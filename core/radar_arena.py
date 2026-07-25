"""
RadarArena — Room 6's environment: a continuous chamber the agent must cross to
the exit while steering around obstacles it can only perceive through its LIGHT
(partial observation), under ICE MOMENTUM (velocity carries between steps). See
Plan.md §Room 6.

This is a `gymnasium.Env` (like Room 5's `ChaseArena`), trained against the same
5-tuple API and the SAME learning recipe (`algorithms/deep_q.py`, unchanged:
Double DQN + reward-scaling + dense straight-line shaping). It is deliberately NOT
an extension of `ChaseArena`, whose observation is enemy-relative positions.

WHY THIS ROOM IS DIFFERENT FROM ROOM 5
--------------------------------------
  * Momentum is back (Room 5 dropped it): the 9 discrete actions set ACCELERATION,
    not displacement, and friction is PER-SECOND (`v ← μ^dt·(v + a·dt)`), the exact
    thing Room 5's old build measured must not be per-step.
  * Partial observation: the net never sees the FULL layout. It sees its own
    `[x, y, vₓ, v_y]` plus the obstacles inside its light radius `X` (below).
  * Generalisation is the thesis: obstacle count and positions are RE-SAMPLED every
    `reset()` (never fixed), so a memorised path cannot work — the light inputs must
    be USED, exactly as Room 5's per-episode enemy respawn forced the enemy channel.

THE LIGHT — EVERYTHING within radius X is detected (user's spec)
---------------------------------------------------------------
The agent carries a lamp that lights the whole disc of radius `X` around it and is
aware of EVERY obstacle inside that disc — there are no rays and no direction bins;
anything in range is simply seen. To feed a fixed-size network, the observation
carries the RELATIVE POSITION `(oₓ−x, o_y−y)/X` of each in-range obstacle, nearest
first, in `MAX_OBSTACLES` fixed slots (unused slots are `(0, 0)` — a real obstacle
can never be at the agent's exact centre, since that is a collision, so zero is an
unambiguous "empty"). Obstacles OUTSIDE `X` are invisible (not in the observation).
Detection is by centre distance (center-to-center, per the user's spec), so the
usable clearance to a lit obstacle is `‖rel‖ − OBSTACLE_RADIUS − AGENT_RADIUS`:
"lit at X" is NOT "collides at X" (documented in the UI too).

MARKOV DISCIPLINE FOR MOVING OBSTACLES
--------------------------------------
A single snapshot of positions is non-Markov once obstacles move — it cannot tell
approaching from receding (the failure that killed lead pursuit and the timeout
penalty in Room 5). So STATIC obstacles are position-only (`obs_dim = 4 + 2·M`, the
default); turning on MOVING obstacles ADDS each lit obstacle's relative VELOCITY
`(vₓ, v_y)/OBS_SPEED_NORM` to its slot → `obs_dim = 4 + 4·M`, which makes the motion
observable (position + velocity is Markov — no frame-stacking needed). The env
exposes `obs_dim` for the room to read. (The central pillar always stays put — only
the extra obstacles drift, so its velocity slot is zero.)

SHAPING avoids Room 6's most dangerous trap on purpose
------------------------------------------------------
Room 5's earlier maze build was silently destroyed by a BFS geodesic field that gave
obstacle cells a huge "unreachable" constant → a −66 spike every step an agent hugged
a wall. Room 6 has obstacles IN the field, so it is the most exposed room. We
deliberately DO NOT use a geodesic field: shaping is straight-line
`shaping_coef·(d_old − d_new)` toward the exit (Plan.md's simpler alternative).
Obstacles then affect only collisions, never the potential, so no spike can ever
fire near one. Collision is a real terminal reward the env pays (−catch_penalty);
the timeout stays DISPLAY-ONLY (never in the learning signal), the app-wide rule.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Fixed geometry (metres). The corner-to-corner run is the lesson; it never
#    moves. The OBSTACLES are what vary per episode. ──
ARENA = 10.0
START = np.array([0.5, 0.5], dtype=np.float64)       # 🤖 bottom-left (fixed start)
EXIT = np.array([9.5, 9.5], dtype=np.float64)        # 🏁 top-right (centre of goal square)
GOAL_HALF = 0.6                                      # 1.2×1.2 m goal square (Chebyshev), as Room 5

# ── Obstacle / agent size — LOCKED. Every obstacle is a disc of this radius; one
#    constant referenced everywhere is the acceptance test (Plan.md §1). ──
OBSTACLE_RADIUS = 0.25                               # 0.5 m obstacles (design decision to lock)
AGENT_RADIUS = 0.25
COLLIDE_DIST = OBSTACLE_RADIUS + AGENT_RADIUS         # centre-to-centre contact ⇒ collision

# Keep obstacles off the start and exit, and apart from each other.
CLEAR_START = 1.5                                    # no obstacle centre within this of START
CLEAR_EXIT = 1.5                                     # ... or of the EXIT centre
MIN_OBSTACLE_GAP = 0.2                               # extra spacing between obstacle SURFACES

# There is ALWAYS a static pillar in the middle of the room (user request): a fixed
# central obstacle straddling the corner-to-corner diagonal, so every layout forces
# the agent to route around the centre. It never moves, even in moving-obstacle mode.
CENTER = np.array([ARENA / 2, ARENA / 2], dtype=np.float64)   # (5, 5)

# ── Time-based physics (Plan.md §2). Friction μ is PER-SECOND (Room 5 measured a
#    per-step μ turns μ=0.8 into a tar pit); dt is small so momentum is smooth. ──
DT = 0.1
THRUST = 3.0                                         # acceleration magnitude (m/s²); beats friction
VMAX = 5.0                                           # speed cap (also the velocity normaliser)

# ── Light sensor. Everything within LIGHT_RADIUS is detected; the observation packs
#    the nearest MAX_OBSTACLES in-range obstacles as relative positions. MAX_OBSTACLES
#    must be ≥ the obstacle-count slider max so "everything in range" is never truncated
#    (at X = arena size, all obstacles can be lit at once). ──
LIGHT_RADIUS = 3.0                                   # X: illumination radius, metres (center-to-center)
MAX_OBSTACLES = 12                                   # M: observation slots = obstacle-count slider max
OBS_SPEED_NORM = 1.5                                 # velocity normaliser (= dynamic-speed slider max)

# The 9 acceleration directions, indexed EXACTLY as Room 5 / Plan.md §Room 5's
# table: dx = a//3 − 1, dy = a%3 − 1. Action 4 = coast (no thrust). Non-zero
# directions are unit-normalised so every thrust has the same magnitude THRUST
# (diagonals included — flat acceleration, as Room 5 chose flat speed).
_RAW = np.array([[a // 3 - 1, a % 3 - 1] for a in range(9)], dtype=np.float64)
_NORM = np.linalg.norm(_RAW, axis=1, keepdims=True)
_NORM[_NORM == 0] = 1.0                              # action 4 (coast) stays (0,0)
ACCEL_DIRS = _RAW / _NORM                            # shape (9, 2), unit directions (coast = 0)


def _seg_point_min_dist(p0, p1, c) -> float:
    """Closest distance from point `c` to the segment p0→p1 (swept-collision math:
    a fast agent must not tunnel through a 0.5 m disc in one tick)."""
    d = p1 - p0
    dd = float(d @ d)
    if dd < 1e-12:
        return float(np.hypot(*(c - p0)))
    t = float((c - p0) @ d / dd)
    t = min(1.0, max(0.0, t))
    closest = p0 + t * d
    return float(np.hypot(*(c - closest)))


class RadarArena(gym.Env):
    """Continuous momentum arena with LIGHT-based obstacle perception (Room 6): the
    agent detects EVERY obstacle within illumination radius X, as relative positions.

    Observation (all normalised to ~[0,1] / [-1,1]), M = MAX_OBSTACLES slots:
      static  : [x/A, y/A, vₓ/VMAX, v_y/VMAX] + M·(relₓ, rel_y)/X   → obs_dim = 4 + 2M
      moving  : the above, each slot also + (vₓ, v_y)/OBS_SPEED_NORM → obs_dim = 4 + 4M
    In-range obstacles fill slots nearest-first; empty slots are 0. 9 discrete
    acceleration actions. Reaching the goal square = +goal_reward; hitting an obstacle
    = −catch_penalty and ends the episode; the step cap truncates (timeout, −100)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_obstacles: int = 6,
        light_radius: float = LIGHT_RADIUS,
        friction: float = 0.5,           # μ per SECOND (retention); lower = more drag
        thrust: float = THRUST,
        obstacle_speed: float = 0.0,     # 0 = static (default, position-only obs)
        max_steps: int = 200,
        random_layout: bool = True,      # re-sample obstacles each reset (off ⇒ fixed seed)
        shaping_coef: float = 1.0,       # dense straight-line progress toward exit
        goal_reward: float = 100.0,
        catch_penalty: float = 100.0,
        layout_seed: int = 12345,        # used when random_layout=False (deterministic warm-up)
    ):
        super().__init__()
        self.n_obstacles = min(int(n_obstacles), MAX_OBSTACLES)
        self.light_radius = float(light_radius)
        self.friction = float(friction)
        self.thrust = float(thrust)
        self.obstacle_speed = float(obstacle_speed)
        self.moving = self.obstacle_speed > 0.0
        self.max_steps = int(max_steps)
        self.random_layout = bool(random_layout)
        self.shaping_coef = float(shaping_coef)
        self.goal_reward = float(goal_reward)
        self.catch_penalty = float(catch_penalty)
        self.layout_seed = int(layout_seed)

        # obs_dim: static is position-only (Markov); moving adds a velocity per slot.
        self.slot_dim = 4 if self.moving else 2
        self.obs_dim = 4 + MAX_OBSTACLES * self.slot_dim

        # Bounds: pos [0,1], vel [-1,1], then M slots of (rel/X) [-1,1] and, if moving,
        # (obstacle velocity / OBS_SPEED_NORM) [-1,1].
        lo = [0.0, 0.0, -1.0, -1.0] + [-1.0] * (MAX_OBSTACLES * self.slot_dim)
        hi = [1.0, 1.0, 1.0, 1.0] + [1.0] * (MAX_OBSTACLES * self.slot_dim)
        self.observation_space = spaces.Box(low=np.array(lo, np.float32),
                                            high=np.array(hi, np.float32))
        self.action_space = spaces.Discrete(9)

        self.agent = START.copy()
        self.vel = np.zeros(2)
        self.obstacles = np.zeros((0, 2))
        self.obs_vel = np.zeros((0, 2))
        self.t = 0

    # ── layout generation ─────────────────────────────────────────────────────
    # Reachability grid (built once): a cell is free unless an obstacle centre is
    # within COLLIDE_DIST of it. Coarse enough to be fast, fine enough (0.4 m) to
    # resolve a real gap (obstacle surfaces sit ≥ MIN_OBSTACLE_GAP apart).
    _RG_STEP = 0.4
    _RG_N = int(ARENA / _RG_STEP) + 1
    _RG_AX = np.linspace(0, ARENA, _RG_N)

    def _reachable(self, obstacles: np.ndarray) -> bool:
        """Is EXIT reachable from START through the free space (agent has radius)?
        One coarse-grid BFS on the WHOLE layout — mirrors `generate_layout`/
        `generate_portals`' keep-only-if-solvable guard, but checked per finished
        layout (not per placement) so generation stays cheap enough to run every
        `reset()` during training."""
        n, ax = self._RG_N, self._RG_AX
        free = np.ones((n, n), dtype=bool)
        for c in obstacles:
            mask = ((ax[None, :] - c[0]) ** 2 +
                    (ax[:, None] - c[1]) ** 2) <= COLLIDE_DIST ** 2
            free[mask] = False

        def idx(p):
            return (int(round(p[1] / self._RG_STEP)), int(round(p[0] / self._RG_STEP)))

        si, ei = idx(START), idx(EXIT)
        if not free[si] or not free[ei]:
            return False
        from collections import deque
        seen = np.zeros_like(free)
        q = deque([si])
        seen[si] = True
        while q:
            r, c = q.popleft()
            if (r, c) == ei:
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and free[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        return False

    def _generate(self, rng) -> np.ndarray:
        """Place `n_obstacles` discs — the FIRST is ALWAYS the fixed central pillar at
        CENTER (user request) — off the start & exit and non-overlapping, then keep the
        layout only if START→EXIT is still traversable; retry otherwise. Falls back to
        whatever was placed (always solvable, and always keeping the centre pillar) if
        the count can't be met — guaranteeing a winnable room."""
        for _attempt in range(40):
            placed = [CENTER.copy()]                     # obstacle 0 = the central pillar
            for _ in range(self.n_obstacles - 1):
                for _try in range(100):
                    pos = rng.uniform(OBSTACLE_RADIUS, ARENA - OBSTACLE_RADIUS, size=2)
                    if np.hypot(*(pos - START)) < CLEAR_START:
                        continue
                    if np.hypot(*(pos - EXIT)) < CLEAR_EXIT:
                        continue
                    if any(np.hypot(*(pos - p)) < 2 * OBSTACLE_RADIUS + MIN_OBSTACLE_GAP
                           for p in placed):
                        continue
                    placed.append(pos)
                    break
            obstacles = np.array(placed, dtype=np.float64)
            if len(placed) == self.n_obstacles and self._reachable(obstacles):
                return obstacles
        # Fallback: keep the centre pillar (index 0); prune extras until solvable.
        while len(placed) > 1 and not self._reachable(np.array(placed)):
            placed.pop()
        return np.array(placed, dtype=np.float64)

    # ── light sensor ──────────────────────────────────────────────────────────
    def _detect(self):
        """The lamp lights the whole disc of radius X: EVERY obstacle inside it is
        detected. Returns `lit` — the indices of in-range obstacles, nearest first
        (centre-to-centre, per the user's spec) — capped at MAX_OBSTACLES slots."""
        if not len(self.obstacles):
            return np.empty(0, dtype=int)
        rel = self.obstacles - self.agent
        dist = np.hypot(rel[:, 0], rel[:, 1])
        lit = np.where(dist <= self.light_radius)[0]
        lit = lit[np.argsort(dist[lit])]               # nearest first
        return lit[:MAX_OBSTACLES]

    # ── obs / info ────────────────────────────────────────────────────────────
    def _obs(self, lit: np.ndarray) -> np.ndarray:
        base = [self.agent[0] / ARENA, self.agent[1] / ARENA,
                self.vel[0] / VMAX, self.vel[1] / VMAX]
        slots = np.zeros((MAX_OBSTACLES, self.slot_dim), dtype=np.float32)
        for s, i in enumerate(lit):
            rel = (self.obstacles[i] - self.agent) / self.light_radius   # in [-1,1]
            slots[s, 0], slots[s, 1] = rel[0], rel[1]
            if self.moving:                            # + relative velocity (Markov)
                v = self.obs_vel[i] / OBS_SPEED_NORM
                slots[s, 2], slots[s, 3] = np.clip(v[0], -1, 1), np.clip(v[1], -1, 1)
        return np.concatenate([np.array(base, np.float32), slots.ravel()])

    def _info(self, lit, outcome=None) -> dict:
        detected = np.zeros(len(self.obstacles), dtype=bool)
        detected[lit] = True
        return {"agent": self.agent.copy(), "vel": self.vel.copy(),
                "obstacles": self.obstacles.copy(), "detected": detected,
                "outcome": outcome}

    # ── gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.agent = START.copy()
        self.vel = np.zeros(2)
        self.t = 0

        if options.get("obstacles") is not None:
            self.obstacles = np.array(options["obstacles"], dtype=np.float64).reshape(-1, 2)
        elif self.random_layout:
            self.obstacles = self._generate(self.np_random)
        else:
            self.obstacles = self._generate(np.random.default_rng(self.layout_seed))

        if self.moving and len(self.obstacles):
            ang = self.np_random.uniform(0, 2 * np.pi, size=len(self.obstacles))
            self.obs_vel = self.obstacle_speed * np.stack([np.cos(ang), np.sin(ang)], 1)
            # The central pillar never moves (user request), even in moving mode.
            at_center = np.all(np.isclose(self.obstacles, CENTER), axis=1)
            self.obs_vel[at_center] = 0.0
        else:
            self.obs_vel = np.zeros((len(self.obstacles), 2))

        lit = self._detect()
        return self._obs(lit), self._info(lit)

    def _move_obstacles(self):
        """Advance moving obstacles one tick, bouncing off the arena walls. Depends
        only on obstacle state, so the dynamics stay Markov given the range-rate obs."""
        if not (self.moving and len(self.obstacles)):
            return
        self.obstacles = self.obstacles + self.obs_vel * DT
        for ax in (0, 1):
            lo = self.obstacles[:, ax] < OBSTACLE_RADIUS
            hi = self.obstacles[:, ax] > ARENA - OBSTACLE_RADIUS
            self.obs_vel[lo | hi, ax] *= -1.0
        self.obstacles = np.clip(self.obstacles, OBSTACLE_RADIUS, ARENA - OBSTACLE_RADIUS)

    def step(self, action: int):
        a0 = self.agent.copy()
        # Momentum: acceleration → per-second friction → integrate. v ← μ^dt(v+a·dt).
        a = self.thrust * ACCEL_DIRS[int(action)]
        v = (self.friction ** DT) * (self.vel + a * DT)
        speed = float(np.hypot(*v))
        if speed > VMAX:
            v *= VMAX / speed
        a1 = a0 + v * DT

        # Wall clamp: stop the component that would leave the arena (agent has radius).
        for ax in (0, 1):
            if a1[ax] < AGENT_RADIUS:
                a1[ax] = AGENT_RADIUS
                v[ax] = 0.0
            elif a1[ax] > ARENA - AGENT_RADIUS:
                a1[ax] = ARENA - AGENT_RADIUS
                v[ax] = 0.0
        self.vel = v

        # Obstacles move, THEN swept collision over the agent's a0→a1 segment against
        # each obstacle's post-move centre (no tunnelling through a 0.5 m disc).
        self._move_obstacles()
        collided = False
        if len(self.obstacles):
            for c in self.obstacles:
                if _seg_point_min_dist(a0, a1, c) < COLLIDE_DIST:
                    collided = True
                    break

        self.agent = a1
        self.t += 1
        lit = self._detect()

        reached = abs(a1[0] - EXIT[0]) < GOAL_HALF and abs(a1[1] - EXIT[1]) < GOAL_HALF
        d_old = float(np.hypot(*(a0 - EXIT)))
        d_new = float(np.hypot(*(a1 - EXIT)))
        reward = self.shaping_coef * (d_old - d_new)   # dense straight-line progress

        terminated, outcome = False, None
        if collided:                                    # a collision precedes the exit
            reward, terminated, outcome = -self.catch_penalty, True, "collided"
        elif reached:
            reward, terminated, outcome = self.goal_reward, True, "escaped"

        truncated = (not terminated) and self.t >= self.max_steps
        if truncated:
            outcome = "timeout"

        info = self._info(lit, outcome)
        return self._obs(lit), float(reward), terminated, truncated, info


# ── rollout helpers for the room (this env has obstacles + a light sensor, not
#    enemies, so it needs its own rollout — deep_q.greedy_rollout is Chase-specific) ─
def greedy_rollout(net, env, seed=None, options=None):
    """One greedy (ε=0) episode. Returns per-step frames (agent, vel, obstacles,
    detected-mask) plus outcome / raw return / step count. Ephemeral — render and
    drop it. Imports torch lazily so `radar_arena` stays importable without torch."""
    import torch
    obs, info = env.reset(seed=seed, options=options)
    frames = [{"agent": info["agent"].copy(), "vel": info["vel"].copy(),
               "obstacles": info["obstacles"].copy(), "detected": info["detected"].copy()}]
    G, t, outcome = 0.0, 0, "timeout"
    while True:
        with torch.no_grad():
            a = int(net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1).item())
        obs, r, term, trunc, info = env.step(a)
        G += r
        t += 1
        frames.append({"agent": info["agent"].copy(), "vel": info["vel"].copy(),
                       "obstacles": info["obstacles"].copy(), "detected": info["detected"].copy()})
        if term or trunc:
            outcome = info.get("outcome") or ("timeout" if trunc else "collided")
            break
    return {"frames": frames, "outcome": outcome, "return": G, "steps": t}


def evaluate_policy(net, make_env, n_eval=200, seed0=0):
    """Greedy escape/collision/timeout rates over `n_eval` fresh layouts drawn from
    seeds `seed0 …`. Used for the held-out Generalisation Score (pass a seed stream
    DISJOINT from training's). Returns (escape, collide, timeout, paths) where paths
    is a list of (T,2) agent trajectories for the trajectory heatmap."""
    env = make_env()
    esc = coll = to = 0
    paths = []
    for i in range(n_eval):
        roll = greedy_rollout(net, env, seed=seed0 + i)
        oc = roll["outcome"]
        esc += oc == "escaped"; coll += oc == "collided"; to += oc == "timeout"
        paths.append(np.array([f["agent"] for f in roll["frames"]]))
    return esc / n_eval, coll / n_eval, to / n_eval, paths
