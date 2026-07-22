"""
ChaseArena — Room 5's environment: an empty continuous arena with one or two
enemies that chase the agent. Reach the exit (+100); an enemy touching you ends
the episode at −100.

This is a `gymnasium.Env` (unlike Rooms 1–4's tabular dict-MDPs), because Room 5
is the Deep Q-Learning room: the network trains against the gymnasium 5-tuple API
exactly as `code examples/dql` does (`reset -> (obs, info)`,
`step -> (obs, reward, terminated, truncated, info)`).

DESIGN (redesigned 2026-07-20, user) — see Plan.md §Room 5:
  * Empty 10×10 m arena, NO walls, DIRECT inertia-free movement (the action IS the
    displacement; there is no momentum, so the ice theme does not apply here).
  * State `[x, y]` + per enemy `[eₓ−x, e_y−y]` (enemy RELATIVE to the agent),
    normalised to the arena → `obs_dim = 2 + 2·n_enemies` (4 with one enemy, 6
    with two). The enemies are given RELATIVE because the network must read where
    they are, not memorise one path.
  * 9 discrete actions = the 8 compass moves + stay, each 1 m (diagonals 1 m total,
    ≈0.707 per axis). Fixed index order — the network's output layer is indexed by
    it, so it must never be reordered.
  * Enemy behaviours (`enemy_kinds`, each a function of OBSERVED state only, so the
    env stays Markov): a **Chaser** (PURE PURSUIT — steps `enemy_speed` metres
    straight at the agent's CURRENT position; myopic, so it can be baited) and, with
    two enemies, a **Flanker** (pursues along a heading rotated `_FLANK_DEG` off the
    direct line, so it curves in from the side). Two pure-pursuit enemies would
    converge onto the same pursuit curve and move in lockstep; the flanker's rotated
    approach keeps them apart and makes 2-enemy mode a real pincer. Two rejected
    alternatives: an exit-guarding interceptor (measured — it camps the one place the
    agent must reach and drops escape to ~1–7%, unwinnable), and lead pursuit that
    predicts the agent's velocity (velocity is not in the observation, so it would
    break the Markov property, the same reason the timeout penalty was rejected).

THE AGENT ALWAYS STARTS AT THE CORNER; THE ENEMIES ARE WHAT VARY.
-----------------------------------------------------------------
The corner-to-corner run is the fixed lesson, so the agent always starts at
`START` (bottom-left). What varies is the ENEMIES: with `random_enemies=True`
(the default) they spawn at fresh random positions each `reset()`, kept clear of
the agent so it is never an instant catch. That randomisation is what makes
"escape rate" a smooth, measurable number AND what forces the network to actually
read the relative-enemy inputs rather than memorise one path — over many
placements a naive beeline escapes some and dies to others, while a good evasive
policy escapes more, and that GAP is the learnable signal.

With `random_enemies=False` the enemies sit at fixed spawns and, since the agent
is fixed too, the whole episode is deterministic — a warm-up / demo mode where the
network only has to solve one configuration.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Fixed geometry (metres). The corner-to-corner run is the lesson; it never
#    moves. Only the enemies (and optionally the agent start) vary per episode. ──
ARENA = 10.0
START = np.array([0.5, 0.5], dtype=np.float64)       # 🤖 bottom-left (fixed start)
EXIT = np.array([9.5, 9.5], dtype=np.float64)        # 🏁 top-right (centre of the goal square)
GOAL_HALF = 0.6                                      # goal is a 1.2×1.2 m square: |x−Eₓ|,|y−E_y| < GOAL_HALF
CATCH_RADIUS = 0.5                                   # within this of an enemy ⇒ caught
STEP = 1.0                                           # agent moves 1 m per decision

# Spawn regions and clear-zones so a fresh episode is never an instant death.
SPAWN_LO, SPAWN_HI = 1.5, 8.5                        # enemy spawn box
MIN_DIST_AGENT_ENEMY = 3.0                           # enemy never spawns on top of the agent
MIN_DIST_ENEMIES = 2.0                               # enemies start apart

# Fixed enemy spawns used when randomisation is OFF, keyed by enemy count. The agent
# is always at START, so the episode is then deterministic — a warm-up mode.
# Positions are clear of the corner, the exit, and each other.
FIXED_ENEMY_SPAWNS = {
    0: np.zeros((0, 2), dtype=np.float64),
    1: np.array([[5.0, 5.0]], dtype=np.float64),
    2: np.array([[3.5, 6.5], [6.5, 3.5]], dtype=np.float64),
    3: np.array([[5.0, 5.0], [2.5, 7.0], [7.0, 2.5]], dtype=np.float64),
}

# ── Enemy behaviours (each a function of OBSERVED state only — the agent's and
#    enemies' positions and the fixed exit — so the env stays Markov). Pure-pursuit
#    enemies collapse onto the same pursuit curve and move in lockstep; different
#    behaviours + mutual repulsion keep them apart and make a real multi-sided
#    pincer rather than one doubled dot. ──
PURSUIT = "pursuit"      # Chaser: aims straight at the agent — trails from behind
FLANK = "flank"          # Flanker: pursuit heading rotated +deg — curves in from one side
AMBUSH = "ambush"        # Ambusher: pursuit heading rotated −deg AND wider — the other side
_FLANK_DEG = 45.0        # the flanker's rotation off the direct line
_AMBUSH_DEG = -72.0      # the ambusher's rotation (opposite side, wider → almost side-on)
_FC, _FS = np.cos(np.radians(_FLANK_DEG)), np.sin(np.radians(_FLANK_DEG))
_AC, _AS = np.cos(np.radians(_AMBUSH_DEG)), np.sin(np.radians(_AMBUSH_DEG))

# Mutual repulsion: enemies push away from each other when closer than this, so the
# hunters can't stack on the same spot — they split to different sides (a pincer).
# Depends only on the enemies' positions, so it stays Markov.
REPEL_RADIUS = 3.0
REPEL_STRENGTH = 1.2

# The three enemy TYPES, in a fixed order (the observation packs active enemies in
# this order, so the network's inputs stay consistent).
ENEMY_TYPES = (PURSUIT, FLANK, AMBUSH)

# Default kinds by count (used when enemy_kinds isn't given explicitly).
DEFAULT_ENEMY_KINDS = {
    0: (),
    1: (PURSUIT,),
    2: (PURSUIT, FLANK),
    3: (PURSUIT, FLANK, AMBUSH),
}

# The 9 moves, indexed exactly as Plan.md §Room 5's table: dx = a//3 − 1,
# dy = a%3 − 1. Non-zero moves are normalised to a 1 m step below.
_RAW = np.array([[a // 3 - 1, a % 3 - 1] for a in range(9)], dtype=np.float64)
_NORM = np.linalg.norm(_RAW, axis=1, keepdims=True)
_NORM[_NORM == 0] = 1.0                              # action 4 (stay) stays (0,0)
MOVES = _RAW / _NORM * STEP                          # shape (9, 2), each row a 1 m step


def _min_dist_moving(a0, a1, e0, e1) -> float:
    """Closest approach between the agent (a0→a1) and an enemy (e0→e1) over one
    step, both moving linearly. This is the SWEPT catch check — it stops the two
    tunnelling past each other in a single 1 m step (endpoint checks miss that)."""
    w = a0 - e0
    v = (a1 - a0) - (e1 - e0)
    vv = float(v @ v)
    t = 0.0 if vv == 0.0 else float(-(w @ v) / vv)
    t = min(1.0, max(0.0, t))
    closest = w + t * v
    return float(np.hypot(closest[0], closest[1]))


class ChaseArena(gym.Env):
    """Continuous chase arena for Room 5. Observation is `2 + 2·n_enemies`-D and
    normalised; 9 discrete actions; 0–3 enemies (each a Chaser / Flanker / Ambusher),
    any of which ends the episode on contact."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        enemy_speed: float = 0.70,      # fraction of the agent's 1 m/step (< 1 to be winnable)
        max_steps: int = 60,
        n_enemies: int = None,          # ignored if enemy_kinds is given
        random_enemies: bool = True,    # random enemy spawn each episode (off ⇒ fixed spawn)
        enemy_kinds=None,               # tuple of active enemy behaviours (0–3); primary
        shaping_coef: float = 1.0,      # dense progress-toward-exit reward per metre gained
        goal_reward: float = 100.0,
        catch_penalty: float = 100.0,
    ):
        super().__init__()
        self.enemy_speed = float(enemy_speed)
        self.max_steps = int(max_steps)
        if enemy_kinds is not None:
            self.enemy_kinds = tuple(enemy_kinds)
        else:
            self.enemy_kinds = DEFAULT_ENEMY_KINDS[1 if n_enemies is None else n_enemies]
        self.n_enemies = len(self.enemy_kinds)
        self.random_enemies = bool(random_enemies)
        self.shaping_coef = float(shaping_coef)
        self.goal_reward = float(goal_reward)
        self.catch_penalty = float(catch_penalty)

        # obs = [x/10, y/10] + per enemy [(eₓ−x)/10, (e_y−y)/10]
        low = np.array([0.0, 0.0] + [-1.0, -1.0] * self.n_enemies, dtype=np.float32)
        high = np.array([1.0, 1.0] + [1.0, 1.0] * self.n_enemies, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(9)

        self.agent = START.copy()
        self.enemies = np.tile(np.array([ARENA / 2, ARENA / 2]), (self.n_enemies, 1))
        self.t = 0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        rel = (self.enemies - self.agent) / ARENA          # (n, 2)
        return np.concatenate([[self.agent[0] / ARENA, self.agent[1] / ARENA],
                               rel.ravel()]).astype(np.float32)

    def _info(self, outcome=None) -> dict:
        return {"agent": self.agent.copy(), "enemies": self.enemies.copy(),
                "outcome": outcome}

    def _enemy_move_dir(self, kind, e0, agent):
        """Unit step direction for an enemy this step. Depends only on the enemy's
        and agent's positions, so the dynamics stay Markov. The Chaser aims straight
        at the agent; the Flanker and Ambusher rotate that heading (opposite signs,
        different magnitudes) so they curve in from different sides."""
        to_agent = agent - e0
        d = float(np.hypot(to_agent[0], to_agent[1]))
        if d < 1e-9:
            return np.zeros(2)
        u = to_agent / d
        if kind == FLANK:
            u = np.array([_FC * u[0] - _FS * u[1], _FS * u[0] + _FC * u[1]])
        elif kind == AMBUSH:
            u = np.array([_AC * u[0] - _AS * u[1], _AS * u[0] + _AC * u[1]])
        return u

    def _sample_enemies(self, agent) -> np.ndarray:
        """`n_enemies` random spawns in the central region, clear of the agent and
        of each other."""
        if self.n_enemies == 0:
            return np.zeros((0, 2), dtype=np.float64)
        placed = []
        for _ in range(self.n_enemies):
            for _try in range(200):
                pos = self.np_random.uniform(SPAWN_LO, SPAWN_HI, size=2)
                if np.hypot(*(pos - agent)) < MIN_DIST_AGENT_ENEMY:
                    continue
                if any(np.hypot(*(pos - p)) < MIN_DIST_ENEMIES for p in placed):
                    continue
                placed.append(pos)
                break
            else:
                placed.append(pos)                          # give up the spacing, keep clear of agent
        return np.array(placed, dtype=np.float64)

    # ── gymnasium API ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # The agent always starts at the corner (unless a test pins it explicitly).
        if options.get("agent_pos") is not None:
            self.agent = np.array(options["agent_pos"], dtype=np.float64)
        else:
            self.agent = START.copy()

        # Enemies: random each episode (default), fixed spawn, or test-pinned.
        if options.get("enemy_pos") is not None:
            self.enemies = np.array(options["enemy_pos"], dtype=np.float64).reshape(self.n_enemies, 2)
        elif self.random_enemies:
            self.enemies = self._sample_enemies(self.agent)
        else:
            self.enemies = FIXED_ENEMY_SPAWNS[self.n_enemies].copy()

        self.t = 0
        return self._obs(), self._info()

    def step(self, action: int):
        a0 = self.agent.copy()
        e0s = self.enemies.copy()

        # Agent and enemies commit SIMULTANEOUSLY: each enemy aims at the agent's
        # pre-move position (pure pursuit), which is what makes it baitable.
        a1 = np.clip(a0 + MOVES[int(action)], 0.0, ARENA)

        e1s = np.empty_like(e0s)
        caught = False
        for i, e0 in enumerate(e0s):
            u = self._enemy_move_dir(self.enemy_kinds[i], e0, a0)   # chaser / flanker / ambusher
            for j, o0 in enumerate(e0s):                            # + mutual repulsion
                if j == i:
                    continue
                away = e0 - o0
                dd = float(np.hypot(away[0], away[1]))
                if 1e-9 < dd < REPEL_RADIUS:
                    u = u + REPEL_STRENGTH * (REPEL_RADIUS - dd) / REPEL_RADIUS * (away / dd)
            nu = float(np.hypot(u[0], u[1]))
            if nu > 1e-9:
                u = u / nu
            e1 = np.clip(e0 + self.enemy_speed * u, 0.0, ARENA)
            e1s[i] = e1
            if _min_dist_moving(a0, a1, e0, e1) < CATCH_RADIUS:
                caught = True

        self.agent, self.enemies = a1, e1s
        self.t += 1

        # Goal is a 1×1 m square centred on EXIT (Chebyshev, not a circle).
        reached = abs(a1[0] - EXIT[0]) < GOAL_HALF and abs(a1[1] - EXIT[1]) < GOAL_HALF
        d_old = float(np.hypot(*(a0 - EXIT)))
        d_new = float(np.hypot(*(a1 - EXIT)))
        reward = self.shaping_coef * (d_old - d_new)       # dense progress toward exit

        terminated, outcome = False, None
        if caught:                                          # a catch mid-step precedes the exit
            reward, terminated, outcome = -self.catch_penalty, True, "caught"
        elif reached:
            reward, terminated, outcome = self.goal_reward, True, "escaped"

        truncated = (not terminated) and self.t >= self.max_steps
        if truncated:
            outcome = "timeout"

        return self._obs(), float(reward), terminated, truncated, self._info(outcome)
