"""
ChaseArena — Room 5's environment: an empty continuous arena with ONE enemy that
chases the agent. Reach the exit (+100); the enemy touching you ends the episode
at −100.

This is a `gymnasium.Env` (unlike Rooms 1–4's tabular dict-MDPs), because Room 5
is the Deep Q-Learning room: the network trains against the gymnasium 5-tuple API
exactly as `code examples/dql` does (`reset -> (obs, info)`,
`step -> (obs, reward, terminated, truncated, info)`).

DESIGN (redesigned 2026-07-20, user) — see Plan.md §Room 5:
  * Empty 10×10 m arena, NO walls, DIRECT inertia-free movement (the action IS the
    displacement; there is no momentum, so the ice theme does not apply here).
  * State `[x, y, eₓ−x, e_y−y]` normalised to the arena → obs_dim = 4. The enemy is
    given RELATIVE to the agent, which is the whole point: the network must read
    where the enemy is, not memorise one path.
  * 9 discrete actions = the 8 compass moves + stay, each 1 m (diagonals 1 m total,
    ≈0.707 per axis). Fixed index order — the network's output layer is indexed by
    it, so it must never be reordered.
  * ONE enemy, PURE PURSUIT (greedy): each step it steps `enemy_speed` metres
    straight toward the agent's CURRENT position. Deliberately myopic (aims where
    you are, not where you'll be) so it can be baited — an agent that arcs around
    it gets it behind and, being slower, it can no longer close before the exit.

WHY THE ENEMY SPAWNS RANDOMLY.
------------------------------
With a fixed enemy the whole env is deterministic, so a policy escapes 0% or 100%
— no band to measure and nothing to generalise. The enemy therefore spawns at a
random position each `reset()` (kept clear of the agent so it is never an instant
catch). Over many placements a naive beeline escapes some and dies to others,
while a good evasive policy escapes more — that GAP is the learnable signal, and
it is why the relative-enemy input has to be in the observation.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Fixed geometry (metres). The corner-to-corner run is the lesson; it never
#    moves, as Rooms 3–4's board never moves. Only the enemy's spawn varies. ──
ARENA = 10.0
START = np.array([0.5, 0.5], dtype=np.float64)       # 🤖 bottom-left
EXIT = np.array([9.5, 9.5], dtype=np.float64)        # 🏁 top-right
GOAL_RADIUS = 0.5                                    # within this of EXIT ⇒ escaped
CATCH_RADIUS = 0.5                                   # within this of enemy ⇒ caught
STEP = 1.0                                           # agent moves 1 m per decision

# Enemy spawn region and the clear-zone around the agent's start, so a fresh
# episode is never an instant death.
SPAWN_LO, SPAWN_HI = 1.5, 8.5
SPAWN_MIN_DIST_FROM_START = 3.0

# The 9 moves, indexed exactly as Plan.md §Room 5's table: dx = a//3 − 1,
# dy = a%3 − 1. Non-zero moves are normalised to a 1 m step below.
_RAW = np.array([[a // 3 - 1, a % 3 - 1] for a in range(9)], dtype=np.float64)
_NORM = np.linalg.norm(_RAW, axis=1, keepdims=True)
_NORM[_NORM == 0] = 1.0                              # action 4 (stay) stays (0,0)
MOVES = _RAW / _NORM * STEP                          # shape (9, 2), each row a 1 m step


def _min_dist_moving(a0, a1, e0, e1) -> float:
    """Closest approach between the agent (a0→a1) and enemy (e0→e1) over one step,
    both moving linearly. This is the SWEPT catch check — it stops the two tunnelling
    past each other in a single 1 m step (endpoint-only checks miss that)."""
    w = a0 - e0
    v = (a1 - a0) - (e1 - e0)
    vv = float(v @ v)
    t = 0.0 if vv == 0.0 else float(-(w @ v) / vv)
    t = min(1.0, max(0.0, t))
    closest = w + t * v
    return float(np.hypot(closest[0], closest[1]))


class ChaseArena(gym.Env):
    """Continuous chase arena for Room 5. Observation is 4-D and normalised;
    9 discrete actions; one pure-pursuit enemy that ends the episode on contact."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        enemy_speed: float = 0.70,      # fraction of the agent's 1 m/step (must be < 1 to be winnable)
        max_steps: int = 60,
        shaping_coef: float = 1.0,      # dense progress-toward-exit reward per metre gained
        goal_reward: float = 100.0,
        catch_penalty: float = 100.0,
    ):
        super().__init__()
        self.enemy_speed = float(enemy_speed)
        self.max_steps = int(max_steps)
        self.shaping_coef = float(shaping_coef)
        self.goal_reward = float(goal_reward)
        self.catch_penalty = float(catch_penalty)

        # obs = [x/10, y/10, (eₓ−x)/10, (e_y−y)/10]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(9)

        self.agent = START.copy()
        self.enemy = np.array([ARENA / 2, ARENA / 2], dtype=np.float64)
        self.t = 0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        rel = (self.enemy - self.agent) / ARENA
        return np.array(
            [self.agent[0] / ARENA, self.agent[1] / ARENA, rel[0], rel[1]],
            dtype=np.float32,
        )

    def _sample_enemy(self) -> np.ndarray:
        """A random spawn inside the central region, kept clear of the agent start."""
        while True:
            pos = self.np_random.uniform(SPAWN_LO, SPAWN_HI, size=2)
            if np.hypot(*(pos - START)) >= SPAWN_MIN_DIST_FROM_START:
                return pos

    # ── gymnasium API ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = START.copy()
        # `options={"enemy_pos": (x, y)}` pins the enemy (used by geometry tests);
        # otherwise it spawns randomly for episode variety.
        if options and options.get("enemy_pos") is not None:
            self.enemy = np.array(options["enemy_pos"], dtype=np.float64)
        else:
            self.enemy = self._sample_enemy()
        self.t = 0
        return self._obs(), {"agent": self.agent.copy(), "enemy": self.enemy.copy()}

    def step(self, action: int):
        a0 = self.agent.copy()
        e0 = self.enemy.copy()

        # Agent and enemy commit SIMULTANEOUSLY: the enemy aims at the agent's
        # pre-move position (pure pursuit), which is what makes it baitable.
        a1 = np.clip(a0 + MOVES[int(action)], 0.0, ARENA)

        to_agent = a0 - e0
        d = float(np.hypot(to_agent[0], to_agent[1]))
        e1 = e0 if d == 0.0 else np.clip(e0 + self.enemy_speed * to_agent / d, 0.0, ARENA)

        self.agent, self.enemy = a1, e1
        self.t += 1

        # Outcomes. A catch mid-step precedes reaching the exit endpoint, so it
        # takes priority; both are terminal.
        caught = _min_dist_moving(a0, a1, e0, e1) < CATCH_RADIUS
        reached = float(np.hypot(*(a1 - EXIT))) < GOAL_RADIUS

        d_old = float(np.hypot(*(a0 - EXIT)))
        d_new = float(np.hypot(*(a1 - EXIT)))
        reward = self.shaping_coef * (d_old - d_new)   # dense progress toward exit

        terminated = False
        outcome = None
        if caught:
            reward = -self.catch_penalty
            terminated = True
            outcome = "caught"
        elif reached:
            reward = self.goal_reward
            terminated = True
            outcome = "escaped"

        truncated = (not terminated) and self.t >= self.max_steps
        if truncated:
            outcome = "timeout"

        info = {"agent": self.agent.copy(), "enemy": self.enemy.copy(), "outcome": outcome}
        return self._obs(), float(reward), terminated, truncated, info
