"""
IcyGridWorld — a 10x10 grid world with three special cell types.

Shared discrete environment for Rooms 1-4. Cells come in these flavours:

  * blocked  — walls the agent cannot step into (a move toward one keeps it put),
  * ice      — slippery cells: a move slips perpendicular with prob `slip`,
  * penalty  — passable cells that yield a negative reward each time they are
               entered (NON-terminal; the episode only ends at the goal).

The goal is the only terminal cell (reward `goal_reward`, +100 by default).

Transitions are generated programmatically and exposed in the same
`.probs` / `.rewards` form the reference DP scripts expect.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

ACTION_SPACE = ("U", "D", "L", "R")

_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_PERPENDICULAR = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}


class IcyGridWorld:
    """A grid with blocked walls, per-cell slippery ice, and passable penalties.

    Parameters
    ----------
    rows, cols  : board dimensions.
    start       : (i, j) start cell.
    goal        : (i, j) terminal goal cell — reward `goal_reward`.
    blocked     : iterable of wall cells the agent cannot enter.
    ice         : iterable of slippery cells. If None, EVERY navigable cell is icy
                  (uniform-slip fallback for later rooms).
    penalties   : dict {cell: reward} of passable negative-reward cells.
    slip        : slip probability on ice cells.
    goal_reward : terminal reward at the goal.
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        start: tuple[int, int] = (9, 0),
        goal: tuple[int, int] = (0, 9),
        blocked=None,
        ice=None,
        penalties=None,
        slip: float = 0.2,
        goal_reward: float = 100.0,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.slip = float(slip)
        self.goal_reward = float(goal_reward)

        self.blocked = set(tuple(b) for b in (blocked or []))
        self.penalties = {tuple(c): float(r) for c, r in dict(penalties or {}).items()}

        # Ice defaults to every navigable cell (uniform slip) when unspecified.
        navigable = [s for s in self.all_states() if s not in self.blocked]
        if ice is None:
            self.ice = set(navigable)
        else:
            self.ice = set(tuple(c) for c in ice)

        # Rewards keyed by RESULTING state: goal + passable penalties.
        self.rewards = {self.goal: self.goal_reward}
        self.rewards.update(self.penalties)

        # Navigable, non-terminal cells get all four actions.
        self.actions = {
            s: ACTION_SPACE
            for s in navigable
            if not self.is_terminal(s)
        }

        self.probs = self._build_probs()
        self.i, self.j = start

    # ------------------------------------------------------------------ #
    # Static structure
    # ------------------------------------------------------------------ #
    def all_states(self):
        return [(i, j) for i in range(self.rows) for j in range(self.cols)]

    def is_terminal(self, s) -> bool:
        return s == self.goal

    def is_blocked(self, s) -> bool:
        return s in self.blocked

    def is_icy(self, s) -> bool:
        return s in self.ice

    def in_bounds(self, i, j) -> bool:
        return 0 <= i < self.rows and 0 <= j < self.cols

    def _step_cell(self, s, a):
        """Deterministic result of action `a` from `s`; wall or edge = stay put."""
        di, dj = _DELTAS[a]
        ni, nj = s[0] + di, s[1] + dj
        if self.in_bounds(ni, nj) and (ni, nj) not in self.blocked:
            return (ni, nj)
        return s

    def _build_probs(self):
        probs = {}
        for s in self.actions:  # navigable, non-terminal
            for a in ACTION_SPACE:
                outcomes: dict[tuple[int, int], float] = {}
                if self.is_icy(s):
                    intended = self._step_cell(s, a)
                    outcomes[intended] = outcomes.get(intended, 0.0) + (1.0 - self.slip)
                    for pa in _PERPENDICULAR[a]:
                        slipped = self._step_cell(s, pa)
                        outcomes[slipped] = outcomes.get(slipped, 0.0) + self.slip / 2.0
                else:
                    dest = self._step_cell(s, a)
                    outcomes[dest] = 1.0
                probs[(s, a)] = outcomes
        return probs

    def get_transition_probs_and_rewards(self):
        """Return (transition_probs, rewards) in (s, a, s') form for the DP code."""
        transition_probs = {}
        rewards = {}
        for (s, a), outcomes in self.probs.items():
            for s2, p in outcomes.items():
                transition_probs[(s, a, s2)] = p
                rewards[(s, a, s2)] = self.rewards.get(s2, 0.0)
        return transition_probs, rewards

    # ------------------------------------------------------------------ #
    # Live simulation (for animating a policy)
    # ------------------------------------------------------------------ #
    def reset(self):
        self.i, self.j = self.start
        return self.start

    def current_state(self):
        return (self.i, self.j)

    def move(self, action):
        outcomes = self.probs[((self.i, self.j), action)]
        states = list(outcomes.keys())
        p = list(outcomes.values())
        idx = np.random.choice(len(states), p=p)
        self.i, self.j = states[idx]
        return self.rewards.get((self.i, self.j), 0.0)

    def game_over(self):
        return self.is_terminal((self.i, self.j))


# ---------------------------------------------------------------------- #
# Random layout generation.
# ---------------------------------------------------------------------- #
def _connected(start, goal, blocked, rows, cols) -> bool:
    """True if `goal` is reachable from `start` avoiding blocked cells."""
    seen = {start}
    q = deque([start])
    while q:
        i, j = q.popleft()
        if (i, j) == goal:
            return True
        for di, dj in _DELTAS.values():
            ni, nj = i + di, j + dj
            nxt = (ni, nj)
            if (0 <= ni < rows and 0 <= nj < cols
                    and nxt not in blocked and nxt not in seen):
                seen.add(nxt)
                q.append(nxt)
    return False


def generate_layout(
    n_blocked: int,
    n_slippery: int,
    n_negative: int,
    seed: int,
    rows: int = 10,
    cols: int = 10,
    start: tuple[int, int] = (9, 0),
    goal: tuple[int, int] = (0, 9),
):
    """Randomly place the three cell types on the board.

    Blocked cells are placed one at a time and only kept if the start still
    reaches the goal, so the board is always solvable. Slippery and negative
    cells are sampled from the remaining free cells and are mutually exclusive.

    Returns (blocked, ice, negatives) as sets of (i, j).
    """
    rng = random.Random(seed)
    free = [s for s in ((i, j) for i in range(rows) for j in range(cols))
            if s != start and s != goal]
    rng.shuffle(free)

    blocked: set = set()
    for c in free:
        if len(blocked) >= n_blocked:
            break
        blocked.add(c)
        if not _connected(start, goal, blocked, rows, cols):
            blocked.discard(c)  # would wall off the goal — skip it

    remaining = [c for c in free if c not in blocked]
    rng.shuffle(remaining)
    ice = set(remaining[:n_slippery])
    negatives = set(remaining[n_slippery:n_slippery + n_negative])
    return blocked, ice, negatives
