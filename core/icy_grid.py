"""
IcyGridWorld — a 10x10 grid world with several special cell types.

Shared discrete environment for Rooms 1-4. Cells come in these flavours:

  * blocked  — walls the agent cannot step into (a move toward one keeps it put),
  * ice      — slippery cells: a move slips perpendicular with prob `slip`,
  * penalty  — passable cells that yield a negative reward each time they are
               entered (NON-terminal; the episode only ends at the goal),
  * teleport — cells that bounce the agent elsewhere the instant it lands on one
               (Room 2's portal traps, which send it back to the start).

The goal is the only terminal cell (reward `goal_reward`, +100 by default).

Transitions are generated programmatically and exposed in the same
`.probs` / `.rewards` form the reference DP scripts expect.

Teleports are **folded into `.probs`**: a transition that physically lands on a
teleport cell is recorded as going straight to its destination. This keeps
`.probs` a true Markov model, so the DP/TD math needs no notion of portals, and
it makes a teleport cell *transient* — reachable-through but never occupied.
`move()` still samples the PHYSICAL landing cell first and records it in
`last_landing`, so an animation can show the agent touching the portal before it
is whisked away (see `core.episode.rollout(..., with_landings=True)`).
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
    teleports   : dict {cell: destination} — landing on `cell` moves the agent to
                  `destination` in the same step (Room 2's portals). Folded into
                  `.probs`, so teleport cells are transient, never occupied.
    rng         : numpy Generator used by move(); defaults to a fresh one. Pass a
                  seeded generator for reproducible rollouts.
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
        teleports=None,
        rng=None,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.slip = float(slip)
        self.goal_reward = float(goal_reward)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.blocked = set(tuple(b) for b in (blocked or []))
        self.penalties = {tuple(c): float(r) for c, r in dict(penalties or {}).items()}
        self.teleports = {tuple(k): tuple(v)
                          for k, v in dict(teleports or {}).items()}

        # Ice defaults to every navigable cell (uniform slip) when unspecified.
        navigable = [s for s in self.all_states() if s not in self.blocked]
        if ice is None:
            self.ice = set(navigable)
        else:
            self.ice = set(tuple(c) for c in ice)

        # Rewards keyed by RESULTING state: goal + passable penalties.
        self.rewards = {self.goal: self.goal_reward}
        self.rewards.update(self.penalties)

        # Navigable, non-terminal cells get all four actions. Teleport cells are
        # excluded: the agent is whisked away the instant it lands on one, so it
        # is never *standing* there to choose an action. Leaving them in would
        # give DP a value for a state that can never be occupied — and would
        # leave a permanent hole in any Q learned from experience.
        self.actions = {
            s: ACTION_SPACE
            for s in navigable
            if not self.is_terminal(s) and s not in self.teleports
        }

        self.probs, self._phys = self._build_probs()
        self._sampler = {
            k: (list(o.keys()), np.cumsum(np.fromiter(o.values(), float, len(o))))
            for k, o in self._phys.items()
        }
        self.i, self.j = start
        self.last_landing = start

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

    def is_teleport(self, s) -> bool:
        return s in self.teleports

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
        """Build the transition model.

        Returns (probs, phys):
          * `phys[(s, a)]` — distribution over the cell physically landed on,
            before any teleport fires. Used by move() so an animation can show
            the agent touching a portal.
          * `probs[(s, a)]` — the same distribution with teleports folded into
            their destinations: the true Markov model the RL math consumes.
        With no teleports the two are identical.
        """
        probs, phys = {}, {}
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
                phys[(s, a)] = outcomes

                folded: dict[tuple[int, int], float] = {}
                for s2, p in outcomes.items():
                    dest = self.teleports.get(s2, s2)
                    folded[dest] = folded.get(dest, 0.0) + p
                probs[(s, a)] = folded
        return probs, phys

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
        self.last_landing = self.start
        return self.start

    def current_state(self):
        return (self.i, self.j)

    def move(self, action):
        """Take one stochastic step; return the reward of the resulting cell.

        Samples the physical landing cell (recorded in `last_landing`), then
        applies any teleport. Sampling is a single uniform draw against a
        precomputed cumulative distribution rather than `np.random.choice`,
        which costs ~10us per call and dominates Room 2's ~1.5M training steps.
        """
        states, cum = self._sampler[((self.i, self.j), action)]
        idx = int(np.searchsorted(cum, self.rng.random() * cum[-1], side="right"))
        landed = states[min(idx, len(states) - 1)]
        self.last_landing = landed
        self.i, self.j = self.teleports.get(landed, landed)
        return self.rewards.get((self.i, self.j), 0.0)

    def game_over(self):
        return self.is_terminal((self.i, self.j))


# ---------------------------------------------------------------------- #
# Random layout generation.
# ---------------------------------------------------------------------- #
def _connected(start, goal, blocked, rows, cols, teleports=None) -> bool:
    """True if `goal` is reachable from `start` avoiding blocked cells.

    `teleports` (if given) is folded in exactly as the transition model folds it:
    stepping onto a teleport cell continues from its destination instead. A
    teleport is therefore never a state you can stand on, which means portals can
    seal the exit off without walling it — see `generate_portals`.
    """
    teleports = teleports or {}
    seen = {start}
    q = deque([start])
    while q:
        i, j = q.popleft()
        if (i, j) == goal:
            return True
        for di, dj in _DELTAS.values():
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            nxt = (ni, nj)
            if nxt in blocked:
                continue
            nxt = teleports.get(nxt, nxt)  # land on a portal -> continue from its exit
            if nxt not in seen:
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


def generate_portals(
    blocked,
    n_portals: int,
    seed: int,
    rows: int = 10,
    cols: int = 10,
    start: tuple[int, int] = (9, 0),
    goal: tuple[int, int] = (0, 9),
    exclude=None,
):
    """Place Room 2's portal traps so they never seal the exit off.

    Portals cannot use the plain `generate_layout` sampler. A portal is a
    *transient* cell — landing on it teleports the agent away before it can act —
    so a portal sitting on the only cell that leads into the goal makes the exit
    unreachable even though nothing is walled, and the whole board silently
    becomes unsolvable (V* collapses to 0 everywhere).

    So portals are placed one at a time and kept only if the start still reaches
    the goal through the FOLDED model — the same incremental guard
    `generate_layout` already applies to walls.

    `exclude` keeps portals off cells already claimed by another type (e.g. the
    ice from `generate_layout`, which is sampled from an independent pool). This
    is cosmetic — a portal cell is transient, so whether it is icy never affects
    the model — but a cell drawn as two hazards at once just reads as a bug.

    Returns a set of (i, j) — possibly fewer than `n_portals` if the rest would
    have stranded the exit. Callers should surface the shortfall.
    """
    rng = random.Random(seed)
    exclude = set(exclude or ())
    free = [s for s in ((i, j) for i in range(rows) for j in range(cols))
            if s != start and s != goal and s not in blocked and s not in exclude]
    rng.shuffle(free)

    portals: dict = {}
    for c in free:
        if len(portals) >= n_portals:
            break
        portals[c] = start
        if not _connected(start, goal, blocked, rows, cols, portals):
            del portals[c]  # would strand the exit — skip it
    return set(portals)
