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


# --------------------------------------------------------------------------- #
# Pure slip physics, shared by every discrete room.
#
# These are free functions rather than methods so a room whose STATE SHAPE this
# class does not model (Room 4's moving guard needs (i, j, phase, coins), which
# is a different augmentation from the shield flag below) can reuse the exact
# same slip distribution instead of copying it. Keeping ONE implementation is
# deliberate: Room 2's memory records how subtle this distribution is (the
# zero-probability-outcome landmine), and two copies would drift.
# --------------------------------------------------------------------------- #
def step_cell(cell, a, rows, cols, blocked):
    """Deterministic result of action `a` from `cell`; wall or edge = stay put."""
    di, dj = _DELTAS[a]
    ni, nj = cell[0] + di, cell[1] + dj
    if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in blocked:
        return (ni, nj)
    return (cell[0], cell[1])


def slip_outcomes(cell, a, rows, cols, blocked, slip):
    """Distribution over the cell physically landed on, splitting `slip` evenly
    across the two perpendicular directions (added only when `slip > 0`)."""
    outcomes: dict[tuple[int, int], float] = {}
    intended = step_cell(cell, a, rows, cols, blocked)
    outcomes[intended] = outcomes.get(intended, 0.0) + (1.0 - slip)
    if slip > 0.0:
        for pa in _PERPENDICULAR[a]:
            slipped = step_cell(cell, pa, rows, cols, blocked)
            outcomes[slipped] = outcomes.get(slipped, 0.0) + slip / 2.0
    return outcomes


class IcyGridWorld:
    """A grid with blocked walls, per-cell slippery ice, passable penalties,
    terminal pits, teleports, and optional shields. Shared discrete env for
    Rooms 1-4.

    Without shields a state is a cell `(i, j)`; with `shields` it becomes
    `(i, j, k)` where `k` marks whether a shield (permanent slip immunity) has
    been collected — needed for the state to stay Markov. Use `cell_of(s)` /
    `shield_of(s)` and `start_state()` so code works for both shapes.
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
        pits=None,
        shields=None,
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
        # Set BEFORE `navigable`/`actions` below: is_terminal() consults it.
        self.pits = {tuple(c): float(r) for c, r in dict(pits or {}).items()}
        self.shields = set(tuple(c) for c in (shields or []))
        # States carry a shield flag ONLY when there is a shield to carry, so
        # every existing room keeps its plain (i, j) states untouched.
        self.stateful = bool(self.shields)
        self.teleports = {tuple(k): tuple(v)
                          for k, v in dict(teleports or {}).items()}

        # Ice defaults to every navigable cell (uniform slip) when unspecified.
        navigable = [c for c in self.cells() if c not in self.blocked]
        if ice is None:
            self.ice = set(navigable)
        else:
            self.ice = set(tuple(c) for c in ice)

        # Rewards keyed by RESULTING state: goal + passable penalties + pits.
        #
        # WARNING: keying by the RESULTING state means a reward on a TELEPORT
        # cell is silently lost — `_build_probs` folds a teleport into its
        # destination, so the lookup happens at the destination and returns 0.
        # Pits are safe because they are terminal (nothing to fold: the agent
        # stops there). Before putting a reward on any transient cell, see the
        # Room 3 section of Plan.md, which documents the fix and why it must
        # average rather than copy.
        # Keyed by resulting STATE, so with shields each rewarded cell is listed
        # once per shield flag: the exit pays +100 whether or not you are holding
        # one.
        self.rewards = {}
        for k in ((0, 1) if self.stateful else (0,)):
            self.rewards[self._state(self.goal, k)] = self.goal_reward
            for c, r in self.penalties.items():
                self.rewards[self._state(c, k)] = r
            for c, r in self.pits.items():
                self.rewards[self._state(c, k)] = r

        # Navigable, non-terminal cells get all four actions. This excludes the
        # goal AND every pit (is_terminal covers both). Teleport cells are
        # excluded: the agent is whisked away the instant it lands on one, so it
        # is never *standing* there to choose an action. Leaving them in would
        # give DP a value for a state that can never be occupied — and would
        # leave a permanent hole in any Q learned from experience.
        # A shielded state on a shield cell is fine; an UNSHIELDED one is not —
        # entering the cell collects the shield, so (shield_cell, 0) can never be
        # occupied. Excluding it is the same reasoning that excludes teleports.
        self.actions = {
            self._state(c, k): ACTION_SPACE
            for c in navigable
            for k in ((0, 1) if self.stateful else (0,))
            if not self.is_terminal(c) and c not in self.teleports
            and not (k == 0 and c in self.shields)
        }

        self.probs, self._phys = self._build_probs()
        self._sampler = {
            key: (list(o.keys()), np.cumsum(np.fromiter(o.values(), float, len(o))))
            for key, o in self._phys.items()
        }
        self.i, self.j = start
        self.k = 1 if start in self.shields else 0
        self.last_landing = start

    # ------------------------------------------------------------------ #
    # Static structure
    # ------------------------------------------------------------------ #
    def cells(self):
        """Every (i, j) on the board, regardless of state shape."""
        return [(i, j) for i in range(self.rows) for j in range(self.cols)]

    def all_states(self):
        """Every state. Cells when there are no shields; (i, j, k) when there are."""
        if not self.stateful:
            return self.cells()
        return [(i, j, k) for (i, j) in self.cells() for k in (0, 1)]

    @staticmethod
    def cell_of(s):
        """The (i, j) of a state, whichever shape it has."""
        return (s[0], s[1])

    @staticmethod
    def shield_of(s) -> int:
        """1 if this state carries a shield. Always 0 in a shield-less room."""
        return s[2] if len(s) > 2 else 0

    def _state(self, cell, k: int):
        """Build a state from a cell + shield flag, matching this grid's shape."""
        return (cell[0], cell[1], k) if self.stateful else (cell[0], cell[1])

    def start_state(self):
        """The state reset() returns — use this, not `.start`, to index V/Q."""
        return self._state(self.start, 1 if self.start in self.shields else 0)

    # These accept a state OR a bare cell: they only ever look at coordinates.
    def is_terminal(self, s) -> bool:
        c = self.cell_of(s)
        return c == self.goal or c in self.pits

    def is_pit(self, s) -> bool:
        return self.cell_of(s) in self.pits

    def is_caught(self, s) -> bool:
        # No moving guard in this env. Defined so the shared episode/rollout code
        # can classify a terminal state uniformly across rooms — Room 4's
        # GuardGrid overrides this with a phase-dependent check.
        return False

    def is_shield(self, s) -> bool:
        return self.cell_of(s) in self.shields

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
        return step_cell((s[0], s[1]), a, self.rows, self.cols, self.blocked)

    def _build_probs(self):
        """Build the transition model, returning (probs, phys): `phys` is the
        distribution over physical landing cells (before teleports), `probs` the
        same with teleports folded into their destinations."""
        probs, phys = {}, {}
        for s in self.actions:  # navigable, non-terminal
            cell, k = self.cell_of(s), self.shield_of(s)
            # A carried shield cancels the slip entirely — that is the whole
            # point of it, and it is exactly why k has to be part of the state.
            slip = self.slip if (self.is_icy(cell) and not k) else 0.0
            for a in ACTION_SPACE:
                outcomes = slip_outcomes(cell, a, self.rows, self.cols,
                                         self.blocked, slip)
                phys[(s, a)] = outcomes

                folded: dict = {}
                for c2, p in outcomes.items():
                    dest = self.teleports.get(c2, c2)
                    # Picked up by TOUCHING the cell — so a shield still counts
                    # if a teleport whisks you off it in the same step.
                    k2 = 1 if (k or c2 in self.shields or dest in self.shields) else 0
                    s2 = self._state(dest, k2)
                    folded[s2] = folded.get(s2, 0.0) + p
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
        self.k = 1 if self.start in self.shields else 0
        self.last_landing = self.start
        return self.current_state()

    def current_state(self):
        return self._state((self.i, self.j), self.k)

    def move(self, action):
        """Take one stochastic step; return the reward of the resulting cell.
        Samples the physical landing cell (recorded in `last_landing`), then
        applies any teleport."""
        cells, cum = self._sampler[(self.current_state(), action)]
        idx = int(np.searchsorted(cum, self.rng.random() * cum[-1], side="right"))
        landed = cells[min(idx, len(cells) - 1)]
        self.last_landing = landed
        dest = self.teleports.get(landed, landed)
        # Mirrors the fold in _build_probs: touching the cell collects it.
        self.k = 1 if (self.k or landed in self.shields
                       or dest in self.shields) else 0
        self.i, self.j = dest
        return self.rewards.get(self.current_state(), 0.0)

    def game_over(self):
        return self.is_terminal(self.current_state())


# ---------------------------------------------------------------------- #
# Random layout generation.
# ---------------------------------------------------------------------- #
def _connected(start, goal, blocked, rows, cols, teleports=None, pits=None) -> bool:
    """True if `goal` is reachable from `start` avoiding blocked cells.
    `teleports` are folded in (step onto one → continue from its destination);
    `pits` are treated as impassable (a route through a pit is not a route)."""
    teleports = teleports or {}
    pits = set(pits or ())
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
            if nxt in blocked or nxt in pits:
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
    exclude=None,
    pits=None,
):
    """Randomly place walls, ice, and negative cells, keeping the board
    solvable (walls added only if start still reaches goal). `pits` marks cells
    the guaranteed route may not use; `exclude` keeps every type off already-
    claimed cells. Returns (blocked, ice, negatives) as sets of (i, j)."""
    rng = random.Random(seed)
    exclude = set(exclude or ())
    pits = set(pits or ())
    free = [s for s in ((i, j) for i in range(rows) for j in range(cols))
            if s != start and s != goal and s not in exclude]
    rng.shuffle(free)

    blocked: set = set()
    for c in free:
        if len(blocked) >= n_blocked:
            break
        blocked.add(c)
        if not _connected(start, goal, blocked, rows, cols, pits=pits):
            blocked.discard(c)  # would wall off the goal — skip it

    remaining = [c for c in free if c not in blocked]
    rng.shuffle(remaining)
    ice = set(remaining[:n_slippery])
    negatives = set(remaining[n_slippery:n_slippery + n_negative])
    return blocked, ice, negatives


def generate_shields(
    blocked,
    n_shields: int,
    seed: int,
    rows: int = 10,
    cols: int = 10,
    start: tuple[int, int] = (9, 0),
    goal: tuple[int, int] = (0, 9),
    exclude=None,
    pits=None,
):
    """Place Room 3's shield pickups, each reachable from the start and able to
    reach the goal without crossing a pit. Returns a set of (i, j), possibly
    fewer than `n_shields` if the board ran out of room."""
    rng = random.Random(seed)
    exclude = set(exclude or ())
    pits = set(pits or ())
    free = [c for c in ((i, j) for i in range(rows) for j in range(cols))
            if c != start and c != goal and c not in blocked
            and c not in exclude and c not in pits]
    rng.shuffle(free)

    shields: set = set()
    for c in free:
        if len(shields) >= n_shields:
            break
        if (_connected(start, c, blocked, rows, cols, pits=pits)
                and _connected(c, goal, blocked, rows, cols, pits=pits)):
            shields.add(c)
    return shields


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
    """Place Room 2's portal traps, one at a time, keeping each only if the
    start still reaches the goal through the folded model (so a portal never
    seals the exit off). `exclude` keeps portals off already-claimed cells.
    Returns a set of (i, j), possibly fewer than `n_portals`."""
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
