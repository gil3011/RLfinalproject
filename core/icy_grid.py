"""
IcyGridWorld — a 10x10 grid world with several special cell types.

Shared discrete environment for Rooms 1-4. Cells come in these flavours:

  * blocked  — walls the agent cannot step into (a move toward one keeps it put),
  * ice      — slippery cells: a move slips perpendicular with prob `slip`,
  * penalty  — passable cells that yield a negative reward each time they are
               entered (NON-terminal; the agent walks on and keeps playing),
  * teleport — cells that bounce the agent elsewhere the instant it lands on one
               (Room 2's portal traps, which send it back to the start),
  * pit      — TERMINAL hazard cells that end the episode with a negative reward
               (Room 3's abyss: falling in is fatal, not a detour).

The goal is terminal (reward `goal_reward`, +100 by default), and so is every
pit. A pit needs no special reward machinery precisely BECAUSE it is terminal:
the agent ends up standing on it, so the resulting-state reward lookup below
carries its penalty like any other cell. Contrast a penalty on a *teleport*
cell, which would be silently lost — see the warning on `rewards`.

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
    """Distribution over the cell physically landed on, applying perpendicular slip.

    `slip` is the total probability of slipping (split evenly across the two
    perpendicular directions). Slip outcomes are added ONLY when `slip > 0`:
    emitting zero-probability outcomes leaves landmines for anything that later
    divides by an outcome's mass (see the Room 3 teleport-reward fold).
    """
    outcomes: dict[tuple[int, int], float] = {}
    intended = step_cell(cell, a, rows, cols, blocked)
    outcomes[intended] = outcomes.get(intended, 0.0) + (1.0 - slip)
    if slip > 0.0:
        for pa in _PERPENDICULAR[a]:
            slipped = step_cell(cell, pa, rows, cols, blocked)
            outcomes[slipped] = outcomes.get(slipped, 0.0) + slip / 2.0
    return outcomes


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
    pits        : dict {cell: reward} of TERMINAL hazard cells — entering one
                  ends the episode with that reward (Room 3's abyss). Like the
                  goal, a pit is excluded from `.actions`: the episode is over,
                  so the agent never chooses an action from it.
    shields     : iterable of pickup cells. Collecting one makes the agent immune
                  to slip FOR THE REST OF THE EPISODE. See the note on state
                  shape below — this is the one option that changes what a state
                  IS, so it is opt-in and off by default.
    slip        : slip probability on ice cells.
    goal_reward : terminal reward at the goal.
    teleports   : dict {cell: destination} — landing on `cell` moves the agent to
                  `destination` in the same step (Room 2's portals). Folded into
                  `.probs`, so teleport cells are transient, never occupied.
    rng         : numpy Generator used by move(); defaults to a fresh one. Pass a
                  seeded generator for reproducible rollouts.

    STATE SHAPE — read this before using `shields`
    ----------------------------------------------
    Without shields a state IS a cell: `(i, j)`. Every room up to Room 2 works
    this way and is completely unaffected by the paragraph below.

    A shield is *carried*, so with `shields` a state becomes `(i, j, k)` where
    `k` is 1 once a shield has been picked up. This is not decoration: whether a
    move slips depends on `k`, so `(i, j)` alone would NOT be Markov — two agents
    on the same icy cell behave differently depending on what they collected ten
    steps ago. Keying the model by the cell alone would leave DP computing a `V*`
    that is quietly wrong (an average over "sometimes shielded"), and it would
    hand the TD learners an unlearnable, self-contradicting target.

    The algorithms need no changes for this: `value_iteration`, `policy_value`
    and `sarsa_control` all treat a state as an opaque dict key. Only code that
    reads coordinates out of a state has to care — use `cell_of(s)` /
    `shield_of(s)` rather than unpacking, and `start_state()` rather than
    `.start`, and such code works for both shapes.
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
        applies any teleport. Sampling is a single uniform draw against a
        precomputed cumulative distribution rather than `np.random.choice`,
        which costs ~10us per call and dominates Room 2's ~1.5M training steps.
        """
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

    `teleports` (if given) is folded in exactly as the transition model folds it:
    stepping onto a teleport cell continues from its destination instead. A
    teleport is therefore never a state you can stand on, which means portals can
    seal the exit off without walling it — see `generate_portals`.

    `pits` are treated as IMPASSABLE, which is not the same as blocked: the agent
    can very much enter a pit, it just never comes out — the episode ends there.
    So a route "through" the abyss is not a route, and a reachability check that
    walked over pits would happily certify an unsolvable board.
    """
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
    """Randomly place the three cell types on the board.

    Blocked cells are placed one at a time and only kept if the start still
    reaches the goal, so the board is always solvable. Slippery and negative
    cells are sampled from the remaining free cells and are mutually exclusive.

    `pits` is the set of cells the guaranteed route may NOT use. It must include
    a board's real pits (Room 3's abyss), or the guard will certify a wall that
    seals the only SAFE route because it found a "path" straight through the
    chasm. A caller may add more cells to demand a route that avoids them too —
    Room 3 adds the *ledge* so walls can never make hugging the cliff the only
    way out. Marking cells here affects the GUARD only; they stay walkable in the
    environment itself.

    `exclude` keeps every type off cells already claimed by another room's
    hazard (Room 3's abyss). Cosmetic for pits — a pit is terminal, so its
    iciness never enters the model — but a cell drawn as two hazards at once
    just reads as a bug. Mirrors `generate_portals`' argument of the same name.

    Returns (blocked, ice, negatives) as sets of (i, j).
    """
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
    """Place Room 3's shield pickups.

    A shield only means anything if the agent can get to it AND still get out,
    so each candidate must be reachable from the start and able to reach the
    goal — both without crossing a pit. An unreachable shield is not a hazard
    (nothing breaks), it is just a lie drawn on the board: the user sees a way to
    beat the ice that no policy can ever take.

    Unlike walls, a shield never makes a board unsolvable, so there is no
    incremental connectivity guard here — placement cannot fail. Returns a set
    of (i, j), possibly smaller than `n_shields` if the board ran out of room.
    """
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
