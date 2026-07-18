"""
GuardGrid — Room 4's environment: Room 3's cliff board plus a moving patrol
guard and a single bonus coin.

This is a SEPARATE env class from `IcyGridWorld`, not an extension of it, and the
choice is deliberate. Room 3's shield augments the state with a latch (`k` flips
0→1 on pickup and never flips back); Room 4 augments it with two structurally
different things:

  * a guard PHASE `g` that advances every single step, deterministically, and
    cycles — nothing like a latch, and
  * a coin MASK `m` whose reward is earned by the bit FLIPPING, i.e. a reward on
    the TRANSITION, which `IcyGridWorld` (rewards keyed by the resulting state)
    cannot express.

So a state here is `(i, j, g, m)`. Bolting a cycling phase and transition-rewards
onto `IcyGridWorld`'s shield machinery would tangle a class three working rooms
depend on. Instead GuardGrid reuses the one thing that MUST stay shared — the
slip physics (`step_cell` / `slip_outcomes`) — and implements its own
augmentation. The RL algorithms never notice: `value_iteration`, `policy_value`,
`sarsa_control` and `q_learning_control` all treat a state as an opaque dict key,
exactly as they do for Room 3's `(i, j, k)`. Only code reading coordinates out of
a state must use `cell_of` / `phase_of` / `mask_of`, mirroring Room 3's
`cell_of` / `shield_of`.

THE GUARD — Markov, and why co-location.
----------------------------------------
The guard patrols a fixed column (rows 0–7 of column `GUARD_COL`) back and forth,
as a cyclic schedule of cells of period `P`. Its position is a pure function of
the phase, so `(cell, g)` stays Markov. Every step advances `g → (g+1) % P`, and
the guard is at `track[g]`.

A catch is CO-LOCATION at the resulting phase: the agent is caught iff it ends a
move on the guard's new cell. This is a pure property of the resulting state
`(c2, g+1)` — `is_caught(s)` needs only `s` — which is what keeps it Markov and
lets `is_terminal` stay a one-argument predicate everywhere DP and TD consult it.
A stationary agent (bumping a wall) can be run over; the agent, knowing the phase,
can predict exactly where the guard will be and time its crossing. The one thing
this model does NOT catch is a SWAP (agent A→B while the guard goes B→A, passing
through each other): they never share a cell at a single phase, so it reads as a
near miss. That is a minor cheese, documented rather than modelled — making it
terminal would require a distinct absorbing "caught" state and buys nothing the
measurements needed.

THE COIN — a reward on the transition.
--------------------------------------
A one-shot coin CANNOT be keyed by the resulting state: once collected its bit
stays set, so re-entering the cell yields an identical `s'` and would pay again,
forever. What earns the coin is the mask bit FLIPPING, visible only in
`(s, s')`. So rewards here are `R(s, a, s')`, built in `_build`. `IcyGridWorld`
has a warning about exactly this (a reward on a transient/teleport cell being
silently dropped); Room 4 is the room that needed the general fix.
"""
from __future__ import annotations

import random

import numpy as np

from core.icy_grid import ACTION_SPACE, slip_outcomes

# Fixed geometry — Room 3's board. The abyss, patrol column, start and exit are
# the lesson, so they never move (🎲 Regenerate only reshuffles walls/ice/coin).
START = (9, 0)
GOAL = (9, 9)
CLIFF = frozenset((9, j) for j in range(1, 9))          # the abyss (terminal)
LEDGE = frozenset((8, j) for j in range(1, 9))          # cells hugging the abyss
GUARD_COL = 5
GUARD_ROWS = range(0, 8)                                  # rows 0–7 of GUARD_COL
# Both "you didn't make it" hazards cost the same, and mirror the +100 exit and
# the −100 scoreboard penalty: fall −100, caught −100, give up −100.
CLIFF_REWARD = -100.0
CAUGHT_REWARD = -100.0


def make_track(col: int = GUARD_COL, rows=GUARD_ROWS):
    """The guard's patrol as a cyclic sequence of cells.

    Sweeps `rows` of `col` top→bottom then back, so the period is
    `2·len(rows) − 2` (the two turn-around cells are not repeated). For rows 0–7
    that is P = 14. The guard advances one cell per environment step.

    The FULL column matters: measured, any patrol that leaves a row permanently
    unoccupied (a shorter sweep, or a "faster" guard that skips rows) hands the
    agent a free crossing and the guard stops mattering. See Plan.md §Room 4.
    """
    rows = list(rows)
    path = rows + rows[-2:0:-1]     # 0..7, 6..1  → no repeated endpoints
    return tuple((r, col) for r in path)


class GuardGrid:
    """Room 4's grid: cliff + moving guard + one coin. States are `(i, j, g, m)`.

    Parameters
    ----------
    blocked   : wall cells.
    ice       : slippery cells.
    coins     : iterable of coin cells (Room 4 ships exactly one). Each gets a bit
                in the mask; entering the cell sets it and pays `coin_value` ONCE.
    track     : the guard's cyclic cell schedule (from `make_track`).
    slip      : slip probability on ice cells.
    goal_reward, coin_value : the two knobs whose RATIO the room is about.
    rng       : Generator for move(); pass a seeded one for reproducible rollouts,
                leave None (fresh entropy) for the ▶️ Play grid so it slips anew.
    """

    def __init__(self, blocked=None, ice=None, coins=(), track=None, slip: float = 0.1,
                 goal_reward: float = 100.0, coin_value: float = 5.0, rng=None):
        self.rows = self.cols = 10
        self.start, self.goal = START, GOAL
        self.slip = float(slip)
        self.goal_reward = float(goal_reward)
        self.coin_value = float(coin_value)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.blocked = set(map(tuple, blocked or []))
        self.ice = set(map(tuple, ice or []))
        self.pits = {c: CLIFF_REWARD for c in CLIFF}
        self.coins = tuple(map(tuple, coins))
        self.track = tuple(track) if track is not None else make_track()
        self.P = len(self.track)
        self._guard_cells = frozenset(self.track)
        self.n_coins = len(self.coins)
        self.n_masks = 1 << self.n_coins
        self._bit = {c: 1 << i for i, c in enumerate(self.coins)}

        navigable = [c for c in self.cells() if c not in self.blocked]
        # A coin cell with its bit UNSET is never a state you can be standing in —
        # entering it collects the coin and sets the bit — so it is excluded, the
        # same reasoning that excludes (shield_cell, unshielded) in Room 3.
        self.actions = {
            (c[0], c[1], g, m): ACTION_SPACE
            for c in navigable
            for g in range(self.P)
            for m in range(self.n_masks)
            if not self.is_terminal((c[0], c[1], g, m))
            and not (c in self._bit and not (m & self._bit[c]))
        }
        self.probs, self._phys, self._rew = self._build()
        self._sampler = {
            key: (list(o.keys()), np.cumsum(np.fromiter(o.values(), float, len(o))))
            for key, o in self._phys.items()
        }
        self.reset()

    # ------------------------------------------------------------------ #
    # Static structure
    # ------------------------------------------------------------------ #
    def cells(self):
        return [(i, j) for i in range(self.rows) for j in range(self.cols)]

    def all_states(self):
        return [(i, j, g, m) for (i, j) in self.cells()
                for g in range(self.P) for m in range(self.n_masks)]

    @staticmethod
    def cell_of(s):
        return (s[0], s[1])

    @staticmethod
    def phase_of(s):
        return s[2]

    @staticmethod
    def mask_of(s):
        return s[3]

    def guard_at(self, g):
        return self.track[g % self.P]

    def start_state(self):
        return (START[0], START[1], 0, 0)

    # These accept a state (i, j, g, m); is_blocked/is_icy take a bare cell.
    def is_caught(self, s) -> bool:
        return self.cell_of(s) == self.guard_at(self.phase_of(s))

    def is_pit(self, s) -> bool:
        return self.cell_of(s) in self.pits

    def is_terminal(self, s) -> bool:
        c = self.cell_of(s)
        return c == self.goal or c in self.pits or self.is_caught(s)

    def is_blocked(self, c) -> bool:
        return c in self.blocked

    def is_icy(self, c) -> bool:
        return c in self.ice

    def is_coin(self, c) -> bool:
        return c in self._bit

    # ------------------------------------------------------------------ #
    # Transition model
    # ------------------------------------------------------------------ #
    def _reward(self, s, s2) -> float:
        """R(s, a, s') — a function of the TRANSITION (see the module docstring).

        Goal / fall / catch pay on the resulting cell; the coin pays on the mask
        bit that newly flipped, which only the (s, s') pair reveals.
        """
        c2 = self.cell_of(s2)
        r = 0.0
        if c2 == self.goal:
            r += self.goal_reward
        if c2 in self.pits:
            r += self.pits[c2]
        if self.is_caught(s2):
            r += CAUGHT_REWARD
        gained = self.mask_of(s2) & ~self.mask_of(s)
        if gained:
            r += self.coin_value * bin(gained).count("1")
        return r

    def _build(self):
        """Build (probs, phys, rewards).

        `phys[(s,a)]`  : distribution over the physical landing CELL (for move()).
        `probs[(s,a)]` : the Markov model over resulting STATES — the physical
                         cell, the advanced phase g+1, and the updated coin mask.
        `rewards[(s,a,s2)]` : R(s, a, s').
        """
        probs, phys, rew = {}, {}, {}
        for s in self.actions:
            c, g, m = self.cell_of(s), self.phase_of(s), self.mask_of(s)
            slip = self.slip if self.is_icy(c) else 0.0
            g2 = (g + 1) % self.P
            for a in ACTION_SPACE:
                out = slip_outcomes(c, a, self.rows, self.cols, self.blocked, slip)
                phys[(s, a)] = out
                folded: dict = {}
                for c2, p in out.items():
                    m2 = m | self._bit.get(c2, 0)
                    s2 = (c2[0], c2[1], g2, m2)
                    folded[s2] = folded.get(s2, 0.0) + p
                probs[(s, a)] = folded
                for s2 in folded:
                    rew[(s, a, s2)] = self._reward(s, s2)
        return probs, phys, rew

    def get_transition_probs_and_rewards(self):
        """Return (transition_probs, rewards) in (s, a, s') form for the DP code."""
        transition_probs = {}
        for (s, a), out in self.probs.items():
            for s2, p in out.items():
                transition_probs[(s, a, s2)] = p
        return transition_probs, dict(self._rew)

    # ------------------------------------------------------------------ #
    # Live simulation (for animating a policy)
    # ------------------------------------------------------------------ #
    def reset(self):
        self.s = self.start_state()
        return self.s

    def current_state(self):
        return self.s

    def move(self, action):
        """Take one stochastic step; return R(s, a, s') for the resulting state."""
        cells, cum = self._sampler[(self.s, action)]
        idx = int(np.searchsorted(cum, self.rng.random() * cum[-1], side="right"))
        c2 = cells[min(idx, len(cells) - 1)]
        s_old = self.s
        m2 = self.mask_of(s_old) | self._bit.get(c2, 0)
        self.s = (c2[0], c2[1], (self.phase_of(s_old) + 1) % self.P, m2)
        return self._reward(s_old, self.s)

    def game_over(self):
        return self.is_terminal(self.s)


# ---------------------------------------------------------------------- #
# Coin placement.
# ---------------------------------------------------------------------- #
# The coin sits on the CENTRAL ledge (columns 3–6 of row 8), never its ends. A
# coin one step from the start or hard against the exit is collected almost for
# free — it must sit far enough out that taking it means genuinely hugging the
# abyss, or it poses no dilemma. Placed BEFORE walls and excluded from them, so a
# wall never lands on it.
_COIN_CELLS = tuple((8, j) for j in range(3, 7))


def place_coin(seed: int):
    """Pick the single bonus coin's cell on the central ledge.

    Reshuffled by 🎲 Regenerate via `seed`. Kept on the ledge because the coin's
    whole purpose is to lure the agent onto the risky route; a coin on the safe
    detour would teach nothing. Returns a 1-tuple of the cell.
    """
    return (random.Random(seed).choice(_COIN_CELLS),)
