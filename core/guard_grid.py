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
    """The guard's patrol as a cyclic sequence of cells: `rows` of `col` swept
    top→bottom then back, period `2·len(rows) − 2`."""
    rows = list(rows)
    path = rows + rows[-2:0:-1]     # 0..7, 6..1  → no repeated endpoints
    return tuple((r, col) for r in path)


class GuardGrid:
    """Room 4's grid: cliff + moving guard + one coin. States are `(i, j, g, m)`
    (cell, guard phase, coin mask). `coins` are cells that pay `coin_value` once
    on entry; `track` is the guard's schedule; `rng` seeds move()."""

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
        """R(s, a, s'): goal/fall/catch pay on the resulting cell; the coin pays
        on the mask bit that newly flipped."""
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
        """Build (probs, phys, rewards): `phys` over physical landing cells,
        `probs` the Markov model over resulting states, `rewards` = R(s,a,s2)."""
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
    """Pick the single bonus coin's cell on the central ledge (seeded by
    `seed`). Returns a 1-tuple of the cell."""
    return (random.Random(seed).choice(_COIN_CELLS),)
