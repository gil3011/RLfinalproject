"""
Shared episode utilities for the discrete rooms (1-4).

`rollout` runs a single policy episode through the REAL stochastic environment
(so slips actually happen) and returns the visited path plus outcome — the data
the ▶️ Play Episode animation replays cell-by-cell.

`scored_return` turns that episode's return into the number shown on screen: the
real return on a win, a flat -100 for any loss. See its docstring for why the
scoreboard and the maths deliberately use different numbers.
"""
from __future__ import annotations

# The flat score shown for ANY lost episode, mirroring the standardized +100 exit:
# escaping is worth its real (positive) return, and every way of NOT escaping —
# falling, being caught, or timing out — is worth exactly this, no matter when or
# how it happened.
LOSS_SCORE = -100.0
# Back-compat alias: rooms import this name in their help text.
TIMEOUT_PENALTY = LOSS_SCORE


def scored_return(G: float, outcome: str) -> float:
    """The episode's score for display: the real discounted return G on a WIN, a
    flat LOSS_SCORE on ANY loss.

    Every non-goal outcome — "fell", "caught", or "timeout" — scores exactly
    LOSS_SCORE, regardless of WHEN it ended (an early fall and a late one score the
    same) and HOW (abyss, guard, or the clock running out). A win shows its true
    discounted G, which is always well above LOSS_SCORE because the exit is the only
    positive reward — so escaping always outranks every way of failing, and all
    failures tie.

    Why REPLACE G rather than penalise it: raw G ranks giving up ABOVE escaping the
    hard way (a wander-and-timeout scores ~0, an escape across cost cells scores
    negative). An earlier version ADDED a penalty on timeout only, which left a fall
    or a catch showing its raw *discounted* value (~-80 for an early fall, less for a
    late one) — inconsistent across losses and across timing. Returning a flat
    LOSS_SCORE for every loss makes the scoreboard read the way a player expects:
    win = your score, loss = -100, full stop. It also cannot double-count the
    environment's own -100 that a fall/catch already paid, because it adds nothing.

    IMPORTANT: this deliberately breaks `G ≈ V(S)`. Averaging scored returns does
    NOT give the value function. This is a SCOREBOARD number, for the player only.
    Everything doing maths — V, Q, the learners' updates, the training curves, the
    DP benchmark — must use the raw G that `rollout` returns, never this.
    """
    return G if outcome == "goal" else LOSS_SCORE


def rollout(grid, policy, gamma: float = 1.0, max_steps: int = 200,
            with_landings: bool = False):
    """Play one episode following `policy` from the grid's start state.

    Parameters
    ----------
    grid   : an IcyGridWorld (its move()/reset() apply the stochastic slip).
    policy : dict {state: action}.
    gamma  : discount factor used to compute the return G, so G is defined the
             same way as the value function V — the expected discounted return.
    max_steps : cap on episode length (a slip-heavy run may never reach the goal).
    with_landings : also return the PHYSICAL landing cell of each step, before
             any teleport fired. Opt-in so rooms without portals (1, 3, 4) keep
             the plain 3-tuple contract.

    Returns
    -------
    path    : list of states visited, start .. terminal (or until the cap).
    G       : discounted return  G = Σ_t γ^t · r_{t+1}  (matches how V is defined).
    outcome : "goal"    — reached the exit,
              "fell"    — ended on a terminal hazard (Room 3's abyss),
              "caught"  — caught by Room 4's patrol guard (also terminal),
              "timeout" — still wandering when the step cap ran out.
              The three losses stay distinct because the KPIs count them
              separately; the SCOREBOARD (`scored_return`) treats all three
              alike, as a flat -100.
    landings (only if `with_landings`) : same length as `path`; `landings[k]`
             is the cell step k physically landed on. It differs from `path[k]`
             exactly when a teleport fired, which is the frame an animation must
             draw to explain the jump. `landings[0]` is the start.
    """
    s = grid.reset()
    path = [s]
    # Only track landings when asked: it reads `grid.last_landing`, which only the
    # teleport-capable IcyGridWorld exposes. A guard/coin grid has no transient
    # landings, so requiring the attribute everywhere would be false coupling.
    landings = [s] if with_landings else None
    G = 0.0
    discount = 1.0

    for _ in range(max_steps):
        if grid.is_terminal(s):
            break
        a = policy.get(s)
        if a is None:
            break
        r = grid.move(a)
        s = grid.current_state()
        G += discount * r
        discount *= gamma
        path.append(s)
        if with_landings:
            landings.append(grid.last_landing)
        if grid.is_terminal(s):
            break

    # Compare the CELL, not the state: an augmented room's states are (i, j, ...),
    # so `s == grid.goal` would never be true and every escape would be
    # misreported as a timeout. Check caught before fell: both are terminal, but
    # they are different deaths and the KPIs count them separately (Room 4).
    if grid.cell_of(s) == grid.goal:
        outcome = "goal"
    elif grid.is_caught(s):
        outcome = "caught"
    elif grid.is_terminal(s):
        outcome = "fell"
    else:
        outcome = "timeout"
    if with_landings:
        return path, G, outcome, landings
    return path, G, outcome
