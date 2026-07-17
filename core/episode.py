"""
Shared episode utilities for the discrete rooms (1-4).

`rollout` runs a single policy episode through the REAL stochastic environment
(so slips actually happen) and returns the visited path plus outcome — the data
the ▶️ Play Episode animation replays cell-by-cell.

`scored_return` turns that episode's return into the number shown on screen,
adding a penalty when the agent never escaped. See its docstring for why the
scoreboard and the maths deliberately use different numbers.
"""
from __future__ import annotations

# Penalty added to a timed-out episode's REPORTED return. Mirrors the
# standardized +100 goal reward: escaping is +100, failing to escape is -100.
TIMEOUT_PENALTY = -100.0


def scored_return(G: float, outcome: str, penalty: float = TIMEOUT_PENALTY):
    """The episode's score for display: G, plus `penalty` if it TIMED OUT.

    Raw G ranks giving up ABOVE escaping the hard way. A run that wanders and
    times out without touching a penalty cell scores G = 0, while a run that
    escapes across two -20 cells scores about -25 — so the scoreboard rewards
    quitting. Adding `penalty` on timeout restores the intended ordering: not
    finishing is always the worst outcome. -100 sits below the worst successful
    return the optimal policy produces (measured at about -47 on the harshest
    settings), so a timeout genuinely ranks last.

    This keys on "timeout" specifically, NOT on "did not reach the goal". A
    Room 3 "fell" episode already paid the environment's real cliff penalty, so
    charging it the timeout penalty too would double-count (-200 for a -100
    fall) and misreport a decisive death as a failure to decide. Rooms 1-2 are
    unaffected: with no pits, "not the goal" and "timeout" are the same set.

    IMPORTANT: this deliberately breaks `G ≈ V(S)`. The scored return of a
    timed-out episode is NOT its discounted return, and averaging scored returns
    does NOT give the value function. This is a SCOREBOARD number, for the player
    only. Everything doing maths — V, Q, MC's updates, the training curves, the
    DP benchmark — must use the raw G that `rollout` returns, never this.
    """
    return G + penalty if outcome == "timeout" else G


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
              "timeout" — still wandering when the step cap ran out.
              A fall is NOT a timeout: it ended the episode decisively and
              already paid the pit's reward, so `scored_return` leaves it alone.
    landings (only if `with_landings`) : same length as `path`; `landings[k]`
             is the cell step k physically landed on. It differs from `path[k]`
             exactly when a teleport fired, which is the frame an animation must
             draw to explain the jump. `landings[0]` is the start.
    """
    s = grid.reset()
    path = [s]
    landings = [s]
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
        landings.append(grid.last_landing)
        if grid.is_terminal(s):
            break

    # Compare the CELL, not the state: a shielded room's states are (i, j, k),
    # so `s == grid.goal` would never be true and every escape would be
    # misreported as a timeout.
    if grid.cell_of(s) == grid.goal:
        outcome = "goal"
    elif grid.is_terminal(s):
        outcome = "fell"
    else:
        outcome = "timeout"
    if with_landings:
        return path, G, outcome, landings
    return path, G, outcome
