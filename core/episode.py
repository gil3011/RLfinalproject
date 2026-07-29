"""
Shared episode utilities for the discrete rooms (1-4).

`rollout` runs a single policy episode through the REAL stochastic environment
(so slips actually happen) and returns the visited path plus outcome — the data
the ▶️ Play Episode animation replays cell-by-cell.

The number shown on screen is simply that episode's real discounted return `G`,
with NO floor of any kind. Every outcome is scored the same way: a win shows the
discounted +100 exit; a fall or catch shows the discounted -100 it paid, so an
early death (≈ -100) is worse than a late one (≈ 0); a timeout shows its raw
discounted return (≈ 0 on a board with no step cost). This makes the scoreboard
match the LEARNER, whose Bellman backup already discounts those same terminal
rewards by γ.
"""
from __future__ import annotations


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
              separately; the scoreboard just shows each episode's real
              discounted return G, no floor.
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
