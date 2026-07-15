"""
Shared episode utilities for the discrete rooms (1-4).

`rollout` runs a single policy episode through the REAL stochastic environment
(so slips actually happen) and returns the visited path plus outcome — the data
the ▶️ Play Episode animation replays cell-by-cell.
"""
from __future__ import annotations


def rollout(grid, policy, gamma: float = 1.0, max_steps: int = 200):
    """Play one episode following `policy` from the grid's start state.

    Parameters
    ----------
    grid   : an IcyGridWorld (its move()/reset() apply the stochastic slip).
    policy : dict {state: action}.
    gamma  : discount factor used to compute the return G, so G is defined the
             same way as the value function V — the expected discounted return.
    max_steps : cap on episode length (a slip-heavy run may never reach the goal).

    Returns
    -------
    path    : list of states visited, start .. terminal (or until the cap).
    G       : discounted return  G = Σ_t γ^t · r_{t+1}  (matches how V is defined).
    outcome : "goal" if the exit was reached, else "timeout".
    """
    s = grid.reset()
    path = [s]
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
        if grid.is_terminal(s):
            break

    outcome = "goal" if s == grid.goal else "timeout"
    return path, G, outcome
