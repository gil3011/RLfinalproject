from __future__ import annotations


def rollout(grid, policy, gamma: float = 1.0, max_steps: int = 200,
            with_landings: bool = False):
    """Play one episode under `policy`; return (path, discounted return G,
    outcome). `with_landings` also returns each step's physical landing cell
    (differs from path only where a teleport fired). Outcome is one of "goal",
    "fell", "caught", "timeout"."""
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
