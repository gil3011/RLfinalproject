from __future__ import annotations

from core.icy_grid import ACTION_SPACE

THETA = 1e-3  # convergence threshold (hardcoded per Plan.md)


def _q_value(grid, rewards, V, s, a, gamma):
    """Expected one-step return of taking action `a` in state `s`."""
    total = 0.0
    for s2, p in grid.probs[(s, a)].items():
        r = rewards.get((s, a, s2), 0.0)
        total += p * (r + gamma * V[s2])
    return total


def greedy_policy(grid, V, gamma, rewards=None):
    """Derive the greedy policy w.r.t. a value function V."""
    if rewards is None:
        _, rewards = grid.get_transition_probs_and_rewards()
    policy = {}
    for s in grid.actions:  # navigable, non-terminal cells
        best_a, best_v = None, float("-inf")
        for a in ACTION_SPACE:
            v = _q_value(grid, rewards, V, s, a, gamma)
            if v > best_v:
                best_v, best_a = v, a
        policy[s] = best_a
    return policy


def value_iteration(grid, gamma: float = 0.9, theta: float = THETA, max_iters: int = 1000):
    """Run value iteration, recording a snapshot per sweep. Returns
    (V, policy, history)."""
    _, rewards = grid.get_transition_probs_and_rewards()
    states = grid.all_states()
    V = {s: 0.0 for s in states}
    history = []

    for _ in range(max_iters):
        biggest_change = 0.0
        for s in grid.actions:  # navigable, non-terminal cells
            old_v = V[s]
            best_v = float("-inf")
            for a in ACTION_SPACE:
                v = _q_value(grid, rewards, V, s, a, gamma)
                if v > best_v:
                    best_v = v
            V[s] = best_v
            biggest_change = max(biggest_change, abs(old_v - V[s]))
        policy = greedy_policy(grid, V, gamma, rewards)
        history.append({"V": dict(V), "policy": policy, "delta": biggest_change})
        if biggest_change < theta:
            break

    return dict(V), history[-1]["policy"], history


def policy_iteration(grid, gamma: float = 0.9, theta: float = THETA, max_iters: int = 1000):
    """Run policy iteration, recording a snapshot per improvement round (full
    evaluation + greedy improvement). Returns (V, policy, history)."""
    _, rewards = grid.get_transition_probs_and_rewards()
    states = grid.all_states()

    # Deterministic initial policy (first available action) — reproducible.
    policy = {s: ACTION_SPACE[0] for s in grid.actions}
    V = {s: 0.0 for s in states}
    prev_V = dict(V)
    history = []

    for _ in range(max_iters):
        # --- policy evaluation (iterative, to convergence) ---
        while True:
            biggest_change = 0.0
            for s in grid.actions:  # navigable, non-terminal cells
                old_v = V[s]
                V[s] = _q_value(grid, rewards, V, s, policy[s], gamma)
                biggest_change = max(biggest_change, abs(old_v - V[s]))
            if biggest_change < theta:
                break

        # --- policy improvement ---
        stable = True
        for s in grid.actions:  # navigable, non-terminal cells
            old_a = policy[s]
            best_a, best_v = None, float("-inf")
            for a in ACTION_SPACE:
                v = _q_value(grid, rewards, V, s, a, gamma)
                if v > best_v:
                    best_v, best_a = v, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False

        delta = max((abs(V[s] - prev_V[s]) for s in states), default=0.0)
        history.append({"V": dict(V), "policy": dict(policy), "delta": delta})
        prev_V = dict(V)

        if stable:
            break

    return dict(V), history[-1]["policy"], history


def policy_value(grid, policy, gamma: float, theta: float = THETA,
                 max_iters: int = 10000):
    """Exact expected discounted return of following `policy`, per state, by
    iterative policy evaluation against the true model. Benchmark only."""
    _, rewards = grid.get_transition_probs_and_rewards()
    V = {s: 0.0 for s in grid.all_states()}
    for _ in range(max_iters):
        biggest_change = 0.0
        for s in grid.actions:  # navigable, non-terminal cells
            a = policy.get(s)
            if a is None:
                continue
            old_v = V[s]
            V[s] = _q_value(grid, rewards, V, s, a, gamma)
            biggest_change = max(biggest_change, abs(old_v - V[s]))
        if biggest_change < theta:
            break
    return V


def expected_steps_to_goal(grid, policy, tol: float = 1e-6, max_iters: int = 20000):
    """Expected steps from start to goal under `policy`, solving the hitting-
    time equations by iteration. Only meaningful when the goal is the only
    terminal (Room 1)."""
    t = {s: 0.0 for s in grid.all_states()}
    for _ in range(max_iters):
        biggest = 0.0
        for s in grid.actions:  # navigable, non-terminal
            old = t[s]
            val = 1.0
            for s2, p in grid.probs[(s, policy[s])].items():
                val += p * t[s2]
            t[s] = val
            biggest = max(biggest, abs(val - old))
        if biggest < tol:
            break
    return t[grid.start_state()]


def success_prob_within(grid, policy, max_steps: int):
    """Probability the start reaches the goal within `max_steps` under `policy`,
    propagating the state distribution forward and absorbing goal mass (mass
    landing on any other terminal is dropped)."""
    dist = {grid.start_state(): 1.0}
    reached = 0.0
    for _ in range(max_steps):
        new: dict = {}
        for s, m in dist.items():
            for s2, p in grid.probs[(s, policy[s])].items():
                new[s2] = new.get(s2, 0.0) + m * p
        # Match on the CELL, not the state: with shields the exit is reachable
        # both shielded and not, and `grid.goal` is a cell, never a state key.
        reached += sum(m for s, m in new.items() if grid.cell_of(s) == grid.goal)
        # Absorb (and discard) mass that fell into a pit — dead runs.
        dist = {s: m for s, m in new.items() if not grid.is_terminal(s)}
        if not dist:
            break
    return reached


ALGORITHMS = {
    "Value Iteration": value_iteration,
    "Policy Iteration": policy_iteration,
}
