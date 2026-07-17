"""
Dynamic Programming for Room 1 — value iteration and policy iteration.

Adapted from the reference `value_iteration.py` / `policy_iteration_*.py`,
extended so BOTH algorithms return a per-iteration `history`: a list of
snapshots {"V", "policy", "delta"}, one per iteration. This drives:
  * the KPIs (iterations = len(history), final Δ = history[-1]["delta"]),
  * the log-scale convergence curve ([h["delta"] for h in history]), and
  * the "view policy at iteration k" scrubber.

`delta` is the largest change in V during that iteration; it drives to 0 as the
algorithm converges, for both methods.

The convergence threshold theta is hardcoded at 1e-3 per the plan.
"""
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
    """Run value iteration, recording a snapshot after every sweep.

    Returns (V, policy, history).
    """
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
    """Run policy iteration, recording a snapshot after every improvement round.

    Each round = full iterative policy evaluation of the current policy, then a
    greedy improvement. `delta` for the round is the largest change in V since the
    previous round's evaluated values. Returns (V, policy, history).
    """
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
    """Exact expected discounted return of following `policy` — per state.

    This is the same iterative policy evaluation `policy_iteration` runs
    internally, exposed standalone so a room can ask "how good is this policy,
    really?" about a policy that was learned WITHOUT a model (Room 2's MC).

    That question can't be answered from the learner's own numbers: MC's
    max_a Q(s,a) is the value of its epsilon-greedy self averaged over its whole
    history, which understates the greedy policy it actually plays. Evaluating
    the policy against the true model gives the honest answer. Benchmark only —
    never fed back to the learner.
    """
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
    """Expected number of steps from the start to the goal under `policy`.

    Solves the hitting-time equations t(s) = 1 + Σ_{s'} P(s'|s, π(s)) · t(s'),
    with t(goal) = 0, by iteration (same shape as policy evaluation). Finite for
    any policy that reaches the goal with probability 1 (e.g. the optimal one).
    """
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
    return t[grid.start]


def success_prob_within(grid, policy, max_steps: int):
    """Probability the start reaches the goal within `max_steps` moves under
    `policy`. Propagates the state distribution forward, absorbing goal mass."""
    dist = {grid.start: 1.0}
    reached = 0.0
    for _ in range(max_steps):
        new: dict = {}
        for s, m in dist.items():
            for s2, p in grid.probs[(s, policy[s])].items():
                new[s2] = new.get(s2, 0.0) + m * p
        reached += new.pop(grid.goal, 0.0)
        dist = new
        if not dist:
            break
    return reached


ALGORITHMS = {
    "Value Iteration": value_iteration,
    "Policy Iteration": policy_iteration,
}
