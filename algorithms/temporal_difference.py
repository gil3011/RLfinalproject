from __future__ import annotations

import numpy as np

from algorithms.monte_carlo import CONSTANT, DECAYING, epsilon_at
from core.icy_grid import ACTION_SPACE

# Which bootstrap target the shared loop uses — the ONE line that separates the
# two algorithms (see `_td_control`).
SARSA = "sarsa"
QLEARNING = "qlearning"

__all__ = ["sarsa_control", "q_learning_control", "SARSA", "QLEARNING",
           "CONSTANT", "DECAYING"]


def _argmax_random(q: dict, rng) -> str:
    """argmax over an action->value dict, breaking ties uniformly at random."""
    best = max(q.values())
    ties = [a for a in ACTION_SPACE if q[a] == best]
    return ties[0] if len(ties) == 1 else ties[int(rng.integers(len(ties)))]


def _epsilon_greedy(Q, s, eps, rng) -> str:
    if rng.random() < eps:
        return ACTION_SPACE[int(rng.integers(len(ACTION_SPACE)))]
    return _argmax_random(Q[s], rng)


def _snapshot(Q, rng):
    """Greedy policy and V(s) = max_a Q(s,a) right now. `rng` must be separate
    from the training generator so snapshotting can't perturb the run."""
    V, policy = {}, {}
    for s, q in Q.items():
        policy[s] = _argmax_random(q, rng)
        V[s] = max(q.values())
    return V, policy


def _td_control(
    grid,
    kind: str,
    gamma: float = 0.95,
    alpha: float = 0.1,
    n_episodes: int = 2000,
    max_steps: int = 200,
    eps_kind: str = DECAYING,
    eps_params: tuple = (1.0, 0.05, 0.998),
    seed: int = 0,
    n_checkpoints: int = 50,
):
    """Shared TD-control loop; `kind` selects only the bootstrap target — SARSA
    uses Q[s2][a2] (action actually taken next), QLEARNING uses max_a Q[s2][a]
    (greedy). Both behave identically otherwise.

    Returns (Q, policy, history, stats): history is ~`n_checkpoints` snapshots
    {V, policy, eps, episode}; stats holds per-episode {returns, steps, success,
    falls, caught, eps}."""
    assert kind in (SARSA, QLEARNING), f"unknown TD kind: {kind!r}"
    rng = np.random.default_rng(seed)
    # Observing a run must not change it — see _snapshot.
    snap_rng = np.random.default_rng(seed + 10_000)
    Q = {s: {a: 0.0 for a in ACTION_SPACE} for s in grid.actions}

    returns = np.zeros(n_episodes)
    steps = np.zeros(n_episodes, dtype=int)
    success = np.zeros(n_episodes, dtype=bool)
    falls = np.zeros(n_episodes, dtype=bool)
    caught = np.zeros(n_episodes, dtype=bool)
    eps_log = np.zeros(n_episodes)

    # Evenly spaced checkpoints, always including the final episode.
    if n_episodes <= n_checkpoints:
        cp_at = set(range(n_episodes))
    else:
        cp_at = set(np.linspace(0, n_episodes - 1, n_checkpoints, dtype=int).tolist())
    history = []

    for k in range(n_episodes):
        eps = epsilon_at(k, eps_kind, eps_params)
        eps_log[k] = eps

        s = grid.reset()
        a = _epsilon_greedy(Q, s, eps, rng)
        G, discount, t = 0.0, 1.0, 0

        for t in range(1, max_steps + 1):
            r = grid.move(a)
            s2 = grid.current_state()
            G += discount * r
            discount *= gamma

            if grid.is_terminal(s2):
                # Terminal: the return is just r — there is no next action to
                # bootstrap from, and Q has no row for a terminal state.
                Q[s][a] += alpha * (r - Q[s][a])
                # Compare the CELL, not the state: in an augmented room a state
                # is (i, j, ...), so `s2 == grid.goal` is never true and every
                # escape would be recorded as a failure.
                success[k] = grid.cell_of(s2) == grid.goal
                falls[k] = grid.is_pit(s2)
                caught[k] = grid.is_caught(s2)
                break

            a2 = _epsilon_greedy(Q, s2, eps, rng)
            target = Q[s2][a2] if kind == SARSA else max(Q[s2].values())
            Q[s][a] += alpha * (r + gamma * target - Q[s][a])
            s, a = s2, a2

        returns[k] = G
        steps[k] = t

        if k in cp_at:
            V, policy = _snapshot(Q, snap_rng)
            history.append({"V": V, "policy": policy, "eps": eps, "episode": k + 1})

    _, final_policy = _snapshot(Q, snap_rng)
    stats = {"returns": returns, "steps": steps, "success": success,
             "falls": falls, "caught": caught, "eps": eps_log}
    return Q, final_policy, history, stats


def sarsa_control(grid, **kwargs):
    """On-policy TD control (SARSA). Bootstraps off the action actually taken next."""
    return _td_control(grid, SARSA, **kwargs)


def q_learning_control(grid, **kwargs):
    """Off-policy TD control (Q-learning). Identical to SARSA except the target
    is `max_a Q[s2][a]`, so it evaluates the greedy policy while behaving
    epsilon-greedily."""
    return _td_control(grid, QLEARNING, **kwargs)
