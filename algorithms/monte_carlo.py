from __future__ import annotations

import numpy as np

from core.icy_grid import ACTION_SPACE

CONSTANT = "Constant ε"
DECAYING = "Decaying ε"


def epsilon_at(k: int, kind: str, params: tuple) -> float:
    """Exploration rate for episode `k` (0-based). CONSTANT: params = (eps,);
    DECAYING: params = (eps_start, eps_min, decay)."""
    if kind == CONSTANT:
        return float(params[0])
    eps_start, eps_min, decay = params
    return float(max(eps_min, eps_start * (decay ** k)))


def _max_dict(d, rng):
    """argmax/max of a dict, breaking ties uniformly at random (as reference)."""
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]
    if len(max_keys) == 1:
        return max_keys[0], max_val
    return max_keys[int(rng.integers(len(max_keys)))], max_val


def _epsilon_greedy(policy, s, eps, rng):
    if rng.random() < (1 - eps):
        return policy[s]
    return ACTION_SPACE[int(rng.integers(len(ACTION_SPACE)))]


def _play_game(grid, policy, eps, rng, max_steps):
    """One episode under the epsilon-greedy policy. Returns (states, actions,
    rewards), with rewards[0] = 0 and each rewards[t+1] = R for states[t+1]."""
    s = grid.reset()
    a = _epsilon_greedy(policy, s, eps, rng)

    states, actions, rewards = [s], [a], [0.0]

    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()
        rewards.append(r)
        states.append(s)
        if grid.game_over():
            break
        a = _epsilon_greedy(policy, s, eps, rng)
        actions.append(a)

    return states, actions, rewards


def monte_carlo_control(
    grid,
    gamma: float = 0.9,
    n_episodes: int = 2000,
    max_steps: int = 300,
    eps_kind: str = DECAYING,
    eps_params: tuple = (1.0, 0.05, 0.998),
    seed: int = 0,
    n_checkpoints: int = 50,
):
    """Run on-policy first-visit MC control. Returns (Q, policy, history, stats):
    history is {episode, V, policy, eps} checkpoints; stats holds per-episode
    {returns, steps, success, eps} (returns is the discounted return)."""
    rng = np.random.default_rng(seed)

    # Random initial policy over navigable, non-terminal cells (as reference).
    policy = {s: ACTION_SPACE[int(rng.integers(len(ACTION_SPACE)))]
              for s in grid.actions}
    Q = {s: {a: 0.0 for a in ACTION_SPACE} for s in grid.actions}
    sample_counts = {s: {a: 0 for a in ACTION_SPACE} for s in grid.actions}

    returns_ = np.zeros(n_episodes)
    steps_ = np.zeros(n_episodes, dtype=int)
    success_ = np.zeros(n_episodes, dtype=bool)
    eps_ = np.zeros(n_episodes)

    if n_episodes <= 0:
        return Q, policy, [], {"returns": returns_, "steps": steps_,
                               "success": success_, "eps": eps_}

    checkpoints = set(
        np.linspace(0, n_episodes - 1, min(n_checkpoints, n_episodes), dtype=int).tolist())
    history = []

    for it in range(n_episodes):
        eps = epsilon_at(it, eps_kind, eps_params)
        states, actions, rewards = _play_game(grid, policy, eps, rng, max_steps)

        # First-occurrence index per (s, a) — the O(T) first-visit test.
        first_seen = {}
        for t, sa in enumerate(zip(states, actions)):
            if sa not in first_seen:
                first_seen[sa] = t

        T = len(states)
        G = 0.0
        for t in range(T - 2, -1, -1):
            s = states[t]
            a = actions[t]

            G = rewards[t + 1] + gamma * G

            if first_seen[(s, a)] == t:  # first-visit
                old_q = Q[s][a]
                sample_counts[s][a] += 1
                lr = 1 / sample_counts[s][a]
                Q[s][a] = old_q + lr * (G - old_q)
                policy[s] = _max_dict(Q[s], rng)[0]

        # After the backward pass G is the return from t=0, i.e. from the start.
        returns_[it] = G
        steps_[it] = T - 1
        # Compare the CELL, not the state — correct for Room 2 either way, but a
        # state is (i, j, k) in a shielded room and this would silently record
        # every escape as a failure there. See IcyGridWorld's STATE SHAPE note.
        success_[it] = grid.cell_of(states[-1]) == grid.goal
        eps_[it] = eps

        if it in checkpoints:
            history.append({
                "episode": it + 1,
                "V": {s: max(Qs.values()) for s, Qs in Q.items()},
                "policy": dict(policy),
                "eps": eps,
            })

    stats = {"returns": returns_, "steps": steps_, "success": success_, "eps": eps_}
    return Q, policy, history, stats


def moving_average(x, window: int):
    """Trailing moving average, same length as `x` (partial at the front)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


