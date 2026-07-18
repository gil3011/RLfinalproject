"""
Temporal-Difference control for Room 3 — SARSA (on-policy).

Adapted from `code examples/sarsa.py`, keeping its update exactly:

    Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])

The whole room lives in that `Q[s2][a2]`. SARSA bootstraps off the action it
will ACTUALLY take next — one drawn from the same epsilon-greedy policy it is
following — so the risk of exploring near a cliff is priced into the value of
standing near the cliff. Q-learning (Room 4) instead bootstraps off
`max_a Q[s2][a]`, the action it *would* take if it never explored, which is why
it will happily learn to walk the ledge and then fall off it while exploring.

Deviations from the reference, all deliberate:

* **No bootstrap off a terminal.** The reference's `while not grid.game_over()`
  loop never forms a target from a terminal state. Room 3 has terminal pits as
  well as a terminal goal, so both end the episode with `Q[s][a] += alpha*(r -
  Q[s][a])`. Bootstrapping through a terminal's all-zero Q row would silently
  damp the -100 fall toward zero and teach the agent that the abyss is cheap.
* **Checkpointed history.** Snapshotting Q every episode is far too large for
  `st.session_state`. Record ~50 evenly-spaced checkpoints of
  {V, policy, eps, episode}; the room's scrubber ranges over checkpoints. Same
  approach as `monte_carlo.py`.
* **Seeded RNG.** The reference calls `np.random.*` globally. Thread an explicit
  `default_rng(seed)` so training is reproducible across `st.cache_data`
  evictions. The display/Play grid stays unseeded so each Play slips differently.
* **Random tie-breaking** among equal-Q actions (the reference's `max_dict`
  behaviour). With a cold, all-zero Q this is what makes early SARSA a random
  walk rather than a march in whichever direction sorts first.
* **Episode stats** (returns/steps/success/falls/eps) for the room's KPIs and
  curves. The return G is DISCOUNTED, matching how V is defined, so the mean
  return of a converged policy is comparable to V(S).

The epsilon schedule is shared with Monte Carlo — same knob, same meaning, so it
is imported rather than forked.
"""
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
    """Greedy policy and V(s) = max_a Q(s,a) as they stand right now.

    `rng` MUST be a separate generator from the training one. Tie-breaking draws
    from it, so snapshotting off the training stream would let the checkpoint
    schedule perturb the very run it is supposed to be observing — and since the
    schedule is derived from n_episodes, a 2,000- and a 5,000-episode run would
    silently diverge inside their common first 2,000 episodes.
    """
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
    """Shared TD-control loop. `kind` selects the bootstrap target and NOTHING else.

      SARSA     : target = Q[s2][a2] — the action actually taken next, so the
                  risk of the agent's own future exploration is priced in.
      QLEARNING : target = max_a Q[s2][a] — the greedy action, as if it would
                  never explore again, so the ledge looks safe from a distance.

    Both draw the NEXT action `a2` epsilon-greedily and take it (the behaviour
    policy is identical); only what they bootstrap from differs. That is the
    whole SARSA-vs-Q-learning lesson, and keeping it to one branch in one loop is
    what guarantees the two are otherwise a controlled comparison — Room 3 was
    burned by a harness that silently ran SARSA twice.

    Returns (Q, policy, history, stats):
      Q       : {state: {action: value}}
      policy  : greedy policy at the END of training
      history : ~`n_checkpoints` snapshots {V, policy, eps, episode}
      stats   : per-episode arrays {returns, steps, success, falls, caught, eps}.
                `caught` is always all-False on a grid with no guard (Room 3);
                Room 4 reads it as its headline death count.
    """
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
    """Off-policy TD control (Q-learning). Bootstraps off the greedy action.

    Adapted from `code examples/q_learning.py`. Identical to SARSA except the
    target is `max_a Q[s2][a]` instead of `Q[s2][a2]` — so it evaluates the
    optimal policy while behaving epsilon-greedily. On Room 4's board it learns
    to walk the ledge for the coin; the DP benchmark shows, per board, whether
    that was brave or merely over-optimistic.
    """
    return _td_control(grid, QLEARNING, **kwargs)
