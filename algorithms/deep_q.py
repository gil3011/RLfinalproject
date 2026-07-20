"""
Deep Q-Learning for Room 5 — adapted from `code examples/dql/dqn.py`.

Keeps the reference's architecture and update: a small MLP `QNetwork`, a
`ReplayBuffer`, a target network, Huber (SmoothL1) loss, `gather` on the taken
action, gradient clipping at 10, Adam. Deviations, all deliberate and measured on
`core/chase_arena.py`:

* **Double DQN target.** A single-network DQN on this env diverged: traced greedy
  policies loitered at the start with `max Q ≈ 6` in reward-units where the best
  achievable return is ~2 — the value function overestimated ~3× and collapsed to
  near-constant across states, so the policy went state-blind. Double DQN (select
  the next action with the online net, evaluate it with the target net) is the
  standard cure for exactly that overestimation, so Room 5 uses it. It is a
  two-line change to the reference's target and nothing else.
* **Reward scaling to the network.** The env pays ±100 (goal / catch) for the
  scoreboard; training on ±100 made the targets large and the net unstable.
  `reward_scale` (default 0.01) scales rewards *for the learner only* — the raw
  ±100 is what the KPIs and the returns curve report.
* **Episode-level epsilon shared with `monte_carlo.epsilon_at`** (CONSTANT /
  DECAYING), so Room 5 means the same thing by ε as Rooms 2–4 rather than the
  reference's step-count exponential decay.
* **Checkpointed weights** (~`n_checkpoints` `state_dict` snapshots) so the room's
  scrubber can replay the greedy policy — and its value field — as they improved.
* **Per-episode stats** (return / steps / outcome / eps) for the KPIs and curves,
  plus per-gradient-step loss and mean predicted Q for the training-diagnostics plot.

Truncation (hitting the step cap) is NOT treated as terminal for bootstrapping —
only an actual catch or escape ends the value backup, exactly as a timeout should.
"""
from __future__ import annotations

import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.monte_carlo import CONSTANT, DECAYING, epsilon_at

# Small net — extra threads cost more than they save here and starve the UI thread.
torch.set_num_threads(2)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

__all__ = ["dqn_control", "q_field", "greedy_rollout", "load_net",
           "QNetwork", "CONSTANT", "DECAYING"]


class ReplayBuffer:
    """Faithful to `code examples/dql/dqn.py`."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Two hidden layers, ReLU — the reference architecture."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _greedy_action(net, obs) -> int:
    with torch.no_grad():
        t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(net(t).argmax(1).item())


def dqn_control(
    make_env,
    *,
    obs_dim: int = 4,
    n_actions: int = 9,
    n_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 3e-4,
    batch_size: int = 64,
    buffer_size: int = 50_000,
    target_update: int = 1_000,
    train_freq: int = 4,
    hidden: int = 128,
    eps_kind: str = DECAYING,
    eps_params: tuple = (1.0, 0.05, 0.99),
    reward_scale: float = 0.01,
    double: bool = True,
    seed: int = 0,
    n_checkpoints: int = 40,
    progress_cb=None,
):
    """Train a DQN on the gymnasium env returned by `make_env()`.

    Returns a dict bundle (all plain numpy / python / state_dicts, safe for
    `st.session_state`):
      net_state   : final policy-net `state_dict`
      hidden      : hidden width (to rebuild the net)
      checkpoints : list of {episode, net_state} snapshots for the scrubber
      stats       : per-episode {returns, steps, outcome, escaped, caught,
                    timeout, eps} + per-grad-step {loss, q_pred}
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    policy_net = QNetwork(obs_dim, n_actions, hidden)
    target_net = QNetwork(obs_dim, n_actions, hidden)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    returns = np.zeros(n_episodes)
    steps = np.zeros(n_episodes, dtype=int)
    escaped = np.zeros(n_episodes, dtype=bool)
    caught = np.zeros(n_episodes, dtype=bool)
    timeout = np.zeros(n_episodes, dtype=bool)
    eps_log = np.zeros(n_episodes)
    outcomes = []
    loss_log, q_log = [], []

    if n_episodes <= n_checkpoints:
        cp_at = set(range(n_episodes))
    else:
        cp_at = set(np.linspace(0, n_episodes - 1, n_checkpoints, dtype=int).tolist())
    checkpoints = []

    step = 0
    for k in range(n_episodes):
        eps = epsilon_at(k, eps_kind, eps_params)
        eps_log[k] = eps
        obs, _ = env.reset(seed=seed * 1_000_003 + k)
        G, t = 0.0, 0
        outcome = "timeout"
        while True:
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                a = _greedy_action(policy_net, obs)
            nobs, r, term, trunc, info = env.step(a)
            G += r                                  # raw undiscounted return (scoreboard scale)
            t += 1
            # done for bootstrap = a real terminal (catch/escape), NOT truncation.
            buffer.push(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor([a]),
                torch.tensor([r * reward_scale], dtype=torch.float32),
                torch.tensor(nobs, dtype=torch.float32),
                torch.tensor([term], dtype=torch.bool),
            )
            obs = nobs
            step += 1

            if step % train_freq == 0 and len(buffer) >= batch_size:
                loss_v, q_v = _optimize(policy_net, target_net, optimizer, buffer,
                                        batch_size, gamma, double)
                loss_log.append(loss_v)
                q_log.append(q_v)
            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if term or trunc:
                outcome = info.get("outcome") or ("timeout" if trunc else outcome)
                break

        returns[k] = G
        steps[k] = t
        escaped[k] = outcome == "escaped"
        caught[k] = outcome == "caught"
        timeout[k] = outcome == "timeout"
        outcomes.append(outcome)

        if k in cp_at:
            checkpoints.append({"episode": k + 1,
                                "net_state": copy.deepcopy(policy_net.state_dict())})
        if progress_cb is not None:
            progress_cb(k + 1, n_episodes)

    env.close()
    stats = {"returns": returns, "steps": steps, "outcome": outcomes,
             "escaped": escaped, "caught": caught, "timeout": timeout,
             "eps": eps_log, "loss": np.array(loss_log),
             "q_pred": np.array(q_log)}
    return {"net_state": copy.deepcopy(policy_net.state_dict()),
            "hidden": hidden, "checkpoints": checkpoints, "stats": stats}


def _optimize(policy_net, target_net, optimizer, buffer, batch_size, gamma, double):
    transitions = buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    states = torch.stack(batch.state)
    actions = torch.stack(batch.action)
    rewards = torch.stack(batch.reward)
    next_states = torch.stack(batch.next_state)
    dones = torch.stack(batch.done)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        if double:
            # select with the online net, evaluate with the target net
            next_a = policy_net(next_states).argmax(1, keepdim=True)
            next_q = target_net(next_states).gather(1, next_a)
        else:
            next_q = target_net(next_states).max(1, keepdim=True).values
        next_q[dones] = 0.0
        targets = rewards + gamma * next_q

    loss = nn.SmoothL1Loss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item()), float(q_values.mean().item())


def load_net(net_state, hidden, obs_dim=4, n_actions=9):
    net = QNetwork(obs_dim, n_actions, hidden)
    net.load_state_dict(net_state)
    net.eval()
    return net


def q_field(net, enemy_xy, arena=10.0, res=50):
    """max_a Q(x, y, ·) sampled on a `res×res` grid, holding the enemy fixed at
    `enemy_xy`. Returns (xs, ys, Z) with Z[j, i] the value at (xs[i], ys[j]) — a
    2-D SLICE of the 4-D value function (the field really depends on the enemy)."""
    xs = np.linspace(0.0, arena, res)
    ys = np.linspace(0.0, arena, res)
    ex, ey = float(enemy_xy[0]), float(enemy_xy[1])
    obs = np.zeros((res * res, 4), dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    flatx, flaty = gx.ravel(), gy.ravel()
    obs[:, 0] = flatx / arena
    obs[:, 1] = flaty / arena
    obs[:, 2] = (ex - flatx) / arena
    obs[:, 3] = (ey - flaty) / arena
    with torch.no_grad():
        q = net(torch.tensor(obs)).max(1).values.numpy()
    return xs, ys, q.reshape(res, res)


def greedy_rollout(net, env, seed=None, options=None):
    """One greedy (ε=0) episode. Returns frames of agent/enemy positions, the
    outcome, the raw undiscounted return, and the step count. Ephemeral — the
    caller renders it and lets it go (never store in session state)."""
    obs, info = env.reset(seed=seed, options=options)
    frames = [{"agent": info["agent"].copy(), "enemy": info["enemy"].copy()}]
    G, t, outcome = 0.0, 0, "timeout"
    while True:
        a = _greedy_action(net, obs)
        obs, r, term, trunc, info = env.step(a)
        G += r
        t += 1
        frames.append({"agent": info["agent"].copy(), "enemy": info["enemy"].copy()})
        if term or trunc:
            outcome = info.get("outcome") or ("timeout" if trunc else "caught")
            break
    return {"frames": frames, "outcome": outcome, "return": G, "steps": t}
