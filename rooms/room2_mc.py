"""
Room 2 — Monte Carlo control (on-policy first-visit, epsilon-greedy).

Task: cross long icy corridors and avoid portal traps that teleport you back to
the start.

Board cell types (see the legend under the board):
  * 🧱 blocked — walls the agent cannot step into; they carve the corridors,
  * 🟦 slippery ice — moves may slip perpendicular (placed by count, as in Room 1),
  * 🌀 portal trap — landing on one teleports the agent back to the start.

Ice is placed by count exactly as in Room 1, so the two rooms differ in what the
agent KNOWS, not in the physics — the comparison the rooms exist to make.

There are deliberately NO 🟥 negative-reward cells here, though Room 1 has them.
They were tried and removed: they break MC control outright. Bumping a wall returns
0 and leaves the agent in place, so once penalty cells push early sampled returns
negative, argmax Q picks "bump" — and that choice then manufactures its own
evidence, since every later episode is a full cap of bumping, confirming Q(bump)=0.
Measured at Room 2's defaults: 0% success and a pure loitering policy, even though
the random walk found the exit in 7% of episodes and V* was 14.4. Not fixable by a
step cost or by optimistic initial values (both still 0%). Room 1's DP is immune —
it backs up through the true model instead of sampled returns — which is why 🟥
cells live there and not here. Do not "restore" them without solving that.

Portals carry NO reward penalty: being sent back to the start simply delays the
exit, and the discount γ is what makes that expensive. That is why γ is a control
here rather than a constant — at γ = 1 a portal would cost nothing at all.

Unlike Room 1, this room has NO step cost, and a lost ▶️ Play episode is instead
scored a flat -100 by the shared `scored_return` so that giving up always ranks
last on the scoreboard. The split is deliberate: Room 1's agent is DP and never
sees a return at all — only the model moves it — so a penalty on the reported G
would be pure decoration there and a step cost is the only thing that can change
its behaviour. Here the penalty is exactly what it claims to be: a score.

Page flow follows docs/UI_STRUCTURE.md:
  Row 1 — About + setup board + 🎮 Environment controls.
  Row 2 — 🧠 Algorithm parameters + 🚀 Train.
  Row 3 — Training results: KPIs, an episode-checkpoint scrubber, the results
          board + ▶️ Play (the plan's "test mode", ε = 0), the learning curves,
          and a DP-benchmark row comparing V_MC against the exact V*.

The DP benchmark is displayed only — the learner never sees it. It exists because
Room 1 already computes the exact answer for this very board, which makes "MC
samples what DP computes" visible instead of theoretical.

On the two value numbers this room shows:
  * V_MC(s) = max_a Q(s,a) is MC's estimate OF ITSELF, and it understates the
    policy badly — measured at ~7.6 when the learned policy was really worth
    ~12.4 out of an optimal 14.8. Two effects compound: Q is the value of the
    epsilon-greedy agent (which keeps moving randomly), and the reference's 1/N
    step size makes Q a plain lifetime average that never forgets the early
    random-walk episodes.
  * So the KPI reports the policy's TRUE value via `policy_value()` — an exact
    model-side evaluation of the very policy ▶️ Play runs. V_MC still appears, in
    the benchmark row, next to the explanation of why the two disagree.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms.dynamic_programming import policy_value, value_iteration
from algorithms.monte_carlo import (CONSTANT, DECAYING, monte_carlo_control,
                                    moving_average)
from core.episode import LOSS_SCORE, rollout, scored_return
from core.icy_grid import IcyGridWorld, generate_layout, generate_portals

START = (9, 0)
GOAL = (0, 9)

_ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
_STEP_DELAY = {"Slow": 0.45, "Normal": 0.22, "Fast": 0.08}
_LEGEND = ("🤖 start · 🏁 goal · 🧱 wall · 🟦 slippery ice · "
           "🌀 portal trap (sends you back to the start)")
_MA_WINDOW = 50


def _make_grid(blocked, ice, portals, slip, goal_reward=100.0, seed=None):
    # seed=None → fresh entropy, so ▶️ Play Episode slips differently each run
    # (the point of watching a stochastic rollout). Training passes an explicit
    # seed for reproducible curves; the DP grids don't touch rng at all.
    return IcyGridWorld(
        start=START, goal=GOAL, blocked=blocked, ice=ice, slip=slip,
        goal_reward=goal_reward, teleports={p: START for p in portals},
        rng=np.random.default_rng(seed))


# ----------------------------------------------------------------------------- #
# Cached compute
# ----------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _train(blocked_t, ice_t, portals_t, slip, goal_reward, gamma, episodes,
           max_steps, eps_kind, eps_params, seed):
    grid = _make_grid(set(blocked_t), set(ice_t), set(portals_t), slip,
                      goal_reward, seed)
    _, _, history, stats = monte_carlo_control(
        grid, gamma=gamma, n_episodes=episodes, max_steps=max_steps,
        eps_kind=eps_kind, eps_params=eps_params, seed=seed)
    return history, stats


@st.cache_data(show_spinner=False)
def _dp_optimal(blocked_t, ice_t, portals_t, slip, goal_reward, gamma):
    """Exact V* for this board — the benchmark MC is measured against."""
    grid = _make_grid(set(blocked_t), set(ice_t), set(portals_t), slip, goal_reward)
    V, policy, _ = value_iteration(grid, gamma=gamma)
    return V, policy


@st.cache_data(show_spinner=False)
def _learned_policy_value(blocked_t, ice_t, portals_t, slip, goal_reward, gamma,
                          policy_t):
    """Exact value of a LEARNED policy — how good it actually is.

    MC's own max_a Q understates the greedy policy it plays (see the benchmark
    row), so the only honest answer comes from evaluating the policy against the
    true model. Displayed only; the learner never sees it.
    """
    grid = _make_grid(set(blocked_t), set(ice_t), set(portals_t), slip, goal_reward)
    return policy_value(grid, dict(policy_t), gamma)


def _regenerate_layout(env, seed, version):
    blocked, ice, _ = generate_layout(env["n_blocked"], env["n_slippery"], 0, seed)
    # Portals come from an independent pool, so keep them off the ice — a cell
    # drawn as two hazards at once just looks broken.
    portals = generate_portals(blocked, env["n_portals"], seed, exclude=ice)
    st.session_state["room2_layout"] = {
        "blocked": blocked, "ice": ice, "portals": portals, "version": version,
        "counts": (env["n_blocked"], env["n_slippery"], env["n_portals"]),
    }


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #
def _cell_shapes(blocked, ice, portals):
    shapes = []
    for (i, j) in ice:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(14,165,233,0.55)", "width": 1.2},
            fillcolor="rgba(56,189,248,0.16)", layer="above"))
    for (i, j) in portals:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(168,85,247,0.9)", "width": 2},
            fillcolor="rgba(168,85,247,0.22)", layer="above"))
    for (i, j) in blocked:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "#111827", "width": 1},
            fillcolor="rgba(55,65,81,0.85)", layer="above"))
    return shapes


def _base_grid(grid, V, policy, show_arrows):
    z = np.zeros((grid.rows, grid.cols))
    text = np.empty((grid.rows, grid.cols), dtype=object)
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if grid.is_blocked(s):
                z[i, j] = np.nan
                text[i, j] = "🧱"
            elif grid.is_teleport(s):
                # Transient — never occupied, so it has no value to show.
                z[i, j] = np.nan
                text[i, j] = "🌀"
            elif s == GOAL:
                # Masked, like the walls. The exit is TERMINAL — you never act
                # from it — so V(exit) is 0, not the +100 it pays on entry.
                # Painting the reward here put a number on a scale labelled V(s)
                # that was not a V at all: every other cell showed the expected
                # return from standing there, and this one showed a reward.
                z[i, j] = np.nan
                text[i, j] = "🏁"
            elif s == START:
                z[i, j] = V.get(s, 0.0)
                text[i, j] = "🤖"
            else:
                z[i, j] = V.get(s, 0.0)
                text[i, j] = _ARROW[policy[s]] if (
                    show_arrows and policy and s in policy) else ""
    return z, text


def _figure(grid, V, policy, show_arrows, trail=None, agent=None,
            portal_flash=False, height=520):
    z, text = _base_grid(grid, V, policy, show_arrows)
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}", textfont={"size": 16},
        colorscale="RdBu", zmid=0, colorbar={"title": "V(s)"},
        hovertemplate="cell (%{y}, %{x})<br>value %{z:.1f}<extra></extra>"))
    if trail:
        fig.add_trace(go.Scatter(
            x=[c for _, c in trail], y=[r for r, _ in trail], mode="lines",
            line={"color": "rgba(17,24,39,0.75)", "width": 3},
            hoverinfo="skip", showlegend=False))
    if agent is not None:
        fig.add_trace(go.Scatter(
            x=[agent[1]], y=[agent[0]], mode="markers",
            marker={"size": 30 if portal_flash else 22,
                    "color": "#a855f7" if portal_flash else "#f59e0b",
                    "line": {"color": "#111827", "width": 2}},
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        shapes=_cell_shapes(grid.blocked, grid.ice, set(grid.teleports.keys())),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=height)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig


def _returns_curve(returns, v_star_start, view_ep):
    n = len(returns)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=returns, mode="markers",
        marker={"size": 3, "color": "rgba(59,130,246,0.35)"}, name="episode G"))
    fig.add_trace(go.Scatter(
        x=x, y=moving_average(returns, _MA_WINDOW), mode="lines",
        line={"color": "#1d4ed8", "width": 2}, name=f"{_MA_WINDOW}-episode average"))
    fig.add_hline(y=v_star_start, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"DP optimal V*(S) = {v_star_start:.1f}")
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"viewing ep {view_ep}")
    fig.update_yaxes(title="discounted return G")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Episode return — MC settles below V* by the ε-gap",
                      legend={"orientation": "h", "y": -0.2})
    return fig


def _steps_curve(steps, success, view_ep):
    n = len(steps)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=steps, mode="markers",
        marker={"size": 3,
                "color": np.where(success, "rgba(16,185,129,0.35)",
                                  "rgba(239,68,68,0.30)")},
        name="steps (green = escaped)"))
    fig.add_trace(go.Scatter(
        x=x, y=moving_average(steps, _MA_WINDOW), mode="lines",
        line={"color": "#047857", "width": 2}, name=f"{_MA_WINDOW}-episode average"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b")
    fig.update_yaxes(title="steps to exit")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Steps per episode — random wandering → efficient routing",
                      legend={"orientation": "h", "y": -0.2})
    return fig


def _epsilon_curve(eps, view_ep):
    """Exploration rate per episode — the ε that was actually in force."""
    fig = go.Figure(go.Scatter(
        x=np.arange(1, len(eps) + 1), y=eps, mode="lines",
        line={"color": "#7c3aed", "width": 2}, name="ε"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"viewing ep {view_ep}")
    fig.update_yaxes(title="ε (chance of a random move)", rangemode="tozero")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Exploration rate — how often the agent ignored its policy")
    return fig


# ----------------------------------------------------------------------------- #
# Controls
# ----------------------------------------------------------------------------- #
def _env_controls():
    st.markdown("##### 🎮 Environment & Physics")
    n_blocked = st.slider("Blocked cells 🧱", 0, 30, 20,
        help="Walls the agent cannot step into — these carve the icy corridors. "
        "Placement always keeps a path from the start to the exit.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 20,
        help="Icy cells (shaded blue) where a move may slip perpendicular. Same "
        "hazard as Room 1 — but here the agent has to discover it by slipping, "
        "rather than reading it off the model.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.2, 0.05,
        help="On an ice cell, the chance a move sends you perpendicular instead "
        "of straight ahead. No effect on solid ground.")
    n_portals = st.slider("Portal traps 🌀", 0, 5, 3,
        help="Landing on a portal teleports you straight back to the start. There "
        "is no reward penalty — the punishment is the time you lose, which only "
        "costs you when γ < 1. Portals are never placed where they would strand "
        "the exit.")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit — the only reward on the board. Monte "
        "Carlo only ever sees it by actually getting there, so everything it "
        "learns is scaled by this number.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Apply the current counts and reshuffle the walls, ice, and portals. "
        "The board only changes when you click this.")
    return {"slip": slip, "n_blocked": n_blocked, "n_slippery": n_slippery,
            "n_portals": n_portals, "goal_reward": goal_reward, "regen": regen}


def _algo_row():
    st.markdown("##### 🧠 Algorithm")
    c1, c2, c3 = st.columns(3)
    gamma = c1.slider("Discount factor γ", 0.50, 0.99, 0.90, 0.01,
        help="How much future reward is worth vs. immediate. This is what makes "
        "portals hurt: the only reward is +100 at the exit, so a trap costs you "
        "only through the extra discounting of a longer route. At γ = 1 a portal "
        "would be free.")
    episodes = c2.select_slider("Training episodes",
        [100, 250, 500, 1000, 2000, 3000, 5000], value=2000,
        help="How many complete episodes MC samples. Monte Carlo learns only from "
        "FINISHED episodes, so too few and the agent never stumbles onto the exit "
        "at all — the values stay flat at zero.")
    max_steps = c3.select_slider("Max steps per training episode",
        [50, 100, 200, 300, 400, 500], value=300,
        help="Cap on each TRAINING episode. Early on the policy is a random walk, "
        "so this must be generous enough to reach the exit by luck — otherwise no "
        "episode ever returns a reward and nothing is learned.")

    e1, e2 = st.columns([1, 3])
    eps_kind = e1.selectbox("Exploration", [DECAYING, CONSTANT],
        help="ε is the chance of ignoring the current best action and trying a "
        "random one. Constant ε keeps exploring forever (and keeps paying for it); "
        "decaying ε explores early, then commits.")
    with e2:
        if eps_kind == CONSTANT:
            eps = st.slider("ε", 0.01, 0.5, 0.30, 0.01,
                help="Fixed exploration rate — the chance of ignoring the policy "
                "and moving at random. Too LOW is the real danger: below ~0.2 the "
                "agent mostly follows its own arbitrary starting policy, almost "
                "never stumbles on the exit, and so learns nothing at all (every "
                "return stays 0 and Q stays flat). Around 0.3 it reliably finds "
                "the exit and matches decaying ε.")
            eps_params = (eps,)
        else:
            d1, d2, d3 = st.columns(3)
            eps_start = d1.slider("ε start", 0.1, 1.0, 1.0, 0.05,
                help="Exploration rate at episode 1. Start at 1.0 for a pure "
                "random walk — with an all-zero Q there is nothing to exploit yet.")
            eps_min = d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01,
                help="Floor ε never drops below, so the agent keeps a little "
                "exploration forever. Set it to 0 to let the policy fully commit.")
            decay = d3.slider("ε decay rate", 0.990, 0.9999, 0.998, 0.0001,
                format="%.4f",
                help="Per-episode multiplier: ε(k) = max(ε min, ε start · rate^k). "
                "Lower = faster commitment. At 0.998, ε reaches 0.05 near episode "
                "1,500 — match this to your episode count.")
            eps_params = (eps_start, eps_min, decay)

    train = st.button("🚀 Train", type="primary", use_container_width=True,
        help="Run Monte Carlo control on the current board.")
    return gamma, episodes, max_steps, eps_kind, eps_params, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 2 · Monte Carlo")
    st.caption(
        "Cross the icy corridors to the exit — and avoid the portal traps that "
        "throw you back to the start.")
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "Room 1 could *compute* the answer because it knew the whole board. "
            "Monte Carlo knows **nothing**: it plays complete episodes and averages "
            "what actually happened. Early on it is a random walk that learns "
            "nothing at all — until it stumbles onto the exit for the first time "
            "and the reward finally has something to flow back through.\n\n"
            "MC also never learns V directly — it averages returns into Q(s,a) and "
            "reads off V(s) = maxₐ Q(s,a). That estimate is famously pessimistic "
            "about itself, which is why the KPI reports the policy's **true** value "
            "and the benchmark row explains the difference.\n\n"
            "**How to use it:** shape the board, set γ and the exploration schedule, "
            "then **🚀 Train**. Scrub the episode history to watch the value function "
            "grow out from the exit, and compare it against the exact DP answer at "
            "the bottom — the same board Room 1 solves in one shot.")

    # --- Row 1: setup board + environment controls -------------------------- #
    board_col, env_col = st.columns([3, 2])
    with board_col:
        setup_board = st.empty()
        setup_caption = st.empty()
        st.caption(_LEGEND)
    with env_col:
        env = _env_controls()

    if st.session_state.get("room2_layout") is None:
        _regenerate_layout(env, seed=0, version=0)
    if env["regen"]:
        v = st.session_state["room2_layout"]["version"] + 1
        _regenerate_layout(env, seed=v, version=v)
        st.session_state.pop("room2_trained_sig", None)

    layout = st.session_state["room2_layout"]
    blocked, ice, portals = layout["blocked"], layout["ice"], layout["portals"]
    grid = _make_grid(blocked, ice, portals, env["slip"], env["goal_reward"])

    zeros = {s: 0.0 for s in grid.all_states()}
    setup_board.plotly_chart(_figure(grid, zeros, {}, show_arrows=False),
                             use_container_width=True, key="room2_setup_board")
    counts_now = (env["n_blocked"], env["n_slippery"], env["n_portals"])
    if counts_now != layout["counts"]:
        setup_caption.caption("⚠️ Counts changed — click 🎲 Regenerate to apply.")
    elif len(portals) < env["n_portals"]:
        setup_caption.caption(
            f"Placed {len(portals)} of {env['n_portals']} portals — the rest would "
            "have sealed the exit off.")
    else:
        setup_caption.caption("Board layout — set the algorithm below and 🚀 Train.")

    # --- Row 2: algorithm parameters ---------------------------------------- #
    st.divider()
    gamma, episodes, max_steps, eps_kind, eps_params, train = _algo_row()

    sig = (layout["version"], env["slip"], env["goal_reward"], gamma, episodes,
           max_steps, eps_kind, eps_params)
    if train:
        st.session_state["room2_trained_sig"] = sig
    if st.session_state.get("room2_trained_sig") != sig:
        return  # not trained for this configuration — no results yet

    # --- Row 3: training results -------------------------------------------- #
    keys = (tuple(sorted(blocked)), tuple(sorted(ice)), tuple(sorted(portals)))
    with st.spinner(f"Sampling {episodes:,} episodes…"):
        history, stats = _train(*keys, env["slip"], env["goal_reward"], gamma,
                                episodes, max_steps, eps_kind, eps_params, 0)
        V_star, pi_star = _dp_optimal(*keys, env["slip"], env["goal_reward"], gamma)

    if not history:
        st.warning("No episodes were run.")
        return

    returns, steps = stats["returns"], stats["steps"]
    success = stats["success"]
    n_cp = len(history)

    st.divider()
    st.markdown("#### Training results")

    last = slice(-100, None)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Success rate (last 100)", f"{success[last].mean():.0%}",
              help="Share of the final 100 training episodes that reached the exit "
              "within the step cap. This is measured while still exploring, so it "
              "sits below what ▶️ Play (ε = 0) achieves.")
    m2.metric("Mean return (last 100)", f"{returns[last].mean():+.1f}",
              help="Average discounted return G over the final 100 episodes — the "
              "same quantity as V, so it is directly comparable to V*(S). Measured "
              "while still exploring, so it sits below the policy's true value.")

    # View controls above the board.
    key = "room2_view_cp"
    if key in st.session_state and st.session_state[key] > n_cp:
        st.session_state[key] = n_cp
    v_col, a_col = st.columns([3, 1])
    with v_col:
        cp_i = st.slider("View checkpoint", 1, n_cp, n_cp, key=key,
            help="Replay the value function and greedy policy as they stood at "
            f"each of {n_cp} checkpoints across training — watch value spread "
            "backward from the exit as episodes accumulate.") if n_cp > 1 else 1
    with a_col:
        show_arrows = st.checkbox("Show policy arrows", value=True,
            help="Overlay the greedy action in each cell for the viewed checkpoint.")

    snap = history[cp_i - 1]
    V, policy, view_ep = snap["V"], snap["policy"], snap["episode"]

    # The exact value of the viewed policy — still needed by the benchmark row,
    # where MC's own (much lower) estimate of itself is put beside it.
    V_greedy = _learned_policy_value(*keys, env["slip"], env["goal_reward"], gamma,
                                     tuple(sorted(policy.items())))
    v_gre_start, v_star_start = V_greedy[START], V_star[START]
    v_mc_start = V.get(START, 0.0)
    m3.metric("Start-state value V(S)", f"{v_mc_start:.1f}",
              help="What MC estimates the start is worth at this checkpoint: "
              "max_a Q(S,a). See the benchmark row for how it compares to the exact "
              "answer, and for what the policy is really worth.")

    m4.metric("ε at checkpoint", f"{snap['eps']:.3f}",
              help="The exploration rate in force at the viewed checkpoint — the "
              "chance the agent ignored its policy and moved at random.")

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play** — greedy, ε = 0")
        play_max_steps = st.slider("Max steps per episode", 10, 500, 200,
            help="Cap for THIS playback only — separate from the training cap "
            "above. On ice the agent can wander; past this it times out.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"],
            "Normal", help="Playback speed of the animated episode.")
        play = st.button("▶️ Play Episode", type="primary",
            use_container_width=True,
            help="The plan's 'test mode': runs the viewed policy with exploration "
            "switched off (ε = 0) across the real, stochastic ice and reports its "
            "discounted return G.")
        episode_slot = st.container()

    results_caption.caption(
        f"Value & greedy policy after **{view_ep:,}** episodes "
        f"(checkpoint {cp_i} of {n_cp})")

    # An episode is EPHEMERAL: it lives only in the run that played it. Nothing
    # goes to session state — a stored rollout outlives the policy it was run
    # against, so scrubbing to another checkpoint would redraw a stale trail over
    # a policy that never produced it.
    if play:
        path, G_ep, outcome, landings = rollout(
            grid, policy, gamma=gamma, max_steps=play_max_steps, with_landings=True)
        # Draw the portal touch as its own frame, else the jump back to the
        # start looks like a rendering glitch rather than a trap firing.
        frames = []
        for k in range(len(path)):
            if landings[k] != path[k]:
                frames.append((list(path[:k]) + [landings[k]], landings[k], True))
            frames.append((list(path[: k + 1]), path[k], False))
        for k, (trail, agent, flash) in enumerate(frames):
            results_board.plotly_chart(
                _figure(grid, V, policy, show_arrows, trail=trail, agent=agent,
                        portal_flash=flash),
                use_container_width=True, key=f"room2_ep_{k}")
            time.sleep(_STEP_DELAY[speed])

        portals_hit = sum(1 for k in range(len(path)) if landings[k] != path[k])
        score = scored_return(G_ep, outcome)
        with episode_slot:
            if outcome == "goal":
                st.success("🏁 Escaped! The agent reached the exit.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return G", f"{score:+.1f}",
                help="On a WIN this is the real discounted return G = Σ γ^t·r₍t+1₎. "
                f"A run that fails to escape scores a flat {LOSS_SCORE:+.0f}, so "
                "giving up always ranks below escaping. One sample of a stochastic "
                "rollout: play again and it will differ. (The training curves below "
                "show the RAW returns MC actually learned from — no penalty.)")
            e2.metric("Steps", len(path) - 1,
                help="Number of moves before the episode ended.")
            e3.metric("Result", "✅" if outcome == "goal" else "❌",
                help="Whether the agent reached the exit within the step cap.")
            if outcome != "goal":
                st.caption(
                    f"Shown as a flat {LOSS_SCORE:+.0f} for not escaping; the raw "
                    f"discounted return was {G_ep:+.1f}.")
            if portals_hit:
                st.caption(f"🌀 Sent back to the start {portals_hit}× this run.")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows),
            use_container_width=True, key="room2_results_board")

    # --- Learning curves ---------------------------------------------------- #
    st.plotly_chart(_returns_curve(returns, V_star[START], view_ep),
                    use_container_width=True)
    st.plotly_chart(_steps_curve(steps, success, view_ep), use_container_width=True)
    st.plotly_chart(_epsilon_curve(stats["eps"], view_ep), use_container_width=True)
    st.caption(
        "ε is the chance the agent ignored its policy and moved at random. Read it "
        "against the two curves above: the return climbs as ε falls, because the "
        "agent stops paying for exploration it no longer needs — and the gap that "
        "remains to V* is largely the exploration it is still doing.")

    # --- DP benchmark row --------------------------------------------------- #
    st.divider()
    st.markdown("#### 📐 Benchmark against the exact answer")
    st.caption(
        "Room 1's Dynamic Programming solves this exact board from the model. "
        "Monte Carlo never sees any of it — it only samples episodes. This is the "
        "same value function, computed vs. learned — and with **Show policy arrows** "
        "on, the same comparison for the policy: every cell where the arrows differ "
        "is one where sampling has not yet found what the model already knows.")

    n1, n2, n3 = st.columns(3)
    n1.metric("V_MC(S) — what MC believes", f"{v_mc_start:.1f}",
              help="max_a Q(S,a), MC's own estimate — the number it learned. Read "
              "the caption below before trusting it as a measure of the policy.")
    n2.metric("True V(S) of that policy", f"{v_gre_start:.1f}",
              help="What the very same policy is actually worth, evaluated exactly "
              "against the model. Higher than MC's own estimate.")
    n3.metric("V*(S) — exact optimal", f"{v_star_start:.1f}",
              help="The best any policy can do on this board — Room 1's answer.")
    st.caption(
        f"**Why the first two disagree.** MC never learns V directly: it averages "
        f"returns into Q(s,a), and V_MC(s) is just max_a Q(s,a). That number "
        f"understates the policy for two compounding reasons. **(1)** Q is the value "
        f"of the *ε-greedy* agent, which keeps making random moves — not of the "
        f"greedy policy ▶️ Play runs. **(2)** Q is a plain average over *every* "
        f"episode ever run (step size 1/N), so the early random-walk returns are "
        f"weighted exactly as heavily as recent ones and are never forgotten. "
        f"So MC's self-estimate reads {v_mc_start:.1f} while the policy it found is "
        f"genuinely worth {v_gre_start:.1f} of a possible {v_star_start:.1f}.")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**V_MC — sampled ({view_ep:,} episodes)**")
        st.plotly_chart(
            _figure(grid, V, policy, show_arrows, height=420),
            use_container_width=True, key="room2_bench_mc")
    with b2:
        st.markdown("**V\\* — computed exactly (DP)**")
        st.plotly_chart(
            _figure(grid, V_star, pi_star, show_arrows, height=420),
            use_container_width=True, key="room2_bench_dp")
