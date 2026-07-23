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
_LEGEND = ("🤖 Start · 🏁 Goal · 🧱 Wall · 🟦 Ice (slippery) · "
           "🌀 Portal (teleport to start)")
_MA_WINDOW = 50


def _make_grid(blocked, ice, portals, slip, goal_reward=100.0, seed=None):
    # seed=None → fresh entropy, so ▶️ Play Episode slips differently each run.
    # Training passes an explicit seed for reproducible curves.
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
    """Exact value of a LEARNED policy — how good it actually is."""
    grid = _make_grid(set(blocked_t), set(ice_t), set(portals_t), slip, goal_reward)
    return policy_value(grid, dict(policy_t), gamma)


def _regenerate_layout(env, seed, version):
    blocked, ice, _ = generate_layout(env["n_blocked"], env["n_slippery"], 0, seed)
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
                z[i, j] = np.nan
                text[i, j] = "🌀"
            elif s == GOAL:
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
        help="Impassable walls that form corridors. A valid path to the exit is always preserved.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 20,
        help="Ice cells where movement may slide sideways.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.2, 0.05,
        help="Chance of sliding perpendicular to the intended direction on ice.")
    n_portals = st.slider("Portal traps 🌀", 0, 5, 3,
        help="Traps that teleport the agent back to the start. No point penalty, but costs time (discounted by γ).")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit. All learned values scale relative to this number.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Generate a new layout with the selected cell counts.")
    return {"slip": slip, "n_blocked": n_blocked, "n_slippery": n_slippery,
            "n_portals": n_portals, "goal_reward": goal_reward, "regen": regen}


def _algo_row():
    st.markdown("##### 🧠 Algorithm")
    c1, c2, c3 = st.columns(3)
    gamma = c1.slider("Discount factor γ", 0.50, 0.99, 0.90, 0.01,
        help="Higher values value future rewards more. This determines how much portals 'hurt' by delaying the goal.")
    episodes = c2.select_slider("Training episodes",
        [100, 250, 500, 1000, 2000, 3000, 5000], value=2000,
        help="Number of complete episodes sampled. Too few episodes prevent the agent from ever finding the exit.")
    max_steps = c3.select_slider("Max steps per training episode",
        [50, 100, 200, 300, 400, 500], value=300,
        help="Step cap per episode. Must be generous enough early on for random walks to find the exit.")

    e1, e2 = st.columns([1, 3])
    eps_kind = e1.selectbox("Exploration", [DECAYING, CONSTANT],
        help="Decaying explores early then commits; Constant maintains a fixed chance of random moves forever.")
    with e2:
        if eps_kind == CONSTANT:
            eps = st.slider("ε", 0.01, 0.5, 0.30, 0.01,
                help="Fixed chance of taking a random move instead of the best action. Values below ~0.2 struggle to find the exit.")
            eps_params = (eps,)
        else:
            d1, d2, d3 = st.columns(3)
            eps_start = d1.slider("ε start", 0.1, 1.0, 1.0, 0.05,
                help="Exploration rate at episode 1 (usually 1.0 for pure random exploration).")
            eps_min = d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01,
                help="The floor ε never drops below.")
            decay = d3.slider("ε decay rate", 0.990, 0.9999, 0.998, 0.0001,
                format="%.4f",
                help="Per-episode multiplier. Lower = faster commitment to greedy actions.")
            eps_params = (eps_start, eps_min, decay)

    train = st.button("🚀 Train", type="primary", use_container_width=True)
    return gamma, episodes, max_steps, eps_kind, eps_params, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 2 · Monte Carlo")
    st.caption("Cross the icy corridors to the exit — and avoid portal traps that throw you back to the start.")
    
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "Unlike Room 1, Monte Carlo knows **nothing** about the board physics. It learns entirely from sampled experience.\n\n"
            "* **Model-Free Learning:** The agent wanders randomly until it stumbles onto the goal, then averages returns back into $Q(s,a)$.\n"
            "* **Portal Traps:** Landing on a portal teleports you back to the start. The penalty is purely time-based, discounted by $\\gamma$.\n"
            "* **Pessimistic Estimates:** MC's self-estimate $V_{MC}(s)$ understates the true policy value because it averages over early random-walk episodes and includes $\\epsilon$-greedy noise.\n"
            "* **Usage:** Configure the environment and schedule -> **🚀 Train** -> Scrub checkpoints -> **▶️ Play Episode** (runs greedy, $\\epsilon=0$)."
        )

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
        setup_caption.caption("⚠️ Counts changed — click **🎲 Regenerate** to apply.")
    elif len(portals) < env["n_portals"]:
        setup_caption.caption(f"Placed {len(portals)} of {env['n_portals']} portals — others would have sealed the exit.")
    else:
        setup_caption.caption("Board layout — select an algorithm below and **🚀 Train**.")

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
              help="Share of final 100 training episodes that reached the exit within the step cap (includes ε-greedy noise).")
    m2.metric("Mean return (last 100)", f"{returns[last].mean():+.1f}",
              help="Average discounted return G over the final 100 training episodes.")

    # View controls above the board.
    key = "room2_view_cp"
    if key in st.session_state and st.session_state[key] > n_cp:
        st.session_state[key] = n_cp
    v_col, a_col = st.columns([3, 1])
    with v_col:
        cp_i = st.slider("View checkpoint", 1, n_cp, n_cp, key=key,
            help="Scrub through training checkpoints to watch value diffusion over time.") if n_cp > 1 else 1
    with a_col:
        show_arrows = st.checkbox("Show policy arrows", value=True)

    snap = history[cp_i - 1]
    V, policy, view_ep = snap["V"], snap["policy"], snap["episode"]

    V_greedy = _learned_policy_value(*keys, env["slip"], env["goal_reward"], gamma,
                                     tuple(sorted(policy.items())))
    v_gre_start, v_star_start = V_greedy[START], V_star[START]
    v_mc_start = V.get(START, 0.0)
    m3.metric("Start-state value V(S)", f"{v_mc_start:.1f}",
              help="MC's estimate of the start state value: max_a Q(S, a).")
    m4.metric("ε at checkpoint", f"{snap['eps']:.3f}")

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play** — greedy, ε = 0")
        play_max_steps = st.slider("Max steps per episode", 10, 500, 200,
            help="Step cap for this test rollout.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"], "Normal")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run a test rollout of the viewed policy with exploration turned off (ε = 0).")
        episode_slot = st.container()

    results_caption.caption(f"Value & greedy policy after **{view_ep:,}** episodes (checkpoint {cp_i} of {n_cp})")

    if play:
        path, G_ep, outcome, landings = rollout(
            grid, policy, gamma=gamma, max_steps=play_max_steps, with_landings=True)
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
                help=f"Total discounted return. Assigns a flat {LOSS_SCORE:+.0f} if the agent times out.")
            e2.metric("Steps", len(path) - 1)
            e3.metric("Result", "✅" if outcome == "goal" else "❌")
            if outcome != "goal":
                st.caption(f"Scored as {LOSS_SCORE:+.0f} for timing out; raw return was {G_ep:+.1f}.")
            if portals_hit:
                st.caption(f"🌀 Sent back to start {portals_hit}× this run.")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows),
            use_container_width=True, key="room2_results_board")

    # --- Learning curves ---------------------------------------------------- #
    st.plotly_chart(_returns_curve(returns, V_star[START], view_ep),
                    use_container_width=True)
    st.plotly_chart(_steps_curve(steps, success, view_ep), use_container_width=True)
    st.plotly_chart(_epsilon_curve(stats["eps"], view_ep), use_container_width=True)
    st.caption("As exploration ($\epsilon$) drops, returns climb and step counts fall. The remaining gap to $V^*$ is largely residual exploration noise.")

    # --- DP benchmark row --------------------------------------------------- #
    st.divider()
    st.markdown("#### 📐 Benchmark against exact Dynamic Programming")
    st.caption("Compares Monte Carlo's sampled estimate against Room 1's exact model solution ($V^*$). Arrow differences show where sampling hasn't converged to the true optimal policy yet.")

    n1, n2, n3 = st.columns(3)
    n1.metric("V_MC(S) — what MC believes", f"{v_mc_start:.1f}",
              help="max_a Q(S,a) — MC's internal estimate of the start state.")
    n2.metric("True V(S) of that policy", f"{v_gre_start:.1f}",
              help="The actual model-evaluated value of the learned greedy policy.")
    n3.metric("V*(S) — exact optimal", f"{v_star_start:.1f}",
              help="The theoretical best possible return, computed via DP.")
    
    st.markdown(
        "**Why $V_{MC}$ understates the True Policy Value:**\n"
        "* **$\epsilon$-Greedy Drag:** $Q$ values reflect the policy *plus* random exploratory moves, while True $V(S)$ evaluates the pure greedy policy ($\epsilon=0$).\n"
        "* **Lifetime Averaging:** Standard MC uses a $1/N$ step size, meaning poor returns from early random-walk episodes permanently drag down the average."
    )

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**V_MC — Sampled ({view_ep:,} episodes)**")
        st.plotly_chart(
            _figure(grid, V, policy, show_arrows, height=420),
            use_container_width=True, key="room2_bench_mc")
    with b2:
        st.markdown("**V\\* — Exact Optimal (DP)**")
        st.plotly_chart(
            _figure(grid, V_star, pi_star, show_arrows, height=420),
            use_container_width=True, key="room2_bench_dp")