from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms.dynamic_programming import (
    ALGORITHMS, THETA, expected_steps_to_goal, success_prob_within)
from core.episode import rollout
from core.icy_grid import IcyGridWorld, generate_layout

START = (9, 0)
GOAL = (0, 9)

_ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
_STEP_DELAY = {"Slow": 0.45, "Normal": 0.22, "Fast": 0.08}
_LEGEND = "🤖 Start · 🏁 Goal · 🧱 Wall · 🟦 Ice (slippery) · 🟥 Penalty cell"


def _make_grid(blocked, ice, negatives, neg_reward, slip, goal_reward):
    return IcyGridWorld(
        start=START, goal=GOAL, blocked=blocked, ice=ice,
        penalties={c: neg_reward for c in negatives}, slip=slip,
        goal_reward=goal_reward)


@st.cache_data(show_spinner=False)
def _solve(blocked_t, ice_t, neg_t, neg_reward, slip, goal_reward, gamma, algo):
    grid = _make_grid(set(blocked_t), set(ice_t), set(neg_t), neg_reward, slip,
                      goal_reward)
    _, _, history = ALGORITHMS[algo](grid, gamma=gamma)
    return history


@st.cache_data(show_spinner=False)
def _expected_steps(blocked_t, ice_t, neg_t, neg_reward, slip, goal_reward, gamma, algo):
    grid = _make_grid(set(blocked_t), set(ice_t), set(neg_t), neg_reward, slip,
                      goal_reward)
    history = _solve(blocked_t, ice_t, neg_t, neg_reward, slip, goal_reward, gamma, algo)
    return expected_steps_to_goal(grid, history[-1]["policy"])


# ----------------------------------------------------------------------------- #
# Layout persistence — only regenerated on click, not on slider drag.
# ----------------------------------------------------------------------------- #
def _regenerate_layout(env, seed, version):
    blocked, ice, negatives = generate_layout(
        env["n_blocked"], env["n_slippery"], env["n_negative"], seed)
    st.session_state["room1_layout"] = {
        "blocked": blocked, "ice": ice, "negatives": negatives,
        "version": version,
        "counts": (env["n_blocked"], env["n_slippery"], env["n_negative"]),
    }


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #
def _cell_shapes(blocked, ice, negatives):
    shapes = []
    for (i, j) in ice:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(14,165,233,0.55)", "width": 1.2},
            fillcolor="rgba(56,189,248,0.16)", layer="above"))
    for (i, j) in negatives:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(239,68,68,0.8)", "width": 2},
            fillcolor="rgba(239,68,68,0.16)", layer="above"))
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
            elif s == GOAL:
                z[i, j] = np.nan
                text[i, j] = "🏁"
            elif s == START:
                z[i, j] = V.get(s, 0.0)
                text[i, j] = "🤖"
            else:
                z[i, j] = V.get(s, 0.0)
                text[i, j] = _ARROW[policy[s]] if (show_arrows and policy and s in policy) else ""
    return z, text


def _figure(grid, V, policy, show_arrows, trail=None, agent=None):
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
            marker={"size": 22, "color": "#f59e0b",
                    "line": {"color": "#111827", "width": 2}},
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        shapes=_cell_shapes(grid.blocked, grid.ice, set(grid.penalties.keys())),
        margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=520)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig


def _convergence_curve(deltas, view_it):
    fig = go.Figure(go.Scatter(
        y=deltas, x=list(range(1, len(deltas) + 1)),
        mode="lines+markers", line={"color": "#3b82f6"}, name="max Δ"))
    fig.add_hline(y=THETA, line_dash="dash", line_color="#ef4444",
                  annotation_text="θ = 1e-3")
    if len(deltas) > 1:
        fig.add_vline(x=view_it, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"viewing #{view_it}")
    fig.update_yaxes(type="log", title="max value change (log)")
    fig.update_xaxes(title="iteration")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=300)
    return fig


# ----------------------------------------------------------------------------- #
# Controls
# ----------------------------------------------------------------------------- #
def _env_controls():
    st.markdown("##### 🎮 Environment & Physics")
    n_blocked = st.slider("Blocked cells 🧱", 0, 20, 8,
        help="Impassable walls. A valid path to the exit is always preserved.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 15,
        help="Ice cells where movement may slide sideways.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.2, 0.05,
        help="Chance of sliding perpendicular to the intended direction on ice.")
    n_negative = st.slider("Negative-reward cells 🟥", 0, 15, 6,
        help="Passable penalty cells that reduce return when entered.")
    neg_reward = st.slider("Negative reward value", -10, -1, -5,
        help="Reward penalty applied when entering a red cell.")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit. Higher values encourage riskier, faster paths.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Generate a new layout with the selected cell counts.")
    return {
        "n_blocked": n_blocked, "n_slippery": n_slippery, "n_negative": n_negative,
        "slip": slip, "neg_reward": neg_reward, "goal_reward": goal_reward,
        "regen": regen,
    }


def _algo_row():
    st.markdown("##### 🧠 Algorithm")
    a1, a2 = st.columns(2)
    algo = a1.selectbox("DP method", list(ALGORITHMS.keys()))
    gamma = a2.slider("Discount factor γ", 0.50, 0.99, 0.90, 0.01,
        help="Higher values plan further ahead; lower values favor immediate rewards.")
    train = st.button("🚀 Train", type="primary", use_container_width=True)
    return algo, gamma, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 1 · Dynamic Programming")
    st.caption("Navigate from start to exit while managing slip risk and penalty cells.")
    
    with st.expander("ℹ️ About this room", expanded=False):
        st.markdown(
            "Dynamic Programming uses the **full environment model** to compute the optimal value function $V(s)$ and policy $\\pi(s)$.\n\n"
            "* **Value Diffusion:** High expected returns near the goal diffuse backward across the grid, decreasing with distance, risk (ice), and penalties (red cells).\n"
            "* **Terminal Goal:** The goal is terminal, so its value is fixed at $V(\\text{goal}) = 0$.\n"
            "* **Usage:** Configure the board and algorithm, click **🚀 Train**, then scrub through iterations or click **▶️ Play Episode** to test the policy."
        )

    # --- Row 1: setup board + environment controls -------------------------- #
    board_col, env_col = st.columns([3, 2])
    with board_col:
        setup_board = st.empty()
        setup_caption = st.empty()
        st.caption(_LEGEND)
    with env_col:
        env = _env_controls()

    # Layout persistence: generate once, then only on 🎲 Regenerate.
    if st.session_state.get("room1_layout") is None:
        _regenerate_layout(env, seed=0, version=0)
    if env["regen"]:
        v = st.session_state["room1_layout"]["version"] + 1
        _regenerate_layout(env, seed=v, version=v)
        st.session_state.pop("room1_trained_sig", None)

    layout = st.session_state["room1_layout"]
    blocked, ice, negatives = layout["blocked"], layout["ice"], layout["negatives"]
    grid = _make_grid(blocked, ice, negatives, env["neg_reward"], env["slip"],
                      env["goal_reward"])

    zeros = {s: 0.0 for s in grid.all_states()}
    setup_board.plotly_chart(_figure(grid, zeros, {}, show_arrows=False),
                             use_container_width=True, key="setup_board")
    counts_now = (env["n_blocked"], env["n_slippery"], env["n_negative"])
    if counts_now != layout["counts"]:
        setup_caption.caption("⚠️ Counts changed — click **🎲 Regenerate** to apply.")
    else:
        setup_caption.caption("Board layout — select an algorithm below and **🚀 Train**.")

    # --- Row 2: algorithm parameters ---------------------------------------- #
    st.divider()
    algo, gamma, train = _algo_row()

    sig = (algo, gamma, env["slip"], env["neg_reward"], env["goal_reward"],
           layout["version"])
    if train:
        st.session_state["room1_trained_sig"] = sig
    if st.session_state.get("room1_trained_sig") != sig:
        return  # not trained for this configuration — no results yet

    # --- Row 3: training results -------------------------------------------- #
    keys = (tuple(sorted(blocked)), tuple(sorted(ice)), tuple(sorted(negatives)))
    history = _solve(*keys, env["neg_reward"], env["slip"], env["goal_reward"],
                     gamma, algo)
    deltas = [h["delta"] for h in history]
    final = history[-1]
    n = len(history)

    exp_steps = _expected_steps(*keys, env["neg_reward"], env["slip"],
                                env["goal_reward"], gamma, algo)

    st.divider()
    st.markdown("#### Training results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iterations to converge", n,
              help="Rounds until max value change dropped below θ (1e-3).")
    m2.metric("Final max Δ", f"{final['delta']:.2e}",
              help="Max value change in the final round.")
    m3.metric("Start-state value V(S)", f"{final['V'][START]:.1f}",
              help="Expected discounted return from the start state.")
    m4.metric("Expected steps to exit", f"{exp_steps:.1f}",
              help="Theoretical average steps to reach the exit under the optimal policy.")

    # View controls on a single row above the board.
    key = "room1_view_iter"
    if key in st.session_state and st.session_state[key] > n:
        st.session_state[key] = n
    v_col, a_col = st.columns([3, 1])
    with v_col:
        view_it = st.slider("View iteration", 1, n, n, key=key,
            help="Scrub through the value function and policy at each iteration.") if n > 1 else 1
    with a_col:
        show_arrows = st.checkbox("Show policy arrows", value=True)
    snap = history[view_it - 1]
    V, policy = snap["V"], snap["policy"]

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play**")
        max_steps = st.slider("Max steps per episode", 10, 50, 50,
            help="Step limit before the episode times out.")
        succ = success_prob_within(grid, policy, max_steps)
        st.metric("Success within cap", f"{succ:.0%}",
            help="Theoretical probability of reaching the goal within the step limit.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"], "Normal")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Simulate one rollout of the selected policy.")
        episode_slot = st.container()

    results_caption.caption(f"Showing **{algo}** · iteration {view_it} of {n}")

    if play:
        path, g, outcome = rollout(grid, policy, gamma=gamma, max_steps=max_steps)
        for k in range(len(path)):
            fig = _figure(grid, V, policy, show_arrows,
                          trail=path[: k + 1], agent=path[k])
            results_board.plotly_chart(fig, use_container_width=True, key=f"ep_{k}")
            time.sleep(_STEP_DELAY[speed])
        with episode_slot:
            if outcome == "goal":
                st.success("🏁 Escaped! The agent reached the exit.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return G", f"{g:+.1f}",
                help="Total discounted return for this run. Average over many runs equals V(S).")
            e2.metric("Steps", len(path) - 1)
            e3.metric("Result", "✅" if outcome == "goal" else "❌")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows),
            use_container_width=True, key="results_board")

    # Convergence graph — its own full-width row below the board + play controls.
    st.plotly_chart(_convergence_curve(deltas, view_it), use_container_width=True)