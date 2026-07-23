from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms.dynamic_programming import policy_value, value_iteration
from algorithms.temporal_difference import CONSTANT, DECAYING, sarsa_control
from algorithms.monte_carlo import moving_average
from core.episode import LOSS_SCORE, rollout, scored_return
from core.icy_grid import IcyGridWorld, generate_layout, generate_shields

START = (9, 0)
GOAL = (9, 9)
CLIFF = frozenset((9, j) for j in range(1, 9))
LEDGE = frozenset((8, j) for j in range(1, 9))
CLIFF_REWARD = -100.0

_ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
_STEP_DELAY = {"Slow": 0.45, "Normal": 0.22, "Fast": 0.08}
_LEGEND = ("🤖 Start · 🏁 Exit · 🕳️ Abyss (terminal fall) · "
           "🧱 Wall · 🟦 Ice (slippery) · 🛡️ Shield (stops slipping)")
_MA_WINDOW = 50


def _make_grid(blocked, ice, shields, slip, goal_reward, seed=None):
    return IcyGridWorld(
        start=START, goal=GOAL, blocked=blocked, ice=ice, shields=shields,
        slip=slip, goal_reward=goal_reward, pits={c: CLIFF_REWARD for c in CLIFF},
        rng=np.random.default_rng(seed))


def _project(grid, table, layer):
    """Flatten a state-keyed table onto the 2D board for one shield layer."""
    if not grid.stateful:
        return dict(table)
    return {grid.cell_of(s): v for s, v in table.items()
            if grid.shield_of(s) == layer}


# ----------------------------------------------------------------------------- #
# Cached compute
# ----------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _train(blocked_t, ice_t, shields_t, slip, goal_reward, gamma, alpha,
           episodes, max_steps, eps_kind, eps_params, seed):
    grid = _make_grid(set(blocked_t), set(ice_t), set(shields_t), slip,
                      goal_reward, seed)
    _, _, history, stats = sarsa_control(
        grid, gamma=gamma, alpha=alpha, n_episodes=episodes, max_steps=max_steps,
        eps_kind=eps_kind, eps_params=eps_params, seed=seed)
    return history, stats


@st.cache_data(show_spinner=False)
def _dp_optimal(blocked_t, ice_t, shields_t, slip, goal_reward, gamma):
    """Exact V* for this board — the benchmark SARSA is measured against."""
    grid = _make_grid(set(blocked_t), set(ice_t), set(shields_t), slip, goal_reward)
    V, policy, _ = value_iteration(grid, gamma=gamma)
    return V, policy


@st.cache_data(show_spinner=False)
def _learned_policy_value(blocked_t, ice_t, shields_t, slip, goal_reward, gamma,
                          policy_t):
    """Exact value of a LEARNED policy — what it is really worth."""
    grid = _make_grid(set(blocked_t), set(ice_t), set(shields_t), slip, goal_reward)
    return policy_value(grid, dict(policy_t), gamma)


def _regenerate_layout(env, seed, version):
    blocked, ice, _ = generate_layout(
        env["n_blocked"], env["n_slippery"], 0, seed,
        start=START, goal=GOAL, exclude=CLIFF, pits=set(CLIFF) | set(LEDGE))
    shields = generate_shields(blocked, env["n_shields"], seed, start=START,
                               goal=GOAL, exclude=set(ice) | set(CLIFF), pits=CLIFF)
    st.session_state["room3_layout"] = {
        "blocked": blocked, "ice": ice, "shields": shields, "version": version,
        "counts": (env["n_blocked"], env["n_slippery"], env["n_shields"]),
    }


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #
def _cell_shapes(grid):
    shapes = []
    for (i, j) in grid.ice:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(14,165,233,0.55)", "width": 1.2},
            fillcolor="rgba(56,189,248,0.16)", layer="above"))
    for (i, j) in grid.shields:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(16,185,129,0.9)", "width": 2},
            fillcolor="rgba(16,185,129,0.20)", layer="above"))
    for (i, j) in CLIFF:
        shapes.append(dict(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "#4c1d95", "width": 2},
            fillcolor="rgba(124,58,237,0.85)", layer="above"))
    for (i, j) in grid.blocked:
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
            c = (i, j)
            if grid.is_blocked(c):
                z[i, j] = np.nan
                text[i, j] = "🧱"
            elif grid.is_pit(c):
                z[i, j] = np.nan
                text[i, j] = "🕳️"
            elif c == GOAL:
                z[i, j] = np.nan
                text[i, j] = "🏁"
            elif c == START:
                z[i, j] = V.get(c, 0.0)
                text[i, j] = "🤖"
            elif grid.is_shield(c):
                z[i, j] = V.get(c, 0.0)
                text[i, j] = "🛡️"
            else:
                z[i, j] = V.get(c, 0.0)
                text[i, j] = _ARROW[policy[c]] if (
                    show_arrows and policy and c in policy) else ""
    return z, text


def _figure(grid, V, policy, show_arrows, trail=None, agent=None, fell=False,
            shielded=False, height=520):
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
        colour = "#ef4444" if fell else ("#10b981" if shielded else "#f59e0b")
        fig.add_trace(go.Scatter(
            x=[agent[1]], y=[agent[0]], mode="markers",
            marker={"size": 30 if fell else 22, "color": colour,
                    "line": {"color": "#111827", "width": 2}},
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        shapes=_cell_shapes(grid),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=height)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig


def _falls_curve(falls, view_ep):
    n = len(falls)
    fig = go.Figure(go.Scatter(
        x=np.arange(1, n + 1), y=np.cumsum(falls), mode="lines",
        line={"color": "#dc2626", "width": 2}, name="cumulative falls"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"viewing ep {view_ep}")
    fig.update_yaxes(title="cumulative falls into abyss", rangemode="tozero")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Cliff falls — slope flattens as the ledge is learned")
    return fig


def _returns_curve(returns, success, v_star_start, view_ep):
    disp = np.where(success, returns, LOSS_SCORE)
    n = len(disp)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=disp, mode="markers",
        marker={"size": 3, "color": "rgba(59,130,246,0.35)"}, name="episode score"))
    fig.add_trace(go.Scatter(
        x=x, y=moving_average(disp, _MA_WINDOW), mode="lines",
        line={"color": "#1d4ed8", "width": 2}, name=f"{_MA_WINDOW}-episode average"))
    fig.add_hline(y=v_star_start, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"DP optimal V*(S) = {v_star_start:.1f}")
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b")
    fig.update_yaxes(title="return (escape = real G · loss = -100)")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Episode score — climbs toward V*, losses floored at -100",
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
    fig.update_yaxes(title="steps in episode")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Steps per episode — early short runs are falls, not shortcuts",
                      legend={"orientation": "h", "y": -0.2})
    return fig


# ----------------------------------------------------------------------------- #
# Controls
# ----------------------------------------------------------------------------- #
def _env_controls():
    st.markdown("##### 🎮 Environment & Physics")
    n_blocked = st.slider("Blocked cells 🧱", 0, 20, 8,
        help="Impassable walls. A safe route avoiding the abyss and ledge is always preserved.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 20,
        help="Ice cells where movement may slide sideways. Never placed on the abyss.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.1, 0.05,
        help="Chance of sliding perpendicular to the intended direction on ice. At 0, all detour caution is driven purely by SARSA's exploration risk.")
    n_shields = st.slider("Shields 🛡️", 0, 2, 1,
        help="Pickups that grant permanent immunity to slipping. A strategic tradeoff: immunity vs. the discounting cost of detouring to grab it.")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit. All learned values scale relative to this and the fixed -100 fall penalty.")
    st.caption(f"🕳️ Abyss falls score **{CLIFF_REWARD:+.0f}**, so this slider sets the reward ratio between escaping and dying.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Reshuffle walls, ice, and shields. The abyss, start, and exit never move.")
    return {"n_blocked": n_blocked, "n_slippery": n_slippery,
            "n_shields": n_shields, "slip": slip,
            "goal_reward": float(goal_reward), "regen": regen}


def _algo_row():
    st.markdown("##### 🧠 Algorithm")
    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.slider("Learning rate α", 0.01, 0.5, 0.10, 0.01,
        help="Step size. Controls how quickly Q-values adapt to new step experiences.")
    gamma = c2.slider("Discount factor γ", 0.50, 0.99, 0.95, 0.01,
        help="Higher values plan further ahead; lower values make immediate safety preferable to distant rewards.")
    episodes = c3.select_slider("Training episodes",
        [500, 1000, 2000, 5000, 10000], value=2000,
        help="Number of training runs. SARSA converges to the best ε-greedy policy, not the pure optimal one.")
    max_steps = c4.select_slider("Max steps per training episode",
        [100, 200, 300, 500], value=200,
        help="Step limit per training episode.")

    e1, e2 = st.columns([1, 3])
    eps_kind = e1.selectbox("Exploration", [DECAYING, CONSTANT],
        help="Decaying explores early then commits; Constant maintains a fixed chance of random moves forever.")
    with e2:
        if eps_kind == CONSTANT:
            eps = st.slider("ε", 0.01, 0.5, 0.30, 0.01,
                help="Fixed chance of taking a random move. SARSA prices this risk directly into its path planning.")
            eps_params = (eps,)
        else:
            d1, d2, d3 = st.columns(3)
            eps_start = d1.slider("ε start", 0.1, 1.0, 1.0, 0.05,
                help="Exploration rate at episode 1 (1.0 = pure random exploration).")
            eps_min = d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01,
                help="The floor ε never drops below.")
            decay = d3.slider("ε decay rate", 0.990, 0.9999, 0.998, 0.0001,
                format="%.4f",
                help="Per-episode multiplier. Lower = faster commitment to greedy actions.")
            eps_params = (eps_start, eps_min, decay)

    train = st.button("🚀 Train", type="primary", use_container_width=True)
    return gamma, alpha, episodes, max_steps, eps_kind, eps_params, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 3 · SARSA")
    st.caption("Cross the slippery ice ledge over the abyss — and do not fall in.")
    
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "SARSA is an **on-policy** Temporal Difference control algorithm that learns from every step.\n\n"
            "* **Step-by-Step Learning:** It updates $Q(s,a)$ toward $r + \\gamma Q(s',a')$ using the *actual* next action $a'$, including $\\epsilon$-greedy exploration noise.\n"
            "* **The Cliff Lesson:** Because SARSA knows it might take a random step next, standing near the abyss is genuinely dangerous to it. It learns a safer detour away from the edge (unlike Q-learning in Room 4, which assumes perfect future play).\n"
            "* **Shield State:** Shields permanently stop ice slipping and expand the state space to $(i, j, \\text{has\\_shield})$.\n"
            "* **Usage:** Configure physics -> **🚀 Train** -> Compare the cautious learned policy against the exact DP optimal $V^*$."
        )

    # --- Row 1: setup board + environment controls -------------------------- #
    board_col, env_col = st.columns([3, 2])
    with board_col:
        setup_board = st.empty()
        setup_caption = st.empty()
        st.caption(_LEGEND)
    with env_col:
        env = _env_controls()

    if st.session_state.get("room3_layout") is None:
        _regenerate_layout(env, seed=0, version=0)
    if env["regen"]:
        v = st.session_state["room3_layout"]["version"] + 1
        _regenerate_layout(env, seed=v, version=v)
        st.session_state.pop("room3_trained_sig", None)

    layout = st.session_state["room3_layout"]
    blocked, ice, shields = layout["blocked"], layout["ice"], layout["shields"]
    grid = _make_grid(blocked, ice, shields, env["slip"], env["goal_reward"])

    zeros = {c: 0.0 for c in grid.cells()}
    setup_board.plotly_chart(_figure(grid, zeros, {}, show_arrows=False),
                             use_container_width=True, key="room3_setup_board")
    counts_now = (env["n_blocked"], env["n_slippery"], env["n_shields"])
    if counts_now != layout["counts"]:
        setup_caption.caption("⚠️ Counts changed — click **🎲 Regenerate** to apply.")
    elif len(shields) < env["n_shields"]:
        setup_caption.caption(f"Placed {len(shields)} of {env['n_shields']} shields — others were unreachable.")
    else:
        setup_caption.caption("Board layout — select an algorithm below and **🚀 Train**.")

    # --- Row 2: algorithm parameters ---------------------------------------- #
    st.divider()
    gamma, alpha, episodes, max_steps, eps_kind, eps_params, train = _algo_row()

    sig = (layout["version"], env["slip"], env["goal_reward"], gamma, alpha,
           episodes, max_steps, eps_kind, eps_params)
    if train:
        st.session_state["room3_trained_sig"] = sig
    if st.session_state.get("room3_trained_sig") != sig:
        return  # not trained for this configuration — no results yet

    # --- Row 3: training results -------------------------------------------- #
    keys = (tuple(sorted(blocked)), tuple(sorted(ice)), tuple(sorted(shields)))
    with st.spinner(f"Running {episodes:,} episodes of SARSA…"):
        history, stats = _train(*keys, env["slip"], env["goal_reward"], gamma,
                                alpha, episodes, max_steps, eps_kind, eps_params, 0)
        V_star, _ = _dp_optimal(*keys, env["slip"], env["goal_reward"], gamma)

    if not history:
        st.warning("No episodes were run.")
        return

    returns, steps = stats["returns"], stats["steps"]
    success, falls = stats["success"], stats["falls"]
    n_cp = len(history)

    st.divider()
    st.markdown("#### Training results")

    last = slice(-100, None)
    n_ep = len(returns)
    n_fell = int(falls.sum())
    n_goal = int(success.sum())
    n_timeout = n_ep - n_goal - n_fell
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🕳️ Falls (training)", f"{n_fell:,}",
              help="Training episodes that ended by falling into the abyss.")
    m2.metric("⏱️ Timeouts (training)", f"{n_timeout:,}",
              help="Training episodes that reached the step limit without escaping or falling.")
    m3.metric("Success rate (last 100)", f"{success[last].mean():.0%}",
              help="Share of final 100 training episodes that escaped (includes ε-greedy noise).")
    st.caption(f"Across all {n_ep:,} training episodes: 🏁 **{n_goal:,}** escaped · 🕳️ **{n_fell:,}** fell · ⏱️ **{n_timeout:,}** timed out.")

    if success[last].mean() < 0.2:
        st.info(
            "⚠️ **Low Escape Rate:** Near a fatal abyss, early random walks die quickly. If the agent accumulates too many falls before finding the exit, it may learn that the entire lower board is lethal and flee upward forever.\n\n"
            "Try **🎲 Regenerate**, switching to **Constant ε**, or increasing training episodes."
        )

    # View controls above the board.
    key = "room3_view_cp"
    if key in st.session_state and st.session_state[key] > n_cp:
        st.session_state[key] = n_cp
    v_col, a_col, s_col = st.columns([2, 1, 1.4])
    with v_col:
        cp_i = st.slider("View checkpoint", 1, n_cp, n_cp, key=key,
            help="Scrub through training checkpoints to watch path planning evolve.") if n_cp > 1 else 1
    with a_col:
        show_arrows = st.checkbox("Show policy arrows", value=True)
    with s_col:
        if grid.stateful:
            layer_lab = st.radio(
                "Value map", ["🛡️ Not collected", "🛡️ Collected"],
                horizontal=True,
                help="Switch layers to see cell values before and after collecting a shield.")
            layer = 1 if "Not" not in layer_lab else 0
        else:
            layer = 0

    snap = history[cp_i - 1]
    V_s, policy_s, view_ep = snap["V"], snap["policy"], snap["episode"]
    V, policy = _project(grid, V_s, layer), _project(grid, policy_s, layer)

    V_greedy = _learned_policy_value(*keys, env["slip"], env["goal_reward"],
                                     gamma, tuple(sorted(policy_s.items())))
    s0 = grid.start_state()
    v_gre_start, v_star_start = V_greedy[s0], V_star[s0]
    m4.metric("V(S) of this policy", f"{v_gre_start:.1f}",
              help="Actual model-evaluated value of the learned greedy policy from the start state.")

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play** — greedy, ε = 0")
        play_max_steps = st.slider("Max steps per episode", 10, 500, 200,
            help="Step limit for this test rollout.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"], "Normal")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run a test rollout of the viewed policy with exploration turned off (ε = 0).")
        episode_slot = st.container()

    results_caption.caption(f"Value & greedy policy after **{view_ep:,}** episodes (checkpoint {cp_i} of {n_cp})")

    if play:
        path, G_ep, outcome = rollout(grid, policy_s, gamma=gamma,
                                      max_steps=play_max_steps)
        cells = [grid.cell_of(s) for s in path]
        for k in range(len(path)):
            fell_here = outcome == "fell" and k == len(path) - 1
            has_shield = bool(grid.shield_of(path[k]))
            board_V = _project(grid, V_s, 1 if has_shield else 0)
            board_pi = _project(grid, policy_s, 1 if has_shield else 0)
            results_board.plotly_chart(
                _figure(grid, board_V, board_pi, show_arrows,
                        trail=cells[: k + 1], agent=cells[k], fell=fell_here,
                        shielded=has_shield),
                use_container_width=True, key=f"room3_ep_{k}")
            time.sleep(_STEP_DELAY[speed])

        picked = any(grid.shield_of(s) for s in path)
        score = scored_return(G_ep, outcome)
        with episode_slot:
            if outcome == "goal":
                st.success("🏁 Escaped! The agent crossed the ledge.")
            elif outcome == "fell":
                st.error("🕳️ Fell into the abyss — run ended.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return G", f"{score:+.1f}",
                help=f"Total discounted return. Assigns a flat {LOSS_SCORE:+.0f} if the agent falls or times out.")
            e2.metric("Steps", len(path) - 1)
            e3.metric("Result", "✅" if outcome == "goal" else "❌")
            if outcome != "goal":
                st.caption(f"Scored as {LOSS_SCORE:+.0f}; raw discounted return was {G_ep:+.1f}.")
            if grid.shields:
                st.caption("🛡️ Shield collected — no slipping." if picked else "🛡️ No shield collected — slipped the whole way.")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows),
            use_container_width=True, key="room3_results_board")
        if grid.stateful:
            st.caption(f"Arrows show the plan for the **{layer_lab}** layer.")

    # --- Learning curves ---------------------------------------------------- #
    st.plotly_chart(_falls_curve(falls, view_ep), use_container_width=True)
    st.caption("Falls level off as the ledge is learned, but never drop to zero while random exploration ($\epsilon > 0$) continues.")
    st.plotly_chart(_returns_curve(returns, success, v_star_start, view_ep),
                    use_container_width=True)
    st.caption("Average returns climb as escapes increase. The remaining gap to $V^*$ reflects SARSA's deliberate caution and exploration noise.")
    st.plotly_chart(_steps_curve(steps, success, view_ep), use_container_width=True)
    st.caption("Short runs early on are fatal falls into the abyss, not efficient shortcuts.")