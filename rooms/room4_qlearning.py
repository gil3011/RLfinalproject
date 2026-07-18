"""
Room 4 — Q-Learning (off-policy temporal-difference control).

Task: cross Room 3's cliff board while a patrol guard sweeps the safe upper
detour, with one bonus coin sitting out on the ledge over the abyss.

What this room exists to show: **Q-learning bootstraps off `max_a Q(s′,a)`** — the
value of acting greedily next, as if it will never explore again — so from a
distance the ledge looks safe and the coin looks free. SARSA bootstraps off the
action it will *actually* take next, exploration and all, so it prices in the risk
of a random step over the edge and detours. Both are trained here, on the SAME
board with identical hyperparameters, because the contrast only means anything as
a controlled experiment (Room 3's returns are a different MDP and cannot be
imported).

The honest, measured finding — and it is the opposite of the tidy story:
**Q-learning walks the ledge and grabs the coin on essentially every board, but
the EXACT optimal policy only agrees a minority of the time.** So Q-learning is
not "aggressive and right" — it is *over-optimistic about the risky route*, and
SARSA is usually the one making the optimal call. The DP answer — the V*(S)
reference line and the one-line route caption — is what says so, per board. Do not
claim Q-learning earns more value: across boards the
two are statistically indistinguishable in `V^π`; only the ROUTE contrast (and
Q-learning's much higher fall count) is robust. See Plan.md §Room 4.

STATE SHAPE. A state here is `(i, j, g, m)`: cell, guard patrol phase, coin
bitmask. Whether a step is fatal depends on where the guard is *now* (phase `g`),
exactly the Markov problem Room 3's shield posed. `GuardGrid` augments the state;
the algorithms treat it as an opaque key and need no changes. This module never
indexes a table by a bare cell — it uses `grid.cell_of/phase_of/mask_of` and
`grid.start_state()`, and projects a state-keyed table onto one `(phase, mask)`
layer to draw it (see `_project`).

Page flow follows docs/UI_STRUCTURE.md, mirroring Room 3 so the rooms differ in
the algorithm, not the dashboard.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms.dynamic_programming import policy_value, value_iteration
from algorithms.monte_carlo import moving_average
from algorithms.temporal_difference import (CONSTANT, DECAYING, q_learning_control,
                                             sarsa_control)
from core.episode import LOSS_SCORE, rollout, scored_return
from core.guard_grid import (CAUGHT_REWARD, CLIFF, CLIFF_REWARD, GOAL, GUARD_COL,
                             GuardGrid, LEDGE, START, make_track, place_coin)
from core.icy_grid import generate_layout

_ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
_STEP_DELAY = {"Slow": 0.45, "Normal": 0.22, "Fast": 0.08}
_LEGEND = ("🤖 start · 🏁 exit · 🕳️ abyss (terminal) · 🚨 patrol guard (terminal if it "
           "catches you) · 🪙 coin (one-time bonus) · 🧱 wall · 🟦 slippery ice")
_MA_WINDOW = 50
_QL, _SA = "Q-learning", "SARSA"
_QL_COLOR, _SA_COLOR = "#dc2626", "#2563eb"


def _make_grid(blocked, ice, coin, slip, goal_reward, coin_value, seed=None):
    # seed=None → fresh entropy so ▶️ Play slips differently each run; training
    # passes an explicit seed for reproducible curves; DP grids never touch rng.
    return GuardGrid(
        blocked=blocked, ice=ice, coins=coin, track=make_track(), slip=slip,
        goal_reward=goal_reward, coin_value=coin_value,
        rng=np.random.default_rng(seed))


# ----------------------------------------------------------------------------- #
# Cached compute
# ----------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _train(kind, blocked_t, ice_t, coin_t, slip, goal_reward, coin_value, gamma,
           alpha, episodes, max_steps, eps_kind, eps_params, seed):
    grid = _make_grid(set(blocked_t), set(ice_t), coin_t, slip, goal_reward,
                      coin_value, seed)
    control = q_learning_control if kind == _QL else sarsa_control
    _, _, history, stats = control(
        grid, gamma=gamma, alpha=alpha, n_episodes=episodes, max_steps=max_steps,
        eps_kind=eps_kind, eps_params=eps_params, seed=seed)
    return history, stats


@st.cache_data(show_spinner=False)
def _dp_optimal(blocked_t, ice_t, coin_t, slip, goal_reward, coin_value, gamma):
    grid = _make_grid(set(blocked_t), set(ice_t), coin_t, slip, goal_reward, coin_value)
    V, policy, _ = value_iteration(grid, gamma=gamma)
    return V, policy


@st.cache_data(show_spinner=False)
def _learned_policy_value(blocked_t, ice_t, coin_t, slip, goal_reward, coin_value,
                          gamma, policy_t):
    grid = _make_grid(set(blocked_t), set(ice_t), coin_t, slip, goal_reward, coin_value)
    return policy_value(grid, dict(policy_t), gamma)


def _regenerate_layout(env, seed, version):
    # Coin first (central ledge), then walls excluding it, the abyss and the
    # patrol column, so a wall never lands on the guard's path or the coin.
    # pits = CLIFF ∪ LEDGE forces a guaranteed route that avoids the cliff EDGE,
    # exactly as Room 3 does — a fatal ledge-only approach reads as a broken room.
    coin = place_coin(seed)
    track = set(make_track())
    blocked, ice, _ = generate_layout(
        env["n_blocked"], env["n_slippery"], 0, seed, start=START, goal=GOAL,
        exclude=set(CLIFF) | track | set(coin), pits=set(CLIFF) | set(LEDGE))
    st.session_state["room4_layout"] = {
        "blocked": blocked, "ice": ice, "coin": coin, "version": version,
        "counts": (env["n_blocked"], env["n_slippery"]),
    }


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #
def _project(grid, table, phase, mask):
    """Flatten a state-keyed table onto the 2D board for one (phase, mask) layer.

    A state is (i, j, g, m); a cell therefore has many values (one per guard phase
    and coin state). A board draws one layer at a time.
    """
    return {grid.cell_of(s): v for s, v in table.items()
            if grid.phase_of(s) == phase and grid.mask_of(s) == mask}


def _greedy_trace(grid, policy, max_len=200):
    """Cells visited following `policy` greedily through the most-likely transition."""
    s, cells = grid.start_state(), []
    for _ in range(max_len):
        cells.append(grid.cell_of(s))
        if grid.is_terminal(s) or s not in policy:
            break
        s = max(grid.probs[(s, policy[s])].items(), key=lambda kv: kv[1])[0]
    return cells


def _takes_coin(grid, policy):
    s = grid.start_state()
    for _ in range(200):
        if grid.is_terminal(s) or s not in policy:
            break
        s = max(grid.probs[(s, policy[s])].items(), key=lambda kv: kv[1])[0]
    return grid.mask_of(s) > 0


def _cell_shapes(grid, phase, mask, coin_cell):
    shapes = []
    for (i, j) in grid.ice:
        shapes.append(dict(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "rgba(14,165,233,0.55)", "width": 1.2},
            fillcolor="rgba(56,189,248,0.16)", layer="above"))
    for (i, j) in CLIFF:
        # Violet abyss (as Room 3): fatal, and it must never read like a mere wall.
        shapes.append(dict(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "#4c1d95", "width": 2},
            fillcolor="rgba(124,58,237,0.85)", layer="above"))
    if coin_cell is not None and not (mask & 1):
        i, j = coin_cell
        shapes.append(dict(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "#b45309", "width": 2},
            fillcolor="rgba(245,158,11,0.30)", layer="above"))
    for (i, j) in grid.blocked:
        shapes.append(dict(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line={"color": "#111827", "width": 1},
            fillcolor="rgba(55,65,81,0.85)", layer="above"))
    # The guard's cell at the viewed phase — crimson, absent from RdBu, distinct
    # from the violet abyss. The agent must not end a move here.
    gi, gj = grid.guard_at(phase)
    shapes.append(dict(type="rect", x0=gj - 0.5, x1=gj + 0.5, y0=gi - 0.5, y1=gi + 0.5,
        line={"color": "#7f1d1d", "width": 2},
        fillcolor="rgba(220,38,38,0.80)", layer="above"))
    return shapes


def _base_grid(grid, V, policy, show_arrows, phase, mask, coin_cell):
    """V/policy are CELL-keyed (already projected to one (phase, mask) layer)."""
    z = np.full((grid.rows, grid.cols), np.nan)
    text = np.empty((grid.rows, grid.cols), dtype=object)
    guard_cell = grid.guard_at(phase)
    for i in range(grid.rows):
        for j in range(grid.cols):
            c = (i, j)
            if grid.is_blocked(c):
                text[i, j] = "🧱"
            elif c in CLIFF:
                text[i, j] = "🕳️"
            elif c == GOAL:
                text[i, j] = "🏁"
            elif c == guard_cell:
                text[i, j] = "🚨"                      # terminal at this phase
            elif coin_cell is not None and c == coin_cell and not (mask & 1):
                text[i, j] = "🪙"                      # not yet collected
            elif c == START:
                z[i, j] = V.get(c, 0.0)
                text[i, j] = "🤖"
            else:
                z[i, j] = V.get(c, 0.0)
                text[i, j] = _ARROW[policy[c]] if (
                    show_arrows and policy and c in policy) else ""
    return z, text


def _figure(grid, V, policy, show_arrows, phase, mask, coin_cell, trail=None,
            agent=None, dead=False, height=520):
    z, text = _base_grid(grid, V, policy, show_arrows, phase, mask, coin_cell)
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
        colour = "#ef4444" if dead else "#f59e0b"
        fig.add_trace(go.Scatter(
            x=[agent[1]], y=[agent[0]], mode="markers",
            marker={"size": 30 if dead else 22, "color": colour,
                    "line": {"color": "#111827", "width": 2}},
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        shapes=_cell_shapes(grid, phase, mask, coin_cell),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=height)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig


def _overlay_figure(grid, ql_cells, sa_cells, coin_cell, height=520):
    """Both learners' greedy ROUTES on one board (phase-0 backdrop, no timing)."""
    z, text = _base_grid(grid, {}, {}, False, 0, 0, coin_cell)
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}", textfont={"size": 16},
        colorscale="RdBu", zmid=0, showscale=False, hoverinfo="skip"))
    for cells, colour, name, dash in (
            (sa_cells, _SA_COLOR, "SARSA — detours", "solid"),
            (ql_cells, _QL_COLOR, "Q-learning — walks the ledge", "dot")):
        fig.add_trace(go.Scatter(
            x=[c for _, c in cells], y=[r for r, _ in cells], mode="lines+markers",
            line={"color": colour, "width": 3, "dash": dash},
            marker={"size": 7, "color": colour}, name=name))
    fig.update_layout(
        shapes=_cell_shapes(grid, 0, 0, coin_cell),
        margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=height,
        legend={"orientation": "h", "y": -0.05})
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig


def _falls_curve(falls_ql, falls_sa, view_ep):
    n = len(falls_ql)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(falls_ql), mode="lines",
        line={"color": _QL_COLOR, "width": 2}, name="Q-learning"))
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(falls_sa), mode="lines",
        line={"color": _SA_COLOR, "width": 2}, name="SARSA"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b")
    fig.update_yaxes(title="cumulative falls into the abyss", rangemode="tozero")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
        title="Falls into the abyss — Q-learning walks the edge, so it falls far more",
        legend={"orientation": "h", "y": -0.2})
    return fig


def _returns_curve(returns_ql, success_ql, returns_sa, success_sa, v_star, view_ep):
    # DISPLAY scoring, matching ▶️ Play: an escape shows its real discounted G, every
    # loss (fall, catch or timeout) is floored to -100. Display only — the learners
    # update off per-step rewards, not episode G, and the stored stats stay raw.
    disp_ql = np.where(success_ql, returns_ql, LOSS_SCORE)
    disp_sa = np.where(success_sa, returns_sa, LOSS_SCORE)
    n = len(disp_ql)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=moving_average(disp_sa, _MA_WINDOW), mode="lines",
        line={"color": _SA_COLOR, "width": 2}, name="SARSA (50-ep avg)"))
    fig.add_trace(go.Scatter(x=x, y=moving_average(disp_ql, _MA_WINDOW), mode="lines",
        line={"color": _QL_COLOR, "width": 2}, name="Q-learning (50-ep avg)"))
    fig.add_hline(y=v_star, line_dash="dash", line_color="#111827",
        annotation_text=f"DP optimal V*(S) = {v_star:.1f}")
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b")
    fig.update_yaxes(title="return  (escape = real G · any loss = -100)")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
        title="Episode score — every loss floored at -100, both trained here",
        legend={"orientation": "h", "y": -0.2})
    return fig


def _steps_curve(steps_ql, steps_sa, view_ep):
    # Two learners, so plot the 50-episode averages (matching the returns curve)
    # rather than Room 3's single-learner scatter.
    n = len(steps_ql)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=moving_average(steps_sa, _MA_WINDOW), mode="lines",
        line={"color": _SA_COLOR, "width": 2}, name="SARSA (50-ep avg)"))
    fig.add_trace(go.Scatter(x=x, y=moving_average(steps_ql, _MA_WINDOW), mode="lines",
        line={"color": _QL_COLOR, "width": 2}, name="Q-learning (50-ep avg)"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b")
    fig.update_yaxes(title="steps in episode", rangemode="tozero")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
        title="Steps per episode — short early runs are deaths, not efficiency",
        legend={"orientation": "h", "y": -0.2})
    return fig


# ----------------------------------------------------------------------------- #
# Controls
# ----------------------------------------------------------------------------- #
def _env_controls():
    st.markdown("##### 🎮 Environment & Physics")
    n_blocked = st.slider("Blocked cells 🧱", 0, 20, 8,
        help="Walls the agent cannot step into. Never placed on the abyss, the "
        "guard's column, or the coin. Placement always keeps a route to the exit "
        "that avoids the cliff edge.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 20,
        help="Icy cells where a move may slip perpendicular — and possibly over "
        "the edge or into the guard's path.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.1, 0.05,
        help="On an ice cell, the chance a move sends you perpendicular instead of "
        "straight ahead. Higher slip makes both routes riskier and mutes the "
        "contrast between the two learners.")
    coin_value = st.slider("Coin value 🪙", 0, 20, 5, 1,
        help="Reward for the one bonus coin, which sits out on the ledge over the "
        "abyss. This is NOT inert here (unlike Room 3's goal slider): it sets how "
        "much the risky ledge route is worth. Measured, the OPTIMAL policy starts "
        "detouring onto the ledge to grab it somewhere between 1 and 10 depending "
        "on the board — below that it ignores the coin; at 0 there is no coin.")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit — the only reward besides the coin. The "
        "fall and a guard catch are both fixed at -100, so this sets the scale the "
        "coin's value is judged against.")
    st.caption(
        f"🕳️ A fall and 🚨 a catch each cost **{CLIFF_REWARD:+.0f}**; on the "
        f"scoreboard every loss — fall, catch, or timeout — scores a flat "
        f"**{LOSS_SCORE:+.0f}**.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Reshuffle the walls, ice and coin. The abyss, the guard's patrol, "
        "the start and the exit never move — that geometry is the lesson.")
    return {"n_blocked": n_blocked, "n_slippery": n_slippery, "slip": slip,
            "coin_value": float(coin_value), "goal_reward": float(goal_reward),
            "regen": regen}


def _algo_row():
    st.markdown("##### 🧠 Algorithm — both SARSA and Q-learning are trained here")
    st.caption(
        "Q-learning is this room's algorithm; SARSA is trained alongside it on the "
        "**same board with identical hyperparameters**, so the comparison below is "
        "controlled rather than a coincidence.")
    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.slider("Learning rate α", 0.01, 0.5, 0.10, 0.01,
        help="How far each step moves Q toward the new estimate.")
    gamma = c2.slider("Discount factor γ", 0.50, 0.99, 0.95, 0.01,
        help="How much future reward is worth versus immediate — sets how far the "
        "distant exit and the coin reach back toward the start.")
    episodes = c3.select_slider("Training episodes", [5000, 10000, 20000, 50000],
        value=20000,
        help="Per learner. This room needs far more than Room 3: its state space "
        "is ~20× larger (guard phase × coin), so at 5,000 episodes the policy value "
        "is still swamped by seed noise. 20,000 is the measured floor for a stable "
        "comparison; both learners together train in a few seconds.")
    max_steps = c4.select_slider("Max steps per training episode", [100, 200, 300, 500],
        value=200, help="Cap on each training episode. Matters little — a fall or a "
        "catch ends an episode early either way.")

    e1, e2 = st.columns([1, 3])
    eps_kind = e1.selectbox("Exploration", [DECAYING, CONSTANT],
        help="ε is the chance of ignoring the current best action and moving at "
        "random. It is also the risk SARSA prices in and Q-learning ignores. "
        "Decaying is the app-wide default — explore hard early, then commit.")
    with e2:
        if eps_kind == CONSTANT:
            eps = st.slider("ε", 0.01, 0.5, 0.30, 0.01,
                help="Fixed exploration rate. Note lower ε does NOT help here — it "
                "means less of the huge state space gets visited, so the tables are "
                "less trained, not safer.")
            eps_params = (eps,)
        else:
            d1, d2, d3 = st.columns(3)
            eps_start = d1.slider("ε start", 0.1, 1.0, 1.0, 0.05,
                help="Exploration rate at episode 1. Start high — a cold, all-zero "
                "Q has nothing worth exploiting yet.")
            eps_min = d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01,
                help="Floor ε never drops below.")
            decay = d3.slider("ε decay rate", 0.990, 0.9999, 0.9995, 0.0001,
                format="%.4f",
                help="Per-episode multiplier ε(k) = max(ε min, ε start · rate^k). "
                "With 20,000 episodes a slower decay than Room 3's keeps exploration "
                "alive long enough to cover the larger state space.")
            eps_params = (eps_start, eps_min, decay)

    train = st.button("🚀 Train both", type="primary", use_container_width=True,
        help="Run SARSA and Q-learning on the current board with these settings.")
    return gamma, alpha, episodes, max_steps, eps_kind, eps_params, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 4 · Q-Learning")
    st.caption("Time the patrol, or brave the ledge for a coin — and see which the "
               "two learners choose.")
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "Room 3's SARSA learned to **back away** from the abyss, because it "
            "prices in the risk of its own next random step. **Q-learning bootstraps "
            "off `max_a Q(s′,a)`** instead — the value of acting perfectly next — so "
            "from a distance the ledge looks safe and the coin looks free.\n\n"
            "This room makes both routes cost something. A **patrol guard** sweeps "
            "the safe upper detour (🚨, terminal if it catches you), so you must "
            "*time* a crossing — or take the ledge past the abyss, where a single "
            "🪙 coin waits. Both algorithms train here on the **same board**, so the "
            "comparison is fair.\n\n"
            "**What to watch:** Q-learning almost always grabs the ledge coin; SARSA "
            "almost always detours. Neither is simply *right* — the **exact DP "
            "answer** at the bottom shows, per board, who called it correctly. "
            "Q-learning tends to take the risk even when optimal would not "
            "(over-optimism); SARSA sometimes detours even when optimal would take "
            "the coin (over-caution). What is reliable is that Q-learning **falls "
            "far more** (the chart) — the classic Cliff-Walking signature.\n\n"
            "**How to use it:** shape the board, **🚀 Train both**, then compare the "
            "two routes on the overlay and against `V*`. Turn the **coin value** up "
            "and watch the optimal policy itself decide the ledge is finally worth it.")

    # --- Row 1: setup board + environment controls -------------------------- #
    board_col, env_col = st.columns([3, 2])
    with board_col:
        setup_board = st.empty()
        setup_caption = st.empty()
        st.caption(_LEGEND)
    with env_col:
        env = _env_controls()

    if st.session_state.get("room4_layout") is None:
        _regenerate_layout(env, seed=0, version=0)
    if env["regen"]:
        v = st.session_state["room4_layout"]["version"] + 1
        _regenerate_layout(env, seed=v, version=v)
        st.session_state.pop("room4_trained_sig", None)

    layout = st.session_state["room4_layout"]
    blocked, ice, coin = layout["blocked"], layout["ice"], layout["coin"]
    coin_cell = coin[0] if coin else None
    grid = _make_grid(blocked, ice, coin, env["slip"], env["goal_reward"],
                      env["coin_value"])

    setup_board.plotly_chart(
        _figure(grid, {}, {}, show_arrows=False, phase=0, mask=0, coin_cell=coin_cell),
        use_container_width=True, key="room4_setup_board")
    counts_now = (env["n_blocked"], env["n_slippery"])
    if counts_now != layout["counts"]:
        setup_caption.caption("⚠️ Counts changed — click 🎲 Regenerate to apply.")
    else:
        setup_caption.caption("Board layout — the guard 🚨 is shown at the start of "
                              "its patrol. Set the algorithm below and 🚀 Train both.")

    # --- Row 2: algorithm parameters ---------------------------------------- #
    st.divider()
    gamma, alpha, episodes, max_steps, eps_kind, eps_params, train = _algo_row()

    sig = (layout["version"], env["slip"], env["goal_reward"], env["coin_value"],
           gamma, alpha, episodes, max_steps, eps_kind, eps_params)
    if train:
        st.session_state["room4_trained_sig"] = sig
    if st.session_state.get("room4_trained_sig") != sig:
        return

    # --- Row 3: training results -------------------------------------------- #
    keys = (tuple(sorted(blocked)), tuple(sorted(ice)), coin)
    common = (env["slip"], env["goal_reward"], env["coin_value"])
    with st.spinner(f"Training SARSA and Q-learning, {episodes:,} episodes each…"):
        hist_ql, stats_ql = _train(_QL, *keys, *common, gamma, alpha, episodes,
                                   max_steps, eps_kind, eps_params, 0)
        hist_sa, stats_sa = _train(_SA, *keys, *common, gamma, alpha, episodes,
                                   max_steps, eps_kind, eps_params, 0)
        V_star, pi_star = _dp_optimal(*keys, *common, gamma)

    if not hist_ql:
        st.warning("No episodes were run.")
        return

    n_cp = len(hist_ql)
    s0 = grid.start_state()
    v_star_start = V_star[s0]

    st.divider()
    st.markdown("#### Training results")

    # --- View controls (pick the learner the KPIs + board reflect) ---------- #
    key = "room4_view_cp"
    if key in st.session_state and st.session_state[key] > n_cp:
        st.session_state[key] = n_cp
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1.2])
    with c1:
        viewed = st.radio("Show learner", [_QL, _SA], horizontal=True,
            help="Which learner the KPIs, board and ▶️ Play reflect. The comparison "
            "charts below always show both.")
    with c2:
        cp_i = st.slider("View checkpoint", 1, n_cp, n_cp, key=key,
            help="Replay the value function and greedy policy as they stood at each "
            f"of {n_cp} checkpoints across training.") if n_cp > 1 else 1
    with c3:
        show_arrows = st.checkbox("Policy arrows", value=True,
            help="Overlay the greedy action in each cell for the viewed layer.")
    with c4:
        phase = st.slider("Guard phase", 0, grid.P - 1, 0,
            help="Where the guard 🚨 stands. A cell's value changes with the guard's "
            "position — this is why the guard phase is part of the state.")

    hist = hist_ql if viewed == _QL else hist_sa
    stats = stats_ql if viewed == _QL else stats_sa
    snap = hist[cp_i - 1]
    V_s, policy_s, view_ep = snap["V"], snap["policy"], snap["episode"]
    mask = 0
    V = _project(grid, V_s, phase, mask)
    policy = _project(grid, policy_s, phase, mask)

    v_learned = _learned_policy_value(*keys, *common, gamma,
                                      tuple(sorted(policy_s.items())))[s0]

    # KPIs for the VIEWED learner — the same set as Room 3, plus the guard catch.
    n_goal = int(stats["success"].sum())
    n_fell = int(stats["falls"].sum())
    n_caught = int(stats["caught"].sum())
    n_timeout = episodes - n_goal - n_fell - n_caught
    last = slice(-100, None)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🕳️ Falls (training)", f"{n_fell:,}",
              help=f"Training episodes where {viewed} fell into the abyss. It walks "
              "the ledge, so it falls far more than SARSA — the Cliff-Walking "
              "signature.")
    m2.metric("🚨 Caught (training)", f"{n_caught:,}",
              help=f"Training episodes where the patrol guard caught {viewed}.")
    m3.metric("⏱️ Timeouts (training)", f"{n_timeout:,}",
              help=f"Training episodes where {viewed} ran out of steps without "
              "reaching the exit — it never finished.")
    m4.metric("Success rate (last 100)", f"{stats['success'][last].mean():.0%}",
              help="Share of the final 100 training episodes that reached the exit. "
              "Measured while still exploring, so it sits below what ▶️ Play (ε = 0) "
              "achieves.")
    m5.metric("V(S) of this policy", f"{v_learned:.1f}",
              help="What the viewed policy is really worth from the start, evaluated "
              "exactly against the model — not the learner's own estimate. The dashed "
              "line on the return curve marks the optimal V*(S).")
    st.caption(
        f"Across all {episodes:,} of **{viewed}**'s training episodes: 🏁 "
        f"**{n_goal:,}** escaped · 🕳️ **{n_fell:,}** fell · 🚨 **{n_caught:,}** "
        f"caught · ⏱️ **{n_timeout:,}** timed out.")

    # Route contrast, in one line — the robust finding.
    dp_coin = _takes_coin(grid, pi_star)
    ql_coin = _takes_coin(grid, hist_ql[-1]["policy"])
    sa_coin = _takes_coin(grid, hist_sa[-1]["policy"])
    yn = lambda b: "**takes** it" if b else "skips it"
    st.caption(
        f"🪙 On this board the exact optimal policy {yn(dp_coin)}; "
        f"Q-learning {yn(ql_coin)}; SARSA {yn(sa_coin)}. Where Q-learning takes the "
        "coin but the optimal skips it, its off-policy optimism has over-valued the "
        "risky route.")

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown(f"**▶️ Play** — {viewed}, greedy (ε = 0)")
        play_max_steps = st.slider("Max steps per episode", 10, 500, 200,
            help="Cap for THIS playback only — separate from the training cap.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"],
            "Normal", help="Playback speed of the animated episode.")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run the viewed policy with exploration off (ε = 0) across the real, "
            "stochastic board. The guard moves in lockstep; watch the agent time it.")
        episode_slot = st.container()

    results_caption.caption(
        f"**{viewed}** — value & greedy policy after **{view_ep:,}** episodes "
        f"(checkpoint {cp_i} of {n_cp}), guard at phase {phase}.")

    # An episode is EPHEMERAL — nothing goes to session state (UI_STRUCTURE).
    if play:
        path, G_ep, outcome = rollout(grid, policy_s, gamma=gamma,
                                      max_steps=play_max_steps)
        for k in range(len(path)):
            sk = path[k]
            dead_here = k == len(path) - 1 and outcome in ("fell", "caught")
            board_V = _project(grid, V_s, grid.phase_of(sk), grid.mask_of(sk))
            board_pi = _project(grid, policy_s, grid.phase_of(sk), grid.mask_of(sk))
            results_board.plotly_chart(
                _figure(grid, board_V, board_pi, show_arrows,
                        phase=grid.phase_of(sk), mask=grid.mask_of(sk),
                        coin_cell=coin_cell,
                        trail=[grid.cell_of(p) for p in path[: k + 1]],
                        agent=grid.cell_of(sk), dead=dead_here),
                use_container_width=True, key=f"room4_ep_{k}")
            time.sleep(_STEP_DELAY[speed])

        got_coin = grid.mask_of(path[-1]) > 0
        score = scored_return(G_ep, outcome)
        with episode_slot:
            if outcome == "goal":
                st.success("🏁 Escaped! The agent crossed to the exit.")
            elif outcome == "fell":
                st.error("🕳️ Fell into the abyss — the run ends here.")
            elif outcome == "caught":
                st.error("🚨 Caught by the guard — the run ends here.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return G", f"{score:+.1f}",
                help="On a WIN this is the real discounted return G = Σ γ^t·r₍t+1₎. "
                f"On ANY loss — falling, being caught, or timing out — the scoreboard "
                f"shows a flat {LOSS_SCORE:+.0f}, no matter when or how, mirroring the "
                "+100 exit. One stochastic sample — play again and it will differ.")
            e2.metric("Steps", len(path) - 1, help="Moves before the episode ended.")
            e3.metric("Result", "✅" if outcome == "goal" else "❌",
                help="Whether the agent reached the exit.")
            if outcome != "goal":
                st.caption(f"Every loss scores a flat {LOSS_SCORE:+.0f}; the raw "
                           f"discounted return this run was {G_ep:+.1f}.")
            if coin_cell is not None:
                st.caption("🪙 Grabbed the coin on the ledge." if got_coin
                           else "🪙 Left the coin — took the safer line.")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows, phase=phase, mask=mask,
                    coin_cell=coin_cell),
            use_container_width=True, key="room4_results_board")

    # --- Path comparison overlay -------------------------------------------- #
    st.markdown("##### 🛣️ Both routes at a glance")
    ql_cells = _greedy_trace(grid, hist_ql[-1]["policy"])
    sa_cells = _greedy_trace(grid, hist_sa[-1]["policy"])
    st.plotly_chart(_overlay_figure(grid, ql_cells, sa_cells, coin_cell),
                    use_container_width=True, key="room4_overlay")
    st.caption(
        "Q-learning's route (red, dotted) versus SARSA's (blue). The guard is drawn "
        "at the start of its patrol; the routes show WHERE each learner goes, not the "
        "timing. Q-learning typically hugs the ledge to reach the 🪙; SARSA detours.")

    # --- Learning curves ---------------------------------------------------- #
    st.plotly_chart(_falls_curve(stats_ql["falls"], stats_sa["falls"], view_ep),
                    use_container_width=True, key="room4_falls")
    st.plotly_chart(_returns_curve(stats_ql["returns"], stats_ql["success"],
                                   stats_sa["returns"], stats_sa["success"],
                                   v_star_start, view_ep),
                    use_container_width=True, key="room4_returns")
    st.caption(
        "Every losing episode is scored -100 here, exactly as ▶️ Play scores it. "
        "Q-learning's average is dragged down more early on because it falls far more "
        "while learning the ledge. The learners never see this number — they update "
        "off per-step rewards — so the flooring is display only.")
    st.plotly_chart(_steps_curve(stats_ql["steps"], stats_sa["steps"], view_ep),
                    use_container_width=True, key="room4_steps")
    st.caption(
        "A very short run is usually a quick death (a fall or a guard catch), not a "
        "fast escape — so early on both averages are pulled down by dying, and they "
        "settle toward each policy's true path length as escapes take over.")
