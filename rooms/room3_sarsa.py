"""
Room 3 — SARSA (on-policy temporal-difference control).

Task: cross the slippery ice ledge over the abyss without falling in.

Board cell types (see the legend under the board):
  * 🕳️ abyss — TERMINAL. Fall in and the episode ends then and there,
  * 🧱 blocked — walls the agent cannot step into (placed by count, as in Rooms 1-2),
  * 🟦 slippery ice — moves may slip perpendicular (placed by count, as in Rooms 1-2),
  * 🛡️ shield — pick one up and the ice stops slipping you FOR THE REST OF THE RUN.

The abyss geometry is FIXED and is the whole lesson: the start and the exit sit
on opposite lips of the same chasm, so the direct route runs along the ledge
(row 8) and the safe route detours upward. 🎲 Regenerate reshuffles the walls,
ice and shields; the abyss, the start, and the exit never move.

THE SHIELD CHANGES WHAT A STATE IS. It is carried, so whether a move slips
depends on what the agent picked up earlier — which means the cell alone is not
Markov, and states here are (i, j, has_shield). `IcyGridWorld` handles that (see
its STATE SHAPE note); the algorithms never notice, because they treat a state as
an opaque key. This module must therefore never index V/Q by a bare (i, j): use
`grid.cell_of` / `grid.shield_of` and `grid.start_state()`. The board is drawn one
shield-layer at a time — see `_project`.

The shield is a TEMPTATION, not a free upgrade, and that is the interesting part.
Measured across 5 random boards: *holding* a shield is worth +6.7 (slip 0.1) to
+12.7 (slip 0.8) at the start — yet the optimal policy detours to pick one up on
only 1 of 5 boards, and at slip 0.8 on none of them. Walking to it costs more
discounting than the immunity gives back. So the honest question the board poses
is "is this detour worth it?", and DP answers it exactly, per board, rather than
asserting.

A caution about reading the shielded layer: SARSA only learns it on states it
actually reaches WHILE holding a shield. Arrows on the shielded layer far from
its shielded route are unvisited noise, not considered opinions — the same
off-distribution caveat that applies to any Q-learned map.

Why the abyss is TERMINAL rather than Sutton & Barto's reset-to-start cliff:
falling is fatal, which needs no new reward machinery (a pit is terminal, so the
agent ends up standing on it and the ordinary resulting-state reward lookup
carries the -100), and it is far more robust to epsilon — measured 94-96% success
at eps = 0.10, 0.30 and decaying alike, where a *passable* penalty cell scores 0%
at eps = 0.10. The classic reset cliff is a penalty on a TELEPORT cell, whose
reward `IcyGridWorld` would silently drop; see the Room 3 section of Plan.md.

What this room exists to show: SARSA bootstraps off the action it will ACTUALLY
take next (`Q[s2][a2]`), drawn from the same epsilon-greedy policy it is
following. So the risk of exploring near a cliff is priced into the value of
standing near the cliff, and the learned route backs away from the edge. That
caution is not free: the policy settles at roughly 63-87% of V*, and the gap IS
the lesson rather than underfitting to be tuned away. Room 4's Q-learning
bootstraps off max_a Q[s2][a] instead and will walk the ledge.

Honest caveat surfaced in the UI: slip PARTIALLY CONFOUNDS the lesson. "Points
away from the hazard" is only a fact about SARSA if the OPTIMAL policy hugs the
ledge. It does at slip = 0 (V* crosses on row 8, SARSA detours to row 1); by
slip = 0.1 the optimal has already backed off to row 7 itself. That is why the
DP V*(S) is still shown (the return-curve reference line and the V(S) KPI) — it
reveals how far SARSA sits from optimal at the user's slip setting instead of
letting SARSA take credit for caution the physics demanded.

Page flow follows docs/UI_STRUCTURE.md:
  Row 1 — About + setup board + 🎮 Environment controls.
  Row 2 — 🧠 Algorithm parameters + 🚀 Train.
  Row 3 — Training results: KPIs, a checkpoint scrubber, the results board +
          ▶️ Play (greedy, eps = 0), and the learning curves.
"""
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
# The abyss: row 9 between the start and the exit. FIXED — this geometry is the
# lesson, so it is never randomized (see the module docstring).
CLIFF = frozenset((9, j) for j in range(1, 9))
# The LEDGE: cells orthogonally touching the abyss (row 8, columns 1-8). Standing
# here is where an ε-greedy random move, or a slip, kills you. Walls must never
# make the ledge the ONLY way to the exit — see _regenerate_layout.
LEDGE = frozenset((8, j) for j in range(1, 9))
# Fixed, not a control. The goal reward is the scale everything is measured
# against (see Plan.md's Core UI Rules), so the room gives the user ONE side of
# that ratio to move; a second slider would only vary the same ratio twice. -100
# also mirrors the standardized +100 exit and the -100 scoreboard penalty for
# never getting out — falling and giving up cost the same.
CLIFF_REWARD = -100.0

_ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
_STEP_DELAY = {"Slow": 0.45, "Normal": 0.22, "Fast": 0.08}
_LEGEND = ("🤖 start · 🏁 exit · 🕳️ abyss (terminal — falling in ends the run) · "
           "🧱 wall · 🟦 slippery ice · 🛡️ shield (collect it and you stop slipping)")
_MA_WINDOW = 50


def _make_grid(blocked, ice, shields, slip, goal_reward, seed=None):
    # seed=None → fresh entropy, so ▶️ Play Episode slips differently each run.
    # Training passes an explicit seed for reproducible curves; the DP grids
    # never touch rng at all.
    return IcyGridWorld(
        start=START, goal=GOAL, blocked=blocked, ice=ice, shields=shields,
        slip=slip, goal_reward=goal_reward, pits={c: CLIFF_REWARD for c in CLIFF},
        rng=np.random.default_rng(seed))


def _project(grid, table, layer):
    """Flatten a state-keyed table onto the 2D board for one shield layer.

    With shields the state is (i, j, k), so a cell has TWO values — what it is
    worth before you have a shield and after. A board can only draw one at a
    time; `layer` picks which. Without shields the table is already cell-keyed
    and passes straight through.
    """
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
    """Exact value of a LEARNED policy — what it is really worth.

    SARSA's own max_a Q understates the greedy policy it plays (Q is the value of
    the epsilon-greedy agent, which keeps exploring), so the only honest answer
    comes from evaluating the policy against the true model. Displayed only; the
    learner never sees it.
    """
    grid = _make_grid(set(blocked_t), set(ice_t), set(shields_t), slip, goal_reward)
    return policy_value(grid, dict(policy_t), gamma)


def _regenerate_layout(env, seed, version):
    # exclude=CLIFF: nothing else may be drawn on the abyss — a pit is never
    # acted from, so ice there could not affect the model anyway, and a cell
    # drawn as two hazards at once just reads as a bug.
    # The wall guard must reject any wall that leaves no route to the exit which
    # AVOIDS THE LEDGE — not merely no route at all. Two separate reasons:
    #   * CLIFF: a fall is fatal, so a "path" through the chasm is not a path.
    #   * LEDGE: measured, ~1 board in 6 walled off the descent to the exit and
    #     left row 8 as the only approach. SARSA — whose entire character is
    #     refusing to walk beside a cliff — then never escapes: 0% success and
    #     V^π = 0 against V* = 59.9, unfixed by 20,000 episodes. The room would
    #     just look broken. Requiring a ledge-free route means any detour SARSA
    #     takes is a CHOICE it made, which is the whole lesson; walls now shape
    #     the route without ever dictating a cliff-hug.
    # Passing them as `pits` marks them impassable FOR THE GUARD ONLY — the ledge
    # stays perfectly walkable in the actual environment.
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
        # Violet, not near-black: the abyss used to sit at rgba(15,23,42) right
        # beside dark-grey walls, and the two read as the same thing — fatal vs
        # merely impassable is the single most important distinction on this
        # board. Violet is the one free slot in the room's palette (walls grey,
        # ice blue, shields green, agent amber) and it is absent from the RdBu
        # value scale, so it never blends into a negative-value cell next to it.
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
    """V and policy here are CELL-keyed — already projected to one shield layer."""
    z = np.zeros((grid.rows, grid.cols))
    text = np.empty((grid.rows, grid.cols), dtype=object)
    for i in range(grid.rows):
        for j in range(grid.cols):
            c = (i, j)
            if grid.is_blocked(c):
                z[i, j] = np.nan
                text[i, j] = "🧱"
            elif grid.is_pit(c):
                # Terminal, exactly like the goal — the agent never acts from a
                # pit, so it has no V. Masked and drawn as an icon only.
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
    """V/policy must be CELL-keyed (run them through `_project` first)."""
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
        # Green while shielded, so "it stopped slipping" is visible on the board
        # rather than something the user has to infer from the trail.
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
    """Cumulative cliff falls — levels off, but never flattens to zero."""
    n = len(falls)
    fig = go.Figure(go.Scatter(
        x=np.arange(1, n + 1), y=np.cumsum(falls), mode="lines",
        line={"color": "#dc2626", "width": 2}, name="cumulative falls"))
    fig.add_vline(x=view_ep, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"viewing ep {view_ep}")
    fig.update_yaxes(title="cumulative falls into the abyss", rangemode="tozero")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Cliff falls — the slope flattens as the ledge is learned")
    return fig


def _returns_curve(returns, success, v_star_start, view_ep):
    # DISPLAY scoring, matching ▶️ Play: an escape shows its real discounted G,
    # every loss (fall or timeout) is floored to -100. This is a scoreboard view
    # only — SARSA never learns from episode G (it updates off per-step rewards),
    # and the stored stats stay raw, so nothing computational is affected.
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
    fig.update_yaxes(title="return  (escape = real G · any loss = -100)")
    fig.update_xaxes(title="episode")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 48, "b": 10}, height=300,
                      title="Episode score — escapes climb toward V*, every loss floored at -100",
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
                      title="Steps per episode — short runs early are falls, not efficiency",
                      legend={"orientation": "h", "y": -0.2})
    return fig


# ----------------------------------------------------------------------------- #
# Controls
# ----------------------------------------------------------------------------- #
def _env_controls():
    st.markdown("##### 🎮 Environment & Physics")
    n_blocked = st.slider("Blocked cells 🧱", 0, 20, 8,
        help="Walls the agent cannot step into. Placement always keeps a route "
        "from the start to the exit that does NOT cross the abyss — a wall that "
        "would leave falling as the only way across is skipped.")
    n_slippery = st.slider("Slippery cells 🟦", 0, 40, 20,
        help="Icy cells (shaded blue) where a move may slip perpendicular. Never "
        "placed on the abyss. Ice makes surprisingly little difference here: the "
        "exact optimal value only falls from 59.9 to 51.9 going from 0 to 40 "
        "cells, because scattered ice rarely lands on the route that matters.")
    slip = st.slider("Slip probability", 0.0, 0.8, 0.1, 0.05,
        help="On an ice cell, the chance a move sends you perpendicular instead "
        "of straight ahead — and possibly over the edge. Set it to 0 to see the "
        "cleanest version of this room's lesson: with no slip the OPTIMAL route "
        "hugs the ledge, so every step SARSA takes away from it is caution that "
        "SARSA chose, not caution the ice forced.")
    n_shields = st.slider("Shields 🛡️", 0, 2, 1,
        help="Pickups that make the agent immune to slipping for the rest of the "
        "run — placed only where it can actually be reached and still leave a way "
        "to the exit. A shield is a temptation, not a free upgrade: HOLDING one is "
        "worth roughly +7 to +13 from the start, but walking over to fetch it costs "
        "discounting, and more often than not the optimal policy decides it is not "
        "worth the trip — a call that depends on your exact board.")
    goal_reward = st.slider("Goal reward 🏁", 10, 1000, 100, 10,
        help="Reward for reaching the exit — the only positive reward on the "
        "board. It is the scale everything else is measured against: the fall is "
        "always -100, so this slider sets how much the exit is worth RELATIVE to "
        "dying. Even at 10 against a -100 fall, escaping still beats loitering — "
        "so the agent goes for it anyway (measured 100% at every setting).")
    st.caption(
        f"🕳️ The fall is fixed at **{CLIFF_REWARD:+.0f}**, and every loss scores "
        f"**{LOSS_SCORE:+.0f}** on the scoreboard — so the slider above moves the "
        "one ratio that matters.")
    regen = st.button("🎲 Regenerate layout", use_container_width=True,
        help="Reshuffle the walls, ice, and shields. The abyss, start, and exit "
        "never move — that geometry is what the room is about.")
    return {"n_blocked": n_blocked, "n_slippery": n_slippery,
            "n_shields": n_shields, "slip": slip,
            "goal_reward": float(goal_reward), "regen": regen}


def _algo_row():
    st.markdown("##### 🧠 Algorithm")
    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.slider("Learning rate α", 0.01, 0.5, 0.10, 0.01,
        help="How far each step moves Q toward the new estimate. Unlike Monte "
        "Carlo's 1/N average, SARSA keeps a constant α forever, so it never stops "
        "adapting — and never fully settles either.")
    gamma = c2.slider("Discount factor γ", 0.50, 0.99, 0.95, 0.01,
        help="How much future reward is worth versus immediate. Low γ makes the "
        "distant +100 exit invisible from the start, so the agent has no reason "
        "to risk the crossing at all.")
    episodes = c3.select_slider("Training episodes",
        [500, 1000, 2000, 5000, 10000], value=2000,
        help="How many episodes SARSA plays. More is NOT reliably better here: "
        "with a constant ε, SARSA converges to the best ε-GREEDY policy, not to "
        "the optimal one, so the gap to V* below does not close with more "
        "episodes. That gap is the room's point.")
    max_steps = c4.select_slider("Max steps per training episode",
        [100, 200, 300, 500], value=200,
        help="Cap on each TRAINING episode. It matters little in this room — a "
        "fall ends an episode outright, so runs are short either way.")

    e1, e2 = st.columns([1, 3])
    eps_kind = e1.selectbox("Exploration", [DECAYING, CONSTANT],
        help="ε is the chance of ignoring the current best action and moving at "
        "random. It is also what SARSA prices in: the agent knows it will keep "
        "making these random moves, so standing next to the abyss is genuinely "
        "dangerous to it. Decaying is the default — it explores hard early, then "
        "commits, and it also produces the MOST cautious policy of any setting "
        "measured (63% of optimal), which is this room's whole point.")
    with e2:
        if eps_kind == CONSTANT:
            eps = st.slider("ε", 0.01, 0.5, 0.30, 0.01,
                help="Fixed exploration rate — the agent keeps exploring at this "
                "rate forever, and keeps paying for it. Caution does NOT rise "
                "smoothly with ε: measured, ε of 0.05/0.10/0.30/0.50 reach "
                "83%/80%/66%/87% of optimal, so 0.50 is both the most "
                "exploratory and the best scoring. 0.30 shows the clearest "
                "detour of the constant settings.")
            eps_params = (eps,)
        else:
            d1, d2, d3 = st.columns(3)
            eps_start = d1.slider("ε start", 0.1, 1.0, 1.0, 0.05,
                help="Exploration rate at episode 1. Start at 1.0 for a pure "
                "random walk — with an all-zero Q there is nothing to exploit.")
            eps_min = d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01,
                help="Floor ε never drops below. Note that ending at a low ε does "
                "NOT make the final policy bold: decaying ε measured as the most "
                "conservative setting of all, at 63% of optimal.")
            decay = d3.slider("ε decay rate", 0.990, 0.9999, 0.998, 0.0001,
                format="%.4f",
                help="Per-episode multiplier: ε(k) = max(ε min, ε start · rate^k). "
                "Lower = faster commitment. Match this to your episode count.")
            eps_params = (eps_start, eps_min, decay)

    train = st.button("🚀 Train", type="primary", use_container_width=True,
        help="Run SARSA on the current board.")
    return gamma, alpha, episodes, max_steps, eps_kind, eps_params, train


# ----------------------------------------------------------------------------- #
# Main render
# ----------------------------------------------------------------------------- #
def render():
    st.markdown("### Room 3 · SARSA")
    st.caption("Cross the slippery ice ledge over the abyss — and do not fall in.")
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "Monte Carlo had to finish a whole episode before it learned anything. "
            "**SARSA learns from every single step**, updating `Q(s,a)` toward "
            "`r + γ·Q(s′,a′)` as it goes — where `a′` is the action it will "
            "**actually take next**, exploration and all.\n\n"
            "That detail is the entire room. Because SARSA knows it will keep "
            "making random moves, standing next to the abyss is genuinely "
            "dangerous *to it* — so it learns to back away from the edge. The "
            "caution costs real value: the policy settles well below the optimal "
            "`V*`, and **that gap is the lesson, not a bug to tune away**. Room 4's "
            "Q-learning assumes it will always act perfectly next, and walks the "
            "ledge instead.\n\n"
            "**How to use it:** shape the board, then **🚀 Train**. Compare the "
            "arrows against the exact DP answer at the bottom — and try "
            "**slip = 0**, where the optimal route hugs the ledge, so every step "
            "SARSA takes away from it is caution it *chose* rather than caution "
            "the ice forced on it.\n\n"
            "The exit is worth **+100** by default and the fall is always **−100**. "
            "Turning the exit down to **10** makes escaping look barely worth the "
            "risk — but it never actually gives up, because loitering pays 0 and "
            "escaping still pays *something*. Discounting shrinks the numbers "
            "without changing which way the arrows point.")

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
        setup_caption.caption("⚠️ Counts changed — click 🎲 Regenerate to apply.")
    elif len(shields) < env["n_shields"]:
        setup_caption.caption(
            f"Placed {len(shields)} of {env['n_shields']} shields — the rest had "
            "nowhere reachable to go.")
    else:
        setup_caption.caption("Board layout — set the algorithm below and 🚀 Train.")

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
              help="Training episodes that ended with the agent falling into the "
              "abyss. It levels off but never stops: ε keeps making random moves "
              "and slip keeps pushing the agent over the edge.")
    m2.metric("⏱️ Timeouts (training)", f"{n_timeout:,}",
              help="Training episodes that ran out of steps without reaching the "
              "exit or falling in — the agent wandered and never finished.")
    m3.metric("Success rate (last 100)", f"{success[last].mean():.0%}",
              help="Share of the final 100 training episodes that reached the "
              "exit. Measured while still exploring, so it sits below what "
              "▶️ Play (ε = 0) achieves.")
    st.caption(
        f"Across all {n_ep:,} training episodes: 🏁 **{n_goal:,}** escaped · "
        f"🕳️ **{n_fell:,}** fell into the abyss · ⏱️ **{n_timeout:,}** timed out.")

    # A run that never finds the exit is a real (if uncommon) SARSA outcome, not
    # a broken room — but "0%" alone just looks like a bug. Say what happened.
    if success[last].mean() < 0.2:
        st.info(
            "**The agent never found the exit — this is a real failure, not a "
            "display bug.** Early on it explores at random, and beside a fatal "
            "abyss a random walk dies within a couple of steps. If it collects "
            "enough of those deaths before it ever stumbles on the exit, it "
            "learns that the whole bottom of the board is lethal — and the exit "
            "sits at the far end of that same bottom row. It then commits to "
            "fleeing upward forever. The dashed line on the return curve marks "
            "V*(S): DP knows a good route exists, because it has the model and "
            "never had to survive learning it.\n\n"
            "Try **🎲 Regenerate** for another board, switch to **Constant ε**, or "
            "raise the episode count.")

    # View controls above the board.
    key = "room3_view_cp"
    if key in st.session_state and st.session_state[key] > n_cp:
        st.session_state[key] = n_cp
    v_col, a_col, s_col = st.columns([2, 1, 1.4])
    with v_col:
        cp_i = st.slider("View checkpoint", 1, n_cp, n_cp, key=key,
            help="Replay the value function and greedy policy as they stood at "
            f"each of {n_cp} checkpoints across training.") if n_cp > 1 else 1
    with a_col:
        show_arrows = st.checkbox("Show policy arrows", value=True,
            help="Overlay the greedy action in each cell for the viewed checkpoint.")
    with s_col:
        if grid.stateful:
            layer_lab = st.radio(
                "Value map", ["🛡️ Not collected", "🛡️ Collected"],
                horizontal=True,
                help="A shield is CARRIED, so every cell has two values: what it "
                "is worth before you have one, and after. The board can only draw "
                "one at a time. 'Not collected' is the layer the agent starts in.")
            layer = 1 if "Not" not in layer_lab else 0
        else:
            layer = 0

    snap = history[cp_i - 1]
    V_s, policy_s, view_ep = snap["V"], snap["policy"], snap["episode"]
    # Cell-keyed views for drawing; the raw state-keyed tables stay for the maths.
    V, policy = _project(grid, V_s, layer), _project(grid, policy_s, layer)

    V_greedy = _learned_policy_value(*keys, env["slip"], env["goal_reward"],
                                     gamma, tuple(sorted(policy_s.items())))
    s0 = grid.start_state()
    v_gre_start, v_star_start = V_greedy[s0], V_star[s0]
    m4.metric("V(S) of this policy", f"{v_gre_start:.1f}",
              help="What the viewed policy is really worth from the start, "
              "evaluated exactly against the model — not SARSA's own estimate of "
              "itself, which is lower because it is the value of the ε-greedy agent "
              "that keeps exploring, not of the greedy policy ▶️ Play runs.")

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play** — greedy, ε = 0")
        play_max_steps = st.slider("Max steps per episode", 10, 500, 200,
            help="Cap for THIS playback only — separate from the training cap "
            "above.")
        speed = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"],
            "Normal", help="Playback speed of the animated episode.")
        play = st.button("▶️ Play Episode", type="primary",
            use_container_width=True,
            help="Run the viewed policy with exploration switched off (ε = 0) "
            "across the real, stochastic ice. With ε = 0 the agent no longer "
            "makes the random moves SARSA was so careful about — but the ice "
            "still slips.")
        episode_slot = st.container()

    results_caption.caption(
        f"Value & greedy policy after **{view_ep:,}** episodes "
        f"(checkpoint {cp_i} of {n_cp})")

    # An episode is EPHEMERAL: it lives only in the run that played it. Nothing
    # goes to session state — a stored rollout outlives the policy it was run
    # against, so scrubbing to another checkpoint would redraw a stale trail over
    # a policy that never produced it.
    if play:
        # The policy passed to rollout must be the STATE-keyed one — the grid
        # steps through (i, j, k) states, not cells.
        path, G_ep, outcome = rollout(grid, policy_s, gamma=gamma,
                                      max_steps=play_max_steps)
        cells = [grid.cell_of(s) for s in path]
        for k in range(len(path)):
            fell_here = outcome == "fell" and k == len(path) - 1
            has_shield = bool(grid.shield_of(path[k]))
            # Follow the agent's own shield status as it plays, so the moment it
            # picks one up the map flips to the layer it is actually acting on.
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
                st.error("🕳️ Fell into the abyss — the run ends here.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return G", f"{score:+.1f}",
                help="On a WIN this is the real discounted return G = Σ γ^t·r₍t+1₎. "
                f"On ANY loss — falling in or timing out — the scoreboard shows a "
                f"flat {LOSS_SCORE:+.0f}, no matter when or how, mirroring the +100 "
                "exit. One sample of a stochastic rollout: play again and it differs.")
            e2.metric("Steps", len(path) - 1,
                help="Number of moves before the episode ended.")
            e3.metric("Result", "✅" if outcome == "goal" else "❌",
                help="Whether the agent reached the exit.")
            if outcome != "goal":
                st.caption(
                    f"Every loss scores a flat {LOSS_SCORE:+.0f}; the raw discounted "
                    f"return this run was {G_ep:+.1f}.")
            if grid.shields:
                st.caption(
                    "🛡️ Picked up the shield — no more slipping from there on."
                    if picked else
                    "🛡️ Never picked up a shield — it slipped the whole way.")
    else:
        results_board.plotly_chart(
            _figure(grid, V, policy, show_arrows),
            use_container_width=True, key="room3_results_board")
        if grid.stateful:
            st.caption(
                "Arrows show the plan for the **" + layer_lab + "** layer — the "
                "same cell can be worth two different things depending on whether "
                "the agent is carrying a shield.")

    # --- Learning curves ---------------------------------------------------- #
    st.plotly_chart(_falls_curve(falls, view_ep), use_container_width=True)
    st.caption(
        "The curve bends as SARSA learns the ledge, but it never goes flat — with "
        "ε > 0 the agent keeps taking random moves near a fatal edge, and on ice "
        "it keeps slipping. A conservative policy reduces falls; it cannot end "
        "them while it is still exploring.")
    st.plotly_chart(_returns_curve(returns, success, v_star_start, view_ep),
                    use_container_width=True)
    st.caption(
        "Every losing episode is scored -100 here, exactly as ▶️ Play scores it, so "
        "wins and losses read the same in training as on the scoreboard. The average "
        "climbs as SARSA escapes more often and settles below V* — the shortfall is "
        "both its ε-caution and the losses it never fully stops making. (SARSA learns "
        "from per-step rewards, not this number, so the flooring is display only.)")
    st.plotly_chart(_steps_curve(steps, success, view_ep), use_container_width=True)
