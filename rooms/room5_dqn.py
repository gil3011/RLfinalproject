"""
Room 5 · Deep Q-Learning — a continuous arena with one chasing enemy.

Unlike Rooms 1–4 (tabular dict-MDPs), Room 5 is continuous and model-free: a
`gymnasium` env (`core/chase_arena.py`) trained with a Double-DQN
(`algorithms/deep_q.py`, adapted from `code examples/dql/dqn.py`). The board is a
Plotly figure over metres, not a cell grid, and the room's visual argument is the
network's value FIELD — a value that exists between the sample points, which no
tabular room can draw.

Follows the shared UI contract (docs/UI_STRUCTURE.md): on-page controls, tooltips
everywhere, train-gated results, ephemeral Play episode.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.chase_arena import (
    ChaseArena, ARENA, START, EXIT, GOAL_RADIUS, CATCH_RADIUS,
)
from algorithms.deep_q import dqn_control, load_net, q_field, greedy_rollout
from algorithms.monte_carlo import CONSTANT, DECAYING, epsilon_at, moving_average
from core.episode import LOSS_SCORE

LEGEND = ("🤖 start (bottom-left) · 🏁 exit (top-right) · 🔴 enemy (fatal on "
          "contact) · value field: **blue = high**, red = low (RdBu, 0-centred)")

_STEP_DELAY = {"Slow": 0.16, "Normal": 0.08, "Fast": 0.03}
_EVAL_SEED = 4242            # fixed spawn for the scrubber/results, so it is stable


# ───────────────────────── board figure ─────────────────────────
def _arena_figure(enemies, agent=None, field=None, path=None, dead=False):
    """Plotly figure of the arena. `enemies` is a single (x, y) or an (n, 2) array;
    `field` is an optional (xs, ys, Z) value slice drawn as an RdBu heatmap
    underneath."""
    fig = go.Figure()

    if enemies is None:
        enemies = []
    else:
        arr = np.asarray(enemies, dtype=float)
        enemies = [arr] if arr.ndim == 1 else list(arr)

    if field is not None:
        xs, ys, Z = field
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=Z, colorscale="RdBu", zmid=0.0,
            colorbar=dict(title="max Q", thickness=12, len=0.9),
            hoverinfo="skip"))

    # exit disc + catch discs are shapes (true radii in metres), layer above the
    # heatmap trace (layer="below" is below TRACES → painted over; see memory).
    fig.add_shape(type="circle", x0=EXIT[0]-GOAL_RADIUS, y0=EXIT[1]-GOAL_RADIUS,
                  x1=EXIT[0]+GOAL_RADIUS, y1=EXIT[1]+GOAL_RADIUS,
                  fillcolor="rgba(38,166,91,0.55)", line=dict(color="white", width=2),
                  layer="above")
    for e in enemies:
        fig.add_shape(type="circle", x0=e[0]-CATCH_RADIUS, y0=e[1]-CATCH_RADIUS,
                      x1=e[0]+CATCH_RADIUS, y1=e[1]+CATCH_RADIUS,
                      fillcolor="rgba(231,76,60,0.30)", line=dict(color="rgba(231,76,60,0.9)", width=1),
                      layer="above")

    if path is not None and len(path) > 1:
        px, py = zip(*path)
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#f1c40f", width=2.5),
                                 hoverinfo="skip", showlegend=False))

    # markers on top: start, exit flag, enemy, agent
    fig.add_trace(go.Scatter(x=[START[0]], y=[START[1]], mode="markers+text",
                             marker=dict(size=16, color="#2ecc71", symbol="square",
                                         line=dict(color="white", width=1)),
                             text=["🤖"], textposition="middle center",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[EXIT[0]], y=[EXIT[1]], mode="text", text=["🏁"],
                             textfont=dict(size=20), hoverinfo="skip", showlegend=False))
    for e in enemies:
        fig.add_trace(go.Scatter(x=[e[0]], y=[e[1]], mode="markers",
                                 marker=dict(size=15, color="#e74c3c",
                                             line=dict(color="white", width=1.5)),
                                 hoverinfo="skip", showlegend=False))
    if agent is not None:
        fig.add_trace(go.Scatter(x=[agent[0]], y=[agent[1]], mode="markers",
                                 marker=dict(size=15, color="#c0392b" if dead else "#3498db",
                                             symbol="x" if dead else "circle",
                                             line=dict(color="white", width=1.5)),
                                 hoverinfo="skip", showlegend=False))

    fig.update_xaxes(range=[0, ARENA], constrain="domain", scaleanchor="y",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, ARENA], constrain="domain",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=0, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


# ───────────────────────── controls ─────────────────────────
def _env_controls():
    st.markdown("##### 🎮 Environment")
    n_enemies = st.radio("Number of enemies", [1, 2], index=0, horizontal=True,
        help="How many enemies chase you. Each adds two inputs to the network "
        "(its position relative to you), so 2 enemies → obs_dim 6. Two hunters are "
        "much harder to shake.")
    speed = st.slider("Enemy speed (× yours)", 0.50, 0.95, 0.75, 0.05,
        help="How fast each enemy chases, as a fraction of your speed. Measured "
        "sweet spot (one enemy): at 0.75 a good policy escapes ~95% while ignoring "
        "the enemy escapes only ~51%. Capped below 1.0 — at equal speed even good "
        "play escapes ~58%.")
    max_steps = st.select_slider("Max steps per episode", [40, 60, 80, 120], 60,
        help="Metres of travel before an episode times out (one decision = 1 m). "
        "A corner-to-corner run is ~14 steps.")
    random_enemies = st.checkbox("Randomize enemy positions each episode", value=True,
        help="You always start in the bottom-left corner. On (default): the enemies "
        "spawn somewhere new every episode, so the network must learn to read where "
        "they are and generalise. Off: the enemies sit at a fixed spot, making the "
        "whole episode deterministic — a warm-up where the net solves one layout.")
    return dict(enemy_speed=speed, max_steps=max_steps, n_enemies=n_enemies,
                random_enemies=random_enemies)


def _algo_row():
    st.markdown("#### 🧠 Deep Q-Network")
    st.caption("A neural net approximates Q(state, action) across the continuous "
               "arena. Trained with Double DQN + experience replay, adapted from "
               "`code examples/dql`.")
    c1, c2, c3, c4 = st.columns(4)
    n_episodes = c1.slider("Training episodes", 100, 1500, 800, 50,
        help="Games played while learning. More episodes → more experience, at a "
        "roughly linear time cost (~10–20 s here).")
    gamma = c2.slider("Discount γ", 0.80, 0.999, 0.99, 0.001,
        help="How much future reward counts. High γ makes reaching the far exit "
        "worth pursuing through many steps.")
    lr = c3.select_slider("Adam learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 3e-4,
        format_func=lambda v: f"{v:.0e}",
        help="Optimiser step size. 3e-4 is the stable default; higher rates can "
        "diverge on this env, lower rates learn slowly.")
    batch = c4.select_slider("Batch size", [32, 64, 128], 64,
        help="Transitions sampled from replay per gradient step.")

    c5, c6, c7 = st.columns(3)
    train_freq = c5.select_slider("Gradient step every N ticks", [1, 2, 4, 8], 4,
        help="How often to run a gradient update, in environment steps. Fewer "
        "updates per step is gentler and more stable here.")
    target_update = c6.select_slider("Target update (steps)", [250, 500, 1000, 2000], 1000,
        help="How often the slow target network is copied from the online net. "
        "Larger = more stable targets.")
    buffer = c7.select_slider("Replay buffer", [5_000, 10_000, 50_000, 100_000], 50_000,
        format_func=lambda v: f"{v//1000}k",
        help="Capacity of the experience-replay memory the net samples from.")

    st.markdown("###### Exploration ε")
    e1, e2 = st.columns([1, 3])
    eps_kind = e1.radio("Schedule", [DECAYING, CONSTANT], index=0,
        help="Decaying starts random and settles toward greedy (the app-wide "
        "default). Constant holds one exploration rate throughout.")
    if eps_kind == DECAYING:
        d1, d2, d3 = e2.columns(3)
        eps_params = (
            d1.slider("ε start", 0.1, 1.0, 1.0, 0.05, help="Exploration at episode 0."),
            d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01, help="Floor ε decays to."),
            d3.slider("ε decay", 0.980, 0.9999, 0.995, 0.0005, format="%.4f",
                      help="Per-episode multiplier: εₖ = max(min, start·decay^k)."),
        )
    else:
        eps_params = (e2.slider("ε (constant)", 0.0, 1.0, 0.10, 0.05,
                                help="Fixed exploration rate every episode."),)

    train = st.button("🚀 Train", type="primary", use_container_width=True,
        help="Train the deep Q-network on this configuration (a few seconds).")
    algo = dict(n_episodes=n_episodes, gamma=gamma, lr=lr, batch=batch,
                train_freq=train_freq, target_update=target_update, buffer=buffer,
                eps_kind=eps_kind, eps_params=eps_params)
    return algo, train


# ───────────────────────── render ─────────────────────────
def render():
    st.markdown("### Room 5 · Deep Q-Learning")
    st.caption("Cross the open arena to the exit while one enemy hunts you — touch "
               "it and the episode ends at −100.")
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown(
            "The arena is **continuous**, so there is no table of states to fill in — "
            "a **neural network** approximates the value of *every* point, including "
            "the ones between where it has been. Each enemy **chases** you with pure "
            "pursuit (it always heads at where you are *now*), so it can be **baited**: "
            "arc around it, get it behind you, and — being slower — it can no longer "
            "catch you before the exit.\n\n"
            "**Try this:** train, then tick *Show the network's value field* to see the "
            "learned landscape (blue = high value). Drag the **episode scrubber** to "
            "watch the policy improve, and **▶️ Play** to run a fresh chase. Ignoring "
            "the enemy and beelining escapes only ~half the time; a policy that reads "
            "the enemy escapes ~95%. Add a **second enemy** (each one adds two inputs to "
            "the net), or untick *Randomize enemy positions* for a fixed, deterministic "
            "warm-up layout.")

    # ── Row 1 — setup board + environment controls ──
    board_col, env_col = st.columns([3, 2])
    with env_col:
        env = _env_controls()
    env_kwargs = dict(enemy_speed=env["enemy_speed"], max_steps=env["max_steps"],
                      n_enemies=env["n_enemies"], random_enemies=env["random_enemies"])
    obs_dim = 2 + 2 * env["n_enemies"]
    with board_col:
        board = st.empty()
        st.caption(LEGEND)
        # preview: a representative spawn (fixed, no training yet)
        prev_env = ChaseArena(**env_kwargs)
        _, prev_info = prev_env.reset(seed=_EVAL_SEED)
        board.plotly_chart(_arena_figure(prev_info["enemies"], agent=prev_info["agent"]),
                           use_container_width=True, key="room5_preview")

    # ── Row 2 — algorithm ──
    st.divider()
    algo, train = _algo_row()

    # ── Train gate ──
    sig = (env["enemy_speed"], env["max_steps"], env["n_enemies"], env["random_enemies"],
           algo["n_episodes"], algo["gamma"], algo["lr"], algo["batch"],
           algo["train_freq"], algo["target_update"], algo["buffer"],
           algo["eps_kind"], algo["eps_params"])
    if train:
        prog = st.progress(0.0, text="Training the deep Q-network…")

        def _cb(done, total):
            if done % 20 == 0 or done == total:
                prog.progress(done / total, text=f"Training… episode {done:,}/{total:,}")

        bundle = dqn_control(
            lambda: ChaseArena(shaping_coef=5.0, **env_kwargs),
            obs_dim=obs_dim,
            n_episodes=algo["n_episodes"], gamma=algo["gamma"], lr=algo["lr"],
            batch_size=algo["batch"], buffer_size=algo["buffer"],
            target_update=algo["target_update"], train_freq=algo["train_freq"],
            eps_kind=algo["eps_kind"], eps_params=algo["eps_params"],
            reward_scale=0.01, double=True, seed=np.random.randint(1_000_000),
            progress_cb=_cb)
        prog.empty()
        st.session_state["room5_bundle"] = bundle
        st.session_state["room5_trained_sig"] = sig

    if st.session_state.get("room5_trained_sig") != sig:
        return
    bundle = st.session_state["room5_bundle"]
    stats = bundle["stats"]

    # ── Row 3 — training results ──
    st.divider()
    st.markdown("#### Training results")

    esc, caught, timeout = stats["escaped"], stats["caught"], stats["timeout"]
    last = slice(-100, None)
    escape_rate = 100 * esc[last].mean()
    steps_ok = stats["steps"][esc]
    mean_steps = steps_ok.mean() if steps_ok.size else float("nan")
    mean_q = stats["q_pred"][-200:].mean() if stats["q_pred"].size else float("nan")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Escape rate (last 100)", f"{escape_rate:.0f}%",
              help="Share of the most recent training episodes that reached the exit.")
    k2.metric("🔴 Caught (training)", f"{int(caught.sum()):,}",
              help="Training episodes ended by the enemy.")
    k3.metric("Mean steps to exit", f"{mean_steps:.1f}" if steps_ok.size else "—",
              help="Average metres travelled on successful escapes (1 step = 1 m).")
    k4.metric("Mean predicted Q", f"{mean_q:.2f}" if stats["q_pred"].size else "—",
              help="The network's own late-training value estimate (scaled units; "
              "no exact answer exists to check it against in a continuous room).")

    # view controls: checkpoint scrubber + value-field toggle
    cps = bundle["checkpoints"]
    vc1, vc2 = st.columns([3, 2])
    cp_i = vc1.slider("View episode (checkpoint)", 1, len(cps), len(cps),
        help="Replay the greedy policy — and its value field — as they stood at "
        "this point in training. Slide left to watch it learn.")
    show_field = vc2.checkbox("Show the network's value field", value=True,
        help="Sample max_a Q over a 50×50 grid, holding the enemies at their shown "
        "positions. It is a 2-D slice of the full value function.")
    cp = cps[cp_i - 1]
    net = load_net(cp["net_state"], bundle["hidden"], obs_dim=obs_dim)

    # a stable greedy rollout at this checkpoint (fixed spawn, so scrubbing is steady)
    view_env = ChaseArena(**env_kwargs)
    roll = greedy_rollout(net, view_env, seed=_EVAL_SEED)
    spawn = roll["frames"][0]
    field = q_field(net, spawn["enemies"]) if show_field else None

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play** — greedy (ε = 0), fresh random spawn")
        speed_sel = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"],
            "Normal", help="Playback speed of the animated episode.")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run the viewed policy with exploration off across a fresh chase — "
            "the enemy spawns somewhere new each time.")
        episode_slot = st.container()

    results_caption.caption(
        f"Greedy policy after **{cp['episode']:,}** episodes "
        f"(checkpoint {cp_i} of {len(cps)}). "
        + ("Value field sliced at the shown enemy position(s)." if show_field else ""))

    # Play is EPHEMERAL — nothing to session state (UI_STRUCTURE).
    if play:
        play_env = ChaseArena(**env_kwargs)
        pr = greedy_rollout(net, play_env)          # unseeded → new spawn each press
        frames = pr["frames"]
        for k in range(len(frames)):
            dead_here = k == len(frames) - 1 and pr["outcome"] == "caught"
            trail = [(f["agent"][0], f["agent"][1]) for f in frames[: k + 1]]
            results_board.plotly_chart(
                _arena_figure(frames[k]["enemies"], agent=frames[k]["agent"],
                              path=trail, dead=dead_here),
                use_container_width=True, key=f"room5_play_{k}")
            time.sleep(_STEP_DELAY[speed_sel])
        # Scoreboard: a WIN shows its real return; ANY loss (caught or timed out)
        # shows a flat −100, mirroring the +100 exit (Rooms 2–4 convention). This is
        # the displayed number only — the learner never sees it (measured: a timeout
        # penalty in the learning signal makes Room 5 time out MORE, not less).
        won = pr["outcome"] == "escaped"
        score = pr["return"] if won else LOSS_SCORE
        with episode_slot:
            if won:
                st.success("🏁 Escaped! Reached the exit.")
            elif pr["outcome"] == "caught":
                st.error("🔴 Caught by an enemy — the run ends here.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return", f"{score:+.1f}",
                help="On a WIN, the real undiscounted return (+100 exit plus shaping). "
                f"On ANY loss — caught or timed out — a flat {LOSS_SCORE:+.0f}, mirroring "
                "the +100 exit. One stochastic sample.")
            e2.metric("Steps", pr["steps"], help="Metres travelled before the run ended.")
            e3.metric("Result", "✅" if won else "❌",
                help="Whether the agent reached the exit.")
            if not won:
                st.caption(f"Every loss scores a flat {LOSS_SCORE:+.0f}; the raw return "
                           f"this run was {pr['return']:+.1f}.")
    else:
        trail = [(f["agent"][0], f["agent"][1]) for f in roll["frames"]]
        results_board.plotly_chart(
            _arena_figure(spawn["enemies"], agent=spawn["agent"], field=field, path=trail),
            use_container_width=True, key="room5_results")

    # ── Graphs ──
    st.divider()
    _graphs(stats)


def _graphs(stats):
    ep = np.arange(1, len(stats["returns"]) + 1)
    outcome = np.array(stats["outcome"])

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("###### Episode return")
        colors = np.where(stats["escaped"], "#2ecc71",
                          np.where(stats["caught"], "#e74c3c", "#f39c12"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=stats["returns"], mode="markers",
                                 marker=dict(size=4, color=colors, opacity=0.5),
                                 name="episode", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ep, y=moving_average(stats["returns"], 50),
                                 mode="lines", line=dict(color="#2c3e50", width=2),
                                 name="50-ep average"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="return")
        st.plotly_chart(fig, use_container_width=True, key="room5_returns")
        st.caption("🟢 escaped · 🔴 caught · 🟠 timed out. The average climbs as the "
                   "policy learns to reach the exit.")

    with g2:
        st.markdown("###### Network training")
        if stats["loss"].size:
            gs = np.arange(1, len(stats["loss"]) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gs, y=stats["loss"], mode="lines",
                                     line=dict(color="#e67e22", width=1), name="TD loss"))
            fig.add_trace(go.Scatter(x=gs, y=stats["q_pred"], mode="lines",
                                     line=dict(color="#8e44ad", width=1), name="mean Q",
                                     yaxis="y2"))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                              xaxis_title="gradient step",
                              yaxis=dict(title="Huber loss"),
                              yaxis2=dict(title="mean Q", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, key="room5_nettrain")
        st.caption("TD (Huber) loss and the network's mean predicted Q per gradient step.")

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("###### Cumulative outcomes")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["escaped"]), mode="lines",
                                 line=dict(color="#2ecc71"), name="escaped"))
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["caught"]), mode="lines",
                                 line=dict(color="#e74c3c"), name="caught"))
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["timeout"]), mode="lines",
                                 line=dict(color="#f39c12"), name="timed out"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="cumulative",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True, key="room5_cumulative")
    with g4:
        st.markdown("###### Exploration rate ε")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=stats["eps"], mode="lines",
                                 line=dict(color="#16a085"), name="ε"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="ε")
        st.plotly_chart(fig, use_container_width=True, key="room5_eps")
        st.caption("Exploration in force per episode; read against the return curve.")
