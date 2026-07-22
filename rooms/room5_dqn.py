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
    ChaseArena, ARENA, START, EXIT, GOAL_HALF, CATCH_RADIUS,
    PURSUIT, FLANK, AMBUSH,
)
from algorithms.deep_q import dqn_control, load_net, q_field, greedy_rollout
from algorithms.monte_carlo import CONSTANT, DECAYING, epsilon_at, moving_average
from core.episode import LOSS_SCORE

LEGEND = ("🤖 start · 🏁 exit (1.2 m square) · 🔴 chaser (head-on) · 🟠 flanker (from a "
          "side) · 🟣 ambusher (sweeps in side-on) — all fatal on contact · value field: "
          "**blue = high**, red = low (RdBu, 0-centred)")

# Per-behaviour colours so the enemies read as different agents.
_KIND_MARKER = {PURSUIT: "#e74c3c", FLANK: "#e67e22", AMBUSH: "#9b59b6"}   # red / orange / purple
_KIND_FILL = {PURSUIT: "rgba(231,76,60,0.28)", FLANK: "rgba(230,126,34,0.28)",
              AMBUSH: "rgba(155,89,182,0.28)"}
_KIND_RING = {PURSUIT: "rgba(231,76,60,0.9)", FLANK: "rgba(230,126,34,0.9)",
              AMBUSH: "rgba(155,89,182,0.9)"}
_KIND_LABEL = {PURSUIT: "🔴 Chaser", FLANK: "🟠 Flanker", AMBUSH: "🟣 Ambusher"}

_STEP_DELAY = {"Slow": 0.16, "Normal": 0.08, "Fast": 0.03}
_EVAL_SEED = 4242            # fixed spawn for the scrubber/results, so it is stable


# ───────────────────────── board figure ─────────────────────────
def _arena_figure(enemies, agent=None, field=None, path=None, dead=False,
                  enemy_kinds=None):
    """Plotly figure of the arena. `enemies` is a single (x, y) or an (n, 2) array;
    `enemy_kinds` colours each by behaviour; `field` is an optional (xs, ys, Z)
    value slice drawn as an RdBu heatmap underneath."""
    fig = go.Figure()

    if enemies is None:
        enemies = []
    else:
        arr = np.asarray(enemies, dtype=float)
        enemies = [arr] if arr.ndim == 1 else list(arr)
    kinds = list(enemy_kinds) if enemy_kinds is not None else [PURSUIT] * len(enemies)

    # arena floor tint (below the value heatmap, so it shows through when the field
    # is off) — gives the board a defined surface even before training.
    fig.add_shape(type="rect", x0=0, y0=0, x1=ARENA, y1=ARENA,
                  fillcolor="rgba(99,120,160,0.10)", line=dict(width=0), layer="below")

    if field is not None:
        xs, ys, Z = field
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=Z, colorscale="RdBu", zmid=0.0,
            colorbar=dict(title="max Q", thickness=12, len=0.9),
            hoverinfo="skip"))

    # exit square + catch discs are shapes (true sizes in metres), layer above the
    # heatmap trace (layer="below" is below TRACES → painted over; see memory).
    fig.add_shape(type="rect", x0=EXIT[0]-GOAL_HALF, y0=EXIT[1]-GOAL_HALF,
                  x1=EXIT[0]+GOAL_HALF, y1=EXIT[1]+GOAL_HALF,
                  fillcolor="rgba(38,166,91,0.55)", line=dict(color="white", width=2),
                  layer="above")
    for e, k in zip(enemies, kinds):
        fig.add_shape(type="circle", x0=e[0]-CATCH_RADIUS, y0=e[1]-CATCH_RADIUS,
                      x1=e[0]+CATCH_RADIUS, y1=e[1]+CATCH_RADIUS,
                      fillcolor=_KIND_FILL[k], line=dict(color=_KIND_RING[k], width=1),
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
    for e, k in zip(enemies, kinds):
        fig.add_trace(go.Scatter(x=[e[0]], y=[e[1]], mode="markers",
                                 marker=dict(size=15, color=_KIND_MARKER[k],
                                             line=dict(color="white", width=1.5)),
                                 hoverinfo="skip", showlegend=False))
    if agent is not None:
        fig.add_trace(go.Scatter(x=[agent[0]], y=[agent[1]], mode="markers",
                                 marker=dict(size=15, color="#c0392b" if dead else "#3498db",
                                             symbol="x" if dead else "circle",
                                             line=dict(color="white", width=1.5)),
                                 hoverinfo="skip", showlegend=False))

    # border frame around the arena (above everything, so it reads over the heatmap)
    fig.add_shape(type="rect", x0=0, y0=0, x1=ARENA, y1=ARENA,
                  fillcolor="rgba(0,0,0,0)", line=dict(color="#64748b", width=3),
                  layer="above")

    fig.update_xaxes(range=[0, ARENA], constrain="domain", scaleanchor="y",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, ARENA], constrain="domain",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=460, margin=dict(l=2, r=2, t=2, b=2),
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


# ───────────────────────── controls ─────────────────────────
def _env_controls():
    st.markdown("##### 🎮 Environment")
    st.caption("Enemies — toggle each (0–3). Each adds two inputs to the network "
               "(its position relative to you), and they repel each other so they "
               "attack from different angles.")
    t1, t2, t3 = st.columns(3)
    on_chaser = t1.checkbox("🔴 Chaser", value=True,
        help="Pure pursuit — heads straight at where you are now. Baitable: arc "
        "around it and it ends up behind you.")
    on_flanker = t2.checkbox("🟠 Flanker", value=False,
        help="Pursues along a path rotated off the direct line, curving in from one "
        "side rather than trailing behind.")
    on_ambusher = t3.checkbox("🟣 Ambusher", value=False,
        help="Sweeps in almost side-on from the opposite side to the flanker — the "
        "most different path of the three.")
    kinds = tuple(k for k, on in [(PURSUIT, on_chaser), (FLANK, on_flanker),
                                  (AMBUSH, on_ambusher)] if on)

    speed = st.slider("Enemy speed (× yours)", 0.50, 0.95, 0.75, 0.05,
        help="How fast each enemy chases, as a fraction of your speed. Sweet spot for "
        "ONE enemy: at 0.75 a good policy escapes ~95% vs ~51% ignoring it. More "
        "enemies are much harder — drop the speed toward 0.50 when running 3.")
    max_steps = st.select_slider("Max steps per episode", [20, 40, 60, 80, 100], 60,
        help="Metres of travel before an episode times out (one decision = 1 m). "
        "A corner-to-corner run is ~14 steps.")
    random_enemies = st.checkbox("Randomize enemy positions each episode (training)",
        value=True,
        help="You always start in the bottom-left corner. On (default): enemies spawn "
        "somewhere new every training episode, so the network must generalise. Off: "
        "enemies sit at a fixed spot — a deterministic warm-up. (Play has its own "
        "toggle, so you can train on one and play the other.)")
    return dict(enemy_speed=speed, max_steps=max_steps, enemy_kinds=kinds,
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
    gamma = c2.slider("Discount γ", 0.50, 0.99, 0.99, 0.01,
        help="How much future reward counts. High γ makes reaching the far exit "
        "worth pursuing through many steps; low γ (toward 0.50) is short-sighted and "
        "can fail to value the distant exit at all.")
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
            "the enemy escapes ~95%. Toggle in a 🟠 *flanker* and 🟣 *ambusher* (up to "
            "three enemies, each with a different approach; they repel each other so "
            "they attack from different angles), or untick *Randomize enemy positions* "
            "for a fixed, deterministic warm-up layout. More enemies are much harder — "
            "drop the enemy speed when you run three.")

    # ── Row 1 — setup board + environment controls ──
    board_col, env_col = st.columns([3, 2])
    with env_col:
        env = _env_controls()
    kinds = env["enemy_kinds"]
    env_kwargs = dict(enemy_speed=env["enemy_speed"], max_steps=env["max_steps"],
                      enemy_kinds=kinds, random_enemies=env["random_enemies"])
    obs_dim = 2 + 2 * len(kinds)
    with board_col:
        board = st.empty()
        st.caption(LEGEND)
        # preview: a representative spawn (fixed, no training yet)
        prev_env = ChaseArena(**env_kwargs)
        _, prev_info = prev_env.reset(seed=_EVAL_SEED)
        board.plotly_chart(_arena_figure(prev_info["enemies"], agent=prev_info["agent"],
                                         enemy_kinds=kinds),
                           use_container_width=True, key="room5_preview")

    # ── Row 2 — algorithm ──
    st.divider()
    algo, train = _algo_row()

    # ── Train gate ──
    sig = (env["enemy_speed"], env["max_steps"], kinds, env["random_enemies"],
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

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Escape rate (last 100)", f"{escape_rate:.0f}%",
              help="Share of the most recent training episodes that reached the exit.")
    k2.metric("🔴 Caught (training)", f"{int(caught.sum()):,}",
              help="Training episodes ended by an enemy.")
    k3.metric("⏱️ Timed out (training)", f"{int(timeout.sum()):,}",
              help="Training episodes that ran out of steps without escaping — a loss. "
              "The −100 for a loss is shown on the scoreboard and the returns curve, but "
              "is deliberately NOT in the learning signal: a timeout depends on elapsed "
              "steps, which the network can't see, so penalising it there poisons "
              "positions rather than teaching (measured — it made timeouts worse).")
    k4.metric("Mean steps to exit", f"{mean_steps:.1f}" if steps_ok.size else "—",
              help="Average metres travelled on successful escapes (1 step = 1 m).")
    k5.metric("Mean predicted Q", f"{mean_q:.2f}" if stats["q_pred"].size else "—",
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
        st.markdown("**▶️ Play** — greedy (ε = 0)")
        play_random = st.checkbox("Randomize enemy positions", value=True,
            help="Independent of the training toggle, so you can train on a fixed "
            "layout and play random (does it generalise?), or train random and play "
            "the fixed layout, or match them. On: fresh random spawn each press.")
        speed_sel = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"],
            "Normal", help="Playback speed of the animated episode.")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run the viewed policy with exploration off for one episode — a fresh "
            "test of what it learned.")
        episode_slot = st.container()

    results_caption.caption(
        f"Greedy policy after **{cp['episode']:,}** episodes "
        f"(checkpoint {cp_i} of {len(cps)}). "
        + ("Value field sliced at the shown enemy position(s)." if show_field else ""))

    # Play is EPHEMERAL — nothing to session state (UI_STRUCTURE).
    if play:
        # Play's enemy randomisation is its own toggle, independent of training's.
        play_env = ChaseArena(**{**env_kwargs, "random_enemies": play_random})
        pr = greedy_rollout(net, play_env)          # unseeded → new spawn each press
        frames = pr["frames"]
        for k in range(len(frames)):
            dead_here = k == len(frames) - 1 and pr["outcome"] == "caught"
            trail = [(f["agent"][0], f["agent"][1]) for f in frames[: k + 1]]
            # keep the value field visible during play, recomputed at THIS frame's
            # enemy positions — so the landscape shifts with the threat as it moves.
            frame_field = q_field(net, frames[k]["enemies"]) if show_field else None
            results_board.plotly_chart(
                _arena_figure(frames[k]["enemies"], agent=frames[k]["agent"],
                              path=trail, dead=dead_here, enemy_kinds=kinds,
                              field=frame_field),
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
            _arena_figure(spawn["enemies"], agent=spawn["agent"], field=field, path=trail,
                          enemy_kinds=kinds),
            use_container_width=True, key="room5_results")

    # ── Graphs ──
    st.divider()
    _graphs(stats)


def _graphs(stats):
    ep = np.arange(1, len(stats["returns"]) + 1)
    outcome = np.array(stats["outcome"])

    # Scored view: any loss (caught OR timed out) floors to −100 in the DISPLAY,
    # mirroring the +100 exit and the Play scoreboard (Rooms 2–4 convention). Pure
    # display transform — the learner updates off per-step rewards, never this.
    scored = np.where(stats["escaped"], stats["returns"], LOSS_SCORE)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("###### Episode return (scored)")
        colors = np.where(stats["escaped"], "#2ecc71",
                          np.where(stats["caught"], "#e74c3c", "#f39c12"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=scored, mode="markers",
                                 marker=dict(size=4, color=colors, opacity=0.5),
                                 name="episode", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ep, y=moving_average(scored, 50),
                                 mode="lines", line=dict(color="#2c3e50", width=2),
                                 name="50-ep average"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="scored return")
        st.plotly_chart(fig, use_container_width=True, key="room5_returns")
        st.caption("🟢 escaped · 🔴 caught · 🟠 timed out. Every loss (caught **or** timed "
                   "out) scores a flat −100, mirroring the +100 exit — a display floor "
                   "only; the learner never sees it. The average climbs as escapes rise.")

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
