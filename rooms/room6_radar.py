"""
Room 6 · Advanced DQL — Dynamic Obstacles & Light-based Vision.

The step up from Room 5: momentum is back (actions set ACCELERATION under per-second
friction), and the agent is PARTIALLY OBSERVED — it never sees obstacle coordinates,
only its LIGHT: a lamp of radius `X` that reveals every obstacle within it, resolved
into `K` direction sectors (nearest obstacle distance per sector, else `X`). Obstacle
layouts are re-sampled EVERY episode (with a fixed pillar always in the middle), so the
headline metric is GENERALISATION: escape rate on held-out layouts training never saw.

Reuses `algorithms/deep_q.py` unchanged (Double DQN + reward-scaling + dense
straight-line shaping — Room 5's measured recipe). No value-field heatmap here: value
depends on the light readings (the layout), so a 2-D max_a Q(x,y) slice would lie. The
honest visuals are the live light beams and the trajectory heatmap.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.radar_arena import (
    RadarArena, ARENA, START, EXIT, GOAL_HALF, OBSTACLE_RADIUS,
    greedy_rollout, evaluate_policy,
)
from algorithms.deep_q import dqn_control, load_net
from algorithms.monte_carlo import CONSTANT, DECAYING, moving_average
from core.episode import LOSS_SCORE

LEGEND = ("🤖 Start · 🏁 Exit · ⬤ Obstacles (fatal on contact; one always in the "
          "middle) · the light circle is the vision radius X, and a beam turns 🔴 "
          "toward any obstacle it reveals")

_STEP_DELAY = {"Slow": 0.16, "Normal": 0.08, "Fast": 0.03}
_EVAL_SEED = 4242            # a fixed held-out layout for the steady results board
_GEN_SEED0 = 10_000_000     # generalisation eval seeds — a stream disjoint from training's
_GEN_N = 150               # held-out layouts scored for the Generalisation Score


# ───────────────────────── board figure ─────────────────────────
def _board_figure(obstacles, agent=None, vel=None, light=None, sector_dirs=None,
                  light_radius=3.0, path=None, dead=False, heat=None, countdown=None):
    fig = go.Figure()

    # arena floor tint (below traces so it shows through)
    fig.add_shape(type="rect", x0=0, y0=0, x1=ARENA, y1=ARENA,
                  fillcolor="rgba(99,120,160,0.10)", line=dict(width=0), layer="below")

    # trajectory heatmap (aggregated rollout paths), if provided
    if heat is not None:
        xs, ys, Z = heat
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=Z, colorscale="Turbo", opacity=0.55,
                                 showscale=False, hoverinfo="skip", zsmooth="best"))

    # the lit disc — the agent sees every obstacle inside this radius
    if agent is not None and light is not None:
        fig.add_shape(type="circle", x0=agent[0]-light_radius, y0=agent[1]-light_radius,
                      x1=agent[0]+light_radius, y1=agent[1]+light_radius,
                      fillcolor="rgba(255,241,150,0.10)",
                      line=dict(color="rgba(255,225,120,0.5)", width=1, dash="dot"),
                      layer="above")

    # goal square
    fig.add_shape(type="rect", x0=EXIT[0]-GOAL_HALF, y0=EXIT[1]-GOAL_HALF,
                  x1=EXIT[0]+GOAL_HALF, y1=EXIT[1]+GOAL_HALF,
                  fillcolor="rgba(38,166,91,0.55)", line=dict(color="white", width=2),
                  layer="above")

    # obstacles (fatal discs)
    for o in np.atleast_2d(obstacles) if len(obstacles) else []:
        fig.add_shape(type="circle", x0=o[0]-OBSTACLE_RADIUS, y0=o[1]-OBSTACLE_RADIUS,
                      x1=o[0]+OBSTACLE_RADIUS, y1=o[1]+OBSTACLE_RADIUS,
                      fillcolor="rgba(120,130,145,0.85)",
                      line=dict(color="#e5e9f0", width=1), layer="above")

    # live light beams from the agent (one per sector; red toward a revealed obstacle)
    if agent is not None and light is not None and sector_dirs is not None:
        for k, u in enumerate(sector_dirs):
            hit = light[k] < light_radius - 1e-6
            reach = light[k] if hit else light_radius
            x1, y1 = agent[0] + u[0] * reach, agent[1] + u[1] * reach
            fig.add_trace(go.Scatter(
                x=[agent[0], x1], y=[agent[1], y1], mode="lines",
                line=dict(color=("rgba(231,76,60,0.95)" if hit else "rgba(255,235,150,0.30)"),
                          width=(2.5 if hit else 1)),
                hoverinfo="skip", showlegend=False))

    # trajectory
    if path is not None and len(path) > 1:
        px, py = zip(*path)
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines",
                                 line=dict(color="#f1c40f", width=2.5),
                                 hoverinfo="skip", showlegend=False))

    # start, exit, agent
    fig.add_trace(go.Scatter(x=[START[0]], y=[START[1]], mode="markers+text",
                             marker=dict(size=16, color="#2ecc71", symbol="square",
                                         line=dict(color="white", width=1)),
                             text=["🤖"], textposition="middle center",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[EXIT[0]], y=[EXIT[1]], mode="text", text=["🏁"],
                             textfont=dict(size=20), hoverinfo="skip", showlegend=False))
    if agent is not None:
        fig.add_trace(go.Scatter(x=[agent[0]], y=[agent[1]], mode="markers",
                                 marker=dict(size=15, color="#c0392b" if dead else "#3498db",
                                             symbol="x" if dead else "circle",
                                             line=dict(color="white", width=1.5)),
                                 hoverinfo="skip", showlegend=False))

    # border frame above everything
    fig.add_shape(type="rect", x0=0, y0=0, x1=ARENA, y1=ARENA,
                  fillcolor="rgba(0,0,0,0)", line=dict(color="#64748b", width=3),
                  layer="above")
    if countdown is not None:
        fig.add_annotation(x=ARENA/2, y=ARENA/2, text=str(countdown), showarrow=False,
                           font=dict(size=96, color="#f8fafc"),
                           bgcolor="rgba(15,23,42,0.55)", borderpad=18)

    fig.update_xaxes(range=[0, ARENA], constrain="domain", scaleanchor="y",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, ARENA], constrain="domain",
                     showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=460, margin=dict(l=2, r=2, t=2, b=2),
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def _trajectory_heat(paths, res=40):
    """2-D histogram of agent positions over many rollout paths → a heatmap grid."""
    if not paths:
        return None
    pts = np.concatenate([p for p in paths if len(p)], axis=0)
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=res, range=[[0, ARENA], [0, ARENA]])
    xs = 0.5 * (xe[:-1] + xe[1:])
    ys = 0.5 * (ye[:-1] + ye[1:])
    Z = H.T                                        # (y, x) for plotly
    Z = np.where(Z > 0, np.log1p(Z), np.nan)       # log scale; blank empty cells
    return xs, ys, Z


# ───────────────────────── controls ─────────────────────────
def _env_controls():
    st.markdown("##### 🎮 Perception & Setup")
    st.caption("The agent sees only its light — every obstacle within radius X, not "
               "their coordinates. Layouts regenerate every episode (one obstacle always "
               "in the middle), so it must learn to *react*, not memorise.")
    c1, c2 = st.columns(2)
    light_radius = c1.slider("⭐ Light radius X (m)", 1.0, 10.0, 3.0, 0.5,
        help="How far the lamp reaches. It reveals EVERY obstacle within X (a true light, "
             "not thin rays). Distance is CENTER-TO-CENTER, so usable clearance is X − 0.5 m: "
             "'lit at X' is not 'collides at X'. More radius = more warning.")
    n_sectors = c2.select_slider("Light directions K", [4, 8, 16, 32], 8,
        help="The lit disc is split into K direction sectors, each reporting its nearest "
             "obstacle. More sectors = finer bearing resolution (the biggest lever on "
             "avoidance) but a larger observation to learn.")
    c3, c4 = st.columns(2)
    n_obstacles = c3.slider("Obstacle count", 2, 12, 6, 1,
        help="Discs (0.5 m each) — one is always the central pillar, the rest placed "
             "randomly every episode, always leaving a traversable path.")
    obstacle_speed = c4.slider("Dynamic-obstacle speed (m/s)", 0.0, 1.5, 0.0, 0.1,
        help="0 = static (default). Above 0, the non-central obstacles drift and bounce "
             "(the central pillar stays put) — the observation switches on a per-sector "
             "range-rate channel so the network can tell approaching from receding.")
    friction = st.slider("Ice friction μ (per second)", 0.2, 0.9, 0.5, 0.05,
        help="Velocity retained per second (momentum carries between steps). Lower = "
             "more drag/grip; higher = more slide. Thrust is tuned to beat friction so "
             "the room stays winnable.")
    random_layout = st.checkbox("Randomize layout each episode (training)", value=True,
        help="Re-sample obstacles every training episode to force generalisation. Untick "
             "for one fixed layout — a deterministic warm-up the net can overfit.")
    return dict(light_radius=light_radius, n_sectors=n_sectors, n_obstacles=n_obstacles,
                obstacle_speed=obstacle_speed, friction=friction,
                random_layout=random_layout)


def _algo_row():
    st.markdown("#### 🧠 Deep Q-Network")
    st.caption("Same recipe as Room 5 — Double DQN + experience replay + reward "
               "scaling — now over a partially-observed light state.")
    c1, c2, c3, c4 = st.columns(4)
    n_episodes = c1.slider("Training episodes", 500, 5000, 1200, 100,
        help="A larger, partially-observed input needs more episodes than Room 5. "
             "Training shows a live progress bar (~80 s; ~89% escape on unseen layouts "
             "by 1,200 at the defaults).")
    gamma = c2.slider("Discount γ", 0.50, 0.99, 0.95, 0.01,
        help="Weight on future reward. The exit is ~13 m away, so keep γ fairly high.")
    lr = c3.select_slider("Adam learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 3e-4,
        format_func=lambda v: f"{v:.0e}",
        help="Optimizer step size. 3e-4 is the stable Room 5 default.")
    batch = c4.select_slider("Batch size", [32, 64, 128], 64)

    c5, c6, c7 = st.columns(3)
    train_freq = c5.select_slider("Gradient step every N ticks", [1, 2, 4, 8], 4,
        help="Network update cadence in environment steps. Higher = more stable/faster.")
    target_update = c6.select_slider("Target update (steps)", [250, 500, 1000, 2000], 1000,
        help="How often to copy weights to the target network. Larger = more stable.")
    buffer = c7.select_slider("Replay buffer", [5_000, 10_000, 50_000, 100_000], 50_000,
        format_func=lambda v: f"{v//1000}k", help="Stored transitions in replay memory.")

    st.markdown("###### Exploration ε")
    e1, e2 = st.columns([1, 3])
    eps_kind = e1.radio("Schedule", [DECAYING, CONSTANT], index=0,
        help="Decaying shifts from exploration to exploitation over training.")
    if eps_kind == DECAYING:
        d1, d2, d3 = e2.columns(3)
        eps_params = (
            d1.slider("ε start", 0.1, 1.0, 1.0, 0.05, help="Exploration rate at episode 0."),
            d2.slider("ε minimum", 0.0, 0.5, 0.05, 0.01, help="Lower bound on ε."),
            d3.slider("ε decay", 0.980, 0.9999, 0.995, 0.0005, format="%.4f",
                      help="Per-episode multiplier: ε = max(min, start × decay^k)."),
        )
    else:
        eps_params = (e2.slider("ε (constant)", 0.0, 1.0, 0.10, 0.05,
                                help="Fixed exploration rate across all episodes."),)

    train = st.button("🚀 Train", type="primary", use_container_width=True)
    algo = dict(n_episodes=n_episodes, gamma=gamma, lr=lr, batch=batch,
                train_freq=train_freq, target_update=target_update, buffer=buffer,
                eps_kind=eps_kind, eps_params=eps_params)
    return algo, train


# ───────────────────────── render ─────────────────────────
def render():
    st.markdown("### Room 6 · Advanced DQL — Light & Obstacles")
    st.caption("Cross a chamber full of obstacles you can only sense through your light. "
               "Layouts change every episode, so the agent must generalise, not memorise.")
    with st.expander("ℹ️ About this room", expanded=False):
        st.markdown(
            "This is the hardest room: **momentum** is back (actions are *accelerations* "
            "under ice friction) and the agent is **partially observed** — it never sees "
            "where the obstacles are, only its **light**.\n\n"
            "* **The light sees everything within radius X.** The lit disc is split into "
            "`K` direction sectors, each reporting the nearest obstacle in it — so nothing "
            "inside the radius is invisible (unlike thin rays). One obstacle is **always in "
            "the middle** of the room.\n"
            "* **Generalisation is the point.** Obstacles are re-randomised every episode. "
            "The headline **Generalisation Score** is the escape rate on *held-out* layouts "
            "the network never trained on, so a high score means learned avoidance, not a "
            "memorised path.\n"
            "* **Light is center-to-center.** A reading is the distance to an obstacle's "
            "*centre*, capped at X — so it reveals sooner than it collides.\n"
            "* **Collision vs. timeout.** Hitting an obstacle scores −100 and *is* a real "
            "learning signal. Running out of steps also shows −100 but is **not** penalised "
            "in training — penalising an arbitrary clock poisons spatial values (Rooms 3, 5).\n"
            "* **No value-field heatmap** here: value depends on the light (the layout), so "
            "a flat Q(x, y) slice would mislead. Watch the **live light** and the "
            "**trajectory heatmap** instead."
        )

    # ── Row 1 — setup board + environment controls ──
    board_col, env_col = st.columns([3, 2])
    with env_col:
        env = _env_controls()
    with board_col:
        board = st.empty()
        st.caption(LEGEND)
        prev_env = RadarArena(n_obstacles=env["n_obstacles"], n_sectors=env["n_sectors"],
                              light_radius=env["light_radius"], friction=env["friction"],
                              obstacle_speed=env["obstacle_speed"])
        _, prev_info = prev_env.reset(seed=_EVAL_SEED)
        board.plotly_chart(
            _board_figure(prev_info["obstacles"], agent=prev_info["agent"],
                          light=prev_info["light"], sector_dirs=prev_env.sector_dirs,
                          light_radius=env["light_radius"]),
            use_container_width=True, key="room6_preview")

    obs_dim = prev_env.obs_dim

    def make_train_env():
        return RadarArena(n_obstacles=env["n_obstacles"], n_sectors=env["n_sectors"],
                          light_radius=env["light_radius"], friction=env["friction"],
                          obstacle_speed=env["obstacle_speed"], max_steps=150,
                          shaping_coef=5.0, random_layout=env["random_layout"])

    # ── Row 2 — algorithm ──
    st.divider()
    algo, train = _algo_row()

    sig = (env["light_radius"], env["n_sectors"], env["n_obstacles"], env["obstacle_speed"],
           env["friction"], env["random_layout"], algo["n_episodes"], algo["gamma"],
           algo["lr"], algo["batch"], algo["train_freq"], algo["target_update"],
           algo["buffer"], algo["eps_kind"], algo["eps_params"])

    if train:
        prog = st.progress(0.0, text="Training the deep Q-network…")

        def _cb(done, total):
            if done % 20 == 0 or done == total:
                prog.progress(done / total, text=f"Training… episode {done:,}/{total:,}")

        bundle = dqn_control(
            make_train_env, obs_dim=obs_dim,
            n_episodes=algo["n_episodes"], gamma=algo["gamma"], lr=algo["lr"],
            batch_size=algo["batch"], buffer_size=algo["buffer"],
            target_update=algo["target_update"], train_freq=algo["train_freq"],
            eps_kind=algo["eps_kind"], eps_params=algo["eps_params"],
            reward_scale=0.01, double=True, seed=np.random.randint(1_000_000),
            progress_cb=_cb)

        # Held-out Generalisation Score — DISJOINT seed stream from training.
        prog.progress(1.0, text="Scoring generalisation on unseen layouts…")
        net = load_net(bundle["net_state"], bundle["hidden"], obs_dim=obs_dim)
        gen_esc, gen_coll, gen_to, _ = evaluate_policy(net, make_train_env,
                                                       n_eval=_GEN_N, seed0=_GEN_SEED0)
        bundle["gen"] = dict(escape=gen_esc, collide=gen_coll, timeout=gen_to)
        prog.empty()
        st.session_state["room6_bundle"] = bundle
        st.session_state["room6_trained_sig"] = sig

    if st.session_state.get("room6_trained_sig") != sig:
        return
    bundle = st.session_state["room6_bundle"]
    stats = bundle["stats"]
    gen = bundle["gen"]

    # ── Row 3 — training results ──
    st.divider()
    st.markdown("#### Training results")

    esc, coll, to = stats["escaped"], stats["caught"], stats["timeout"]
    escape_rate = 100 * esc[-100:].mean()
    coll_rate = 100 * coll.mean()
    to_rate = 100 * to.mean()
    mean_q = stats["q_pred"][-200:].mean() if stats["q_pred"].size else float("nan")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Escape rate (last 100)", f"{escape_rate:.0f}%",
              help="Share of the last 100 TRAINING episodes that reached the exit.")
    k2.metric("🎲 Generalisation score", f"{100*gen['escape']:.0f}%",
              help=f"Escape rate on {_GEN_N} HELD-OUT layouts the network never trained on. "
                   "This is the headline — learned avoidance, not memorisation.")
    k3.metric("💥 Collision rate", f"{coll_rate:.0f}%",
              help="Share of training episodes that ended by hitting an obstacle.")
    k4.metric("⏱️ Timed out", f"{to_rate:.0f}%",
              help="Share of training episodes that ran out of steps. Shown as −100 for "
                   "display, but unpenalized during training.")
    k5.metric("Mean predicted Q", f"{mean_q:.2f}" if stats["q_pred"].size else "—",
              help="The network's average value estimate over the last 200 training steps.")

    # view controls
    cps = bundle["checkpoints"]
    vc1, vc2 = st.columns([3, 2])
    cp_i = vc1.slider("View episode (checkpoint)", 1, len(cps), len(cps),
        help="Scrub through training checkpoints to watch the policy improve.")
    show_heat = vc2.checkbox("Show trajectory heatmap", value=True,
        help="Overlay where the greedy policy tends to travel, aggregated over many "
             "held-out layouts — the paths it has learned to prefer.")
    cp = cps[cp_i - 1]
    net = load_net(cp["net_state"], bundle["hidden"], obs_dim=obs_dim)

    # steady results board: greedy rollout on the fixed held-out layout
    view_env = RadarArena(n_obstacles=env["n_obstacles"], n_sectors=env["n_sectors"],
                          light_radius=env["light_radius"], friction=env["friction"],
                          obstacle_speed=env["obstacle_speed"], max_steps=150)
    roll = greedy_rollout(net, view_env, seed=_EVAL_SEED)

    heat = None
    if show_heat:
        # Cache per checkpoint — scrubbing/toggling other controls shouldn't re-run
        # 50 rollouts every time (they only depend on this checkpoint's weights).
        cache = bundle.setdefault("heat_cache", {})
        if cp_i not in cache:
            _, _, _, paths = evaluate_policy(
                load_net(cp["net_state"], bundle["hidden"], obs_dim=obs_dim),
                lambda: RadarArena(n_obstacles=env["n_obstacles"], n_sectors=env["n_sectors"],
                                   light_radius=env["light_radius"], friction=env["friction"],
                                   obstacle_speed=env["obstacle_speed"], max_steps=150),
                n_eval=50, seed0=_GEN_SEED0)
            cache[cp_i] = _trajectory_heat(paths)
        heat = cache[cp_i]

    res_board_col, res_ctrl_col = st.columns([3, 2])
    with res_board_col:
        results_board = st.empty()
        results_caption = st.empty()
    with res_ctrl_col:
        st.markdown("**▶️ Play**")
        play_count = st.slider("Obstacles this run", 2, 12, env["n_obstacles"], 1,
            help="Choose how many obstacles for THIS play episode — independent of "
                 "training. The light sensor is a fixed size, so the trained policy "
                 "handles any count (one is always the central pillar).")
        speed_sel = st.select_slider("Animation speed", ["Slow", "Normal", "Fast"], "Normal")
        play = st.button("▶️ Play Episode", type="primary", use_container_width=True,
            help="Run one greedy (ε = 0) episode on a fresh, unseen random layout.")
        gen_room = st.button("🎲 Generate Random Room", use_container_width=True,
            help="Roll the frozen policy on a brand-new layout and re-score generalisation.")
        episode_slot = st.container()

    results_caption.caption(
        f"Greedy policy at episode **{cp['episode']:,}** ({cp_i}/{len(cps)}) on the "
        "fixed reference layout." + (" Heatmap aggregates 50 held-out rollouts." if show_heat else ""))

    if play or gen_room:
        # Both play a single episode on a FRESH unseen layout (ephemeral), with the
        # user-chosen obstacle count for this run.
        play_env = RadarArena(n_obstacles=play_count, n_sectors=env["n_sectors"],
                              light_radius=env["light_radius"], friction=env["friction"],
                              obstacle_speed=env["obstacle_speed"], max_steps=150)
        pr = greedy_rollout(net, play_env, seed=np.random.randint(_GEN_SEED0, 2**31 - 1))
        frames = pr["frames"]
        sd = play_env.sector_dirs
        for c in (3, 2, 1):
            results_board.plotly_chart(
                _board_figure(frames[0]["obstacles"], agent=frames[0]["agent"],
                              light=frames[0]["light"], sector_dirs=sd,
                              light_radius=env["light_radius"], heat=heat, countdown=c),
                use_container_width=True, key=f"room6_cd_{c}")
            time.sleep(0.6)
        for kf in range(len(frames)):
            dead_here = kf == len(frames) - 1 and pr["outcome"] == "collided"
            trail = [(f["agent"][0], f["agent"][1]) for f in frames[: kf + 1]]
            results_board.plotly_chart(
                _board_figure(frames[kf]["obstacles"], agent=frames[kf]["agent"],
                              light=frames[kf]["light"], sector_dirs=sd,
                              light_radius=env["light_radius"], path=trail,
                              dead=dead_here, heat=heat),
                use_container_width=True, key=f"room6_play_{kf}")
            time.sleep(_STEP_DELAY[speed_sel])

        won = pr["outcome"] == "escaped"
        score = pr["return"] if won else LOSS_SCORE
        with episode_slot:
            if won:
                st.success("🏁 Escaped the chamber!")
            elif pr["outcome"] == "collided":
                st.error("💥 Hit an obstacle — the run ends here.")
            else:
                st.warning("⏱️ Timed out before reaching the exit.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Return", f"{score:+.1f}",
                help="Undiscounted return (+100 for exit plus shaping; flat −100 for a loss).")
            e2.metric("Steps", pr["steps"])
            e3.metric("Result", "✅" if won else "❌")
            if not won:
                st.caption(f"Display score floored at {LOSS_SCORE:+.0f} (raw: {pr['return']:+.1f}).")
    else:
        trail = [(f["agent"][0], f["agent"][1]) for f in roll["frames"]]
        last = roll["frames"][-1]        # final agent + obstacle positions (they match)
        dead0 = roll["outcome"] == "collided"
        results_board.plotly_chart(
            _board_figure(last["obstacles"], agent=last["agent"], light=last["light"],
                          sector_dirs=view_env.sector_dirs, light_radius=env["light_radius"],
                          path=trail, dead=dead0, heat=heat),
            use_container_width=True, key="room6_results")

    # ── Graphs ──
    st.divider()
    _graphs(stats)


def _graphs(stats):
    ep = np.arange(1, len(stats["returns"]) + 1)
    scored = np.where(stats["escaped"], stats["returns"], LOSS_SCORE)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("###### Episode return (scored)")
        colors = np.where(stats["escaped"], "#2ecc71",
                          np.where(stats["caught"], "#e74c3c", "#f39c12"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=scored, mode="markers",
                                 marker=dict(size=4, color=colors, opacity=0.5),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ep, y=moving_average(scored, 50), mode="lines",
                                 line=dict(color="#2c3e50", width=2), name="50-ep average"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="scored return")
        st.plotly_chart(fig, use_container_width=True, key="room6_returns")
        st.caption("🟢 Escaped · 🔴 Collided · 🟠 Timed out. Losses display as −100 "
                   "(display floor only; unpenalized during training).")
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
                              xaxis_title="gradient step", yaxis=dict(title="Huber loss"),
                              yaxis2=dict(title="mean Q", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, key="room6_nettrain")
            st.caption("Temporal-difference loss and mean predicted Q per gradient step.")
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("###### Cumulative outcomes")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["escaped"]), mode="lines",
                                 line=dict(color="#2ecc71"), name="escaped"))
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["caught"]), mode="lines",
                                 line=dict(color="#e74c3c"), name="collided"))
        fig.add_trace(go.Scatter(x=ep, y=np.cumsum(stats["timeout"]), mode="lines",
                                 line=dict(color="#f39c12"), name="timed out"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="cumulative",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True, key="room6_cumulative")
    with g4:
        st.markdown("###### Exploration rate ε")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=stats["eps"], mode="lines",
                                 line=dict(color="#16a085"), name="ε"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="episode", yaxis_title="ε")
        st.plotly_chart(fig, use_container_width=True, key="room6_eps")
        st.caption("Exploration schedule over training.")
