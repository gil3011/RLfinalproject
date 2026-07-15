# Room UI Structure — the shared contract

Every room in the RL Escape Room app follows the **same page structure** so the
experience is consistent from room to room. The *parameters* differ per room, and
some rooms add *extra graphs*, but the skeleton, flow, and conventions below are
fixed. Room 1 (`rooms/room1_dp.py`) is the reference implementation.

---

## Shared modules

| Module | Responsibility |
| --- | --- |
| `streamlit_app.py` | Entry point. Calls `configure_page()`, `room_selector()`, dispatches to `roomN_*.render()`. |
| `core/layout.py` | `configure_page()` and `room_selector()`. **The sidebar holds only the room selector** — nothing else. |
| `core/icy_grid.py` | `IcyGridWorld` — the shared discrete env for Rooms 1–4 (blocked / ice / passable-penalty cells) + `generate_layout()`. |
| `core/episode.py` | `rollout(grid, policy, gamma, max_steps)` — one stochastic episode; returns `(path, G, outcome)` with **discounted** `G = Σ γᵗ·r₍ₜ₊₁₎`. |
| `algorithms/*.py` | Algorithm implementations, **adapted from `code examples/`** (same math). Extended only to return per-iteration history and expose metrics. |
| `rooms/roomN_*.py` | One module per room, each exposing a single `render()` function. |

---

## Page skeleton (top → bottom)

1. **Header** — `st.markdown("### Room N · <Algorithm>")` + a one-line task caption.
2. **About** — `st.expander("ℹ️ About this room", expanded=True)`: 2–4 sentences on
   the idea and a "how to use it" line. No long theory (see Core UI Rules).
3. **Row 1 — Setup** — `st.columns([3, 2])`:
   - **Left (board):** an `st.empty()` board placeholder + a **legend** caption.
     Shows the environment immediately, before training.
   - **Right:** `🎮 Environment & Physics` controls (+ a `🎲 Regenerate` button
     when the layout is randomized).
4. **Row 2 — Algorithm** — `st.divider()` then a full-width `🧠 Algorithm` section:
   method selector + hyperparameters, followed by the primary **`🚀 Train`** button.
5. **Train gate** — compute a `sig` tuple of every env + algorithm parameter. Store
   it on train; if `st.session_state["<room>_trained_sig"] != sig`, **`return`**
   (results stay hidden until the user trains this exact configuration).
6. **Row 3 — Training results** (only past the gate) — `st.divider()` +
   `#### Training results`, then:
   - **KPI row:** `st.columns(3–4)` of `st.metric`.
   - **View-controls row above the board:** iteration scrubber + display toggles
     (e.g. "Show policy arrows"), on a single row.
   - **Results board (left) + ▶️ Play controls (right):** `st.columns([3, 2])`.
     Play controls = max-steps slider, any per-room live metric (e.g.
     success-within-cap), animation speed, the **▶️ Play Episode** button, and the
     episode result.
   - **Graphs row(s) below:** learning / convergence curves full-width.

---

## Conventions (must hold in every room)

- **On-page parameters.** All controls live on the page. The sidebar is *only* the
  room selector.
- **Tooltips everywhere.** Every widget passes a `help="…"` string.
- **Goal reward = +100**, fixed across all rooms.
- **Train-gated.** Results/analytics/Play appear only after `🚀 Train`. Changing any
  environment or algorithm parameter (reflected in `sig`) returns the room to its
  pre-train state — never show stale results against a changed setup.
- **Layout changes only on click.** Randomized layouts are generated once and kept
  in `st.session_state`; only `🎲 Regenerate` reshuffles them — never a slider drag.
  Show a hint when the count sliders no longer match the shown layout.
- **Play Episode.** Animates one rollout on the *results* board using the currently
  viewed policy, then reports the **discounted return G** (same γ as V), the step
  count, and a **success/timeout** result. The result metric shows an **icon only**
  (✅ / ❌); the spelled-out outcome goes in the banner above it.
- **Board.** A Plotly heatmap/canvas with a legend caption; special cell types must
  be visually distinct and named in the legend.
- **Guard session state.** Episode dicts stored in session must be read defensively
  (`ep.get("sig") == sig and "<expected key>" in ep`) so a schema change across a
  code edit can't crash a long-running session.

---

## Algorithm sourcing

Algorithms are **adapted from `code examples/`**, keeping the same update math
(e.g. `dynamic_programming.py` mirrors the reference `value_iteration.py` /
`policy_iteration_probabilistic.py`: same Bellman backups, same `θ = 1e-3`
convergence, transitions/rewards via `grid.get_transition_probs_and_rewards()`).
Permitted extensions: iterating only navigable cells (`grid.actions`), recording a
per-iteration `history` of `{V, policy, delta}` snapshots, and exposing exact
metrics (e.g. `expected_steps_to_goal`, `success_prob_within`). Document any
deviation from the reference in the module docstring.

---

## Verification

The in-app browser's **screenshot / click tools time out** on this Streamlit +
Plotly app. Verify UI logic headlessly with `streamlit.testing.v1.AppTest`:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("streamlit_app.py", default_timeout=120).run()
assert not at.exception
[b for b in at.button if "Train" in b.label][0].click().run()
[b for b in at.button if "Play" in b.label][0].click().run()
assert not at.exception
```

Check: no exceptions pre/post train/play, the expected widgets and KPI metrics are
present, and env logic (transition sums, reachability, metric-vs-empirical) holds
in a plain headless script.

---

## `render()` skeleton

```python
def render():
    st.markdown("### Room N · <Algorithm>")
    st.caption("<one-line task>")
    with st.expander("ℹ️ About this room", expanded=True):
        st.markdown("...")

    # Row 1 — setup board + environment controls
    board_col, env_col = st.columns([3, 2])
    with board_col:
        board = st.empty(); board_caption = st.empty(); st.caption(LEGEND)
    with env_col:
        env = _env_controls()          # + 🎲 Regenerate if randomized

    # ... resolve/persist layout (regenerate only on click) ...
    board.plotly_chart(_figure(...))   # env preview before training

    # Row 2 — algorithm row
    st.divider()
    algo_params, train = _algo_row()

    sig = (...env..., ...algo...)
    if train:
        st.session_state["roomN_trained_sig"] = sig
    if st.session_state.get("roomN_trained_sig") != sig:
        return                         # train gate

    # Row 3 — training results
    st.divider(); st.markdown("#### Training results")
    # KPI metrics → view-controls row → results board + ▶️ Play → graphs below
```
