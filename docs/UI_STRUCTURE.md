# Room UI Structure — the shared contract

Every room in the RL Escape Room app follows the **same page structure** so the
experience is consistent from room to room. The *parameters* differ per room, and
some rooms add *extra graphs*, but the skeleton, flow, and conventions below are
fixed. Room 1 (`rooms/room1_dp.py`) is the reference implementation for the skeleton;
Rooms 2–3 add the pattern for a *learning* room (checkpoint scrubber, learning curves,
DP-benchmark row) — `rooms/room3_sarsa.py` is the most recent and closest to what Room 4
needs.

---

## Shared modules

| Module | Responsibility |
| --- | --- |
| `streamlit_app.py` | Entry point. Calls `configure_page()`, `room_selector()`, dispatches to `roomN_*.render()`. |
| `core/layout.py` | `configure_page()` and `room_selector()`. **The sidebar holds only the room selector** — nothing else. |
| `core/icy_grid.py` | `IcyGridWorld` — the shared discrete env for Rooms 1–4. Cell types: blocked / ice / passable-penalty / teleport / **pit** (terminal hazard) / **shield** (pickup that cancels slip). Generators: `generate_layout()`, `generate_portals()`, `generate_shields()`. |
| `core/episode.py` | `rollout(grid, policy, gamma, max_steps)` — one stochastic episode; returns `(path, G, outcome)` with **discounted** `G = Σ γᵗ·r₍ₜ₊₁₎`. `outcome` is `"goal"` / **`"fell"`** (ended on a terminal hazard) / `"timeout"`. Pass `with_landings=True` for a 4th value: the pre-teleport landing cell of each step (Room 2's portal animation). Also `scored_return(G, outcome)` / `TIMEOUT_PENALTY` — the **scoreboard** number (see below). |
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
- **Goal reward defaults to +100** in every room, so rooms stay comparable. Rooms 1–2
  expose it as a `10`–`1000` slider; a new room should default to 100 and only add
  the slider if it teaches something with it.
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
- **Mask any cell that has no V.** Walls, teleports, **pits**, and **the goal** are drawn
  with `z = np.nan` and an icon only. Terminals are terminal — the agent never acts from
  one, so `V(goal)` is 0, *not* the reward it pays on entry; painting the reward there puts
  a number on a scale labelled `V(s)` that is not a V. The scale is `RdBu` with
  `zmid=0`, so **high V is blue** and negative is red — do not describe it as "warm =
  good" in About text.
- **Hazards must not read like scenery.** Room 3's abyss shipped as near-black beside
  dark-grey walls, so *fatal* and *merely impassable* looked identical — the single most
  important distinction on that board. Give each cell type a colour that is distinct from
  the others **and** absent from `RdBu`, or it will blend into the value cells around it
  (Room 3: violet abyss, grey walls, blue ice, green shields, amber agent).
- **Scoreboard ≠ maths.** `scored_return(G, outcome)` shows the real discounted `G`
  on a win but a **flat −100 for ANY loss** (fell / caught / timed out), mirroring the
  +100 goal, so escaping always ranks above every way of failing and all failures tie.
  It deliberately breaks `G ≈ V(S)` and is for the player only: V, Q, learning updates,
  and benchmarks must use the raw `G`. It **replaces** G with −100 rather than adding a
  penalty, so it cannot double-count the −100 a fall/catch already paid (changed
  2026-07-18 from the old "add penalty on timeout only" behaviour, at the user's request;
  `LOSS_SCORE` in `core/episode.py`, with `TIMEOUT_PENALTY` kept as a back-compat alias).
- **The returns CURVE now shows this scored view too (Rooms 3–4, added 2026-07-18).**
  For player consistency ("−100 for a loss, in training as well"), the returns curve
  floors every losing training episode to −100 in its **display**. This is a pure
  display transform: the stored per-episode `stats["returns"]` stay raw, the learners
  update off per-step rewards (never episode G), and V/Q/benchmark are untouched — so
  the learned policy is bit-identical with or without it. Consequence to state honestly
  in the caption: the moving average now sits below `V*` for *two* reasons (ε-caution
  **and** the −100 loss floor), so it no longer reads as "mean return ≈ V^π". Room 2's
  curve is still raw.
- **Never move that penalty into the learning signal.** Measured in Room 3: at −100 it is
  a no-op; at −500 it collapses the room to 0% escape / 100% timeout. Timing out depends
  on elapsed steps `t`, which is not in the state — the cap is an artifact of training,
  not of the world — so the penalty poisons whatever cell the clock stopped in, and it
  redefines the problem out from under every DP benchmark in the app.
- **Episodes are ephemeral — never store them in session state.** Render the rollout
  (animation, banner, metrics) inside the run that played it, from local variables,
  and let it vanish on the next rerun. A stored episode outlives the policy it was
  run against: guarding it with the train `sig` is *not enough*, because `sig` does
  not include the scrubber position, so moving the scrubber would redraw a stale
  trail over a policy that never produced it. Ephemeral rollouts remove that whole
  class of staleness by construction rather than by remembering a guard.
- **Only three things belong in session state:** the generated **layout**, the
  **trained signature** (the train gate), and widget `key=`s. Nothing else.
- **The Play grid must be unseeded.** Build the env the rollout runs on with
  `rng=None` (fresh entropy) so every press slips differently — that is the entire
  point of watching a stochastic rollout. Only *training* passes an explicit seed,
  for reproducible curves. Streamlit rebuilds the grid on every rerun, so a fixed
  seed on the display grid silently replays the identical episode forever.

---

## A state is not always a cell

Up to Room 2 a state **is** a cell, `(i, j)`. From Room 3 it may not be: a shield is
*carried*, so whether a move slips depends on what was collected earlier, and `(i, j)`
alone stops being Markov. States become `(i, j, has_shield)` whenever `shields=` is
passed (`grid.stateful`). Room 4's guard position will pose the same question.

- **Getting this wrong is silent, not loud.** Nothing crashes: DP just computes a `V*`
  that is quietly an average over "sometimes shielded", and the TD learners chase a
  self-contradicting target.
- **The algorithms need no changes.** `value_iteration`, `policy_value`, `sarsa_control`
  and `monte_carlo_control` all treat a state as an opaque dict key. Keep it that way —
  augmenting state is the *environment's* job.
- **Never read coordinates out of a state by unpacking or comparing.** Use
  `grid.cell_of(s)`, `grid.shield_of(s)`, `grid.start_state()`, `grid.cells()`. In
  particular `s == grid.goal` and `V[START]` are **bugs** on an augmented board — they
  are silently always-False / KeyError-or-wrong. Both shipped once: `rollout` reported
  every escape as a timeout, and a `success[k] = s2 == grid.goal` pinned a KPI at 0%
  while the agent escaped perfectly.
- **A board draws one layer at a time.** With shields a cell has two values (before and
  after pickup). Project with a helper (`room3_sarsa._project`) and name the layer in the
  UI. Remember the off-layer is largely **off-distribution** for a learner — it only
  learns states it actually occupies — so those arrows are noise, not opinions.

---

## Environment invariants

- **The exit is always reachable.** Every layout generator must guarantee it, and
  the guard has to reason about the *folded* transition model, not just walls —
  Room 2's portals strand the goal without blocking a single cell. Place hazards
  one at a time, keep each only if `_connected(...)` still holds.
- **Reachable is not enough — the route must be one the learner can survive learning.**
  Pass `pits=` to `generate_layout()` on any board with terminal hazards, or the guard
  "proves" a path straight through them (measured: **23/40 boards stranded**). Room 3
  goes further and passes `CLIFF ∪ LEDGE`, demanding a route that avoids the cliff edge
  entirely: walls that left the ledge as the only approach made **~1 board in 6**
  unwinnable for SARSA (0%, `V^π` = 0 vs `V*` = 59.9, unfixed by 20,000 episodes) while
  DP solved them fine — which just reads as a broken room. Constrain the *generator*, not
  the environment: the ledge stays walkable, so avoiding it remains the agent's choice.
- **A cell the agent can never stand on is not a state.** Teleport cells are kept
  out of `grid.actions`, so they never get a DP value or a Q entry. Same for terminals
  (goal and pits) and for `(shield_cell, unshielded)`, which entering makes impossible.
- **`.probs` is the true Markov model.** Anything exotic (teleports today) is
  folded into the destination so the algorithm modules stay ignorant of it.
  `move()` may expose the pre-fold detail for animation only.

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

So far: `dynamic_programming.py` (Room 1), `monte_carlo.py` (Room 2, from
`monte_carlo_no_es.py`), `temporal_difference.py` (Room 3 — `sarsa_control`, from
`sarsa.py`; Room 4's Q-learning belongs beside it). The ε schedule (`CONSTANT` /
`DECAYING` / `epsilon_at`) is shared from `monte_carlo.py` — import it rather than
forking it, so the rooms cannot drift apart on what ε means.

- **Decaying ε is the default in every room that exposes the control.** Room 1 is DP and
  has none.
- **Never bootstrap off a terminal.** With terminal hazards, both the goal and a fall must
  end the episode on `Q[s][a] += α·(r − Q[s][a])`. Bootstrapping through a terminal's
  all-zero Q row silently damps the hazard toward zero.
- **Model-side benchmarks are display-only.** `policy_value()` gives the *true* value of a
  learned policy, which is the honest counterweight to a learner's estimate of itself
  (both MC's and SARSA's understate the greedy policy they play, since Q is the value of
  the ε-greedy agent). The learner must never see it.

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

**The AppTest gotcha:** the room selector's `st.selectbox` uses a `format_func`, which
breaks `AppTest.selectbox.set_value(n)`. To test one room headlessly, write a small probe
script that calls `roomN_*.render()` directly and `AppTest` *that*, rather than switching
rooms through the selector.

**Test the number the user reads, not the code you just fixed.** A `success[k] = s2 ==
grid.goal` pinned Room 3's KPI at 0% while the agent escaped perfectly, and 40 passing
checks sailed past it — because they measured success through `rollout()` (already fixed
for augmented states) instead of through the `stats` the KPI actually displays. Where you
can, assert against an **independent witness** rather than a parallel code path: only the
exit pays a positive reward and only a hazard pays a negative one, so `sign(G)` confirms
the success/fall flags without reusing their logic.

**Observing a run must not change it.** Snapshot/checkpoint code must draw from its own
RNG. Room 3's `_snapshot` broke argmax ties off the *training* generator, so the
checkpoint schedule — derived from `n_episodes` — perturbed the very run it recorded, and
2,000- and 5,000-episode runs diverged inside their shared first 2,000. Assert that a
short run and a long run agree over their common prefix, and that the checkpoint count
does not move the trajectory.

**Suspiciously exact agreement is a bug, not a finding.** A dropped branch made a
SARSA-vs-Q-learning comparison run SARSA twice, producing bit-identical Q-tables and a
convincing false conclusion. When two things that should differ agree perfectly, check the
harness before believing the result.

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
