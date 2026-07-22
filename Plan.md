# Plan.md – Interactive Reinforcement Learning Escape Room

## 1. Overview and Core Principles

This document outlines the design and architecture for a Streamlit-based interactive Web application. The platform visualizes the evolution of Reinforcement Learning (RL) algorithms through a 6-stage "Escape Room" game with progressively increasing difficulty.

### Core Architecture & UI Rules

* **Unifying Physical Theme:** Slippery surfaces (Ice physics) connect the rooms. In Rooms 1–4 (discrete grid), this is modeled as stochastic slip probabilities. In Room 6 (continuous space), it is modeled as low friction, inertia, and momentum. **Room 5 is the stated exception** — it was redesigned 2026-07-20 to direct, inertia-free movement (one chasing enemy, empty arena) for simplicity, so the ice theme does not apply there; see §Room 5.
* **Minimal Text UI:** The user interface must remain clean and visual. Do not include lengthy theoretical or algorithmic explanations on the screen.
* **Task-Only Descriptions:** Each room will display only a brief 1–2 sentence description of the **task/mission** (e.g., navigating hazards, dodging guards) at the top of the main view.
* **Tooltip Parameter Explanations:** All controls, hyperparameters, and environment settings (rendered **on the page**, not the sidebar) must use Streamlit's native question-mark tooltip feature (using the `help="..."` argument inside widgets) to explain their mathematical or mechanical purpose.
* **Goal Reward — default +100, adjustable:** Reaching the exit is the only positive reward in Rooms 1–3, 5 and 6. **Room 4 is the deliberate exception** — its bonus coin is a second positive reward, and it has to be, because the room's whole question is whether a small certain gain is worth a route that can kill you. That question cannot be posed with the exit as the only payout. The exception is scoped to Room 4 and is not a licence to sprinkle rewards elsewhere; note it also forces `R(s,a,s')` on the shared env (see §Room 4). The goal reward **defaults to +100** everywhere so rooms stay comparable, but Rooms 1–3 expose it as a slider (`10`–`1000`) because it is the scale everything else is measured against — the negative cells, the slip risk, and the discounting all only mean something relative to it. *(This relaxes the original "fixed +100 in every room" rule. Rooms 4–6 should default to 100 and only expose the slider if the room actually teaches something with it.)* **Expose only ONE side of the ratio.** Room 3 briefly had sliders for both the goal reward and its cliff penalty, which is two controls over one quantity; the hazard is now pinned at −100 and the goal reward is the single scale knob. Note the slider is *inert* in a room whose only rewards are the exit and a hazard — it scales V without moving the policy (measured: 100% escape from goal 10 to 1000). It earns its place by making that scale-invariance visible, not by changing behaviour.
* **Code Implementation Source:** The underlying RL algorithms must not be coded from scratch; developers must integrate and adapt the baseline algorithm implementations located in the local **`code examples`** directory.

---

## 2. Streamlit UI Layout Architecture

Every room shares a standardized **on-page** layout (all parameters live on the
page, not the sidebar — the sidebar holds only the room selector). The page reads
top-to-bottom as a guided flow:

```text
+---------------------------------------------------------------------------+
|  Sidebar: Room Selector (Levels 1-6) only                                 |
+---------------------------------------------------------------------------+
|  Title + short task description                                           |
|  ℹ️ About this room (expander, open by default)                           |
+------------------------------------+--------------------------------------+
|  Row 1 — Setup                     |                                      |
|  Left (~60%): live board (env      |  Right (~40%): 🎮 Environment &      |
|  preview before training) + legend |  Physics controls (+ 🎲 Regenerate)  |
+------------------------------------+--------------------------------------+
|  Row 2 — 🧠 Algorithm (full width): method / hyperparameters + 🚀 Train    |
+---------------------------------------------------------------------------+
|  Row 3 — Training Results (after 🚀 Train, full width):                    |
|  * 3-4 KPI metrics (st.metric)                                            |
|  * View-controls row above the board (iteration scrubber, display toggles)|
|  * Results board (left) + ▶️ Play controls (right: max-steps, per-room    |
|    metrics, speed, Play, episode result — G / steps / ✅❌)                |
|  * Learning / convergence curve(s) in their own full-width row(s) below   |
+---------------------------------------------------------------------------+
```

> The concrete, component-by-component contract every room implements —
> including the shared modules and a `render()` skeleton — lives in
> [`docs/UI_STRUCTURE.md`](docs/UI_STRUCTURE.md). New rooms must follow it so
> the app stays visually consistent (parameters and extra graphs differ per
> room; the structure does not).

* **Train-gated flow:** Results, analytics, and the Play control only appear
  **after** the user clicks **🚀 Train**. Before training, the page shows just the
  About text, the environment board, and the setup controls. Changing any
  environment or algorithm parameter returns the room to the pre-train state so
  stale results are never shown against a changed setup.

* **Standardized ▶️ Play Episode Control:** Every room includes a **▶️ Play Episode**
  button that animates a single rollout of the current (learned or optimal) policy
  on the live game board, stepping through the real stochastic environment — so the
  user watches the agent actually escape or slip into a hazard. In the discrete
  rooms (1–4) it animates the agent cell-by-cell; in the continuous rooms (5–6) it
  is the smooth trajectory animation described per room. A **max-steps-per-episode**
  slider caps the rollout length. After the episode the room reports its
  **discounted return G = Σ γ^t·r₍t+1₎** (defined the same way as V, so a successful
  run from the start averages ≈ V(S)), the **step count**, and **success/timeout**.

* **Episodes are ephemeral.** A rollout is rendered only in the run that played it
  and is never stored in `st.session_state` — a stored episode outlives the policy
  it was run against, so moving a scrubber would redraw a stale trail over a policy
  that never produced it. See `docs/UI_STRUCTURE.md`.

* **Scoring a loss (Rooms 2–4).** Raw G ranks giving up *above* escaping the hard
  way: a run that wanders and times out scores ~0, while an escape that crossed cost
  cells can score negative. So the *reported* return (`core.episode.scored_return`)
  shows a **flat −100 for ANY lost episode** — fell, caught, or timed out — mirroring
  the +100 exit, so escaping always ranks above every way of failing and all failures
  tie. This is a **scoreboard** number only — it deliberately breaks `G ≈ V(S)`, and
  nothing doing maths (V, Q, the learners' updates, the DP benchmark) may use it. The
  returns *curve* is the one **display** that mirrors it — see the next bullet. Room 1
  does **not** do this: its agent is DP and never sees a return at all.

* **It REPLACES G with −100 on any loss; it does not ADD a penalty (changed 2026-07-18,
  user request: "the G of every loss should be −100 no matter when or how it loses").**
  The earlier version added −100 only on `outcome == "timeout"` and left a fall/catch
  showing its raw *discounted* G (~−80 for an early fall, less for a late one) — which
  is inconsistent across losses and across timing. Replacing rather than adding also
  means it can never double-count the environment's own −100 that a fall/catch already
  paid. A win still shows its true discounted G, which (the exit being the only positive
  reward) is always well above −100, so the ordering "win > any loss" holds.

* **The returns CURVE mirrors this scoring too (Rooms 3–4, 2026-07-18: "−100 for a loss,
  in training as well").** The returns curve floors every losing *training* episode to
  −100 in its DISPLAY, so the chart reads like the Play scoreboard. It is a pure display
  transform — `stats["returns"]` stays raw, the learners update off per-step rewards
  (never episode G), and V/Q/benchmark are untouched, so the learned policy is identical
  with or without it (verified). Honest consequence: the moving average now sits below
  `V*` for two reasons — ε-caution **and** the −100 loss floor — so it no longer reads as
  "mean return ≈ V^π"; the captions say so. Room 2's curve is still raw (not requested).

* **Keep the "didn't finish" penalty on the SCOREBOARD — never in the learning signal.**
  Tested in Room 3: at −100 it is a no-op, and at −500 it *destroys* the room (0% escape,
  100% timeout). Timing out depends on elapsed steps `t`, and `t` is not in the state —
  the cap is an artifact of training, not a fact about the world — so the penalty lands on
  whatever cell the clock happened to stop in and poisons it. It also silently redefines
  the problem, so the learner would optimise something `V*` does not measure and every DP
  benchmark in the app would stop meaning anything. The Markov way to punish dithering is
  a **step cost**; Rooms 1 and 3 both measured one as unnecessary.

---

## 3. Room-by-Room Technical Specifications

### Room 1: Dynamic Programming (Bellman Equations)

* **Task Description:** Navigate a static icy labyrinth from the starting tile to the exit while factoring in the risk of slipping.
* **State & Action Space:** 10x10 discrete grid (100 states), 4 directional actions. The board has three special cell types: **🧱 blocked** cells (walls the agent cannot enter), **🟦 slippery ice** cells (moves may slip perpendicular), and **🟥 negative-reward** cells (*passable* — the agent can cross them but pays a negative reward each entry; only the goal is terminal). A legend under the board identifies each type.
* **On-page Controls (with tooltips):**
* **Environment & Physics:** Count sliders for **blocked cells** (`0`–`20`, default `8`), **slippery cells** (`0`–`40`, default `15`), and **negative-reward cells** (`0`–`15`, default `6`); **slip probability** (`0.0`–`0.8`, applied only on ice cells); **negative reward value** (`-10`–`-1`, default `-5` — deliberately kept small against the goal reward, so crossing a 🟥 cell is a cost worth weighing and never a reason to abandon the exit); **goal reward** (`10`–`1000`, default `100`); and a **🎲 Regenerate layout** button. Placement is random from a fixed seed and always keeps the start→exit reachable; Regenerate reshuffles the seed.
* **Algorithm (its own row):** DP method selector — **Value Iteration** vs. **Policy Iteration** — and discount factor ($\gamma$) between `0.5` and `0.99`, followed by the **🚀 Train** button. *(Note: Convergence threshold $\theta$ is hardcoded in the background at `1e-3`).* The board layout only changes on **🎲 Regenerate** (or first load), never while the count sliders are dragged.


* **KPI Metrics:** Iterations to convergence, maximum delta in the final iteration, expected starting-state value V(S), and **expected steps to exit** (model-exact mean moves from start under the optimal policy). Beside the Play controls, a live **success-within-cap** metric shows the model-exact probability the viewed policy reaches the exit within the current max-steps cap (updates as the slider moves).
* **Visualizations:**
* Live value heatmap overlay on the grid showing reward diffusion from the exit backward, with slippery ice and negative-reward cells highlighted, and **walls *and the exit itself* masked**. The exit is terminal — the agent never acts from it, so V(exit) = 0, not the reward it pays on entry. Painting the reward there put a number on a scale labelled V(s) that was not a V at all. Note the scale is Plotly `RdBu` with `zmid=0`: **high V is blue**, negative is red.
* Log-scale convergence curve showing delta dropping to zero, with a marker for the iteration currently being viewed.
* **Training-results board + iteration scrubber:** The results section shows the board again (with ▶️ Play controls beside it) and a "view iteration" slider that replays the value function and greedy policy as they were at each iteration — revealing the contrast between Value Iteration (intermediate policies are poor until late) and Policy Iteration (intermediate policies are already strong). The convergence curve sits in its own row below.
* **▶️ Play Episode:** A "max steps per episode" slider caps the rollout; the room animates one episode of the *currently-viewed* policy across the stochastic ice (where a slip can push it off course or into a penalty cell) and reports its **discounted return G** (computed like V), step count, and success/timeout.


* **Implementation Note:** Utilize the Dynamic Programming reference scripts in the `code examples` folder.
* **No step cost.** One was built and removed. It was meant to stop the agent "giving up" (loitering forever is worth 0, so if every route to the exit crosses 🟥 cells, quitting can beat escaping). Measurement showed the problem does not occur on any board this room can generate — **48/48 random configs escape at 100%** across γ ∈ {0.5…0.99}, penalty ∈ {−20, −100}, slip ∈ {0, 0.4, 0.8}, with *no* step cost. It only appeared in a hand-built layout where 🟥 cells *ring* the exit, which `🎲 Regenerate` never produces. Clamping the negative reward to `-10` makes it less reachable still. Do not re-add a step cost without a reproducible board that needs it.

---

### Room 2: Monte Carlo ($\epsilon$-greedy)

* **Task Description:** Traverse long icy corridors and avoid portal traps that instantly teleport you back to the starting point.
* **State & Action Space:** 10x10 discrete grid, 4 directional actions. Ice is **placed by count exactly as in Room 1** (🟦 slippery cells + a slip probability that applies only on them), so Rooms 1 and 2 differ in what the agent *knows*, not in the physics — which is the comparison the two rooms exist to make. The board adds one new cell type: **🌀 portal traps**, which teleport the agent back to the start on entry. Blocked cells carve the "corridors".

> **No 🟥 negative-reward cells here — they were built and removed.** They break MC control outright. Bumping a wall returns 0 and leaves the agent in place, so once penalty cells push early sampled returns negative, `argmax Q` picks "bump" — and that choice then *manufactures its own evidence*, because every later episode is a full cap of bumping, confirming $Q(\text{bump}) = 0$ while real actions keep their negative estimates. Measured at Room 2's defaults: **0% success and a pure loitering policy**, even though the random walk found the exit in **7%** of episodes and $V^*$ was **14.4**. Not fixable by a step cost (still 0%) nor by optimistic initial values at $Q_0 \in \{5, 20, 100\}$ (still 0%). Portals are innocent — 0 cells/3 portals reaches 100%; 6 cells/0 portals reaches 0%. Room 1's DP is immune because it backs up through the true model rather than sampled returns, which is exactly why 🟥 cells live there and not here. Do not "restore" them for board symmetry without solving that.

* **On-page Controls (with tooltips):**
* **Environment:** **Blocked cells** (`0`–`30`, default `20` — more walls than Room 1, to form corridors), **slippery cells** (`0`–`40`, default `20`), slip probability (`0.0` to `0.8`), portal trap count (`0` to `5`, default `3`), **goal reward** (`10`–`1000`, default `100`), and a **🎲 Regenerate layout** button.
* **Algorithm:** Discount factor $\gamma$ (`0.5`–`0.99`, default `0.90`), training episodes (`100` to `5,000`), max steps per **training** episode (`50` to `500`, default `300`), exploration mode selector: **Constant $\epsilon$** (single slider, default **`0.30`**) vs. **Decaying $\epsilon$** (initial `1.0`, minimum `0.05`, decay `0.998`), then **🚀 Train**.

> **$\gamma$ is mandatory here, not cosmetic.** The goal reward is the only reward and portals carry *no* reward penalty — so with $\gamma = 1$ every successful path scores identically and a portal costs literally nothing. Discounting alone is what makes long paths and portal traps expensive. Making $\gamma$'s role visible is a large part of this room's teaching value.

> **Constant $\epsilon$ defaults to 0.30 because low $\epsilon$ silently breaks the room.** Measured over 8 seeds × 2,000 episodes: $\epsilon = 0.05$ fails **6/8**, $\epsilon = 0.10$ fails **4/8**, $\epsilon = 0.30$ fails **0/8** (policies worth 9.3–14.4 vs $V^* = 14.84$), $\epsilon = 1.00$ fails **5/8**. It is a *sweet spot*, not a monotone knob, and both ends fail for opposite reasons: too low and the agent follows its own arbitrary starting policy, never reaches the exit, so every return is 0 and Q stays flat; at $\epsilon = 1$ it never exploits, reaches the exit in ~2% of episodes, and the extracted greedy policy stays essentially random. Constant `0.30` matches decaying $\epsilon$. Note the failures are *coin flips*, not deterministic — a single lucky run at $\epsilon = 0.10$ proves nothing.

* **Portal semantics:** Entering a portal deals **no reward penalty** — it only teleports the agent to the start, and $\gamma$ does the punishing. Because the teleport is folded into the transition model, a portal cell is a **transient** state the agent is never *standing* on; portal cells are masked in the value heatmap and rendered as 🌀.
* **KPI Metrics:** Success rate over the last 100 episodes, moving-average return over the last 100 episodes, **start-state value $V(S)$** ($\max_a Q(S,a)$ at the viewed checkpoint), and the $\epsilon$ in force at the viewed checkpoint.

> **$V_{MC}(S)$ understates the policy — say so where it is shown.** MC never learns $V$: it averages returns into $Q(s,a)$ and reads $V_{MC}(s) = \max_a Q(s,a)$. Measured at **7.6** when the learned policy was really worth **12.4** out of an optimal **14.8**. Two effects compound: (1) $Q$ is the value of the *ε-greedy* agent, not of the greedy policy ▶️ Play runs; (2) the reference's $lr = 1/N$ makes $Q$ a plain lifetime average, weighting the early random-walk returns exactly as heavily as recent ones. The benchmark row therefore shows $V_{MC}(S)$ beside the policy's **true** value from `policy_value()` — an exact model-side evaluation, verified against 20,000 empirical rollouts (13.842 exact vs 13.838 ± 0.038) — plus $V^*(S)$, with a caption naming both causes.

* **Step size stays at the reference's $1/N$** (sample average), not a constant $\alpha$ — faithful to `monte_carlo_no_es.py` and to the textbook definition of MC as "average the observed returns", even though constant $\alpha$ measurably converges faster (10.12 vs 7.65 estimate, 13.89 vs 12.41 policy, at 20k episodes). Deliberate choice; do not "fix" it.
* **Visualizations:**
* Episode return scatter with a 50-episode moving average, plus a **dashed reference line at the DP-optimal $V^*(S)$**.
* Steps-to-goal curve demonstrating the transition from random exploration to efficient routing (green = escaped, red = timed out).
* **Exploration-rate curve:** the $\epsilon$ actually in force per episode. Read against the return curve, it shows the return climbing as $\epsilon$ falls.
* **Episode scrubber** (mirrors Room 1's iteration scrubber): replays the greedy policy and $V_{MC}$ as they stood at each of ~50 recorded **checkpoints**. All three curves carry a marker for the viewed episode.
* **DP benchmark row (extra, full-width):** three numbers — $V_{MC}(S)$ (what MC believes), the true $V^{\pi}(S)$ of that same policy, and $V^*(S)$ — above $V_{MC}$ vs. $V^*$ heatmaps side by side, **both drawing their policy arrows** off the shared *Show policy arrows* toggle, so every cell where the arrows differ is one where sampling has not yet found what the model already knows. The benchmark is displayed only; the learner never sees it.
* **▶️ Play Episode:** the standardized control *is* the plan's "test mode" — it animates the greedy ($\epsilon = 0$) policy, drawing a **portal touch as its own frame** (via `rollout(..., with_landings=True)`) so the jump back to the start reads as a trap firing rather than a rendering glitch. Its max-steps slider is a **view-time** cap, separate from the training max-steps above. A timed-out run takes the **−100 timeout penalty** on its reported G (see §2).

* **Implementation Notes:** Adapt `code examples/monte_carlo_no_es.py` (on-policy first-visit MC control, $\epsilon$-soft). Required deviations, documented in the module docstring:
* **First-visit check:** the reference's `if (s, a) not in states_actions[:t]` is $O(T^2)$ per episode; at max settings (5,000 × 500) that is ~$10^9$ operations and will hang the app. Precompute a first-occurrence index → $O(T)$, identical semantics.
* **Checkpointed history:** snapshotting Q every episode (5,000 × 100 × 4) is far too large for `st.session_state`. Record ~50 evenly-spaced checkpoints of `{V, policy, eps, episode}`; the scrubber ranges over checkpoints.
* **Seeded RNG:** the reference calls `np.random.*` globally. Thread an explicit `default_rng(seed)` so training is reproducible across `st.cache_data` evictions. The **display/Play grid must stay unseeded** so each Play slips differently.
* **Random tie-breaking:** keep the reference's `max_dict` behaviour (uniform choice among argmax ties) — with a cold, all-zero Q this is exactly what makes early MC a random walk.
* **Cold start is expected, not a bug.** Until the agent first stumbles onto the exit, every return is 0 and Q stays flat. This is why max-steps defaults to 300 and why low episode counts honestly show nothing — it *is* the Monte Carlo lesson.

**Other exploration strategies (surveyed, not adopted):** the `code examples` folder contains **no** UCB, Thompson sampling, or softmax/Boltzmann. It offers only $\epsilon$-greedy, **Exploring Starts** (`monte_carlo_es.py`), and **optimistic initial values** (`optimistic.py` / `optimistic_initial_values.py`, bandit-only). Both alternatives fix the low-$\epsilon$ failure (0/8 each), but neither beat simply raising $\epsilon$, and each costs something: Exploring Starts begins each episode from a **random cell**, so per-episode returns are no longer "return from the start" and the return curve's $V^*(S)$ reference line stops being the same quantity; optimistic initial values has no grid-world version in the reference and produced the noisiest policies. Revisit only if exploration becomes a teaching surface in its own right.

**Shared-code changes this room required (`core/icy_grid.py`):**
* `IcyGridWorld` gained an optional `teleports={cell: dest}`. `_build_probs` folds the teleport into the destination so `.probs` remains a true Markov model (DP, and later Rooms 3–4, keep working unchanged). `move()` samples the **physical** landing cell first and records it in `last_landing`, so the Play animation can render the portal moment. Room 1 passes `teleports=None` and is unaffected.
* `move()` no longer calls `np.random.choice` per step (~10 µs → ~2 µs, over ~1.5M training steps). Per-`(s, a)` cumulative distributions + a single `rng.random()`. Same distribution; Room 1 gets a free speedup. Worst case the UI allows (5,000 × 500) trains in ~10 s, cached behind a spinner.
* Walls and ice reuse `generate_layout()`. Portals **cannot** reuse that sampler and get their own `generate_portals()`: a portal is *transient*, so one sitting on the only cell leading into the goal strands the exit even though nothing is walled — the board silently becomes unsolvable and $V^*$ collapses to 0 everywhere (this happened on the very first seed tried). `generate_portals()` places them one at a time, keeping each only if the start still reaches the goal through the **folded** model, mirroring the guard `generate_layout()` already applies to walls. `_connected()` gained an optional `teleports` argument for this, and `generate_portals()` an `exclude` set so portals are never drawn on top of the ice (cosmetic only — a portal's iciness never enters the model). Verified across 360 seed × portal-count combinations.
* Teleport cells are excluded from `grid.actions` — the agent is never *standing* on one, so it is not a state. This also keeps them out of DP's $V$ and MC's $Q$.

---

### Room 3: SARSA (On-Policy TD)

* **Task Description:** Carefully cross a slippery ice ledge suspended over a deep abyss without falling off the edge.
* **State & Action Space:** 10x10 discrete grid (Cliff Walking layout), 4 directional actions. **Start `(9,0)`, exit `(9,9)`, and the abyss is row 9, columns 1–8** — the start and the exit sit on opposite lips of the same chasm, so the direct route runs along the ledge (row 8) and the safe route detours upward. This geometry *is* the lesson and is therefore **fixed**, not randomized.

> **A fall is TERMINAL — the episode ends.** This is a deliberate departure from Sutton & Barto's Cliff Walking (where a fall costs −100 and *resets to the start* with the episode continuing), and it was chosen over the reset variant for two measured reasons. **(1) It needs no new reward machinery.** A fall resolves *to* the cliff cell, so `IcyGridWorld`'s existing resulting-state reward lookup carries the −100 correctly — verified: entering `(9,4)` returns exactly `-100.0` and `game_over` is `True`. The reset variant does **not** work: see the trap below. **(2) It is dramatically more robust to ε.** Measured at cliff = −100, γ = 0.95, α = 0.1: the terminal cliff reaches **94–96% success at ε = 0.10, ε = 0.30, and decaying ε alike**, whereas a *passable* penalty cell scores **0% at ε = 0.10** (3 goal-hits in 3,000 episodes).

> **⚠️ Trap — do not "restore" the classic reset-to-start cliff without fixing the reward fold.** A reset cliff is a penalty cell that is *also* a teleport, and `IcyGridWorld.rewards` is keyed by the **resulting** state while `_build_probs` folds a teleport into its destination. So a fall resolves to `start`, the reward is looked up *at* `start`, and it is **0.0** — measured: `move()` onto the cliff returns `0.0` and every DP reward on that transition is `0.0`. The agent cannot feel the cliff at all. Rooms 1–2 never exposed this because Room 2's portals deliberately carry **no** reward; Room 3 would have been the first room to put a reward on a transient cell. The fix (prototyped and verified, not merged — nothing needs it today) is to attribute rewards to the **physical landing** cell and fold them as a conditional expectation, `E[r|s,a,s'] = Σ_l p(l)·R(l) / Σ_l p(l)`. It must genuinely *average*, not copy: with an icy start, action `R` reaches `start` both by hitting the cliff (−100, p = 0.9) **and** by bumping the wall (0, p = 0.05), giving `E[r] = -94.737`. It must also guard zero mass — at slip = 0 the model still emits zero-probability slip outcomes, which divide by zero. Verified against reality once fixed: exact `V*(S) = 42.371` vs 4,000 rollouts at `42.522 ± 0.399`.

* **On-page Controls (with tooltips):**
* **Environment:** **blocked cells** (`0`–`20`, default `8`, as in Room 1), **slippery cells** (`0`–`40`, default `20`), slip probability (`0.0`–`0.8`, **default `0.1`**), **shields** (`0`–`2`, default `1`), **goal reward** (`10`–`1000`, default `100`, as in Rooms 1–2), and **🎲 Regenerate layout** — which reshuffles the walls, ice and shields. The abyss, start, and exit never move.

> **The cliff fall is FIXED at −100; the goal reward is the slider.** Both were briefly exposed, which was a mistake: only the *ratio* between them means anything, so two sliders just move the same quantity twice. The fall is pinned to −100 to mirror the standardized +100 exit and the −100 scoreboard penalty for never getting out — falling and giving up cost the same — leaving the goal reward as the one honest scale control, exactly as Rooms 1–2 have it.

> **🛡️ Shields change what a STATE IS — this is the room's one deep change.** A shield is *carried*, so whether a move slips depends on what was picked up earlier: the cell alone is **not Markov**, and states become `(i, j, has_shield)`. Getting this wrong is silent, not loud — DP would compute a `V*` that is quietly an average over "sometimes shielded", and the TD learners would chase a self-contradicting target. `IcyGridWorld` therefore augments the state itself (`shields=` ⇒ `stateful`), and the **algorithms need no changes at all**: `value_iteration`, `policy_value` and `sarsa_control` already treat a state as an opaque dict key. Only code reading coordinates out of a state must use `cell_of()` / `shield_of()` / `start_state()` instead of unpacking — `rollout`'s `s == grid.goal` and `success_prob_within`'s `grid.start` both had to change, or every escape would have been misreported as a timeout. Verified: `V(shielded) ≥ V(unshielded)` on all 90 shared cells, shielded moves on ice have exactly one outcome at p=1.0, and the augmented model matches reality (exact `V^π` 37.79 vs 3,000 rollouts at 37.80 ± 0.05).

> **The shield is a TEMPTATION, not a free upgrade — and that is the point.** Measured over 5 boards: *holding* one is worth **+6.7 (slip 0.1) to +12.7 (slip 0.8)** at the start, yet the optimal policy detours to collect it on only **1 of 5 boards**, and at slip 0.8 on **none**. Fetching it costs more discounting than the immunity returns. So the room does not *claim* the shield is good; the benchmark row computes, per board, what holding one is worth and whether `V*` actually bothers — an honest question with an exact answer, which is exactly what having Room 1's DP on tap is for. An earlier draft asserted "any detour on the shielded layer is pure ε-caution"; measurement killed it (see below).

> **⚠️ The shielded layer is mostly off-distribution — do not read it as SARSA's considered opinion.** SARSA only learns `(cell, shielded=1)` for cells it actually reaches *while holding* a shield. Tracing its greedy policy from `(start, shielded=1)` — a state a shielded agent has no reason to occupy — returned "never crosses" on 3 of 5 seeds: unvisited noise, not caution. The honest trace starts from the real `start_state()`. This killed the tidy claim that the shielded layer isolates ε-caution from physics (it also turned out that at slip 0.3 the optimal crosses the ledge *without* the shield at all, so there is no such isolation to show). The UI now says this out loud rather than inviting the user to over-read those arrows.
* **Algorithm:** Learning rate ($\alpha$, `0.01`–`0.5`, default `0.1`), discount $\gamma$ (`0.5`–`0.99`, default `0.95`), training episodes, max steps per training episode, and an exploration mode selector — **Decaying ε (default**, `1.0` → `0.05`, decay `0.998`) vs. **Constant ε** (default `0.30` when chosen), then **🚀 Train**.

> **Decaying ε is the default in every room that has the control** (Rooms 2 and 3; Room 1 is DP and has no ε). In Room 3 the measurements independently favour it: decaying produces the **most cautious policy of any setting swept — 63% of `V*`**, below every constant ε including 0.30 — which is precisely the behaviour this room exists to show. It is also the setting Room 2 found reliable on sparse reward, so the two rooms now agree.

> **Exploration still matters more than it looks — the magnitude of the cliff is what bites, not the penalty per se.** The start sits *on* the abyss row, so a harsh cliff teaches the agent to flee the entire bottom of the board — and fleeing is also the direction of the exit. At ε = 0.10, cliff values of 0, −1 and −10 all reach 100% and only −100 collapses; with a *passable* penalty that is fatal (0%), while the terminal cliff survives it (94–96% at every ε). Constant ε = 0.30 remains the best of the constant options for legibility.

* **KPI Metrics:** **training failures broken out by cause** — falls into the abyss vs timeouts (added 2026-07-18 at the user's request: "how many times did the agent fail, by a fall or a timeout") — plus success rate over the last 100 episodes and the learned policy's **true** value V(S) from `policy_value()` at the viewed checkpoint. A caption tallies all episodes as escaped / fell / timed out.
* **Visualizations:**
* **Cumulative cliff falls chart.** It levels off but **never reaches zero** — measured 7–25 falls in the last 100 episodes across ε settings. Slip and ε both keep pushing the agent in; that residual is honest and worth not hiding.
* Q-table arrow overlay showing SARSA's conservative tendency to point away from the hazard, plus the value heatmap. **Cliff cells are terminal, so they are masked (`z = np.nan`, icon only)** exactly like the goal — the agent never acts from one, so it has no V.
* **DP benchmark row** — ⚠️ **REMOVED from the UI 2026-07-18 (user: "TMI").** The learned-vs-`V*` side-by-side heatmaps and the three value metrics (SARSA's estimate / true `V^π(S)` / `V*(S)`) are gone. `V*(S)` survives as the dashed reference line on the return curve, and the learned policy's true `V(S)` stays as a KPI; the exact-vs-learned comparison is no longer foregrounded. The code and `policy_value()`/`value_iteration()` machinery remain — only the display was cut. **Room 3 still does NOT train Q-learning** — that contrast is Room 4's reveal.
* **▶️ Play Episode:** animates the greedy (ε = 0) policy crossing the ledge. A fall is a **third outcome**, not a timeout — see the required shared change below.

> **Slip partially confounds the room's own lesson — say so rather than hide it.** "SARSA points away from the hazard" is only a fact *about SARSA* if the optimal policy hugs the ledge. It does at slip = 0: measured `V*` crosses the abyss span on **row 8** (the ledge) while SARSA detours to **row 1** (`V^π` 29.20 vs `V*` 59.87). But by slip = 0.1 the *optimal* policy has already backed off to **row 7**, and SARSA crosses at row 6 — only one row apart, so the visual contrast is real but muted. This is why the DP benchmark row matters: it shows the user what optimal actually does at their slip setting, instead of letting them credit SARSA for caution the physics demanded. **The About text should nudge the user to slide slip to 0 to see the pure textbook effect.** (Measured with Q-learning for reference, not shipped: at slip = 0 it hugs row 8 and hits `V^π` = 59.87 = `V*` exactly.)

* **Implementation Notes:** Adapt `code examples/sarsa.py` (on-policy TD control). Its update is `Q[s][a] += α·(r + γ·Q[s2][a2] − Q[s][a])`, bootstrapping off the action *actually taken next* — that is the whole room. Required deviations, documented in the module docstring:
* **No bootstrap off a terminal.** The reference's `while not grid.game_over()` loop never forms a target from a terminal state. With the cliff terminal, both the goal and a fall must use `Q[s][a] += α·(r − Q[s][a])`. Bootstrapping off a terminal's all-zero Q row silently damps the −100.
* **Checkpointed history + seeded RNG**, exactly as Room 2 does, for the scrubber and for reproducibility across `st.cache_data` evictions. The display/Play grid stays **unseeded**.
* ⚠️ **Snapshotting must use a SEPARATE RNG from training.** `_snapshot` breaks argmax ties randomly; drawing those from the training generator lets the checkpoint schedule perturb the run it is meant to be observing. Since the schedule derives from `n_episodes`, a 2,000- and a 5,000-episode run **silently diverged inside their common first 2,000 episodes** — caught by an impossible measurement (2 wins at 2,000 episodes vs 4,658 at 5,000, which cannot fit in the extra 3,000). This alone accounted for most of an apparent "decaying ε fails at the defaults" result. Observing a run must not change it.
* ⚠️ **When comparing on-policy vs off-policy anywhere, assert the two actually differ.** A dropped `kind` branch made a SARSA-vs-Q-learning harness silently run SARSA twice; it produced bit-identical Q-tables (364/364 entries, max difference exactly `0.0`) and a plausible-looking "the lesson is destroyed" result. Three exact ties is not a finding, it is a bug. ~16% of non-terminal updates have `Q[s2][a2] ≠ max_a Q[s2,a]`, so any correct comparison **must** diverge.

**Shared-code changes this room requires:**
* **`core/icy_grid.py` — optional `shields=`,** which switches states to `(i, j, k)` (see the note above). Off by default, so Rooms 1–2 keep plain `(i, j)` states and are untouched. Adds `cell_of()`, `shield_of()`, `start_state()`, `cells()`, `is_shield()`, and `generate_shields()` (which keeps a pickup only if it is reachable *and* can still reach the goal without crossing a pit — an unreachable shield is not a hazard, just a lie drawn on the board).
* ⚠️ **`generate_layout()` must be passed `pits=` on a board that has them.** Its wall guard calls `_connected`, which without pits will happily "prove" a route straight through the abyss and so certify a wall that seals the only SAFE path. **Measured: 23 of 40 boards are stranded** at 20 walls without it, and 0 of 40 with it. This is the same bug class as Room 2's portals, in the opposite direction. It also gained `exclude=` so nothing is drawn on the abyss.
* ⚠️ **Room 3 passes `pits = CLIFF ∪ LEDGE`, demanding a route that avoids the ledge — not merely a route.** The exit has exactly one entrance, `(8,9)`, so walls that block the descent down column 9 leave row 8 (every cell of it touching the abyss) as the *only* approach. SARSA, whose entire character is refusing to walk beside a cliff, then never escapes: **measured 0% success and `V^π` = 0 against `V*` = 59.9, and 20,000 episodes does not fix it** (2–4 wins total) — while DP solves the same board fine, because it has the model and never had to survive learning. This hit **~1 board in 6**, which just reads as a broken room. With the guard: **0/12 pathological boards, and all 8 walls still place on every seed.** The ledge stays fully walkable in the environment — this constrains the *generator* only, so any detour SARSA takes remains a choice it made, which is the lesson.
* **`core/icy_grid.py` — multi-terminal support.** `is_terminal()` hardcodes `s == self.goal`. Room 3 needs terminal *hazard* cells (e.g. an optional `pits={cell: reward}`): `is_terminal` returns true for the goal **or** a pit; pits join `self.rewards`; and `self.actions` already excludes terminals, so pits correctly stop being states. Rooms 1–2 pass nothing and are unaffected.
* **`_connected()` must treat pits as impassable.** A fall ends the episode, so a path *through* the abyss is not a path. The reachability guard must exclude pits or it will certify an unsolvable board — the same class of bug Room 2's portals caused, in the opposite direction.
* **`core/episode.py` — a fall is a third outcome.** `rollout()` computes `outcome = "goal" if s == grid.goal else "timeout"`, so a fall is **reported as a timeout** — verified: a suicidal policy returns `outcome='timeout'` having ended on a cliff cell. It needs `"fell"`. And `scored_return()` adds `TIMEOUT_PENALTY` whenever `outcome != "goal"`, which would hit a fall with a **second** −100 (−200 total) on top of the one the environment already paid. It must key on `outcome == "timeout"` instead. Rooms 1–2 are unaffected (with no pits, `!= "goal"` and `== "timeout"` are equivalent).
* **Verify the DP metrics against pits.** `expected_steps_to_goal()` and `success_prob_within()` were written when the goal was the only terminal; confirm they treat a fall as an absorbing non-goal outcome rather than assuming every episode eventually reaches the exit.

> **A learnable "didn't finish" penalty was investigated and REJECTED — there is nothing to fix, and it breaks what it touches.** The worry is reasonable: with the exit worth as little as 10 against a −100 fall, surely the agent should refuse to cross and loiter for a safe 0? **It never does.** Measured across 12 corners of the sliders (γ ∈ {0.5, 0.7, 0.9, 0.95} × goal ∈ {10, 100} × slip ∈ {0.1, 0.8}): **100% escape in every single one**, with only ~25 of 4,000 training episodes timing out (0.6%). The reason is structural — loitering pays exactly 0 while escaping pays *something*, so `V*(S)` stays positive (5.32 at goal 10; still 0.002 at γ = 0.5 / goal 10 / slip 0.8), and γ scales the numbers without changing which way the arrows point. Room 1's "giving up" problem needs *passable* penalty cells lining the route; a terminal cliff you simply walk around cannot create it.
>
> Adding the penalty to the **learning signal** anyway was measured: at −100 it is a **no-op** (100% → 100%; training timeouts 21 → 37). At −500 it is **catastrophic** — the greedy policy collapses to **0% escape / 100% timeout**, with training timeouts exploding 21 → 1,044. The failure is instructive and predictable: **timing out depends on elapsed steps `t`, and `t` is not in the state.** The step cap is an artifact of training, not a fact about the world. So the penalty is charged to whichever cell the clock happened to run out in, poisoning arbitrary states, which causes more timeouts, which poisons more states. It also silently redefines the problem: the learner would optimise something `V*` does not measure, so the room could report a policy "beating" `V*` and the whole DP benchmark would stop meaning anything.
>
> The Markov way to punish dithering is a **step cost** — a real reward on every transition that DP prices exactly (`V*(S)` 5.32 → −4.03 at −1.0/step). It also changes no behaviour here (100% throughout), which is the same conclusion Room 1 reached when it built and removed one. **So the −100 for not finishing stays exactly where it is: on the scoreboard** (`core.episode.scored_return`), where it costs nothing to be right and reads as a deliberate mirror of the cliff — fall −100, give up −100, escape positive.

**Measured defaults (ice-by-count × episodes sweep, 3 seeds, slip 0.1, cliff −100, γ 0.95, α 0.1, ε 0.30):**

* **Ice count barely changes the room — 🟦 20 is a safe default.** Exact `V*(S)` by count: **59.87 (0) → 53.92 (10) → 52.87 (20) → 52.35 (30) → 51.90 (40)**. Nearly the whole effect lands in the first 10 cells and then plateaus, because scattered ice rarely falls on the critical path. SARSA reached **100% success at every count**, and the greedy policy never fell. Ice-by-count is therefore *much* gentler than the "ice everywhere" configuration the earlier probes used (`V*` 40.17) — those numbers were the pessimistic extreme, not the shipped board.
* **Training is effectively free — ~0.9–1.1 s for 5,000 episodes** (vs ~10 s for Room 2's MC at the same count), because a terminal cliff plus a wall-free board makes episodes short. Room 3 can afford generous episode counts; **default 2,000**.
* **More episodes does NOT reliably improve the policy** — `V^π` as a share of `V*` runs **51–84%**, non-monotone in episode count (e.g. ice 0: 61%, 51%, 65%, 62% at 500/1k/2k/5k). This is not underfitting to be tuned away: with constant ε, SARSA converges to the **ε-soft** optimum, not `V*`, and that residual gap **is** the room's lesson. Do not "fix" it by raising the episode cap.
* **Max steps per training episode barely matters** (100/200/300/500 all reach 100% in ~0.4 s) — episodes end quickly against a terminal cliff. **Default 200.**
* **Cliff falls never reach zero: 10–16 per last-100 episodes** at every setting swept. Slip and ε keep pushing the agent in. The chart should level off at a nonzero floor, and that is honest.

> **ε does NOT cleanly control SARSA's conservatism — do not teach that it does.** The intuitive story ("higher ε ⇒ more exploration risk priced in ⇒ wider detour") was measured and **does not hold monotonically**. `V^π` as a share of `V*` over 5 seeds × 3,000 episodes: **ε 0.05 → 83%, ε 0.10 → 80%, ε 0.30 → 66%, ε 0.50 → 87%, decaying → 63%.** The detour does widen from ε 0.05 to 0.30, but ε 0.50 reverses it and scores best of all — while *decaying* ε, which **ends** at 0.05, produces the **most** conservative policy of any setting, so a low final ε plainly does not mean a bold policy. Variance across seeds is large. This is why **decaying is the default** (widest detour, and consistent with Room 2) — a ranking, not a curve. Any UI text claiming "turn ε up to see more caution" would state something the measurements contradict.



---

### Room 4: Q-Learning (Off-Policy TD)

**Status: designed and measured 2026-07-17, NOT yet built.** Everything below that
carries a number was measured on a scratchpad prototype of the environment
(`GuardGrid`) driven by the *real* `algorithms/dynamic_programming.py` and a
prototype `td_control`. The prototype deliberately implements the same interface
`IcyGridWorld` exposes, which is how we know the shared algorithms need no
changes. Four of this section's claims are corrections of the sketch that stood
here before; three more are corrections of measurements taken earlier the same
day. Read the "claims that died" note at the end before adding anything here.

* **Task Description:** Cross the hazardous ledge over the abyss while a patrol guard sweeps the safe detour, with an optional bonus coin sitting out on the ledge itself.
* **State & Action Space:** 10x10 discrete grid **× guard patrol phase × coin flag** — states are `(i, j, g, m)`, **not** `(i, j)`. 4 directional actions. See the state-shape note below; the summary table in §4 previously said "Discrete (10x10)" and was simply wrong.

> **The guard breaks the Markov property exactly as Room 3's shield did.** Whether
> a move is safe depends on where the guard is *now*, so `(i, j)` alone is not a
> state. `docs/UI_STRUCTURE.md` ("A state is not always a cell") already predicted
> this room would pose the question. The answer is the same: **the environment
> augments the state; the algorithms are not told.** `value_iteration`,
> `policy_value` and the TD learners treat a state as an opaque dict key and needed
> **zero** changes against the prototype — verified, not assumed.

**Geometry (fixed — it is the lesson, exactly as Room 3's abyss is):** Room 3's
board is reused — start `(9,0)`, exit `(9,9)`, abyss on row 9 columns 1–8, ledge
on row 8. Room 4 adds a guard patrolling **column 5, rows 0–7**, back and forth,
as a cyclic schedule of period **P = 14**. Walls, ice and the coin are reshuffled
by 🎲 Regenerate; the abyss, the patrol column, start and exit never move.

> **Why a column, and why the FULL column.** Start and exit sit on opposite sides,
> so every route must cross every column. With the guard on col 5 rows 0–7, the
> *only* guard-free crossings are row 8 (the ledge) and row 9 (the abyss) — so the
> room forces a real choice: hug the ledge, or time a crossing. **This does not
> survive a shorter patrol.** Measured over 5 boards: a rows-3–6 patrol leaves rows
> 0–2 and 7 free, and a "fast" guard implemented as *skipping two rows per step*
> occupies only rows 0/2/4/6 — leaving rows 1/3/5/7 free **forever**, which the
> optimal policy walks straight through (5/5 boards dodged). Both shorter tracks
> score *better* on `V^π`/`V*` (up to 92–95% vs 76–84%) precisely **because they are
> degenerate** — the guard has stopped existing. A higher score is not a better room.

> **⚠️ The guard patrol speed selector (Slow/Normal/Fast) is REJECTED — speed is
> not a free knob, it IS the state space.** Speed lives in the phase, so it changes
> the state count and it changes the geometry. *Faster* without gaps is impossible:
> the guard must occupy each row in sequence, so skipping rows is the only way to
> speed it up, and skipping rows opens permanent free crossings (above). *Slower*
> means dwelling, which doubles the period to P = 28 and the states to 5,600, where
> SARSA measured **22%/31%/74%** of `V*` across 3 seeds. Ship the guard at one
> speed. If a speed control is ever wanted, it must be re-derived, not restored.

* **On-page Controls (with tooltips):** Deliberately mirrors Room 3, so the rooms
  differ in the *algorithm*, not the dashboard.
* **Environment:** blocked cells (`0`–`20`, default `8`), slippery cells (`0`–`40`, default `20`), slip probability (`0.0`–`0.8`, default `0.1`), **coin value (`0`–`20`, default `5`)**, **goal reward** (`10`–`1000`, default `100`), and 🎲 Regenerate. **No guard-speed slider** (see above). **No shields** — that is Room 3's lesson, and a fourth state dimension would take states to ~22k.
* **Algorithm:** identical to Room 3 — α (`0.01`–`0.5`, default `0.1`), γ (`0.5`–`0.99`, default `0.95`), episodes (**default `20,000` — see below, Room 3's 2,000 is far too few here**), max steps per training episode (default `200`), and the shared ε selector with **Decaying as the default** (per the app-wide rule). Both learners get the *same* hyperparameters — that is what makes the comparison controlled.

> **⚠️ The goal-reward slider is NOT inert here, unlike in Room 3.** Room 3 measured
> it as pure scale (100% escape from goal 10 to 1000) and kept it only to make
> scale-invariance visible. In Room 4 it sets the **exit : coin ratio**, which is
> the exact quantity the room is about, so it earns its place by changing behaviour.
> The fall and the guard are both pinned at **−100** — one side of the ratio only,
> per the Core UI Rules.

> **Coin value `0`–`20`, default `5` — the sketch's `0.0`–`2.0` was measured and
> rejected.** The value at which the *optimal* policy detours onto the ledge to
> collect ranges **1 to 10** across boards × slip × γ (20 configs; it never fails to
> flip). So `0.0`–`2.0` is not merely weak, it is *marginal*: on some boards inert,
> on others already enough, so the default lands arbitrarily per board. `0`–`20`
> spans "ignore the coin" to "obviously take it" on every board measured. At the
> default board (seed 0, slip 0.1, γ 0.95) the flip sits at **3**: `V*(S)` is 45.56
> at coin 0, 45.63 at coin 2 (**the optimal ignores the coin entirely**), and 48.80
> at coin 5, where it walks 6 ledge cells to collect.

> **⚠️ ONE coin, not two — a second coin is not a bigger version of the first.**
> Two coins measured at **37–49% state-action coverage** vs one coin's **68–98%**,
> and Q-learning's policy collapsed (46%, 29% and 53% of `V*` at 5k/20k/60k
> episodes — *non-monotone, and not fixed by 12× the budget*). The cause is not the
> state count, which was the first and wrong diagnosis (see below): it is that a
> **coin mask is a history dimension the agent only reaches by deliberately doing a
> rare thing**, so masks `01`/`10`/`11` stay unvisited, whereas the guard phase
> cycles with `t` and is covered for free. Adding a *second* coin costs a factor of
> two in states and buys nothing but an untrained table.

* **KPI Metrics (AS BUILT, updated 2026-07-18 — user: "do the same KPIs as Room 3").**
  Metric tiles for the **viewed** learner (chosen by the Show-learner radio), matching
  Room 3's set plus the guard catch: 🕳️ Falls (training) · 🚨 Caught (training) ·
  ⏱️ Timeouts (training) · Success rate (last 100) · V(S) of this policy, above a tally
  caption (escaped / fell / caught / timed out over all episodes). Replaced an earlier
  two-learner table. The head-to-head lives in the falls + returns curves (both learners)
  and the route caption; `V*(S)` is the return curve's dashed line, not a tile (as Room 3).

> **⚠️ "Maximum reward achieved on the final trajectory" and "total guard
> collisions" are both REJECTED as KPIs.** The first is one sample of a stochastic
> rollout — Rooms 2 and 3 both converged on a benchmark row instead, for this exact
> reason. The second was measured and **does not hold**: an early single-seed result
> suggested a tidy "SARSA dies to the guard, Q-learning dies to the abyss" split
> (SARSA 386 catches / 1,734 falls; Q-learning 110 / 3,786). Across **5 seeds the
> catch counts are noise** — SARSA 164–308, Q-learning 73–1,104, no consistent
> ordering. **Falls are the signal that survives: Q-learning falls 2–4× more than
> SARSA on 5/5 boards** (2,981–5,836 vs 5,903–15,802 over 20,000 episodes). That is
> the classic Cliff-Walking signature and the one number to put on screen.

**Measured defaults (full P=14 track, 1 coin @ 5, slip 0.1, γ 0.95, α 0.1, ε 0.30 constant, 5 seeds):**

* **20,000 episodes, not Room 3's 2,000.** At 5,000 episodes SARSA scores 59% of `V*` with a **standard deviation of 20** — the room would read as random. At 20,000: SARSA 80 (sd 17); at 50,000: 84 (sd 15). Training both learners costs **~7.5 s** at 20,000 (SARSA ~4.6 s + Q-learning ~2.9 s), against Room 2's ~10 s ceiling. Affordable; cache it behind a spinner.
* **DP is affordable and stays exact:** value iteration over 2,800 states converges in **23 sweeps / ~0.5 s** — cheap enough to keep the exact `V*(S)` on tap for the return-curve reference line and the route caption (the full benchmark *row* was later removed as TMI — see Visualizations).
* **Lower ε does NOT help — it makes the room worse.** ε of 0.30/0.10/0.05 measured coverage 73–87%/40–59%/39–52% for SARSA. Less exploration means less coverage means a less-trained table. This is the opposite of the intuitive fix and mirrors Room 2's "ε is a sweet spot, not a monotone knob".
* **Decaying ε (default 1.0 → 0.05, decay 0.9995) is measured and is the shipped default.** Slower decay than Room 3's 0.998 because 20,000 episodes over a ~20× larger state space need exploration kept alive longer. Over 5 boards it trains cleanly and the headline holds: **Q-learning takes the coin 5/5, agrees with optimal 2/5; SARSA takes it 0/5, agrees 3/5; Q-learning falls 2–4× more** (≈2,700–8,000 vs SARSA's ≈1,500–2,000). Decaying ε is *gentler* on falls than constant 0.30 (which drove 3,000–15,000) while preserving the contrast. **Nuance worth keeping honest:** with the central coin worth 5, the optimal takes it on ~2/5 boards, so the two learners err in OPPOSITE directions — Q-learning over-takes, SARSA over-detours — rather than "Q-learning wrong, SARSA right". The room's About text and route caption say this; do not simplify it back.

> **⚠️ THE HEADLINE — and it inverts the sketch that stood here.** The old text
> promised "Q-Learning's risky, direct edge path vs. SARSA's safe, detour path",
> implying Q-learning is the clever one. Measured at the defaults over 5 boards:
> **Q-learning takes the ledge coin on 5/5 boards but agrees with the exact optimal
> on only 2/5. SARSA agrees with the optimal on 5/5.** So Q-learning is not
> "aggressive and right" — it is **systematically over-optimistic about the risky
> route**, and SARSA is the one making the optimal call. The room should teach what
> was measured: *Q-learning walks the ledge, and the DP answer tells you, per board,
> whether that was brave or just wrong.* That is a better lesson than the sketch's,
> and it is the entire reason the room keeps the exact DP answer on tap (as the `V*`
> reference line and the route caption) — so the learner can be shown to be wrong.

> **⚠️ Do NOT claim Q-learning earns more value.** `V^π` as a share of `V*` over 5
> seeds: **SARSA 93/91/57/68/60 (mean 74), Q-learning 89/64/78/81/80 (mean 78)** —
> indistinguishable against that spread, and each wins on some boards. The **route**
> contrast is robust (5/5 vs 2/5); the **value** contrast is not. Say the first,
> never the second.

> **Room 4 must train BOTH algorithms on its OWN board.** It cannot import Room 3's
> curves: Room 4's MDP (guard + coin, augmented state) is not Room 3's, so plotting
> the two rooms' returns on shared axes would compare different MDPs and mean
> nothing. Run SARSA as a baseline *inside* Room 4 with identical hyperparameters —
> that is what makes the path overlay a controlled experiment rather than a
> coincidence, and it is why Room 3 stays pure SARSA. **Assert the two learners
> actually diverge** (Room 3 shipped a harness that silently ran SARSA twice):
> verified on the prototype, `max |ΔQ|` = 100.75/100.0/100.0 across 3 seeds.

* **Visualizations:**
* Comparative return chart plotting Q-Learning vs. SARSA on the same axes, both trained here, with a dashed `V*(S)` reference line as in Rooms 2–3.
* **Cumulative falls chart, one line per learner** — the Cliff-Walking signature (Q-learning's line climbs 2–4× faster on every board measured).
* **Steps-per-episode chart (added 2026-07-18, user request "like Room 3")** — Room 3's steps curve, but two 50-episode-average lines (Q-learning / SARSA) to match Room 4's two-learner convention rather than Room 3's single-learner scatter. A short early run is a quick death (fall or catch), not a fast escape; the averages settle toward each policy's true path length as escapes take over.
* **Path Comparison Overlay:** both final greedy paths on one board — Q-learning's ledge/coin route vs SARSA's detour. The board draws **one guard phase at a time**; name the phase in the UI (the same "a board draws one layer at a time" rule shields imposed in Room 3).
* **DP benchmark row** — ⚠️ **REMOVED from the UI 2026-07-18 (user: "TMI"), same as Room 3.** The learned-vs-`V*` heatmaps and the own-estimate/true/optimal metrics are gone. What survives: `V*(S)` as a compact KPI and as the dashed line on the return curve, and the one-line **route contrast** caption (does the optimal / Q-learning / SARSA take the coin?) — which is the honest per-board answer to the coin's question and the room's actual lesson, so it stays. `value_iteration()`/`policy_value()` remain in the code; only the heavy display was cut.
* **▶️ Play Episode:** animates the greedy (ε = 0) policy of the selected learner. A **catch is a terminal outcome like a fall** — `rollout` must report it (see below).

**Shared-code changes — AS BUILT (diverged from the design note, deliberately).**
The design note here said "generalize `IcyGridWorld` to `(i, j, *aug)`". At build
time that was rejected as the highest-regression-risk option against three working
rooms: the guard's mechanics are structurally unlike the shield latch (a phase that
*cycles every step*, and a *phase-dependent terminal*), and cramming them into the
shield machinery would tangle the shared class. Instead:

* **`core/icy_grid.py` — extract the slip physics to free functions** `step_cell()`
  and `slip_outcomes()`, and refactor `IcyGridWorld` to call them. This keeps ONE
  implementation of the (subtle) slip distribution — the thing that MUST stay
  shared — without forcing Room 4's different state shape into the class. Rooms 1–3
  re-verified byte-identical (Room 3 `V*(S)` = 53.473 unchanged; Room 2-ish 14.844).
  Also added a one-line `is_caught(s) → False` so the shared `episode.py` can
  classify terminals uniformly.
* **`core/guard_grid.py` — a NEW `GuardGrid` env**, not an extension of
  `IcyGridWorld`. State `(i, j, g, m)`; reuses `step_cell` / `slip_outcomes`;
  implements its own `cell_of` / `phase_of` / `mask_of` / `start_state`. The RL
  algorithms treat the state as an opaque key and needed **zero** changes — verified
  (`max |ΔQ|` ≈ 100 between the two learners confirms they genuinely diverge). This
  is the same interface the scratchpad prototype validated.
* ⚠️ **Rewards ARE a function of the transition, `R(s, a, s')`** (in `GuardGrid._build`
  / `_reward`), not of the resulting state. A one-shot coin cannot be keyed by `s'`:
  once collected the bit stays set, so re-entry yields an identical `s'` and would pay
  forever. The mask **bit flip** is the reward, visible only in `(s, s')`. Same bug
  class as Room 3's shelved teleport-reward fix. `get_transition_probs_and_rewards()`
  emits `rewards[(s, a, s2)]` and `_q_value` already consumes it, so **DP needed
  nothing**.
* **A guard catch is terminal but is NOT a `pits` entry** — it is phase-dependent, so
  `is_terminal` consults `is_caught(s) = cell_of(s) == track[phase_of(s)]`. **Catch is
  co-location at the resulting phase**, which keeps it a pure state property (Markov)
  and lets `is_terminal` stay one-argument. The SWAP case (agent and guard passing
  through each other) is a near miss under this model — documented, not modelled; a
  distinct absorbing "caught" state would be needed and buys nothing measured.
* **`core/episode.py` — a catch is a fourth outcome `"caught"`**, checked before
  `"fell"`; `scored_return` leaves it alone exactly as it leaves a fall alone
  (already paid −100). Also made `rollout`'s landing-tracking conditional on
  `with_landings` so it no longer requires every grid to expose `last_landing`
  (`GuardGrid` has no transient landings).
* **`generate_layout(exclude=CLIFF ∪ track ∪ coin, pits=CLIFF ∪ LEDGE)`** — walls
  never land on the guard's column (it would walk through walls) or the coin.
  Coin placed first on the central ledge (cols 3–6) via `place_coin`, then excluded.
* **`algorithms/temporal_difference.py` — `sarsa_control` and `q_learning_control`
  now share `_td_control(kind)`**, differing **only** in the bootstrap target
  (`Q[s2][a2]` vs `max_a Q[s2][a]`). Adapted from `code examples/q_learning.py`.
  Asserts `kind` is one of the two (Room 3's "run SARSA twice" trap). `stats` gained
  a `caught` array (all-False on a guard-less grid). **Never bootstraps off a
  terminal** — same rule as Room 3.

> **⚠️ Claims that died to measurement in this room — three of mine, in one
> sitting. The pattern is always a metric that looked reasonable.**
> 1. **"A coin worth 1 flips the route on 19/20 configs."** False. The metric counted
>    *any* ledge cell on the path, and the route clips `(8,1)` incidentally leaving
>    the start. Coin 0 and coin 1 produced **byte-identical paths** at `V = 45.55`.
>    The honest criterion is **coins collected** — they sit mid-ledge and cannot be
>    touched by accident.
> 2. **"5,600 states is too many for tabular TD."** False, and nearly written in as a
>    finding. A *slow guard* at the same 5,600 states holds **92–95%** coverage,
>    while *two coins* at 5,600 collapses to **37–49%**. It was never the count — it
>    is which dimension you add (see the one-coin note).
> 3. **"Median visits per (s,a) is 2–4, so nothing is converged."** Misleading. Median
>    visits barely moves with the budget (4 → 5 → 5 across a 10× increase) while
>    SARSA's score climbs 59 → 80 → 84, because visits **concentrate on the learned
>    path** and the median just measures off-distribution states that do not matter.
>    A whole-table statistic cannot judge a policy.
>
> Add nothing to this section that has not survived a probe designed to kill it.
> `docs/UI_STRUCTURE.md`'s rule applies with force here: **suspiciously tidy
> agreement is a bug, not a finding.**



---

### Room 5: Deep Q-Learning (Continuous Arena, One Chasing Enemy)

> **Task redesigned 2026-07-20 (user: "simple yet challenging — a moving enemy; if he
> catches you the episode ends with −100").** The previous spec — three patrol guards,
> two static walls, low-friction momentum — was built, then **stashed and reset**
> (`git stash@{0}` "room5 wip"). This is a deliberate strip-down to the room's essence:
> an **empty** arena, **one enemy that chases the agent**, and **direct, inertia-free
> movement**. **Build status (2026-07-20): BUILT end-to-end and verified.** `core/chase_arena.py`
> (`gymnasium.Env`, passes `check_env`, `code examples/dql` 5-tuple API); `algorithms/deep_q.py`
> (Double-DQN training, adapted from `code examples/dql/dqn.py`); `rooms/room5_dqn.py` (the UI),
> registered in `streamlit_app.py`; `torch`/`gymnasium` added to `requirements.txt`. AppTest
> passes the full flow (train → scrubber → Play, no exceptions). The enemy-speed band **and**
> the learned escape rate are measured (below).
>
> **✅ THE DQN FINDING — Double DQN is what makes the room work; a single-net DQN collapses.**
> Traced (not assumed): a vanilla DQN on this env **diverged and went state-blind** — greedy
> policies loitered at the start corner with `max Q ≈ 6` in reward-units where the best return
> is ~2 (≈3× overestimation), escaping **0–35%**, *below* the 51% beeline, and **bimodal** across
> seeds (a run either learned or collapsed to 0%). Two standard, faithful fixes cure it together:
> **(1) Double DQN** (select the next action with the online net, evaluate with the target net)
> and **(2) reward scaling to the network** (the env keeps ±100 for the scoreboard; the learner
> trains on ±1). With both, plus shaping_coef 5 and Adam 3e-4: **speed 0.75 → 93% escape over 3
> seeds (100/81/99), no collapse** — right at the scripted-skilled ceiling of 95%, capturing the
> full 51%→93% gap. This mirrors the old maze build's lesson ("lr and reward scale decide it"),
> now pinned to the specific cure. Room defaults: 800 episodes, decaying ε (1.0→0.05, 0.995),
> Double DQN on, reward_scale 0.01.

* **Task Description:** Cross an empty continuous 10×10 m room from the bottom-left corner to the top-right exit (a 1×1 m square) while up to three enemies hunt you across the open floor. Touching any enemy ends the episode at −100.

> **⚠️ Room 5 drops the ice theme — deliberate user decision.** §1's unifying rule models
> Rooms 5–6's slipperiness as low friction / inertia. Room 5 now uses **direct movement**
> (the agent lands exactly where it aims; no momentum), chosen for the simplest possible
> redesign. So "slippery surfaces connect all six rooms" holds for Rooms 1–4 and 6 only,
> and Room 5 is the stated exception. Momentum is the single physics knob to re-add if the
> theme is ever wanted back — the enemy and reward machinery are independent of it.

* **State & Action Space:** Continuous state `[x, y]` + per enemy `[eₓ−x, e_y−y]` — the agent's position plus each **enemy's position relative to the agent** → `obs_dim = 2 + 2·n_enemies` (**2** with no enemies up to **8** with three). No velocity components: movement is inertia-free, so the action *is* the displacement and there is no momentum to carry in the state. **9 discrete actions** assign a displacement $(dx, dy) \in \{-1,0,1\}$, **normalised so every non-zero move travels exactly 1 m** (diagonals therefore ≈0.707 m per axis — flat *speed*, not flat components). **One decision = one metre**; a corner-to-corner run is ~14 steps.

* **The enemy is the whole room.** One marker **spawns at a random position each episode** (central region, kept ≥3 m clear of the agent's corner so it is never an instant catch) and, each step, moves a fixed fraction of the agent's speed **straight toward the agent's current position** (greedy pursuit). The agent cannot simply beeline: half the time the enemy sits across the direct line, so the agent must **arc around** it — exactly the behaviour the relative-enemy input exists to let the network learn. The enemy being **slower than the agent** is what keeps the room winnable: once the agent gets the enemy behind it, a slower pursuer can no longer close before the exit.

> **Why the enemy spawns RANDOMLY (not at a fixed point).** With a fixed enemy the whole
> env is deterministic, so any policy escapes 0% or 100% — no band to measure, and nothing
> for the network to generalise (it would memorise one trajectory). A random spawn makes
> "escape rate" a smooth number **and forces the network to actually read the relative-enemy
> input** rather than memorise a path. Agent start and exit stay fixed (the corner-to-corner
> run is the lesson, as in Rooms 3–4); only the enemy moves.

> **✅ MEASURED 2026-07-20 — the speed band is real, not hoped-for (5 placement seeds ×
> 400–500 random spawns, `scratchpad/measure_band.py`).** Escape rate of a **naive beeline**
> (ignore the enemy) vs a **scripted evasive** ceiling (attract-to-exit + repel-from-enemy
> potential field): **speed 0.75 → beeline 51% ± 1, skilled 95% ± 1, a 44-point gap** — the
> largest in the sweep, and the learnable advantage the DQN exists to capture. The gap is a
> hump: it falls below ~0.65 (enemy too slow — beeline creeps up) and above ~0.85 (even
> skilled play drops — 73% at 0.90, 58% at equal speed 1.0, the old "bimodal" ceiling). So
> the **enemy-speed slider is capped below 1.0, range `0.50`–`0.95`, default `0.75`** — the
> sweet spot. This is the room's guarantee stated as a number: at the default, a good policy
> escapes ~95% where ignoring the enemy escapes ~50%.
  * **Catch = contact:** the episode ends at −100 the instant `‖agent − enemy‖ < catch_radius` (≈0.5 m). Resolve the catch **after both have moved**, and test it against the **swept** agent segment too, so the pair cannot tunnel past each other in a single 1 m step (verify).

> **The old "greedy chaser is BIMODAL" worry was resolved by the random spawn.** The
> earlier build saw an enemy at equal speed *always* catch and one notch slower *never*
> catch — but that was a **fixed, deterministic** setup, where outcome is all-or-nothing.
> Randomising the spawn and measuring (above) shows a wide, smooth band with a 44-point
> naive-vs-skilled gap at 0.75. The chaser is kept as-is; no reaction lag or turn-rate cap
> is needed (both were considered and are **not** used — they would have grown the
> observation past 4 dims). Do not "restore" patrols.
* **The agent's 9 moves** — the 8 compass directions plus a stay-put, indexed in this fixed order (the network's output layer is indexed by it, so it must never be reordered):

  | # | $(V_x, V_y)$ | direction | # | $(V_x, V_y)$ | direction | # | $(V_x, V_y)$ | direction |
  |---|---|---|---|---|---|---|---|---|
  | 0 | $(-1,-1)$ | ↙ down-left | 3 | $(0,-1)$ | ↓ down | 6 | $(1,-1)$ | ↘ down-right |
  | 1 | $(-1,0)$ | ← left | 4 | $(0,0)$ | · stay put | 7 | $(1,0)$ | → right |
  | 2 | $(-1,1)$ | ↖ up-left | 5 | $(0,1)$ | ↑ up | 8 | $(1,1)$ | ↗ up-right |

  The four cardinal moves travel 1 m along one axis; the four diagonals travel 1 m total (≈0.707 m per axis); action 4 holds position (0 m) — mostly useless against a pursuer, but kept so the action space matches Room 6. There are **no walls**; only the arena boundary clamps a move. (The moves are a displacement, not a velocity — with no momentum the two coincide; the $(V_x,V_y)$ labels above read as "metres this step".)
* **Rewards (drive the curves below):** exit **+100**, caught **−100**, plus a potential-based shaping term toward the exit. With no walls the path to the exit is a straight line, so the shaping potential is just the **Euclidean** distance to the exit — the through-the-walls geodesic machinery the old spec needed is gone. Goal reward is fixed at +100 (no slider).

> **✅ A timeout penalty in the LEARNING signal was requested, measured, and REJECTED —
> Room 3's lesson reproduced here (2026-07-20, user: "make timeout −100, force the agent to
> exit").** There is nothing to force: at the defaults the trained agent **already times out
> 0%** of the time (escape 93%, caught 6%). Adding −100 on timeout *to the learner* made it
> **worse** — timeouts rose 0%→5% and escape fell 93%→90%, exactly the non-Markov poison
> §2 documents (elapsed time `t` is not in the 4-D state, so the penalty lands on whatever
> *position* the clock stopped in). A **step cost** — the Markov alternative — was worse
> still: it makes the agent rush the exit and run into the enemy (caught 6%→13–34%, escape
> down to 56–73%). So the −100 for a loss stays **on the scoreboard only** (`LOSS_SCORE`,
> shown on a timed-out ▶️ Play, mirroring the +100 exit and Rooms 2–4); the learner never
> sees it. Same conclusion as Rooms 1 and 3: punish dithering, if at all, never through a
> timeout the state cannot see.

**On-page controls (every widget carries a `help=` tooltip):**

* **🎮 Environment** (Row-1 board panel):
  * **Enemy speed (× yours)** — slider `0.50`–`0.95`, default **`0.75`** (MEASURED — peak naive-vs-skilled gap; see the band above). Capped below 1.0 because at equal speed even a good policy escapes only ~58%. Renamed from "Patrol speed" because the enemy now chases rather than sweeps. (No 🎲 Regenerate — geometry is fixed, as in Rooms 3–4. No goal-reward slider.)
  * **Max steps per episode** — select `{40, 60, 80, 120}`, default **`60`**.
  * **Enemies — three on/off toggles, 0–3 active** (default just the Chaser), added
    2026-07-20 (user: "add a third enemy with different algorithm, allow the user to turn
    each on or off, 0–3 possible"). Each active enemy adds two inputs (`obs_dim = 2 + 2n`,
    2→8) — Room 5's cheap echo of Room 4's "each coin doubles the tabular table": another
    hunter is **two more floats**, not a state explosion. **Three different behaviours so
    they never move as one**, each a function of *observed* positions only (Markov):
    🔴 **Chaser** (pure pursuit, head-on), 🟠 **Flanker** (heading rotated +45°, curves in
    from a side), 🟣 **Ambusher** (rotated −72°, sweeps in almost side-on from the other
    side), plus **mutual repulsion** so they split to different sides instead of stacking.
    Rejected (measured): an exit-guarding interceptor (camps the one cell the agent must
    reach → escape ~1–7%, unwinnable) and lead pursuit (needs the agent's velocity, not in
    the obs → non-Markov). **Difficulty scales steeply with count** (measured escape):
    **1 enemy ~95%, 2 ~93% (0.75, 800 ep), 3 ~70% (0.55, 1500 ep — needs the lower speed and
    more episodes; ~17% at 0.75/800)**. The scripted-evader ceiling for 3 is ~50–64%, so 3
    is genuinely hard but winnable — the About text and the speed tooltip tell the user to
    drop the speed and raise episodes for three.
  * **Randomize enemy positions each episode** — checkbox (default **on**), added 2026-07-20
    (user; revised same day from an agent-start toggle to an *enemy* toggle — "agent starts at
    the same point but the enemies at random locations when checked"). The **agent always
    starts at the corner** (the fixed lesson). **On:** enemies spawn randomly each episode —
    what forces the net to read the relative-enemy inputs and generalise, and what makes
    escape-rate a smooth number. **Off:** enemies sit at fixed spawns (`FIXED_ENEMY_SPAWNS`)
    and, with the agent fixed too, the episode is **deterministic** — a warm-up where the net
    only has to solve one layout.

> **✅ MEASURED (speed 0.75, 800 ep).** Random enemies (the default): **1 enemy 89%, 2 enemies
> 86%** escape — two enemies barely dents it, because the extra threat is only two more input
> floats, not a state explosion (the room's point, and the deliberate contrast with Room 4's
> "a second coin doubles the table"). Fixed enemies (deterministic warm-up): **winnable** —
> the 1-enemy layout escapes in 18 steps, and the flanking 2-enemy layout `[(3.5,6.5),
> (6.5,3.5)]` is solved by the DQN at 800 episodes (17 steps) even though the scripted evader
> is caught by that pincer. None of the four combinations is impossible.
* **🧠 Deep Q-Network** (Row-2 algorithm section), laid out in four columns + an ε row. ⚠️ **These defaults were tuned for the old spec (16-dim obs, walls, three guards) — re-check them on the simpler MDP.** `obs_dim` drops 16→4 and the reward landscape is smoother, so the network may converge in **fewer** episodes; the geodesic wall-cell shaping bug that dominated the old build's fate no longer exists (no walls), but the learning rate is still the first thing to sweep.
  * **Training episodes** — slider `100`–`1500`, default **`500`**.
  * **Discount γ** — slider `0.50`–`0.99`, default **`0.99`** (widened down to 0.50 on user request 2026-07-20; low γ is short-sighted and can fail to value the distant exit).
  * **Adam learning rate** — select `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`, default **`3e-4`**.
  * **Batch size** — select `{16, 32, 64, 128}`, default **`64`**.
  * **Gradient step every N ticks** (`train_freq`) — select `{1, 2, 4, 8}`, default **`4`**.
  * **Replay buffer** — select `{1k, 5k, 10k, 50k, 100k}`, default **`50k`**.
  * **Target update (steps)** — select `{100, 250, 500, 1k, 2k}`, default **`500`**.
  * **Exploration ε** — selector **Decaying (default) / Constant**, matching Rooms 2–4 and sharing `monte_carlo.epsilon_at`; on its own full-width row. Decaying exposes **ε start** (`0.1`–`1.0`, def `1.0`), **ε minimum** (`0.0`–`0.5`, def `0.05`), **ε decay rate** (`0.980`–`0.9999`, def `0.99`); Constant exposes a single ε slider.
  * **No random-seed control** — each 🚀 Train draws a fresh seed, so re-training a poor run gives a different outcome.
* **🚀 Train** trains synchronously behind an `st.spinner` (seconds at these defaults — no background thread), then gates all results until the `sig` of the current setup matches.

**Board (continuous arena, not a cell grid):** a square Plotly figure over `x, y ∈ [0, 10]` m (`scaleanchor` + `constrain="domain"` on **both** axes so metres stay square and the arena fills the frame).

* **Static layer:** 🤖 start (bottom-left), 🏁 exit (green **1×1 m square**, top-right — enlarged from a 0.5 m circle on user request 2026-07-20; reached = Chebyshev `|x−Eₓ|,|y−E_y| < ½`), and **0–3 enemies** (🔴 chaser / 🟠 flanker / 🟣 ambusher, each fatal on contact, colour-coded by behaviour) at their current positions. **No walls.** ⚠️ These are Plotly **shapes/markers drawn with `layer="above"`** — `layer="below"` means below *traces*, and the value field is a Heatmap trace, so anything "below" is painted over and vanishes.
* **⭐ Value layer:** the trained network sampled as $\max_a Q(x, y, \cdot)$ on a **50×50 grid** and drawn as an `RdBu` heatmap, `zmid=0` (high value **blue**). This is the room's visual argument for function approximation — a value field that exists *between* the sample points, which no tabular room can show. ⚠️ **The value now depends on the enemy's position** (`obs = [x, y, eₓ−x, e_y−y]`), which a 2-D board cannot show all at once: sample it **holding the enemy at its currently-drawn position** and say so in the caption — the heatmap is a **slice** of a 4-D function, not the whole thing. (The old fixed-guard version dodged this; the chaser makes it explicit, which is fair to show.) Toggled by a **"Show the network's value field"** checkbox.
* **▶️ Play Episode** animates one greedy rollout, **one metre per frame**, with the enemies drawn at their true per-frame positions (the trajectory records them); the agent marker turns **red** on a catch. Reports the outcome (✅ escaped / 🔴 caught / ⏱️ timed out), step count, and return (a **flat −100 for any loss**, mirroring the +100 exit; the raw is shown in a caption). Play has **its own "Randomize enemy positions" checkbox** (user 2026-07-20), independent of the training one, so you can train on a fixed layout and play random (does it generalise?), the reverse, or match them.
* A **checkpoint scrubber** ("view episode N") replays the greedy rollout captured at intervals during training, so the policy can be watched improving.

**KPI Metrics** (`st.metric` row, for the trained network):

* **Escape rate (last 100 episodes)** — share of recent training episodes that reached the exit.
* **🔴 Caught** — training episodes ended by an enemy.
* **⏱️ Timed out** — training episodes that ran out of steps (a loss), added on user request 2026-07-20. The −100 for a loss shows on the scoreboard and the (scored) returns curve but is **never** in the learning signal — a timeout depends on elapsed steps the network can't see, so penalising it there poisons positions (measured: it *increases* timeouts). Same rule as Rooms 1/3.
* **Mean steps to exit** — averaged over successful episodes (one step = one metre).
* **Mean predicted Q** — the network's own late-training value estimate (there is no exact answer to check it against in a continuous room).

**Graphs (full-width analytics row below the board):**

* **Episode return (scored)** — per-episode return scatter with a moving-average line; every loss (caught **or** timed out) is floored to −100 in the DISPLAY, mirroring the Play scoreboard (Rooms 2–4 convention). Pure display transform — the learner updates off per-step rewards, never this.
* **Network training** — TD loss (Huber) and mean predicted Q on a shared dual-axis plot, over gradient steps.
* **Cumulative outcomes** — running totals of escaped / caught / timed-out.
* **Exploration rate** — the ε schedule over episodes (Room 2's convention).



---

### Room 6: Advanced DQL (Dynamic Obstacles & Radar Raycasting)

* **Task Description:** Escape a hazardous chamber filled with static and moving obstacles by relying exclusively on forward-facing radar sensors.
* **State & Action Space:** Continuous state vector $(X, Y, V_x, V_y)$ + $K$ radar distance readings. Discrete acceleration actions.
* **On-page Controls (with tooltips):**
* **Perception & Setup:** ⭐ **Agent Radar Range ($X$ meters)** (`1.0` to `10.0` meters), radar ray count $K$ (`4`, `8`, `16`, or `32`), static obstacle count (`2` to `12`), dynamic obstacle speed, ice friction $\mu$. **Goal Reward: +100**.
* **Neural Network:** Identical network hyperparameters to Room 5 + training episode slider (`500` to `5,000`).


* **KPI Metrics:** Obstacle collision rate per episode, **Generalization Score** (success rate on unseen rooms), average near-miss braking distance.
* **Visualizations & Interactive Features:**
* **Live Radar Visualization:** Animate the agent with $K$ transparent ray lines that turn **red** the instant an obstacle enters the $X$-meter detection threshold.
* Trajectory heatmap aggregating the paths of the last 50 episodes to show policy stabilization.
* 🎲 **"Generate Random Room" Button:** Instantly clears and generates a novel obstacle layout to test if the neural network learned generalized collision avoidance rather than room memorization.



---

## 4. Algorithm & Room Summary

| Room | Algorithm | State Space | Environmental Knowledge | Behavioral Policy | Key Challenge |
| --- | --- | --- | --- | --- | --- |
| **1** | Dynamic Programming | Discrete (10x10) | Full Model (100%) | Mathematically Optimal | Calculating expectations against slip probability |
| **2** | Monte Carlo | Discrete (10x10) | Model-Free (Episodes) | High Variance / Empirical | Escaping teleport portals and infinite loops |
| **3** | SARSA (On-Policy) | Discrete (10x10 **× shield flag**) | Model-Free (Step-by-Step) | **Conservative & Safe** | Crossing an icy ledge over a fatal abyss |
| **4** | Q-Learning (Off-Policy) | Discrete (10x10 **× guard phase × coin flag**) | Model-Free (Step-by-Step) | **Over-optimistic about risk** | Timing a patrol, or braving the ledge for a coin |
| **5** | Deep Q-Learning | Continuous (10x10m, +enemy-relative) | Model-Free (NN Approx) | Evasive & Reactive | Outrunning a chasing enemy to the exit |
| **6** | Advanced DQL | Continuous + Radar | Limited Sensor Rays | **Generalizable Avoidance** | Navigating dynamic obstacles with radar range $X$ |

---

## 5. Implementation Guidelines & Tech Stack

* **Algorithm Source Code:** All training logic, replay buffers, Bellman updates, and network architectures must be imported directly from the **`code examples`** repository folder.
* **UI Tooltips:** Every Streamlit input widget (sliders, selectboxes, number inputs) must include the `help` parameter containing a 1–2 sentence explanation of the variable.
* **Core Libraries:**
* **Web Framework:** `streamlit` (utilizing `st.session_state` for training persistence, `st.metric` for KPIs, and `st.tabs` for analytics).
* **Math & Grids:** `numpy`, `scipy`.
* **Deep Learning (Rooms 5–6):** `torch` (PyTorch) imported from the code examples for defining the Deep Q-Networks.
* **Rendering:** `plotly.graph_objects` for grid heatmaps and dual-line charts; custom HTML5 Canvas components or Plotly animations for continuous physics tracking.