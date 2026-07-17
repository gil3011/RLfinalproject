# Plan.md – Interactive Reinforcement Learning Escape Room

## 1. Overview and Core Principles

This document outlines the design and architecture for a Streamlit-based interactive Web application. The platform visualizes the evolution of Reinforcement Learning (RL) algorithms through a 6-stage "Escape Room" game with progressively increasing difficulty.

### Core Architecture & UI Rules

* **Unifying Physical Theme:** Slippery surfaces (Ice physics) connect all 6 rooms. In Rooms 1–4 (discrete grid), this is modeled as stochastic slip probabilities. In Rooms 5–6 (continuous space), it is modeled as low friction, inertia, and momentum.
* **Minimal Text UI:** The user interface must remain clean and visual. Do not include lengthy theoretical or algorithmic explanations on the screen.
* **Task-Only Descriptions:** Each room will display only a brief 1–2 sentence description of the **task/mission** (e.g., navigating hazards, dodging guards) at the top of the main view.
* **Tooltip Parameter Explanations:** All controls, hyperparameters, and environment settings (rendered **on the page**, not the sidebar) must use Streamlit's native question-mark tooltip feature (using the `help="..."` argument inside widgets) to explain their mathematical or mechanical purpose.
* **Goal Reward — default +100, adjustable:** Reaching the exit is the only positive reward in every room. It **defaults to +100** everywhere so rooms stay comparable, but Rooms 1–2 expose it as a slider (`10`–`1000`) because it is the scale everything else is measured against — the negative cells, the slip risk, and the discounting all only mean something relative to it. *(This relaxes the original "fixed +100 in every room" rule. Rooms 3–6 should default to 100 and only expose the slider if the room actually teaches something with it.)*
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

* **Scoring a timeout (Room 2 only).** Raw G ranks giving up *above* escaping the
  hard way: a run that wanders and times out scores 0, while an escape that crossed
  penalty cells can score negative. Room 2 therefore adds a **−100 timeout penalty**
  to the *reported* return (`core.episode.scored_return`), mirroring the +100 goal,
  so not finishing always ranks last. This is a **scoreboard** number only — it
  deliberately breaks `G ≈ V(S)`, and nothing doing maths (V, Q, MC's updates, the
  curves, the DP benchmark) may use it. Room 1 does **not** do this: its agent is DP
  and never sees a return at all, so a penalty on G there would be pure decoration.

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

* **KPI Metrics:** Total cumulative cliff falls during training, success rate and mean return over the last 100 episodes, and the **start-state value V(S)** at the viewed checkpoint. As in Room 2, report the learned policy's **true** value from `policy_value()` beside SARSA's own estimate — the two disagree for the same reason (Q is the value of the *ε-greedy* agent, not of the greedy policy ▶️ Play runs).
* **Visualizations:**
* **Cumulative cliff falls chart.** It levels off but **never reaches zero** — measured 7–25 falls in the last 100 episodes across ε settings. Slip and ε both keep pushing the agent in; that residual is honest and worth not hiding.
* Q-table arrow overlay showing SARSA's conservative tendency to point away from the hazard, plus the value heatmap. **Cliff cells are terminal, so they are masked (`z = np.nan`, icon only)** exactly like the goal — the agent never acts from one, so it has no V.
* **DP benchmark row**, mirroring Room 2: SARSA's estimate, the policy's true `V^π(S)`, and `V*(S)` side by side. **Room 3 does NOT train Q-learning** — that contrast is Room 4's reveal (see the note there).
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

* **Task Description:** Cross the hazardous bridge while dodging a moving patrol guard and collecting optional bonus coins placed near the abyss.
* **State & Action Space:** 10x10 discrete grid + dynamic guard position, 4 directional actions.
* **On-page Controls (with tooltips):**
* **Environment:** Slip probability, guard patrol speed (Slow/Normal/Fast), coin bonus value (`0.0` to `2.0`). **Goal Reward: +100**.
* **Algorithm:** Identical hyperparameters to Room 3 ($\alpha$, episodes, exploration rate) to ensure a controlled side-by-side comparison.


* **KPI Metrics:** Maximum reward achieved on the final trajectory, total guard collisions.

> **Room 4 must train BOTH algorithms on its OWN board.** The comparison below cannot import Room 3's curves: Room 4's environment (moving guard + coins, and a state space extended by the guard position) is not Room 3's, so plotting the two rooms' returns "on the same axes" would compare different MDPs and mean nothing. Run SARSA as a baseline *inside* Room 4, on Room 4's board, with identical hyperparameters — which is also what makes the path-comparison overlay a controlled experiment rather than a coincidence. This is why Room 3 stays pure SARSA and leaves the contrast here. The contrast is confirmed to exist: on Room 3's cliff at slip = 0, Q-learning hugs the ledge and reaches `V^π` = `V*` = 59.87 exactly while SARSA detours and settles at 29.20. **Note it weakens as slip rises** (see Room 3's slip note) — so if Room 4's board is slippery, budget for the contrast being muted, and see Room 3's warning about asserting that the two learners actually diverge.

* **Visualizations:**
* Comparative reward chart plotting Q-Learning vs. SARSA on the same axes, both trained here.
* **Path Comparison Overlay:** Visualizing both final paths on the game board simultaneously—Q-Learning's risky, direct edge path vs. SARSA's safe, detour path.
* **▶️ Play Episode:** Animate the learned policy taking its aggressive edge route past the guard, contrasting with SARSA's cautious detour.



---

### Room 5: Deep Q-Learning (Continuous Physics & Momentum)

* **Task Description:** Slide across a continuous 10x10 meter icy arena while counteracting strong side-wind currents pushing you off course.
* **State & Action Space:** Continuous state vector $(X, Y, V_x, V_y)$, discrete acceleration actions $\Delta V \in \{-1, 0, 1\}$ updated every 0.02 seconds.
* **On-page Controls (with tooltips):**
* **Physics:** Ice friction coefficient $\mu$ (`0.80` for normal floor to `0.98` for extreme ice), wind zone force (`-3.0` to `+3.0` $m/s^2$), goal target radius (`0.3` to `1.0` meters). **Goal Reward: +100**.
* **Neural Network:** Adam learning rate (`1e-4` to `1e-2`), batch size (`16` to `128`), replay buffer size (`1,000` to `10,000`), target network update frequency.


* **KPI Metrics:** Average time to goal (seconds), average velocity, mean predicted Q-value.
* **Visualizations:**
* Neural network TD-error / loss curve over training steps.
* Smooth 60fps HTML5/Plotly animation showing real-time sliding trajectories and momentum correction against wind fields.



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
| **3** | SARSA (On-Policy) | Discrete (10x10) | Model-Free (Step-by-Step) | **Conservative & Safe** | Surviving a narrow bridge over a cliff |
| **4** | Q-Learning (Off-Policy) | Discrete (10x10) | Model-Free (Step-by-Step) | **Aggressive & Risky** | Dodging a moving guard for bonus coins |
| **5** | Deep Q-Learning | Continuous (10x10m) | Model-Free (NN Approx) | Smooth & Momentum-Aware | Counteracting low friction and wind zones |
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