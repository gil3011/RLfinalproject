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

* **Task Description:** Carefully cross a narrow, slippery ice bridge suspended over a deep abyss without falling off the edge.
* **State & Action Space:** 10x10 discrete grid (Cliff Walking layout), 4 directional actions.
* **On-page Controls (with tooltips):**
* **Environment:** Slip probability (`0.0` to `0.8`), cliff fall penalty (`-10` to `-100` plus step reset). **Goal Reward: +100**.
* **Algorithm:** Learning rate ($\alpha$) between `0.01` and `0.5`, training episodes, epsilon mode selector.


* **KPI Metrics:** Total cumulative cliff falls during training, stabilized average reward, safe path step length.
* **Visualizations:**
* Cumulative cliff falls chart showing leveling-off as safety rules are learned.
* Real-time Q-table overlay showing directional arrows inside each grid cell, highlighting SARSA's conservative tendency to point away from the hazard.
* **▶️ Play Episode:** Animate the learned policy crossing the bridge, showing SARSA hugging the safe interior away from the cliff edge.



---

### Room 4: Q-Learning (Off-Policy TD)

* **Task Description:** Cross the hazardous bridge while dodging a moving patrol guard and collecting optional bonus coins placed near the abyss.
* **State & Action Space:** 10x10 discrete grid + dynamic guard position, 4 directional actions.
* **On-page Controls (with tooltips):**
* **Environment:** Slip probability, guard patrol speed (Slow/Normal/Fast), coin bonus value (`0.0` to `2.0`). **Goal Reward: +100**.
* **Algorithm:** Identical hyperparameters to Room 3 ($\alpha$, episodes, exploration rate) to ensure a controlled side-by-side comparison.


* **KPI Metrics:** Maximum reward achieved on the final trajectory, total guard collisions.
* **Visualizations:**
* Comparative reward chart plotting Q-Learning vs. SARSA on the same axes.
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