# Plan.md – Interactive Reinforcement Learning Escape Room

## 1. Overview and Core Principles

This document outlines the design and architecture for a Streamlit-based interactive Web application. The platform visualizes the evolution of Reinforcement Learning (RL) algorithms through a 6-stage "Escape Room" game with progressively increasing difficulty.

### Core Architecture & UI Rules

* **Unifying Physical Theme:** Slippery surfaces (Ice physics) connect all 6 rooms. In Rooms 1–4 (discrete grid), this is modeled as stochastic slip probabilities. In Rooms 5–6 (continuous space), it is modeled as low friction, inertia, and momentum.
* **Minimal Text UI:** The user interface must remain clean and visual. Do not include lengthy theoretical or algorithmic explanations on the screen.
* **Task-Only Descriptions:** Each room will display only a brief 1–2 sentence description of the **task/mission** (e.g., navigating hazards, dodging guards) at the top of the main view.
* **Tooltip Parameter Explanations:** All controls, hyperparameters, and environment settings (rendered **on the page**, not the sidebar) must use Streamlit's native question-mark tooltip feature (using the `help="..."` argument inside widgets) to explain their mathematical or mechanical purpose.
* **Standardized Goal Reward:** To maintain consistency across all environments, reaching the final goal in **every room** always grants a fixed reward of **+100**.
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

---

## 3. Room-by-Room Technical Specifications

### Room 1: Dynamic Programming (Bellman Equations)

* **Task Description:** Navigate a static icy labyrinth from the starting tile to the exit while factoring in the risk of slipping.
* **State & Action Space:** 10x10 discrete grid (100 states), 4 directional actions. The board has three special cell types: **🧱 blocked** cells (walls the agent cannot enter), **🟦 slippery ice** cells (moves may slip perpendicular), and **🟥 negative-reward** cells (*passable* — the agent can cross them but pays a negative reward each entry; only the goal is terminal). A legend under the board identifies each type.
* **On-page Controls (with tooltips):**
* **Environment & Physics:** Count sliders for **blocked cells** (`0`–`20`), **slippery cells** (`0`–`40`), and **negative-reward cells** (`0`–`15`); **slip probability** (`0.0`–`0.8`, applied only on ice cells); **negative reward value** (`-100`–`-1`, factored into V); and a **🎲 Regenerate layout** button. Placement is random from a fixed seed and always keeps the start→exit reachable; Regenerate reshuffles the seed. **Goal Reward: +100**.
* **Algorithm (its own row):** DP method selector — **Value Iteration** vs. **Policy Iteration** — and discount factor ($\gamma$) between `0.5` and `0.99`, followed by the **🚀 Train** button. *(Note: Convergence threshold $\theta$ is hardcoded in the background at `1e-3`).* The board layout only changes on **🎲 Regenerate** (or first load), never while the count sliders are dragged.


* **KPI Metrics:** Iterations to convergence, maximum delta in the final iteration, expected starting-state value V(S), and **expected steps to exit** (model-exact mean moves from start under the optimal policy). Beside the Play controls, a live **success-within-cap** metric shows the model-exact probability the viewed policy reaches the exit within the current max-steps cap (updates as the slider moves).
* **Visualizations:**
* Live value heatmap overlay on the grid showing reward diffusion from the exit backward, with slippery ice and negative-reward cells highlighted and walls masked.
* Log-scale convergence curve showing delta dropping to zero, with a marker for the iteration currently being viewed.
* **Training-results board + iteration scrubber:** The results section shows the board again (with ▶️ Play controls beside it) and a "view iteration" slider that replays the value function and greedy policy as they were at each iteration — revealing the contrast between Value Iteration (intermediate policies are poor until late) and Policy Iteration (intermediate policies are already strong). The convergence curve sits in its own row below.
* **▶️ Play Episode:** A "max steps per episode" slider caps the rollout; the room animates one episode of the *currently-viewed* policy across the stochastic ice (where a slip can push it off course or into a penalty cell) and reports its **discounted return G** (computed like V), step count, and success/timeout.


* **Implementation Note:** Utilize the Dynamic Programming reference scripts in the `code examples` folder.

---

### Room 2: Monte Carlo ($\epsilon$-greedy)

* **Task Description:** Traverse long icy corridors and avoid portal traps that instantly teleport you back to the starting point.
* **State & Action Space:** 10x10 discrete grid, 4 directional actions.
* **On-page Controls (with tooltips):**
* **Environment:** Slip probability (`0.0` to `0.8`), portal trap count (`0` to `5`), max steps per episode (`50` to `500`). **Goal Reward: +100**.
* **Algorithm:** Training episodes (`100` to `5,000`), exploration mode selector: **Constant $\epsilon$** (single slider) vs. **Decaying $\epsilon$** (initial, minimum, and decay rate inputs). *(Note: Discount factor is hardcoded to `1.0`).*


* **KPI Metrics:** Success rate over the last 100 episodes, moving average reward, current epsilon value.
* **Visualizations:**
* Episode reward scatter plot with a 50-episode moving average trendline.
* Steps-to-goal curve demonstrating the transition from random exploration to efficient routing.
* Test mode execution ($\epsilon = 0$) animating the learned policy.



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