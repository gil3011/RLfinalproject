# Best Training Parameters — RL Escape Room

This document records, for each room, the recommended **default environment**
(challenging but not the hardest the sliders allow) and the **best training
parameters** for that default, backed by measured sweeps. Each room also lists
guidance for *if you change the environment* — which knobs to move, and when a
different algorithm/setting becomes preferable.

Two kinds of rooms:

- **Planning rooms (Room 1, DP)** are *exact*: Value Iteration and Policy
  Iteration both converge to the **optimal** policy regardless of
  hyperparameters. "Best params" here means the settings that make the value
  function and policy *pedagogically legible* and fast to converge — not more
  accurate (they're already optimal).
- **Learning rooms (Rooms 2–6: MC, SARSA, Q-learning, DQN, Advanced DQL)** are
  where hyperparameters genuinely decide success or failure. Those sections
  carry real tuning sweeps.

All rooms share: goal reward **fixed at +100**, start `(9,0)`, goal `(0,9)`,
10×10 grid (Rooms 1–4).

### Shipped defaults vs. measured-best (read this first)

For **UI consistency**, the app ships a set of *uniform* defaults that differ
from the per-room measured optimum below. Each room's "Best default" table
reports what the sweeps found; the app loads the uniform values, and the user can
dial in the measured-best from the tooltips.

| Knob | Uniform shipped default | Notes |
|---|---|---|
| Discount γ | **0.90** (all rooms) | Measured-best varies: 0.95 (R1–R4, R6), 0.80 (R5). 0.90 is a fair compromise. |
| Exploration | **Decaying ε**, start 1.0, min 0.05, **decay 0.998** (all rooms) | Matches a ~1500–2000-episode budget. R3's measured-best is Constant ε=0.10 — switch to it via the selectbox. |
| Training episodes (R2–R4) | **2000** (range 500–10000) | Measured-best: R2 ≈3000, R4 ≥10000. R4 *under-trains* at 2000 (its state space is large). |
| Play-episode max steps (R1–R4) | **10–50 slider** | Enough for the tabular boards' path lengths. |

**Two places where the uniform default is knowingly suboptimal — raise them for
the best result:**
- **Room 3 exploration:** switch to **Constant ε = 0.10** (decaying-from-1.0 is
  the worst schedule on a terminal cliff, and on the current slip-0.40 default it
  collapses whole seeds to the flee-upward policy — see R3 below).
- **Room 4 episodes:** raise to **10000** — 2000 under-trains both learners and
  muddies the SARSA-vs-Q-learning contrast (see R4 below).

---

## Room 1 · Dynamic Programming

Icy grid navigation with slippery ice and passable penalty cells. DP uses the
full model to compute `V(s)` and the optimal policy.

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Blocked cells 🧱 | 0–20 | **10** | Walls; a start→goal path is always preserved (BFS-guaranteed). |
| Slippery cells 🟦 | 0–40 | **30** | Ice. **Density matters more than slip probability** (see below). |
| Slip probability | 0.0–0.8 | **0.35** | Chance of sliding perpendicular on ice. |
| Negative cells 🟥 | 0–15 | **8** | Passable penalty cells. |
| Negative reward | −10…−1 | **−6** | Penalty on entering a red cell. |
| Goal reward 🏁 | — | **100 (fixed)** | Slider **removed** — see change log. |

**Change log (this pass):**
- **Removed the Goal-reward slider (was 10–1000)** and hardcoded `goal=100`.
  Rationale: it violated the cross-room "+100 fixed" convention, and for DP it
  mostly rescales `V(s)` without changing the policy — the meaningful risk knob
  is the *ratio* of penalty to goal, already controlled by the negative-reward
  slider.
- **Raised ice default 15 → 30** and slip 0.20 → 0.35. Reason below.

**Why the ice default was raised (key finding):** with only ~15–20 ice cells the
optimal path routes *around* the ice entirely, so slip probability is inert —
E[steps] stays at the Manhattan distance (18) and V(start) is identical for
slip 0.0 through 0.8. Worse, the app's initial board (`seed=0`) was exactly such
a dead board. At **30 ice** the ice sits on the shortest path even on `seed=0`,
so the risk/reward tradeoff Room 1 is meant to teach actually appears.

### Training parameters

| Param | Options / Range | **Best default** |
|---|---|---|
| DP method | Value Iteration · Policy Iteration | **Value Iteration** |
| Discount γ | 0.50–0.99 | **0.95** |
| θ (convergence) | fixed 1e-3 | — |

### Best parameters — measured (default board, seed 0)

**γ sweep** (slip 0.35). Both methods reach the *same* optimal policy; success is
100% at every γ because the board is solvable — but the value signal and
risk-aware routing require a high γ:

| γ | VI iters | PI iters | V(start) | E[steps] | succ@50 |
|---|---|---|---|---|---|
| 0.50 | 14 | 8 | 0.00 | 18.9 | 100% |
| 0.70 | 21 | 9 | 0.17 | 18.9 | 100% |
| 0.80 | 25 | 9 | 1.76 | 18.9 | 100% |
| 0.90 | 26 | 8 | 14.3 | 18.9 | 100% |
| **0.95** | **31** | **9** | **37.5** | **18.9** | **100%** |
| 0.97 | 33 | 8 | 54.4 | 20.0 | 100% |
| 0.99 | 34 | 7 | 80.1 | 20.1 | 100% |

- **γ = 0.95 is the sweet spot.** Below ~0.8 the +100 goal is discounted to near
  zero over the ~18-step path, so `V(start) ≈ 0` — the value heatmap goes flat
  and uninformative even though the policy still escapes. Above 0.97 the agent
  starts over-detouring (E[steps] creeps up) and VI takes more sweeps.
- **VI vs PI:** identical optimal policy always. PI converges in far fewer
  *outer* iterations (7–9) but each runs a full inner policy-evaluation loop, so
  wall-clock is comparable — PI is actually *slower* at high γ (≈110 ms vs a few
  ms for VI at γ=0.99). **Default to Value Iteration** (simpler, faster here);
  Policy Iteration is the better teaching contrast, not a speed win.

**Slip sensitivity** (γ 0.95) — confirms the raised-ice default makes slip bite:

| slip | V(start) | E[steps] |
|---|---|---|
| 0.00 | 41.8 | 18.0 |
| 0.20 | 39.1 | 19.0 |
| 0.35 | 37.5 | 18.9 |
| 0.50 | 36.6 | 19.3 |
| 0.80 | 34.4 | 20.7 |

### If you change the environment

- **Fewer ice cells (< ~20):** slip becomes irrelevant (path routes around ice);
  the room degenerates to a trivial shortest-path problem. Keep ice ≥ 25 to keep
  the risk lesson alive, or lower `n_blocked` so ice is forced onto the corridor.
- **Very high slip (≥ 0.65) + harsh penalties (≤ −8):** the safe detour wins
  decisively; expect longer E[steps] and lower V(start), but still 100% success —
  DP is immune to the greedy-loop failure that bites the model-free rooms.
- **Lower γ intentionally (to demo myopia):** drop to ~0.7 to show the value
  function flattening; the policy still escapes, which is itself the lesson that
  *DP's optimal policy is far more γ-robust than its value function*.
- **Larger/denser boards or a finer θ:** prefer **Value Iteration** — Policy
  Iteration's inner evaluation loop gets expensive as γ→1. If you ever need the
  exact fixed point fast at γ≈0.99, VI with θ=1e-3 is the cheaper route here.

---

## Room 2 · Monte Carlo

On-policy first-visit MC control, ε-greedy. The board is randomly generated:
walls + ice + **portal traps** that teleport the agent back to the start (a
time penalty, discounted by γ — no point penalty). Model-free: the agent learns
only from sampled episodes. Benchmarked against Room 1's exact DP (V*).

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Blocked cells 🧱 | 0–30 | **10** | Corridors; start→goal path always preserved. |
| Slippery cells 🟦 | 0–40 | **20** | Ice. |
| Slip probability | 0.0–0.8 | **0.35** | |
| Portal traps 🌀 | 0–5 | **5** | Signature mechanic — teleport to start. Some may be dropped if they'd seal the exit. |
| Goal reward 🏁 | — | **100 (fixed)** | Slider **removed** (was 10–1000). |

**Change log:** removed goal-reward slider (fixed 100). Default board reworked to
**10 / 20 / 0.35 / 5** — a **more open** board (walls 22→10, ice 25→20) that
leans on the **portal traps** (raised to the slider max, 5) as the signature
challenge rather than dense walls. Effect: the board is *easier to navigate*, so
V\* rises (~35→41 at γ=0.95) and the MC-vs-DP value gap shrinks (~18%→9%), but the
three-way V\*/trueV/V_MC separation the room teaches survives cleanly. Negative
cells were removed in an earlier pass (harsh penalties starve MC of successful
episodes — see project memory). *(Earlier passes: 20/20/0.2/3 → 22/25/0.35/4 →
current.)*

### Training parameters

| Param | Range / Options | **Best default** |
|---|---|---|
| Discount γ | 0.50–0.99 | **0.95** |
| Training episodes | 100…5000 | **3000** |
| Max steps / episode | 50…500 | **300** |
| Exploration | Decaying · Constant ε | **Decaying** |
| ε start / min / decay | — | **1.0 / 0.05 / 0.999** |

### Best parameters — measured (default board 10/20/0.35/5, 4 layouts × 2 train seeds vs DP)

MC reaches **100% training success** on this board at any reasonable setting
(portals + generous step cap + ε-from-1.0 always eventually find the goal), and
the more open board makes convergence *easier* than the previous default. The
discriminating metric is the **value-quality gap** = how far the learned greedy
policy's *true* value (model-evaluated) sits below the exact optimal V*.

**γ (key knob), 2000 episodes:**

| γ | V* | learned trueV | V_MC (MC's own) | gap to V* |
|---|---|---|---|---|
| 0.90 | 16.0 | 14.0 | 8.2 | 13% |
| **0.95** | **41.0** | **37.3** | **25.2** | **9%** |
| 0.97 | 58.9 | 55.8 | 41.1 | 5% |

γ=0.95 chosen for **pedagogy, not raw quality**: V*, learned-trueV, and MC's
pessimistic V_MC are all clearly separated (41 / 37 / 25) — the whole point of the
room (MC *understates* its own policy *and* is suboptimal vs DP). The open board
shrinks the gap vs the old default (9% vs 18%) but keeps the three-way split
legible; at γ≥0.97 the gaps compress and the lesson gets subtle.

**Episodes** (γ=0.95, decay matched so ε hits floor near the end):

| episodes | trueV | success | worst-seed success | gap |
|---|---|---|---|---|
| 500 | 29.5 | 87% | 0% | 28% |
| 1000 | 35.0 | 100% | 100% | 15% |
| 2000 | 37.3 | 100% | 100% | 9% |
| **3000** | **38.3** | **100%** | **100%** | **7%** |

**ε schedule** (γ=0.95, 2000 episodes):

| schedule | trueV | gap |
|---|---|---|
| decay 0.998 | 37.3 | 9% |
| **decay 0.999** | **38.3** | **7%** |
| constant 0.10 | 33.3 | 19% |
| constant 0.30 | 37.1 | 9% |

- **500 episodes is the failure floor** — some seeds still collapse (worst-seed
  success = 0). 1000+ is reliable; quality keeps climbing toward 3000+.
- **Slower decay (0.999) beats faster (0.998)** at a fixed budget: keeping ε high
  across the whole run gives better Q-coverage. Match decay to the episode budget
  so ε reaches its floor near the end.
- **Constant ε is the instructive failure:** 0.10 gives the largest value gap
  (19% — too little exploration to polish the policy); 0.30 matches decaying on
  value but decaying still edges it and is safer.

### If you change the environment

- **More portals / higher slip:** raises variance and the value gap — bump
  **episodes to 5000** and use the slower **decay 0.999** (or lower) to keep
  exploring. Consider **γ ≥ 0.95** so the goal signal survives the longer,
  portal-lengthened paths.
- **Fewer episodes are forced (speed):** raise the decay *rate down* (faster
  decay, e.g. 0.995 for 1000 episodes) so ε still reaches its floor, and accept a
  larger gap — but never below ~1000 episodes on a board this size.
- **Harsh negative cells (if re-added):** MC starves (timeout return beats
  braving penalty → agent idles). Keep penalties mild (≥ −8) and slip ≥ 0.1, or
  switch to a TD room (Room 3/4) which bootstraps and tolerates it far better.
- **Deterministic board (slip 0):** MC greedy rollouts can loop; keep slip ≥ 0.1
  or rely on the ε floor to break loops.

---

## Room 3 · SARSA

On-policy TD control on a **cliff-walk**: start `(9,0)` and exit `(9,9)` sit at
the ends of the bottom row; the 8 cells between them are a **terminal abyss**
(−100). Because SARSA bootstraps off the action it *actually* takes next
(ε-greedy), it prices its own exploration risk into standing near the edge and
learns a **cautious detour** — the contrast with Q-learning (Room 4). Shields
grant permanent slip-immunity and expand the state to `(i, j, has_shield)`.

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Blocked cells 🧱 | 0–20 | **15** | Safe route always preserved. |
| Slippery cells 🟦 | 0–40 | **24** | Never on the abyss. |
| Slip probability | 0.0–0.8 | **0.40** | Raised to make the ledge genuinely lethal — but with the right schedule SARSA still handles it. |
| Shields 🛡️ | 0–2 | **1** | Slip-immunity pickup; doubles the state space. |
| Goal reward 🏁 | — | **100 (fixed)** | Slider **removed**; with the −100 abyss this is a 1:1 escape:die ratio. |
| Abyss reward | — | −100 (fixed) | |

**Change log:** removed goal-reward slider; board hardened to **15 / 24 / 0.40**
(walls 8→15, ice 20→24, slip 0.20→0.40). This pushes the room deep into the
high-slip regime the old default deliberately avoided — but the key finding is
that **the collapse is schedule-driven, not slip-driven**: at slip 0.40 with
*low constant ε* (0.05–0.10) SARSA still escapes ~96–99% (worst seed 94%), while
*decaying-from-1.0* now genuinely collapses seeds to the "flee upward forever"
policy (mean 87%, **worst seed 0%**). Higher slip simply widens the gap between
the right schedule (low constant ε) and the wrong one — sharpening the room's
whole lesson. *(Earlier passes: slip 0.10 → 0.20 → current 0.40.)*

### Training parameters

| Param | Range / Options | **Best default** |
|---|---|---|
| Learning rate α | 0.01–0.5 | **0.10** |
| Discount γ | 0.50–0.99 | **0.95** |
| Episodes | 500…10000 | **2000** |
| Max steps / episode | 100…500 | **200** |
| Exploration | **Constant ε** · Decaying | **Constant, ε = 0.10** |

### Best parameters — measured (default board 15/24/0.40, 4 layouts × 2 seeds vs DP, V*≈45.7)

The metrics that matter: **training falls** (episodes lost to the abyss),
**last-100 success**, **worst-seed success** (the collapse detector), and the
**learned policy's true value** vs V*. The gap to V* is *expected and desirable*
— it is SARSA's deliberate caution, not under-training.

**Exploration schedule is the dominant knob (α=0.10, γ=0.95, 2000 ep):**

| schedule | success | worst seed | falls | trueV | gap to V* |
|---|---|---|---|---|---|
| decaying 1.0→0.05 (0.998) | 87% | **0%** | 314 | 24.2 | 47% |
| constant ε = 0.05 | 99% | 96% | 64 | 37.9 | 17% |
| **constant ε = 0.10** | **96%** | **94%** | **100** | **36.2** | **21%** |
| constant ε = 0.15 | 97% | 90% | 137 | 32.4 | 29% |
| constant ε = 0.30 | 86% | 80% | 318 | 28.7 | 37% |

**Key insight — decaying-from-1.0 is the *worst* choice on a terminal cliff, and
the harder board makes it fatal.** Pure early exploration means the agent falls
constantly and learns the entire lower board is lethal, then flees upward; at
slip 0.40 this now collapses a seed *entirely* (worst-seed success 0%, mean 87%).
Low **constant** ε keeps it near the good path (few falls) while still pricing in
a fixed risk, so it stays cautious — and it does not collapse.

**ε is the caution dial:** ε=0.05 gives the best raw value (gap 17%) but SARSA
then barely differs from Q-learning; ε=0.10 keeps near-best value (gap 21%) *and*
visible caution; ε≥0.15 over-detours (falls climb, value falls). **0.10 remains
the chosen default** for the performance/pedagogy balance — but on this harder
board **ε=0.05 is the best-performing** and the safe floor if you want maximum
escape (see "if you change the environment").

**α (constant ε=0.10):** 0.05 / 0.10 / 0.20 all reach ~96–97% success; α=0.10
gives the best true value (36.2) and is the reliable pick. **Episodes:** 2000
suffices with constant ε; 5000 barely improves it (gap 21%→18%, but 2× the falls)
— constant-ε SARSA plateaus.

### If you change the environment

- **Even higher slip (≥0.5) or denser ice** (the default is already slip 0.40):
  raises fall risk further — **lower ε to 0.05** (the safe floor here) to avoid
  the flee-upward collapse, and/or add a **shield** so the agent can buy
  slip-immunity. Never use decaying ε in this regime — it collapses seeds. If it
  still collapses at ε=0.05, the board is past SARSA's comfort zone; drop slip
  back.
- **Want maximum escape performance (not the caution lesson):** ε=0.05 constant,
  α=0.10 — closest to optimal here.
- **Want to *exaggerate* the SARSA-vs-Q-learning contrast:** raise ε to 0.20–0.30
  (bigger, safer detour) — but expect a larger gap and a few more timeouts.
- **Decaying ε only makes sense** if the hazard is non-terminal (a mild penalty,
  not a cliff). On a terminal cliff, always prefer low constant ε.
- **Larger goal:die ratio (if the abyss penalty were softened):** the optimal
  path shifts toward the risky ledge; SARSA's caution gap widens — a good demo,
  but keep episodes ≥ 2000.

---

## Room 4 · Q-learning (vs SARSA)

Same TD engine as Room 3, but **SARSA and Q-learning train simultaneously on the
identical board** for a controlled comparison. The cliff returns, plus a **patrol
guard** (sweeps column 5, catching you is terminal) and a **bonus coin** on the
ledge. State is `(i, j, guard_phase, coin_mask)` — P=14 phases × 2 masks, so
~2800 states (hence the large episode budget). Q-learning is off-policy
(bootstraps `max_a Q`): from afar the ledge looks safe and the coin free.

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Blocked cells 🧱 | 0–20 | **15** | Safe detour always preserved. |
| Slippery cells 🟦 | 0–40 | **30** | |
| Slip probability | 0.0–0.8 | **0.30** | Raised — but note this **muddies the coin contrast** (see caution below). |
| Coin value 🪙 | 0–20 | **5** | **The contrast dial** — see below. Slider kept. |
| Goal reward 🏁 | — | **100 (fixed)** | Slider **removed**. |
| Fall / Catch | — | −100 (fixed) | |

**Change log:** removed goal-reward slider; board hardened to **15 / 30 / 0.30**
(walls 8→15, ice 20→30, slip 0.20→0.30).

> ⚠️ **Caution — this bump blurs Room 4's lesson (opposite of Room 3).** Measured
> on the new board (decaying ε, coin=5, 20k ep, 3 layouts × 2 seeds): both
> learners drop to ~83% success (from ~87/98%), falls explode (Q-learning ~5955,
> SARSA ~1960 per 20k), and true values fall (Q ~27, SARSA ~19.5, from ~39/28).
> **The clean coin contrast degrades:** the exact-optimal DP itself now takes the
> coin on only ~1/3 layouts (coin placement + slip noise vary board to board), so
> "Q-learning over-takes / SARSA skips" no longer reads crisply (Q 2/6, SARSA 1/6
> runs). SARSA's *caution signature survives* (it still falls ~3× less), but if you
> want the sharp signature lesson, **keep slip ≤ 0.20**. Best-params below are
> unchanged; they were measured on the earlier lower-slip board.

### Training parameters

| Param | Range / Options | **Best default** |
|---|---|---|
| Learning rate α | 0.01–0.5 | **0.10** |
| Discount γ | 0.50–0.99 | **0.95** |
| Episodes (each learner) | 5000…50000 | **20000** |
| Max steps / episode | 100…500 | **200** |
| Exploration | Decaying · Constant | **Decaying, 1.0 → 0.05 (0.9995)** |

### Best parameters — measured (default board, coin at (8,6), 3 seeds, V*≈49.7)

**Coin value is the pedagogical dial** (α=0.10, γ=0.95, 20k ep, decaying):

| coin value | DP-optimal | Q-learning | SARSA | contrast |
|---|---|---|---|---|
| 3 | skips | **takes** 3/3 | skips 0/3 | clean over-optimism |
| **5** | **skips** | **takes 3/3** | **skips 0/3** | **clean (default)** |
| 8 | takes | takes 3/3 | takes 1/3 | fading (coin genuinely worth it) |
| 12 | takes | takes | takes | none |

At **coin=5**, exact-optimal DP *skips* the coin, yet Q-learning greedily *takes*
it (walks the ledge) while SARSA correctly *detours* — the signature lesson.
Q-learning falls ~2.6× more during training (≈4400 vs ≈1700 / 20k episodes). Its
final greedy policy still scores higher than SARSA's (trueV ≈39 vs ≈28): on this
board SARSA's caution costs more than Q-learning's recklessness — both suboptimal,
opposite directions.

**Exploration schedule — the important result, and it is the OPPOSITE of Room 3:**

| schedule (20k ep) | Q-learning | SARSA | contrast intact? |
|---|---|---|---|
| **decaying 1.0→0.05** | takes coin, 87% succ | **skips coin**, 98% succ | **yes** |
| constant ε = 0.10 | takes coin, 80% succ | **takes coin** (35 tV), 94% | **no — broken** |
| constant ε = 0.20 | takes coin, 66% succ | takes coin, 84% | no |

**Decaying ε is required here.** With constant low ε, SARSA is no longer cautious
enough — it also takes the coin and the SARSA-vs-Q-learning contrast collapses.
The high *early* exploration of decaying ε is what teaches SARSA the ledge is
dangerous (it slips off repeatedly while ε is high), producing its cautious
signature. (Contrast Room 3, where a *terminal* cliff punished high early
exploration; here the guard/coin board with 20k episodes does not collapse SARSA,
so the early exploration is a feature, not a trap.)

**Episodes:** 10k already shows the contrast; 20k is the reliable default; 50k
barely improves Q-learning (trueV 41 vs 39) while doubling falls. **α=0.10** and
**γ=0.95** as in Room 3.

### If you change the environment

- **Raise coin value to ≥ 8:** the coin becomes genuinely optimal — DP takes it
  too, and "Q-learning over-valued it" is no longer true. Use this to show the
  *other* lesson (SARSA leaving real value on the table).
- **Higher slip or a nastier guard:** keep **decaying ε** and consider **50k
  episodes** so both learners cover the enlarged risk landscape; do NOT switch to
  constant low ε (it erases the contrast).
- **Want SARSA to visibly escape more reliably at any cost (not the contrast):**
  that is Room 3's regime — constant low ε — but it defeats Room 4's purpose.
- **Much larger state space (more coins/longer patrol):** raise episodes first
  (state count scales with P × 2^coins); α can rise to ~0.2 to speed early
  learning, but watch for the SARSA seed-collapse seen in Room 3 at α ≥ 0.2.

---

## Room 5 · Deep Q-Learning

The first **continuous** room: a `gymnasium` arena (10×10 m, no walls), direct
inertia-free movement, 9 discrete actions. A small MLP approximates
Q(state, action). State is `[x, y]` + per-enemy relative `[eₓ−x, e_y−y]`
(obs_dim = 2 + 2·n). **Double DQN + reward-scaling** are what make it converge
(vanilla DQN overestimates ~3× and goes state-blind). Dense progress-to-exit
shaping (`shaping_coef=5.0`) gives an immediate learning signal.

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Enemies (Chaser/Flanker/Ambusher) | toggles | **Chaser + Flanker** | 2-enemy pincer. |
| Enemy speed (× yours) | 0.50–0.95 | **0.75** | < 1 so it's winnable. |
| Max steps / episode | 20–100 | **60** | Corner-to-corner ≈ 14 m. |
| Randomize enemy spawns | on/off | **on** | Forces generalization; makes escape-rate a smooth signal. |
| Goal / catch reward | — | +100 / −100 (fixed) | Not exposed — already consistent. |

**Difficulty tiers** (greedy escape, default training): 1 enemy ≈ **100%**
(trivial) · 2 enemies ≈ **95–98%** (real pincer) · 3 enemies ≈ **55–64%** (the
hard wall). **Change log:** default raised from 1 chaser to **Chaser + Flanker** —
challenging but well short of the 3-enemy wall.

### Training parameters

| Param | Range / Options | **Best default** |
|---|---|---|
| Training episodes | 100…2000 | **1500** |
| Discount γ | 0.50–0.99 | **0.80** |
| Adam learning rate | 1e-4…3e-3 | **1e-3** |
| Batch size | 32/64/128 | 64 |
| Gradient step every N | 1/2/4/8 | 4 |
| Target update (steps) | 250…2000 | 1000 |
| Replay buffer | 5k…100k | 50k |
| Exploration | Decaying · Constant | **Decaying 1.0→0.05, decay 0.997** |

### Best parameters — measured (default 2-enemy board, greedy eval over 200 spawns, 3–4 seeds)

**Learning rate — the biggest lever, and it contradicts the old "3e-4 is safest":**

| lr | escape (±std, min) |
|---|---|
| 1e-4 | 92% (±10, 77) |
| 3e-4 | 89–93% (±5) |
| **1e-3** | **97% (±2, 94)** |
| 3e-3 | 90% (±10, 76) — unstable |

**Episodes:** 400 → 71%, 800 → ~90%, **1500 → 97–98%**. The harder 2-enemy board
needs the larger budget (1 enemy was already 100% at 800).

**γ:** robust across 0.70–0.99 (all ~90–95%) *because the dense shaping supplies an
immediate signal*. **γ=0.80 is best**: γ=0.99 reaches similar escape but with
much **longer paths** (24 vs 18 steps) — high γ discounts the shaping less and the
agent dawdles. Low γ beelines efficiently.

**Confirmed best combo:** **γ=0.80, lr=1e-3, 1500 episodes → 98% (±2, min 96)**,
18-step paths. (Double DQN + `reward_scale=0.01` are non-negotiable — single-net
DQN diverges here regardless of the above.)

Secondary knobs (batch 64, train-freq 4, target-update 1000, buffer 50k) were
left at reference defaults — they did not move the needle on this env and are kept
visible for teaching.

### If you change the environment

- **3 enemies (the hard wall):** escape drops to ~55–64%. Raise **episodes to
  2000**, keep **lr 1e-3**, and consider **lowering enemy speed to 0.60–0.65** —
  or accept that 3 same-speed hunters is near the ceiling of what this net/obs can
  do. Do not raise γ (it lengthens paths without helping escape).
- **Higher enemy speed (≥0.9) with 1–2 enemies:** still ~95%+; no retuning needed.
- **Single enemy (easy mode):** 800 episodes at lr 1e-3 already hits ~100%; drop
  episodes to save time.
- **If training diverges** (mean-Q blows up, escape collapses): lower **lr to
  3e-4**, confirm Double DQN is on, and check `reward_scale` is ≤ 0.01. This is
  the failure the architecture was designed around.
- **Longer episodes / bigger arena:** raise **max_steps** and **buffer** together
  so replay still covers whole trajectories.

---

## Room 6 · Advanced DQL — Light & Dynamic Obstacles

The hardest room: **momentum** returns (actions set acceleration under per-second
friction) and the agent is **partially observed** — it sees only its `[x,y,vₓ,v_y]`
plus every obstacle inside its **light radius X** (relative position, nearest-first;
plus relative velocity when obstacles move). Layouts re-randomize every episode
(one fixed central pillar), so the headline metric is the **Generalisation Score**:
greedy escape rate on 150 held-out layouts never trained on. Same DQN recipe as
Room 5 (Double DQN + reward-scaling + straight-line shaping), unchanged.

### Environment parameters

| Param | Range | **Default** | Notes |
|---|---|---|---|
| Light radius X (m) | 1.0–10.0 | **3.0** | Center-to-center; usable clearance = X − 0.5. |
| Obstacle count | 2–12 | **6** | **The difficulty lever** (see below). One is always the central pillar. |
| Dynamic-obstacle speed | 0.0–1.5 | **0.5** | **Changed from 0 (static).** Showcases the room's headline dynamic feature + velocity channel. |
| Ice friction μ (per sec) | 0.2–0.9 | **0.5** | Momentum retention. |
| Randomize layout | on/off | **on** | Required for the generalisation metric. |
| Goal / collision | — | +100 / −100 (fixed) | Not exposed. |

**Change log:** enabled **moving obstacles (speed 0.5)** by default — the room is
named "Dynamic Obstacles" but shipped static. The velocity channel keeps motion
learnable (Markov-observable), but moving *is* mildly harder and noisier than
static (GEN ≈81% ±7 vs ≈90% ±4 at 2000 ep, γ 0.90) — a fair "challenging but not
hardest" showcase of the advanced feature, not a free lunch.

### Training parameters

| Param | Range / Options | **Best default** |
|---|---|---|
| Training episodes | 500…5000 | **1200** (2000 is measurably better) |
| Discount γ | 0.50–0.99 | **0.90** (0.95 also fine) |
| Adam learning rate | 1e-4…3e-3 | **3e-4** |
| Batch / train-freq / target / buffer | — | 64 / 4 / 1000 / 50k |
| Exploration | Decaying · Constant | **Decaying 1.0→0.05, decay 0.998** |

### Best parameters — measured (Generalisation Score, 2 seeds × 150 held-out layouts)

**Learning rate — 3e-4 wins, and this is the OPPOSITE of Room 5:**

| lr (6-static) | 1200 ep | 2000 ep |
|---|---|---|
| **3e-4** | **81%** | **90%** |
| 1e-3 | 77% | 79% |

On this partially-observed, higher-dimensional input, the larger step (1e-3) that
helped Room 5's simple 6-D state *hurts* here — keep **3e-4**.

**Episodes — the biggest lever:** 6-static GEN **81% → 90%** going 1200 → 2000,
collisions 15% → 8%; 8-static **68% → 77%**. (Shipped default stays 1200 for speed
per the user; raise to 2000 for the best score.)

**Obstacle count is the difficulty dial:**

| obstacles | GEN @1200 ep | GEN @2000 ep |
|---|---|---|
| 6 (static) | 81% | 90% (±4) |
| 6 (moving @0.5) | ~85% | **81% (±7)** |
| 8 (static) | 68% (±9) | 77% (±1) |
| 12 | — | slider max (hardest) |

**Moving obstacles (the shipped default):** the velocity channel keeps them
learnable, but they're mildly harder *and noisier* than static (81% ±7 vs 90% ±4
at 2000 ep). Not "free" — but a legitimate challenging showcase. (Older "~0.66
moving" memory notes were a pre-fix build; the range-rate fix lifted this a lot.)

**γ:** 0.90 and 0.95 both give ~90% on 6-static (exit ~13 m away wants a fairly
high γ); **0.90 confirmed** as the app-wide default with no loss here.

### If you change the environment

- **Want the hardest winnable room:** raise **obstacle count to 8–10** and
  **episodes to 2000+**; keep **lr 3e-4**. Expect GEN in the 60–75% range and
  higher seed variance.
- **Faster moving obstacles (≥1.0):** still learnable thanks to the velocity
  channel, but shrink the **light radius** below ~3 only if you want to starve the
  agent of warning (X − 0.5 is the real clearance; a small X removes reaction time).
- **Do NOT raise lr to 1e-3** here (it helped Room 5 but hurts this
  partially-observed input) and do not drop below ~1000 episodes.
- **Static + few obstacles** is the easy warm-up: 6-static at 2000 ep ≈ 90%.
- **More episodes always help** more than any other single knob on this env —
  reach for episodes before touching the secondary DQN knobs.

---

*Verification: every room was checked headlessly via `AppTest.from_file` on a
wrapper that calls `roomN_*.render()`; all six render without exceptions at the
documented defaults. DQN escape/generalisation numbers are greedy (ε=0) evals over
fixed held-out seed streams, disjoint from training.*
