# Revised Technical Specification: Hybrid Neuro-Symbolic Level Mechanics Engine (with Viewport Targeting)

## 0) What changes vs the prior spec

1. **Add a first-class “Viewport / Target Level” module** so the “level in question” can be selected, tracked, and swapped dynamically intraday (manual focus + auto focus). This matches how you’ll actually use it (e.g., “we’re heading to 15m OR Low; infer *there* / at nearby strikes”).
2. **Make tradeability multi-scale**: keep the canonical **$2 in 5m** outcome labeling, but also model **$1 and $2 time-to-threshold** as separate event-time distributions (already in schema). This reconciles the analyst’s “$1 move” gating idea with your canonical outcome definition. 
3. **Promote normalization + invariances** (distance-from-current-price / ATR-normalization) into the core math spec (important for both ML + retrieval). 
4. **Add “attempt number / defense deterioration”** as explicit derived features (touch clustering) to support repeated tests at the same level while the “focus” persists. 

Everything else remains the same: a **hybrid** of (i) feasibility constraints (“physics”), (ii) boosted-tree scorer, and (iii) retrieval/pattern memory.   

---

## 1) Core objective (what the system outputs)

Given a **target level** (L) (e.g., OR_LOW, PM_HIGH, STRIKE, CALL_WALL) and the current market state, output:

1. **Tradeability**
   (p_{\text{tradeable}} = \Pr(\text{not CHOP})) under the canonical “meaningful move” definition. 
2. **Direction conditional on tradeability**
   (p_{\text{break}} = \Pr(\text{BREAK}\mid \text{tradeable})), (p_{\text{bounce}} = 1-p_{\text{break}}). 
3. **Strength and speed**

   * (\mathbb{E}[\text{strength_signed}]) and robust quantiles
   * (\Pr(T_{$1}\le \tau)) and (\Pr(T_{$2}\le \tau)) for key (\tau) values (time-to-threshold distributions). 
4. **Retrieval context** (K nearest analogs): outcome distribution + strength/time summaries for “show me similar level tests.” 

---

## 2) First-class “Viewport / Target Level” module

### 2.1 LevelUniverse (what can be targeted)

At any intraday time (t), define a universe of level entities ( \mathcal{U}(t)), including at minimum:
PM_HIGH/LOW, OR_HIGH/LOW, SESSION_HIGH/LOW, SMA_200/400, VWAP, ROUND, STRIKE, CALL_WALL, PUT_WALL.  

Each level entity is:
[
\ell = (\text{kind}, \text{price}(t), \text{valid_from}, \text{valid_to}, \text{dynamic?})
]

* **Static-ish**: PM_H/L, OR_H/L once formed, ROUND, STRIKE grid.
* **Dynamic**: VWAP, SMA_200/400, SESSION_H/L (updates with new highs/lows), CALL_WALL/PUT_WALL if your wall computation updates intraday.

### 2.2 Viewport definition (the “analysis window” you focus on)

A **Viewport** at time (t) is a ranked subset:
[
\mathcal{V}(t) \subset \mathcal{U}(t)
]
that the engine will actively score and update.

Viewport membership can be:

* **Manual focus**: user says “focus OR_LOW_15m” or “focus STRIKE 485”.
* **Auto focus**: system chooses top-N “relevant targets” near spot.

### 2.3 Auto-focus selection (mathematical scoring)

Let spot be (S(t)). For each level (\ell\in\mathcal{U}(t)), compute a **relevance score**:
[
R(\ell,t) = w_d \cdot f_d(|S(t)-\ell.\text{price}(t)|) ;+; w_v \cdot f_v(\text{approach_velocity}) ;+; w_c \cdot f_c(\text{confluence}) ;+; w_g \cdot f_g(\text{gamma})
]

Minimum requirements:

* Proximity uses the existing **critical decision zone** concept: your dataset is filtered to a monitor band (distance (\le $0.25)) as a “critical decision zone.” 

  * Use two radii:

    * **Scan radius** (larger) to *consider* targets.
    * **Monitor band** (tight) to enter “event mode.” (Use the existing (0.25) as the default monitor band.)
* Confluence comes from your confluence feature group. 
* Gamma terms should consider sign/regime via `fuel_effect` and magnitude via `gamma_exposure`. 

Auto-focus returns:

* a ranked list of targets ( \ell_1,\dots,\ell_N)
* plus “why” metadata (dominant contributors to (R)).

### 2.4 Focus persistence and “level migration”

The viewport must support **persistence** (you stay focused on OR_LOW as price approaches) and **migration** (once that level is resolved, focus shifts).

Define per-level finite-state:

* FAR → APPROACHING → IN_MONITOR_BAND → TOUCH/TEST → CONFIRMATION → RESOLVED

Rules:

* Enter APPROACHING when within scan radius and approach velocity points toward the level.
* Enter IN_MONITOR_BAND when (|S(t)-L|\le) monitor band. 
* Define TOUCH at first time the distance crosses 0 or hits an epsilon zone around the level price.
* After RESOLVED, keep the level in viewport only if (a) it remains close, or (b) it is explicitly pinned by the user.

### 2.5 “Infer at that strike” while approaching OR_LOW

This is handled by allowing both of the following to be simultaneously targeted in the viewport:

* the structural level (e.g., OR_LOW)
* the nearest strike(s) (STRIKE grid) around that structural level

Confluence features explicitly measure stacked levels near the target. 
So the system can answer:

* “At OR_LOW: what happens?” and
* “At STRIKE K: what happens?”
  without conflating them, while still capturing their relationship via confluence.

---

## 3) Timing and leakage control (unchanged, but now tied to viewport states)

Your barrier/tape metrics are defined using a **forward window of 60 seconds from the touch timestamp**.  
This is only valid if the decision is made after observing that window. 

Therefore define for each TOUCH/TEST episode:

* (t_0): touch timestamp
* (t_1 = t_0 + 60s): confirmation decision time

Two-stage structure:

* **Stage A (Intent)**: any time from APPROACHING → TOUCH, uses only information (\le t_0). 
* **Stage B (Confirmation)**: at (t_1), uses barrier/tape measurements from ([t_0,t_1]). 

---

## 4) Ground truth labels (canonical)

Use the `features.json` outcome methodology as canonical:

* Actionable threshold: **$2.00 (2 strikes)**
* Lookforward horizon: **5 minutes**
* CHOP means “didn’t move $2.00 either direction”
* UNDEFINED if forward window incomplete. 

Also use:

* time-to-threshold metrics at **$1.00 and $2.00**. 

### 4.1 Re-anchor labels to the confirmation time

If Stage B uses information from ([t_0,t_1]), then Stage B labels must be defined on ([t_1, t_1+5m]) to preserve causality. (Same semantics; shifted anchor.)

Let direction (d\in{+1,-1}) encode approach direction (UP = +1, DOWN = −1). 

Define:

* (M^+ = \max_{t\in[t_1,t_1+5m]} d\cdot(S(t)-S(t_1)))
* (M^- = \max_{t\in[t_1,t_1+5m]} -d\cdot(S(t)-S(t_1)))

Then:

* BREAK if (M^+\ge 2)
* BOUNCE if (M^-\ge 2)
* CHOP otherwise

Strength targets:

* (\text{strength_signed} := M^+ - M^-)
* (\text{strength_abs} := \max(M^+,M^-))

Time-to-threshold:

* (T_{$1}): first (t\ge t_1) s.t. (d(S(t)-S(t_1))\ge 1)
* (T_{$2}): first (t\ge t_1) s.t. (d(S(t)-S(t_1))\ge 2) 

### 4.2 Multi-scale tradeability targets (to match analyst insight)

Define two binary events:

* tradeable_1 := (\mathbb{I}(T_{$1}\le 5m))
* tradeable_2 := (\mathbb{I}(T_{$2}\le 5m))

Use tradeable_2 as the canonical “not CHOP,” but train both (T_{$1}) and (T_{$2}) distributions so the system can:

* gate early on “is *any* move likely soon?”
* and still preserve the canonical $2 move decision.

---

## 5) Feature specification (existing + required derived features)

### 5.1 Use the existing feature contract as authoritative

Your feature taxonomy already matches the goal: confluence/level context, approach context, barrier/tape mechanics, and dealer gamma/fuel + velocity. 

Key mechanics features are explicitly defined:

* Barrier physics with states and continuous measures; includes the 60s forward window definition.  
* Tape physics (`tape_imbalance`, `tape_velocity`, `sweep_detected`). 
* Fuel/gamma (`gamma_exposure`, `fuel_effect` semantics AMPLIFY/DAMPEN). 
* Dealer velocity (`gamma_flow_velocity`, accelerations, `dealer_pressure_accel`). 
* Confluence (`confluence_weighted_score`, `confluence_pressure`). 
* Pressure indicator channels (liquidity/tape/gamma pressures + accel) for possible future sequence encoders. 

### 5.2 Mandatory normalization / invariances (for ML + retrieval)

Implement normalization consistent with best practices:

* Encode technical levels as **distance-from-current-price** features, ideally normalized by ATR/recent volatility. 
* Encode prices as **percentage difference from current mid/spot** to reduce non-stationarity. 

For any raw dollar distance feature (x):

* ATR-normalized: (x_{\text{atr}} = x / (\text{ATR}+\epsilon))
* or spot-normalized: (x_{%} = x/S(t))

For sparse barrier measures (e.g., wall_ratio sparsity noted in dataset stats), include both:

* indicator (I(x\neq 0))
* and a signed log magnitude ( \operatorname{sign}(x)\log(1+|x|)). 

### 5.3 Required derived features (additions)

These are not “new data sources,” just higher-level transforms from what you already compute.

#### (A) Confluence alignment

Add a boolean / categorical “alignment” feature describing whether nearby stacked levels reinforce resistance/support *in the approach direction*.

Example: for approach direction UP (testing resistance), alignment is “stacked overhead” if price is below both SMAs and SMA slopes are negative and spread widening (this idea is explicitly recommended). 

#### (B) Touch clustering + attempt index + defense deterioration

Your `prior_touches` exists, but repeated probing needs a more structured encoding.  

Define a **touch cluster** for a given (date, level_kind, level_price, direction):

* cluster touches within (\Delta t) minutes and within (\Delta p) dollars of the exact level price.

Within each cluster compute:

* `attempt_index` (1,2,3,…)
* trends across attempts:

  * (\Delta) barrier replenishment / delta-liq trend
  * tape velocity trend
  * tape imbalance trend
    These explicitly encode “defense strengthening vs weakening” across repeated tests.

#### (C) Viewport-relative features (critical for “infer at OR_LOW vs infer at strike”)

Every model input must be computed *relative to the current target level (\ell)*:

* `distance` and `direction` are always w.r.t. that target. 
* Confluence features become “secondary levels near the target,” not near spot in general. 

This is the mathematical mechanism that makes viewport targeting consistent: the same scoring engine works for *any* chosen target.

---

## 6) Modeling architecture (hybrid stack)

### 6.1 Layer 1 — Feasibility / “Physical possibility” constraints

Implement deterministic masks that prevent directions that are mechanically implausible. This is a core shared idea across docs.  

Output per target:
[
m_{\text{allow}}(\ell,t) \in {0,1}^2 = (\text{allow_break}, \text{allow_bounce})
]

Feasibility inputs must use:

* barrier states/continuous metrics (VACUUM vs WALL semantics) 
* tape metrics (imbalance/velocity/sweep) 
* gamma regime (AMPLIFY/DAMPEN meaning) 

### 6.2 Layer 2 — Parametric scorer (tree ensemble)

Use a boosted-tree family (LightGBM/XGBoost/CatBoost) as the first decision brain given current data scale and feature richness.  

#### Heads / objectives (minimum set)

1. (p_{\text{tradeable_2}}) (non-CHOP under the $2 move definition)
2. (p_{\text{break}}) on tradeable examples
3. strength regression (signed) + optional quantiles
4. discrete-time survival (hazard) for (T_{$1}) and (T_{$2}) (or equivalent reach-probabilities)

This matches the recommended “CHOP as first-class” decomposition. 

### 6.3 Layer 3 — Retrieval / similarity memory

Build a vector similarity index over historical level tests and compute empirical probabilities from neighbors:
[
P_{\text{bounce}}^{\text{kNN}}=\frac{\sum_{i=1}^{K} w_i,\mathbb{I}(\text{neighbor}*i=\text{BOUNCE})}{\sum*{i=1}^{K} w_i}
]
(as explicitly described). 

Retrieval should use:

* metadata filters: (level_kind_name, direction) and optionally gamma-regime buckets.
* an embedding built from normalized engineered features initially; reserve PatchTST for later embedding learning.  

### 6.4 Gating/ensemble between tree scorer and retrieval

If the parametric scorer and retrieval sharply disagree, downgrade confidence (hallucination check). This is consistent with the hybrid “scoring + sanity check” view. 

Define a mixture:
[
p_{\text{break}}^{\text{final}}=\lambda,p_{\text{break}}^{\text{tree}}+(1-\lambda),p_{\text{break}}^{\text{kNN}}
]
where (\lambda) is higher when neighbor similarity is low / neighbor outcomes are high-entropy.

Finally apply feasibility mask:

* if allow_break=0 → (p_{\text{break}}^{\text{final}}=0)
* if allow_bounce=0 → (p_{\text{break}}^{\text{final}}=1)

---

## 7) Inference-time system behavior (now viewport-driven)

### 7.1 Continuous scanning

At each inference tick/time step (t):

1. Build/update (\mathcal{U}(t)) (LevelUniverse snapshot).
2. Update (\mathcal{V}(t)) using:

   * pinned targets (manual)
   * auto-focus ranking
3. For each (\ell\in\mathcal{V}(t)), compute target-relative features and output Stage A scores.

### 7.2 Event mode when in monitor band

When (|S(t)-\ell.\text{price}(t)| \le) monitor band (default from data quality filter), treat as an active “level test episode.” 

* Stage A continues to update as approach evolves.
* When TOUCH is detected, set (t_0).
* At (t_1=t_0+60s), compute confirmation features (barrier/tape measured during ([t_0,t_1])) and output Stage B. 

### 7.3 Multi-target outputs and ranking

Because the viewport may contain multiple targets (e.g., OR_LOW and nearby strikes), produce a per-target “action summary” and rank by a scalar utility proxy, e.g.:
[
\text{Score}(\ell)=p_{\text{tradeable_2}}(\ell)\cdot \mathbb{E}[\text{strength_abs}(\ell)]
]
(or a more conservative quantile-based score).

This gives you the practical effect: “while moving down, which target level has the cleanest mechanical setup?”

---

## 8) Training and evaluation protocol

### 8.1 Data source

Use the canonical vectorized dataset output path. 
Dataset stats show ~3.3k signals across 4 days; treat this as small-data and prefer robust baselines. 

### 8.2 Walk-forward splits

Split by date (strict walk-forward) to avoid temporal leakage (no random shuffle within day). 

### 8.3 Ablations

Evaluate at minimum:

1. TA/context only
2. Mechanics only (barrier/tape/gamma/pressure)
3. Full

This tests the hypothesis: confluence sets the stage; mechanics decide resolution. 

### 8.4 Calibration checks

Because outputs are probabilities used for gating and ranking:

* reliability curves for (p_{\text{tradeable}}) and (p_{\text{break}})
* Brier scores for discrete-time reach probabilities (\Pr(T_{$1}\le\tau)), (\Pr(T_{$2}\le\tau))

---

## 9) Implementation deliverables (what the coding agent must build)

1. **Viewport/Target module**

   * LevelUniverse snapshot builder
   * Viewport manager (manual focus + auto focus + persistence/migration rules)
2. **Touch/episode state machine**

   * FAR→APPROACHING→IN_MONITOR_BAND→TOUCH→CONFIRMATION→RESOLVED
   * emits (target, t0, t1) episodes
3. **Leakage-safe label anchoring for confirmation**

   * Stage B labels anchored at (t_1), not (t_0) (definition above)
4. **Derived feature engineering**

   * confluence alignment
   * touch clustering / attempt_index / deterioration trends
   * strict target-relative computation for all features
5. **Hybrid inference stack**

   * feasibility mask (m_{\text{allow}})
   * tree heads: tradeable, direction, strength, time-to-threshold distributions
   * retrieval index and empirical outcome/strength/time summaries
   * ensemble + disagreement downgrade + feasibility gating
6. **Walk-forward training + ablations + calibration evaluation**

---

## 10) Why this resolves the “viewport complication”

The complication disappears once the system treats **(target level, direction)** as the fundamental conditioning variable and ensures *every feature and label is defined relative to that target*. Your existing pipeline already “generates signals for all levels near spot,” which is the right foundation. 

The new piece is simply: **a principled way to choose which of those levels are “in question” right now**, and to keep that choice stable enough to support repeated tests (attempt-index features) while still allowing focus to migrate as price moves.
