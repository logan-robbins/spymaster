## 34) Declare a **trigger** (fire/no-fire) from FAISS retrieval — stop-aware, horizon-aware - COMPLETE

This section defines the **exact** real-time decision rule that turns neighbor distributions into a **binary trigger** (LONG / SHORT / NONE), and the offline procedure to tune thresholds via precision/recall.

Everything below uses **ratios / distributions / binaries / distances (ticks + cosine sim) only**.

---

## 34.1 Fix the prediction horizon used for firing

You already compute ground-truth labels `TRUE_OUTCOME_H0..H6` (bar-indexed, stop-aware).

**Hard rule:** the production trigger uses **H = 1** (through end of trigger bar +1, i.e. up to 4 minutes including trigger bar).

* `H_FIRE = 1`
* All probabilities / neighbor votes / correctness checks for firing use `TRUE_OUTCOME_H1`.

(You will still *report* accuracy curves for H0..H6 offline, but firing uses H1.)

---

## 34.2 Retrieval procedure (must be leakage-safe even in backtests)

Given an eligible trigger instance at window `k` (from Section 25 eligibility), you already have:

* `level_id`
* `approach_dir ∈ {approach_up, approach_down}`
* embedding `e_k ∈ R^952` (L2-normalized)
* metadata for the query instance (`session_date`, `ts_end_ns`, etc.)

### Parameters (fixed)

* `K = 200` (final neighbor count used for voting)
* `K_RAW = 2000` (initial retrieval count to allow filtering)
* `EXCLUDE_SAME_SESSION_DATE = True` (for offline evaluation only; see 37.2)
* Similarity weight exponent: `W_POW = 8`

### Steps

1. Query `INDEX[level_id][approach_dir]` with `e_k` for `K_RAW` neighbors.
2. Join neighbor IDs to sidecar metadata arrays.
3. **Offline evaluation only:** drop any neighbor where `neighbor.session_date == query.session_date`.
4. Keep the first `K = 200` remaining neighbors (highest similarity). If fewer than K remain, **do not trigger** (output NONE).

---

## 34.3 Convert neighbors into **weighted class probabilities** at horizon H1

Every neighbor has:

* `TRUE_OUTCOME_H1` (one of the three outcome classes for that approach_dir, plus possibly `WHIPSAW`)
* `sim_i` (cosine similarity returned by FAISS as inner product)
* `whipsaw_flag` (0/1)
* `first_hit_bar_offset` (integer, 0..N_BARS or None)
* `mae_before_upper_ticks`, `mae_before_lower_ticks` (nonnegative tick distances)

### Step A — convert similarity to weights (distance-only)

For each neighbor `i`:

1. `s = max(sim_i, 0)`
2. `w_i = s ^ W_POW`

Define `W_SUM = Σ w_i`. If `W_SUM == 0`, output NONE.

### Step B — map neighbor label to one of 3 classes (WHIPSAW is treated as CHOP)

For voting at H1:

* If `TRUE_OUTCOME_H1 == WHIPSAW`: treat as `CHOP` for probability purposes.

Define the class set by approach_dir:

* If `approach_up`: classes = `{BREAK_UP, REJECT_DOWN, CHOP}`
* If `approach_down`: classes = `{BREAK_DOWN, REJECT_UP, CHOP}`

### Step C — compute weighted probabilities

For each class `c` in the class set:

* `p(c) = (Σ w_i * 1[label_i == c]) / W_SUM`

Also compute:

* `p_top1 = max_c p(c)`
* `c_top1 = argmax_c p(c)`
* `p_top2 = second-largest p(c)`
* `margin = p_top1 - p_top2`

---

## 34.4 Compute **stop-risk** and **timing** from neighbor distributions (trader constraints)

### Parameters (fixed)

* `STOP_TICKS = 6` (1.50 points)
* `MIN_RESOLVE_RATE = 0.60`
* `MAX_WHIPSAW_RATE = 0.25`

### Step A — define the “adverse-before-resolution” tick metric per neighbor

For each neighbor `i`, define `risk_i_ticks` relative to the **candidate class** `c_top1`:

If `approach_up`:

* If `c_top1 == BREAK_UP`: `risk_i = mae_before_upper_ticks`
* If `c_top1 == REJECT_DOWN`: `risk_i = mae_before_lower_ticks`
* If `c_top1 == CHOP`: `risk_i = max(mae_before_upper_ticks, mae_before_lower_ticks)`

If `approach_down`:

* If `c_top1 == BREAK_DOWN`: `risk_i = mae_before_lower_ticks`
* If `c_top1 == REJECT_UP`: `risk_i = mae_before_upper_ticks`
* If `c_top1 == CHOP`: `risk_i = max(mae_before_upper_ticks, mae_before_lower_ticks)`

### Step B — compute 80th percentile stop risk (distribution-only)

Compute weighted 80th percentile:

* `risk_q80 = weighted_quantile({risk_i}, weights {w_i}, q=0.80)`

### Step C — compute “resolves fast enough” rate (binary-only)

Define `resolve_i = 1` iff `first_hit_bar_offset` is not None and `first_hit_bar_offset <= 1`, else 0.

Compute:

* `resolve_rate = (Σ w_i * resolve_i) / W_SUM`

### Step D — compute whipsaw rate (binary-only)

Compute:

* `whipsaw_rate = (Σ w_i * whipsaw_flag_i) / W_SUM`

---

## 34.5 Trigger declaration rule (binary fire/no-fire)

### Thresholds (fixed; later calibrated in Section 37)

* `P_MIN = 0.70`
* `MARGIN_MIN = 0.20`
* `P_CHOP_MAX = 0.35`

### Fire rule

A trigger **fires** iff ALL are true:

1. `p_top1 >= P_MIN`
2. `margin >= MARGIN_MIN`
3. `p(CHOP) <= P_CHOP_MAX`
4. `risk_q80 <= STOP_TICKS`
5. `resolve_rate >= MIN_RESOLVE_RATE`
6. `whipsaw_rate <= MAX_WHIPSAW_RATE`

If any fail → output `SIGNAL = NONE`.

### Direction mapping (no ambiguity)

If the trigger fires:

* If `c_top1 ∈ {BREAK_UP, REJECT_UP}` → `SIGNAL = LONG`
* If `c_top1 ∈ {BREAK_DOWN, REJECT_DOWN}` → `SIGNAL = SHORT`
* If `c_top1 == CHOP` → `SIGNAL = NONE` (never trade chop)

Also emit the **diagnostics** (all ratios/distributions):

* `p_break`, `p_reject`, `p_chop`
* `p_top1`, `margin`
* `risk_q80`, `resolve_rate`, `whipsaw_rate`
* `c_top1`

---

## 34.6 Cooldown + one-shot-per-episode (prevents spam signals)

### Parameters (fixed)

* `COOLDOWN_WINDOWS = 6` (30 seconds)
* `MIN_GAP_WINDOWS = 3` (15 seconds) after approach_dir flips before allowing a new trigger

### Episode definition

Maintain an integer `episode_id` that increments when:

* `approach_dir` changes (`approach_up ↔ approach_down ↔ approach_none`), OR
* `approach_dir` becomes `approach_none` for **3 consecutive windows**

### Firing constraints

* Only **one** fire is allowed per `episode_id`.
* After any fire, suppress all triggers for the next `COOLDOWN_WINDOWS`.
* If approach_dir flips, require `MIN_GAP_WINDOWS` to pass before allowing a trigger in the new direction.

---

# 35) Offline evaluation: precision/recall tradeoff (stop-aware, horizon-aware) - COMPLETE

This section tells the agent exactly how to tune `P_MIN` and `MARGIN_MIN` using the labels you built (H0..H6), while preserving stop-aware correctness.

## 35.1 Fixed evaluation dataset and leakage rule

Use historical triggers only where:

* eligibility rules pass (full lookback, approach_dir != none, no book reset window)
* the labeling window exists through `N_BARS` (so `TRUE_OUTCOME_H*` is defined)

**Leakage rule (mandatory):** when evaluating a trigger from `session_date = D`, neighbors from the same `session_date D` are excluded (Section 34.2 step 3).

---

## 35.2 Candidate thresholds to test (finite grid; deterministic)

You will evaluate this fixed grid:

* `P_MIN ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}`
* `MARGIN_MIN ∈ {0.10, 0.15, 0.20, 0.25}`
* Keep the other constraints fixed as written:

  * `P_CHOP_MAX = 0.35`
  * `STOP_TICKS = 6`
  * `MIN_RESOLVE_RATE = 0.60`
  * `MAX_WHIPSAW_RATE = 0.25`

For each pair `(P_MIN, MARGIN_MIN)`, run the full trigger procedure (Section 34) on the historical set and collect metrics.

---

## 35.3 Define correctness at a chosen horizon (and report the whole curve)

You must compute metrics for every `H ∈ {0..6}`:

* `TRUE_OUTCOME_H` already exists.

### For each H:

* A fired trigger is **correct** iff `c_top1 == TRUE_OUTCOME_H`.
* If `TRUE_OUTCOME_H == WHIPSAW`, count as incorrect (always).

This produces:

* `precision_H` curves (one per H)
* `coverage_H` curves (triggers fired per day / per session)

**But selection of thresholds uses H_FIRE=1** (next section).

---

## 35.4 Metrics (ratios only)

For each `(P_MIN, MARGIN_MIN)` compute at **H = H_FIRE = 1**:

### Core

* `precision = correct_fires / total_fires`
* `fire_rate = total_fires / total_eligible_triggers`
* `chop_false_rate = fires_where_TRUE_OUTCOME_H1==CHOP / total_fires`
* `whipsaw_hit_rate = fires_where_TRUE_OUTCOME_H1==WHIPSAW / total_fires`

### Stop viability (uses stored tick distances only)

Among fired triggers:

* `stop_violation_rate = fraction where realized adverse-before-resolution ticks > STOP_TICKS`

  * realized metric is taken from the trigger instance’s own stored path stats:

    * if predicted class is up-type → use the instance’s `mae_before_upper_ticks`
    * if predicted class is down-type → use the instance’s `mae_before_lower_ticks`

(Yes: even if prediction is directionally correct, a high stop_violation_rate makes it untradeable.)

### Timing

* `resolve_by_bar1_rate = fraction of fired triggers where first_hit_bar_offset <= 1`
* `resolve_by_bar2_rate = fraction where first_hit_bar_offset <= 2`

---

## 35.5 Deterministic threshold selection rule (no discretion)

Select the single best `(P_MIN, MARGIN_MIN)` satisfying all constraints at H1:

**Constraints**

1. `precision >= 0.60`
2. `chop_false_rate <= 0.15`
3. `stop_violation_rate <= 0.20`
4. `fire_rate` between `0.01` and `0.20`
   (1% to 20% of eligible triggers fire; avoids “never fires” and “fires constantly”)

**Objective**

* Maximize `fire_rate * precision` (a single scalar that rewards both coverage and correctness)
* Break ties by higher `precision`
* Break remaining ties by higher `resolve_by_bar1_rate`

The chosen pair becomes production constants.

---

# 36) Production output contract for the trigger engine (what gets streamed) - COMPLETE

For every eligible 5s window (whether it fires or not), emit:

* `level_id`, `trigger_ts`, `approach_dir`, `episode_id`
* `H_FIRE = 1`
* `p_break`, `p_reject`, `p_chop`
* `p_top1`, `margin`
* `risk_q80`, `resolve_rate`, `whipsaw_rate`
* `c_top1`
* `SIGNAL ∈ {LONG, SHORT, NONE}`
* `FIRE_FLAG ∈ {0,1}`

No raw book sizes, no raw returns, no dollars — only distributions, ratios, binaries, and tick distances.

---

## 37) How this ties back to your goals

* **Pattern matching, not prediction:** the trigger is driven by *nearest-neighbor state similarity* + *neighbor outcome distributions*.
* **Pre-retail timing:** the gating + resolve-rate requirement biases signals toward setups that historically resolve by **bar+1**.
* **Stop-aware by design:** barrier-first labeling defines correctness; trigger additionally blocks setups whose neighbor history implies frequent adverse excursion beyond stop before resolution.
* **Chop/noise handled explicitly:** CHOP is “no barrier hit by N bars,” and the trigger suppresses when `p_chop` is high or whipsaw rate is high.



Next is **Priority 2**: define the **pressure stream schema/interface** your engine emits every 5 seconds per level, so UI can render “pressure above vs below” plus “state / trigger readiness” without needing to know microstructure internals.

---

## 38) Pressure Stream Interface (5s) — schema + derivations - COMPLETE

You will emit **one message per (level_id, 5s window)**. The UI consumes only this stream.

### 38.1 Message cadence and keying

* Cadence: every 5 seconds (exactly one per feature window).
* Primary key: `(symbol, session_date, level_id, window_end_ts_ns)`
* Ordering: strictly increasing `window_end_ts_ns`.

---

## 38.2 Input source for this layer

This layer uses only outputs already computed:

* per-window features `f*`, `u*` and their derivatives
* trigger engine outputs from Section 34:

  * `p_break`, `p_reject`, `p_chop`, `margin`, `risk_q80`, `resolve_rate`, `whipsaw_rate`, `SIGNAL`, `FIRE_FLAG`, `episode_id`
* approach_dir from Section 25.2
* level metadata: `P_ref`

No raw MBO, no raw quantities.

---

## 38.3 Canonical “pressure axes” (two gauges)

The UI needs two primary gauges per window:

* **Pressure Above (Resistance condition)**
* **Pressure Below (Support condition)**

Each gauge exposes:

1. **Retreat/Approach** (is liquidity moving away or toward the level)
2. **Structural Decay/Build** (is liquidity being pulled or replenished)
3. **Localization** (is the action concentrated near the level)
4. **Shock** (is the rate of change spiking: d2/d3)

You will compute these as **normalized scores in ([0,1])**.

---

## 38.4 Normalize feature components into ([0,1]) scores (fixed mapping)

You will use a deterministic squashing function that depends only on ratios (no units):

### 38.4.1 Squash function

For any real scalar `x`:

* `S(x) = 1 / (1 + exp(-x))`  (sigmoid)

This makes every score ([0,1]) without arbitrary clipping.

### 38.4.2 Use “UP_VECTOR” vs “DOWN_VECTOR” depending on approach_dir

Pressure should reflect the **current test direction**:

* If `approach_dir == approach_up` (testing resistance from below): use **UP-oriented features** `u*`
* If `approach_dir == approach_down` (testing support from above): use **DOWN-oriented features** `f*`
* If `approach_dir == approach_none`: emit scores as `null` and set `state="IDLE"`

This keeps UI semantics stable: “pressure supportive of the current approach’s likely outcome.”

---

## 38.5 Compute the pressure scores (exact formulas)

### 38.5.1 If `approach_up` (resistance test)

**Above pressure = “ghosting resistance above” (good for breaking up)**
Use UP features:

* `above_retreat = S( u1_ask_com_disp_log )`
* `above_decay  = S( u5_ask_pull_add_log_rest )`
* `above_local  = S( u7_ask_near_pull_share_rest - 0.5 )`
  (center at 0.5 so >0.5 becomes positive)
* `above_shock  = S( d2_u5_ask_pull_add_log_rest )`
  (acceleration of decay is the cleanest “shock” proxy)

**Below pressure = “support building below” (good for breaking up)**

* `below_approach = S( u8_bid_com_approach_log )`
* `below_build    = S( u12_bid_add_pull_log_rest )`
* `below_local    = S( u10_bid_near_share_rise )`
* `below_shock    = S( d2_u12_bid_add_pull_log_rest )`

### 38.5.2 If `approach_down` (support test)

**Above pressure = “resistance builds above” (good for breaking down)**
Use DOWN features but interpret “pressure above” as **what helps continuation down**:

* `above_retreat = S( f1_ask_com_disp_log )`
  (asks moving away up increases vacuum, helps down snap; keep as-is)
* `above_decay   = S( f2_ask_pull_add_log_rest )`
* `above_local   = S( f2_ask_near_pull_share_rest - 0.5 )`
* `above_shock   = S( d2_f2_ask_pull_add_log_rest )`

**Below pressure = “support evaporates below” (good for breaking down)**

* `below_recede = S( f3_bid_com_disp_log )`
* `below_decay  = S( f4_bid_pull_add_log_rest )`
* `below_local  = S( f4_bid_near_pull_share_rest - 0.5 )`
* `below_shock  = S( d2_f4_bid_pull_add_log_rest )`

---

## 38.6 Aggregate pressure scores (single scalars per side)

These are what the UI should primarily plot.

### 38.6.1 Per-side aggregate

For any side (above/below), define:

* `pressure_side = mean([retreat_or_recede, decay_or_build, local, shock])`

So you output:

* `pressure_above` (0..1)
* `pressure_below` (0..1)

### 38.6.2 Vacuum score (combined)

* `vacuum_score = mean([pressure_above, pressure_below])`

This is the “setup quality” gauge.

---

## 38.7 Discrete state machine (what UI displays)

Use only current window scalars and trigger engine outputs.

### 38.7.1 Fixed thresholds

* `WATCH_VACUUM = 0.60`
* `ARMED_VACUUM = 0.70`
* `FIRE_FLAG` is already computed by Section 34.

### 38.7.2 State definition

If `approach_dir == approach_none`:

* `state = "IDLE"`

Else:

* If `FIRE_FLAG == 1`: `state = "FIRE"`
* Else if `vacuum_score >= ARMED_VACUUM` and `margin >= 0.20`: `state = "ARMED"`
* Else if `vacuum_score >= WATCH_VACUUM`: `state = "WATCH"`
* Else: `state = "TRACK"`

---

## 38.8 Stream payload (exact JSON schema)

Emit exactly this structure (fields may be `null` only when `state == IDLE`):

```json
{
  "ts_end_ns": 0,
  "symbol": "ES",
  "session_date": "YYYY-MM-DD",
  "level_id": "PRE_MARKET_HIGH",
  "p_ref": 0.0,
  "approach_dir": "approach_up | approach_down | approach_none",

  "pressure": {
    "above": {
      "retreat": 0.0,
      "decay_or_build": 0.0,
      "localization": 0.0,
      "shock": 0.0,
      "score": 0.0
    },
    "below": {
      "retreat_or_recede": 0.0,
      "decay_or_build": 0.0,
      "localization": 0.0,
      "shock": 0.0,
      "score": 0.0
    },
    "vacuum_score": 0.0
  },

  "retrieval": {
    "h_fire": 1,
    "p_break": 0.0,
    "p_reject": 0.0,
    "p_chop": 0.0,
    "margin": 0.0,
    "risk_q80_ticks": 0.0,
    "resolve_rate": 0.0,
    "whipsaw_rate": 0.0
  },

  "signal": {
    "state": "IDLE | TRACK | WATCH | ARMED | FIRE",
    "fire_flag": 0,
    "signal": "LONG | SHORT | NONE",
    "episode_id": 0
  }
}
```

---
