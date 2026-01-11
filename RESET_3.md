## 20) Vector dimension count (based on **everything** we built) - COMPLETE

Per 5-second window you already emit:

* **DOWN orientation**: 17 base features × (base + d1 + d2 + d3) = **17 × 4 = 68**
* **UP orientation**: 17 base features × (base + d1 + d2 + d3) = **17 × 4 = 68**

So the **per-window feature frame** (no rollups) is:

* `x_k ∈ R^(68 + 68) = R^136`

That’s the atomic “state at window k”.

---

## 21) Build a **single vector that represents multiple lookback windows** (5 / 15 / 45 / 120 seconds) - COMPLETE

You will not concatenate raw windows (too time-shift sensitive). You will build a **multi-horizon rollup embedding** from the last 120 seconds that is stable and still encodes dynamics.

### 21.1 Fixed horizons (in 5s windows)

Because 1 window = 5s:

* `H1 = 1` window  (5s)
* `H3 = 3` windows (15s)
* `H9 = 9` windows (45s)
* `H24 = 24` windows (120s)

You will only create an embedding at window `k` if **k has at least 24 prior windows** available (i.e., a full 120s history exists).

---

## 22) Define the canonical **per-window vector** `x_k` (136-d) and its ordering - COMPLETE

You must use a deterministic ordering so historical + live vectors match bit-for-bit.

### 22.1 Canonical feature list (17) — DOWN (use these exact names/order)

`F_DOWN = [`

1. `f1_ask_com_disp_log`
2. `f1_ask_slope_convex_log`
3. `f1_ask_near_share_delta`
4. `f1_ask_reprice_away_share_rest`
5. `f2_ask_pull_add_log_rest`
6. `f2_ask_pull_intensity_rest`
7. `f2_ask_near_pull_share_rest`
8. `f3_bid_com_disp_log`
9. `f3_bid_slope_convex_log`
10. `f3_bid_near_share_delta`
11. `f3_bid_reprice_away_share_rest`
12. `f4_bid_pull_add_log_rest`
13. `f4_bid_pull_intensity_rest`
14. `f4_bid_near_pull_share_rest`
15. `f5_vacuum_expansion_log`
16. `f6_vacuum_decay_log`
17. `f7_vacuum_total_log`
    `]`

### 22.2 Canonical feature list (17) — UP (use these exact names/order)

`F_UP = [`

1. `u1_ask_com_disp_log`
2. `u2_ask_slope_convex_log`
3. `u3_ask_near_share_decay`
4. `u4_ask_reprice_away_share_rest`
5. `u5_ask_pull_add_log_rest`
6. `u6_ask_pull_intensity_rest`
7. `u7_ask_near_pull_share_rest`
8. `u8_bid_com_approach_log`
9. `u9_bid_slope_support_log`
10. `u10_bid_near_share_rise`
11. `u11_bid_reprice_toward_share_rest`
12. `u12_bid_add_pull_log_rest`
13. `u13_bid_add_intensity`
14. `u14_bid_far_pull_share_rest`
15. `u15_up_expansion_log`
16. `u16_up_flow_log`
17. `u17_up_total_log`
    `]`

### 22.3 Assemble `x_k` (136 dims)

For each `name` in `F_DOWN` in order, append **exactly**:

* `[ name, d1_name, d2_name, d3_name ]`

This yields 17×4=68 dims.

Then for each `name` in `F_UP` in order, append:

* `[ name, d1_name, d2_name, d3_name ]`

Total:

* `x_k ∈ R^136`

### 22.4 Non-finite safety rule (mandatory)

Before `x_k` is used anywhere:

* Replace any `NaN`, `+inf`, `-inf` with `0`.

---

## 23) Multi-horizon rollup embedding `v_k` (952-d) - COMPLETE

You will convert the last 120 seconds of `x` frames into **one** embedding.

### 23.1 Required rollup statistics per dimension

For each component `j` of `x` (j=1..136), define:

Let `x_t = x_k[j]` (current window), and let `x_{k-m}[j]` be the value m windows ago.

Compute **exactly these 7 numbers**:

1. `last_5s`

* `r1 = x_k[j]`

2. `mean_15s` (last 3 windows)

* `r2 = mean(x_{k-2}[j], x_{k-1}[j], x_k[j])`

3. `slope_15s` (trend across last 3 windows)

* `r3 = (x_k[j] - x_{k-2}[j]) / 2`

4. `mean_45s` (last 9 windows)

* `r4 = mean(x_{k-8}[j] ... x_k[j])`

5. `slope_45s`

* `r5 = (x_k[j] - x_{k-8}[j]) / 8`

6. `mean_120s` (last 24 windows)

* `r6 = mean(x_{k-23}[j] ... x_k[j])`

7. `slope_120s`

* `r7 = (x_k[j] - x_{k-23}[j]) / 23`

### 23.2 Final vector order and dimension

Construct `v_k` by iterating `j = 1..136` in order and appending:

* `[r1, r2, r3, r4, r5, r6, r7]`

So:

* `v_k ∈ R^(136 × 7) = R^952`

---

## 24) Normalize for FAISS (mandatory, fixed pipeline) - COMPLETE

You must make distances meaningful. Do **robust per-dimension scaling**, then cosine normalization.

### 24.1 Fit robust scalers offline (one-time)

Using a large historical corpus of `v_k` vectors for the instrument:

For each dimension `d` (1..952), compute:

* `MED[d]` = median over all historical vectors
* `MAD[d]` = median(|v[d] - MED[d]|) over all historical vectors

Store `MED[1..952]`, `MAD[1..952]`.

### 24.2 Apply scaling online (every vector)

Given `v_k`, produce `z_k`:

For each dimension `d`:

* `z[d] = (v[d] - MED[d]) / (1.4826 * MAD[d] + 1e-6)`
* Clip: `z[d] = min(8, max(-8, z[d]))`

### 24.3 L2 normalize (cosine-ready)

* `norm = sqrt(sum(z[d]^2))`
* If `norm == 0`: do not index/query this vector.
* Else: `e_k = z / norm`

This `e_k` is the final FAISS vector.

---

## 25) Decide *when* a vector is eligible (gating) - COMPLETE

You only care about level-conditional states near the level. Enforce hard gating so the index doesn’t fill with irrelevant states.

### 25.1 Compute end-of-window price (required)

From MBO:

* Maintain `last_trade_price_int` updated on trade prints (your feed will have a trade action; use whatever your implementation uses consistently).
* At each window end, define `px_end_int = last_trade_price_int`.

Compute:

* `dist_ticks = (px_end_int - P_REF_INT) / TICK_INT`  (can be negative)

### 25.2 Define approach direction (required, deterministic)

At window `k`, compute short-term price trend using 15s:

* `px_end_int[k]` and `px_end_int[k-3]` (15s ago)
* `trend = px_end_int[k] - px_end_int[k-3]`

Then:

* `approach_up = (dist_ticks < 0) AND (abs(dist_ticks) <= 20) AND (trend > 0)`
* `approach_down = (dist_ticks > 0) AND (abs(dist_ticks) <= 20) AND (trend < 0)`
* else: `approach_none`

### 25.3 Eligibility rule

A vector `e_k` is **eligible** only if all are true:

1. Full lookback exists: windows `k-23 .. k` exist
2. `approach_dir ∈ {approach_up, approach_down}`
3. The window was not invalidated by a book reset (`action == 'R'` rule from prior spec)

---

## 26) FAISS index layout (fixed, no filtering tricks) - COMPLETE

FAISS doesn’t do metadata filtering well; you will enforce separations via multiple indices.

### 26.1 Indices to maintain

For each `level_id` you track (e.g., `"PRE_MARKET_HIGH"`), maintain exactly two FAISS indices:

* `INDEX[level_id]["approach_up"]`   (vectors eligible under approach_up)
* `INDEX[level_id]["approach_down"]` (vectors eligible under approach_down)

Each index has:

* dimension `d = 952`
* metric = **inner product** (because vectors are L2-normalized → cosine similarity)

### 26.2 Index type (mandatory)

Use an approximate index suitable for large scale:

* HNSW with inner product:

  * `M = 32`
  * `efConstruction = 200`
  * `efSearch = 64`

(You can change performance params later, but the *logic* assumes cosine/IP on normalized vectors.)

---

## 27) Metadata sidecar (required) - COMPLETE

FAISS returns integer IDs. You must store metadata in aligned arrays keyed by the insertion ID.

For each inserted vector, store:

* `id` (monotonic int)
* `ts_end_ns`
* `session_date`
* `symbol`
* `level_id`
* `P_ref` (float) and `P_REF_INT` (int)
* `approach_dir` (`approach_up` or `approach_down`)
* **Outcome labels** (Section 28)

---

## 31) **Stop-aware, bar-indexed labeling** (this is the missing stage) - COMPLETE

You will label **every eligible 5s trigger window** with **path-dependent outcomes** measured on **2-minute bars**, with **chop/noise** and **stop-out-before-target** handled explicitly.

This labeling is what lets you answer:

* “Was the setup correct?”
* “Was it correct by trigger bar +1? +2?”
* “Did it stop-out first (invalidating the break/reject)?”

---

## 31.1 Fixed constants for labeling (use exactly these)

### Bar resolution

* `BAR_NS = 120_000_000_000` (2 minutes)

### Lookahead

* `N_BARS = 6` (evaluate up to 12 minutes forward)
* You will compute results for **every** horizon `H ∈ {0,1,2,3,4,5,6}` where:

  * `H=0` = through the **end of the trigger bar**
  * `H=1` = through the **end of trigger bar +1**
  * …
  * `H=6` = through the **end of trigger bar +6**

### Break / reject thresholds (level-anchored barriers)

* `THRESH_TICKS = 8` (2.00 ES points)
* `UPPER_BARRIER_INT = P_REF_INT + THRESH_TICKS * TICK_INT`
* `LOWER_BARRIER_INT = P_REF_INT - THRESH_TICKS * TICK_INT`

**Trader rule encoded:** For an “up” outcome to be valid, the **down barrier must NOT be hit first**, and vice versa. This is exactly the “stop hit before target nullifies the break/reject” requirement.

### Deterministic tie-break

If both barriers are hit at the exact same timestamp (rare):

* Treat as `WHIPSAW` (noise) and count as incorrect for both directions.

---

## 31.2 Build the 2-minute bar index (from trade prices)

You must build a time → bar mapping for the session using a single canonical trade price series.

1. Define `trade_px_int(ts)` as the last traded price at timestamp `ts` (from trade prints).
2. For each trade print with timestamp `ts_trade`, assign:

   * `bar_id = floor(ts_trade / BAR_NS)`
3. For each `bar_id`, store:

   * `bar_start_ts = bar_id * BAR_NS`
   * `bar_end_ts = bar_start_ts + BAR_NS`
   * `high_int = max(trade_px_int within bar)`
   * `low_int  = min(trade_px_int within bar)`
   * `close_int = last trade_px_int in bar`

You will use this bar index only for reporting “bar+1 / bar+2” horizons.
Actual barrier hits must be determined using event timestamps (Section 31.4).

---

## 31.3 Define the “trigger instance” (one per eligible 5s window)

For every eligible vector window `k` (per the gating rules you already implemented):

1. `trigger_ts = window_end_ts_ns[k]` (end of the 5s feature window)
2. `trigger_bar_id = floor(trigger_ts / BAR_NS)`
3. Define horizon end timestamps:

   * For each `H ∈ {0..N_BARS}`
     `horizon_end_ts(H) = (trigger_bar_id + H + 1) * BAR_NS`
     (end of bar `trigger_bar_id + H`)

Store these once per trigger instance:

* `trigger_ts`
* `trigger_bar_id`
* `horizon_end_ts(0..N_BARS)`

---

## 31.4 Compute **first-hit times** to the two barriers (path-dependent, timestamp-accurate)

For each trigger instance, you must compute barrier hit order using the **trade prints** between `trigger_ts` and `horizon_end_ts(N_BARS)`.

### 31.4.1 Scan rules

Scan the trade price stream (in time order) for trades with:

* `ts_trade ∈ (trigger_ts, horizon_end_ts(N_BARS)]`

Track the earliest timestamps:

* `t_hit_upper` = first `ts_trade` where `trade_px_int >= UPPER_BARRIER_INT`
* `t_hit_lower` = first `ts_trade` where `trade_px_int <= LOWER_BARRIER_INT`

If never hit, value is `None`.

### 31.4.2 Determine the “first hit” (with whipsaw handling)

* If both are `None` → `first_hit = NONE`
* If only one exists → `first_hit = UPPER` or `LOWER`
* If both exist:

  * If `t_hit_upper < t_hit_lower` → `first_hit = UPPER`
  * If `t_hit_lower < t_hit_upper` → `first_hit = LOWER`
  * If equal → `first_hit = WHIPSAW`

Also compute:

* `first_hit_ts` = corresponding timestamp (or None)
* `first_hit_bar_offset = floor(first_hit_ts / BAR_NS) - trigger_bar_id` (if hit)
* `whipsaw_flag = 1` if both hits exist within lookahead (regardless of order), else 0
* `second_hit_ts` and `second_hit_bar_offset` if both exist and not equal (store; used as a “chop/noise risk” metric)

---

## 31.5 Convert first-hit side into **true outcome class** (depends on approach direction)

You already compute `approach_dir ∈ {approach_up, approach_down}` for each trigger.

For each trigger, define the **true semantic outcome** using `first_hit`:

### If `approach_up` (price rising into resistance)

* `first_hit == UPPER` → `TRUE_OUTCOME = BREAK_UP`
* `first_hit == LOWER` → `TRUE_OUTCOME = REJECT_DOWN`
* `first_hit == NONE`  → `TRUE_OUTCOME = CHOP`
* `first_hit == WHIPSAW` → `TRUE_OUTCOME = WHIPSAW`

### If `approach_down` (price falling into support)

* `first_hit == LOWER` → `TRUE_OUTCOME = BREAK_DOWN`
* `first_hit == UPPER` → `TRUE_OUTCOME = REJECT_UP`
* `first_hit == NONE`  → `TRUE_OUTCOME = CHOP`
* `first_hit == WHIPSAW` → `TRUE_OUTCOME = WHIPSAW`

This is your **ground truth** for “break vs reject vs chop/noise,” and it is **stop-aware** because whichever barrier hits first invalidates the opposite-direction trade.

---

## 31.6 Horizon-specific labels: “correct by trigger bar +H”

You must generate labels for each `H ∈ {0..N_BARS}` so you can test bar+1, bar+2, etc.

For each `H`:

1. Determine whether either barrier was hit **by** `horizon_end_ts(H)`:

   * `upper_hit_by_H = (t_hit_upper != None) AND (t_hit_upper <= horizon_end_ts(H))`
   * `lower_hit_by_H = (t_hit_lower != None) AND (t_hit_lower <= horizon_end_ts(H))`

2. Determine `first_hit_by_H`:

   * If neither hit → `NONE`
   * If only one hit → that one
   * If both hit → compare timestamps (same tie rule → `WHIPSAW`)

3. Map `first_hit_by_H` into `TRUE_OUTCOME_H` using the same approach_dir mapping from Section 31.5.

Store:

* `TRUE_OUTCOME_H0, TRUE_OUTCOME_H1, ..., TRUE_OUTCOME_H6`

This is exactly what allows:

* “accuracy at trigger bar +1” = compare predictions to `TRUE_OUTCOME_H1`
* “accuracy at trigger bar +2” = compare predictions to `TRUE_OUTCOME_H2`

---

## 31.7 Trader-centric risk metrics (mandatory; used to judge stop pressure even when correct)

For each trigger instance, compute **excursions** over the lookahead window to quantify “did it almost stop out / did it fake out”:

Over `ts ∈ (trigger_ts, horizon_end_ts(N_BARS)]`:

* `min_px_int = min(trade_px_int)`
* `max_px_int = max(trade_px_int)`

Compute:

* `mfe_up_ticks = max(0, (max_px_int - P_REF_INT) / TICK_INT)`  (best push above level)
* `mfe_down_ticks = max(0, (P_REF_INT - min_px_int) / TICK_INT)` (best push below level)

Also compute **pre-resolution adverse excursion** for each direction (captures “stop risk before target”):

* If `first_hit == UPPER` (up resolved first):

  * `mae_before_upper_ticks = max(0, (P_REF_INT - min_px_int before t_hit_upper) / TICK_INT)`
* If `first_hit == LOWER` (down resolved first):

  * `mae_before_lower_ticks = max(0, (max_px_int before t_hit_lower - P_REF_INT) / TICK_INT)`
* If `NONE/WHIPSAW`: compute both over the whole horizon.

Store these as metadata alongside the vector. This keeps the trader goal front-and-center: “Even if it breaks, did it threaten the stop first?”

---

# 32) Using these labels to measure “did we identify the trigger state correctly?” - COMPLETE

You will evaluate the model (retrieval vote, later transformer, etc.) against `TRUE_OUTCOME_H`.

## 32.1 Define the model’s predicted class at each trigger

At each trigger, your FAISS retrieval returns a neighbor label distribution. Convert that into one predicted class **deterministically**:

* `PRED_OUTCOME = argmax( p(BREAK_*), p(REJECT_*), p(CHOP) )`
* If `CHOP` is not explicitly modeled, define it as:
  `p(CHOP) = 1 - p(BREAK_*) - p(REJECT_*)` (clamp to [0,1])

Do this separately per `approach_dir` (since the class set differs):

* approach_up classes: `{BREAK_UP, REJECT_DOWN, CHOP}`
* approach_down classes: `{BREAK_DOWN, REJECT_UP, CHOP}`
  (`WHIPSAW` is never predicted; it is treated as noise in evaluation.)

---

## 32.2 Accuracy at trigger bar +H (what you asked for)

For each horizon `H ∈ {0..6}` compute:

1. **Directional accuracy**

* Count a prediction correct if:

  * `PRED_OUTCOME == TRUE_OUTCOME_H`
* Treat `TRUE_OUTCOME_H == WHIPSAW` as incorrect for all predictions.

2. **Trade-valid accuracy (stop-aware by construction)**
   This is already enforced: if the “wrong-way” barrier hit first by time H, `TRUE_OUTCOME_H` flips and the prediction becomes incorrect. That matches: “stop-out first nullifies the break.”

3. **Chop handling**

* If `TRUE_OUTCOME_H == CHOP` and you predicted BREAK/REJECT → that is a false trigger.
* If you predicted CHOP and it later resolves after H → correct for H, incorrect for longer horizons where resolution occurs.

Store/report the full curve:

* `acc_H0, acc_H1, acc_H2, ..., acc_H6`

This answers directly: “are we most accurate at bar+1 or bar+2?”

---

## 32.3 Additional must-report metrics (because traders care)

For each horizon H, compute these rates:

* `hit_break_H` = fraction where `PRED == BREAK_*` and `TRUE_OUTCOME_H == BREAK_*`
* `hit_reject_H` = fraction where `PRED == REJECT_*` and `TRUE_OUTCOME_H == REJECT_*`
* `false_break_H` = predicted BREAK but true is REJECT or CHOP or WHIPSAW
* `false_reject_H` = predicted REJECT but true is BREAK or CHOP or WHIPSAW
* `overtrade_H` = predicted BREAK/REJECT when true is CHOP (classic noise/chop failure)

Also compute:

* distribution of `first_hit_bar_offset` for correct predictions
  (how often it resolves in 0,1,2 bars—this is your timing edge).

---

**IMPORTANT**
You must label each 2 minute bar with a candle_id starting from 0830EST (1 hour pre-market) they should be 0 indexed. COMPLETE
