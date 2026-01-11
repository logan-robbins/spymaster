## 40) Walk-Forward Backtest Runner (deterministic, leakage-free)

This runner replays historical MBO chronologically and produces: (a) pressure stream messages, (b) trigger decisions, (c) stop-aware horizon labels, (d) precision/recall trade metrics (ratios only).

### 40.1 Inputs (required)

1. **Sessions list**: ordered array of `session_date` (YYYY-MM-DD), strictly increasing.
2. **MBO source per session**: path/URI to raw Databento MBO events for that date.
3. **Levels per session**: for each `(session_date, level_id)` provide:

   * `p_ref` (float) and `P_REF_INT` (int)
   * `level_id` string (e.g., `PRE_MARKET_HIGH`)
4. **Symbol**: `ES`
5. **All constants** already fixed in earlier spec (ticks, windows, thresholds, horizons).

### 40.2 Determinism requirements (mandatory)

* Sort MBO by `(ts_event, sequence)` before processing.
* All window boundaries are computed from `WINDOW_NS` and `BAR_NS` via integer floor division.
* FAISS index build/search must run with **fixed threading** (set threads=1) for reproducibility.
* Use walk-forward insertion order (see 41.4) so HNSW construction is consistent across runs.

---

## 41) Per-session processing pipeline (run once per session_date, for each level_id)

You will run the following steps **for each** `level_id` independently (because bucket membership and “resting-in-bucket” timestamps are level-relative).

### 41.1 Precompute the 2-minute bar index (shared across levels)

From trade prints only (same as labeling spec):

1. Build 2-minute bars keyed by `bar_id = floor(ts / BAR_NS)`.
2. Store per bar: `high_int`, `low_int`, `close_int`, `bar_end_ts`.
3. Also build a time-ordered trade series `(ts_trade, trade_px_int)` to support barrier-hit scanning.

### 41.2 Run the 5-second feature engine (per level_id)

Execute the already-final “perfect” feature engine for that level:

1. Replay the session’s MBO events once.
2. Emit per 5s window:

   * all `f*`, `u*`, and all `d1/d2/d3`
   * `window_start_ts_ns`, `window_end_ts_ns`
   * `px_end_int` = last trade price at window end (for gating)

### 41.3 Build embeddings (per window)

For each 5s window `k`:

1. Build `x_k` (136-d) in canonical order.
2. Build `v_k` (952-d) via 5/15/45/120s rollup.
3. Apply robust scaling (MED/MAD) → `z_k`.
4. L2 normalize → `e_k`.

If `norm == 0`, mark window “embedding invalid” (do not query, do not insert).

### 41.4 Compute approach_dir + eligibility (per window)

Use the deterministic approach_dir definition already specified:

* compute `dist_ticks` from `px_end_int - P_REF_INT`
* compute 15s trend from `px_end_int[k] - px_end_int[k-3]`
* set `approach_up / approach_down / approach_none`
* eligibility requires: full 120s lookback, approach_dir != none, no book reset window, embedding valid.

### 41.5 Stop-aware horizon labels (per eligible window)

For each eligible window (trigger instance):

1. Define `trigger_ts = window_end_ts_ns`.
2. Compute `TRUE_OUTCOME_H0..H6` using barrier-first labeling:

   * barriers are `P_REF ± THRESH_TICKS`
   * outcome is determined by which barrier is hit **first** by time horizon end
   * no barrier hit by horizon → CHOP
   * tie → WHIPSAW
3. Compute and store risk metrics (tick distances only):

   * `first_hit_bar_offset`
   * `whipsaw_flag`
   * `mae_before_upper_ticks`, `mae_before_lower_ticks`
   * `mfe_up_ticks`, `mfe_down_ticks`

### 41.6 Retrieval prediction + trigger decision (per eligible window)

Using the **current FAISS indices that only contain prior sessions** (see Section 42 walk-forward ordering):

1. Query the correct index:

   * `INDEX[level_id][approach_dir]`
2. Compute weighted class distribution at `H_FIRE=1`:

   * `p_break`, `p_reject`, `p_chop`
   * `margin`, `c_top1`
3. Compute distribution-based constraints:

   * `risk_q80_ticks`, `resolve_rate`, `whipsaw_rate`
4. Apply the trigger rule:

   * output `FIRE_FLAG`, `SIGNAL`, `episode_id`, `state`
5. Emit the **pressure stream message** (Section 38), using approach_dir-appropriate features and retrieval outputs.

### 41.7 Per-window logging (mandatory)

Write one row per eligible trigger window containing only:

* identifiers (date, ts_end, level_id, approach_dir)
* prediction distributions + trigger diagnostics
* `TRUE_OUTCOME_H0..H6` and risk metrics
* **no PnL, no dollar metrics, no raw book sizes**

---

## 42) Walk-forward index build and evaluation (leakage-free by construction)

This ordering ensures the index never contains “future” or same-session neighbors at query time.

### 42.1 Define warmup requirement

* `WARMUP_SESSIONS = 20`
* If total sessions < 20: **abort** backtest (insufficient data to fit scaler and seed index).

### 42.2 Fit scalers (MED/MAD) using warmup sessions only

1. For each of the first 20 sessions:

   * compute embeddings `v_k` for all eligible windows for all levels
2. Pool all warmup `v_k` vectors across levels and approach_dirs.
3. Compute `MED[1..952]` and `MAD[1..952]`.
4. Freeze these scalers for the entire run.

### 42.3 Initialize FAISS indices (empty)

For each level_id:

* create `INDEX[level_id]["approach_up"]` and `INDEX[level_id]["approach_down"]` (cosine/IP HNSW as specified)

### 42.4 Walk-forward loop (core rule)

For sessions in chronological order:

#### Phase A: calibration run (sessions 1..WARMUP_SESSIONS)

For session `i`:

1. If `i == 1`: skip querying (index empty), **only compute labels and store per-window logs**.
2. If `i >= 2`:

   * query indices built from sessions `< i`
   * compute per-window prediction distributions + diagnostics
   * DO NOT apply a single threshold yet (store the prediction scalars required for thresholding)
3. After processing session `i`, **insert** that session’s eligible vectors into indices (see 42.5).

After session 20 completes, you now have:

* a leakage-free set of “predictions vs TRUE_OUTCOME_H*” from sessions 2..20
* seeded indices containing sessions 1..20

#### Phase B: evaluation run (sessions WARMUP+1..end)

1. Choose thresholds from calibration logs (Section 43).
2. For each session `i > 20`:

   * query indices built from sessions `< i`
   * apply trigger decision rule with chosen thresholds
   * log outcomes and emit pressure stream
   * insert session’s vectors after processing

### 42.5 Index insertion rule (what goes into FAISS)

Insert **only eligible windows** (approach_up or approach_down) with:

* vector `e_k`
* sidecar metadata:

  * `session_date`, `ts_end_ns`, `level_id`, `approach_dir`
  * `TRUE_OUTCOME_H1` (and optionally all H0..H6 in sidecar arrays)
  * `whipsaw_flag`, `first_hit_bar_offset`
  * `mae_before_upper_ticks`, `mae_before_lower_ticks`

Do not insert approach_none windows.

---

## 43) Threshold calibration runner (uses calibration logs only)

This is the “precision/recall tradeoff” step, fully deterministic.

### 43.1 Calibration dataset

Use only sessions 2..20 predictions (since 1 has empty index), across all levels.

Each row must already contain:

* `p_break`, `p_reject`, `p_chop`, `margin`, `c_top1`
* `risk_q80_ticks`, `resolve_rate`, `whipsaw_rate`
* `TRUE_OUTCOME_H1`, `whipsaw_flag`
* `mae_before_upper_ticks`, `mae_before_lower_ticks`
* `approach_dir`

### 43.2 Apply the fixed threshold grid (no discretion)

Evaluate all `(P_MIN, MARGIN_MIN)` pairs from the fixed grid (already specified).

For each pair:

1. Apply the fire rule to each calibration row (binary).
2. For fired rows compute:

   * correctness at H1: `c_top1 == TRUE_OUTCOME_H1` and `TRUE_OUTCOME_H1 != WHIPSAW`
   * `stop_violation` using the row’s own MAE metric appropriate to `c_top1`
3. Aggregate metrics (ratios only):

   * `precision`, `fire_rate`, `chop_false_rate`, `whipsaw_hit_rate`, `stop_violation_rate`, `resolve_by_bar1_rate`

### 43.3 Select thresholds deterministically

Pick the unique best pair using the selection rule already defined:

* satisfy constraints
* maximize `fire_rate * precision`
* tie-break by precision then resolve_by_bar1_rate

Freeze the chosen pair for Phase B evaluation.

---

## 44) Evaluation report outputs (what the backtest produces)

All outputs are ratios/distributions/binaries/tick distances only.

### 44.1 Per-trigger table (row = eligible 5s window)

Must include:

* identifiers: `session_date`, `ts_end_ns`, `level_id`, `approach_dir`, `episode_id`
* prediction: `p_break`, `p_reject`, `p_chop`, `margin`, `c_top1`
* trigger: `FIRE_FLAG`, `SIGNAL`, `state`
* constraints: `risk_q80_ticks`, `resolve_rate`, `whipsaw_rate`
* truth: `TRUE_OUTCOME_H0..H6`, `first_hit_bar_offset`, `whipsaw_flag`
* path risk: `mae_before_upper_ticks`, `mae_before_lower_ticks`, `mfe_up_ticks`, `mfe_down_ticks`

### 44.2 Per-session summary (group by session_date, level_id, approach_dir)

Compute at minimum:

* `eligible_count`
* `fires_count`
* `fire_rate = fires/eligible`
* `precision_H1 = correct_fires_H1 / fires`
* `chop_false_rate_H1`
* `stop_violation_rate_H1`
* `resolve_by_bar1_rate`
* Horizon precision curve for fired signals:

  * `precision_H0..precision_H6`

### 44.3 Global summary (across all evaluation sessions and all levels)

Same metrics as 44.2, aggregated:

* overall + broken out by `level_id` and by `approach_dir`

---

## 45) Mandatory sanity checks (runner must assert these)

1. **Label monotonicity:** for each trigger, once `TRUE_OUTCOME_H` becomes BREAK/REJECT at some H, it must remain the same for all larger H (WHIPSAW only allowed as tie case).
2. **Stop-aware consistency:** if `TRUE_OUTCOME_H1` is BREAK, then the opposite barrier must not have occurred earlier than the target barrier within that horizon.
3. **No leakage:** in walk-forward mode, ensure index contains only sessions strictly earlier than the session being evaluated.
4. **Non-finite cleanup:** confirm no NaN/Inf exists in any emitted probability/pressure fields (replace with 0 before output).
