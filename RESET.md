# Level-Conditional Liquidity Vacuum Features (ES)

Compute a 5-second state vector around a single predetermined level (P_{ref}) that captures **(1) Ask retreat**, **(2) Ask decay (resting pull > add)**, **(3) Bid recession**, **(4) Bid evaporation (resting pull > add)** using **Databento MBO** events.

## Assets

### SCHEMA

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Market Data Message",
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "ts_recv": {
        "type": "integer",
        "description": "Capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch."
      },
      "size": {
        "type": "integer",
        "description": "The order quantity."
      },
      "ts_event": {
        "type": "integer",
        "description": "Matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch."
      },
      "channel_id": {
        "type": "integer",
        "minimum": 0,
        "description": "Channel ID assigned by Databento as an incrementing integer starting at zero."
      },
      "rtype": {
        "type": "integer",
        "description": "Record type. Each schema corresponds with a single rtype value."
      },
      "order_id": {
        "type": "integer",
        "description": "Order ID assigned at the venue."
      },
      "publisher_id": {
        "type": "integer",
        "description": "Publisher ID assigned by Databento, denoting dataset and venue."
      },
      "flags": {
        "type": "integer",
        "description": "Bit field indicating event end, message characteristics, and data quality."
      },
      "instrument_id": {
        "type": "integer",
        "description": "Numeric instrument ID."
      },
      "ts_in_delta": {
        "type": "integer",
        "description": "Matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv."
      },
      "action": {
        "type": "string",
        "enum": ["A", "C", "M", "R", "T", "F", "N"],
        "description": "Event action: Add, Cancel, Modify, Clear book, Trade, Fill, or None."
      },
      "sequence": {
        "type": "integer",
        "description": "Message sequence number assigned at the venue."
      },
      "side": {
        "type": "string",
        "enum": ["A", "B", "N"],
        "description": "Initiating side: Ask, Bid, or None."
      },
      "symbol": {
        "type": "string",
        "description": "Requested symbol for the instrument."
      },
      "price": {
        "type": "integer",
        "description": "Order price as a signed integer where 1 unit equals 1e-9 (0.000000001)."
      }
    },
    "required": [
      "ts_recv",
      "size",
      "ts_event",
      "channel_id",
      "rtype",
      "order_id",
      "publisher_id",
      "flags",
      "instrument_id",
      "ts_in_delta",
      "action",
      "sequence",
      "side",
      "symbol",
      "price"
    ]
}  
```

### SAMPLE DATA

- mbo_preview.json

---

## 0) Inputs (from raw MBO)

Each event row provides at least:

* `ts_event` (int, ns since epoch)
* `action` ∈ {`A`,`C`,`M`,`R`,`T`,`F`,`N`}
* `side` ∈ {`A`=Ask, `B`=Bid, `N`}
* `price` (int, where **1 unit = 1e-9** price)
* `size` (int qty)
* `order_id` (int)
* `sequence` (int; use for deterministic ordering)
* `symbol` (string)

Also provided externally:

* `P_ref` (float price for the level being monitored; e.g., pre-market high)
* `symbol_target` (e.g., `"ES"`)

---

## 1) Fixed constants (use exactly these)

### Price / tick

* `PRICE_SCALE = 1e-9`
* `TICK_SIZE = 0.25` (ES)
* `TICK_INT = round(TICK_SIZE / PRICE_SCALE) = 250_000_000`
* `P_REF_INT = round(P_ref / PRICE_SCALE)` (convert once)

### Spatial bandwidth (local influence zone)

* `DELTA_TICKS = 20` (influence zone = 5.00 points)
* `NEAR_TICKS = 5` (near slice = 1.25 points)
* `FAR_TICKS_LOW = 15`
* `FAR_TICKS_HIGH = 20` (far slice = last 1.50–5.00 points in zone)

### Time bucketing + “resting” threshold

* `WINDOW_NS = 5_000_000_000` (5s)
* `REST_NS = 500_000_000` (500ms; resting-in-band filter)

### Numerics

* `EPS_QTY = 1` (contracts)
* `EPS_DIST_TICKS = 1` (tick)
* `LOG(x)` = natural log

---

## 2) Deterministic ordering

1. Filter rows where `symbol == symbol_target`.
2. Sort ascending by `(ts_event, sequence)`.
3. Drop events with `action == 'N'` (no-op).

---

## 3) Define band buckets relative to (P_{ref})

You must assign **every active order** to exactly one bucket at all times based on its current `(side, price)`.

### 3.1 Helper: ticks away from level

All divisions below are **integer ticks** (assume ES prices fall on tick grid; if not, round to nearest tick before dividing).

* For asks (`side='A'`) above level:
  `ticks = (price_int - P_REF_INT) / TICK_INT`
* For bids (`side='B'`) below level:
  `ticks = (P_REF_INT - price_int) / TICK_INT`

### 3.2 Buckets

**Ask-side buckets (only if price_int > P_REF_INT):**

* If `ticks < 1` or `ticks > DELTA_TICKS` → `OUT`
* Else if `ticks <= NEAR_TICKS` → `ASK_ABOVE_NEAR`
* Else if `ticks >= FAR_TICKS_LOW` → `ASK_ABOVE_FAR`
* Else → `ASK_ABOVE_MID`

**Bid-side buckets (only if price_int < P_REF_INT):**

* If `ticks < 1` or `ticks > DELTA_TICKS` → `OUT`
* Else if `ticks <= NEAR_TICKS` → `BID_BELOW_NEAR`
* Else if `ticks >= FAR_TICKS_LOW` → `BID_BELOW_FAR`
* Else → `BID_BELOW_MID`

If `side == 'N'` → `OUT`.

### 3.3 Influence membership

* Ask influence = bucket ∈ {`ASK_ABOVE_NEAR`,`ASK_ABOVE_MID`,`ASK_ABOVE_FAR`}
* Bid influence = bucket ∈ {`BID_BELOW_NEAR`,`BID_BELOW_MID`,`BID_BELOW_FAR`}

---

## 4) Maintain the “shadow book” (per `order_id`)

Maintain a map `orders[order_id] -> OrderState`:

* `side` (`A` or `B`)
* `price_int` (current)
* `qty` (current remaining)
* `bucket` (current band bucket from Section 3)
* `bucket_enter_ts` (ts_event when the order **most recently entered this bucket**)

**Bucket-enter rule:**
Whenever an order’s bucket changes (including `OUT -> IN` or `IN -> OUT`, or `NEAR <-> MID <-> FAR`), set `bucket_enter_ts = current ts_event`.

---

## 5) Windowing control

Define:

* `window_id = floor(ts_event / WINDOW_NS)`
* `window_start_ts = window_id * WINDOW_NS`
* `window_end_ts = window_start_ts + WINDOW_NS`

You will emit **one feature row per (window_id)**.

---

## 6) Per-window accumulators (reset every window)

Maintain these counters for the current window (W_k):

### Ask-above (influence zone)

* `ask_add_qty` (total added quantity into ask influence zone)
* `ask_pull_rest_qty` (quantity removed from ask influence zone via **C/M**, **resting-in-bucket**)
* `ask_pull_rest_qty_near` (subset of `ask_pull_rest_qty` removed from `ASK_ABOVE_NEAR`)
* `ask_reprice_away_rest_qty` (resting-in-bucket, modify moved farther from level)
* `ask_reprice_toward_rest_qty` (resting-in-bucket, modify moved closer to level)

### Bid-below (influence zone)

* `bid_add_qty`
* `bid_pull_rest_qty` (via **C/M**, resting-in-bucket)
* `bid_pull_rest_qty_near` (subset from `BID_BELOW_NEAR`)
* `bid_reprice_away_rest_qty`
* `bid_reprice_toward_rest_qty`

**Critical intent rule:**
Events with `action == 'F'` (fills) update the book state but do **not** contribute to pull/add/reprice features (you are measuring “provider intent”, not taker consumption).

---

## 7) Snapshot metrics at window start and end

You must compute the following from the **current shadow book state**:

For each snapshot time (t), define:

### 7.1 Ask-above snapshot (all active asks in influence zone)

Let (S^{ask}(t)) be all active orders with:

* `side='A'` AND bucket ∈ {`ASK_ABOVE_NEAR`,`ASK_ABOVE_MID`,`ASK_ABOVE_FAR`}

Compute:

* `ask_depth_total(t) = Σ qty`
* `ask_depth_near(t) = Σ qty where bucket==ASK_ABOVE_NEAR`
* `ask_depth_far(t)  = Σ qty where bucket==ASK_ABOVE_FAR`
* `ask_com_price_int(t) = (Σ price_int * qty) / max(ask_depth_total(t), EPS_QTY)`

### 7.2 Bid-below snapshot (all active bids in influence zone)

Let (S^{bid}(t)) be all active orders with:

* `side='B'` AND bucket ∈ {`BID_BELOW_NEAR`,`BID_BELOW_MID`,`BID_BELOW_FAR`}

Compute:

* `bid_depth_total(t)`
* `bid_depth_near(t)`
* `bid_depth_far(t)`
* `bid_com_price_int(t) = (Σ price_int * qty) / max(bid_depth_total(t), EPS_QTY)`

### 7.3 Convert COM displacement to ticks (always positive)

At any snapshot time (t):

* Ask displacement ticks:
  `D_ask_ticks(t) = max((ask_com_price_int(t) - P_REF_INT) / TICK_INT, 0)`
* Bid displacement ticks:
  `D_bid_ticks(t) = max((P_REF_INT - bid_com_price_int(t)) / TICK_INT, 0)`

---

## 8) Event processing (this is the core)

Process events sequentially (already sorted). For each event `e`:

### 8.1 Window boundary handling

If `event.window_id` changes from `current_window_id`:

1. Compute **end snapshot** for the previous window (Section 7 at end-of-window state).
2. Compute features for that window (Section 9).
3. Emit feature row.
4. Reset all accumulators (Section 6).
5. Set new window start snapshot from current state (Section 7 at start).
6. Update `current_window_id`.

### 8.2 Handle action == 'R' (clear book)

When `action == 'R'`:

1. Clear `orders` completely.
2. Reset all accumulators to zero.
3. Mark the **current window as invalid** and emit **no row** for it.
4. Resume processing from the next event; treat the next event’s window as a fresh start (start snapshot from empty book).

(Reason: `R` is feed/book reset, not market intent; including it would create false “mass pull” signals.)

### 8.3 Read old state (pre-event)

Let `old = orders.get(order_id)`; if missing:

* For `A`: proceed (new order).
* For `C/M/F`: ignore the event for feature counting and state updates (cannot infer deltas reliably).

If present, record:

* `old_side`, `old_price_int`, `old_qty`, `old_bucket`, `old_bucket_enter_ts`

Define:

* `old_in_ask_influence = (old_side=='A' AND old_bucket in ask influence buckets)`
* `old_in_bid_influence = (old_side=='B' AND old_bucket in bid influence buckets)`

### 8.4 Apply the event to produce the new state

#### action == 'A' (add)

* Create/overwrite `orders[order_id]` with:

  * `side = e.side`
  * `price_int = e.price`
  * `qty = e.size`
  * `bucket = bucket(side, price_int)` (Section 3)
  * `bucket_enter_ts = e.ts_event`

#### action == 'C' (cancel)

* Remove the order from `orders` (new state = non-existent)

#### action == 'M' (modify)

* Update existing order to:

  * `price_int = e.price`
  * `qty = e.size`
  * `bucket = recompute bucket(...)`
  * If bucket changed: set `bucket_enter_ts = e.ts_event` else keep old `bucket_enter_ts`

#### action == 'F' (fill)

* Reduce existing order:

  * `qty_new = old_qty - e.size`
  * If `qty_new <= 0`: remove order
  * Else update `qty = qty_new` (bucket unchanged; bucket_enter_ts unchanged)

#### action == 'T'

* Do not change the shadow book state (treat as informational trade print).

---

## 9) Update per-window accumulators from each event (using band-delta logic)

You must compute **add vs pull** as **quantity deltas inside the influence zone**, so that reprices across boundaries count correctly.

### 9.1 Compute q_old_in_zone and q_new_in_zone

For asks:

* `q_old_ask_zone = old_qty if old_in_ask_influence else 0`
* `q_new_ask_zone = new_qty if new_state exists AND side=='A' AND new_bucket in ask influence buckets else 0`

For bids:

* `q_old_bid_zone = old_qty if old_in_bid_influence else 0`
* `q_new_bid_zone = new_qty if new_state exists AND side=='B' AND new_bucket in bid influence buckets else 0`

Define deltas:

* `delta_ask = q_new_ask_zone - q_old_ask_zone`
* `delta_bid = q_new_bid_zone - q_old_bid_zone`

### 9.2 Count adds (unfiltered)

If `delta_ask > 0`: `ask_add_qty += delta_ask`
If `delta_bid > 0`: `bid_add_qty += delta_bid`

This counts:

* true adds (`A`) into the zone,
* modifies that move into the zone,
* size increases inside the zone.

### 9.3 Count pulls (resting-only, intent-only)

A “pull” is any **C/M** event that removes quantity from the zone, but you only count it if the liquidity was **resting in its bucket**.

For asks:
If `delta_ask < 0` AND `action ∈ {'C','M'}`:

1. `age_ns = e.ts_event - old_bucket_enter_ts`
2. If `age_ns >= REST_NS`:

   * `pull = -delta_ask`
   * `ask_pull_rest_qty += pull`
   * If `old_bucket == 'ASK_ABOVE_NEAR'`: `ask_pull_rest_qty_near += pull`

For bids:
If `delta_bid < 0` AND `action ∈ {'C','M'}`:

1. `age_ns = e.ts_event - old_bucket_enter_ts`
2. If `age_ns >= REST_NS`:

   * `pull = -delta_bid`
   * `bid_pull_rest_qty += pull`
   * If `old_bucket == 'BID_BELOW_NEAR'`: `bid_pull_rest_qty_near += pull`

**Do not count `action == 'F'` as pull**, even though it reduces zone quantity.

---

## 10) Count “reprice away vs toward” (resting-only, modify-only)

These features capture Condition (1) and (3) as **direct evidence of the book stepping away from the level**, independent of cancels.

### 10.1 Ask-side repricing (Condition 1)

If `action == 'M'` AND `old_side=='A'` AND `old_bucket` is in ask influence buckets:

1. `age_ns = e.ts_event - old_bucket_enter_ts`
2. If `age_ns >= REST_NS`:

   * `dist_old = (old_price_int - P_REF_INT) / TICK_INT` (ticks above)
   * If `new_price_int <= P_REF_INT`: treat as **toward**
   * Else `dist_new = (new_price_int - P_REF_INT) / TICK_INT`
   * If `dist_new > dist_old`: `ask_reprice_away_rest_qty += old_qty`
   * If `dist_new < dist_old`: `ask_reprice_toward_rest_qty += old_qty`
   * If equal: add nothing

### 10.2 Bid-side repricing (Condition 3)

If `action == 'M'` AND `old_side=='B'` AND `old_bucket` is in bid influence buckets:

1. `age_ns = e.ts_event - old_bucket_enter_ts`
2. If `age_ns >= REST_NS`:

   * `dist_old = (P_REF_INT - old_price_int) / TICK_INT` (ticks below)
   * If `new_price_int >= P_REF_INT`: treat as **toward**
   * Else `dist_new = (P_REF_INT - new_price_int) / TICK_INT`
   * If `dist_new > dist_old`: `bid_reprice_away_rest_qty += old_qty`
   * If `dist_new < dist_old`: `bid_reprice_toward_rest_qty += old_qty`
   * If equal: add nothing

---

## 11) Compute features at the end of each 5s window

At window finalization you already have:

* start snapshot metrics (Section 7 at window start)
* end snapshot metrics (Section 7 at window end)
* accumulators (Sections 6, 9, 10)

Compute exactly these **base features**:

### 11.1 Condition 1 — Ask Retreat

Let (D^{ask}*{start} = D_ask_ticks(t*{start})), (D^{ask}*{end} = D_ask_ticks(t*{end}))

1. **Ask CoM Displacement Log Ratio**

* `f1_ask_com_disp_log = LOG((D_ask_end + EPS_DIST_TICKS) / (D_ask_start + EPS_DIST_TICKS))`

2. **Ask Slope Convexity Log Ratio (Far vs Near depth)**

* `f1_ask_slope_convex_log = LOG((ask_depth_far_end + EPS_QTY) / (ask_depth_near_end + EPS_QTY))`

3. **Ask Near Share Delta**

* `near_share_start = ask_depth_near_start / (ask_depth_total_start + EPS_QTY)`
* `near_share_end   = ask_depth_near_end   / (ask_depth_total_end   + EPS_QTY)`
* `f1_ask_near_share_delta = near_share_end - near_share_start`

4. **Ask Reprice Away Share (Resting-only)**

* `den = ask_reprice_away_rest_qty + ask_reprice_toward_rest_qty`
* If `den == 0`: `f1_ask_reprice_away_share_rest = 0.5`
* Else `f1_ask_reprice_away_share_rest = ask_reprice_away_rest_qty / (den + EPS_QTY)`

---

### 11.2 Condition 2 — Resistance Decay (Resting Pull > Add)

5. **Ask Resting Pull/Add Log Ratio**

* `f2_ask_pull_add_log_rest = LOG((ask_pull_rest_qty + EPS_QTY) / (ask_add_qty + EPS_QTY))`

6. **Ask Resting Pull Intensity (normalized by starting depth)**

* `f2_ask_pull_intensity_rest = ask_pull_rest_qty / (ask_depth_total_start + EPS_QTY)`

7. **Ask Near Pull Share (Resting-only)**

* `f2_ask_near_pull_share_rest = ask_pull_rest_qty_near / (ask_pull_rest_qty + EPS_QTY)`

---

### 11.3 Condition 3 — Bid Recession

Let (D^{bid}*{start} = D_bid_ticks(t*{start})), (D^{bid}*{end} = D_bid_ticks(t*{end}))

8. **Bid CoM Displacement Log Ratio**

* `f3_bid_com_disp_log = LOG((D_bid_end + EPS_DIST_TICKS) / (D_bid_start + EPS_DIST_TICKS))`

9. **Bid Slope Convexity Log Ratio (Far vs Near depth)**

* `f3_bid_slope_convex_log = LOG((bid_depth_far_end + EPS_QTY) / (bid_depth_near_end + EPS_QTY))`

10. **Bid Near Share Delta**

* `near_share_start = bid_depth_near_start / (bid_depth_total_start + EPS_QTY)`
* `near_share_end   = bid_depth_near_end   / (bid_depth_total_end   + EPS_QTY)`
* `f3_bid_near_share_delta = near_share_end - near_share_start`

11. **Bid Reprice Away Share (Resting-only)**

* `den = bid_reprice_away_rest_qty + bid_reprice_toward_rest_qty`
* If `den == 0`: `f3_bid_reprice_away_share_rest = 0.5`
* Else `f3_bid_reprice_away_share_rest = bid_reprice_away_rest_qty / (den + EPS_QTY)`

---

### 11.4 Condition 4 — Support Evaporation (Resting Pull > Add)

12. **Bid Resting Pull/Add Log Ratio**

* `f4_bid_pull_add_log_rest = LOG((bid_pull_rest_qty + EPS_QTY) / (bid_add_qty + EPS_QTY))`

13. **Bid Resting Pull Intensity (normalized by starting depth)**

* `f4_bid_pull_intensity_rest = bid_pull_rest_qty / (bid_depth_total_start + EPS_QTY)`

14. **Bid Near Pull Share (Resting-only)**

* `f4_bid_near_pull_share_rest = bid_pull_rest_qty_near / (bid_pull_rest_qty + EPS_QTY)`

---

## 12) Compute enforced “both-sides-must-happen” composites (still derived only from 1–4)

These are included because your hypothesis requires **simultaneity**.

15. **Vacuum Expansion Log** (asks retreat + bids recede)

* `f5_vacuum_expansion_log = f1_ask_com_disp_log + f3_bid_com_disp_log`

16. **Vacuum Decay Log** (both sides pulling)

* `f6_vacuum_decay_log = f2_ask_pull_add_log_rest + f4_bid_pull_add_log_rest`

17. **Vacuum Total Log**

* `f7_vacuum_total_log = f5_vacuum_expansion_log + f6_vacuum_decay_log`

---

## 13) Derivatives (d1/d2/d3) for every feature (1–17)

For each base feature `fX`:

* `d1_fX[k] = fX[k] - fX[k-1]`
* `d2_fX[k] = d1_fX[k] - d1_fX[k-1]`
* `d3_fX[k] = d2_fX[k] - d2_fX[k-1]`

For the first window(s) where prior values don’t exist, set missing derivatives to `0`.

---

## 14) Final emitted row schema (per window)

Emit exactly one row per valid 5s window:

* `window_start_ts_ns`
* `window_end_ts_ns`
* `P_ref` (float) and/or `P_REF_INT` (int)
* The 17 base features: `f1_...` through `f7_...`
* All derivatives: `d1_*, d2_*, d3_*` for each of the 17 features

**Do not include raw quantities in the final emitted row.** (Raw values are allowed internally during computation only.)

## Plan

1. [x] Add dataset config and contract for the liquidity vacuum feature rows.
2. [x] Implement MBO feature computation + stage using `mbo_preview.json`.
3. [x] Register the new dataset and stage in the pipeline.
4. [x] Verify the stage on the preview data with `uv run`.
5. [x] Validate all action/bucket cases using preview-derived scenarios.
6. [x] Run the validation harness on preview data.

## 16) Build the **UPWARD PUSH / BREAK-UP** feature vector (price pushes **up through** (P_{ref}))

You will **not** change any raw processing, shadow-book logic, bucketing, or windowing from the previous spec. You will compute the original 17 features (Sections 11–12) exactly as written, then deterministically transform them into a new 17-feature vector whose **polarity is “UP”**.

### 16.1 Naming

* Let the original (already emitted) features be `f1..f7` exactly as defined.
* Create a second set of features with prefix `u_` (UP-oriented) and emit them alongside (plus their d1/d2/d3, Section 16.4).

### 16.2 UP-oriented feature transforms (explicit 1-to-1 mapping)

**Ask-side (resistance above level) — keep the “ghosting” mechanics**
These features already represent *resistance retreat/decay*, which is supportive of an upward push. Keep them, except where a sign must be flipped so “more ghosting” is numerically larger.

1. `u1_ask_com_disp_log = f1_ask_com_disp_log`

2. `u2_ask_slope_convex_log = f1_ask_slope_convex_log`

3. `u3_ask_near_share_decay = -f1_ask_near_share_delta`
   Reason: for an UP push, “near ask share goes down” is supportive; flipping makes “more decay” larger.

4. `u4_ask_reprice_away_share_rest = f1_ask_reprice_away_share_rest`
   (“Away” = asks repricing higher away from the level → supportive)

5. `u5_ask_pull_add_log_rest = f2_ask_pull_add_log_rest`

6. `u6_ask_pull_intensity_rest = f2_ask_pull_intensity_rest`

7. `u7_ask_near_pull_share_rest = f2_ask_near_pull_share_rest`

---

**Bid-side (support below level) — invert from “evaporation” to “support building”**
Your base bid features (`f3`, `f4`) were oriented so positive meant **bids moving down/pulling** (support loss). For an UP push you need the opposite: bids **move up toward** the level and **adds dominate** pulls.

8. `u8_bid_com_approach_log = -f3_bid_com_disp_log`
   (Approach toward level = distance decreases)

9. `u9_bid_slope_support_log = -f3_bid_slope_convex_log`
   (Base was `log(far/near)`; support building makes near larger → invert)

10. `u10_bid_near_share_rise = f3_bid_near_share_delta`
    (Already aligned: near share increasing is supportive)

11. `u11_bid_reprice_toward_share_rest = 1 - f3_bid_reprice_away_share_rest`
    (Base “away” = bids repricing lower; invert to “toward” = repricing higher)

12. `u12_bid_add_pull_log_rest = -f4_bid_pull_add_log_rest`
    (Base was `log(pull/add)`; invert to represent `log(add/pull)`)

13. **Replace the “pull intensity” concept with its true inverse: add intensity**
    Compute using the same internal intermediates already required by the base engine (`bid_add_qty`, `bid_depth_total_start`):

* `u13_bid_add_intensity = bid_add_qty / (bid_depth_total_start + EPS_QTY)`

14. **Invert the location of any remaining pulls** (if pulls exist, you prefer them NOT to be near the level):

* `u14_bid_far_pull_share_rest = 1 - f4_bid_near_pull_share_rest`

---

**UP composites (must be recomputed from the UP-oriented components, not from base f5–f7):**
15) `u15_up_expansion_log = u1_ask_com_disp_log + u8_bid_com_approach_log`

16. `u16_up_flow_log = u5_ask_pull_add_log_rest + u12_bid_add_pull_log_rest`

17. `u17_up_total_log = u15_up_expansion_log + u16_up_flow_log`

### 16.3 Output rules (still enforced)

* Do **not** output raw quantities.
* Output only `u1..u17`, plus their derivatives in Section 16.4.

### 16.4 Derivatives for UP features

For every `uX` (X=1..17), compute:

* `d1_uX[k] = uX[k] - uX[k-1]`
* `d2_uX[k] = d1_uX[k] - d1_uX[k-1]`
* `d3_uX[k] = d2_uX[k] - d2_uX[k-1]`
  Missing prior values → set to `0`.

---

## 17) Use the engine for **price coming DOWN toward (P_{ref})** (support test)

Nothing in the raw feature computation changes. What changes is **which oriented vector corresponds to “rejection” vs “breakthrough”**.

### 17.1 Define two canonical outcome vectors (always computed every window)

You will always compute both:

* **DOWN outcome vector** = the original base features (already defined in prior spec):
  `DOWN_VECTOR = { f1..f7 and their d1/d2/d3 }`
  Interpretation: “Support below is receding/evaporating” (and concurrent above ghosting per your base spec), which is consistent with **downward continuation away from the level**.

* **UP outcome vector** = the UP-oriented features (Section 16):
  `UP_VECTOR = { u1..u17 and their d1/d2/d3 }`
  Interpretation: “Support below is strengthening while resistance above is ghosting,” consistent with **upward movement away from / through the level**.

You will not conditionally compute one or the other. Compute both every window.

---

## 18) **Approach-direction mapping** (how to interpret “rejection” vs “breakthrough”)

Define `approach_dir` externally from your price stream (last trade / mid / 2m candle trend). This spec does not dictate how you compute it; it only dictates how to **map outcomes** once you have it.

### 18.1 If price is coming **UP** toward (P_{ref}) (resistance test)

* **Rejection (down from level)** → use `DOWN_VECTOR` (`f*`)
* **Breakthrough (up through level)** → use `UP_VECTOR` (`u*`)

### 18.2 If price is coming **DOWN** toward (P_{ref}) (support test)

* **Rejection (up from level / bounce)** → use `UP_VECTOR` (`u*`)
* **Breakthrough (down through level / continuation)** → use `DOWN_VECTOR` (`f*`)

This is the only “flip” required for the second permutation: **the label “rejection” swaps which outcome-vector it points to when the approach direction flips.**

---

## 19) Final emit requirement for multi-permutation use

For every 5s window, emit a single row containing:

* the original base feature set: `f1..f7`, `d1_f*`, `d2_f*`, `d3_f*`
* the UP-oriented feature set: `u1..u17`, `d1_u*`, `d2_u*`, `d3_u*`

Downstream systems (retrieval / classification / visualization) must select:

* `UP_VECTOR` vs `DOWN_VECTOR` according to Sections 18.1–18.2 based on approach direction and the semantic meaning of “reject” vs “break.”

---

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

---

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

---