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
