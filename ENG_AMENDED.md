# MBP-10 Feature Engineering Specification (Amended)

## Section 0 — Data Contract

### 0.1 Input Source

- **Provider:** Databento
- **Schema:** MBP-10 (Market-By-Price, 10 levels)
- **Instrument:** ES Futures (E-mini S&P 500)
- **Tick Size:** 0.25 points (4 ticks = 1 point)

### 0.2 Input Columns

Each row represents a single book event. The relevant columns are:

| Column | Type | Description |
|--------|------|-------------|
| `ts_event` | uint64 | Event timestamp in **nanoseconds** since Unix epoch (UTC) |
| `ts_recv` | uint64 | Receive timestamp (not used for feature computation) |
| `action` | uint8 | Event action type (see mapping below) |
| `side` | char | `'B'` = Bid, `'A'` = Ask |
| `price` | int64 | Price in fixed-point (divide by 1e9 to get float; for ES, typically price × 1e9) |
| `size` | uint32 | Size at this level after the event |
| `depth` | uint8 | Level index (0 = best bid/ask, 9 = deepest visible) |
| `flags` | uint8 | Bitfield flags (not used for this spec) |
| `sequence` | uint64 | Sequence number for ordering within same timestamp |
| `bid_px_00` ... `bid_px_09` | int64 | Bid prices at levels 0–9 (post-event state) |
| `bid_sz_00` ... `bid_sz_09` | uint32 | Bid sizes at levels 0–9 (post-event state) |
| `bid_ct_00` ... `bid_ct_09` | uint32 | Bid order counts at levels 0–9 (post-event state) |
| `ask_px_00` ... `ask_px_09` | int64 | Ask prices at levels 0–9 (post-event state) |
| `ask_sz_00` ... `ask_sz_09` | uint32 | Ask sizes at levels 0–9 (post-event state) |
| `ask_ct_00` ... `ask_ct_09` | uint32 | Ask order counts at levels 0–9 (post-event state) |

**Price Conversion:**
- Raw prices are in fixed-point integer format
- To convert to float: `price_float = price_int / 1e9`
- For ES futures, prices will be values like 6800.00, 6800.25, etc.

### 0.3 Action Mapping

| Action Value | Name | Description |
|--------------|------|-------------|
| 0 | Modify | Existing order modified (size change at a price level) |
| 1 | Clear | Book cleared (all levels wiped) |
| 2 | Add | New order added to the book |
| 3 | Cancel | Order canceled/removed from the book |
| 4 | Trade | Trade executed (aggressor removes liquidity) |
| 5 | Fill | (Treat same as Trade for this spec) |

### 0.4 Side Mapping

| Side Value | Name | Description |
|------------|------|-------------|
| `'B'` or `'b'` | Bid | Buy side of the book |
| `'A'` or `'a'` | Ask | Ask/Offer side of the book |

For **Trade** events, `side` indicates the **resting order's side** that was hit:
- `side = 'A'` → Aggressor was a buyer (bought from resting ask) → **Agg Buy**
- `side = 'B'` → Aggressor was a seller (sold to resting bid) → **Agg Sell**

### 0.5 Book State Interpretation

**Critical:** The `bid_px_00..09`, `bid_sz_00..09`, `ask_px_00..09`, `ask_sz_00..09` arrays in each row represent the **post-event** book state.

To compute features correctly, you must maintain:
- **Pre-event state:** The book arrays from the **previous** event
- **Post-event state:** The book arrays in the **current** row

When processing event at row `i`:
1. Pre-state = book arrays from row `i-1` (or initialized state if `i=0`)
2. Post-state = book arrays from row `i`
3. Compute features using pre-state where specified
4. After processing, pre-state ← post-state for next iteration

### 0.6 Timestamp and Bar Boundaries

**Timestamp unit:** Nanoseconds (1 second = 1,000,000,000 nanoseconds)

**5-second bar boundaries:**
```
bar_duration_ns = 5_000_000_000  (5 seconds in nanoseconds)
bar_start = floor(ts_event / bar_duration_ns) * bar_duration_ns
bar_end = bar_start + bar_duration_ns
```

**Boundary ownership rule:** An event with `ts_event` exactly equal to a bar boundary belongs to the **new** bar (the bar that starts at that timestamp), not the closing bar.

### 0.7 Processing Order

Events must be processed in strict order:
1. Primary sort: `ts_event` ascending
2. Secondary sort: `sequence` ascending (for events with identical timestamps)

### 0.8 Initialization

**At the start of each trading session (or data file):**
- Pre-state book arrays initialized to the first event's post-state arrays
- `t_last` initialized to the first event's `ts_event`
- All TWA accumulators initialized to 0
- All SUM accumulators initialized to 0

**At the start of each 5-second bar:**
- All SUM accumulators reset to 0
- TWA accumulators reset to 0
- `t_last` set to `bar_start` (not the first event's time)
- Pre-state book arrays carry forward from end of previous bar

### 0.9 Empty Bar Handling

If a 5-second bar contains **zero events**:
- Output a row for that bar
- EOB features = prior bar's EOB values (state carried forward)
- TWA features = same as EOB (state was constant for full 5 seconds)
- All SUM features = 0 (no events occurred)
- `bar5s_meta_msg_cnt_sum` = 0

### 0.10 Output Format

- **Format:** Parquet (columnar, compressed with snappy)
- **One row per 5-second bar**
- **Timestamp column:** `bar_ts` = bar start time in nanoseconds (UTC)
- **Symbol column:** `symbol` = instrument identifier (e.g., "ESH5")
- **All feature columns:** float64 (even counts, for consistency and NaN support)

### 0.11 Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `POINT` | 1.0 | One index point (4 ticks for ES) |
| `BAR_DURATION_NS` | 5,000,000,000 | 5 seconds in nanoseconds |
| `EPSILON` | 1e-9 | Small value for numerical stability in divisions |
| `WALL_Z_THRESHOLD` | 2.0 | Z-score threshold for "strong" wall detection |

### 0.12 Edge Case Handling

| Scenario | Handling |
|----------|----------|
| Division by zero (e.g., empty book side) | Return 0.0 (not NaN) for imbalance ratios; use `max(denominator, EPSILON)` |
| Price level > 10 points from P_ref | Exclude from banded features; do not clamp |
| Price level ≤ 0 points from P_ref | Include in `p0_1` band |
| No wall found (no Z ≥ threshold) | Set `dist_pts = NaN`, `levelidx = -1` |
| Negative size (data error) | Clamp to 0 |
| Price = 0 or invalid | Skip event for flow calculations; still update book state |
| Book side completely empty | All sizes = 0; depth = 0; imbalances = 0.0 |

---

## Section 1 — Naming Convention

All features are 5-second bars, so prefix everything with:

* `bar5s_`

Then use a consistent structure:

* `bar5s_<family>_<detail>_<agg>`

Where:

* `<family>` ∈ `{meta, state, depth, ladder, shape, flow, trade, wall}`
* `<agg>` ∈ `{twa, eob, sum}`

  * `twa` = time-weighted average over the 5s bar
  * `eob` = end-of-bar snapshot (state at bar close)
  * `sum` = sum over events in the bar (for flows/trades/counts)

**No feature contains an absolute price.**
Allowed: differences (spread), distances-from-reference (in points), ratios, counts, sizes.

---

## Section 2 — Constants and Internal-Only Values

### 2.1 Point Distance Unit

* Define `POINT = 1.0` in ES/NQ price terms
* ES tick size = 0.25, so 1 point = 4 ticks
* All distance features are expressed in points, not ticks

### 2.2 Internal-Only Reference Price (Not Output as a Feature)

Compute at each event from the **pre-event** top-of-book:

Let:
* `B0_px` = `bid_px_00` (best bid price, pre-event)
* `A0_px` = `ask_px_00` (best ask price, pre-event)
* `B0_sz` = `bid_sz_00` (best bid size, pre-event)
* `A0_sz` = `ask_sz_00` (best ask size, pre-event)

Internal microprice (size-weighted mid):
```
P_ref(t) = (A0_px(t) × B0_sz(t) + B0_px(t) × A0_sz(t)) / (B0_sz(t) + A0_sz(t))
```

**Edge case:** If `B0_sz + A0_sz = 0`, use simple midpoint:
```
P_ref(t) = (A0_px(t) + B0_px(t)) / 2
```

You will **not** store `P_ref` as a feature; it's only used to define relative bands/distances.

### 2.3 Point-Distance Bands (Relative to P_ref)

Use these point ranges (designed for MBP-10 depth):

| Band Name | Distance Range (points) |
|-----------|------------------------|
| `p0_1` | 0 < d ≤ 1 |
| `p1_2` | 1 < d ≤ 2 |
| `p2_3` | 2 < d ≤ 3 |
| `p3_5` | 3 < d ≤ 5 |
| `p5_10` | 5 < d ≤ 10 |

Distance in **points** for a book level at price `p`:

* Ask side: `d = (p - P_ref) / POINT`
* Bid side: `d = (P_ref - p) / POINT`

**Band assignment rules:**
* If `d ≤ 0`, assign to `p0_1` band (price at or through the reference)
* If `d > 10`, **exclude** from all banded features (do not clamp)
* Assign to the **first** band where `d` falls within the range

---

## Section 3 — Implementation Order

### Step 1 — Sort and Bar Events

Process the feed in ascending order:
* Primary: `ts_event`
* Secondary: `sequence`

Define 5-second bar boundaries:
* `bar_start = floor(ts_event / BAR_DURATION_NS) * BAR_DURATION_NS`
* `bar_end = bar_start + BAR_DURATION_NS`

Maintain a bar accumulator structure that resets at each bar boundary.

### Step 2 — Maintain Pre-Event and Post-Event Book State

For each event, you need access to **both** the pre-event and post-event book state.

**Pre-event state** (arrays before this event):
* `pre_bid_px_00..09`, `pre_bid_sz_00..09`, `pre_bid_ct_00..09`
* `pre_ask_px_00..09`, `pre_ask_sz_00..09`, `pre_ask_ct_00..09`

**Post-event state** (arrays from current row):
* `bid_px_00..09`, `bid_sz_00..09`, `bid_ct_00..09`
* `ask_px_00..09`, `ask_sz_00..09`, `ask_ct_00..09`

**State transition per event:**
1. Read current row's arrays as post-state
2. Compute features using pre-state (for TWA accumulation, P_ref, flow deltas)
3. After processing: `pre_state ← post_state`

Also maintain:
* `t_last` = timestamp of last processed event (for TWA dt calculation)

### Step 3 — TWA (Time-Weighted Average) Accumulation

For each state variable `X(t)` that needs TWA:

**Within a bar, when an event arrives at time `t_e`:**
1. Compute `dt = t_e - t_last` (in nanoseconds)
2. Accumulate: `sum_X += X(pre_state) × dt`
3. Update: `t_last = t_e`
4. Then update pre_state ← post_state

**At bar close (when crossing into a new bar or at end of data):**
1. Compute `dt_end = bar_end - t_last`
2. Final accumulation: `sum_X += X(current_pre_state) × dt_end`
3. Compute TWA: `X_twa = sum_X / BAR_DURATION_NS`

**Important:** `t_last` at bar start should be set to `bar_start`, so the first event's `dt` captures time from bar start to first event.

### Step 4 — Flow Accumulation (Sum Over Events)

Flows capture the **change** in book liquidity from Add/Cancel/Modify events.

#### 4.1 Compute Per-Event ΔQ for Add/Cancel/Modify

For an event with `action ∈ {Add, Cancel, Modify}` (action values 0, 2, 3):

**Identify the affected price `p_evt`:**
* Use the `price` field from the event row

**Compute previous size at that price (from pre-state):**
* Scan `pre_bid_px_00..09` (if side='B') or `pre_ask_px_00..09` (if side='A')
* Find index where price matches `p_evt` (use tolerance of 0 for exact match on integers)
* If found: `Q_prev = pre_<side>_sz_<index>`
* If not found in top 10: `Q_prev = 0`

**Compute new size at that price (from post-state):**
* Scan `bid_px_00..09` or `ask_px_00..09` for `p_evt`
* If found: `Q_new = <side>_sz_<index>`
* If not found in top 10: `Q_new = 0`

**Compute delta:**
```
ΔQ = Q_new - Q_prev
AddVol = max(ΔQ, 0)
RemVol = max(-ΔQ, 0)
```

#### 4.2 Bucket the Event into a Point Band

Compute point distance using **pre-event** P_ref:
```
P_ref = microprice from pre-state (Section 2.2)

If side = 'A' (Ask):
    d = (p_evt - P_ref) / POINT
If side = 'B' (Bid):
    d = (P_ref - p_evt) / POINT
```

Assign to band:
* `d ≤ 0` → `p0_1`
* `0 < d ≤ 1` → `p0_1`
* `1 < d ≤ 2` → `p1_2`
* `2 < d ≤ 3` → `p2_3`
* `3 < d ≤ 5` → `p3_5`
* `5 < d ≤ 10` → `p5_10`
* `d > 10` → **exclude** (do not accumulate)

#### 4.3 Accumulate Flow Sums

Within each bar, maintain accumulators per side (`bid`, `ask`) and per band (`p0_1`, `p1_2`, `p2_3`, `p3_5`, `p5_10`):

**Volume accumulators:**
* `flow_add_vol_<side>_<band>_sum += AddVol`
* `flow_rem_vol_<side>_<band>_sum += RemVol`
* `flow_net_vol_<side>_<band>_sum += (AddVol - RemVol)`

**Count accumulators (by action type):**
* If `action = 2` (Add): `flow_cnt_add_<side>_<band>_sum += 1`
* If `action = 3` (Cancel): `flow_cnt_cancel_<side>_<band>_sum += 1`
* If `action = 0` (Modify): `flow_cnt_modify_<side>_<band>_sum += 1`

#### 4.4 Trade Events (Separate from Flow)

For `action = 4` (Trade) or `action = 5` (Fill):

**Do NOT compute ΔQ or include in flow accumulators.** Trades are tracked separately.

Accumulate:
* `trade_vol_sum += size`
* `trade_cnt_sum += 1`

By aggressor side (remember: `side` in trade event = resting side):
* If `side = 'A'`: aggressor was buyer → `trade_aggbuy_vol_sum += size`
* If `side = 'B'`: aggressor was seller → `trade_aggsell_vol_sum += size`

Signed trade volume:
* `trade_signed_vol_sum = trade_aggbuy_vol_sum - trade_aggsell_vol_sum`

#### 4.5 Clear Events

For `action = 1` (Clear):

* Increment `meta_clear_cnt_sum += 1`
* Optionally compute clear removal volume (total depth wiped):
```
ClearRemVol = sum(pre_bid_sz_00..09) + sum(pre_ask_sz_00..09)
```
* Accumulate separately: `meta_clear_vol_sum += ClearRemVol`

**Do NOT include clear events in flow_rem accumulators** — they are tracked separately to avoid polluting normal cancel/modify removal signals.

---

## Section 4 — Feature Definitions

### A) Meta Features (SUM)

Event counts over the bar:

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_meta_msg_cnt_sum` | Total events in bar (all action types) |
| `bar5s_meta_clear_cnt_sum` | Count of Clear events (action=1) |
| `bar5s_meta_add_cnt_sum` | Count of Add events (action=2) |
| `bar5s_meta_cancel_cnt_sum` | Count of Cancel events (action=3) |
| `bar5s_meta_modify_cnt_sum` | Count of Modify events (action=0) |
| `bar5s_meta_trade_cnt_sum` | Count of Trade events (action=4 or 5) |

---

### B) State Features (TWA + EOB)

All state features use **pre-event** book state for TWA accumulation and **final post-event** state for EOB.

#### B1) Spread in Points

```
Spread_pts(t) = (A0_px(t) - B0_px(t)) / POINT
```

| Feature Name | Aggregation |
|--------------|-------------|
| `bar5s_state_spread_pts_twa` | Time-weighted average of Spread_pts |
| `bar5s_state_spread_pts_eob` | End-of-bar Spread_pts |

#### B2) Top-of-Book Imbalance (OBI0)

```
OBI0(t) = (B0_sz(t) - A0_sz(t)) / (B0_sz(t) + A0_sz(t) + EPSILON)
```

Range: [-1, +1] where +1 = all bid, -1 = all ask

| Feature Name | Aggregation |
|--------------|-------------|
| `bar5s_state_obi0_twa` | Time-weighted average |
| `bar5s_state_obi0_eob` | End-of-bar snapshot |

#### B3) Total Depth Imbalance (OBI10)

```
BidDepth10(t) = sum(bid_sz_00..09)
AskDepth10(t) = sum(ask_sz_00..09)
OBI10(t) = (BidDepth10(t) - AskDepth10(t)) / (BidDepth10(t) + AskDepth10(t) + EPSILON)
```

| Feature Name | Aggregation |
|--------------|-------------|
| `bar5s_state_obi10_twa` | Time-weighted average |
| `bar5s_state_obi10_eob` | End-of-bar snapshot |

#### B4) Cross-Depth Imbalance by Band (CDI)

For each band `band ∈ {p0_1, p1_2, p2_3, p3_5, p5_10}`:

```
BelowQty(band, t) = sum of bid sizes where point distance d falls in band
AboveQty(band, t) = sum of ask sizes where point distance d falls in band

CDI(band, t) = (BelowQty - AboveQty) / (BelowQty + AboveQty + EPSILON)
```

| Feature Name | Aggregation |
|--------------|-------------|
| `bar5s_state_cdi_<band>_twa` | Time-weighted average |
| `bar5s_state_cdi_<band>_eob` | End-of-bar snapshot |

---

### C) Depth Features (TWA + EOB)

#### C1) Total Depth by Side

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_depth_bid10_qty_twa` | TWA of sum(bid_sz_00..09) |
| `bar5s_depth_bid10_qty_eob` | EOB sum(bid_sz_00..09) |
| `bar5s_depth_ask10_qty_twa` | TWA of sum(ask_sz_00..09) |
| `bar5s_depth_ask10_qty_eob` | EOB sum(ask_sz_00..09) |

#### C2) Banded Depth (Absolute Quantities)

For each band:

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_depth_below_<band>_qty_twa` | TWA of BelowQty(band) |
| `bar5s_depth_below_<band>_qty_eob` | EOB BelowQty(band) |
| `bar5s_depth_above_<band>_qty_twa` | TWA of AboveQty(band) |
| `bar5s_depth_above_<band>_qty_eob` | EOB AboveQty(band) |

#### C3) Banded Depth (Fractional)

```
BelowFrac(band) = BelowQty(band) / (BidDepth10 + EPSILON)
AboveFrac(band) = AboveQty(band) / (AskDepth10 + EPSILON)
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_depth_below_<band>_frac_twa` | TWA of BelowFrac(band) |
| `bar5s_depth_below_<band>_frac_eob` | EOB BelowFrac(band) |
| `bar5s_depth_above_<band>_frac_twa` | TWA of AboveFrac(band) |
| `bar5s_depth_above_<band>_frac_eob` | EOB AboveFrac(band) |

---

### D) Ladder Geometry Features (EOB Only)

Price gaps between adjacent levels, in points. No absolute prices output.

#### D1) Ask Gaps

```
AskGapPts_n = (ask_px_{n+1} - ask_px_n) / POINT,  for n = 0..8
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_ladder_ask_gap_max_pts_eob` | max(AskGapPts_0..8) |
| `bar5s_ladder_ask_gap_mean_pts_eob` | mean(AskGapPts_0..8) |

#### D2) Bid Gaps

```
BidGapPts_n = (bid_px_n - bid_px_{n+1}) / POINT,  for n = 0..8
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_ladder_bid_gap_max_pts_eob` | max(BidGapPts_0..8) |
| `bar5s_ladder_bid_gap_mean_pts_eob` | mean(BidGapPts_0..8) |

**Edge case:** If a level has price = 0 (empty level), exclude that gap from min/max/mean calculation. If fewer than 2 valid levels, output NaN.

---

### E) Shape Features (EOB Only)

Per-level book shape. No prices, only sizes and counts.

#### E1) Raw Sizes

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_shape_bid_sz_l00_eob` ... `bar5s_shape_bid_sz_l09_eob` | bid_sz_00..09 at bar end |
| `bar5s_shape_ask_sz_l00_eob` ... `bar5s_shape_ask_sz_l09_eob` | ask_sz_00..09 at bar end |

#### E2) Raw Counts

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_shape_bid_ct_l00_eob` ... `bar5s_shape_bid_ct_l09_eob` | bid_ct_00..09 at bar end |
| `bar5s_shape_ask_ct_l00_eob` ... `bar5s_shape_ask_ct_l09_eob` | ask_ct_00..09 at bar end |

#### E3) Fractional Sizes

```
BidSzFrac_n = bid_sz_n / (sum(bid_sz_00..09) + EPSILON)
AskSzFrac_n = ask_sz_n / (sum(ask_sz_00..09) + EPSILON)
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_shape_bid_sz_frac_l00_eob` ... `bar5s_shape_bid_sz_frac_l09_eob` | BidSzFrac_0..9 |
| `bar5s_shape_ask_sz_frac_l00_eob` ... `bar5s_shape_ask_sz_frac_l09_eob` | AskSzFrac_0..9 |

#### E4) Fractional Counts

```
BidCtFrac_n = bid_ct_n / (sum(bid_ct_00..09) + EPSILON)
AskCtFrac_n = ask_ct_n / (sum(ask_ct_00..09) + EPSILON)
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_shape_bid_ct_frac_l00_eob` ... `bar5s_shape_bid_ct_frac_l09_eob` | BidCtFrac_0..9 |
| `bar5s_shape_ask_ct_frac_l00_eob` ... `bar5s_shape_ask_ct_frac_l09_eob` | AskCtFrac_0..9 |

---

### F) Flow Features (SUM)

All flow features are sums over events in the bar.

#### F1) Volume Flows by Side and Band

For each `<side>` ∈ `{bid, ask}` and `<band>` ∈ `{p0_1, p1_2, p2_3, p3_5, p5_10}`:

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_flow_add_vol_<side>_<band>_sum` | Sum of AddVol for this side+band |
| `bar5s_flow_rem_vol_<side>_<band>_sum` | Sum of RemVol for this side+band |
| `bar5s_flow_net_vol_<side>_<band>_sum` | Sum of (AddVol - RemVol) for this side+band |

#### F2) Event Counts by Side, Band, and Action

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_flow_cnt_add_<side>_<band>_sum` | Count of Add events |
| `bar5s_flow_cnt_cancel_<side>_<band>_sum` | Count of Cancel events |
| `bar5s_flow_cnt_modify_<side>_<band>_sum` | Count of Modify events |

#### F3) Normalized Net Volume (Optional but Recommended)

```
NetVolNorm = NetVol / max(DepthBandQty_twa, 1)
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_flow_net_volnorm_<side>_<band>_sum` | Normalized net volume |

---

### G) Trade Features (SUM)

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_trade_cnt_sum` | Total trade count |
| `bar5s_trade_vol_sum` | Total trade volume |
| `bar5s_trade_aggbuy_vol_sum` | Volume from aggressive buyers |
| `bar5s_trade_aggsell_vol_sum` | Volume from aggressive sellers |
| `bar5s_trade_signed_vol_sum` | aggbuy_vol - aggsell_vol |

---

### H) Wall Features (EOB Only)

Wall detection identifies unusually large resting orders relative to the local book.

#### H1) Z-Score Computation

For each side (bid/ask), at end of bar:

1. Let `Q_n` = size at level n (n = 0..9)
2. Compute log-transformed sizes: `q_n = ln(1 + Q_n)`
3. Compute mean: `μ = mean(q_0..q_9)`
4. Compute std: `σ = std(q_0..q_9)` (population std, not sample)
5. Compute z-scores: `Z_n = (q_n - μ) / max(σ, EPSILON)`

#### H2) Max Z-Score

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_wall_bid_maxz_eob` | max(Z_0..Z_9) for bid side |
| `bar5s_wall_ask_maxz_eob` | max(Z_0..Z_9) for ask side |
| `bar5s_wall_bid_maxz_levelidx_eob` | argmax(Z_n) for bid (0..9) |
| `bar5s_wall_ask_maxz_levelidx_eob` | argmax(Z_n) for ask (0..9) |

#### H3) Nearest Strong Wall

A "strong wall" is defined as a level with `Z_n ≥ WALL_Z_THRESHOLD` (default 2.0).

For each side, find the **closest** level (smallest n) where `Z_n ≥ threshold`:

**Distance in points:**
```
For bid: dist = (P_ref - bid_px_n) / POINT
For ask: dist = (ask_px_n - P_ref) / POINT
```

| Feature Name | Computation |
|--------------|-------------|
| `bar5s_wall_bid_nearest_strong_dist_pts_eob` | Point distance to nearest strong bid wall |
| `bar5s_wall_ask_nearest_strong_dist_pts_eob` | Point distance to nearest strong ask wall |
| `bar5s_wall_bid_nearest_strong_levelidx_eob` | Level index of nearest strong bid wall |
| `bar5s_wall_ask_nearest_strong_levelidx_eob` | Level index of nearest strong ask wall |

**If no strong wall found on a side:** Set `dist_pts = NaN`, `levelidx = -1`

---

## Section 5 — Complete Feature List

### 5.1 Meta (6 features)

```
bar5s_meta_msg_cnt_sum
bar5s_meta_clear_cnt_sum
bar5s_meta_add_cnt_sum
bar5s_meta_cancel_cnt_sum
bar5s_meta_modify_cnt_sum
bar5s_meta_trade_cnt_sum
```

### 5.2 State (22 features)

```
bar5s_state_spread_pts_twa
bar5s_state_spread_pts_eob
bar5s_state_obi0_twa
bar5s_state_obi0_eob
bar5s_state_obi10_twa
bar5s_state_obi10_eob
bar5s_state_cdi_p0_1_twa
bar5s_state_cdi_p0_1_eob
bar5s_state_cdi_p1_2_twa
bar5s_state_cdi_p1_2_eob
bar5s_state_cdi_p2_3_twa
bar5s_state_cdi_p2_3_eob
bar5s_state_cdi_p3_5_twa
bar5s_state_cdi_p3_5_eob
bar5s_state_cdi_p5_10_twa
bar5s_state_cdi_p5_10_eob
```

### 5.3 Depth (44 features)

```
bar5s_depth_bid10_qty_twa
bar5s_depth_bid10_qty_eob
bar5s_depth_ask10_qty_twa
bar5s_depth_ask10_qty_eob

bar5s_depth_below_p0_1_qty_twa
bar5s_depth_below_p0_1_qty_eob
bar5s_depth_below_p1_2_qty_twa
bar5s_depth_below_p1_2_qty_eob
bar5s_depth_below_p2_3_qty_twa
bar5s_depth_below_p2_3_qty_eob
bar5s_depth_below_p3_5_qty_twa
bar5s_depth_below_p3_5_qty_eob
bar5s_depth_below_p5_10_qty_twa
bar5s_depth_below_p5_10_qty_eob

bar5s_depth_above_p0_1_qty_twa
bar5s_depth_above_p0_1_qty_eob
bar5s_depth_above_p1_2_qty_twa
bar5s_depth_above_p1_2_qty_eob
bar5s_depth_above_p2_3_qty_twa
bar5s_depth_above_p2_3_qty_eob
bar5s_depth_above_p3_5_qty_twa
bar5s_depth_above_p3_5_qty_eob
bar5s_depth_above_p5_10_qty_twa
bar5s_depth_above_p5_10_qty_eob

bar5s_depth_below_p0_1_frac_twa
bar5s_depth_below_p0_1_frac_eob
bar5s_depth_below_p1_2_frac_twa
bar5s_depth_below_p1_2_frac_eob
bar5s_depth_below_p2_3_frac_twa
bar5s_depth_below_p2_3_frac_eob
bar5s_depth_below_p3_5_frac_twa
bar5s_depth_below_p3_5_frac_eob
bar5s_depth_below_p5_10_frac_twa
bar5s_depth_below_p5_10_frac_eob

bar5s_depth_above_p0_1_frac_twa
bar5s_depth_above_p0_1_frac_eob
bar5s_depth_above_p1_2_frac_twa
bar5s_depth_above_p1_2_frac_eob
bar5s_depth_above_p2_3_frac_twa
bar5s_depth_above_p2_3_frac_eob
bar5s_depth_above_p3_5_frac_twa
bar5s_depth_above_p3_5_frac_eob
bar5s_depth_above_p5_10_frac_twa
bar5s_depth_above_p5_10_frac_eob
```

### 5.4 Ladder (4 features)

```
bar5s_ladder_ask_gap_max_pts_eob
bar5s_ladder_ask_gap_mean_pts_eob
bar5s_ladder_bid_gap_max_pts_eob
bar5s_ladder_bid_gap_mean_pts_eob
```

### 5.5 Shape (80 features)

```
bar5s_shape_bid_sz_l00_eob ... bar5s_shape_bid_sz_l09_eob  (10)
bar5s_shape_ask_sz_l00_eob ... bar5s_shape_ask_sz_l09_eob  (10)
bar5s_shape_bid_ct_l00_eob ... bar5s_shape_bid_ct_l09_eob  (10)
bar5s_shape_ask_ct_l00_eob ... bar5s_shape_ask_ct_l09_eob  (10)
bar5s_shape_bid_sz_frac_l00_eob ... bar5s_shape_bid_sz_frac_l09_eob  (10)
bar5s_shape_ask_sz_frac_l00_eob ... bar5s_shape_ask_sz_frac_l09_eob  (10)
bar5s_shape_bid_ct_frac_l00_eob ... bar5s_shape_bid_ct_frac_l09_eob  (10)
bar5s_shape_ask_ct_frac_l00_eob ... bar5s_shape_ask_ct_frac_l09_eob  (10)
```

### 5.6 Flow (90 features)

For each combination of:
- `<side>` ∈ `{bid, ask}` (2 values)
- `<band>` ∈ `{p0_1, p1_2, p2_3, p3_5, p5_10}` (5 values)

Volume features (3 per combination × 10 combinations = 30):
```
bar5s_flow_add_vol_<side>_<band>_sum
bar5s_flow_rem_vol_<side>_<band>_sum
bar5s_flow_net_vol_<side>_<band>_sum
```

Count features (3 per combination × 10 combinations = 30):
```
bar5s_flow_cnt_add_<side>_<band>_sum
bar5s_flow_cnt_cancel_<side>_<band>_sum
bar5s_flow_cnt_modify_<side>_<band>_sum
```

Normalized net (1 per combination × 10 combinations = 10):
```
bar5s_flow_net_volnorm_<side>_<band>_sum
```

**Explicit enumeration of flow features:**
```
bar5s_flow_add_vol_bid_p0_1_sum
bar5s_flow_add_vol_bid_p1_2_sum
bar5s_flow_add_vol_bid_p2_3_sum
bar5s_flow_add_vol_bid_p3_5_sum
bar5s_flow_add_vol_bid_p5_10_sum
bar5s_flow_add_vol_ask_p0_1_sum
bar5s_flow_add_vol_ask_p1_2_sum
bar5s_flow_add_vol_ask_p2_3_sum
bar5s_flow_add_vol_ask_p3_5_sum
bar5s_flow_add_vol_ask_p5_10_sum

bar5s_flow_rem_vol_bid_p0_1_sum
bar5s_flow_rem_vol_bid_p1_2_sum
bar5s_flow_rem_vol_bid_p2_3_sum
bar5s_flow_rem_vol_bid_p3_5_sum
bar5s_flow_rem_vol_bid_p5_10_sum
bar5s_flow_rem_vol_ask_p0_1_sum
bar5s_flow_rem_vol_ask_p1_2_sum
bar5s_flow_rem_vol_ask_p2_3_sum
bar5s_flow_rem_vol_ask_p3_5_sum
bar5s_flow_rem_vol_ask_p5_10_sum

bar5s_flow_net_vol_bid_p0_1_sum
bar5s_flow_net_vol_bid_p1_2_sum
bar5s_flow_net_vol_bid_p2_3_sum
bar5s_flow_net_vol_bid_p3_5_sum
bar5s_flow_net_vol_bid_p5_10_sum
bar5s_flow_net_vol_ask_p0_1_sum
bar5s_flow_net_vol_ask_p1_2_sum
bar5s_flow_net_vol_ask_p2_3_sum
bar5s_flow_net_vol_ask_p3_5_sum
bar5s_flow_net_vol_ask_p5_10_sum

bar5s_flow_cnt_add_bid_p0_1_sum
bar5s_flow_cnt_add_bid_p1_2_sum
bar5s_flow_cnt_add_bid_p2_3_sum
bar5s_flow_cnt_add_bid_p3_5_sum
bar5s_flow_cnt_add_bid_p5_10_sum
bar5s_flow_cnt_add_ask_p0_1_sum
bar5s_flow_cnt_add_ask_p1_2_sum
bar5s_flow_cnt_add_ask_p2_3_sum
bar5s_flow_cnt_add_ask_p3_5_sum
bar5s_flow_cnt_add_ask_p5_10_sum

bar5s_flow_cnt_cancel_bid_p0_1_sum
bar5s_flow_cnt_cancel_bid_p1_2_sum
bar5s_flow_cnt_cancel_bid_p2_3_sum
bar5s_flow_cnt_cancel_bid_p3_5_sum
bar5s_flow_cnt_cancel_bid_p5_10_sum
bar5s_flow_cnt_cancel_ask_p0_1_sum
bar5s_flow_cnt_cancel_ask_p1_2_sum
bar5s_flow_cnt_cancel_ask_p2_3_sum
bar5s_flow_cnt_cancel_ask_p3_5_sum
bar5s_flow_cnt_cancel_ask_p5_10_sum

bar5s_flow_cnt_modify_bid_p0_1_sum
bar5s_flow_cnt_modify_bid_p1_2_sum
bar5s_flow_cnt_modify_bid_p2_3_sum
bar5s_flow_cnt_modify_bid_p3_5_sum
bar5s_flow_cnt_modify_bid_p5_10_sum
bar5s_flow_cnt_modify_ask_p0_1_sum
bar5s_flow_cnt_modify_ask_p1_2_sum
bar5s_flow_cnt_modify_ask_p2_3_sum
bar5s_flow_cnt_modify_ask_p3_5_sum
bar5s_flow_cnt_modify_ask_p5_10_sum

bar5s_flow_net_volnorm_bid_p0_1_sum
bar5s_flow_net_volnorm_bid_p1_2_sum
bar5s_flow_net_volnorm_bid_p2_3_sum
bar5s_flow_net_volnorm_bid_p3_5_sum
bar5s_flow_net_volnorm_bid_p5_10_sum
bar5s_flow_net_volnorm_ask_p0_1_sum
bar5s_flow_net_volnorm_ask_p1_2_sum
bar5s_flow_net_volnorm_ask_p2_3_sum
bar5s_flow_net_volnorm_ask_p3_5_sum
bar5s_flow_net_volnorm_ask_p5_10_sum
```

### 5.7 Trade (5 features)

```
bar5s_trade_cnt_sum
bar5s_trade_vol_sum
bar5s_trade_aggbuy_vol_sum
bar5s_trade_aggsell_vol_sum
bar5s_trade_signed_vol_sum
```

### 5.8 Wall (8 features)

```
bar5s_wall_bid_maxz_eob
bar5s_wall_ask_maxz_eob
bar5s_wall_bid_maxz_levelidx_eob
bar5s_wall_ask_maxz_levelidx_eob
bar5s_wall_bid_nearest_strong_dist_pts_eob
bar5s_wall_ask_nearest_strong_dist_pts_eob
bar5s_wall_bid_nearest_strong_levelidx_eob
bar5s_wall_ask_nearest_strong_levelidx_eob
```

---

## Section 6 — Feature Count Summary

| Family | Count |
|--------|-------|
| Meta | 6 |
| State | 16 |
| Depth | 44 |
| Ladder | 4 |
| Shape | 80 |
| Flow | 70 |
| Trade | 5 |
| Wall | 8 |
| **Total** | **233** |

Plus 2 identifier columns:
- `bar_ts` (bar start timestamp, nanoseconds)
- `symbol` (instrument identifier)

---

## Section 7 — Validation Checks

The implementation should include these sanity checks:

### 7.1 Per-Event Checks

- `ts_event` is monotonically non-decreasing (after sorting)
- `action` is in valid range {0, 1, 2, 3, 4, 5}
- `side` is in {'A', 'B', 'a', 'b'}
- `size >= 0`
- `depth` is in range [0, 9]

### 7.2 Per-Bar Checks

- `meta_msg_cnt_sum >= 0`
- `meta_add_cnt_sum + meta_cancel_cnt_sum + meta_modify_cnt_sum + meta_trade_cnt_sum + meta_clear_cnt_sum <= meta_msg_cnt_sum`
- `-1 <= obi0 <= 1`
- `-1 <= obi10 <= 1`
- `-1 <= cdi_* <= 1`
- `spread_pts >= 0` (ask should be >= bid)
- `sum(below_*_frac) ≈ 1.0` (within floating point tolerance)
- `sum(above_*_frac) ≈ 1.0`
- `sum(bid_sz_frac_l00..l09) ≈ 1.0`
- `sum(ask_sz_frac_l00..l09) ≈ 1.0`

### 7.3 Data Quality Flags (Optional)

Consider adding these meta features to flag data issues:

- `bar5s_meta_empty_bid_book_flag` = 1 if BidDepth10 was 0 at any point
- `bar5s_meta_empty_ask_book_flag` = 1 if AskDepth10 was 0 at any point
- `bar5s_meta_crossed_book_flag` = 1 if bid_px_00 > ask_px_00 at any point

---

## Section 8 — Implementation Notes

### 8.1 Memory Efficiency

- Process data in streaming fashion (one event at a time)
- Only keep current pre-state and accumulator structures in memory
- Write completed bars to output file incrementally

### 8.2 Performance Considerations

- Pre-compute P_ref once per event, reuse for all band calculations
- Use vectorized operations for summing book arrays
- Price scanning for ΔQ calculation can use binary search if prices are sorted

### 8.3 Testing Strategy

1. **Unit test TWA logic:** Create synthetic events at known times, verify TWA equals expected weighted average
2. **Unit test flow logic:** Create Add/Cancel sequence at known prices, verify ΔQ computed correctly
3. **Unit test band assignment:** Create events at known distances from P_ref, verify band assignment
4. **Integration test:** Process one day of real data, verify:
   - No NaN in non-wall features (except where documented)
   - Feature ranges are sensible
   - Row count matches expected bar count for that session
5. **Regression test:** Store expected output for a fixed input sample, verify exact reproducibility

---

## Appendix A — Databento MBP-10 Schema Reference

For reference, the Databento MBP-10 schema includes these fields:

| Field | Type | Description |
|-------|------|-------------|
| ts_event | uint64 | Exchange timestamp (nanoseconds) |
| ts_recv | uint64 | Receive timestamp (nanoseconds) |
| rtype | uint8 | Record type |
| publisher_id | uint16 | Publisher ID |
| instrument_id | uint32 | Instrument ID |
| action | uint8 | Action: 0=Modify, 1=Clear, 2=Add, 3=Cancel, 4=Trade, 5=Fill |
| side | char | 'A'=Ask, 'B'=Bid |
| depth | uint8 | Level index (0-9) |
| price | int64 | Price in fixed-point (÷1e9 for float) |
| size | uint32 | Size at level |
| flags | uint8 | Flags bitfield |
| ts_in_delta | int32 | (internal use) |
| sequence | uint32 | Sequence number |
| bid_px_00..09 | int64[10] | Bid prices by level |
| ask_px_00..09 | int64[10] | Ask prices by level |
| bid_sz_00..09 | uint32[10] | Bid sizes by level |
| ask_sz_00..09 | uint32[10] | Ask sizes by level |
| bid_ct_00..09 | uint32[10] | Bid counts by level |
| ask_ct_00..09 | uint32[10] | Ask counts by level |

---

## Appendix B — Band Assignment Pseudocode

```
function assign_band(p_evt, P_ref, side):
    if side == 'B':  # Bid
        d = (P_ref - p_evt) / POINT
    else:  # Ask
        d = (p_evt - P_ref) / POINT
    
    if d <= 0:
        return 'p0_1'
    elif d <= 1:
        return 'p0_1'
    elif d <= 2:
        return 'p1_2'
    elif d <= 3:
        return 'p2_3'
    elif d <= 5:
        return 'p3_5'
    elif d <= 10:
        return 'p5_10'
    else:
        return None  # Exclude from banded features
```

---

## Appendix C — TWA Pseudocode

```
# At bar start
t_last = bar_start
sum_X = 0

# For each event in bar
for event in bar_events:
    t_e = event.ts_event
    dt = t_e - t_last
    
    X_pre = compute_X(pre_state)
    sum_X += X_pre * dt
    
    t_last = t_e
    pre_state = event.post_state

# At bar end
dt_end = bar_end - t_last
X_final = compute_X(pre_state)
sum_X += X_final * dt_end

X_twa = sum_X / BAR_DURATION_NS
```

---

## Appendix D — Flow ΔQ Pseudocode

```
function compute_delta_q(event, pre_state, post_state):
    if event.action not in {Add, Cancel, Modify}:
        return None
    
    p_evt = event.price
    side = event.side
    
    # Find Q_prev in pre-state
    Q_prev = 0
    if side == 'B':
        for i in range(10):
            if pre_state.bid_px[i] == p_evt:
                Q_prev = pre_state.bid_sz[i]
                break
    else:  # Ask
        for i in range(10):
            if pre_state.ask_px[i] == p_evt:
                Q_prev = pre_state.ask_sz[i]
                break
    
    # Find Q_new in post-state
    Q_new = 0
    if side == 'B':
        for i in range(10):
            if post_state.bid_px[i] == p_evt:
                Q_new = post_state.bid_sz[i]
                break
    else:  # Ask
        for i in range(10):
            if post_state.ask_px[i] == p_evt:
                Q_new = post_state.ask_sz[i]
                break
    
    delta_q = Q_new - Q_prev
    add_vol = max(delta_q, 0)
    rem_vol = max(-delta_q, 0)
    
    return add_vol, rem_vol
```

---

## Appendix E — Wall Z-Score Pseudocode

```
function compute_wall_features(book_side_sizes):
    # book_side_sizes = array of 10 sizes [sz_0, sz_1, ..., sz_9]
    
    # Log transform
    q = [ln(1 + sz) for sz in book_side_sizes]
    
    # Statistics
    mu = mean(q)
    sigma = std(q)  # population std
    
    # Z-scores
    Z = [(q_n - mu) / max(sigma, EPSILON) for q_n in q]
    
    # Max Z
    max_z = max(Z)
    max_z_idx = argmax(Z)
    
    # Nearest strong wall (Z >= threshold)
    nearest_strong_idx = -1
    for i in range(10):
        if Z[i] >= WALL_Z_THRESHOLD:
            nearest_strong_idx = i
            break
    
    return max_z, max_z_idx, nearest_strong_idx
```
