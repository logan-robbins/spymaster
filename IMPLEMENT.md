# IMPLEMENT.md — Radar Architecture (Single Product Plan)

## Objective (Trader-Facing)
Build one product that answers, continuously and in real time:

- **What is happening above and below price right now?**
- **Is there something stopping price from moving higher/lower?**
- **Are obstacles building or eroding (d1/d2/d3)?**
- **How is strike-based GEX changing as spot moves and as time passes?**

This is **not** a Bookmap clone. We will use **our physics** (vacuum / slope / localization / shock) computed from Databento **MBO** (market-by-order) as the canonical source.

---

## Facts / Constraints (Do Not Violate)
- **Canonical microstructure source is Databento MBO** (schema `mbo`, `rtype=160`) for both futures and options.
  - **Timestamps**: `ts_event` and `ts_recv` are **nanoseconds since UNIX epoch**.
  - **Price encoding**: `price` is `int64` fixed-point where **1 unit = 1e-9**. Our canonical conversion is `price = price_int * 1e-9`.
  - **ES tick constants (hard-coded for this product)**:
    - `TICK_SIZE = 0.25`
    - `TICK_INT = int(round(TICK_SIZE / 1e-9)) = 250_000_000`
  - **Actions**: `action ∈ {A,C,M,R,T,F,N}` (Add, Cancel, Modify, cleaR book, Trade, Fill, None).
  - **Sides**: `side ∈ {A,B,N}` (Ask, Bid, None).
- **Databento MBO snapshots must be handled correctly** (hard requirement):
  - Snapshot records are marked with `F_SNAPSHOT` and begin with an `action='R'` record (clear book), followed by `action='A'` records to rebuild state.
  - The order book is **not guaranteed valid** until the snapshot sequence reaches a record with `F_LAST`. (If the last snapshot record does not have `F_LAST`, book state is invalid until the next record with `F_LAST`.)
  - Snapshot `ts_recv` may be a synthetic snapshot-generation timestamp (flagged `F_BAD_TS_RECV`). **Do not order by `ts_recv`**; order by `ts_event, sequence`.
- **Option contract metadata is not inferred from strings** (hard requirement):
  - `underlying`, `right`, `expiration`, and `strike` must be sourced from Databento **Instrument Definitions** keyed by `instrument_id` (expiration is nanoseconds since epoch; strike is fixed-point 1e-9).
  - We may carry these fields in parquet for convenience, but they are treated as **derived reference data** from the definitions feed, not a parsing heuristic.
- **NBBO is derived from MBO** book reconstruction (best bid/ask). We do not require separate NBBO tables to build the product.
- **Trades are derived from MBO** (trade prints are action `T`). We do not require separate trades tables to build the product.
- **Open interest is not present in MBO.** For OI-based GEX we ingest Databento options statistics for the dates in question.
- **Online-safe**: every feature at time \(t\) uses only data \(\le t\). No lookahead.
- **Overlay shapes are explicit**:
  - Futures: **time × price** surfaces (absolute price levels).
  - Options: **time × strike** surfaces (absolute strikes; strikes are price-levels in ES points).

---

## Current Pipeline Reality (What Exists Today, and the Gap)

### Futures (`future_mbo`)
- Stage `SilverComputeMboLevelVacuum5s` in `backend/src/data_eng/stages/silver/future_mbo/compute_level_vacuum_5s.py` outputs `silver.future_mbo.mbo_level_vacuum_5s`.
- It anchors to premarket high `P_ref`, tracks per-order bucket membership across windows, and emits `approach_dir` from trade prints. This is correct for level-based retrieval, not for a spot-centered overlay.
- `gold.future_mbo.mbo_pressure_stream` nulls pressure when `approach_dir == approach_none`, which creates **blind spots** if you want a continuous overlay around spot.

**Gap**: We want “what’s above/below price now” continuously. That requires **spot-anchored** (not level-anchored) physics and a **surface** that can be drawn above/below the live price line.

### Options (`future_option_mbo`)
- Stage `SilverComputeGex5s` in `backend/src/data_eng/stages/silver/future_option_mbo/compute_gex_5s.py` outputs `silver.future_option_mbo.gex_5s`.
- It computes a **gamma-weighted depth proxy** using the options book and a depth-weighted `ref_price`. It does **not** incorporate open interest or a futures spot join.

**Gap**: For strike-level GEX that matches common trading usage, we need **OI + gamma** per strike/expiry. OI comes from the statistics feed; the rest can be derived from MBO.

---

## Execution Plan (Living, Pipeline-Aligned)
1) Define dataset entries and Avro contracts for `book_snapshot_1s`, `wall_surface_1s`, `vacuum_surface_1s`, `radar_vacuum_1s`, `physics_bands_1s`, `gex_surface_1s`, plus a **normalization calibration artifact** (`gold.hud.physics_norm_calibration`). Status: **DONE** (Phase 1 Contracts Created).
2) Add silver stages in `future_mbo` to build book snapshot + wall surface from a single book reconstruction pass, and to derive `vacuum_surface_1s`, `radar_vacuum_1s`, `physics_bands_1s`. Status: **DONE** (Snapshot & Wall Implemented & Verified).
3) Replace `silver.future_option_mbo.gex_5s` with `silver.future_option_mbo.gex_surface_1s` built from options MBO + OI stats + futures spot, with a **fixed signed convention** (defined below). Status: **CODE COMPLETE** (Pending Verification).
4) Add the **Serving Layer** (API + WebSocket) that streams a rolling 30-minute HUD window with bounded surfaces and fixed quantization. Status: todo.
5) Register stages in `pipeline.py` and ensure cross-product inputs are built before dependent stages. Status: **DONE** (Registered).
6) Run the verification checks in this document (including normalization + bounded texture limits). Status: todo.

---

## Product Output: Three Physics Surfaces + One Summary

### 1) Futures Wall Surface (time × price)
Shows “what is sitting there” (obstacles) and whether it is eroding.

### 2) Futures Vacuum Surface (time × price)
Shows “is liquidity pulling away or building” at each price level (our vacuum physics in a paintable form).

### 3) Options GEX Surface (time × strike)
Shows strike-level gamma exposure as a continuously evolving obstacle field.

### 4) Above/Below Summary (time series)
Compact signals for “up is easier than down right now” and vice versa, derived from the surfaces.

---

## Data Contracts Engineers Must Implement

### A0) Databento Instrument Definitions (required reference feed)
This is required to correctly resolve **option expiration + strike** (and, eventually, tick size / display format) without fragile string parsing.

**Dataset key**: `bronze.shared.instrument_definitions`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `bronze/source=databento/symbol={symbol}/table=instrument_definitions`
- contract: `src/data_eng/contracts/bronze/shared/instrument_definitions.avsc`

**Stage**:
- `BronzeIngestInstrumentDefinitions` in `backend/src/data_eng/stages/bronze/shared/ingest_instrument_definitions.py`

**Row grain**: one row per instrument definition message (time series).

**Required fields** (minimum):
- `ts_event` (ns since epoch; definition timestamp)
- `instrument_id` (uint32)
- `security_update_action` (A/M/D)
- `security_type` (must include option-on-future identifiers)
- `underlying_id` and `underlying`
- `expiration` (ns since epoch; last eligible trade time)
- `strike_price` (int64 fixed-point 1e-9)
- `raw_symbol` (for debugging / reconciliation)

**Hard rule (online-safe)**:
- Any feature computed for session date `dt` must use the instrument definition state as-of the session window start for `dt` (no lookahead).

### A) Futures Book Snapshot (derived from futures MBO)
**Dataset key**: `silver.future_mbo.book_snapshot_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_mbo/symbol={symbol}/table=book_snapshot_1s`
- contract: `src/data_eng/contracts/silver/future_mbo/book_snapshot_1s.avsc`

**Stage**:
- `SilverComputeBookSnapshot1s` in `backend/src/data_eng/stages/silver/future_mbo/compute_book_snapshot_1s.py`

**Purpose**: deterministic spot reference and basic integrity signals.

**Required fields** (per 1s window):
- `window_start_ts_ns`, `window_end_ts_ns`
- `best_bid_price_int`, `best_bid_qty`
- `best_ask_price_int`, `best_ask_qty`
- `mid_price` (double; derived)
- `mid_price_int` (int64; derived by rounding `mid_price` to the nearest tick, in `price_int` units)
- `last_trade_price_int` (from action `T`, the last trade at-or-before `window_end_ts_ns` for the instrument)
- `spot_ref_price_int` (int64; **the canonical spot reference for the HUD**):
  - `spot_ref_price_int = last_trade_price_int` (primary)
  - if no trade has occurred yet in-session, use `best_bid_price_int` when `book_valid=true`, else `0`
- `book_valid` (false after any `action='R'` until the book is known valid again; see snapshot rules above)

**Why**: Everything else needs a spot reference. We cannot depend on external NBBO/trades feeds because those are derivable from MBO.

---

### B) Futures Wall Surface (derived from futures MBO)
**Dataset key**: `silver.future_mbo.wall_surface_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_mbo/symbol={symbol}/table=wall_surface_1s`
- contract: `src/data_eng/contracts/silver/future_mbo/wall_surface_1s.avsc`

**Stage**:
- `SilverComputeWallSurface1s` in `backend/src/data_eng/stages/silver/future_mbo/compute_wall_surface_1s.py`

**Row grain**: one row per \((window_end_ts_ns, price_int, side)\)
- `side` ∈ {`B`, `A`}

**Hard bounds (to make a real-time HUD feasible)**:
- Define `HUD_MAX_TICKS = 600` (150 ES points).
- For each window, compute `spot_ref_price_int` (from `book_snapshot_1s` at the same `window_end_ts_ns`).
- Emit rows **only** for price levels in:
  - `price_int ∈ [spot_ref_price_int - HUD_MAX_TICKS*TICK_INT, spot_ref_price_int + HUD_MAX_TICKS*TICK_INT]`
- Also emit `rel_ticks = (price_int - spot_ref_price_int) / TICK_INT` as an **integer** in \([-HUD_MAX_TICKS, +HUD_MAX_TICKS]\).

**Required identity fields** (per row):
- `spot_ref_price_int` (int64; copied from `book_snapshot_1s`)
- `rel_ticks` (int32)

**Required measures** (per row):
- `depth_qty_start`
- `depth_qty_end`
- `add_qty` (adds within window at this price/side)
- `pull_qty_total` (cancels/mods reducing size within window at this price/side)
- `depth_qty_rest` (subset of depth_qty_end where order age at current price ≥ REST_NS)
- `pull_qty_rest` (subset where order age at its current price ≥ REST_NS at time of pull)
- `fill_qty` (reductions attributable to fills within window at this price/side)
- `d1_depth_qty`, `d2_depth_qty`, `d3_depth_qty` (computed over time for the same \((price_int, side)\))
- `window_valid`

**Why**: This is the minimal “heatmap substrate” that can be overlaid on price and supports erosion directly.

---

### C) Futures Vacuum Surface (our physics, paintable)
**Dataset key**: `silver.future_mbo.vacuum_surface_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_mbo/symbol={symbol}/table=vacuum_surface_1s`
- contract: `src/data_eng/contracts/silver/future_mbo/vacuum_surface_1s.avsc`

**Stage**:
- `SilverComputeVacuumSurface1s` in `backend/src/data_eng/stages/silver/future_mbo/compute_vacuum_surface_1s.py`

**Row grain**: one row per \((window_end_ts_ns, price_int, side)\).

**Hard bounds**: identical to `wall_surface_1s` (same `HUD_MAX_TICKS`, same `spot_ref_price_int`, same `rel_ticks`).

**Required identity fields** (per row):
- `spot_ref_price_int` (int64)
- `rel_ticks` (int32)

**Required measures** (derived from wall_surface_1s):
- `pull_intensity_rest = pull_qty_rest / (depth_qty_start + ε)`
- `pull_add_log = log((pull_qty_rest + ε) / (add_qty + ε))`
- `d1_pull_add_log`, `d2_pull_add_log`, `d3_pull_add_log`
- `wall_strength_log = log(depth_qty_rest + 1)` (visual stability using **resting** depth; exact transform is documented and consistent)
- `wall_erosion = -min(d1_depth_qty, 0)` (non-negative “erosion magnitude”)
- `vacuum_score` (0..1; **hard-specified normalization below**; continuous — **never gated to null by approach_dir**)

**Why**: This is our “vacuum / obstacle” language in a form that can be drawn as a heatmap above/below price.

---

### C.1) Vacuum score normalization (hard spec; cross-day stable)
We define a **fixed 0..1 mapping** so the HUD color scale is stable across dates.

**Inputs** (per row):
- \(x_1 = pull\_add\_log\)
- \(x_2 = \log(1 + pull\_intensity\_rest)\)
- \(x_3 = \log(1 + wall\_erosion / (depth\_qty\_start + \varepsilon))\)
- \(x_4 = d2\_pull\_add\_log\)  (shock = acceleration of pulls vs adds)

**Normalization calibration (required artifact)**:
- Maintain `gold.hud.physics_norm_calibration` per symbol with robust bounds for each \(x_k\):
  - \(lo_k = Q_{05}(x_k)\), \(hi_k = Q_{95}(x_k)\) computed over the **first 3 trading hours** across the **most recent 20 sessions**.
- This calibration is refreshed **daily** and is the **single source of truth** for HUD color mapping.

**Per-row normalized components**:
- \(n_k = clamp((x_k - lo_k) / (hi_k - lo_k), 0, 1)\)

**Final vacuum score**:
- `vacuum_score = (n1 + n2 + n3 + n4) / 4`

This makes `vacuum_score` comparable across days and stable for WebGL color mapping.

---

### D) Futures Spot-Anchored Vacuum Features (reuse our existing feature family, but continuous)
**Dataset key**: `silver.future_mbo.radar_vacuum_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_mbo/symbol={symbol}/table=radar_vacuum_1s`
- contract: `src/data_eng/contracts/silver/future_mbo/radar_vacuum_1s.avsc`

**Stage**:
- `SilverComputeRadarVacuum1s` in `backend/src/data_eng/stages/silver/future_mbo/compute_radar_vacuum_1s.py`

**Purpose**: keep the existing vacuum vocabulary (COM displacement, slopes, pull shares, repricing shares, etc.) but compute it continuously around spot (not around PM high).

**Required fields**:
- `window_start_ts_ns`, `window_end_ts_ns`
- `spot_ref_price` (double) and `spot_ref_price_int`
- `approach_dir` (derived from MBO trade prints / trend; must not be used to null the output)
- The full feature set currently produced by `mbo_level_vacuum_5s`, with one change:
  - **Replace** `P_ref`/`P_REF_INT` with `spot_ref_price`/`spot_ref_price_int`.

**Critical rule (do not implement incorrectly)**:
- Do **not** persist per-order “bucket” labels across windows in a moving spot reference frame.
- Instead, for each 1s window:
  - pick a deterministic `spot_ref_price_int` for that window (from book_snapshot_1s),
  - compute start/end snapshots and within-window accumulators relative to that same spot_ref.
This preserves feature meaning and prevents “reference-motion artifacts.”

---

### E) Options OI-Based GEX Surface (derived from options MBO + OI statistics)
**Dataset key**: `silver.future_option_mbo.gex_surface_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_option_mbo/symbol={symbol}/table=gex_surface_1s`
- contract: `src/data_eng/contracts/silver/future_option_mbo/gex_surface_1s.avsc`

**Stage**:
- `SilverComputeGexSurface1s` in `backend/src/data_eng/stages/silver/future_option_mbo/compute_gex_surface_1s.py`

**Row grain**: one row per \((window_end_ts_ns, underlying, strike_price_int)\).

**Hard bounds (to make a real-time HUD feasible)**:
- Define `GEX_STRIKE_STEP_POINTS = 5` (ES option strikes are on a 5-point grid for the contracts we visualize).
- Define `GEX_MAX_STRIKE_OFFSETS = 30` (±150 points around spot; aligns to the futures HUD price axis).
- Define `GEX_MAX_DTE_DAYS = 45` (only include expirations with \(0 < DTE \le 45\) days).
- For each window:
  - compute `underlying_spot_ref` (from `book_snapshot_1s`)
  - define `strike_ref_points = round(underlying_spot_ref / 5) * 5`
  - emit strikes `strike_points = strike_ref_points + i*GEX_STRIKE_STEP_POINTS` for `i ∈ [-GEX_MAX_STRIKE_OFFSETS, +GEX_MAX_STRIKE_OFFSETS]`
  - convert to fixed-point `strike_price_int = int(round(strike_points / 1e-9))`

**Required inputs**:
- Underlying spot per window (from `silver.future_mbo.book_snapshot_1s`).
- Option mid premium per window (derived by reconstructing option best bid/ask from **options MBO**).
- Open interest from `silver.future_option.statistics_clean` (statistics feed joined by option identity and time).
- **Instrument Definitions** (hard requirement): use `bronze.shared.instrument_definitions` keyed by `instrument_id` to obtain:
  - `expiration` (ns since epoch; last eligible trade time)
  - `strike_price` (fixed-point 1e-9)
  - `right` (C/P)
  - `underlying` symbol

**Required fields**:
- `underlying_spot_ref` (double)
- `strike_price_int` (int64; fixed-point 1e-9)
- `strike_points` (double)
- `gex_call_abs` (double; magnitude)
- `gex_put_abs` (double; magnitude)
- `gex_abs` (double; barrier magnitude)
- `gex` (double; call-minus-put signed field)
- `gex_imbalance_ratio = gex / (gex_abs + ε)` (double; \([-1, +1]\); used for HUD color)
- `d1_gex_abs`, `d2_gex_abs`, `d3_gex_abs` (double)
- `d1_gex`, `d2_gex`, `d3_gex` (double)
- `d1_gex_imbalance_ratio`, `d2_gex_imbalance_ratio`, `d3_gex_imbalance_ratio` (double)

**Signed convention (hard spec; no ambiguity)**:
- We do **not** label `gex` as “dealer-signed gamma.” We define the sign to match the trader-facing “call/long vs put/short” narrative:
  - Compute `gex_call_abs` and `gex_put_abs` as **positive magnitudes** per strike (summing across expirations within `GEX_MAX_DTE_DAYS`).
  - Define the signed field as **call-minus-put**:
    - `gex = gex_call_abs - gex_put_abs`
- Additionally, the surface must expose a **barrier magnitude** field used for obstacle visualization:
  - `gex_abs = gex_call_abs + gex_put_abs`

**GEX unit (hard spec)**:
- We output “**gex_per_1pt**”:
  - `gex_call_abs = Σ_exp ( gamma(exp, strike) * open_interest(exp, strike, C) * FUTURES_MULTIPLIER )`
  - `gex_put_abs  = Σ_exp ( gamma(exp, strike) * open_interest(exp, strike, P) * FUTURES_MULTIPLIER )`
  - where `FUTURES_MULTIPLIER = 50` for ES.
This unit is stable, interpretable at the level granularity we care about (ES points), and directly supports a strike heatmap.

---

### F) Above/Below Summary Bands (derived from futures surfaces)
**Dataset key**: `silver.future_mbo.physics_bands_1s`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `silver/product_type=future_mbo/symbol={symbol}/table=physics_bands_1s`
- contract: `src/data_eng/contracts/silver/future_mbo/physics_bands_1s.avsc`

**Stage**:
- `SilverComputePhysicsBands1s` in `backend/src/data_eng/stages/silver/future_mbo/compute_physics_bands_1s.py`

**One row per window**, including:
- `mid_price`
- Tick bands (using the same constants as our current vacuum logic):
  - AT: 0–2 ticks
  - NEAR: 3–5 ticks
  - MID: 6–14 ticks
  - FAR: 15–20 ticks
- Aggregates for each band above and below spot (derived from wall_surface + vacuum_surface):
  - wall_strength / wall_erosion / vacuum_score components
- A small number of composite summary scores:
  - `above_score`, `below_score`, `vacuum_total_score`

**Why**: fast interpretability (status panel + alerts) while the surfaces provide full detail.

---

### F.1) Above/below composite scores (hard spec; stable semantics)
We define “above” as asks in positive `rel_ticks` and “below” as bids in negative `rel_ticks`. Scores are **0..1** and are designed to answer:

- `above_score`: “How easy is it for price to move upward right now (based on obstacles above)?”
- `below_score`: “How easy is it for price to move downward right now (based on obstacles below)?”

**Band selection for the live ‘tug’**:
- Use only the **AT + NEAR** bands (0–5 ticks) for the immediate tug.
- MID + FAR are still persisted and used by “headlights” later, but do not dominate the immediate tug.

**Per-band components (each already 0..1 due to `gold.hud.physics_norm_calibration`)**:
- `wall_strength_norm`: normalized `wall_strength_log` aggregated across the band
- `wall_erosion_norm`: normalized `log(1 + wall_erosion/(depth_qty_start+ε))` aggregated across the band
- `vacuum_norm`: mean `vacuum_score` across the band

**Band-level ease score**:
- `ease = 0.50 * vacuum_norm + 0.35 * wall_erosion_norm + 0.15 * (1 - wall_strength_norm)`

**Final window scores**:
- `above_score = 0.60 * ease_AT_above + 0.40 * ease_NEAR_above`
- `below_score = 0.60 * ease_AT_below + 0.40 * ease_NEAR_below`
- `vacuum_total_score = (above_score + below_score) / 2`

---

### G) HUD Normalization Calibration (required; shared)
**Dataset key**: `gold.hud.physics_norm_calibration`

**Dataset entry** (datasets.yaml; parquet; partition_keys: symbol, dt):
- path: `gold/hud/symbol={symbol}/table=physics_norm_calibration`
- contract: `src/data_eng/contracts/gold/hud/physics_norm_calibration.avsc`

**Stage**:
- `GoldBuildHudPhysicsNormCalibration` in `backend/src/data_eng/stages/gold/hud/build_physics_norm_calibration.py`

**Row grain**: one row per `metric_name` (per symbol), written once per session date.

**Required fields**:
- `metric_name` (string)
- `q05` (double)
- `q95` (double)
- `lookback_sessions` (int; always 20)
- `session_window` (string; always “first_3h”)
- `asof_dt` (string; the dt when the calibration was computed)

**Required metrics**:
- Futures surfaces:
  - `wall_strength_log`
  - `pull_add_log`
  - `log1p_pull_intensity_rest`
  - `log1p_erosion_norm`
  - `d2_pull_add_log`
- Options surface:
  - `gex_abs`

**Hard rule**:
- `vacuum_surface_1s`, `physics_bands_1s`, and HUD serving must **fail fast** if `gold.hud.physics_norm_calibration` is missing for the symbol.

---

## Pipeline Alignment (Stage Map and Dependencies)

### Futures (`future_mbo`) silver stages
- Build `book_snapshot_1s` and `wall_surface_1s` from a single MBO pass. Use one stage that writes both outputs to keep book state identical and to avoid drift between datasets.
- `SilverComputeVacuumSurface1s` uses `silver.future_mbo.wall_surface_1s` **and** `gold.hud.physics_norm_calibration` (for the hard 0..1 normalization) and emits `silver.future_mbo.vacuum_surface_1s`.
- `SilverComputeRadarVacuum1s` uses `bronze.future_mbo.mbo` plus `silver.future_mbo.book_snapshot_1s` so each window has a deterministic spot reference.
- `SilverComputePhysicsBands1s` uses `silver.future_mbo.book_snapshot_1s`, `silver.future_mbo.wall_surface_1s`, `silver.future_mbo.vacuum_surface_1s`, and `gold.hud.physics_norm_calibration`.

### Options (`future_option_mbo`) silver stages
- `SilverComputeGexSurface1s` uses `bronze.future_option_mbo.mbo`, `bronze.shared.instrument_definitions`, `silver.future_option.statistics_clean`, and `silver.future_mbo.book_snapshot_1s`. Output is `silver.future_option_mbo.gex_surface_1s`.
- `GoldBuildGexEnrichedTriggerVectors` must read `gex_surface_1s` to keep gold features consistent with the new OI-based surface.

### Pipeline registration
- In `pipeline.py`, list the new silver stages for `future_mbo` in dependency order.
- In `pipeline.py`, replace `SilverComputeGex5s` with `SilverComputeGexSurface1s` for `future_option_mbo`.

---

## Implementation Details (What To Do, and Why)

### 0) Stage and dataset pattern (match current pipeline)
- Follow the StageIO pattern in `compute_level_vacuum_5s.py` and `compute_gex_5s.py`: idempotency via `_SUCCESS`, contract enforcement, and manifest lineage via `read_manifest_hash`.
- New datasets are parquet partitions keyed by `symbol, dt` with Avro contracts under `backend/src/data_eng/contracts/silver/{product_type}/`.
- Stage naming should follow the `silver_compute_*_1s` convention so the CLI runner and logs stay consistent.

**Why**: keeps the new work consistent with the current pipeline and the existing runner expectations.

### 1) Futures MBO → Book Reconstruction (single source of truth)
Maintain:
- active orders: `order_id → {side, price_int, qty, ts_enter_price}`  
  - `ts_enter_price` updates when the order changes price (or is newly added)
- aggregated depth: `depth[side][price_int] = total_qty`

Process MBO actions:
- `A`: add order, increase depth
- `M`: update order (price and/or qty), move depth accordingly
- `C`: cancel order, decrease depth
- `F`: fill reduces qty (and depth); delete order if qty→0
- `T`: trade print (derive last trade price series)
- `R`: clear book (reset).
  - If `flags` indicate `F_SNAPSHOT`, this `R` is the start of a Databento snapshot stream. Keep ingesting snapshot `A` records until the snapshot completes at a record with `F_LAST`.
  - `book_valid=false` from the snapshot start until `F_LAST` is observed.
  - Any 1s window that overlaps an incomplete snapshot is `window_valid=false`.
- `N`: no-op record; ignore for state.

**Why**: this produces NBBO, mid, and the heatmap substrate directly from MBO (no external tables).

### 2) Windowing + Snapshot Discipline
- Keep 1-second windows consistent with the rest of the pipeline unless explicitly changed.
- For each window:
  - snapshot depth at window start and end
  - accumulate add/pull/fill flow by price/side within the window
  - emit `book_snapshot_1s` and `wall_surface_1s`

**Why**: it makes d1/d2 meaningful and prevents reference-frame artifacts.

### 3) “Resting pull” definition (must be stable)
- Use a single REST threshold (REST_NS) and define “resting” relative to **time at the current price**, tracked by `ts_enter_price`.

**Why**: it matches the idea “resting liquidity vs fleeting noise,” and does not break when spot moves.

### 4) Futures Vacuum Physics (continuous, not gated)
- Derive vacuum primitives per price/side from wall_surface.
- Derive band-level summaries above/below spot_ref.
- Compute spot-anchored versions of our existing vacuum features (COM displacement, slopes, pull shares, shock, etc.) per window.
- Do **not** null output based on an “approach” classification; approach is ancillary metadata.

**Why**: traders want continuous visibility, not only when price is near a pre-defined level.

### 5) Options OI-Based GEX (derived from MBO + statistics)
- Derive underlying spot from futures MBO.
- Derive option mid premiums from options MBO (best bid/ask) and resolve `expiration` + `strike_price` from Instrument Definitions keyed by `instrument_id`.
- Join open interest from statistics (`silver.future_option.statistics_clean`) using the same option identity (underlying/right/strike/exp).
- Compute IV + gamma per option contract (Black-76), then aggregate to the bounded strike grid:
  - sum `gex_call_abs` and `gex_put_abs` across expirations with \(0 < DTE \le GEX_MAX_DTE_DAYS\)
  - emit `gex_abs`, `gex`, and `gex_imbalance_ratio` per strike, plus d1/d2/d3 for the same strike over time.

**Why**: strike obstacles move continuously as spot moves; this surface directly overlays above/below price.

---

## What To Remove From Prior Plans (Incorrect)
- Do not implement “continuous vacuum” by simply setting `p_ref_int = current_mid_price` while keeping per-order bucket state. That creates incorrect “bucket-enter time” semantics and reference-motion artifacts.
- Do not treat the current `future_option_mbo.gex_5s` output as OI-based GEX; it is a proxy and must be replaced by `gex_surface_1s`.
- Do not use `gold.future_mbo.mbo_pressure_stream` as an overlay input; it nulls pressure outside `approach_dir` and hides continuous structure.

---

---
## Frontend Visualization Spec (WebGL / "The HUD")

### 1) Core Engine & Coordinate System
*   **Technology**: WebGL required (Pixi.js, Regl, or Three.js) for custom shaders.
*   **Viewport**:
    *   **X-Axis (Time)**: Rolling window. Visible: $T_{-30m}$ to $T_{now}$ (History) + $T_{now}$ to $T_{+5m}$ (Future Void).
    *   **Y-Axis (Price)**: Auto-centering on Spot, but lockable.
    *   **Z-Order**: Background (Vacuum) -> Midground (Walls) -> Foreground (Price/Tug) -> Overlay (Projections).

### 2) The Physics Layers ("X-Ray Vision")
We render the **Wall Surface** and **Vacuum Surface** as a composite texture.
*   **The Wall (Obstacles)**:
    *   **Core (Resting Depth)**: Rendered as **Solid/Saturated** blocks. Represents "Real" liquidity ($age > 500ms$).
    *   **Fog (Fleeting Depth)**: Rendered as **Diffuse/Glowing** clouds around the core. Represents "Spoof/Noise" ($age < 500ms$).
    *   *Visual Insight*: A "Brick" stops price. A "Cloud" lets price pass.
*   **Erosion effects (The "Active" Physics)**:
    *   **Vacuum (Pulls)**: Use a **Dissolve Shader**. The Wall "evaporates" into particles drifting away from price.
    *   **Smash (Fills)**: Use **Impact Flashes**. Bright white/yellow bursts on the wall edge upon trade impact.

### 3) The "Tug" (Spot & Net Pressure)
Visualizing the immediate forces acting on the price.
*   **Spot Indicator**: The current price is a bright, pulsing line or orb.
*   **Physics Vectors ("The Tug")**:
    *   An animated circular "Force Field" around the Spot.
    *   **distortion**: If `vacuum_score_above > vacuum_score_below`, the field stretches **upward** (egg shape), visually "pulling" the price up.
    *   **color**: Shifts Red (Resistance dominating) vs Green (Vacuum dominating).
    *   *User Feel*: "I can feel the market wanting to go up."

### 4) Future Projection ("The Headlights")
Rendering the model's forward-looking outputs into the $T > 0$ void.
*   **Ghost Candles**: Semi-transparent candles projecting the *most likely* path.
*   **Probability Fan**: A cone gradient indicating the confidence interval (P10 to P90).
*   **Dynamic Updates**: As the physics change (walls dissolve), the "Headlights" swing instantly to the new likely path.

---

## Serving & Streaming Spec (Hard Requirement)
The HUD must be able to render a rolling 30-minute window at interactive FPS with deterministic, bounded data volume.

### 0) API surface (hard spec)
- **Bootstrap (HTTP)**: `GET /v1/hud/bootstrap?symbol={symbol}&dt={dt}&end_ts_ns={now}`
  - returns Arrow IPC (single batch) containing the last 30 minutes of:
    - `book_snapshot_1s`
    - pre-quantized HUD texture columns (see below)
- **Live stream (WebSocket)**: `WS /v1/hud/stream?symbol={symbol}`
  - streams one Arrow IPC record batch every 1 second (one new column)

### 1) Transport (2025 best practice)
- **Binary columnar payloads**: Apache Arrow IPC is the canonical transport for:
  - bootstrapping a 30-minute window
  - streaming incremental 1-second updates
- Payloads are compressed (zstd) and decoded into TypedArrays on the client for direct WebGL texture upload.

### 2) Query engine (lake → HUD)
- The server queries parquet directly from the lake using DuckDB with predicate pushdown on:
  - `symbol`, `dt`, and `window_end_ts_ns` range
- The server emits:
  - `book_snapshot_1s` for the same time range
  - `wall_surface_1s` + `vacuum_surface_1s` converted into dense, bounded textures in `rel_ticks` space
  - `gex_surface_1s` mapped into the **same 1201-height price-axis texture** (strikes land on integer tick rows; values are 0 elsewhere)

### 3) HUD tile shape (fixed; guarantees WebGL feasibility)
- **Time axis**:
  - history: last 30 minutes = 1800 columns at 1s cadence
  - future void: 5 minutes = 300 columns (rendered empty client-side)
- **Price axis**:
  - `HUD_MAX_TICKS = 600` ⇒ height = `2*HUD_MAX_TICKS + 1 = 1201`
- This fits within common `MAX_TEXTURE_SIZE` constraints and avoids unbounded growth.

### 4) Streaming cadence
- The server pushes **one new column every 1 second** (the new window).
- Each update includes:
  - `window_end_ts_ns`
  - `spot_ref_price_int`
  - `wall_core_u8[1201]` (resting depth intensity)
  - `wall_fog_u8[1201]` (fleeting depth intensity)
  - `vacuum_u8[1201]` (vacuum_score intensity)
  - `fills_u8[1201]` (fill impact intensity)
  - `gex_abs_u8[1201]` (gamma obstacle magnitude at strikes)
  - `gex_imbalance_u8[1201]` (call-minus-put imbalance ratio mapped to 0..255)

### 5) Caching
- The server maintains an in-memory ring buffer for the last 30 minutes per symbol to ensure:
  - instant reconnects
  - deterministic redraws
  - no repeated parquet scans for the most recent window

---

## Verification & Testing (Must-Pass)

### Futures book integrity
- Depth is never negative.
- `best_bid_price_int < best_ask_price_int` whenever `book_valid=true`.
- `R` resets do not leak stale orders into subsequent windows (window_valid flag enforced).

### Futures overlay sanity
- A large stable wall produces stable `wall_strength_log` at that price and low erosion.
- A rapidly pulled wall produces negative `d1_depth_qty` and elevated `pull_intensity_rest`.
- Above/below band scores shift smoothly as price moves through levels (no artificial discontinuities).
- HUD texture bounds are enforced: every emitted row satisfies `abs(rel_ticks) <= HUD_MAX_TICKS`.
- `vacuum_score` and `above_score/below_score` are stable across days due to the fixed `gold.hud.physics_norm_calibration` mapping.

### Options GEX sanity
- Gamma is highest near-the-money; as spot moves away, gamma decreases (holding OI constant).
- If OI updates for a contract, GEX changes proportionally and is reflected in d1/d2.
