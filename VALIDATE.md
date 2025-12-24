# VALIDATE - Data Pipeline and Live Stream Checks

This document defines what the pipeline ingests, what it produces, and how to validate correctness (statistical and mathematical) for both historical batch runs and real-time viewing.

Authoritative references:
- `backend/features.json` (schema + output path + baseline stats)
- `PHASE_3.md` (architecture and invariants)
- `FRONTEND.md` (frontend payload contract)

---

## 1) Historical (Batch) Pipeline: Ingest -> Produce

### Inputs (required)
- ES futures trades + MBP-10 depth from Databento DBN files (via `DBNIngestor`).
- SPY option trades from Bronze tier (Polygon flatfiles via `BronzeReader`).
- Config constants from `src.common.config.CONFIG` (touch band, monitor band, thresholds).

### Process (authoritative batch pipeline)
`src.pipeline.vectorized_pipeline` runs the Phase 3 vectorized pipeline:
1. Build 1-minute OHLCV from ES trades and convert ES -> SPY (ES price / 10).
2. Generate LevelUniverse and detect touches by OHLCV high/low crossing within `CONFIG.TOUCH_BAND`.
3. Compute target-relative features (barrier, tape, fuel, confluence, mean-reversion, approach context).
4. Label outcomes anchored to confirmation time `t1 = t0 + 60s` and compute time-to-threshold for $1/$2.
5. Filter to monitor band (`distance <= CONFIG.MONITOR_BAND`, default $0.25).

### Output (final format expected)
Single Parquet file:
- Path: `data/lake/gold/research/signals_vectorized.parquet` (from `backend/features.json`)
- Schema: must match `backend/features.json` exactly

Minimum column groups required (full list in `backend/features.json`):
- Identity: `event_id`, `ts_ns`, `confirm_ts_ns`, `date`, `symbol`
- Level: `spot`, `level_price`, `level_kind`, `level_kind_name`, `direction`, `direction_sign`, `distance`, `distance_signed`, `atr`, `distance_atr`, `distance_pct`
- Context/Confluence/Mean Reversion: `is_first_15m`, `dist_to_pm_*`, `sma_*`, `confluence_*`
- Physics: `barrier_*`, `tape_*`, `gamma_exposure`, `fuel_effect`, `gamma_bucket`
- Dealer velocity + pressure: `gamma_flow_*`, `dealer_pressure*`, `*_pressure`
- Approach context: `approach_*`, `prior_touches`, `attempt_index`, `attempt_cluster_id`, `*_trend`, `bars_since_open`
- Outcome: `outcome`, `anchor_spot`, `future_price_5min`, `excursion_max`, `excursion_min`, `strength_signed`, `strength_abs`, `tradeable_1`, `tradeable_2`, `time_to_threshold_1`, `time_to_threshold_2`

---

## 2) Real-Time View: Ingest -> Stream

### Live inputs (required for full physics)
- ES futures trades + MBP-10 on NATS:
  - `market.futures.trades`
  - `market.futures.mbp10`
- SPY options trades on NATS:
  - `market.options.trades`

### Replay inputs (for historical replays)
`src.ingestor.replay_publisher` can publish:
- ES futures trades + MBP-10 from DBN files
- Optional SPY options trades from Bronze (`REPLAY_INCLUDE_OPTIONS=true`)

### Process (services-only)
1. **Ingestor** publishes normalized events to NATS.
2. **Core service** subscribes to `market.futures.*` + `market.options.trades`:
   - Updates MarketState
   - Computes physics signals at `CONFIG.SNAP_INTERVAL_MS`
   - Publishes `levels.signals`
   - Publishes `market.flow` (per-strike option aggregates for UI)
3. **Gateway** subscribes and emits `/ws/stream` payload with normalized level fields and optional viewport scores.

### Output (final formats expected)

Core -> NATS `levels.signals` (nested physics payload):
```
{
  "ts": <unix_ms>,
  "spy": {"spot": <float>, "bid": <float>, "ask": <float>},
  "levels": [
    {
      "id": "...",
      "price": <float>,
      "kind": "OR_LOW",
      "direction": "SUPPORT|RESISTANCE",
      "distance": <float>,
      "barrier": {...},
      "tape": {...},
      "fuel": {...},
      "runway": {...},
      "signal": "BREAK|REJECT|CONTESTED|NEUTRAL",
      "confidence": "HIGH|MEDIUM|LOW",
      "note": "..."
    }
  ],
  "viewport": {"ts": <unix_ms>, "targets": [...]}  # only if enabled
}
```

Gateway -> `/ws/stream` (frontend contract):
```
{
  "flow": { "<ticker>": { "cumulative_volume": ..., "net_gamma_flow": ..., ... } },
  "levels": {
    "ts": <unix_ms>,
    "spy": { "spot": <float>, "bid": <float>, "ask": <float> },
    "levels": [
      {
        "id": "...",
        "level_price": <float>,
        "level_kind_name": "OR_LOW",
        "direction": "UP|DOWN",
        "distance": <float>,
        "barrier_state": "...",
        "barrier_delta_liq": <float>,
        "barrier_replenishment_ratio": <float>,
        "wall_ratio": <float>,
        "tape_imbalance": <float>,
        "tape_velocity": <float>,
        "tape_buy_vol": <int>,
        "tape_sell_vol": <int>,
        "sweep_detected": <bool>,
        "gamma_exposure": <float>,
        "fuel_effect": "AMPLIFY|DAMPEN|NEUTRAL",
        "approach_velocity": <float>,
        "approach_bars": <int>,
        "approach_distance": <float>,
        "prior_touches": <int>,
        "bars_since_open": <int>,
        "break_score_raw": <float>,
        "break_score_smooth": <float>,
        "signal": "BREAK|BOUNCE|CHOP",
        "confidence": "HIGH|MEDIUM|LOW"
      }
    ]
  },
  "viewport": {
    "ts": <unix_ms>,
    "targets": [
      {
        "level_id": "...",
        "level_kind_name": "...",
        "level_price": <float>,
        "direction": "UP|DOWN",
        "distance": <float>,
        "distance_signed": <float>,
        "p_tradeable_2": <float>,
        "p_break": <float>,
        "p_bounce": <float>,
        "strength_signed": <float>,
        "strength_abs": <float>,
        "time_to_threshold": {"t1": {...}, "t2": {...}},
        "utility_score": <float>,
        "viewport_state": "...",
        "stage": "stage_a|stage_b",
        "pinned": <bool>,
        "relevance": <float>
      }
    ]
  }
}
```

---

## 3) Statistical Sanity Checks (Batch Dataset)

Use `backend/features.json` statistics as the baseline. Expect these to be in the same order of magnitude; deviations indicate missing data or math bugs.

Key expectations:
- **Outcome distribution**: BREAK/BOUNCE should dominate; CHOP and UNDEFINED should be small.
- **Barrier state**: NEUTRAL is majority; VACUUM/WALL/ABSORPTION/CONSUMED are present but sparse.
- **Fuel effect**: DAMPEN typically dominates; AMPLIFY is smaller but non-zero.
- **Sparse features**: `wall_ratio` and `barrier_delta_liq` should be mostly zero with a small non-zero tail.
- **Non-zero coverage**:
  - `tape_velocity` non-zero for almost all rows
  - `gamma_exposure` non-zero for almost all rows
  - `distance` always within `[0, 0.25]` by monitor-band filter

If any of the above collapse to all-zeros or a single class, the upstream data source is missing or the conversion window is wrong.

---

## 4) Mathematical Invariants (Batch Dataset)

These must hold to rule out math/labeling errors:

**Time and anchoring**
- `confirm_ts_ns == ts_ns + 60s` (60 * 1e9) for labeled rows.
- `anchor_spot` equals the OHLCV close at the bar used for `ts_ns`.
- `time_to_threshold_1/2` are either NaN or in `[0, 300]` seconds.
- `tradeable_1 == 1` iff `time_to_threshold_1` is finite; same for `tradeable_2`.

**Level geometry**
- `distance == abs(level_price - spot)` (within float tolerance).
- `distance_signed == level_price - spot`.
- `direction == "UP"` implies `distance_signed >= 0`; `direction == "DOWN"` implies `distance_signed <= 0`.
- `direction_sign == 1` for UP and `-1` for DOWN.

**Normalization**
- `distance_atr == distance / (atr + epsilon)` and `distance_pct == distance / spot`.
- Signed versions use `distance_signed` in the numerator.

**Outcome consistency**
Let threshold be `$2.00` and direction be the approach direction.
- `excursion_max` is the max move in the break direction from `anchor_spot`.
- `excursion_min` is the max move in the opposite direction from `anchor_spot`.
- `strength_signed == excursion_max - excursion_min`.
- `strength_abs == max(excursion_max, excursion_min)`.
- Outcome logic (from `vectorized_pipeline.py`):
  - `BREAK`: `excursion_max >= 2.0` and `excursion_max > excursion_min`
  - `BOUNCE`: `excursion_min >= 2.0`
  - `CHOP`: neither condition met
  - `UNDEFINED`: forward window missing

**Attempt clustering**
- `attempt_index` starts at 1 and increments per cluster.
- When `attempt_index == 1`, trend features (`*_trend`) should be ~0.

---

## 5) Real-Time Validation (Services)

**NATS subjects present**
- `market.futures.trades` and `market.futures.mbp10` are non-empty.
- `market.options.trades` is non-empty.

**Core outputs**
- `levels.signals` published at `CONFIG.SNAP_INTERVAL_MS`.
- `levels.signals.levels` non-empty once spot and MBP-10 are flowing.
- `market.flow` non-empty when options trades are flowing.

**Gateway outputs**
- `/ws/stream` payload includes `levels` with flattened fields.
- Direction is `UP|DOWN`, signal is `BREAK|BOUNCE|CHOP`.
- `viewport` appears only when `VIEWPORT_SCORING_ENABLED=true` and model/index files exist.

If `barrier_state` is always `NEUTRAL`, or `tape_velocity` is always 0, the ES streams are missing or misaligned. If `gamma_exposure` is always 0, options trades or greeks are missing.

---

## 6) Failure Modes and Likely Causes

- **All NEUTRAL barrier states**: ES MBP-10 not ingested or wrong symbol/time alignment.
- **All-zero gamma_exposure**: no options trades, missing Greeks, or 0DTE filter not matching.
- **Outcomes all UNDEFINED**: missing forward window; OHLCV coverage incomplete.
- **Distance > 0.25**: monitor-band filter not applied or spot conversion wrong.
- **Direction mismatches**: incorrect sign or distance_signed computation.

---

## 7) Historical vs Real-Time Summary

- **Historical** uses `vectorized_pipeline` to produce the training dataset at `data/lake/gold/research/signals_vectorized.parquet` with labels anchored at confirmation time.
- **Real-time** uses the service chain (ingestor -> core -> gateway) to stream live physics signals and optional viewport ML scoring over `/ws/stream`.

Both paths must be consistent with `backend/features.json` and the Phase 3 target-relative invariants. If the historical dataset passes the mathematical checks but real-time signals diverge, the issue is almost always ingestion timing or missing NATS feeds.
