# VP Derivative-Only Directional Micro-Regime Analysis

## Overview

This analysis path is explicitly:

- derivative-only for signal construction
- no model fitting or historical training
- single-instrument runtime (`future_mbo:MNQH6` in this environment)

The objective is to detect directional micro regime changes as force derivatives
shift on both sides of spot, then score whether those shifts lead to tradable
continuation.

## Fixed Evaluation Window

Canonical replay target:

- date: `2026-02-06`
- instrument: `MNQH6`
- evaluation window: `09:00:00` to `12:00:00` ET

Important mechanics:

- Correct order-book construction still requires pre-window replay.
- With `--start-time 09:00`, pipeline runs:
  1. book bootstrap to `08:30` ET (cacheable)
  2. VP warmup from `08:30` to `09:00` ET
  3. emit/evaluate from `09:00` onward

So evaluation is `09:00-12:00`, while bootstrap/warmup remain mandatory for correctness.

## Data Path

Snapshots are captured directly from:

- `stream_events()` in `backend/src/vacuum_pressure/stream_pipeline.py`

No websocket transport is used by the analysis loop.
Dense grid shape is fixed (`K=50`, `101` buckets).

Per snapshot fields used:

- force variants: `pressure_variant`, `vacuum_variant`
- derivatives: `v_*`, `a_*`, `j_*`

## Directional Spectrum Construction

### 1. Spatial layers

For first pass, directional layers are built over symmetric bands:

- `+/-4`, `+/-8`, `+/-16`

Side slices:

- above side: `k>0` within band
- below side: `k<0` within band

### 2. Per-layer composite

For each band side:

```text
P_layer = mean(pressure_variant over side/band)
V_layer = mean(vacuum_variant over side/band)
C_layer = (P_layer - V_layer) / (P_layer + V_layer + eps)
```

Interpretation:

- `+1` pressure-dominant
- `0` transition/chop
- `-1` vacuum-dominant

### 3. Micro-window rollups

Trailing windows:

- `25,50,100,200` snapshots

Rollup is deterministic weighted blend of trailing means:

```text
roll(C) = sum_w alpha_w * trailing_mean(C, w), alpha_w ∝ 1/sqrt(w)
```

### 4. Derivative slope stack

For rolled composite:

```text
d1 = d(roll(C))/dt
d2 = d(d1)/dt
d3 = d(d2)/dt
```

Event-time deltas are used (`dt` from snapshot timestamps).
Each derivative is normalized with trailing robust z-score (median/MAD).

Layer directional score:

```text
score = 0.55*tanh(z(d1)/3) + 0.30*tanh(z(d2)/3) + 0.15*tanh(z(d3)/3)
```

### 5. Pressure / transition / vacuum states

With threshold `theta` (`--spectrum-threshold`):

- `score >= +theta` -> `pressure`
- `score <= -theta` -> `vacuum`
- otherwise -> `transition`

### 6. Side composites and directional edge

Band scores are side-aggregated with weights `1/sqrt(band)`:

- `above_side_score`
- `below_side_score`

Directional edge:

```text
direction_edge = below_side_score - above_side_score
```

Directional state:

- `up` when `direction_edge >= edge_threshold`
- `down` when `direction_edge <= -edge_threshold`
- `flat` otherwise

Posture state keeps trader-facing force vocabulary:

- `bullish_release`: above vacuum + below pressure
- `bearish_release`: above pressure + below vacuum
- `two_sided_vacuum`
- `two_sided_pressure`
- `transition`

## Trade-Style Evaluation Target

No synthetic labels are used.
Evaluation is event-driven and trader-style:

- detect directional switch events (`up`/`down` transitions with cooldown)
- at each event, enter at current mid-price
- success criterion:
  - reach `+TP` in predicted direction before `-SL` adverse move
  - defaults: `TP=8 ticks`, `SL=4 ticks`
  - timeout at `--max-hold-snapshots` if neither side is hit

Reported metrics:

- `tp_before_sl_rate`
- `sl_before_tp_rate`
- `timeout_rate`
- `events_per_hour`
- `median_time_to_outcome_ms`

## Hourly Stability Gate

The 3-hour window is split into:

- `09:00-10:00`
- `10:00-11:00`
- `11:00-12:00`

Gate checks:

- minimum events per hour (`--stability-min-signals-per-hour`)
- relative drift across hours for:
  - `tp_before_sl_rate`
  - `events_per_hour`
- max allowed drift: `--stability-max-drift`

This is a stability-first acceptance criterion, not an absolute PnL threshold.

## Forward Projection (Per Bucket, Per Horizon)

Per-bucket projection remains available and derivative-driven:

```text
P_hat = clip(P0 + dP_dt*h + 0.5*d2P_dt2*h^2, 0, inf)
V_hat = clip(V0 + dV_dt*h + 0.5*d2V_dt2*h^2, 0, inf)
C_hat = (P_hat - V_hat) / (P_hat + V_hat + eps)
```

This keeps the pressure -> transition -> vacuum interpretation at bucket level,
including focused buckets such as `k=+8` or `k=-8`.

## Reproduction

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --mode regime \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:00 \
  --eval-start 09:00 \
  --eval-end 12:00 \
  --directional-bands 4,8,16 \
  --micro-windows 25,50,100,200 \
  --tp-ticks 8 \
  --sl-ticks 4 \
  --max-hold-snapshots 1200 \
  --projection-horizons-ms 250,500,1000,2500 \
  --projection-rollup-windows 8,32,96 \
  --projection-buckets -8,8
```

## Test Methodology

Tests are in:

- `backend/tests/test_analyze_vp_signals_regime.py`
- `backend/tests/test_single_instrument_config_lock.py`

Strategy validation requirements:

- no synthetic strategy data
- replay real MNQ data
- use deterministic golden metrics for `09:00-12:00 ET`

Real replay tests are gated behind:

- `VP_ENABLE_REAL_REPLAY_TESTS=1`

Golden file:

- `backend/tests/golden_mnq_20260206_0900_1200.json`

Run:

```bash
cd backend
uv run pytest -q
VP_ENABLE_REAL_REPLAY_TESTS=1 uv run pytest -q
```

## Future Considerations

Planned follow-up evaluations:

1. Per-cell (grid-level) directional regime detection instead of only band rollups.
2. Wider temporal aggregation per grid cell before derivative extraction (multi-timescale cell synthesis).

These are not enabled in first pass and are intentionally deferred.

