# VP Derivative Signal Predictive Power Analysis

## Overview

Quantitative evaluation of vacuum-pressure (VP) derivative signals for directional mid-price prediction on MNQ (Micro E-mini Nasdaq 100) futures. The analysis tests whether the spatial structure of order book flow derivatives — velocity, acceleration, and jerk of depth changes across 101 price levels — contains statistically significant predictive information for short-horizon price direction.

## Dataset

| Parameter | Value |
|-----------|-------|
| Symbol | MNQH6 |
| Date | 2026-02-06 |
| Product type | future_mbo |
| Tick size | $0.25 |
| Grid radius | +/- 50 ticks around spot (101 levels) |
| Throttle | 25ms (max 40 snapshots/sec) |
| Training window | 09:25 - 09:45 ET (41,759 snapshots) |
| Evaluation window | 09:45 - 09:48 ET (6,693 snapshots) |
| Total snapshots | 48,452 |
| Mid price range | $24,701.00 - $24,948.75 (991 ticks) |

## Methodology

### Data Capture

Grid snapshots captured via `stream_events()` synchronous generator (no WebSocket overhead, no real-time pacing). Each snapshot produces a `(101, 15)` array of bucket fields extracted into a pre-allocated numpy buffer with dynamic growth.

### Feature Engineering

**20 spatial features** computed per snapshot via vectorized summation over bid/ask zone masks:

**PV Net Edge** (5 features) — the core directional signal combining pressure and vacuum asymmetry:
```
pv_net_edge = sum(pressure[k<0]) - sum(pressure[k>0]) + sum(vacuum[k>0]) - sum(vacuum[k<0])
```
Positive = bullish (pressure below + vacuum above exceeds reverse). Computed at full range, near (|k|<=5), mid (5<|k|<=20), far (20<|k|<=50), and 1/|k|-weighted.

**Depth Tilt** (4 features) — static depth asymmetry:
```
depth_tilt = sum(rest_depth[k<0]) - sum(rest_depth[k>0])
```
Full, near, mid, far field versions.

**Velocity Tilts** (4 features) — first-derivative flow asymmetry:
- `v_add_tilt`: bid-side add velocity excess (support building faster than resistance)
- `v_pull_tilt`: ask-side pull velocity excess (resistance weakening = bullish)
- `v_fill_tilt`: ask-side fill velocity excess (fills eating through asks = bullish)
- `v_depth_tilt`: bid-side depth velocity excess

**Acceleration Tilts** (4 features) — second-derivative flow asymmetry:
- Same structure as velocity, using `a_*` fields

**Jerk Features** (3 features) — third-derivative regime change detection:
- `jerk_magnitude`: max|j_add| + max|j_pull| across all k (unsigned, measures regime transition intensity)
- `j_add_tilt`, `j_pull_tilt`: directional jerk asymmetry

### Temporal Smoothing

Each spatial feature is z-scored at 4 EWM lookback windows:
- L=5 (~125ms), L=15 (~375ms), L=50 (~1.25s), L=150 (~3.75s)
- Half-life parameterization: `alpha = 1 - exp(-ln(2)/L)` via `pandas.ewm(halflife=L)`
- Z-score: `(EWM - rolling_mean) / rolling_std` with minimum 10 observations

Total: 20 features x 4 lookbacks = **80 z-scored signals**.

### Prediction Target

Forward mid-price return in ticks at three horizons:
- H=25 (~0.6s): std=12.79 ticks
- H=100 (~2.5s): std=25.37 ticks
- H=500 (~12.5s): std=59.50 ticks

### Evaluation Metrics

For each of 240 signal x horizon pairs:
- **Rank IC**: Spearman correlation of signal vs forward return
- **Hit rate**: directional accuracy (% where sign(signal) == sign(return))
- **t-stat**: IC / (1/sqrt(n)), statistical significance

### Composite Signal

Equal-weight combination of top-5 signals by |IC| on training set, sign-corrected so positive composite = bullish.

### Regime Conditioning

Evaluation window split at median jerk magnitude into DIRECTIONAL (high jerk) and CHOP (low jerk) sub-periods.

## Results

### Top Individual Signals (H=100, ~2.5s horizon, sorted by |Eval IC|)

| Signal | Train IC | Eval IC | Eval Hit | Eval t-stat |
|--------|----------|---------|----------|-------------|
| a_fill_tilt_L150 | -0.069 | **-0.226** | 44.9% | -18.8 |
| j_add_tilt_L150 | +0.096 | **+0.205** | 55.2% | +17.0 |
| j_pull_tilt_L150 | -0.103 | **-0.201** | 45.0% | -16.6 |
| a_add_tilt_L150 | +0.056 | **+0.202** | 55.3% | +16.7 |
| a_depth_tilt_L50 | +0.054 | **+0.200** | 55.7% | +16.5 |
| a_pull_tilt_L150 | -0.060 | **-0.199** | 45.3% | -16.5 |
| a_fill_tilt_L50 | -0.014 | **-0.192** | 44.1% | -15.9 |
| v_fill_tilt_L50 | -0.046 | **-0.185** | 45.6% | -15.3 |
| v_depth_tilt_L50 | +0.047 | **+0.174** | 53.7% | +14.4 |
| a_add_tilt_L50 | +0.023 | **+0.171** | 54.3% | +14.1 |
| a_pull_tilt_L50 | -0.018 | **-0.167** | 46.0% | -13.7 |
| pv_net_edge_kw_L50 | +0.063 | **+0.164** | 53.2% | +13.5 |
| pv_net_edge_L50 | +0.051 | **+0.163** | 54.4% | +13.4 |
| v_add_tilt_L50 | +0.050 | **+0.159** | 52.3% | +13.1 |
| v_pull_tilt_L50 | -0.051 | **-0.157** | 47.9% | -12.9 |

### Composite Signal Performance

| Metric | Value |
|--------|-------|
| Components | j_pull_tilt_L150, j_add_tilt_L150, jerk_magnitude_L150, a_fill_tilt_L150, pv_net_edge_mid_L50 |
| Rank IC (eval) | **+0.168** |
| Hit Rate (eval) | **55.0%** |
| Cumulative PnL | **+20,210 ticks** |
| PnL ($) | **$10,105** (MNQ $2/tick) |
| n_eval | 6,593 snapshots |

### Regime Analysis (Eval Window)

| Regime | IC | Hit Rate | PnL (ticks) | n |
|--------|-----|----------|-------------|------|
| DIRECTIONAL (high jerk) | +0.066 | 51.7% | +6,204 | 3,294 |
| CHOP (low jerk) | **+0.246** | **58.2%** | **+14,006** | 3,299 |

## Key Findings

### 1. Acceleration and jerk signals dominate

The highest out-of-sample ICs come from 2nd and 3rd derivative fields: `a_fill_tilt`, `j_add_tilt`, `j_pull_tilt`, `a_add_tilt`, `a_depth_tilt`. These capture regime transitions in order flow before they manifest in velocity or price. The fill acceleration tilt (`a_fill_tilt_L150`) achieves the single highest eval IC at -0.226.

### 2. Longer lookbacks outperform

L150 (~3.75s) and L50 (~1.25s) consistently produce stronger signals than L5/L15. This indicates the predictive information lives in the 1-4 second smoothing horizon rather than sub-second tick noise. The VP engine's built-in EMA derivative chain (tau=2s velocity, 5s acceleration, 10s jerk) combined with the analysis z-score lookback creates an effective smoothing window of 5-15 seconds.

### 3. Prediction quality is strongest in CHOP regimes

IC = 0.246 with 58.2% hit rate during low-jerk (choppy) periods vs IC = 0.066 in directional periods. This is the correct profile for a depth-structure-based signal: when price is range-bound, the order book's spatial pressure/vacuum balance provides the most useful information about which direction the next move will break. During strong directional moves, momentum dominates and book structure is less informative.

### 4. Signal holds out-of-sample

The composite signal shows consistent sign between training and evaluation windows. The eval IC of +0.168 with t-stat well above 2 indicates statistical significance. Individual signal ICs frequently strengthen from train to eval, suggesting the 09:45-09:48 window (post-open, settling into the session) has richer microstructure than the volatile 09:25-09:45 open.

### 5. Velocity features confirm the derivative hierarchy

Velocity tilts (v_add, v_pull, v_fill, v_depth at L50) cluster in the IC=0.15-0.19 range — strong but consistently below their acceleration counterparts. This confirms the derivative hierarchy: change-of-change (acceleration) is more predictive than change (velocity), and change-of-change-of-change (jerk) provides the earliest regime transition detection.

## Reproduction

### Prerequisites

- Raw data: `backend/lake/raw/source=databento/product_type=future_mbo/symbol=MNQ/table=market_by_order_dbn/glbx-mdp3-20260206.mbo.dbn`
- Book cache (optional, built automatically): `backend/lake/cache/vp_book/MNQH6_2026-02-06_*.pkl`

### Step 1: Warm cache (optional, saves ~6 min on first run)

```bash
cd backend
uv run scripts/warm_cache.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:25
```

### Step 2: Run analysis

```bash
cd backend
uv run scripts/analyze_vp_signals.py \
  --product-type future_mbo \
  --symbol MNQH6 \
  --dt 2026-02-06 \
  --start-time 09:25 \
  --eval-start 09:45 \
  --eval-minutes 3 \
  --throttle-ms 25 \
  --composite-horizon 100 \
  --top-k 5
```

Runtime: ~44 minutes (dominated by VP engine processing of ~48K snapshots through full derivative chain).

### Step 3: Vary parameters

```bash
# Different eval window
uv run scripts/analyze_vp_signals.py --eval-start 09:50 --eval-minutes 5

# Shorter lookback horizons
uv run scripts/analyze_vp_signals.py --composite-horizon 25 --top-k 3

# Different start time (changes warmup boundary and cache key)
uv run scripts/analyze_vp_signals.py --start-time 09:30 --eval-start 09:50
```

### Analysis script CLI

```
usage: analyze_vp_signals.py [-h] [--product-type PRODUCT_TYPE]
                             [--symbol SYMBOL] [--dt DT]
                             [--start-time START_TIME]
                             [--eval-start EVAL_START]
                             [--eval-minutes EVAL_MINUTES]
                             [--throttle-ms THROTTLE_MS]
                             [--composite-horizon COMPOSITE_HORIZON]
                             [--top-k TOP_K]
                             [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

## Files

| File | Purpose |
|------|---------|
| `backend/scripts/analyze_vp_signals.py` | Signal analysis script (this study) |
| `backend/scripts/warm_cache.py` | Pre-build book state cache |
| `backend/src/vacuum_pressure/stream_pipeline.py` | VP pipeline with `stream_events()` and `ensure_book_cache()` |
| `backend/src/vacuum_pressure/event_engine.py` | VP engine with derivative chain math |
