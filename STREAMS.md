# STREAMS: Normalized Signal Streams + Forward Projections (2-minute bars)

> **IMPLEMENTATION STATUS**: Sections 1-5, 6, 7.1-7.2, 8-10 are **COMPLETE**. Sections 7.3-7.4, 11-12 remain **TODO**.

---

## IMPLEMENTATION GUIDE FOR AI CODING AGENTS

### 🎯 START HERE: System Status & Next Steps

**CURRENT STATE (December 30, 2025)**:
- ✅ **Streams computation is LIVE** - 5 canonical streams + derivatives working
- ✅ **Tested successfully** on 2025-12-16: 437 bars, all validation checks passed
- ✅ **Projection models COMPLETE** - Quantile polynomial forecasting (20-min ahead)
- ✅ **State machine rules COMPLETE** - 14 alert types with hysteresis
- ❌ **UI integration NOT COMPLETE** - Angular components not built

**IMMEDIATE NEXT TASK**: Implement Section 7.4 (Real-Time API) or Section 12 (UI Integration) - see roadmap below.

**QUICK START (for testing existing implementation)**:
```bash
cd backend
# Run pipeline for a date with state table data
uv run python -m scripts.run_pentaview_pipeline --date 2025-12-16
# Validate output
uv run python -m scripts.validate_pentaview --date 2025-12-16
```

### What This System Does
**Pentaview** transforms the ES level-interaction pipeline's 30-second state table into continuous, interpretable **streams** that emit scalar values in `[-1, +1]` every 2-minute bar. These streams provide TA-style signals (momentum, flow, barrier dynamics) plus forward projections (20-minute quantile forecasts).

### Architecture Overview

```
Stage 16 State Table (30s cadence)
       ↓
Pentaview Pipeline
       ↓
┌──────────────────────────────────────────────┐
│  Stream Normalization (robust/zscore/tanh)  │  ← backend/src/ml/stream_normalization.py
└──────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────┐
│  Aggregate 30s → 2-min bars                  │  ← backend/src/pipeline/stages/compute_streams.py
└──────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────┐
│  Stream Builder (5 canonical + derivatives)  │  ← backend/src/ml/stream_builder.py
│  • Σ_M (Momentum): velocity/accel/jerk      │
│  • Σ_F (Flow): OFI/tape/aggression          │
│  • Σ_B (Barrier): liq consumption w/dir_sign│
│  • Σ_D (Dealer): gamma regime amplifier      │
│  • Σ_S (Setup): proximity/quality scaler     │
│  • Σ_P (Pressure): 0.55*M + 0.45*F          │
│  • Σ_R (Structure): 0.70*B + 0.30*S         │
│  • Derivatives: slope/curvature/jerk (EMA)   │
└──────────────────────────────────────────────┘
       ↓
Gold Layer: gold/streams/pentaview/version=3.1.0/
```

### Key Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `backend/src/ml/stream_normalization.py` | Robust normalization (median/MAD + tanh) | ✅ COMPLETE |
| `backend/src/ml/stream_builder.py` | 5 canonical streams + derivatives | ✅ COMPLETE |
| `backend/src/pipeline/stages/compute_streams.py` | 30s→2min aggregation + DCT | ✅ COMPLETE |
| `backend/src/pipeline/pipelines/pentaview_pipeline.py` | Pipeline orchestration | ✅ COMPLETE |
| `backend/scripts/compute_stream_normalization.py` | Compute norm stats from state table | ✅ COMPLETE |
| `backend/scripts/run_pentaview_pipeline.py` | Run pipeline for date/range | ✅ COMPLETE |
| `backend/src/ml/stream_projector.py` | Quantile polynomial projection models | ✅ COMPLETE |
| `backend/src/ml/stream_state_machine.py` | TA-style rule engine (exhaustion/divergence) | ✅ COMPLETE |

### Critical Design Decisions

1. **Barrier Sign Convention**: `Σ_B = tanh(dir_sign * s_local)` where `dir_sign = +1 if direction=='UP' else -1`. This ensures barrier stream has **market semantics** (positive=favors up, negative=favors down) regardless of approach direction.

2. **Normalization Strategy**: 
   - Robust (median/MAD + tanh) for heavy-tailed features (OFI, tape, barrier deltas)
   - Z-score (mean/std + tanh) for symmetric features (velocity, acceleration)
   - Stratified by `time_bucket` (T0_15, T15_30, T30_60, T60_120, T120_180)
   - All streams bounded in `(-1, +1)`

3. **DCT Trajectory Encoding**: 40-bar (20-minute) windows for 4 series (d_atr, ofi_60s, barrier_delta_liq_log, tape_imbalance). Use first 8 coefficients to derive `trend_score` (c1+c2) and `chop_score` (high-freq energy ratio).

4. **Dealer as Amplifier**: `Σ_D` is **non-directional**. Use as multiplier: `A_adj = clamp(A * (1.0 + 0.35*Σ_D), -1, +1)`. Positive Σ_D = fuel regime (amplifies moves), negative = pin regime (dampens).

5. **Derivatives via EMA**: Smooth raw streams first (`halflife=3 bars`), then compute discrete differences for slope/curvature/jerk. Prevents noise amplification.

### Data Flow

```
Input:  silver/state/es_level_state/date=YYYY-MM-DD/*.parquet (86 columns)
        ↓ (1729 samples @ 30s for 2025-12-16)
Compute: gold/streams/normalization/current.json (58 features, 5 strata)
        ↓
Aggregate: 30s → 2-min bars (87 bars per level per day)
        ↓
Build Streams: 5 canonical (Σ_M, Σ_F, Σ_B, Σ_D, Σ_S) + 2 merged (Σ_P, Σ_R) + 3 composites
        ↓
Compute Derivatives: EMA smooth + slope/curvature/jerk for 4 streams
        ↓
Output:  gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD/stream_bars.parquet (32 columns)
         (437 bars for 5 levels: OR_HIGH, OR_LOW, PM_LOW, SMA_90, EMA_20)
```

### State Table → Stream Mapping (Quick Reference)

| Stream | Source Columns from State Table | Formula Weight |
|--------|--------------------------------|----------------|
| **Σ_M** | velocity_{1,3,5,10,20}min<br>acceleration_{1,3,5,10,20}min<br>jerk_{1,3,5}min<br>momentum_trend_{3,5,10,20}min | 0.40*vel + 0.30*acc<br>+ 0.15*jerk + 0.15*trd |
| **Σ_F** | ofi_60s, ofi_near_level_60s<br>ofi_acceleration<br>tape_imbalance, tape_velocity<br>flow_alignment (missing*)<br>DCT(ofi), DCT(tape) | 0.25*ofi + 0.20*ofi_level<br>+ 0.25*imb + 0.15*acc<br>+ 0.10*aln + 0.05*shape |
| **Σ_B** | barrier_delta_liq_log (missing*)<br>barrier_delta_3min<br>barrier_state_encoded<br>barrier_replenishment_ratio<br>**+ dir_sign multiplier** | dir_sign × tanh(<br>0.50*consume + 0.25*rate<br>+ 0.15*state + 0.10*repl) |
| **Σ_D** | fuel_effect_encoded<br>gamma_exposure<br>gex_ratio<br>net_gex_2strike | 0.45*fuel + 0.25*(-ge)<br>+ 0.15*(-ratio)<br>+ 0.15*(-abs(local)) |
| **Σ_S** | distance_signed_atr<br>approach_velocity, approach_bars<br>attempt_index, prior_touches<br>level_stacking_{5,10}pt<br>time_since_last_touch<br>DCT(d_atr) | proximity + approach<br>+ freshness + confluence<br>+ cleanness - chop |
| **Σ_P** | Σ_M, Σ_F | tanh(0.55*M + 0.45*F) |
| **Σ_R** | Σ_B, Σ_S | tanh(0.70*B + 0.30*S) |

**\*Missing features**: Currently defaulted to 0.0. Add to Stage 16 for full implementation.

### Testing

```bash
# 1. Compute normalization stats (required first)
cd backend
uv run python -m scripts.compute_stream_normalization \
  --data-root data \
  --lookback-days 60 \
  --end-date 2024-12-31 \
  --output-name current

# 2. Run pipeline for single date
uv run python -m scripts.run_pentaview_pipeline \
  --date 2024-12-16 \
  --data-root data \
  --canonical-version 3.1.0

# 3. Verify output
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/gold/streams/pentaview/version=3.1.0/date=2024-12-16/stream_bars.parquet')
print(df[['timestamp', 'level_kind', 'sigma_m', 'sigma_f', 'sigma_p']].head())
"

# 4. Run validation checks
uv run python -m scripts.validate_pentaview --date 2024-12-16 --data-root data
```

### What's Left to Implement

**MEDIUM PRIORITY:**
1. **Section 7.4: Real-Time API** - WebSocket inference schemas for projection + alert broadcasting.
   - **Complexity**: MEDIUM (API design, integration with core service)
   - **Estimated effort**: 2-3 hours
   - **Dependencies**: Sections 6 & 10 (NOW COMPLETE)

**LOW PRIORITY:**
3. **Section 12: UI Encoding** - Angular/frontend visualization rules (color hue, intensity, arrow glyphs).
   - **Complexity**: LOW (frontend component updates)
   - **Estimated effort**: 1-2 hours
   - **Dependencies**: Stream WebSocket integration

### Current Implementation Capabilities

**✅ You can already:**
- Compute 5 canonical streams (Momentum, Flow, Barrier, Dealer, Setup) from state table
- Compute 2 merged streams (Pressure, Structure) + alignment metrics
- Calculate derivatives (slope, curvature, jerk) for all directional streams
- Run batch processing for date ranges
- Output normalized, bounded streams in [-1, +1] with clear sign semantics
- **Forecast future stream values** - 20-minute quantile projections (q10/q50/q90)
- **Generate rule-based alerts** - 14 alert types with hysteresis (exhaustion, divergence, etc.)
- **Position-aware exit scoring** - LONG/SHORT recommendations (HOLD/REDUCE/EXIT)

**❌ You cannot yet:**
- Broadcast projections + alerts in real-time (no WebSocket API integration)
- Visualize streams in UI (no Angular components)

---

## 0) Product goal

Produce **continuous, interpretable streams** from the ES level-interaction pipeline that:

- Emit **scalar values in [-1, +1]** each **2-minute bar close**.
- Have well-defined **sign semantics** (buy vs sell / up vs down) so a trader can read a chart instantly.
- Support **1st/2nd/3rd derivatives** (slope/curvature/jerk) for “acceleration / deceleration” style TA.
- Are “mergeable” into higher-level **super-streams** without losing interpretability.
- Are forecastable: project **median + uncertainty band** out **H bars** (H=10 → 20 minutes) at every bar close.

This spec defines:

1. How to map raw columns into **feature families**.
2. How to normalize each family robustly.
3. The **canonical streams** and optional **merged streams**.
4. Derivative and synergy semantics (color + slope + jerk).
5. Projection model spec (smooth multi-horizon quantile forecasts) + schemas.

---

## 1) Column meaning audit (feature families) ✅ **COMPLETE**

**Implementation**: `backend/src/ml/stream_builder.py` - Feature family mappings implemented in `compute_*_stream()` functions.

### 1.1 Price kinematics (directional)
These are “pure price motion” descriptors at multiple windows.

- `velocity_{1,3,5,10,20}min`  → 1st derivative of price (preferably ATR-normalized)
- `acceleration_{1,3,5,10,20}min` → 2nd derivative
- `jerk_{1,3,5,10,20}min` → 3rd derivative / inflection
- `momentum_trend_{3,5,10,20}min` → persistence / directional stability

**Interpretation**: These should already be market-signed (up positive, down negative).

### 1.2 Aggression + flow (directional)
These quantify **who is lifting / hitting** and whether flow is accelerating.

- `ofi_{30s,60s,120s,300s}` → order flow imbalance
- `ofi_near_level_{30s,60s,120s,300s}` → OFI restricted to liquidity near the level
- `ofi_acceleration` → 2nd derivative of flow
- `tape_imbalance[t-k]` → market order imbalance (buy vs sell)
- `tape_velocity[t-k]` → how fast the tape is printing
- `tape_log_ratio` → log(buy_vol/sell_vol)
- `flow_alignment` → whether flow agrees with approach direction / local trend

**Interpretation**: These should be market-signed (buy positive, sell negative).

### 1.3 Barrier / liquidity dynamics (directional *after sign alignment*)
These quantify whether a level is being **defended (replenished)** or **absorbed (consumed)**.

Typical columns:

- `barrier_delta_{1,3,5}min` → barrier consumption rate
- `barrier_pct_change_{1,3,5}min` → relative change in barrier size
- `barrier_delta_liq_log[t-k]` → log change in resting liquidity
- `wall_ratio_log[t-k]` → log(bid_liq/ask_liq) asymmetry near level
- `barrier_state_encoded` → categorical state (e.g., [-2..+2])
- `barrier_replenishment_ratio` → rebuild / consume
- `barrier_replenishment_trend` → trend of rebuilding
- `barrier_delta_liq_trend` → trend of consumption
- `mass_proxy` → log1p(barrier_depth) or similar

**Interpretation**:

- Raw barrier metrics are often **side-relative** (depends on whether we are attacking from below or above).
- To make a single stream with consistent market meaning, we must multiply by a **direction sign**.

Define:

- `dir_sign = +1` when `direction == UP` (approaching from below / bullish attack)
- `dir_sign = -1` when `direction == DOWN` (approaching from above / bearish attack)

Then we can build a barrier stream that is market-signed:

- Positive → conditions favor continuation **up**
- Negative → conditions favor continuation **down**

### 1.4 Dealer / gamma regime (non-directional amplifier)
These quantify whether the options dealer environment **amplifies** moves ("fuel") or **dampens** ("pin").

- `gamma_exposure` (and history)
- `fuel_effect_encoded` ∈ {+1 amplify, 0 neutral, -1 dampen}
- `gex_ratio`, `gex_asymmetry`
- `net_gex_2strike`, `gex_above_1strike`, `gex_below_1strike`, tail measures

**Interpretation**:

- Treat the **primary dealer stream as regime**, not buy/sell direction.
- Use it as a **multiplier** on directional conviction.

### 1.5 Setup / context quality (non-directional)
These measure whether the approach is “clean” and the level interaction is meaningful.

- `d_atr[t-k]` → distance-to-level in ATR units (proximity)
- `approach_velocity`, `approach_bars`, `approach_distance_atr`
- `prior_touches`, `attempt_index`, `time_since_last_touch_sec`
- `level_stacking_{2,5,10}pt` → confluence / clustering of levels

**Interpretation**:

- Setup is **not buy/sell**. It is a **confidence/quality** scaler.

### 1.6 Trajectory shape encodings (DCT blocks)
DCT coefficients are compact summaries of recent trajectory shape.

- `DCT(d_atr)[c0..c7]`
- `DCT(ofi_60s)[c0..c7]`
- `DCT(barrier_delta_liq_log)[c0..c7]`
- `DCT(tape_imbalance)[c0..c7]`

**Interpretation** (recommended usage):

- `c1` ≈ trend component
- `c2` ≈ curvature / concavity
- higher `c` → higher frequency / "chop" energy

Use DCT primarily to derive:

- **trend_score** (low-frequency) and
- **chop_score** (high-frequency energy ratio)

rather than dumping all coefficients directly into a stream.

---

## 2) Normalization to [-1, +1] (robust + stable) ✅ **COMPLETE**

**Implementation**: `backend/src/ml/stream_normalization.py`
- `compute_stream_normalization_stats()` - Computes median/MAD/mean/std from 60-day lookback
- `normalize_feature_robust()` - Robust normalization: `(x - median) / (1.4826 * MAD)` + tanh squashing
- `normalize_feature_zscore()` - Z-score normalization: `(x - mean) / std` + tanh squashing
- `normalize_feature()` - Dispatcher with stratified stats lookup by time_bucket
- Stats saved to: `gold/streams/normalization/current.json` (58 features, 5 time buckets)

### 2.1 Feature-level normalization
Every raw feature feeding a stream must be mapped to a comparable scale.

**Signed continuous features** (velocity, OFI, tape imbalance, deltas, ratios already log’d):

```
robust_z(x; median, mad) = (x - median) / (1.4826 * mad + eps)
clip_z = clamp(robust_z, -z_clip, +z_clip)         # z_clip ~ 6
norm_signed = tanh(clip_z / z_scale)               # z_scale ~ 2
# output in (-1, +1)
```

**Positive-only magnitude features** (mass_proxy, depth):

```
robust_z_pos = (x - median) / (1.4826 * mad + eps)
clip_z = clamp(robust_z_pos, -z_clip, +z_clip)
mag_0_1 = sigmoid(clip_z)                           # output in (0,1)
# use as a multiplier/modulator, not a directional contributor
```

**Encoded categorical** (`barrier_state_encoded` in [-2..+2]):

```
norm_cat = clamp(barrier_state_encoded / 2.0, -1, +1)
```

**Fuel effect** (`fuel_effect_encoded` in {-1,0,+1}):

```
fuel = fuel_effect_encoded                           # already in [-1,+1]
```

### 2.2 Where do the medians/MADs come from?

Store robust stats on the training universe with **stratification** to prevent regime bleed:

- instrument (ES)
- `time_bucket` (e.g., open / mid / late)
- optionally `or_active` and/or `vol_bucket`

At inference, choose the appropriate stats bucket; fall back to global stats if missing.

### 2.3 Stream-level squashing
After aggregating normalized components into a stream raw score `s_raw`:

```
Σ = tanh(s_raw)
# ensures Σ ∈ (-1, +1)
```

---

## 3) Canonical streams (keep these separate) ✅ **COMPLETE**

**Implementation**: `backend/src/ml/stream_builder.py`
- `compute_momentum_stream()` - Σ_M: 0.40*vel + 0.30*acc + 0.15*jerk + 0.15*trend
- `compute_flow_stream()` - Σ_F: OFI + tape + DCT shape
- `compute_barrier_stream()` - Σ_B: **WITH dir_sign CORRECTION** for market semantics
- `compute_dealer_stream()` - Σ_D: fuel + gamma (non-directional amplifier)
- `compute_setup_stream()` - Σ_S: proximity + confluence + DCT cleanness
- `compute_all_streams()` - Orchestrates all 5 + merged streams (Σ_P, Σ_R) + composites

**Critical**: Barrier stream applies `dir_sign = +1 if direction=='UP' else -1` to ensure market-signed output.

**Design rule**: each stream answers **one question** and uses mostly one feature family.

### 3.1 Σ_M: MOMENTUM (directional)
**Question**: Is price pressure building or dissipating?

Inputs (normalized):

- `velocity_*` (multi-scale)
- `acceleration_*` (multi-scale)
- `jerk_*` (short scale emphasized)
- `momentum_trend_*`

Aggregation (example):

```
vel = wavg([norm(v1), norm(v3), norm(v5), norm(v10), norm(v20)], decay=0.7)
acc = wavg([norm(a1), norm(a3), norm(a5), norm(a10), norm(a20)], decay=0.7)
jer = wavg([norm(j1), norm(j3), norm(j5)], decay=0.8)
trd = wavg([norm(mt3), norm(mt5), norm(mt10), norm(mt20)], decay=0.7)

s_raw = 0.40*vel + 0.30*acc + 0.15*jer + 0.15*trd
Σ_M = tanh(s_raw)
```

Semantics:

- `Σ_M > 0` → upward momentum regime
- `Σ_M < 0` → downward momentum regime

### 3.2 Σ_F: FLOW (directional)
**Question**: Is aggressive buying/selling increasing, and is it aligned with trend?

Inputs (normalized):

- `ofi_*`, `ofi_near_level_*`
- `ofi_acceleration`
- `tape_imbalance[t-k]`, `tape_velocity[t-k]`
- `tape_log_ratio`, `flow_alignment`
- DCT-derived: `trend_score(ofi)` and `trend_score(tape_imbalance)`

Aggregation (example):

```
ofi_core   = norm(ofi_60s)
ofi_level  = norm(ofi_near_level_60s)
imb        = mean([norm(tape_imb[t-2]), norm(tape_imb[t-1]), norm(tape_imb[t0])])
acc_flow   = norm(ofi_acceleration)
aln        = norm(flow_alignment)
shape      = 0.5*trend_score(DCT_ofi) + 0.5*trend_score(DCT_tape)

s_raw = 0.25*ofi_core + 0.20*ofi_level + 0.25*imb + 0.15*acc_flow + 0.10*aln + 0.05*shape
Σ_F = tanh(s_raw)
```

Semantics:

- `Σ_F > 0` → net aggressive buying
- `Σ_F < 0` → net aggressive selling

### 3.3 Σ_B: BARRIER (directional *via dir_sign*)
**Question**: Is the level’s resting liquidity being absorbed (break) or defended (reject)?

Key requirement: **market-signed output**.

Define:

- `dir_sign = +1` if `direction == UP`, else `-1`.

Build two latent sub-scores:

- **absorption_score** (consumption dominates)
- **defense_score** (replenishment dominates)

Example mapping (assumes negative delta_liq means consumption):

```
consume = mean([ -norm(barrier_delta_liq_log[t-2]), -norm(barrier_delta_liq_log[t-1]), -norm(barrier_delta_liq_log[t0]) ])
rate    = norm(barrier_delta_3min)
wall    = mean([norm(wall_ratio_log[t-1]), norm(wall_ratio_log[t0])])
state   = norm_cat(barrier_state_encoded)

repl    = norm(barrier_replenishment_ratio - 1.0)          # >0 means rebuilding
repl_tr = norm(barrier_replenishment_trend)

absorption_score = 0.45*consume + 0.25*(-rate) + 0.15*(-state) + 0.15*(-repl)
# defense_score is implicit via repl/state/wall; keep symmetric if desired

s_raw_local = absorption_score
Σ_B = tanh( dir_sign * s_raw_local )
```

Notes:

- If your pipeline already signs barrier deltas by direction, then drop `dir_sign`.
- If `barrier_state_encoded` uses opposite polarity, flip it once here and never again.

Semantics:

- `Σ_B > 0` → barrier conditions favor continuation **up**
- `Σ_B < 0` → barrier conditions favor continuation **down**

### 3.4 Σ_D: DEALER / GAMMA REGIME (non-directional)
**Question**: Are dealers likely to dampen (pin) or amplify (fuel) moves?

Inputs:

- `fuel_effect_encoded` (primary)
- `gamma_exposure` / `gex_ratio` / `net_gex_2strike`

Aggregation (example):

```
fuel   = fuel_effect_encoded                           # +1 amplify, -1 dampen
ge     = norm(gamma_exposure)
ratio  = norm(gex_ratio)
local  = norm_signed(net_gex_2strike)

s_raw = 0.45*fuel + 0.25*(-ge) + 0.15*(-ratio) + 0.15*(-abs(local))
Σ_D = tanh(s_raw)
```

Recommended semantics:

- `Σ_D > 0` → **amplification regime** ("fuel")
- `Σ_D < 0` → **dampening regime** ("pin")

(If your internal gamma convention differs, flip once here.)

### 3.5 Σ_S: SETUP / QUALITY (non-directional)
**Question**: Is this approach + level context clean enough to trust signals + projections?

Inputs:

- proximity: `d_atr[t0]`
- approach: `approach_velocity`, `approach_bars`, `approach_distance_atr`
- memory: `attempt_index`, `prior_touches`, `time_since_last_touch_sec`
- confluence: `level_stacking_*`
- trajectory cleanliness: DCT-derived chop/trend of `d_atr`

Aggregation (example):

```
proximity = exp(-abs(d_atr[t0]) / 1.5)                # in (0,1]
recency   = exp(-time_since_last_touch_sec / 900.0)   # in (0,1]

freshness = 1.0 if attempt_index <= 2 else 0.6 if attempt_index <= 4 else 0.3
confluence = clamp((level_stacking_5pt + level_stacking_10pt) / 12.0, 0, 1)

approach_speed = tanh(norm(approach_velocity))         # [-1,1]
clean_trend    = 0.5 + 0.5*trend_score(DCT_d_atr)      # [0,1]
chop_penalty   = chop_score(DCT_d_atr)                 # [0,1] higher=choppier

q_0_1 = 0.28*proximity + 0.18*(0.5+0.5*approach_speed) + 0.18*freshness + 0.16*confluence + 0.10*recency + 0.10*clean_trend - 0.10*chop_penalty
q_0_1 = clamp(q_0_1, 0, 1)

Σ_S = 2*q_0_1 - 1                                      # map to [-1,+1]
```

Semantics:

- `Σ_S > +0.5` → high-quality setup
- `Σ_S < -0.5` → degraded / noisy setup

---

## 4) Derivatives: slope, curvature, jerk (TA-grade) ✅ **COMPLETE**

**Implementation**: `backend/src/ml/stream_builder.py` - `compute_derivatives()`
- Applies EMA smoothing (halflife=3 bars) to raw stream values
- Computes discrete differences: `slope = Σ̄[t] - Σ̄[t-1]`, `curvature = slope[t] - slope[t-1]`, `jerk = curvature[t] - curvature[t-1]`
- Returns: `{smooth, slope, curvature, jerk}` for each stream
- Used for exhaustion detection (positive stream, negative slope = fading pressure)

### 4.1 Smooth first, then differentiate
Raw stream values will be noisy. Use an EWMA-smoothed stream for derivatives.

```
Σ̄_x[t] = ema(Σ_x[t], halflife_bars = 3)

slope_x[t]     = Σ̄_x[t] - Σ̄_x[t-1]          # 1st difference (per bar)
curvature_x[t] = slope_x[t] - slope_x[t-1]    # 2nd difference
jerk_x[t]      = curvature_x[t] - curvature_x[t-1]
```

Optional (more stable): linear-regression slope over last k bars:

```
slope_k = linreg_slope( Σ̄_x[t-k+1 : t] )
```

### 4.2 Derivative semantics (directional streams)

For `x ∈ {M, F, B}`:

- `Σ_x > 0` and `slope_x > 0` → bullish pressure building
- `Σ_x > 0` and `slope_x < 0` → bullish pressure fading (exhaustion)
- `Σ_x < 0` and `slope_x < 0` → bearish pressure building
- `Σ_x < 0` and `slope_x > 0` → bearish pressure fading

**Jerk usage**:

- `jerk_x` spike opposite current position bias → early regime-change warning
- Use jerk as a **booster / alert**, not a standalone entry.

### 4.3 Color + sign convention for UI

Directional stream line color = `sign(Σ̄_x)`:

- `Σ̄_x >= 0` → green family
- `Σ̄_x < 0` → red family

Slope glyph = `sign(slope_x)`:

- strong up: `▲`
- weak up: `△`
- flat: `─`
- weak down: `▽`
- strong down: `▼`

Interpretation example:

- Green line + `▼` slope glyph → “still net buying, but buying is decelerating”
- Red line + `▲` slope glyph → “still net selling, but selling is decelerating”

---

## 5) Synergy + merged "super-streams" ✅ **COMPLETE**

**Implementation**: `backend/src/ml/stream_builder.py` - `compute_all_streams()`
- **Σ_P (Pressure)**: `tanh(0.55*Σ_M + 0.45*Σ_F)` - Primary TA-style line
- **Σ_R (Structure)**: `tanh(0.70*Σ_B + 0.30*Σ_S)` - Microstructure support
- **Alignment**: `(Σ_M + Σ_F + Σ_B)/3 * (0.6 + 0.4*(0.5 + 0.5*Σ_S))` - Setup-weighted consensus
- **Divergence**: `std([Σ_M, Σ_F, Σ_B])` - Directional disagreement metric
- **Alignment_adj**: `clamp(A * (1.0 + 0.35*Σ_D), -1, +1)` - Dealer-scaled conviction

Keep the 5 canonical streams for diagnostics.

For decisioning + compact UI, compute **two merged directional streams** plus one composite.

### 5.1 Σ_P: PRESSURE (Momentum + Flow)
**Question**: Are price movement and aggression aligned?

```
Σ_P = tanh( 0.55*Σ_M + 0.45*Σ_F )

# Optional booster: reward agreement in slopes, penalize divergence
boost = 0.15 * tanh( sign(slope_M) * sign(slope_F) * min(abs(slope_M), abs(slope_F)) / s0 )
Σ_P = clamp( Σ_P + boost, -1, +1 )
```

Semantics:

- `Σ_P` is the cleanest “single line” for a TA-style chart.
- Color: green/red by sign.
- Slope: building/fading pressure.

### 5.2 Σ_R: STRUCTURE (Barrier + Setup)
**Question**: Is microstructure + context supportive of continuation?

```
# Setup is a confidence scaler
Σ_R = tanh( 0.70*Σ_B + 0.30*Σ_S )

# Alternative (setup as weight): Σ_R = Σ_B * (0.6 + 0.4*(0.5 + 0.5*Σ_S))
```

Semantics:

- `Σ_R` positive → structure supports upward continuation; negative supports downward.

### 5.3 Alignment + divergence (consensus diagnostics)

Directional alignment (exclude dealer, treat setup as weight):

```
A_dir = (Σ_M + Σ_F + Σ_B) / 3
A = A_dir * (0.6 + 0.4*(0.5 + 0.5*Σ_S))
```

Divergence:

```
D = std([Σ_M, Σ_F, Σ_B])     # 0..~1
```

### 5.4 Dealer as multiplier

Use dealer regime to scale conviction (not to flip direction):

```
A_adj = clamp( A * (1.0 + kD*Σ_D), -1, +1 )
# kD ~ 0.25..0.50
```

Interpretation:

- `Σ_D > 0` (fuel) increases the impact of alignment
- `Σ_D < 0` (pin) reduces the impact

---

## 6) Projection model (2-min cadence, smooth curves, quantiles) ✅ **COMPLETE**

**Status**: IMPLEMENTED (December 30, 2025) - Transforms historical streams into forward-looking guidance.

**Implementation**: `backend/src/ml/stream_projector.py` + training pipeline scripts.

**Recommended Approach**:
1. **Create training dataset** from historical stream bars (Section 7.3 schema):
   - Input: L=20 bars of stream history + derivatives + context
   - Target: H=10 future stream values (Σ_P[t+1..t+10])
   - One sample per bar per level (filter to high-quality setups: Σ_S > -0.25)

2. **Model Architecture** (per stream, e.g., Pressure):
   - Input: Stream history (20 bars) + derivatives + cross-stream context + static features
   - Output: 9 coefficients (3 quantiles × 3 polynomial coefficients)
     - `q10: {a1, a2, a3}` - Lower bound
     - `q50: {a1, a2, a3}` - Median forecast
     - `q90: {a1, a2, a3}` - Upper bound
   - Loss: Pinball loss on generated polynomial path (Section 6.4)

3. **Files to Create**:
   - `backend/src/ml/stream_projector.py` - Model training + inference
   - `backend/scripts/train_projection_models.py` - Training script
   - `backend/scripts/build_projection_dataset.py` - Dataset builder

4. **Why Polynomial Coefficients?**
   - Guarantees smooth curves (no jagged forecasts)
   - Direct TA interpretation: a1=slope, a2=curvature, a3=jerk
   - Easy to visualize (11 points: current + 10 future)

5. **Training Data Construction** (pseudo-code for AI agent):
```python
# For each historical stream bar at timestamp t with sufficient future:
def build_training_sample(stream_bars_df, idx, lookback=20, horizon=10):
    if idx < lookback or idx + horizon >= len(stream_bars_df):
        return None  # Insufficient history or future
    
    bar = stream_bars_df.iloc[idx]
    
    # Extract lookback history (L=20 bars before t)
    hist_start = idx - lookback
    hist_slice = stream_bars_df.iloc[hist_start:idx]
    
    # Extract future targets (H=10 bars after t)
    future_slice = stream_bars_df.iloc[idx+1:idx+1+horizon]
    
    return {
        'sample_id': f"{bar.timestamp}_{bar.level_kind}",
        'level_kind': bar.level_kind,
        'direction': bar.direction,
        'time_bucket': get_time_bucket(bar.minutes_since_open),
        
        # Lookback history
        'sigma_p_hist': hist_slice['sigma_p'].values,  # [L]
        'sigma_m_hist': hist_slice['sigma_m'].values,
        # ... other streams ...
        'slope_p_hist': hist_slice['sigma_p_slope'].values,
        
        # Targets (what we're trying to predict)
        'sigma_p_target': future_slice['sigma_p'].values,  # [H]
        'sigma_m_target': future_slice['sigma_m'].values,
        # ... other streams ...
        
        # Context
        'atr': bar.atr,
        'setup_weight': bar.sigma_s,  # Sample weight
    }
```

6. **Model Training Strategy**:
   - Separate model per stream (Pressure, Flow, Barrier, etc.)
   - Optionally separate by level_kind and/or direction for regime specificity
   - Use gradient boosting (LightGBM/XGBoost) or neural network
   - Multi-task output: 9 coefficients (q10/q50/q90 × a1/a2/a3)
   - Weight samples by `setup_weight = (sigma_s + 1) / 2` to emphasize high-quality setups

### 6.1 What to predict
Predict **future stream values** at horizons `h = 1..H` (H=10).

Targets:

- `Σ_P[t+h]` (primary)
- optionally also `Σ_M, Σ_F, Σ_B, Σ_R` (diagnostic projections)

### 6.2 Recommended model: Quantile Polynomial Projection head
We want **fluid projected curves** with interpretable slope/curvature.

For each stream `x`, parameterize the forecast path as a low-degree polynomial in horizon `h`:

```
ŷ_x,q(h) = clamp( Σ̄_x[t] + a1_x,q*h + 0.5*a2_x,q*h^2 + (1/6)*a3_x,q*h^3, -1, +1 )

# Choose degree:
# - Linear:   a2=a3=0
# - Quadratic: a3=0
# - Cubic: include a3
```

The model outputs coefficients `{a1,a2,a3}` for each quantile `q ∈ {0.10, 0.50, 0.90}`.

This guarantees:

- smoothness across horizons (no jagged multi-step path)
- direct linkage to TA language: slope (a1), curvature (a2), jerk (a3)

### 6.3 Inputs (per stream model)
At bar close `t`, with lookback `L=20` bars:

- Stream history: `Σ_x[t-L+1..t]`
- Derivatives history: `slope_x[t-L+1..t]`, `curvature_x[...]`
- Cross-stream context: `{Σ_M, Σ_F, Σ_B, Σ_D, Σ_S}` histories (short)
- Stream-specific raw features (example):
  - Momentum model: velocity/accel/jerk multi-scale
  - Flow model: OFI/tape features
  - Barrier model: barrier + replenishment + wall ratio + state
  - Setup model: distance/attempt/confluence
  - Dealer model: gamma fields
- Static context: `level_kind`, `direction`, `time_bucket`, `attempt_index`, confluence
- Known future: `minutes_since_open[t+h]`, `or_active[t+h]` (optional)

### 6.4 Loss (pinball on the generated path)
For each stream `x`, each horizon `h`, each quantile `q`:

```
L = Σ_x Σ_h Σ_q  w_h * pinball( Σ_x[t+h] - ŷ_x,q(h), q )

w_h = decay^(h-1)          # decay ~ 0.90
```

### 6.5 Output objects
Return quantile bands as **11 points** (current + 10 future):

- `x_axis = [t, t+1, ..., t+H]`
- `median = [Σ̄_x[t], ŷ_q50(1), ..., ŷ_q50(H)]`
- `lower  = [Σ̄_x[t], ŷ_q10(1), ..., ŷ_q10(H)]`
- `upper  = [Σ̄_x[t], ŷ_q90(1), ..., ŷ_q90(H)]`

---

## 7) Schemas (pre-train + post-train) ⚠️ **PARTIALLY COMPLETE**

**Status**: 
- ✅ 7.1 BAR_FEATURES_SCHEMA - Implemented (2-min bars from state table aggregation)
- ✅ 7.2 STREAM_BAR_SCHEMA - Implemented (output schema with 32 columns)
- ❌ 7.3 TRAINING_SAMPLE_SCHEMA - TODO (for projection model training)
- ❌ 7.4 INFERENCE_REQUEST/RESPONSE_SCHEMA - TODO (for real-time API)

### 7.1 Schema: 2-minute BAR FEATURES (post-aggregation)
One row per 2-minute bar close per active level-episode.

```
BAR_FEATURES_SCHEMA:

# Identifiers
episode_id:            string
timestamp_ns:          int64
bar_index:             int32
symbol:                string      # "ES"

# Context
level_kind:            string      # {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90, EMA_20, ...}
direction:             string      # {UP, DOWN}
time_bucket:           string      # {T0_15, T15_30, ...}
level_price:           float64
spot:                  float64
atr:                   float64
attempt_index:         int32
prior_touches:         int32
time_since_last_touch_sec: float32
level_stacking_2pt:    int32
level_stacking_5pt:    int32
level_stacking_10pt:   int32

# Kinematics
velocity_1min:         float32
velocity_3min:         float32
velocity_5min:         float32
velocity_10min:        float32
velocity_20min:        float32
acceleration_1min:     float32
acceleration_3min:     float32
acceleration_5min:     float32
acceleration_10min:    float32
acceleration_20min:    float32
jerk_1min:             float32
jerk_3min:             float32
jerk_5min:             float32
jerk_10min:            float32
jerk_20min:            float32
momentum_trend_3min:   float32
momentum_trend_5min:   float32
momentum_trend_10min:  float32
momentum_trend_20min:  float32

# Flow
ofi_30s:               float32
ofi_60s:               float32
ofi_120s:              float32
ofi_300s:              float32
ofi_near_level_30s:    float32
ofi_near_level_60s:    float32
ofi_near_level_120s:   float32
ofi_near_level_300s:   float32
ofi_acceleration:      float32

tape_imbalance_t0:     float32
tape_imbalance_t1:     float32
tape_imbalance_t2:     float32
tape_imbalance_t3:     float32
tape_imbalance_t4:     float32

tape_velocity_t0:      float32
...

tape_log_ratio:        float32
flow_alignment:        float32

# Barrier
barrier_delta_1min:    float32
barrier_delta_3min:    float32
barrier_delta_5min:    float32
barrier_pct_change_1min: float32
barrier_pct_change_3min: float32
barrier_pct_change_5min: float32

barrier_delta_liq_log_t0: float32
barrier_delta_liq_log_t1: float32
...
wall_ratio_log_t0:     float32
...
barrier_state_encoded: int32       # expected [-2..+2]
barrier_replenishment_ratio: float32
barrier_replenishment_trend: float32
barrier_delta_liq_trend: float32
mass_proxy:            float32

# Dealer / gamma
fuel_effect_encoded:   int32       # {-1,0,+1}
gamma_exposure:        float32
gex_ratio:             float32
gex_asymmetry:         float32
net_gex_2strike:       float32
gex_above_1strike:     float32
gex_below_1strike:     float32
call_gex_above_2strike: float32
put_gex_below_2strike:  float32

# DCT blocks (optional; if present)
dct_d_atr_c0..c7:      float32
...
```

### 7.2 Schema: STREAM BAR (derived, emitted to UI)

```
STREAM_BAR_SCHEMA:

episode_id:   string
timestamp_ns:int64
bar_index:   int32

# Canonical streams (current)
Σ_M: float32
Σ_F: float32
Σ_B: float32
Σ_D: float32
Σ_S: float32

# Optional merged streams
Σ_P: float32
Σ_R: float32
A:   float32
D:   float32
A_adj: float32

# Smoothed values + derivatives (per stream you display)
Σ̄_P: float32
slope_P: float32
curvature_P: float32
jerk_P: float32

Σ̄_F: float32
slope_F: float32
...
```

### 7.3 Schema: TRAINING SAMPLE (supervised, anchored at bar t) ❌ **TODO**

**Status**: NOT IMPLEMENTED - Needed for projection model training (Section 6)

**How to Build**: Create `backend/scripts/build_projection_dataset.py` that:
1. Loads all historical stream bars from `gold/streams/pentaview/`
2. For each bar with sufficient history (L=20) and future (H=10):
   - Extract lookback: `sigma_p_hist`, `slope_p_hist`, all cross-streams
   - Extract targets: `sigma_p_target[1..10]` (future stream values)
   - Compute sample weight from `sigma_s`
3. Save to `gold/training/projection_samples/date=YYYY-MM-DD/*.parquet`

```
TRAINING_SAMPLE_SCHEMA:

sample_id:            string
episode_id:           string
timestamp_ns:         int64
bar_index:            int32

# Context
level_kind:           string
direction:            string
time_bucket:          string
attempt_index:        int32
level_stacking_5pt:   int32
spot:                 float32
atr:                  float32

# Lookback history (L=20)
Σ_P_hist:             float32[L]
Σ_M_hist:             float32[L]
Σ_F_hist:             float32[L]
Σ_B_hist:             float32[L]
Σ_D_hist:             float32[L]
Σ_S_hist:             float32[L]

slope_P_hist:         float32[L]
curvature_P_hist:     float32[L]

# Stream-specific raw history (optional)
velocity_5min_hist:   float32[L]
ofi_60s_hist:         float32[L]
barrier_delta_3min_hist: float32[L]
gamma_exposure_hist:  float32[L]
d_atr_hist:           float32[L]

# DCT features at anchor (optional)
dct_d_atr:            float32[8]
dct_ofi:              float32[8]
dct_barrier:          float32[8]
dct_tape:             float32[8]

# Known future (H=10)
minutes_since_open_fut: float32[H]
or_active_fut:          int32[H]

# Targets: future values (H=10)
Σ_P_target:           float32[H]   # Σ_P[t+1..t+H]
Σ_M_target:           float32[H]
Σ_F_target:           float32[H]
Σ_B_target:           float32[H]
Σ_R_target:           float32[H]

# Sample weights (optional)
setup_weight:         float32      # derived from Σ_S at anchor
```

### 7.4 Schema: INFERENCE REQUEST / RESPONSE (UI-ready) ❌ **TODO**

**Status**: NOT IMPLEMENTED - Real-time API schema for projection inference

**How to Build**: Create WebSocket message schemas in `backend/src/gateway/` and `frontend/src/app/models/`:
1. Request contains: current stream histories + context
2. Response contains: current streams + derivatives + projection curves (11 points × 3 quantiles)
3. Wire to existing WebSocket infrastructure in gateway

**Integration Point**: Add to `backend/src/core/main.py` real-time service - call projection model every 2-min bar close.

```
INFERENCE_REQUEST_SCHEMA:
request_id: string
episode_id: string
timestamp_ns: int64
bar_index: int32

# Context
level_kind: string
direction: string
time_bucket: string
attempt_index: int32
level_stacking_5pt: int32
spot: float32
atr: float32

# Lookback histories (L=20)
Σ_P_hist, Σ_M_hist, Σ_F_hist, Σ_B_hist, Σ_D_hist, Σ_S_hist: float32[L]
slope_P_hist, curvature_P_hist: float32[L]

# Optional raw histories + DCT blocks
...

# Known future (H=10)
minutes_since_open_fut: float32[H]
or_active_fut: int32[H]
```

```
INFERENCE_RESPONSE_SCHEMA:
request_id: string
episode_id: string
timestamp_ns: int64
bar_index: int32

# Current values (computed)
streams:
  momentum: {value: float32, smooth: float32, slope: float32, curvature: float32, jerk: float32}
  flow:     {value: float32, smooth: float32, slope: float32, curvature: float32, jerk: float32}
  barrier:  {value: float32, smooth: float32, slope: float32, curvature: float32, jerk: float32}
  dealer:   {value: float32, smooth: float32}
  setup:    {value: float32, smooth: float32}
  pressure: {value: float32, smooth: float32, slope: float32, curvature: float32, jerk: float32}
  structure:{value: float32, smooth: float32}

composites:
  alignment: float32
  divergence: float32
  alignment_adj: float32

# Projection curves: 11 points (t..t+H)
curves:
  pressure:
    x: int32[H+1]
    q50: float32[H+1]
    q10: float32[H+1]
    q90: float32[H+1]
  flow: { ... }
  barrier: { ... }

alerts: [ {type: string, severity: string, message: string} ]
```

---

## 8) Keep / update / discard from the submitted analyst spec ✅ **COMPLETE**

**Status**: Design guidance followed in implementation.
- Kept: Canonical streams, derivatives, quantile outputs, alignment/divergence
- Updated: DCT moved to Setup, Barrier sign fixed, Dealer as amplifier, added super-streams
- Discarded: Index-based references, hard-coded gamma interpretations

### Keep

- The **idea of canonical normalized streams** and showing their **derivatives**.
- Separating **directional** (momentum/flow/barrier) from **context/amplifier** (setup/dealer).
- Multi-horizon **quantile** outputs (q10/q50/q90) for UI confidence bands.
- Computing **alignment** and **divergence** as first-class diagnostics.

### Update

1. **Move `DCT(d_atr)` out of MOMENTUM**.
   - Distance-to-level shape is setup/context, not price momentum.

2. **Fix BARRIER sign semantics**.
   - Barrier must be market-signed via `dir_sign` (or explicit pipeline sign). Otherwise alignment math breaks.

3. **Treat DEALER as an amplifier regime**, not a directional stream.
   - Use `Σ_D` primarily as a multiplier on directional conviction, not an additive alignment term.

4. **Consolidate tape-shape features**.
   - `DCT(tape_imbalance)` belongs in FLOW (aggression trajectory), not SETUP.

5. **Add merged super-streams** (`Σ_P`, `Σ_R`) to support “one-glance TA”.
   - Keep the canonical streams, but give UI a compact “Pressure” line as the main signal.

### Discard / avoid

- Index-based references ("25-29") in production logic. Use **explicit column names**.
- Hard-coded interpretations of gamma sign unless anchored to your internal convention. Flip **once** in `Σ_D`.
- Mixing too many unrelated families into one stream (e.g., momentum + distance + shape) — it reduces interpretability.

---

## 9) Implementation pseudocode (core) ✅ **COMPLETE**

**Status**: All helper functions and stream computation implemented in `backend/src/ml/stream_builder.py`
- `wavg()` - Exponentially-weighted average
- `trend_score()` - DCT low-frequency energy (c1+c2)
- `chop_score()` - DCT high-frequency ratio
- `ema()` - Exponential moving average
- `compute_streams()` - Main stream computation function (Section 9.2)

### 9.1 Helpers

```
function wavg(values[], decay):
  # values ordered shortest→longest scale
  # weights: [1, decay, decay^2, ...]
  w = []
  for i in range(len(values)):
    w[i] = decay^i
  return sum(w[i]*values[i]) / sum(w)

function trend_score(DCT_c[]):
  # emphasize c1 trend, c2 curvature
  return tanh( 0.8*norm(DCT_c[1]) + 0.2*norm(DCT_c[2]) )

function chop_score(DCT_c[]):
  # high-frequency energy ratio
  low = sum(|c1|, |c2|)
  hi  = sum(|c3|..|c7|)
  return clamp( hi / (low + hi + eps), 0, 1 )

function ema(x_t, prev, halflife_bars):
  α = 1 - exp( ln(0.5) / halflife_bars )
  return α*x_t + (1-α)*prev
```

### 9.2 Compute streams at bar close

```
function compute_streams(bar_row, stats):

  # --- normalize primitives ---
  v = [norm(velocity_1min), norm(velocity_3min), ...]
  a = [norm(acceleration_1min), ...]
  j = [norm(jerk_1min), norm(jerk_3min), norm(jerk_5min)]
  mt = [norm(momentum_trend_3min), ...]

  # MOMENTUM
  Σ_M = tanh(0.40*wavg(v,0.7) + 0.30*wavg(a,0.7) + 0.15*wavg(j,0.8) + 0.15*wavg(mt,0.7))

  # FLOW
  shape = 0.5*trend_score(DCT_ofi) + 0.5*trend_score(DCT_tape)
  Σ_F = tanh(0.25*norm(ofi_60s) + 0.20*norm(ofi_near_level_60s) + 0.25*mean(norm(tape_imb_t0..t2))
            +0.15*norm(ofi_acceleration) + 0.10*norm(flow_alignment) + 0.05*shape)

  # BARRIER (dir_sign)
  dir_sign = +1 if direction=="UP" else -1
  consume = mean([-norm(barrier_delta_liq_log_t0), -norm(barrier_delta_liq_log_t1), -norm(barrier_delta_liq_log_t2)])
  repl    = norm(barrier_replenishment_ratio - 1.0)
  state   = clamp(barrier_state_encoded/2.0, -1, +1)
  s_local = tanh(0.50*consume + 0.25*(-norm(barrier_delta_3min)) + 0.15*(-state) + 0.10*(-repl))
  Σ_B = tanh(dir_sign * s_local)

  # DEALER (amplify/dampen)
  fuel = fuel_effect_encoded
  Σ_D = tanh(0.45*fuel + 0.25*(-norm(gamma_exposure)) + 0.15*(-norm(gex_ratio)) + 0.15*(-abs(norm(net_gex_2strike))))

  # SETUP / QUALITY
  proximity = exp(-abs(d_atr_t0)/1.5)
  recency   = exp(-time_since_last_touch_sec/900)
  freshness = 1.0 if attempt_index<=2 else 0.6 if attempt_index<=4 else 0.3
  confluence = clamp((level_stacking_5pt + level_stacking_10pt)/12.0,0,1)
  clean_trend = 0.5 + 0.5*trend_score(DCT_d_atr)
  chop_pen = chop_score(DCT_d_atr)
  q = clamp(0.28*proximity + 0.18*(0.5+0.5*tanh(norm(approach_velocity))) + 0.18*freshness + 0.16*confluence
           +0.10*recency + 0.10*clean_trend - 0.10*chop_pen, 0, 1)
  Σ_S = 2*q - 1

  # SUPER-STREAMS
  Σ_P = tanh(0.55*Σ_M + 0.45*Σ_F)
  Σ_R = tanh(0.70*Σ_B + 0.30*Σ_S)

  A_dir = (Σ_M + Σ_F + Σ_B)/3
  A = A_dir * (0.6 + 0.4*(0.5 + 0.5*Σ_S))
  D = std([Σ_M, Σ_F, Σ_B])
  A_adj = clamp(A * (1.0 + 0.35*Σ_D), -1, +1)

  return {Σ_M,Σ_F,Σ_B,Σ_D,Σ_S, Σ_P,Σ_R, A,D,A_adj}
```

---


## 10) TA-style state machine (optional but recommended) ✅ **COMPLETE**

**Status**: IMPLEMENTED (December 30, 2025) - Rule-based interpretation layer for discretionary trading

**Implementation**: `backend/src/ml/stream_state_machine.py` with functions:
- `detect_exhaustion_continuation()` - Buying/selling pressure patterns
- `detect_flow_divergence()` - Flow-momentum mismatch detection
- `detect_barrier_phase()` - Barrier support/opposition/weakening
- `detect_quality_gates()` - Setup quality and dealer regime gates
- `compute_exit_score()` - Position-aware hold/reduce/exit scoring
- `StreamStateMachine` - Hysteresis to prevent alert flickering

**Features**: 14 alert types, confidence scores, sustained alerts (>5s), LONG/SHORT position awareness.

These are **interpretation layers** for UI + discretionary execution. They are derived purely from streams + derivatives.

### 10.1 Exhaustion / continuation / reversal (using PRESSURE)

Let `P = Σ̄_P`, `P1 = slope_P`, `P2 = curvature_P`, `P3 = jerk_P`.

```
CONTINUATION_UP:
  P > +0.35 and P1 > +0.05

EXHAUSTION_UP ("buying slowing"):
  P > +0.35 and P1 < 0 for n_bars

REVERSAL_RISK_UP (early warning):
  P > +0.35 and P1 < 0 and P2 < 0 and abs(P3) > j_thresh

CONTINUATION_DOWN:
  P < -0.35 and P1 < -0.05

EXHAUSTION_DOWN:
  P < -0.35 and P1 > 0 for n_bars

REVERSAL_RISK_DOWN:
  P < -0.35 and P1 > 0 and P2 > 0 and abs(P3) > j_thresh
```

### 10.2 Flow–momentum divergence (reversal / squeeze / trap detector)

```
FLOW_DIVERGENCE:
  sign(Σ̄_F) != sign(Σ̄_M)
  and abs(Σ̄_F) > 0.30
  and abs(Σ̄_M) > 0.30

FLOW_CONFIRMATION:
  sign(Σ̄_F) == sign(Σ̄_M)
  and min(abs(Σ̄_F), abs(Σ̄_M)) > 0.35
```

Interpretation examples:

- `Σ̄_M > 0` but `Σ̄_F < 0` → price drifting up while sell aggression dominates → fragile / short-covering prone.
- `Σ̄_M < 0` but `Σ̄_F > 0` → price drifting down while buy aggression dominates → absorption / squeeze risk.

### 10.3 Barrier phase (break vs reject microstructure)

Use `Σ̄_B` (market-signed) and its slope.

```
BARRIER_BREAK_SUPPORT:
  sign(Σ̄_B) == sign(Σ̄_P)
  and abs(Σ̄_B) > 0.30

BARRIER_OPPOSES_PRESSURE:
  sign(Σ̄_B) != sign(Σ̄_P)
  and abs(Σ̄_B) > 0.30

BARRIER_WEAKENING:
  abs(Σ̄_B) > 0.30 and sign(slope_B) != sign(Σ̄_B)
```

### 10.4 Setup gating

When `Σ_S` is low, treat projections as low confidence and suppress aggressive alerts.

```
LOW_QUALITY:
  Σ_S < -0.25

HIGH_QUALITY:
  Σ_S > +0.25
```

### 10.5 Dealer gating

When `Σ_D` is high (fuel), increase urgency and tighten reversal thresholds.

```
FUEL_REGIME:
  Σ_D > +0.25

PIN_REGIME:
  Σ_D < -0.25
```


### 10.6 Position-aware hold/reduce/exit score (optional)

Define `pos_sign = +1` for LONG, `-1` for SHORT.

```
Apos  = pos_sign * A_adj
P1pos = pos_sign * slope_P        # slope of Pressure in position direction
F1pos = pos_sign * slope_F
B1pos = pos_sign * slope_B

E_exit = tanh(
  0.45*Apos
+ 0.25*P1pos
+ 0.15*F1pos
+ 0.15*B1pos
)

# Interpret:
#  E_exit > +0.50 -> HOLD / ADD
#  E_exit in [-0.20, +0.50] -> HOLD / TRAIL
#  E_exit in [-0.50, -0.20] -> REDUCE
#  E_exit < -0.50 -> EXIT

# Jerk booster (early warning):
# if pos_sign*jerk_P < -j_thresh: E_exit -= 0.15
```

---

## 11) Projection inference pseudocode (Quantile Polynomial Projection) ✅ **COMPLETE**

**Status**: IMPLEMENTED (December 30, 2025) - Part of Section 6 implementation

**Implementation**: Built into `backend/src/ml/stream_projector.py`:
- `build_polynomial_path()` - Generate 11-point smooth curve from polynomial coefficients
- `compute_projected_slope()` - Compute projected slope at horizon h
- `compute_projected_curvature()` - Compute projected curvature at horizon h
- `project_stream_curves()` - Main inference function with uncertainty bands

**Usage**: Real-time inference generates 20-minute projection curves (q10/q50/q90) from current bar.

### 11.1 Model outputs

For each projected stream `x` you support (`pressure`, optionally `flow`, `barrier`, `momentum`, `structure`):

```
model_out[x] = {
  q10: {a1: float, a2: float, a3: float},
  q50: {a1: float, a2: float, a3: float},
  q90: {a1: float, a2: float, a3: float}
}
```

### 11.2 Build the projected curve

```
function build_curve(Σ̄_t, coeffs, H):
  curve = array length H+1
  curve[0] = Σ̄_t

  for h in 1..H:
    y = Σ̄_t + coeffs.a1*h + 0.5*coeffs.a2*h*h + (1.0/6.0)*coeffs.a3*h*h*h
    curve[h] = clamp(y, -1, +1)

  return curve
```

### 11.3 Projected slope/curvature at horizon h (for UI tooltips)

From the polynomial:

```
slopê(h)     = d/dh ŷ(h) = a1 + a2*h + 0.5*a3*h^2
curvaturê(h) = d^2/dh^2 ŷ(h) = a2 + a3*h
```

### 11.4 Projected alignment (optional)

If you project `{Σ̂_M, Σ̂_F, Σ̂_B}` then:

```
Â_dir(h) = (Σ̂_M(h) + Σ̂_F(h) + Σ̂_B(h)) / 3
Â(h) = clamp( Â_dir(h) * (0.6 + 0.4*(0.5 + 0.5*Σ_S)), -1, +1 )
Â_adj(h) = clamp( Â(h) * (1.0 + 0.35*Σ_D), -1, +1 )
```

---

## 12) Minimal UI encoding rules (sign + color + slope) ❌ **TODO - LOW PRIORITY**

**Status**: NOT IMPLEMENTED - Frontend/Angular visualization rules

**What to Build**: Angular components in `frontend/src/app/` for:
- Color hue mapping: `sign(Σ̄_x)` → green/red families
- Intensity mapping: `0.3 + 0.7*abs(Σ̄_x)`
- Arrow glyph selection: `slope_x` → ▲/△/─/▽/▼
- Dealer/Setup color schemes (distinct from directional streams)

**Use Case**: Chart visualization - trader sees stream values, colors, and slope glyphs at a glance.

### 12.1 Directional streams (Pressure / Flow / Barrier)

- **Color hue** = sign of smoothed value `Σ̄_x`.
- **Intensity** = `0.3 + 0.7*abs(Σ̄_x)`.
- **Arrow glyph** = sign + magnitude of `slope_x`.

### 12.2 Non-directional streams (Dealer / Setup)

- Dealer: hue encodes `fuel vs pin` (amplify vs dampen).
- Setup: hue encodes quality (good vs degraded).

These should be visually distinct from buy/sell hues.

---

## APPENDIX: Quick Reference for Extension/Modification

### How to Add a New Stream

1. **Define the stream function** in `backend/src/ml/stream_builder.py`:
```python
def compute_my_stream(bar_row, stats, stratum=None) -> float:
    # Normalize raw features
    feat1 = normalize_feature('feature1', bar_row.get('feature1', 0.0), stats, stratum)
    feat2 = normalize_feature('feature2', bar_row.get('feature2', 0.0), stats, stratum)
    
    # Aggregate
    s_raw = 0.6 * feat1 + 0.4 * feat2
    
    # Squash to [-1, +1]
    return float(np.tanh(s_raw))
```

2. **Add to `compute_all_streams()`**:
```python
sigma_my = compute_my_stream(bar_row, stats, stratum)
return {
    'sigma_m': sigma_m,
    # ... existing streams ...
    'sigma_my': sigma_my  # Add here
}
```

3. **Update output schema** in `backend/src/pipeline/stages/compute_streams.py` if needed.

### How to Modify Stream Weights

Edit coefficients in `backend/src/ml/stream_builder.py`:
- **Momentum**: Currently `0.40*vel + 0.30*acc + 0.15*jerk + 0.15*trd`
- **Flow**: Currently `0.25*ofi + 0.20*ofi_level + 0.25*imb + 0.15*acc + 0.10*aln + 0.05*shape`
- **Pressure**: Currently `tanh(0.55*Σ_M + 0.45*Σ_F)`

**After changing weights**, recompute stream bars:
```bash
uv run python -m scripts.run_pentaview_pipeline --date 2024-12-16
```

### How to Add Features to Normalization

1. **Classify feature** in `backend/src/ml/stream_normalization.py`:
   - Add to `STREAM_ROBUST_FEATURES` (heavy-tailed) or
   - Add to `STREAM_ZSCORE_FEATURES` (symmetric) or
   - Add to `STREAM_PASSTHROUGH_FEATURES` (already normalized)

2. **Recompute normalization stats**:
```bash
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 60 --end-date 2024-12-31
```

### How to Debug Stream Values

1. **Check normalization stats**:
```bash
cat data/gold/streams/normalization/current.json | jq '.global_stats.ofi_60s'
```

2. **Inspect stream bars**:
```python
import pandas as pd
df = pd.read_parquet('data/gold/streams/pentaview/version=3.1.0/date=2024-12-16/stream_bars.parquet')
print(df[['timestamp', 'sigma_m', 'sigma_f', 'sigma_b']].describe())
```

3. **Check for missing features** (warnings in logs):
```
grep "No normalization stats for feature" backend/logs/*.log
```

### Critical Invariants to Maintain

1. **All streams must be bounded in [-1, +1]** - Use `tanh()` final squashing
2. **Barrier stream must apply dir_sign** - Ensures market semantics (positive=up, negative=down)
3. **Dealer is non-directional** - Use as multiplier, not additive alignment term
4. **Derivatives require smoothing** - Apply EMA first (halflife=3) to prevent noise amplification
5. **Normalization must be stratified** - Different regimes (time_bucket) need different stats

### Performance Notes

- **Stream computation**: ~0.11s for 437 bars (5 levels) = 3,972 bars/sec
- **Normalization**: 58 features × 5 strata = 290 feature-stratum pairs
- **Memory**: Stream bars are ~12 KB/day (437 rows × 32 cols × 4 bytes)

### Implementation Roadmap for Remaining TODOs

**Phase 1: Projection Model (Section 6, 11)** - CRITICAL
```bash
# Step 1: Build training dataset
uv run python -m scripts.build_projection_dataset \
  --start 2024-11-01 --end 2024-12-31 \
  --lookback 20 --horizon 10

# Step 2: Train projection models (one per stream)
uv run python -m scripts.train_projection_models \
  --stream pressure --epochs 100 --model-type lightgbm

# Step 3: Test inference
uv run python -c "
from src.ml.stream_projector import StreamProjector
proj = StreamProjector.load('data/models/projection_pressure_v1.joblib')
coeffs = proj.predict(current_bar_features)
curve = build_curve(sigma_p_current, coeffs['q50'], H=10)
print(curve)  # 11 points: current + 10 future
"
```

**Phase 2: State Machine (Section 10)** - MEDIUM
```bash
# Add rule engine
# File: backend/src/ml/stream_state_machine.py
# Usage in real-time service:
from src.ml.stream_state_machine import detect_alerts
alerts = detect_alerts(stream_bar, history_df)
# Returns: [{type: 'EXHAUSTION_UP', severity: 'WARNING', ...}]
```

**Phase 3: UI Integration (Section 12)** - LOW
```bash
# Angular components in frontend/src/app/
# File: stream-chart.component.ts
# Renders streams with color/intensity/glyphs per Section 12
```

### Testing Checklist

After modifications, verify:
- [ ] All stream values in [-1, +1]
- [ ] Barrier stream changes sign with direction (UP vs DOWN)
- [ ] Derivatives computed for sigma_p, sigma_m, sigma_f, sigma_b
- [ ] No NaN values in output
- [ ] Pipeline completes without errors
- [ ] Output file size reasonable (~10-15 KB/day)

### Verified Test Results (2025-12-16)

✅ **Validation Status**: ALL CHECKS PASSED
```
✓ All streams bounded in [-1, +1]
✓ No NaN values
✓ Derivatives present (slope, curvature, jerk)
✓ Barrier directional bias confirmed (UP mean=0.121, DOWN mean=-0.067)
```

✅ **Stream value ranges** (all properly bounded):
```
sigma_m (Momentum):  [-0.454, 0.397]
sigma_f (Flow):      [-0.477, 0.383]
sigma_b (Barrier):   [-0.335, 0.326]
sigma_d (Dealer):    [-0.378, 0.148]
sigma_s (Setup):     [-0.488, 0.417]
sigma_p (Pressure):  [-0.411, 0.266]
sigma_r (Structure): [-0.312, 0.328]
alignment:           [-0.262, 0.195]
divergence:          [0.021, 0.300]
alignment_adj:       [-0.274, 0.204]
```

✅ **Output**: 437 stream bars across 5 levels (OR_HIGH, OR_LOW, PM_LOW, SMA_90, EMA_20)
✅ **Performance**: 3,972 bars/sec (0.11s for 437 bars)
✅ **No NaN values** in output
✅ **File size**: ~12 KB/day (437 rows × 32 cols)

⚠️ **Known Issue**: `flow_alignment` and `barrier_delta_liq_log` features not present in current state table. These are **derived features** that need to be added to Stage 16 (materialize_state_table.py). Current implementation defaults to 0.0 when missing.

**To Fix**:
1. Add `flow_alignment = ofi_60s * (-sign(distance_signed_atr))` to state table computation
2. Add `barrier_delta_liq_log = sign(barrier_delta_liq) * log1p(|barrier_delta_liq|)` to state table
3. Rerun es_pipeline Stage 16, then recompute stream normalization

### Troubleshooting

**Problem**: Stream values outside [-1, +1]
- **Check**: Final `tanh()` squashing present in stream functions?
- **Check**: Normalization applied before aggregation?

**Problem**: Barrier stream doesn't flip sign with direction
- **Check**: `dir_sign = +1 if direction=='UP' else -1` applied in `compute_barrier_stream()`?
- **Verify**: Print barrier values for same level with UP vs DOWN approaches

**Problem**: Derivatives are noisy/spiky
- **Check**: EMA smoothing applied with `halflife=3` before computing differences?
- **Try**: Increase halflife to 5 for more smoothing

**Problem**: Missing features warning spam in logs
- **Check**: Are features present in state table? Run: `df.columns.tolist()`
- **Fix**: Add missing features to state table (Stage 16) or remove from stream computation

**Problem**: Pipeline fails with "No normalization stats"
- **Check**: `gold/streams/normalization/current.json` exists?
- **Fix**: Run `scripts/compute_stream_normalization.py` first

**Problem**: Empty stream bars output
- **Check**: State table exists for date? Check `silver/state/es_level_state/date=YYYY-MM-DD/`
- **Check**: State table has required columns? Run validation script

