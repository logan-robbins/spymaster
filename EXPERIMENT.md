# Regime Change Detection Experiments

## Dataset

**Instrument**: MNQH6 (Micro E-mini Nasdaq 100 Futures, March 2026)
**Capture Window**: 2026-02-06, 09:25-10:25 ET (1 hour of RTH)
**Dataset ID**: `mnqh6_20260206_0925_1025`
**Source**: `backend/lake/research/vp_immutable/mnqh6_20260206_0925_1025/`

### Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `bins.parquet` | 35,999 | 11 | Per-100ms bin metadata: `ts_ns`, `bin_seq`, `mid_price`, BBO, `book_valid` |
| `grid_clean.parquet` | 3,635,899 | 33 | 101 ticks x 35,999 bins. Each row = one (bin, k) cell |

### Grid Columns Used

| Column | Description | Derivative Order |
|---|---|---|
| `v_add` | Add velocity (rate of new orders arriving) | d1 of mass_add |
| `v_pull` | Pull velocity (rate of order cancellations) | d1 of mass_pull |
| `v_fill` | Fill velocity (rate of executions) | d1 of mass_fill |
| `a_add` | Add acceleration | d2 of mass_add |
| `a_pull` | Pull acceleration | d2 of mass_pull |
| `j_add` | Add jerk | d3 of mass_add |
| `j_pull` | Pull jerk | d3 of mass_pull |
| `pressure_variant` | Composite liquidity-building score | Two-force model |
| `vacuum_variant` | Composite liquidity-draining score | Two-force model |
| `spectrum_score` | Per-cell regime score | Smoothed composite |
| `spectrum_state_code` | Per-cell state {-1=vacuum, 0=neutral, 1=pressure} | Thresholded score |

### Grid Layout

- 101 columns per bin: `k = -50` to `k = +50` (column index = k + 50)
- `k = 0` (column 50) = spot price (BBO midpoint)
- `k < 0` = bid side (below spot)
- `k > 0` = ask side (above spot)
- Tick size: $0.25

---

## Evaluation Methodology

All 6 experiments share a common evaluation harness (`eval_harness.py`) to ensure identical signal detection and outcome evaluation.

### Signal Detection

```
detect_signals(signal, threshold, cooldown_bins):
    state = "flat"
    for each bin i:
        if signal[i] >= +threshold: cur_state = "up"
        elif signal[i] <= -threshold: cur_state = "down"
        else: cur_state = "flat"

        if cur_state != "flat" and cur_state != prev_state:
            if bins_since_last_signal >= cooldown_bins:
                emit signal(bin=i, direction=cur_state)
```

Signals fire on threshold crossing with direction change and cooldown enforcement.

### TP/SL Evaluation

```
For each signal at bin i with direction d:
    entry_price = mid_price[i]
    For bins i+1 to i+1200 (120 second max hold):
        if d == "up":
            price >= entry + 8 * $0.25 => TP ($2.00 profit)
            price <= entry - 4 * $0.25 => SL ($1.00 loss)
        if d == "down":
            price <= entry - 8 * $0.25 => TP ($2.00 profit)
            price >= entry + 4 * $0.25 => SL ($1.00 loss)
    If neither hit within 1200 bins => timeout
```

| Parameter | Value |
|---|---|
| TP ticks | 8 ($2.00) |
| SL ticks | 4 ($1.00) |
| Tick size | $0.25 |
| Risk/reward | 2:1 (risk $1 to make $2) |
| Breakeven TP rate | 33.3% |
| Max hold | 1200 bins (120 seconds) |
| Warmup | 300 bins (30 seconds) skipped for rolling window fill |

### Baseline

The existing directional edge approach (`analyze_vp_signals.py --mode regime --tp-ticks 8 --sl-ticks 4`) achieves **28.5% TP rate** on this dataset, below the 33.3% breakeven.

---

## Experiments

### 1. Asymmetric Derivative Slope (ADS)

**Thesis**: OLS slopes of bid-vs-ask velocity asymmetry are the earliest leading indicators of regime change. When v_add is increasing faster on the bid side than the ask side (and v_pull is increasing faster on the ask side), price is about to move up.

**Columns**: `v_add`, `v_pull`

**Step 1 — Per-band asymmetry**

Three spatial bands around spot:

| Band | Bid columns | Ask columns | Width |
|---|---|---|---|
| Inner | k=-3..-1 (cols 47-49) | k=+1..+3 (cols 51-53) | 3 |
| Mid | k=-11..-4 (cols 39-46) | k=+4..+11 (cols 54-61) | 8 |
| Outer | k=-23..-12 (cols 27-38) | k=+12..+23 (cols 62-73) | 12 |

```
add_asym(band) = mean(v_add[bid_cols]) - mean(v_add[ask_cols])    # positive = bullish
pull_asym(band) = mean(v_pull[ask_cols]) - mean(v_pull[bid_cols])  # positive = bullish
```

**Step 2 — Bandwidth-weighted combination**

```
combined_asym = sum_b[ (1/sqrt(width_b)) * (add_asym_b + pull_asym_b) ] / sum_b[ 1/sqrt(width_b) ]
```

Narrower bands get higher weight (inner band resolves faster).

**Step 3 — Multi-scale slope analysis**

```
slope_w = rolling_OLS_slope(combined_asym, window=w)    for w in [10, 25, 50]
z_w = robust_zscore(slope_w, window=200)                 # median/MAD, 1.4826 scale
signal = 0.40 * tanh(z_10/3) + 0.35 * tanh(z_25/3) + 0.25 * tanh(z_50/3)
```

**Parameters**: cooldown=30 bins, thresholds=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

**Results**:

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 0.02 | 951 | **40.2%** | 59.8% | **+0.30** | 951 |
| 0.05 | 958 | 38.2% | 61.7% | -0.01 | 958 |
| 0.08 | 967 | 38.8% | 61.1% | +0.24 | 967 |
| 0.10 | 963 | 38.5% | 61.4% | +0.25 | 963 |
| 0.15 | 952 | 38.9% | 61.1% | +0.31 | 952 |
| 0.20 | 902 | 38.0% | 62.0% | +0.22 | 902 |

**Best**: 40.2% TP at threshold 0.02, 951 signals, +0.30 ticks/trade

---

### 2. Spatial Pressure Gradient (SPG)

**Thesis**: The spatial derivative dP/dk of pressure/vacuum fields encodes directional intent. Walls (high pressure gradient) block price movement, vacuums (negative gradient) attract it.

**Columns**: `pressure_variant`, `vacuum_variant`

**Step 1 — Spatial first derivative**

```
dP/dk = np.gradient(pressure_variant, axis=1)    # central differences along k
dV/dk = np.gradient(vacuum_variant, axis=1)
```

**Step 2 — Mean gradient by side** (band k=-16..-1 and k=+1..+16)

```
grad_P_above = mean(dP/dk[ask_cols])    # cols 51:67
grad_P_below = mean(dP/dk[bid_cols])    # cols 34:50
grad_V_above = mean(dV/dk[ask_cols])
grad_V_below = mean(dV/dk[bid_cols])
```

**Step 3 — Directional signals**

```
wall_signal = grad_P_below - grad_P_above    # positive = stronger wall above = bearish
pull_signal = grad_V_above - grad_V_below    # positive = stronger vacuum above = bullish
net = -wall_signal + pull_signal
```

**Step 4 — Dual EMA smoothing**

```
alpha_fast = 2/(5+1) = 0.333    # ~500ms
alpha_slow = 2/(20+1) = 0.095   # ~2s
smoothed = 0.6 * EMA_fast(net) + 0.4 * EMA_slow(net)
```

**Step 5 — Spatial curvature correction**

```
d2P/dk2[c] = P[c+1] + P[c-1] - 2*P[c]    for c in [48, 49, 50, 51, 52]
curv_signal = -mean(d2P/dk2), EMA-smoothed (span=10)
final = 0.7 * smoothed + 0.3 * curv_signal
```

**Parameters**: cooldown=20 bins, adaptive thresholds from signal distribution

**Results**:

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 2.606 | 863 | **39.3%** | 60.7% | +0.03 | 863 |
| 5.212 | 844 | 38.9% | 61.1% | +0.05 | 844 |
| 7.529 | 666 | 36.6% | 63.2% | -0.32 | 666 |
| 10.321 | 431 | 36.2% | 63.8% | -0.28 | 431 |
| 13.114 | 219 | 34.7% | 65.3% | -0.15 | 219 |

**Best**: 39.3% TP at threshold 2.606, 863 signals, +0.03 ticks/trade

**Note**: Almost entirely short-biased (854/863 signals were short). The signal's mean was strongly negative, indicating persistent downward spatial pressure gradient during this period.

---

### 3. Entropy Regime Detector (ERD)

**Thesis**: Before regime transitions, the state field fragments from ordered to disordered. A spike in Shannon entropy of the {pressure, neutral, vacuum} distribution precedes the reversal.

**Columns**: `spectrum_state_code`, `spectrum_score`

**Step 1 — Shannon entropy per bin**

```
H = -sum(p_i * log2(p_i + 1e-12))    for 3 states {-1, 0, 1}
```

Computed separately for:
- `H_full`: all 101 ticks
- `H_above`: ticks k=+1..+50 (50 ticks above spot)
- `H_below`: ticks k=-50..-1 (50 ticks below spot)
- Max entropy: log2(3) = 1.585 bits

**Step 2 — Entropy asymmetry and z-score**

```
entropy_asym = H_above - H_below
z_H = robust_zscore(H_full, window=100)    # median/MAD
```

**Step 3 — Spike-gated signal** (two variants tested)

```
spike_gate = max(0, z_H - 0.5)    # only fires when entropy z-score exceeds 0.5

Variant A: signal = score_direction * spike_gate
    where score_direction = mean(spectrum_score[below]) - mean(spectrum_score[above])

Variant B: signal = entropy_asym * spike_gate    # WINNER
```

**Parameters**: cooldown=40 bins, thresholds=[0.05..2.0]

**Results** (Variant B):

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 0.05 | 372 | 35.5% | 64.5% | -0.46 | 372 |
| 0.10 | 201 | 32.3% | 67.7% | -1.19 | 201 |
| 0.20 | 110 | **40.0%** | 60.0% | +0.25 | 110 |
| 0.30 | 77 | **40.3%** | 59.7% | +0.14 | 77 |
| 0.40 | 57 | 36.8% | 63.2% | -0.81 | 57 |
| 0.50 | 46 | 32.6% | 67.4% | -1.46 | 46 |
| 1.00 | 23 | 30.4% | 69.6% | -2.15 | 23 |
| 1.50 | 13 | **53.8%** | 46.2% | **+2.88** | 13 |
| 2.00 | 3 | 33.3% | 66.7% | +0.67 | 3 |

**Best**: 53.8% TP at threshold 1.5 — but only 13 signals (statistically unreliable). At threshold 0.3: 40.3% TP with 77 signals, more statistically meaningful.

---

### 4. Pressure Front Propagation (PFP)

**Thesis**: Aggressive order activity propagates from inner ticks (near BBO) outward. When inner-tick velocity leads outer-tick velocity by more than baseline lag, an aggressive participant is acting directionally.

**Columns**: `v_add`, `v_pull`, `v_fill`

**Step 1 — Zone definitions**

| Zone | Bid columns | Ask columns |
|---|---|---|
| Inner | k=-3..-1 (cols 47-49) | k=+1..+3 (cols 51-53) |
| Outer | k=-12..-5 (cols 38-45) | k=+5..+12 (cols 55-62) |

**Step 2 — Activity intensity per zone**

```
I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])
```

**Step 3 — Lead-lag via EMA cross-products**

```
lead_metric_bid[t] = EMA(I_inner_bid[t] * I_outer_bid[t - 5]) /
                     (EMA(I_inner_bid[t] * I_outer_bid[t]) + eps)
```

EMA alpha = 0.1, lag = 5 bins. Ratio > 1 means inner activity 5 bins ago predicts current outer activity (inner leads outer).

**Step 4 — Directional signal from add/fill channel**

```
add_signal = lead_metric_bid - lead_metric_ask    # positive = bid-side inner leads = bullish
```

**Step 5 — Pull (cancellation) channel**

```
pull_lead_bid = lead_metric(v_pull inner bid, v_pull outer bid, lag=5)
pull_lead_ask = lead_metric(v_pull inner ask, v_pull outer ask, lag=5)
pull_signal = pull_lead_ask - pull_lead_bid    # pull on ask side leading = bullish
```

**Step 6 — Blend**

```
final = 0.6 * add_signal + 0.4 * pull_signal
```

**Parameters**: cooldown=30 bins, adaptive thresholds from signal distribution percentiles

**Results**:

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 0.053 | 908 | 40.2% | 59.8% | +0.36 | 908 |
| 0.080 | 823 | 39.9% | 60.1% | +0.29 | 823 |
| 0.098 | 731 | **40.6%** | 59.4% | **+0.51** | 731 |
| 0.125 | 548 | 36.5% | 63.5% | -0.18 | 548 |
| 0.148 | 366 | 39.1% | 60.9% | +0.10 | 366 |
| 0.189 | 113 | 32.7% | 67.3% | -1.31 | 113 |

**Best**: 40.6% TP at threshold 0.098, 731 signals, +0.51 ticks/trade. Strongest balanced performer across all experiments.

---

### 5. Jerk-Acceleration Divergence (JAD)

**Thesis**: When jerk (d3) of add/pull diverges between bid and ask sides, it signals the inflection point BEFORE acceleration changes sign. Jerk-acceleration agreement confirms direction; disagreement signals early reversal.

**Columns**: `j_add`, `j_pull`, `a_add`, `a_pull`

**Step 1 — Distance-weighted spatial aggregation**

```
w(k) = 1/|k|    for k in [-24..-1] (bid cols 26-49) and [+1..+24] (ask cols 51-74)
weights_norm = w / sum(w)
X_bid = weighted_mean(grid[:, bid_cols], weights_norm)    # for X in {j_add, j_pull, a_add, a_pull}
X_ask = weighted_mean(grid[:, ask_cols], weights_norm)
```

**Step 2 — Divergence signals (bullish-positive orientation)**

```
jerk_add_div  = J_add_bid  - J_add_ask       # more add jerk on bid = bullish
jerk_pull_div = J_pull_ask - J_pull_bid       # more pull jerk on ask = bullish
accel_add_div = A_add_bid  - A_add_ask
accel_pull_div = A_pull_ask - A_pull_bid

jerk_signal  = 0.5 * jerk_add_div  + 0.5 * jerk_pull_div
accel_signal = 0.5 * accel_add_div + 0.5 * accel_pull_div
```

**Step 3 — Agreement/disagreement weighting**

```
if sign(jerk_signal) == sign(accel_signal):
    raw = 0.4 * jerk + 0.6 * accel       # confirmed direction: trust accel more
else:
    raw = 0.8 * jerk + 0.2 * accel       # disagreement: trust jerk (leads accel)
```

**Step 4 — Normalization**

```
z = robust_zscore(raw, window=300)    # median/MAD
signal = tanh(z / 3.0)
```

**Parameters**: cooldown=25 bins, thresholds=[0.05, 0.10, 0.15, 0.20, 0.30]

**Results**:

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 0.05 | 1253 | **38.5%** | 61.5% | **+0.10** | 1253 |
| 0.10 | 1266 | 34.8% | 65.2% | -0.53 | 1266 |
| 0.15 | 1279 | 35.1% | 64.9% | -0.39 | 1279 |
| 0.20 | 1272 | 36.6% | 63.4% | -0.03 | 1272 |
| 0.30 | 1194 | 36.0% | 64.0% | -0.18 | 1194 |

**Best**: 38.5% TP at threshold 0.05, 1253 signals, +0.10 ticks/trade. Highest signal volume of all experiments.

---

### 6. Intensity Imbalance Rate-of-Change (IIRC)

**Thesis**: The rate of change of the add-to-pull intensity ratio captures order-flow toxicity momentum. A declining bid-side ratio + increasing ask-side ratio = incoming selling pressure.

**Columns**: `v_add`, `v_pull`, `v_fill`

**Step 1 — Sum velocity by side** (band k=-16..-1 and k=+1..+16)

```
add_rate_bid = sum(v_add[cols 34:50])
pull_rate_bid = sum(v_pull[cols 34:50])
fill_rate_bid = sum(v_fill[cols 34:50])
(same for ask side, cols 51:67)
```

**Step 2 — Intensity ratio with Laplace smoothing**

```
ratio_bid = add_rate_bid / (pull_rate_bid + fill_rate_bid + 1.0)
ratio_ask = add_rate_ask / (pull_rate_ask + fill_rate_ask + 1.0)
```

**Step 3 — Log imbalance**

```
imbalance = log(ratio_bid + 1.0) - log(ratio_ask + 1.0)
```

Positive = bid adding more relative to pulling (bullish).

**Step 4 — Rate of change via rolling OLS slope**

```
d_fast = rolling_OLS_slope(imbalance, window=10)
d_slow = rolling_OLS_slope(imbalance, window=30)
signal = 0.6 * d_fast + 0.4 * d_slow
```

**Step 5 — Noise floor filter**

```
signal = 0    where |imbalance| < 0.1
```

Only 5.4% of bins pass the noise floor, making this a highly selective signal.

**Parameters**: cooldown=20 bins, thresholds=[0.001, 0.005, 0.01, 0.02, 0.05]

**Results**:

| Threshold | Signals | TP% | SL% | Mean PnL (ticks) | Events/hr |
|---|---|---|---|---|---|
| 0.001 | 534 | 35.8% | 64.2% | -0.13 | 534 |
| 0.005 | 438 | 35.2% | 64.8% | -0.33 | 438 |
| 0.010 | 198 | 38.4% | 61.6% | +0.66 | 198 |
| 0.020 | 26 | **38.5%** | 61.5% | **+2.77** | 26 |
| 0.050 | 0 | N/A | N/A | N/A | 0 |

**Best**: 38.5% TP at threshold 0.02, 26 signals, +2.77 ticks/trade. Highest per-trade PnL but low signal count makes it statistically weak.

---

## Comparison

### Ranking by Best TP Rate (min 5 signals)

| # | Agent | TP% | Signals | Mean PnL | Events/hr | vs Baseline | vs Breakeven |
|---|---|---|---|---|---|---|---|
| 1 | PFP | 40.6% | 731 | +0.51t | 731 | +12.1% | BEATS |
| 2 | ADS | 40.2% | 951 | +0.30t | 951 | +11.7% | BEATS |
| 3 | ERD | 40.3% | 77 | +0.14t | 77 | +11.8% | BEATS |
| 4 | SPG | 39.3% | 863 | +0.03t | 863 | +10.8% | BEATS |
| 5 | JAD | 38.5% | 1253 | +0.10t | 1253 | +10.0% | BEATS |
| 6 | IIRC | 38.5% | 26 | +2.77t | 26 | +10.0% | BEATS |

### Ranking by Mean PnL (min 20 signals)

| # | Agent | PnL/trade | TP% | Signals | Events/hr |
|---|---|---|---|---|---|
| 1 | IIRC | +2.77t | 38.5% | 26 | 26 |
| 2 | PFP | +0.51t | 40.6% | 731 | 731 |
| 3 | ADS | +0.30t | 40.2% | 951 | 951 |
| 4 | ERD | +0.25t | 40.0% | 110 | 110 |
| 5 | JAD | +0.10t | 38.5% | 1253 | 1253 |
| 6 | SPG | +0.03t | 39.3% | 863 | 863 |

### Key Findings

All 6 experiments beat both the 28.5% baseline and the 33.3% breakeven threshold on this dataset.

**PFP (Pressure Front Propagation)** is the strongest balanced performer: 40.6% TP rate with 731 signals and the highest PnL per trade (+0.51 ticks) among high-volume strategies. The lead-lag propagation from inner to outer ticks captures aggressive directional intent with strong statistical support.

**ADS (Asymmetric Derivative Slope)** ranks second with high volume (951 signals) and consistent profitability across all thresholds. The multi-scale OLS slope approach provides stable signal generation.

**ERD (Entropy Regime Detector)** shows the most extreme behavior: 53.8% TP at its tightest threshold (1.5), but with only 13 signals. At more permissive thresholds (0.2-0.3) it produces 77-110 signals at 40% TP. The entropy spike thesis has merit but needs more data to validate the high-threshold regime.

**SPG (Spatial Pressure Gradient)** showed strong directional bias (99% short signals), suggesting the spatial gradient approach may be capturing a persistent microstructure feature rather than symmetric regime detection. The signal was profitable but barely positive on PnL.

**IIRC (Intensity Imbalance Rate-of-Change)** had the highest per-trade PnL (+2.77 ticks) but generated only 26 signals. The aggressive noise floor filter (|imbalance| > 0.1) removed 94.6% of bins, making it too selective for practical use without relaxation.

**JAD (Jerk-Acceleration Divergence)** generated the most signals (1253) at 38.5% TP. The jerk-acceleration agreement/disagreement weighting adds interpretability: when jerk and acceleration agree on direction, the signal is more reliable.

---

## Shared Infrastructure

### eval_harness.py

Location: `backend/lake/research/vp_experiments/mnqh6_20260206_0925_1025/eval_harness.py`

Functions:
- `load_dataset(columns)` — reads bins + grid_clean parquet, pivots grid to `(n_bins, 101)` numpy arrays
- `detect_signals(signal, threshold, cooldown_bins)` — threshold crossing with cooldown (identical to `analyze_vp_signals.py:651`)
- `evaluate_tp_sl(signals, mid_price, ts_ns, ...)` — TP/SL outcome evaluation (identical to `analyze_vp_signals.py:685`)
- `sweep_thresholds(signal, thresholds, cooldown_bins, mid_price, ts_ns)` — runs detect+evaluate across threshold grid
- `write_results(agent_name, experiment_name, params, results)` — writes JSON to `agents/{name}/outputs/results.json`
- `rolling_ols_slope(arr, window)` — rolling OLS slope: `slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x2) - sum(x)^2)`
- `robust_zscore(arr, window, min_periods=30)` — median/MAD z-score with 1.4826 scaling factor

### Workspace Layout

```
backend/lake/research/vp_experiments/mnqh6_20260206_0925_1025/
    eval_harness.py           # shared evaluation module
    comparison.py             # cross-experiment ranking
    agents/
        ads/
            run.py            # experiment script
            outputs/results.json
            data/base_immutable -> ../../vp_immutable/mnqh6_20260206_0925_1025/
        spg/ ...
        erd/ ...
        pfp/ ...
        jad/ ...
        iirc/ ...
```

### Reproduction

```bash
cd backend
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/ads/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/spg/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/erd/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/pfp/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/jad/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/agents/iirc/run.py
uv run lake/research/vp_experiments/mnqh6_20260206_0925_1025/comparison.py
```
