# Experiment-Driven Projection Bands

## Summary

Render directional pressure bands in the frontend projection zone (15% right side of heatmap) driven by the top 3 experiment signals (PFP, ADS, ERD) computed in-browser from Arrow grid data already arriving at 10Hz. Purple color family, Gaussian intensity falloff, confidence decay across 4 horizons (250ms, 500ms, 1s, 2.5s). No backend changes.

## Decision Record

- **Approach**: Pure frontend computation (Approach A). All data needed is already in Arrow payload.
- **Horizons**: All 4 (250ms, 500ms, 1s, 2.5s) as discrete columns in the projection zone.
- **Color**: Purple ramp (bright purple = strong prediction, dark = no prediction). Distinct from green pressure ramp.
- **Confidence decay**: Progressive fade across horizons (h250=1.0, h500=0.8, h1000=0.6, h2500=0.4).
- **Experiments**: Top 3 by TP% x signal volume: PFP (40.6%), ADS (40.2%), ERD (40.3%).
- **Taylor projections**: Retained in Arrow deserialization but not rendered. Experiments replace them visually.
- **Band style**: Bands only, no center line. Directional skew of the envelope tells the story.

## Signal Computation

### PFP (Pressure Front Propagation) — weight 0.40

Inputs: `v_add`, `v_fill`, `v_pull` from 4 spatial zones.

Zones:
- Inner bid: k=-3..-1 (cols 47-49), Inner ask: k=+1..+3 (cols 51-53)
- Outer bid: k=-12..-5 (cols 38-45), Outer ask: k=+5..+12 (cols 55-62)

Per-zone intensity: `I = mean(v_add[zone] + v_fill[zone])`

Lead-lag via EMA cross-products (alpha=0.1, lag=5 bins):
```
lead_metric_bid = EMA(I_inner_bid[t] * I_outer_bid[t-5]) / (EMA(I_inner_bid[t] * I_outer_bid[t]) + eps)
```

Directional signals:
```
add_signal = lead_metric_bid - lead_metric_ask
pull_signal = pull_lead_ask - pull_lead_bid
final = 0.6 * add_signal + 0.4 * pull_signal
```

State: 5-bin lag buffer + 4 EMA accumulators. Warmup: 5 bins (500ms).

### ADS (Asymmetric Derivative Slope) — weight 0.35

Inputs: `v_add`, `v_pull` from 3 spatial bands.

Bands:
- Inner: k=-3..-1 / k=+1..+3 (width 3)
- Mid: k=-11..-4 / k=+4..+11 (width 8)
- Outer: k=-23..-12 / k=+12..+23 (width 12)

Per-band asymmetry:
```
add_asym = mean(v_add[bid]) - mean(v_add[ask])
pull_asym = mean(v_pull[ask]) - mean(v_pull[bid])
```

Bandwidth-weighted: `combined = sum(1/sqrt(w) * (add_asym + pull_asym)) / sum(1/sqrt(w))`

Multi-scale slope:
```
slope_w = incremental_OLS_slope(combined, window=w) for w in [10, 25, 50]
z_w = robust_zscore(slope_w, window=200)
signal = 0.40 * tanh(z_10/3) + 0.35 * tanh(z_25/3) + 0.25 * tanh(z_50/3)
```

State: 200-bin ring buffer + 3 incremental OLS accumulators. Warmup: 200 bins (20s).

### ERD (Entropy Regime Detector) — weight 0.25

Inputs: `spectrum_state_code`, `spectrum_score` across all 101 ticks.

Shannon entropy: `H = -sum(p_i * log2(p_i + 1e-12))` for 3 states {-1, 0, 1}

Computed for H_full (101 ticks), H_above (k=+1..+50), H_below (k=-50..-1).

Signal (Variant B):
```
entropy_asym = H_above - H_below
z_H = robust_zscore(H_full, window=100)
spike_gate = max(0, z_H - 0.5)
signal = entropy_asym * spike_gate
```

State: 100-bin ring buffer. Warmup: 100 bins (10s).

### Blending

```
composite = 0.40 * pfp + 0.35 * ads + 0.25 * erd
```

Dynamic warmup: only warm signals contribute. Weights renormalize to sum to 1.0 among warm signals. PFP comes online first (500ms), then ERD (10s), then ADS (20s).

## Band Geometry

At each of the 4 horizon positions in the projection zone:

1. Band center offset from spot row: `center = spot_row + composite * spread_ticks[horizon]`
   - spread_ticks: h250=2, h500=4, h1000=6, h2500=8
2. Band half-width: 6 ticks (fixed)
3. Per-row intensity: Gaussian falloff from center, `exp(-d^2 / (2 * sigma^2))`, sigma=2.5 ticks
4. Confidence alpha: h250=1.0, h500=0.8, h1000=0.6, h2500=0.4
5. Color: `R = 0.35*I*255, G = 0.08*I*255, B = 0.55*I*255` (purple/violet)
6. No center line — band shape encodes direction

## Rendering Architecture

Projection bands are NOT part of the scrolling heatmap buffer. They repaint from scratch every frame.

Dedicated buffer: `Uint8ClampedArray(4 * HMAP_LEVELS * 4)` — 4 horizon columns x grid height x RGBA.

Render pipeline (in `renderHeatmap()`):
1. Clear projection buffer to dark background
2. For each horizon column: compute band center, Gaussian intensity per row, apply confidence alpha, write purple RGBA
3. Blit via offscreen canvas (4px x HMAP_LEVELS) scaled to projection zone area
4. Extend spot line as horizontal dashed cyan into projection zone
5. Render horizon labels ("250ms", "500ms", "1s", "2.5s") on time axis

Frame budget: ~324 exp() calls = ~1.6us on M4. Invisible.

## State Management

Single `ExperimentEngine` class encapsulating all rolling state:

```
PFP: Float64Array(5*4) lag buffer + Float64Array(4) EMA numer/denom + cursor
ADS: Float64Array(200) combined_asym ring + Float64Array(50*3) OLS rings + cursors
ERD: Float64Array(100) entropy ring + cursor
Total: ~3KB
```

Incremental OLS: maintain running sums (sum_y, sum_xy) in ring buffers. X-values fixed (0..w-1), so sum_x and sum_x2 are precomputed constants. O(1) per bin.

Robust z-score: sorted insertion array for rolling window. Binary search + splice for insert/remove. Median = middle element. MAD = median(|x_i - median|) * 1.4826. O(w) per bin, w=200 max.

## Integration Points

All changes in `frontend/src/vacuum-pressure.ts` and `frontend/vacuum-pressure.html`.

New code (~250-350 lines):
1. `ExperimentEngine` class — computation engine with `update(grid)` and `reset()`
2. `renderProjectionBands()` — band painting in projection zone
3. `renderProjectionTimeLabels()` — horizon labels on time axis

Modified code (~30 lines):
4. `pushHeatmapColumnFromGrid()` — call `experimentEngine.update(currentGrid)`
5. `renderHeatmap()` — call `renderProjectionBands()` (replaces blank tint)
6. `renderTimeAxis()` — call `renderProjectionTimeLabels()`
7. `resetStreamState()` — call `experimentEngine.reset()`
8. `applyRuntimeConfig()` — instantiate `ExperimentEngine`

## Compute Budget

| Experiment | FLOPs/bin | History depth | Memory |
|-----------|-----------|---------------|--------|
| PFP | ~100 | 5 bins (500ms) | ~160B |
| ADS | ~900 | 200 bins (20s) | ~2.4KB |
| ERD | ~500 | 100 bins (10s) | ~800B |
| Total | ~1,500 | 200 bins max | ~3.4KB |

Render: ~324 exp() = ~1.6us. Total per-frame: <0.1ms. Inference rate: 10Hz (every 100ms bin).
