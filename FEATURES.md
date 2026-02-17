# Features and Formulas

This file defines the runtime features and exact formulas currently used by the canonical vacuum-pressure pipeline.

## 1. Runtime Geometry and Time Semantics

Definitions:
- `PRICE_SCALE = 1e-9`
- `cell_width_ms` is fixed bin width in milliseconds (default `100` ms).
- `grid_radius_ticks` is serve-time visible half-window around spot (default `50` ticks, so `101` rows).
- `n_absolute_ticks` is full internal engine grid size (default `8192`).

Time boundaries:
- `emit_after_ns = ET(dt + start_time).to_utc_ns`
- `warmup_start_ns = emit_after_ns - 0.5 hours` for futures/equities in current config.

Bin semantics:
- Bin duration: `cell_width_ns = cell_width_ms * 1_000_000`
- Each emitted bin is half-open in event time: `[bin_start_ns, bin_end_ns)`.
- Event routing behavior:
  - Events with `ts_ns < warmup_start_ns`: skipped.
  - Events with `warmup_start_ns <= ts_ns < emit_after_ns`: processed by engine, not emitted.
  - For streaming bins, when `ts_ns >= bin_end_ns`, previous bin is emitted, then bin advances by `cell_width_ns` until `ts_ns < bin_end_ns`.

Wall-time streaming pace:
- Async stream enforces `await sleep(cell_width_ms / 1000.0)` between emitted bins after first bin.

## 2. Price/Tick Mapping

Tick integer size:
- Futures: `tick_int = round(tick_size / PRICE_SCALE)`
- Equities: `tick_int = round(bucket_size_dollars / PRICE_SCALE)`

Price conversion:
- `price_dollars = price_int * PRICE_SCALE`

Absolute price to engine row index:
- `tick_abs = round(price_int / tick_int)`
- `idx = tick_abs - anchor_tick_idx + floor(n_ticks / 2)`
- Valid iff `0 <= idx < n_ticks`, otherwise out-of-range.

Inverse mapping:
- `tick_abs = anchor_tick_idx - floor(n_ticks / 2) + idx`
- `price_int = tick_abs * tick_int`

Spot reference:
- If both BBO sides valid:
  - `raw_mid = (best_bid + best_ask) / 2`
  - `spot_ref_price_int = round_to_tick(raw_mid)` using `floor(raw_mid / tick_int + 0.5) * tick_int`
  - `mid_price_dollars = (best_bid + best_ask) * 0.5 * PRICE_SCALE`
- Else: `spot_ref_price_int = 0`, `mid_price_dollars = 0.0`

## 3. Engine Mechanics and Derivative Chain

Per touched tick, current rest depth:
- `rest_depth = depth_bid + depth_ask`

Mass decay/update (`tau_rest_decay = 30s`):
- If `dt_s > 0`:
  - `decay = exp(-dt_s / tau_rest_decay)`
  - `add_mass = add_mass * decay + add_delta`
  - `pull_mass = pull_mass * decay + pull_delta`
  - `fill_mass = fill_mass * decay + fill_delta`
- Else (`dt_s <= 0`): add deltas directly.

Continuous-time EMA alpha:
- `alpha(dt, tau) = 1 - exp(-dt / tau)` (clamped to `1.0` when `dt/tau > 50`).

Derivative chain from value change (`v/a/j` for `rest_depth`):
- `rate = (new_value - prev_value) / dt`
- `v_new = alpha_v * rate + (1 - alpha_v) * v_prev`
- `dv_rate = (v_new - v_prev) / dt`
- `a_new = alpha_a * dv_rate + (1 - alpha_a) * a_prev`
- `da_rate = (a_new - a_prev) / dt`
- `j_new = alpha_j * da_rate + (1 - alpha_j) * j_prev`

Derivative chain from event delta (`v/a/j` for add/pull/fill):
- `rate = delta / dt`
- Same `v_new`, `a_new`, `j_new` chain as above.

Time constants:
- `tau_v = 2s`
- `tau_a = 5s`
- `tau_j = 10s`

## 4. Two-Force Model

Coefficients:
- `c1 = 1.0` (`v_add`)
- `c2 = 0.5` (`max(v_rest_depth, 0)`)
- `c3 = 0.3` (`max(a_add, 0)`)
- `c4 = 1.0` (`v_pull`)
- `c5 = 1.5` (`v_fill`)
- `c6 = 0.5` (`max(-v_rest_depth, 0)`)
- `c7 = 0.3` (`max(a_pull, 0)`)

Per tick:
- `pressure_variant = c1*v_add + c2*max(v_rest_depth, 0) + c3*max(a_add, 0)`
- `vacuum_variant = c4*v_pull + c5*v_fill + c6*max(-v_rest_depth, 0) + c7*max(a_pull, 0)`

Interpretation:
- `pressure_variant`: liquidity building/replenishing.
- `vacuum_variant`: liquidity draining/consumed.

## 5. Spectrum Kernel (Per Cell, Independent)

Inputs each bin:
- `pressure[i]`, `vacuum[i]` for each absolute tick cell `i`.

Composite signal:
- `composite[i] = (pressure[i] - vacuum[i]) / (abs(pressure[i]) + abs(vacuum[i]) + 1e-12)`

Window rollup:
- Let windows be `W = {w1, w2, ...}` with normalized weights `rw_j`.
- For each window `wj`, maintain trailing rolling sum over at most `wj` latest composite values.
- `mean_j[i] = rolling_sum_j[i] / min(composite_count, wj)`
- `rolled[i] = sum_j rw_j * mean_j[i]`

Time-step for derivatives:
- `dt_s = (ts_ns - prev_ts_ns) / 1e9` if positive; otherwise `default_dt_s = cell_width_ms / 1000`.

Derivatives:
- `d1 = (rolled - prev_rolled) / dt_s` (or zeros initially)
- `d2 = (d1 - prev_d1) / dt_s` (or zeros initially)
- `d3 = (d2 - prev_d2) / dt_s` (or zeros initially)

Robust z-score per derivative stream:
- If `count < zscore_min_periods`: `z = 0`
- Else over trailing `zscore_window_bins` history:
  - `med = median(hist)`
  - `mad = median(abs(hist - med))`
  - `scale = 1.4826 * mad`
  - For `scale > 1e-9`: `z = (x - med) / scale`, else `0`

Final score (derivative weights normalized to sum 1):
- `score = w1*tanh(z1 / tanh_scale) + w2*tanh(z2 / tanh_scale) + w3*tanh(z3 / tanh_scale)`
- Clamp and sanitize:
  - `score = clip(score, -1, 1)`
  - `score = nan_to_num(score, nan=0, posinf=1, neginf=-1)`

State code:
- `+1` if `score >= neutral_threshold`
- `-1` if `score <= -neutral_threshold`
- `0` otherwise

Forward projection per horizon `h_ms`:
- `h = h_ms / 1000`
- `score_d1 = (score - prev_score) / dt_s` (or zeros initially)
- `score_d2 = (score_d1 - prev_score_d1) / dt_s` (or zeros initially)
- `proj_score_h = clip(score + score_d1*h + 0.5*score_d2*h^2, -1, 1)`
- Then `nan_to_num` with same bounds.

## 6. Serve-Time Grid Extraction

Internal computation is full-grid (`n_absolute_ticks`), then serve-time slices visible window around spot.

Window extraction:
- `center_idx = spot_to_idx(spot_ref_price_int)`
- Requested index range: `[center_idx - grid_radius_ticks, center_idx + grid_radius_ticks]`
- Out-of-bound sides are zero-padded.

Relative coordinate:
- For each emitted row with absolute index `abs_idx`:
  - `k = abs_idx - center_idx`
- Therefore visible rows are always `k in [-grid_radius_ticks, +grid_radius_ticks]`.

Emitted row fields include:
- Mechanics: `add_mass`, `pull_mass`, `fill_mass`, `rest_depth`
- Derivatives: `v_*`, `a_*`, `j_*`
- Forces: `pressure_variant`, `vacuum_variant`
- Spectrum: `spectrum_score`, `spectrum_state_code`, `proj_score_h{ms}`
- Metadata: `last_event_id`

## 7. Frontend Heatmap and Signal-Panel Formulas

### 7.1 Heatmap color model

For each visible bucket row:
- Uses depth and spectrum score:
  - `depthN = min(1, log1p(depth) / log1p(maxDepth))`
  - `flowN = tanh(netFlow / FLOW_NORM_SCALE)` where `netFlow = spectrum_score * FLOW_NORM_SCALE`
  - `lum = 0.04 + depthN * 0.56`
- RGB palette switches on `flowN` sign and magnitude thresholds (`>0.03`, `<-0.03`, else neutral).

### 7.2 Right-panel aggregate features

Aggregation by `k` relative to spot:
- For `k > 0` (above spot):
  - `pressureAbove += pressure_variant`
  - `vacuumAbove += vacuum_variant`
  - `restDepthAbove += rest_depth`
- For `k < 0` (below spot):
  - `pressureBelow += pressure_variant`
  - `vacuumBelow += vacuum_variant`
  - `restDepthBelow += rest_depth`

Core derived metrics:
- `bullEdge = pressureBelow + vacuumAbove`
- `bearEdge = pressureAbove + vacuumBelow`
- `netEdge = bullEdge - bearEdge`
- `forceTotal = bullEdge + bearEdge`
- `conviction = abs(netEdge) / forceTotal` when `forceTotal > 0`, else `0`
- `restDepthTilt = restDepthBelow - restDepthAbove`
- `restDepthTotal = restDepthAbove + restDepthBelow`

Alignment flags:
- `upAligned = (vacuumAbove > pressureAbove) and (pressureBelow > vacuumBelow)`
- `downAligned = (pressureAbove > vacuumAbove) and (vacuumBelow > pressureBelow)`

State classification:
- `UP BIAS` if `netEdge > 5` and `upAligned`
- `DOWN BIAS` if `netEdge < -5` and `downAligned`
- `UP LEAN` if `netEdge > 2`
- `DOWN LEAN` if `netEdge < -2`
- Else `CHOP`

Long guidance:
- `HOLD` if `netEdge > 5` and `upAligned`
- `TIGHTEN` if `netEdge > -2`
- Else `EXIT`

Short guidance:
- `HOLD` if `netEdge < -5` and `downAligned`
- `TIGHTEN` if `netEdge < 2`
- Else `EXIT`

Risk flag:
- `SUPPORT FAILING` if `vacuumBelow > max(pressureBelow*1.15, 2)`
- `RESISTANCE THINNING` if `vacuumAbove > max(pressureAbove*1.15, 2)`
- `UP ADVANTAGE` or `DOWN ADVANTAGE` if `abs(netEdge) >= 2`
- Else `BALANCED`

Marker normalization:
- `edgeNorm = tanh(netEdge / 200) * 0.5 + 0.5`

Projection status in UI:
- `proj_score_h{ms}` values are parsed from stream, but predictive model panel is currently hardcoded to `NOT ENABLED`.

## 8. Producer Latency Telemetry Formulas

Per emitted bin (when enabled):
- `first_ingest_to_grid_ready_us = (grid_ready_ns - first_ingest_ns) / 1000`
- `last_ingest_to_grid_ready_us = (grid_ready_ns - last_ingest_ns) / 1000`
- `grid_ready_to_queue_put_done_us = (queue_put_done_ns - grid_ready_ns) / 1000`
- `first_ingest_to_queue_put_done_us = (queue_put_done_ns - first_ingest_ns) / 1000`
- `last_ingest_to_queue_put_done_us = (queue_put_done_ns - last_ingest_ns) / 1000`
- `queue_block_us = (queue_put_done_ns - queue_put_start_ns) / 1000`

Window filter semantics:
- Record bin iff it overlaps configured window:
  - Reject if `bin_end_ns <= window_start_ns`
  - Reject if `bin_start_ns >= window_end_ns`

## 9. Current Default Parameter Set (MNQH6 Config)

From `backend/src/vacuum_pressure/instrument.yaml`:
- `grid_radius_ticks = 50`
- `cell_width_ms = 100`
- `n_absolute_ticks = 8192`
- `spectrum_windows = [5, 10, 20, 40]`
- `spectrum_rollup_weights = [1.0, 1.0, 1.0, 1.0]` (normalized internally)
- `spectrum_derivative_weights = [0.55, 0.30, 0.15]` (normalized internally)
- `spectrum_tanh_scale = 3.0`
- `spectrum_threshold_neutral = 0.15`
- `zscore_window_bins = 300`
- `zscore_min_periods = 75`
- `projection_horizons_bins = [1, 2, 3, 4]` (derived ms via `cell_width_ms`)
