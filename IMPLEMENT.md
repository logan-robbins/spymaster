1. **[BACKEND → FRONTEND CONTRACT] Add tick-native fields to eliminate float/rounding drift (fixes $5 alignment + tick buckets). [COMPLETE]**
   1.1. In `silver.future_mbo.book_snapshot_1s` → frontend `snap`, include `spot_ref_price_int` (already defined backend-side) in the stream payload, not just `mid_price`.
   1.2. In `silver.future_option_mbo.gex_surface_1s` → frontend `gex`, include at least one of:

   * `strike_price_int` (already exists backend-side), and ALSO the tick-aligned spot anchor used for that window (`spot_ref_price_int` or `underlying_spot_ref_price_int`)
   * OR directly include `rel_ticks` computed backend-side for each strike row.
     1.3. Update the frontend schema (`frontend_data.json`) to mark the new fields as required (for alignment-critical paths) and document units explicitly (“tick-int scaled by 1e-9; 1 tick = 250000000”).
     1.4. Update the WebSocket / Arrow serializer mapping so these new fields are actually transmitted (not just present in silver parquet/arrow).

2. **[BACKEND: GEX GRID INTEGRITY] Guarantee the GEX surface actually contains the full 25-strike ladder every second (fixes “missing strikes above/below”). [COMPLETE]**
   2.1. Enforce the grid spec from the backend schema: `base = round(spot/5)*5`, strikes = `base + 5*k` for `k ∈ [-12..+12]` (25 rows). 
   2.2. After aggregating option-derived gex by strike, **left-join** onto the full 25-strike grid and fill missing strikes with zeros:

   * `gex_abs = 0`, `gex = 0`, `gex_imbalance_ratio = 0`, derivatives = 0.
     (This is the canonical fix if you currently only output strikes that had trading/OI activity.)
     2.3. Add an invariant check per window: `count(rows)==25` and `sorted(strike_points)` is strictly `step=5`. If violated, emit a “GEX_GRID_INVALID” metric/log and still send the completed grid by repairing values (don’t silently drop).
     2.4. Quantize `strike_points` *exactly* from integer strikes (preferred): `strike_points = strike_price_int * PRICE_SCALE` (avoid float accumulation). 

3. **[BACKEND: GEX → TICKS] Output a tick-native strike placement so the frontend never uses continuous buckets. [COMPLETE]**
   3.1. Compute `rel_ticks_gex` in backend using integer arithmetic (no floats):

   * `spot_ref_price_int_window` = the same anchor used for futures rel_ticks that second (`spot_ref_price_int`)
   * `rel_ticks_gex = (strike_price_int - spot_ref_price_int_window) / TICK_INT` (must be integer, and must be multiples of 20 for ES options strikes). 
     3.2. Add invariant checks:
   * `abs(rel_ticks_gex) <= 240` (because ±60 points × 4 ticks/point = ±240 ticks)
   * `rel_ticks_gex % 20 == 0` (because $5 increments = 20 ticks)
   * if violated: snap to nearest `20*ticks` bucket and log. 

4. **[FRONTEND: STOP MIXING DOLLARS WITH TICKS] Convert the entire grid-layer rectification math to tick-space (fixes the vertical scale mismatch). [COMPLETE]** 
   4.1. Define an explicit “render coordinate system”: **Y = ticks**, not dollars.
   4.2. Store `spotHistory` in tick units (integer tick index), not “points”.

   * `spot_tick = spot_ref_price_int / TICK_INT` (preferred)
   * fallback: `spot_tick = round(price_points / 0.25)` using symmetric rounding (see task 7).
     4.3. Update shader uniforms: replace `uSpotRef` (points) with `uSpotRefTick` (ticks). Replace `uSpotHistory` values to be ticks.
     4.4. Rewrite the rectification formula (conceptual, exact):
   * `heightTicks = uHeight` (801)
   * `absTick = uSpotRefTick + (vUv.y * heightTicks - heightTicks/2)`
   * `histSpotTick = sample(uSpotHistory, x).r` (tick index)
   * `relTickInHist = absTick - histSpotTick`
   * `yUvSample = (relTickInHist + heightTicks/2) / heightTicks`
   * sample `uDataTex` at `(xUvSample, yUvSample)`
     (No tick-size multipliers anywhere inside the shader if everything is in ticks.) 

5. **[FRONTEND: PLANE SCALE MUST MATCH TICKS] Make the world-space plane height exact, not a hard-coded 200. [COMPLETE]** 
   5.1. Compute plane height in world “points” from ticks:

   * `height_points = TOTAL_TICKS * TICK_SIZE` = `801 * 0.25 = 200.25` 
     5.2. Use `height_points` everywhere you scale the mesh / camera view so that 1 tick row is exactly 0.25 points tall in world space.

6. **[FRONTEND: 1 SECOND = 1 COLUMN] Replace continuous UV scrolling with discrete column addressing (fixes “is every pixel 1 second wide?”). [COMPLETE]** 
   6.1. Stop using `RepeatWrapping + uHeadOffset + continuous x = vUv.x + offset` for time alignment.
   6.2. In shader, compute an integer column index from `vUv.x` and sample the exact texel center:

   * `col = floor(vUv.x * WIDTH)`
   * map to ring-buffer column: `texCol = (head + col) % WIDTH` (or `(head - (WIDTH-1-col)) % WIDTH` depending on which edge is “now”)
   * `xUvSample = (texCol + 0.5) / WIDTH`
     6.3. Pass `head` as an integer uniform (or float but representing integer) and pass `WIDTH` explicitly to shader.
     6.4. Add an automated visual sanity test: render a known “vertical bar every 10 seconds” pattern and verify it is exactly 10 columns apart (no interpolation drift).

7. **[FRONTEND: ROUNDING MUST BE SYMMETRIC] Eliminate `Math.round` half-tick asymmetry (critical for gex strikes when spot is mid). [COMPLETE]** 
   7.1. Implement a deterministic symmetric rounding function for float→tick:

   * `round_half_away_from_zero(x)` OR “banker’s rounding” consistently, but MUST be symmetric for negative values (JS `Math.round(-4.5)=-4` is NOT symmetric).
     7.2. Prefer integer price_int / tick_int everywhere so this rounding never runs in production for core placement.

8. **[FRONTEND: GEX PLACEMENT MUST USE TICK BUCKETS] Place GEX rows using backend-provided `rel_ticks` (or tick ints), never float diffs. [COMPLETE]**
   8.1. If backend emits `rel_ticks_gex`: use it directly (assert multiples of 20).
   8.2. If backend emits only `strike_price_int` and `spot_ref_price_int`: compute `rel_ticks_gex = (strike_price_int - spot_ref_price_int)/TICK_INT` (integer).
   8.3. Frontend should hard-assert `rel_ticks_gex % 20 == 0`; if not, snap to nearest 20-tick bucket and increment a counter.

9. **[FRONTEND: TEXTURE FILTERING] Force pixel-perfect buckets (fixes “continuous numbers for buckets”). [COMPLETE]** 
   9.1. Use `NearestFilter` for `minFilter` and `magFilter` for ALL physics grids when in “bucketed” mode (at least for wall + gex + per-tick physics).
   9.2. If you want smoothing, do it explicitly (convolution over buckets), not via bilinear texture sampling.

10. **[FRONTEND: COLOR SCALE CONSISTENCY] Normalize wall encoding (currently mixes 0..255 RGB with 0..1 alpha). [COMPLETE]** 
    10.1. Choose ONE representation for all layers:

* Option A: store all RGBA as 0..1 floats in texture; shader outputs directly.
* Option B: store all RGBA as 0..255; shader divides by 255 for every layer.
  10.2. Apply the chosen representation consistently to wall/vacuum/physics/gex so one layer doesn’t clamp to white and mask others.

11. **[FRONTEND: LAYER VISIBILITY] Ensure GEX is not being visually masked by vacuum/physics overlays. [COMPLETE]** 
    11.1. Render order fix: ensure gex is drawn AFTER vacuum if vacuum is black-alpha overlay (otherwise it darkens gex).
    11.2. If keeping Z-sorting, set explicit `renderOrder` per layer (don’t rely on z alone with transparent blending).
    11.3. Add a debug toggle: show ONLY gex layer (others hidden) to validate $5 alignment independent of masking.

12. **[FRONTEND: TIME GAP FILL] Maintain “1 column = 1 second” even if the stream skips windows. [COMPLETE]** 
    12.1. Track last `window_end_ts_ns` advanced.
    12.2. If `Δt_seconds > 1`, advance the ring buffer `Δt_seconds` times inserting empty/carry-forward columns so the x-axis remains true-time aligned.

13. **[FRONTEND: TIMESTAMP SAFETY] Never write rows into the “current head column” unless their `window_end_ts_ns` matches it.** 
    13.1. For each update function (wall/vacuum/gex/physics), filter rows: keep only rows where `row.window_end_ts_ns == advancedTickTs`.
    13.2. If you want late-arriving data support: compute the correct column index for that timestamp and write there (bounded by history window), else drop with metric.

14. **[BACKEND: ISSUE #2 — TICK-LEVEL PHYSICS SURFACE] Add a per-tick physics stream (bands are not bucketed enough).**
    14.1. Create `silver.future_mbo.physics_surface_1s` (new dataset) with at minimum:

* `window_end_ts_ns`, `rel_ticks` (int), `side` (A/B), `physics_score` (0..1), optionally `physics_score_signed` (-1..+1).
  14.2. Compute per tick using the same components already defined for `physics_bands_1s`: 
* `wall_strength_log = log(depth_qty_rest + 1)`
* `erosion_log1p = log1p(max(-d1_depth_qty, 0) / (depth_qty_start + EPS))`
* `vacuum_norm = vacuum_score` (already 0..1)
* Normalize `wall_strength_log` and `erosion_log1p` using the same q05/q95 calibration rule: `clip((val-q05)/(q95-q05),0,1)` 
* `ease = 0.50*vacuum_norm + 0.35*erosion_norm + 0.15*(1 - wall_strength_norm)` 
* Direction assignment per tick:

  * if `side=='A' and rel_ticks>0`: `physics_score_signed = +ease`
  * if `side=='B' and rel_ticks<0`: `physics_score_signed = -ease`
  * else 0. 

15. **[FRONTEND: ISSUE #2 RENDERING] Render tick-physics as true buckets (1 tick tall × 1 second wide).** 
    15.1. Replace the current “physics gradient” writing logic with per-row writes at `rel_ticks`.
    15.2. Use alpha proportional to `abs(score_signed)` and color = sign (green/red).
    15.3. Do NOT smear via shader smoothstep; any smoothing must be explicit and optional.

16. **[DISSIPATION MODEL] Implement bucketed temporal decay (so “pressure dissipates at specific levels”).** 
    16.1. For the per-tick physics grid (and optionally wall/vacuum), change the per-column initialization from “clear to 0” to “copy previous column and decay”:

* `new_cell = old_cell * exp(-Δt/τ)` for each tick row.
  16.2. Then apply current-second injections (writes) on top of the decayed field (e.g., max or additive with clamp).
  16.3. Keep `τ` configurable per layer (physics vs wall vs vacuum).

17. **[GEX VISUAL BUCKET HEIGHT] Enforce “few ticks high, not 1 full strike” while still aligning to $5 levels. [COMPLETE]** 
    17.1. Compute the strike center row as tick index (multiples of 20).
    17.2. Draw a constant small band height `h_ticks` (e.g., 2–4 ticks) around that strike row (centered), independent of strike spacing.
    17.3. Do NOT scale the band height up to 20 ticks (a full $5 strike block) unless explicitly in “coarse strike mode”.

18. **[DATA INTEGRITY TESTS] Add automated checks for issue #1 (incoming GEX correctness) and issue #2 (tick bucket correctness).**
    18.1. Per window, validate GEX:

* rows==25, strikes monotonic, `Δstrike==5`, `abs(imbalance)<=1`, `gex_abs>=0`.
  18.2. Per row, validate tick mapping:
* `price_int == spot_ref_price_int + rel_ticks*TICK_INT` for wall/vacuum. 
  18.3. Per row, validate GEX tick mapping:
* `rel_ticks_gex % 20 == 0` and within ±240. 

19. **[FRONTEND DEBUG OVERLAY] Make misalignment obvious in one glance.** 
    19.1. Draw horizontal guide lines at:

* every 1 point (4 ticks)
* every $5 strike (20 ticks)
  relative to spot (row 0).
  19.2. Display per-tick diagnostics in HUD: current `spot_ref_tick`, current `head`, last window_end_ts, and per-stream row counts for that second.

20. **[REMOVE AMBIGUOUS “MID” ANCHORING] Decide one canonical spot anchor for grid alignment and use it everywhere. [COMPLETE]**
    20.1. Canonical for bucket visuals: `spot_ref_price_int` (tick-aligned) from `book_snapshot_1s`. 
    20.2. If you still want a smoother visible line, render mid_price as a separate cosmetic line, but DO NOT let it drive tick anchoring / rectification.
