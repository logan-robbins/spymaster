# IMPLEMENT.md - Streaming-First Radar Architecture

## Objective
Build one product that answers, continuously and in real time:
- What is happening above and below price right now
- Are obstacles building or eroding (d1/d2/d3)
- How strike-based GEX changes as spot moves and as time passes

This product is driven by Databento MBO as the canonical microstructure source.

## Current scope (2026-01-06 only)
We have full data for 2026-01-06. All streaming work, refactors, and verification target only this date. Do not expand to multi-date runs yet.

## Non-negotiables
- Canonical source is Databento MBO (schema mbo, rtype=160) for futures and options.
- Order by ts_event, sequence. Do not order by ts_recv.
- Price encoding: price_int is fixed-point 1e-9; TICK_SIZE = 0.25; TICK_INT = 250_000_000.
- Actions: A, C, M, R, T, F, N. Sides: A, B, N.
- Snapshot handling: snapshot starts with action R and F_SNAPSHOT; book_valid stays false until F_LAST is observed; any window overlapping an incomplete snapshot is window_valid=false.
- Spot reference for futures is last trade price; if none yet in-session, use best bid when book_valid=true; else 0.
- Options metadata is sourced from instrument definitions keyed by instrument_id. Do not parse symbols.
- Online-safe: features at time t use only data <= t.
- Futures overlay is time x price; options overlay is time x strike.

## Streaming-first architecture
One canonical futures order book engine, one apply per MBO message, all futures features derived from the same state and per-window accumulators. This removes drift across outputs and keeps latency bounded for real time.

## Order Book Engine (Futures MBO)
Responsibilities and state:
- orders: order_id -> side, price_int, qty, ts_enter_price (ts_enter_price resets when price changes).
- depth: side -> price_int -> total_qty.
- book_valid flag, snapshot_in_progress flag, last_trade_price_int.
- per-window accumulators for flow and resting logic.

Action handling:
- A adds order and depth.
- C reduces or removes order and depth.
- M changes price and or qty; depth moves accordingly; ts_enter_price resets on price change.
- F reduces qty and depth; delete when qty hits 0.
- T sets last_trade_price_int.
- R clears book and starts snapshot; book_valid stays false until F_LAST.
- N is ignored.

Expose queries needed by feature extraction:
- Best bid and best ask
- Depth at price for a side
- Prices in range for HUD bounds
- Resting depth at price based on ts_enter_price and REST_NS

Window boundary behavior:
- Emit book_snapshot_1s, wall_surface_1s, and radar_vacuum_1s using the same book state and the same window accumulators.
- Reset window accumulators only; keep book state.
- window_valid reflects book_valid and snapshot state.

## Windowed accumulators (1s)
Track these per price_int and side during a window:
- add_qty, pull_qty_total, pull_qty_rest, fill_qty
- depth_qty_start and depth_qty_end
- depth_qty_rest from orders with age >= REST_NS
- d1/d2/d3 depth deltas per price_int and side across windows

Why: wall and vacuum physics depend on within-window flow and resting age, not just a static snapshot.

## Futures outputs (1s cadence)
book_snapshot_1s:
- window_start_ts_ns, window_end_ts_ns
- best_bid_price_int, best_bid_qty
- best_ask_price_int, best_ask_qty
- mid_price, mid_price_int
- last_trade_price_int
- spot_ref_price_int
- book_valid

wall_surface_1s:
- One row per window_end_ts_ns, price_int, side within HUD bounds
- spot_ref_price_int and rel_ticks
- depth_qty_start, depth_qty_end, depth_qty_rest
- add_qty, pull_qty_total, pull_qty_rest, fill_qty
- d1/d2/d3 depth deltas
- window_valid

vacuum_surface_1s:
- Derived from wall_surface_1s and gold.hud.physics_norm_calibration
- Continuous 0..1 vacuum_score; never gated by approach_dir

radar_vacuum_1s:
- Uses the same book state and window accumulators as wall_surface_1s
- Bucket membership is computed per window relative to spot_ref_price_int
- Do not persist per-order bucket labels across windows

physics_bands_1s:
- Aggregates wall and vacuum into AT, NEAR, MID, FAR bands above and below spot
- Emits above_score, below_score, vacuum_total_score

## Options outputs (1s cadence)
gex_surface_1s:
- Uses options MBO, instrument definitions, options statistics, and futures spot_ref
- Derive option mid prices from the options book state
- Join OI by option identity
- Compute gamma and aggregate to bounded strike grid
- 0DTE filter is enforced by expiration date in UTC-5

## Serving and streaming
- Real-time service applies MBO once and emits 1s batches.
- In-memory ring buffer keeps the last 30 minutes per symbol.
- Arrow IPC is used for bootstrap and stream.
- HUD bounds: HUD_MAX_TICKS = 600, height = 1201, 1s cadence.

## Reference example
Databento order book example: EXAMPLE.py

## Verification
- Depth never negative.
- best_bid_price_int < best_ask_price_int whenever book_valid=true.
- Snapshot handling sets window_valid=false until F_LAST is observed.
- All rows satisfy abs(rel_ticks) <= HUD_MAX_TICKS.
- vacuum_score is normalized using gold.hud.physics_norm_calibration.

## Immediate Tasks (living list)
4) Ensure vacuum_surface_1s and physics_bands_1s read shared outputs and enforce calibration presence. Status: DONE.
5) Build the streaming service that applies MBO once, emits 1s Arrow batches, and maintains the ring buffer with HUD bounds. Status: DONE.
6) Align gex_surface_1s to streaming: options book mid, OI join, instrument definitions, futures spot_ref. Status: DONE.
