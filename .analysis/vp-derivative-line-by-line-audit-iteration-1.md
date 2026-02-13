# Vacuum/Pressure Derivative Audit (Iteration 1, Updated)

## Intent correction from latest discussion
The target product definition is now explicit:
- Pressure/vacuum are derived from **derivatives of order-flow mechanics**, not from raw levels alone.
- Math must be identical in live and replay.
- For futures MBO, each relative tick bucket around spot has its own evolving local state.
- If a bucket receives no updates, its rendered shade persists until the next update in that bucket.

This document records how current code differs from that intent.

## 1) Current compute path (what exists now)

### 1.1 1-second data model
- Silver outputs are emitted per 1-second window:
  - futures: `backend/src/data_eng/stages/silver/future_mbo/book_engine.py:208`
  - equities: `backend/src/data_eng/stages/silver/equity_mbo/book_engine.py:214`
- Per-window fields include additive flow and end-state depth per bucket:
  - `add_qty`, `pull_qty`, `fill_qty`, `pull_qty_rest`
  - `depth_qty_end`, `depth_qty_rest`
- `rel_ticks` is relative to the window spot reference:
  - futures: `backend/src/data_eng/stages/silver/future_mbo/book_engine.py:283`
  - equities: `backend/src/data_eng/stages/silver/equity_mbo/book_engine.py:300`

### 1.2 Replay math path
- Replay computes via `run_full_pipeline` and composite derivatives:
  - `backend/src/vacuum_pressure/formulas.py:701`
  - derivative chain at `backend/src/vacuum_pressure/formulas.py:538`
- Replay is not the same engine as live.

### 1.3 Live math path
- Live computes Bernoulli fields and `net_lift` in `IncrementalSignalEngine.process_window`:
  - `backend/src/vacuum_pressure/incremental.py:881`
  - lift equations: `backend/src/vacuum_pressure/incremental.py:1017`
- "5s/15s/60s" are currently EMA timescale chains on 1-second `net_lift`, not canonical bucket-horizon recomputations:
  - `backend/src/vacuum_pressure/incremental.py:716`

## 2) Mismatch vs target intent

1. Live and replay math differ now, but target requires one canonical math path.
2. Current "pressure/vacuum" is largely formed from level-time aggregations before derivatives; target requires derivative-led force definitions at bucket level.
3. Current timescale lanes are filter labels, not true local-force horizons at each relative bucket.
4. Frontend heatmap is net-flow/depth color and does not render derivative-defined force surfaces:
   - current color mapping: `frontend2/src/vacuum-pressure.ts:697`
   - parsed but unused force-intensity fields in color logic: `frontend2/src/vacuum-pressure.ts:1837`
5. Frontend anchoring uses `mid_price` flow mapping rather than strict spot-ref frame semantics for force-field indexing:
   - spot set: `frontend2/src/vacuum-pressure.ts:1817`
   - row map: `frontend2/src/vacuum-pressure.ts:841`

## 3) Rendering persistence requirement status
- Required behavior: "no updates at a level => shade persists."
- Current buffer behavior already shifts prior columns forward and only writes new rightmost-column values:
  - shift existing pixels: `frontend2/src/vacuum-pressure.ts:819`
  - initialize new column dark then overwrite where rows exist: `frontend2/src/vacuum-pressure.ts:824`
- This persistence exists in time-history columns, but not yet as derivative-defined force shading.

## 4) Direct conclusion
The current product is not yet the intended continuous derivative-force system.  
It is still primarily a 1-second windowed feature pipeline with divergent live/replay math and non-force-native rendering.

See implementation handoff: `.analysis/IMPLEMENT.md`.
