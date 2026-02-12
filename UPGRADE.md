# Vacuum Pressure Runtime Configuration Upgrade

Date: 2026-02-12  
Owner: Market Data / UI Platform

## 1) Objective

Make `vacuum-pressure` the official UI path for liquidity-vacuum visualization with runtime instrument configuration so one stream endpoint can serve:

- `equity_mbo` symbols (e.g., `QQQ`)
- `future_mbo` contracts (e.g., `MNQH6`, `ESH6`, `SIH6`)

No options integration in this phase. Design must leave a clean extension path for options later.

## 2) Current State and Gaps

Current implementation is effectively equity-only and hardcoded around QQQ assumptions:

- Engine reads only `silver/product_type=equity_mbo/...` (`backend/src/vacuum_pressure/engine.py`).
- Server emits fixed `bucket_size=0.50` and `tick_size=0.50` (`backend/src/vacuum_pressure/server.py`).
- Formulas assume fixed bucket dollars and proximity constants in equity units (`backend/src/vacuum_pressure/formulas.py`).
- Frontend hardcodes `BUCKET_DOLLARS = 0.50` and fixed UI scale constants (`frontend2/src/vacuum-pressure.ts`).

Result: futures silver data exists but cannot be consumed correctly without semantic drift.

## 3) Target Runtime Contract

## 3.1 Runtime Identity

Every stream start must be resolved by:

- `product_type` (required): `equity_mbo` or `future_mbo`
- `symbol` (required):
- Equities: ticker (`QQQ`)
- Futures: resolved contract (`MNQH6`, not root `MNQ` in phase 1)
- `dt` (required): `YYYY-MM-DD`

## 3.2 Runtime Instrument Config

At stream start, backend resolves a config and sends it to frontend before first data batch.

Minimum config fields:

- `product_type`
- `symbol`
- `symbol_root` (for futures)
- `price_scale` (current system: `1e-9`)
- `tick_size` (native instrument tick)
- `bucket_size_dollars` (the unit represented by `rel_ticks` in loaded silver table)
- `rel_tick_size` (explicit alias of bucket size for UI)
- `grid_max_ticks`
- `contract_multiplier`
- `qty_unit` (shares/contracts)
- `price_decimals`
- `config_version` (hash or semantic version for observability)

Important semantic note:

- For `equity_mbo`, current silver `rel_ticks` is bucketized at `$0.50` (not native $0.01 tick).
- For `future_mbo`, current silver `rel_ticks` is native tick-based from per-product config.

Frontend must always render using `rel_tick_size`, never hardcoded constants.

## 4) Backend Changes

## 4.1 Add Runtime Config Resolver

Create a dedicated resolver module for vacuum-pressure runtime metadata, sourced from:

1. Existing futures product config in `backend/src/data_eng/config/products.yaml`.
2. New equity defaults (phase 1 global defaults, optional symbol overrides).
3. Optional per-symbol override layer for exceptions.

Recommended precedence:

1. Explicit symbol override
2. Product-root defaults
3. Global product-type defaults
4. Fail fast if unresolved

Expected outcome:

- One authoritative config object used by server, engine, and formulas.

## 4.2 Generalize Engine Input Selection

Update vacuum-pressure engine flow to select silver datasets by runtime `product_type` instead of hardcoded `equity_mbo`.

Required behavior:

- `equity_mbo` -> load:
- `silver/product_type=equity_mbo/symbol={symbol}/table=book_snapshot_1s/dt={dt}`
- `silver/product_type=equity_mbo/symbol={symbol}/table=depth_and_flow_1s/dt={dt}`
- `future_mbo` -> load:
- `silver/product_type=future_mbo/symbol={symbol}/table=book_snapshot_1s/dt={dt}`
- `silver/product_type=future_mbo/symbol={symbol}/table=depth_and_flow_1s/dt={dt}`

Fail-fast requirements:

- If partition missing, error must include exact missing path and exact runner command to produce it.
- If required columns missing, raise explicit schema mismatch with missing columns list.

## 4.3 Parameterize Formula Layer

Convert vacuum-pressure formula constants into a runtime parameter object passed from resolver:

- `bucket_size_dollars`
- proximity decay semantics (keep interpretation stable across instruments)
- near/depth ranges in **dollar space**, then converted to rel-tick counts per stream config

This avoids distortion when switching between:

- equity $0.50 buckets
- futures native tick buckets

Keep default behavior numerically identical for current QQQ path.

## 4.4 Stream Protocol Upgrade

Update websocket contract in `backend/src/vacuum_pressure/server.py`:

- Query params must include `product_type`.
- First control message must include full runtime config block.
- `batch_start` continues per-window timing and surface list.

Compatibility strategy:

- Keep existing fields for one transition window.
- Add new config fields immediately.
- Remove legacy hardcoded `tick_size/bucket_size` after frontend migration cutover.

## 4.5 CLI Upgrade

Update `backend/scripts/run_vacuum_pressure.py` interface:

- Add required `--product-type`.
- Keep `--symbol`, `--dt`, `--port`, `--compute-only`.
- Print resolved runtime config at startup.

## 4.6 Caching and Keying

Cache keys must include:

- `product_type`
- `symbol`
- `dt`
- `config_version`

Prevents stale cross-instrument reuse.

## 4.7 Official Readiness Checks

Before stream starts:

- Validate dataset readiness for both required silver tables.
- Validate runtime config resolved and coherent.
- Emit structured startup log with resolved config and row counts.

## 5) Frontend Changes

## 5.1 Runtime Config Consumption

Replace all hardcoded instrument constants in `frontend2/src/vacuum-pressure.ts` with server-driven config:

- bucket size used for row/price mapping
- price axis label interval
- gridline interval
- rounding precision
- any normalization constants tied to tick/bucket meaning

## 5.2 URL Contract

Update page query parameters:

- require `product_type`
- keep `symbol`, `dt`, `speed`, `skip`

Example (futures):  
`/vacuum-pressure.html?product_type=future_mbo&symbol=MNQH6&dt=2026-02-06&speed=1`

## 5.3 UI State and Diagnostics

Expose resolved metadata in top bar:

- product type
- symbol
- tick size
- rel tick size
- multiplier

This is operationally useful for debugging incorrect scaling.

## 5.4 Shared Stream Client Pattern

Align vacuum-pressure client behavior with existing velocity client pattern:

- explicit handling of stream metadata
- strict parsing for control vs binary frames
- reconnect preserving original params

## 5.5 Backward Compatibility

If config metadata absent during transition:

- display a visible warning banner
- fall back once to legacy `0.50` behavior
- log deprecation warning

Remove fallback after migration window.

## 6) Data Lake and Symbol Policy

Phase 1 policy:

- Futures require resolved contract symbol in stream request (`MNQH6`, `ESH6`).
- Equities use ticker symbol.

Future root resolution (`MNQ` -> `MNQH6`) can be added later, but should not block runtime-config rollout.

## 7) Phased Delivery Plan

1. **Protocol and Resolver First**
- Add runtime resolver and metadata emission.
- Keep old formula/frontend assumptions temporarily.

2. **Engine and Formula Parameterization**
- Product-type-based loading.
- Remove hardcoded equity path.
- Parameterize formula constants.

3. **Frontend Dynamic Rendering**
- Remove hardcoded bucket/tick assumptions.
- Render from runtime config.

4. **Cutover and Cleanup**
- Make `product_type` required.
- Remove legacy fallback fields and behavior.
- Promote page as official UI path.

## 8) Acceptance Criteria

All must pass:

1. `equity_mbo + QQQ + 2026-02-06` matches current output characteristics (no regression).
2. `future_mbo + MNQH6 + 2026-02-06` streams and renders with correct vertical price mapping.
3. Stream startup log includes resolved config and data row counts.
4. Frontend shows runtime metadata and uses dynamic tick/bucket math only.
5. Missing partitions fail fast with actionable command text.

## 9) Testing Requirements

Backend tests:

- config resolution by product type and symbol
- dataset path selection by product type
- formula invariants under different `bucket_size_dollars`
- websocket control message includes runtime config

Frontend tests:

- dynamic mapping from rel ticks to absolute price using server config
- axis precision updates with tick size
- reconnect retains runtime params

Integration tests:

- smoke stream for one equity and one future
- regression test for current QQQ reference day

## 10) Risks and Mitigations

Risk: mixing bucket semantics between equity and futures breaks signal comparability.  
Mitigation: explicit `rel_tick_size` contract and formula parameterization in dollar space.

Risk: stale cache after config changes.  
Mitigation: include `config_version` in cache key and startup logs.

Risk: frontend silently using defaults when config missing.  
Mitigation: explicit warning banner and temporary-only fallback.

## 11) Out of Scope (This Upgrade)

- Options rendering in vacuum-pressure UI
- Root-symbol-to-contract auto-resolution for futures
- Reworking silver generation bucket definitions

These are phase-2+ enhancements after runtime config and cross-instrument streaming are stable.
