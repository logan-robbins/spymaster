# Spymaster - LLM Ops Reference

## Critical Rules
- ALL CODE IS OLD: overwrite/delete/extend as needed
- Do not modify raw data unless explicitly instructed
- Use nohup + verbose logging for long-running commands, check every 15s
- Work backward from entry points to find current implementation
- If pipeline/features change: update avro contracts, datasets.yaml, {product}_data.json

## Product Types
- future_mbo, future_option_mbo, equity_mbo, equity_option_cmbp_1

## Symbol Convention
- **Bronze (futures)**: Parent symbol (ES) → partitioned by contract (ESH6, ESM6, etc.)
- **Bronze (equities)**: Ticker symbol (QQQ)
- **Silver/Gold**: Full contract (ESH6) or ticker (QQQ)
- Contract selection map required before silver/gold with parent symbols

---

## 1. DOWNLOADING DATA

### Raw Data Location
`backend/lake/raw/source=databento/product_type={product_type}/`

Job trackers: `backend/logs/*jobs.json`

### Current Data Range
2026-01-05 through 2026-01-29 (18 trading days)

### Raw Data Formats
- Primary: `.dbn` (Databento native)
- Fallback: `.parquet`
- Decompress: `find . -name "*.dbn.zst" -exec zstd -d --rm {} \;`

### Download Scripts
Location: `backend/scripts/batch_download_*.py`

**Futures + Futures Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols ES \
    --include-futures \
    --options-schemas mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```
- Flat-file only: requests are always `delivery=download`, `split_duration=day`, and submitted as one session-date per job.
- Active-contract pipeline (ES futures options):
  1. Download futures definitions for `ES.FUT`
  2. Download `ohlcv-1d` for `ES.FUT`
  3. Resolve active contract by max daily volume (front-month-like mapping)
  4. Download options definitions from GLBX (`ALL_SYMBOLS`)
  5. Filter 0DTE options where `underlying == active_contract` and `expiration UTC date == session date`
  6. Submit options MBO/statistics jobs with `stype_in=raw_symbol`

**Equities + Equity Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_equities.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols QQQ \
    --equity-schemas mbo \
    --options-schemas cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```
- Flat-file only: requests are always `delivery=download`, `split_duration=day`, and submitted as one session-date per job.
- Equity 0DTE nuance (QQQ options): filter OPRA definitions by `underlying == QQQ`, `instrument_class in {C,P}`, and `expiration UTC date == session date`.

### LLM Request Routing (Instrument -> Pipeline)
- Normalize date text to `YYYY-MM-DD` before building commands. Example: `Feb 06 2026` -> `2026-02-06`.
- If request contains `ES`: use `scripts/batch_download_futures.py` with `--symbols ES`.
- If request contains `QQQ`: use `scripts/batch_download_equities.py` with `--symbols QQQ`.
- If request contains both `ES` and `QQQ`: run both scripts (futures + equities pipelines).
- Single-date request: set `--start` and `--end` to the same date.
- Multi-date request: set `--start` to first date and `--end` to last date.
- Current supported downloader symbols are strict: futures script supports `ES`, equities script supports `QQQ`. Unsupported symbols should fail fast.

### LLM Command Examples
Single date request: "download Feb 06 2026 data for ES and QQQ"
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start 2026-02-06 --end 2026-02-06 \
    --symbols ES \
    --include-futures \
    --options-schemas mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &

nohup uv run python scripts/batch_download_equities.py daemon \
    --start 2026-02-06 --end 2026-02-06 \
    --symbols QQQ \
    --equity-schemas mbo \
    --options-schemas cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```

Multi-date request: "download ES and QQQ from 2026-02-03 to 2026-02-10"
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start 2026-02-03 --end 2026-02-10 \
    --symbols ES \
    --include-futures \
    --options-schemas mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &

nohup uv run python scripts/batch_download_equities.py daemon \
    --start 2026-02-03 --end 2026-02-10 \
    --symbols QQQ \
    --equity-schemas mbo \
    --options-schemas cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```

### Contract Selection Map
```bash
cd backend
uv run python -m src.data_eng.mbo_contract_day_selector --output-path lake/selection/mbo_contract_day_selection.parquet
```
- Maps session_date → front-month contract
- Built from `bronze/source=databento/product_type=future_mbo`
- Filters for dates with premarket trades (05:00-08:30 ET)
- Required before silver/gold when using base symbol ES (independent of the downloader's day-by-day flat-file contract mapping)

---

## 2. DATA PIPELINE

### Entry Point
`backend/src/data_eng/runner.py`

### Layer Flow
Bronze (normalize) → Silver (book reconstruction) → Gold (feature engineering)

### Key Files
- Stage registry: `backend/src/data_eng/pipeline.py`
- Dataset definitions: `backend/src/data_eng/config/datasets.yaml`
- Avro contracts: `backend/src/data_eng/contracts/`
- Stage implementations: `backend/src/data_eng/stages/{bronze,silver,gold}/{product_type}/`

### Commands
```bash
cd backend

# Bronze: parent symbol
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer bronze --symbol ES --dt YYYY-MM-DD --workers 4

# Silver: full contract
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer silver --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Gold: full contract
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer gold --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Add --overwrite to rebuild
```

### Output Path Pattern
`lake/{bronze,silver,gold}/product_type={PRODUCT_TYPE}/symbol={SYMBOL}/table={TABLE}/dt={DATE}/`

### Lake File Visibility
Most `part-*.parquet` files are `.gitignore`d. Verify with:
```bash
uv run python -c "import os; print(os.listdir('lake/..../dt=YYYY-MM-DD'))"
```

### Current Coverage (2026-01-05..2026-01-29)
- Bronze: All 4 product types have full coverage (18 trading days)
- Silver: Partial (run pipeline for additional dates as needed)

### Dependencies
- `equity_option_cmbp_1` silver requires `equity_mbo` silver (for spot reference)
- `future_option_mbo` silver/gold requires selection map

---

## 3. ML

### Calibration
```bash
cd backend
uv run python -m scripts.fit_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma.json`

```bash
uv run python -m scripts.eval_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma_eval.json`

---

## 4. SERVING

### Backend Server
- Entry: `backend/src/serving/velocity_main.py`
- Stream service: `backend/src/serving/velocity_streaming.py`
- Endpoints: `backend/src/serving/routers/`

```bash
cd backend
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &
```

**WebSocket:** `ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=YYYY-MM-DD`

Query params: `speed`, `skip_minutes`

Requires: `backend/data/physics/physics_beta_gamma.json`

---

## 5. UI

### Frontend
Location: `frontend2/`

- Entry: `src/main.ts`
- WebSocket client: `src/ws-client.ts`
- Layout: `index.html`

```bash
cd frontend2
npm install
npm run dev        # http://localhost:5174
npm run build
npm run preview
```

### Particle Wave Tester
URL: http://localhost:5175/particle-wave.html
Files: `frontend2/particle-wave.html`, `frontend2/src/particle-wave.ts`

---

## 6. FEATURE DOCUMENTATION

Canonical definitions (LIVING DOCUMENTS):
- `futures_data.json` - Futures/futures options
- `equities_data.json` - Equities/equity options

---

## 7. DATA FILTERS

Location: `backend/src/data_eng/filters/`

| Layer | Action | Files |
|-------|--------|-------|
| Bronze | Hard reject | `bronze_hard_rejects.py` |
| Silver | Soft flag | `price_filters.py`, `size_filters.py` |
| Gold | Strict filter | `gold_strict_filters.py` |

---

## 8. VALIDATION SCRIPTS

```bash
cd backend
# Standard validation
uv run python scripts/validate_silver_future_mbo.py
uv run python scripts/validate_silver_equity_mbo.py
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD
uv run python scripts/validate_silver_equity_option_cmbp_1.py

# Institution-grade audit (grunt)
uv run python scripts/grunt_limitation_analysis.py
uv run python scripts/validate_institutional_fixes.py
```

### Audit Results Reference (2026-02-08 Full Pipeline Audit)
All 4 pipelines audited with synthetic tests. 169 tests passing.
- future_mbo (ESH6): Grade A, 3 bugs fixed (NaN boundary propagation, rel_ticks dtype, double accumulator reset), 29 tests
- future_option_mbo (ESH6): Grade A, 2 bugs fixed (pandas axis=1 deprecation, stale test assertion), 66 tests
- equity_mbo (QQQ): Grade A, 2 bugs fixed (depth_qty_rest clamping, redundant accumulator reset), 30 tests
- equity_option_cmbp_1 (QQQ): Grade A, 0 math bugs (2 comment fixes in datasets.yaml), 44 tests

### future_option_mbo Institutional Fixes (2026-02-02)
- Added `accounting_identity_valid` boolean field to schema
- Clamped `depth_qty_rest` to `min(depth_qty_rest, depth_qty_end)`
- 91.29% of rows have valid accounting identity
- Zero-flow rows: 0% violations (proves formula is mathematically correct)
- Use `depth_qty_end` as AUTHORITATIVE, `accounting_identity_valid` to filter

---

## 9. PROCESS MANAGEMENT

```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Backend
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend
kill $(lsof -t -iTCP:8001)
kill $(lsof -t -iTCP:5174)
tail -20 /tmp/backend.log
```

---

## Quick Start

1. Download data (section 1)
2. Generate contract selection map (section 1)
3. Run pipeline: bronze → silver → gold (section 2)
4. Fit calibration (section 3)
5. Start backend (section 4)
6. Start frontend (section 5)
7. Open: http://localhost:5174
