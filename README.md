# Spymaster - LLM Ops Reference

## Quick Access: Particle Wave Tester
**URL:** http://localhost:5175/particle-wave.html
**File:** [`frontend2/particle-wave.html`](frontend2/particle-wave.html) + [`frontend2/src/particle-wave.ts`](frontend2/src/particle-wave.ts)
**Run:** `cd frontend2 && npm run dev` then open the URL above

Interactive 2D particle wave simulation with:
- Wave parameters: wavelength (10 units), frequency (1 Hz), speed (10 units/s), amplitude
- Physics tensor: viscosity, decay, distance-to-object, permeability, object density
- Tunable object properties: density, position, dimensions
- Real-time visualization with pause/resume and tensor view modes

## Critical Rules
- ALL CODE IS OLD: overwrite/delete/extend as needed
- Do not modify raw data unless explicitly instructed
- Use nohup + verbose logging for long-running commands, check every 15s
- Work backward from entry points to find current implementation
- If pipeline/features change: update avro contracts, datasets.yaml, {product}_data.json

## Product Types
- future_mbo, future_option_mbo, equity_mbo, equity_option_cmbp_1

## Symbol Convention Gotcha
- **Bronze layer (future_mbo)**: Uses full contract (ESH6) - data partitioned by contract
- **Bronze layer (equities)**: Uses ticker symbol (QQQ)
- **Silver/Gold layers**: Use full contract (ESH6) or ticker (QQQ)
- Contract selection map required: run mbo_contract_day_selector before silver/gold with parent symbols

---

## 1. DOWNLOADING DATA

### Raw Data Inventory
**Inventory source-of-truth:** `backend/lake/raw/source=databento/` + job trackers in `backend/logs/*jobs.json`

**Current Holdings (2026-01-05 to 2026-01-29, all 18 trading days):**
- ES futures: 22 days (incl. Sunday sessions) - 16 GB  
- **ES futures options: 18 days - ~133 GB ✅ 0DTE ONLY** (2 days parquet, 16 days dbn)
- QQQ equity: 18 days - ~40 GB
- **QQQ equity options (cmbp-1): 18 days - ~80 GB ✅ 0DTE ONLY** (decompressed from .dbn.zst)
- QQQ equity options (statistics): 18 days - ~2 MB (ex-MLK day 2026-01-19)

**Raw Data Formats:**
- Primary: `.dbn` (Databento native format)
- Fallback: `.parquet` (future_option_mbo pipeline auto-detects and loads with format conversion)
- Note: After downloading .dbn.zst files, decompress with: `find . -name "*.dbn.zst" -exec zstd -d --rm {} \;`

### Raw Data Scripts
Location: `backend/scripts/batch_download_*.py`

**Futures + Futures Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_futures.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols ES \
    --include-futures \
    --options-schemas definition,mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```
Output: `lake/raw/source=databento/product_type={future_mbo,future_option_mbo}/`
Job tracker: `logs/futures_jobs.json`

**Equities + Equity Options:**
```bash
cd backend
nohup uv run python scripts/batch_download_equities.py daemon \
    --start YYYY-MM-DD --end YYYY-MM-DD \
    --symbols SPY,QQQ \
    --equity-schemas mbo \
    --options-schemas definition,cmbp-1,statistics \
    --poll-interval 60 \
    --log-file logs/equities.log > logs/equities_daemon.out 2>&1 &
```
Output: `lake/raw/source=databento/product_type={equity_mbo,equity_option_cmbp_1}/`
Job tracker: `logs/equity_options_jobs.json`

**Monitor/Resume:**
```bash
tail -f logs/{futures,equities}.log
uv run python scripts/batch_download_*.py poll --log-file logs/*.log
```

### Contract Selection (prerequisite)
```bash
cd backend
uv run python -m src.data_eng.mbo_contract_day_selector --dates YYYY-MM-DD --output-path lake/selection/mbo_contract_day_selection.parquet
```
Maps session_date → front-month contract. Required before silver/gold with parent symbols.
Current selection map rows (2026-02-01): 19 trading days from 2026-01-05 through 2026-01-29.
- All dates use ESH6 (March 2026 expiry)
- Excluded: 2026-01-11, 2026-01-18, 2026-01-25 (Sunday sessions with no premarket trades)
Selection map is built from `bronze/source=databento/product_type=future_mbo` and only includes dates with bronze MBO data
and premarket trades (05:00-08:30 ET). Missing dates in this map are skipped by silver/gold when using base symbol ES.

---

## 2. DATA PIPELINE

### Entry Point
`backend/src/data_eng/runner.py` - Orchestrates all stages

### Layer Flow
Bronze (normalize) → Silver (book reconstruction) → Gold (feature engineering)

### Stage Registry
`backend/src/data_eng/pipeline.py` - Maps (product_type, layer) → stage classes

### Dataset Definitions
`backend/src/data_eng/config/datasets.yaml` - Schema registry for all tables

### Avro Contracts
`backend/src/data_eng/contracts/` - Type definitions for all data products

### Commands
```bash
cd backend

# Bronze: parent symbol (ES)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer bronze --symbol ES --dt YYYY-MM-DD --workers 4

# Silver: full contract (ESH6)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer silver --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Gold: full contract (ESH6)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer gold --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Add --overwrite to rebuild
```

Output: `lake/{bronze,silver,gold}/product_type={PRODUCT_TYPE}/symbol={SYMBOL}/table={TABLE}/dt={DATE}/`

### Lake File Visibility Gotcha
Most `part-*.parquet` data files under `backend/lake/**` are `.gitignore`d. Some tooling will only show `_MANIFEST.json` + `_SUCCESS`.

To verify the actual data file exists for a partition, use Python:
```bash
cd backend
uv run python -c "import os; print(os.listdir('lake/silver/product_type=.../symbol=.../table=.../dt=YYYY-MM-DD'))"
```

### Current Bronze Coverage (2026-01-05..2026-01-29)
- `future_mbo`: 22 dates (19 weekdays + 3 Sunday sessions: 2026-01-11, 2026-01-18, 2026-01-25)
- `future_option_mbo`: 18 trading days (full coverage)
- `equity_mbo`: 18 trading days (full coverage)
- `equity_option_cmbp_1`: 18 trading days (full coverage) - 2026-01-05 through 2026-01-29

### Current Silver Coverage (2026-01-05..2026-01-29)
- `future_mbo`: `book_snapshot_1s` + `depth_and_flow_1s` on dt={2026-01-05,2026-01-06}
- `equity_mbo`: `book_snapshot_1s` + `depth_and_flow_1s` on dt={2026-01-07,2026-01-15,2026-01-27} (first-hour window)
- `equity_option_cmbp_1`: `book_snapshot_1s` + `depth_and_flow_1s` on dt={2026-01-07,2026-01-15,2026-01-27} (first-hour window; depends on `silver.equity_mbo.book_snapshot_1s`)

### Stage Implementations
- Bronze stages: `backend/src/data_eng/stages/bronze/{product_type}/`
- Silver stages: `backend/src/data_eng/stages/silver/{product_type}/` (book engines)
- Gold stages: `backend/src/data_eng/stages/gold/{product_type}/` (feature engineering)

---

## 3. ML

### Calibration (Stage F)
Requires: Last 20 trading days (09:30–12:30 ET) in gold layer

**Fit parameters:**
```bash
cd backend
uv run python -m scripts.fit_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma.json`

**Evaluate:**
```bash
cd backend
uv run python -m scripts.eval_lookahead_beta_gamma
```
Output: `backend/data/physics/physics_beta_gamma_eval.json`

Requirement: Confidence monotonicity across horizons

---

## 4. SERVING

### Backend Server
Entry: `backend/src/serving/velocity_main.py`
Stream service: `backend/src/serving/velocity_streaming.py`
Endpoints: `backend/src/serving/routers/`

**Start:**
```bash
cd backend
nohup uv run python -m src.serving.velocity_main > /tmp/backend.log 2>&1 &
sleep 3 && lsof -iTCP:8001 -sTCP:LISTEN
```

**WebSocket:**
`ws://localhost:8001/v1/velocity/stream?symbol=ESH6&dt=YYYY-MM-DD`
Query params: `speed` (playback multiplier), `skip_minutes` (skip N minutes at start)

**Protocol:**
Per 1s window: batch_start → surface_header + Arrow IPC (snap, velocity, options, forecast)

Requires: `backend/data/physics/physics_beta_gamma.json` (fit Stage F first)

---

## 5. UI

### Frontend Location
`frontend2/`

### Main Files
- Entry: `src/main.ts`
- Grid renderers: `src/{velocity,options,forecast,spot}-*.ts`
- WebSocket client: `src/ws-client.ts`
- Layout: `index.html`

### Commands
```bash
cd frontend2
npm install
npm run dev        # Dev server: http://localhost:5174
npm run build      # Production build
npm run preview    # Preview production
```

**Check Status:**
```bash
lsof -iTCP:5174 -sTCP:LISTEN
```

### Features
- Restart Stream button (top-left): reconnect WebSocket, clear state
- Zoom: vertical (scroll), horizontal (Cmd+scroll or trackpad swipe), independent up to 8x
- Pan: click-drag vertically
- Price axis: auto-scales with zoom level

---

## Feature Documentation

**Canonical feature definitions:**
- `futures_data.json` - Futures/futures options features
- `equities_data.json` - Equities/equity options features

These are LIVING DOCUMENTS. Reference these for current feature semantics, not this README.

---

## Data Validation Status

### future_mbo Silver Layer (validated 2026-02-01)
**Dates validated:** 2026-01-06, 2026-01-07, 2026-01-08, 2026-01-15, 2026-01-22
**Status:** PASS

**Bug fixed (prior):** `depth_qty_rest > depth_qty_end` desync caused by Databento MBO fill events reporting trade size instead of order fill amount for aggressor orders. Fix: cap fill quantity at order's remaining quantity in `book_engine.py::_fill_order()`.

**Code cleanup:** Removed unused classes `ApproachDirState` and `RadarDerivativeState` from `book_engine.py`. Removed DEBUG print statements.

**Findings:**
- 0 null/NaN/Inf values across all columns
- 0 crossed books (bid always < ask)
- 0 negative quantity values
- All formulas verified: mid_price, mid_price_int, depth_qty_start, rel_ticks (100% match)
- `depth_qty_rest <= depth_qty_end` (constraint holds)
- `pull_qty_rest <= pull_qty` (constraint holds)
- Spread: 1-2 ticks (mostly 1 tick = $0.25)
- 100% book_valid rate
- 100% window_valid rate

**Price ranges (ES/ESH6):**
- 2026-01-06: $6945.62 - $6977.12
- 2026-01-07: $6977.38 - $7006.62
- 2026-01-08: $6940.38 - $6967.12
- 2026-01-15: $6992.38 - $7017.12
- 2026-01-22: $6925.88 - $6963.50

**Tables:**
- `book_snapshot_1s`: 10,801 rows per day (3-hour window 09:30-12:30 ET)
- `depth_and_flow_1s`: ~3.7-4.3M rows per day (401 price levels × 2 sides × seconds)

**Validation script:**
```bash
cd backend
uv run python scripts/validate_silver_future_mbo.py
```

### equity_mbo Silver Layer (validated 2026-02-01)
**Dates validated:** 2026-01-07, 2026-01-15, 2026-01-27
**Status:** PASS

**Findings:**
- 0 null values across all columns
- 100% book_valid (all windows valid)
- 0 crossed books (ask always > bid)
- 0 negative quantity values
- Side balance ~50/50 bid/ask (1.6-1.7% imbalance)
- rel_ticks range: [-100, 100] ($50 grid from spot)
- Spread: $0.01-$0.05 (typical equity spread)

**Formula Validation:**
- `mid_price`: (bid + ask) * 0.5 * 1e-9 - max error 0.00e+00
- `mid_price_int`: round((bid + ask) * 0.5) - max error 0
- `depth_qty_start`: depth_qty_end - add_qty + pull_qty + fill_qty - max error 0.00e+00
- `rel_ticks`: (price_int - spot_ref) / BUCKET_INT - 100% match

**Bronze-Silver Volume Match:**
- Add volume: ~10-13% (expected: silver processes 10-min window vs full-day bronze)
- Fill volume: ~9-12% (same window effect)

**Price ranges (QQQ):**
- 2026-01-07: $622.60-$624.11
- 2026-01-15: $624.92-$627.03
- 2026-01-27: $627.34-$629.61

**Tables:**
- `book_snapshot_1s`: 601 rows per day (10-minute dev window)
- `depth_and_flow_1s`: ~119K rows per day (bucketed depth at $0.50 granularity)

**Run silver pipeline:**
```bash
cd backend
uv run python -m src.data_eng.runner --product-type equity_mbo --layer silver --symbol QQQ --dt YYYY-MM-DD --workers 4
```

**Validation script:**
```bash
cd backend
uv run python scripts/validate_silver_equity_mbo.py
```

### future_option_mbo Silver Layer (validated 2026-02-01)
**Dates validated:** 2026-01-06, 2026-01-07, 2026-01-14, 2026-01-23
**Status:** PASS (prior) — revalidate after strike bucketing removal

**Performance (optimized 2026-02-01):**
- Full day pipeline: **~48 seconds** (previously ~5+ minutes)
- 74% reduction in engine processing time
- Optimizations: replaced Numba typed lists with pre-allocated numpy arrays, added JIT caching

**Bug fixed (prior):** `depth_qty_start` formula violation caused by aggregating per-instrument flows to $5 strike buckets. When multiple instruments map to the same bucket, the formula `depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty` can produce negative values. Fix: track `depth_qty_start` explicitly in `options_book_engine.py` before events modify depth, then aggregate properly.

**Bug fixed (2026-02-01):** Removed strike bucketing in `future_option_mbo` flow surface. Use original strike values directly and compute `rel_ticks` as `(strike - spot)` in ticks (no rounding), with strike grid anchored to nearest $5.

**Rebuild (2026-02-02):** Re-ingested bronze `future_option_mbo` for 2026-01-05..2026-01-29. Silver rebuilt for selection-map dates: 2026-01-06, 2026-01-07, 2026-01-14, 2026-01-15, 2026-01-22, 2026-01-23. (Weekends + 2026-01-19 had no raw data.)

**Findings:**
- 0 negative `depth_qty_start` values (constraint now holds after fix)
- 0 null/NaN/Inf values across all columns
- Valid warm-up: first window ref_price in expected ES range ($6900-$7100)
- Accounting identity mismatch previously ~8-10% due to strike bucketing; rerun validation after removal
- 100% 1-second windows (timestamp ordering verified)
- Right distribution: 50/50 C/P (calls/puts)
- Side distribution: 50/50 A/B (ask/bid)
- rel_ticks reflect strike - spot in ticks (strike spacing = 20 ticks; offset varies with spot)

**Price ranges (ESH6):**
- 2026-01-07: $6977-$7007
- 2026-01-14: $6923-$6979
- 2026-01-23: $6926-$6964

**Tables:**
- `book_snapshot_1s`: ~1M rows per day (option instruments × seconds)
- `depth_and_flow_1s`: ~907K rows per day (21 strike offsets × 2 rights × 2 sides × seconds)

**Note:** `pull_qty_rest` outputs zeros - not used by gold layer downstream (tracking removed for performance).

**Validation script:**
```bash
cd backend
uv run python scripts/grunt_validate_future_option_mbo.py
# Or for specific date:
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD --verbose
```

### equity_option_cmbp_1 Silver Layer (validated 2026-02-01)
**Dates validated:** 2026-01-07, 2026-01-09, 2026-01-15, 2026-01-16, 2026-01-27
**Status:** PASS

**Bug fixed:** Missing `_reset_accumulators()` method in `cmbp1_book_engine.py` caused AttributeError. Added method to clear acc_add/acc_pull dictionaries between windows.

**Findings:**
- 0 null values across all columns
- 0 crossed books (ask always > bid)
- 100% book_valid=True, 100% window_valid=True
- Rights: C and P only (calls/puts)
- Sides: A and B only (ask/bid)
- rel_ticks aligned to $1 grid (all even values, multiples of 2)
- All quantity fields non-negative
- Accounting identity holds: `depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty` (max_abs_error=0)

**Note:** `pull_qty_rest` and `fill_qty` are always 0 for CMBP-1 data (BBO-only, no order tracking).

**Price ranges (QQQ options):**
- 2026-01-09: spot $620-623, unique instruments: 175, snap rows: 102,850
- 2026-01-16: spot $624.5-626, unique instruments: 271, snap rows: 159,262
- 2026-01-27: spot $627.5-629.5, unique instruments: 149, snap rows: 88,228

**Tables:**
- `book_snapshot_1s`: 88K-159K rows per day (599 windows × 149-271 instruments)
- `depth_and_flow_1s`: 122,400 rows per day (600 windows × 51 strikes × 2 rights × 2 sides)

**Dependencies:** Requires `silver.equity_mbo.book_snapshot_1s` for spot reference prices.

**Run silver pipeline:**
```bash
cd backend
uv run python -m src.data_eng.runner --product-type equity_option_cmbp_1 --layer silver --symbol QQQ --dt YYYY-MM-DD --workers 4
```

**Validation script:**
```bash
cd backend
uv run python scripts/validate_silver_equity_option_cmbp_1.py
```

---

## Institutional-Grade Data Filters

**Location:** `backend/src/data_eng/filters/`

**3-Layer Strategy:**

| Layer | Action | Files |
|-------|--------|-------|
| **Bronze** | Hard reject impossible values | `bronze_hard_rejects.py` |
| **Silver** | Soft flag suspicious data (preserve for forensics) | `price_filters.py`, `size_filters.py` |
| **Gold** | Strict filter flagged rows for clean ML features | `gold_strict_filters.py` |

**Thresholds (NYSE Rule 128 / CME / Academic):**
- Price deviation: 3σ z-score from rolling median, or 3-7% from reference (tier-based)
- Size outliers: >99.9th percentile or >10x median
- IV anomalies: <1% or >500% implied volatility
- Fat-finger: Option price >$100,000 (e.g., $187,187 found in dataset)
- Crossed markets: bid >= ask

**Test filters:**
```bash
cd backend
uv run python -m scripts.test_filters
```

**Filter flags added to data:**
- `price_outlier_flag`: Extreme price deviation from reference
- `crossed_market_flag`: Bid >= ask (data quality issue)
- `spread_anomaly_flag`: Spread > 10x median spread
- `size_outlier_flag`: Size > 99.9th percentile
- `iv_anomaly_flag`: IV outside [1%, 500%]
- `arbitrage_violation_flag`: No-arbitrage bounds violated

---

## Process Management

**Check running:**
```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Backend
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend
```

**Kill:**
```bash
kill $(lsof -t -iTCP:8001)
kill $(lsof -t -iTCP:5174)
```

**Debug backend:**
```bash
tail -20 /tmp/backend.log
```

---

## Quick Start

1. Download data (section 1)
2. Generate contract selection (section 1)
3. Run pipeline: bronze → silver → gold (section 2)
4. Fit Stage F (section 3)
5. Start backend (section 4)
6. Start frontend (section 5)
7. Open: http://localhost:5174
