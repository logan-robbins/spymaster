# Spymaster - LLM Ops Reference

## Critical Rules
- Do not modify raw data unless explicitly instructed
- Use nohup + verbose logging for long-running commands, check every 15s
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

### Current Raw Data (as of 2026-02-09)
All raw data is `.dbn` format (Databento native). Date: **2026-02-06 only**.

| Symbol | Product Types | Size |
|--------|--------------|------|
| SI | future_mbo (283 MB), future_option_mbo (1.0 GB + 5.3 MB stats), definition JSON | ~1.3 GB |
| MNQ | future_mbo (2.7 GB), future_option_mbo (1.7 GB + 6.8 MB stats), definition (820 KB) | ~4.4 GB |
| QQQ | equity_mbo (2.0 GB), equity_option_cmbp_1 (6.4 GB), statistics (1.8 MB), definition (3.7 MB) | ~8.4 GB |
| ES | metadata JSON only, no `.dbn` data files | — |

Decompress: `find . -name "*.dbn.zst" -exec zstd -d --rm {} \;`

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

### LLM Request Routing (Instrument -> Pipeline)
- Normalize date text to `YYYY-MM-DD`. Example: `Feb 06 2026` -> `2026-02-06`.
- `ES`, `NQ`, `SI`, `GC`, `CL`, `6E`, `MNQ` → `scripts/batch_download_futures.py`
- `QQQ`, `AAPL`, `SPY` (any OPRA ticker) → `scripts/batch_download_equities.py`
- Both → run both scripts.
- Single date: `--start` and `--end` same. Multi-date: range.

### Contract Selection Map
```bash
cd backend
uv run python -m src.data_eng.mbo_contract_day_selector --output-path lake/selection/mbo_contract_day_selection.parquet
```
- Required before silver/gold when using base symbol ES
- Built from `bronze/source=databento/product_type=future_mbo`

---

## 2. DATA PIPELINE

### Entry Point
`backend/src/data_eng/runner.py`

### Layer Flow
Bronze (normalize) → Silver (book reconstruction) → Gold (feature engineering)

### Key Files
- Stage registry: `backend/src/data_eng/pipeline.py`
- Dataset definitions: `backend/src/data_eng/config/datasets.yaml`
- Product config: `backend/src/data_eng/config/products.yaml`
- Bronze session windows: `backend/src/data_eng/utils.py` → `session_window_ns()`
- Avro contracts: `backend/src/data_eng/contracts/`
- Stage implementations: `backend/src/data_eng/stages/{bronze,silver,gold}/{product_type}/`
- Book engines: `backend/src/data_eng/stages/silver/{equity,future}_mbo/book_engine.py`
- Filters: `backend/src/data_eng/filters/`

### Per-Product Configuration
`backend/src/data_eng/config/products.yaml`

| Root | tick_size | grid_max_ticks | strike_step | strike_ticks | multiplier |
|------|-----------|----------------|-------------|--------------|------------|
| ES   | 0.25      | 200            | $5          | 20           | 50.0       |
| MES  | 0.25      | 200            | $5          | 20           | 5.0        |
| NQ   | 0.25      | 400            | $5          | 20           | 20.0       |
| MNQ  | 0.25      | 400            | $5          | 20           | 2.0        |
| GC   | 0.10      | 200            | $5          | 50           | 100.0      |
| SI   | 0.005     | 200            | $0.25       | 50           | 5000.0     |
| CL   | 0.01      | 200            | $0.50       | 50           | 1000.0     |
| 6E   | 0.00005   | 200            | $0.005      | 100          | 125000.0   |

Runner extracts root from symbol (e.g., MNQH6 → MNQ) and passes `ProductConfig` to all stages.

### Commands
```bash
cd backend

# Bronze: parent symbol (no --overwrite; delete partition dir to rebuild)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer bronze --symbol ES --dt YYYY-MM-DD --workers 4

# Silver: full contract (--overwrite supported)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer silver --symbol ESH6 --dt YYYY-MM-DD --workers 4

# Gold: full contract (--overwrite supported)
uv run python -m src.data_eng.runner --product-type {PRODUCT_TYPE} --layer gold --symbol ESH6 --dt YYYY-MM-DD --workers 4
```

### Output Path Pattern
`lake/{bronze,silver,gold}/product_type={PRODUCT_TYPE}/symbol={SYMBOL}/table={TABLE}/dt={DATE}/`

### Bronze Ingestion Windows

Controlled by `session_window_ns(session_date, product_type)` in `backend/src/data_eng/utils.py`. Snapshot (`F_SNAPSHOT=32`) and Clear (`action=R`) records are exempt from the time filter (their `ts_event` preserves original order placement time, which predates the session window).

| Product Type | Bronze Window | Session Start |
|---|---|---|
| `equity_mbo` | 02:00–16:00 ET | XNAS Clear at ~03:05 ET |
| `equity_option_cmbp_1` | 02:00–16:00 ET | Same as equities |
| `future_mbo` | 00:00–24:00 UTC | GLBX snapshot at 00:00 UTC (1 Clear + ~6K snapshot Adds) |
| `future_option_mbo` | 00:00–24:00 UTC | Same as futures |

### Silver Warmup

Book engines warm up from bronze start before the output window. Must reach back past session start for zero orphan orders.

| Product Type | Warmup | Output Start | Reaches Back To |
|---|---|---|---|
| `equity_mbo` | 8 hours | 09:30 ET | 01:30 ET (before 02:00 ET bronze) |
| `future_mbo` | 15 hours | 09:30 ET (14:30 UTC) | 23:30 UTC prior day (before 00:00 UTC bronze) |

### Databento MBO Flags (`u8` bitmask)

Canonical source: `databento-python/databento/common/enums.py`

| Flag | Value | Bit | Meaning |
|---|---|---|---|
| `F_LAST` | 128 | 7 | Last record in event for instrument_id |
| `F_TOB` | 64 | 6 | Top-of-book message |
| `F_SNAPSHOT` | 32 | 5 | Sourced from replay/snapshot server |
| `F_MBP` | 16 | 4 | Aggregated price level |
| `F_BAD_TS_RECV` | 8 | 3 | ts_recv is inaccurate |
| `F_MAYBE_BAD_BOOK` | 4 | 2 | Unrecoverable gap detected |

### Current Coverage (as of 2026-02-09)
- Raw: SI, MNQ, QQQ for 2026-02-06 only (ES metadata only)
- Bronze: MNQH6 future_mbo (51.4M rows incl. 6,811 snapshot), QQQ equity_mbo (38.1M rows, 0 orphans) + equity_option_cmbp_1
- Silver: MNQH6 future_mbo (10,801 snap + 8.6M flow), QQQ equity_mbo (601 snap + 121K flow) + equity_option_cmbp_1
- Gold: Empty (needs pipeline run)

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

`batch_start` messages include product metadata: `tick_size`, `tick_int`, `strike_ticks`, `grid_max_ticks`.

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

## 7. VALIDATION

```bash
cd backend
uv run python scripts/validate_silver_future_mbo.py
uv run python scripts/validate_silver_equity_mbo.py
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD
uv run python scripts/validate_silver_equity_option_cmbp_1.py
```

Tests: `uv run pytest tests/streaming/ -v`

---

## 8. PROCESS MANAGEMENT

```bash
lsof -iTCP:8001 -sTCP:LISTEN   # Backend (velocity server)
lsof -iTCP:8002 -sTCP:LISTEN   # Backend (vacuum pressure server)
lsof -iTCP:5174 -sTCP:LISTEN   # Frontend
kill $(lsof -t -iTCP:8001)
kill $(lsof -t -iTCP:5174)
tail -20 /tmp/backend.log
```

---

## 9. VACUUM PRESSURE DETECTOR

- Formulas: `backend/src/vacuum_pressure/formulas.py`
- Engine: `backend/src/vacuum_pressure/engine.py`
- Server: `backend/src/vacuum_pressure/server.py`
- CLI: `backend/scripts/run_vacuum_pressure.py`
- Frontend: `frontend2/vacuum-pressure.html`, `frontend2/src/vacuum-pressure.ts`

```bash
# 1. Requires silver equity_mbo
cd backend
uv run python -m src.data_eng.runner --product-type equity_mbo --layer silver --symbol QQQ --dt 2026-02-06 --workers 4

# 2. Start vacuum/pressure server (port 8002)
uv run python scripts/run_vacuum_pressure.py --symbol QQQ --dt 2026-02-06 --port 8002

# 3. Start frontend (separate terminal)
cd frontend2 && npm run dev

# 4. Open: http://localhost:5174/vacuum-pressure.html?symbol=QQQ&dt=2026-02-06&speed=10

# Compute-only (save to parquet)
uv run python scripts/run_vacuum_pressure.py --symbol QQQ --dt 2026-02-06 --compute-only
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
