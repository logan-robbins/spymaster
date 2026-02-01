# Spymaster - LLM Ops Reference

## Critical Rules
- ALL CODE IS OLD: overwrite/delete/extend as needed
- Do not modify raw data unless explicitly instructed
- Use nohup + verbose logging for long-running commands, check every 15s
- Work backward from entry points to find current implementation
- If pipeline/features change: update avro contracts, datasets.yaml, {product}_data.json

## Product Types
- future_mbo, future_option_mbo, equity_mbo, equity_option_cmbp_1

## Symbol Convention Gotcha
- **Bronze layer**: Use parent symbol (ES)
- **Silver/Gold layers**: Use full contract (ESH6)
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
