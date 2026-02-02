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
    --options-schemas definition,mbo,statistics \
    --poll-interval 60 \
    --log-file logs/futures.log > logs/futures_daemon.out 2>&1 &
```

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

### Contract Selection Map
```bash
cd backend
uv run python -m src.data_eng.mbo_contract_day_selector --output-path lake/selection/mbo_contract_day_selection.parquet
```
- Maps session_date → front-month contract
- Built from `bronze/source=databento/product_type=future_mbo`
- Filters for dates with premarket trades (05:00-08:30 ET)
- Required before silver/gold when using base symbol ES

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
uv run python scripts/validate_silver_future_mbo.py
uv run python scripts/validate_silver_equity_mbo.py
uv run python scripts/validate_silver_future_option_mbo.py --dt YYYY-MM-DD
uv run python scripts/validate_silver_equity_option_cmbp_1.py
uv run python scripts/grunt_validate_future_option_mbo.py
```

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
