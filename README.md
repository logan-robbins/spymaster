# Spymaster: Level Interaction Similarity Retrieval System

I am attempting to build a paltform specifically for market/dealer physics in the first 3 hours of trading (when volume is the highest). The goal is not to predict price (initially), but to retrieve similar "setups" and their labeled outcomes. I chose 6 specific levels: Pre-market high/low, 15 min opening range high/low, SMA_90 (2 min bars) and EMA_20 (2 min bars). a "setup" should be a vectorized snapshot of N lookback timeseries bars, where each bar represents the market state at that time-- RELATIVE to the specific "level" in question. The SETUP.png is a perfect example of what we are trying to model at each bar as the day progresses. The grey dotted lines are the pre-market high/low. The horizontal red and green lines are the 15 min open range high/low. The yellow/purple dotted lines are the SMA_90/EMA_20. There is clear reaction to these levels almost every single day. Sometimes the price breaks the levels, some times it does not. Sometimes it chops. We are trying to model and attribute the movements to various market physics forces. Then, we go a step further and say: at the 16 minute bar close, retrieve top 50 similar vectors and filter. What were their outcomes? the price break through the level or reject from the level. The hypothesis is that by looking at the approach over time (in granular windows computed within the large lookback window of say, 20 minutes), we can identify features that support break/reject probabilities. As a trader who follows these strict TA levels in the first 3 hours, the goal is that my platform UI begins retrieving/smoothing probabilistic signals based on the historical simliarity of the setup as the price approaches one of the levels we monitor. This can help me answer the question: if the price approaches the 15min opening range high from below, with f features, historically it has broken through with 77% probability. 



**System**: Retrieves historically similar market setups when price approaches technical levels, presenting empirical outcome distributions.

## Quick Facts

- **Data Source**: ES futures + ES 0DTE options (Databento GLBX.MDP3)
- **Levels**: 6 kinds (PM_HIGH/LOW, OR_HIGH/LOW, SMA_90/EMA_20)
- **Outcomes**: BREAK/REJECT/CHOP 
- **Episode Vectors**: N dimensions (Source) 
- **Retrieval**: FAISS similarity search 
- **Pipeline**: N stages (bronze → silver → gold → indices)

---

## Quick Start

**Current Data Range**: June 2 - Sept 30, 2025 (110 trading days)  
**Pipeline Range** (with 3-day warmup): June 5 - Sept 30, 2025 (107 days)

### 1. Setup Environment

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Install dependencies (uv manages all Python packages)
uv sync
```

### 2. Download & Prepare Data

The pipeline requires both **ES futures** and **ES options** in the Bronze layer:

| Instrument | Schemas Required | Bronze Location |
|------------|------------------|-----------------|
| **ES Futures** | trades, mbp-10 (order book) | `data/bronze/futures/{trades,mbp10}/symbol=ES/` |
| **ES Options** | trades, mbp-1 (NBBO), statistics | `data/bronze/options/{trades,nbbo,statistics}/underlying=ES/` |

**Data Flow Summary:**

```bash
cd backend

#ES OPTIONS:
#  Databento API 
#    → download_es_options_fast.py → data/raw/databento/options/
#    → convert_options_dbn_to_bronze.py → data/bronze/options/{trades,nbbo,statistics}/
#  Check ES options bronze data
ls data/bronze/options/trades/underlying=ES/ | head
ls data/bronze/options/nbbo/underlying=ES/ | head
ls data/bronze/options/statistics/underlying=ES/ | head

#  Step 1: Download ES Options DBN
uv run python scripts/download_es_options_fast.py \
  --start 2025-06-02 \
  --end 2025-09-30 \
  --workers 8

# Step 2: Convert DBN files to Bronze Parquet
uv run python scripts/convert_options_dbn_to_bronze.py --all

#ES FUTURES:
#  Databento API (or local DBN files)
#    → download_es_futures_fast.py (optional) → data/raw/databento/futures/{trades,MBP-10}/
#    → backfill_bronze_futures.py → data/bronze/futures/{trades,mbp10}/
#  Check ES futures bronze data
ls data/bronze/futures/trades/symbol=ES/ | head
ls data/bronze/futures/mbp10/symbol=ES/ | head

# Step 1: Download ES Futures DBN
uv run python scripts/download_es_futures_fast.py \
  --start 2025-06-02 \
  --end 2025-09-30 \
  --workers 4

# Step 2: Convert Futures DBN files to Bronze Parquet with backfill
  uv run python -m scripts.backfill_bronze_futures --all --workers 4

# Validate Bronze Data
uv run python scripts/validate_backfill.py --date 2025-06-20

```

### 3. Run Pipeline

#### 3a. Bronze → Silver (Feature Engineering)

```bash
cd backend

# Run Bronze→Silver pipeline  with 8 parallel workers
# Complete data range: June 5 - September 30, 2025 (107 days)
# Use nohup to run in background without blocking
nohup uv run python -m scripts.run_pipeline \
  --pipeline bronze_to_silver \
  --start 2025-06-05 \
  --end 2025-09-30 \
  --workers 8 \
  --write-outputs \
  --canonical-version 4.0.0 \
  > logs/{log_name}.out 2>&1 &

# Monitor progress
tail -f logs/{log_name}.out

# Check completion status
grep -E "✅|❌" logs/{log_name}.out | tail -20
```

#### 3b. Silver → Gold (Episode Construction)

```bash
cd backend

# Run Silver→Gold pipeline  with 8 parallel workers
nohup uv run python -m scripts.run_pipeline \
  --pipeline silver_to_gold \
  --start 2025-06-05 \
  --end 2025-09-30 \
  --workers 8 \
  --write-outputs \
  --canonical-version 4.0.0 \
  > logs/{log_name}.out 2>&1 &


uv run python scripts/compute_normalization_stats.py  

```

#### 3c. Validate Pipeline Output

```bash
cd backend

# Validate specific date
uv run python scripts/validate_es_pipeline.py --date 2025-06-20

uv run python scripts/validate_stage_14_label_outcomes.py --date 2025-09-15
uv run python scripts/validate_stage_16_materialize_state_table.py --date 2025-09-15
uv run python scripts/validate_stage_17_construct_episodes.py --date 2025-09-15

# Validate date range
uv run python scripts/validate_es_pipeline.py --start 2025-06-05 --end 2025-09-30
```

### 4. Build Retrieval System

```bash

# Build FAISS indices (60 partitions)
uv run python -c "
from pathlib import Path
from src.ml.index_builder import BuildIndicesStage

stage = BuildIndicesStage(
    episodes_dir=Path('data/gold/episodes/es_level_episodes/version=4.0.0'),
    output_dir=Path('data/gold/indices/es_level_indices/version=4.0.0')
)
result = stage.execute()
print(f'Built {result[\"n_partitions_built\"]} indices')
"
```

---

## Development

### Running Services (Real-Time Mode)

```bash
# Terminal 1: NATS infrastructure
docker-compose up nats -d

# Terminal 2: Ingestion (replay mode for testing)
cd backend
export REPLAY_DATE=2025-09-15
uv run python -m src.ingestion.databento.replay

# Terminal 3: Core service (with retrieval engine)
uv run python -m src.core.main

# Terminal 4: Gateway (WebSocket API)
uv run python -m src.gateway.main


```
