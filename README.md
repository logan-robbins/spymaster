# Spymaster: Level Interaction Similarity Retrieval System

**IMPORTANT**
This is a paltform specifically built to visualize market/dealer physics in the first 3 hours of trading (when volume is the highest). The goal is not to predict price, but to retrieve similar "setups" and their labeled outcomes.
SETUP.png is a perfect example of what we are trying to model. 
Here is the core value prop we answer for the trader: "I am watching PM High at 6800. Price is approaching from below. I see call and long physics outpacing put/short physics at 6799 showing that people expect the price go above. At the same time, I see call/long physics at 6800 outpacing put/short physics. At the same time, I see call/long physics at 6801 outpacing put/short physics. BUT At 6802, I see MASSIVE put/short/resting limit sells. Represening both negative sentiment/positioning, and massive liquidity increasing that will make it tough for the price to go above 6802." WE answer THAT specific question- in both directions, for 4-5 key levels (not every singel point/strike). The exhaustive feature permutations in both directions are important for our model. THIS must be in the core of every line of code we write.


**System**: Retrieves historically similar market setups when price approaches technical levels, presenting empirical outcome distributions.

## Quick Facts

- **Data Source**: ES futures + ES 0DTE options (Databento GLBX.MDP3)
- **Levels**: 5 types (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90)
- **Outcomes**: BREAK/REJECT/CHOP 
- **Retrieval**: FAISS similarity search 
- **Pipeline**: Single-level architecture (run once per level type)

---

## Quick Start

**Current Data Range**: June 2 - Sept 30, 2025 (110 trading days)  
**Pipeline Range** (with 1-day warmup): June 4 - Sept 30, 2025 

### 1. Setup Environment

```bash
cd backend

# Set Databento API key
echo "DATABENTO_API_KEY=your_key_here" >> .env

# Install dependencies (uv manages all Python packages)
uv sync
```

### 2. Download & Prepare Data

```bash
cd backend

#  Step 1: Download ES Options DBN
uv run python scripts/download_es_options_fast.py \
  --start 2025-06-02 \
  --end 2025-09-30 \
  --workers 8

# Step 2: Convert DBN files to Bronze Parquet
uv run python scripts/convert_options_dbn_to_bronze.py --all

#  Check ES futures bronze data
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

Refer to backend/src/data_eng/README.md

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
**IMPORTANT**
This section is under construction and may need to be updated

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
