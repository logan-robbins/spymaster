# Market data pipeline (Bronze → Silver → Gold)

Stage-based data pipeline for market data organized by product type.

- **Bronze**: DBN → Parquet (minimal transformations: drop ts_in_delta, extract flag bits, filter spreads)
- **Silver**: UTC to EST timezone conversion
- **Gold**: first 3 hours of Regular Trading Hours (09:30–12:30 America/New_York)

Configuration:
- Dataset definitions: `config/datasets.yaml`
- Avro contracts: `contracts/{layer}/{product_type}/*.avsc`
- Stages: `stages/{layer}/{product_type}/`
- Raw DBN files: `lake/raw/source=databento/product_type={future|future_option}/symbol={symbol}/table={name}/`

---

## Lake Layout

```
lake/
├── bronze/source=databento/product_type={future|future_option}/symbol={symbol}/table={name}/dt=YYYY-MM-DD/
├── silver/product_type={future|future_option}/symbol={symbol}/table={name}/dt=YYYY-MM-DD/
└── gold/product_type={future|future_option}/symbol={symbol}/table={name}/dt=YYYY-MM-DD/
```

Each partition (`dt=YYYY-MM-DD/`) contains:
- `part-00000.parquet` (data file)
- `_MANIFEST.json` (row count, contract hash, lineage)
- `_SUCCESS` (completion marker)

Hour-level partitioning (`hour=HH/`) is supported but not currently used.

---

## Product Types

**future** - Futures (MBP-10 order book data)
- Format: Parquet
- Tables: `market_by_price_10`

**future_option** - Future Options (ES options)
- Format: Parquet
- Tables: `trades`, `nbbo`, `statistics`

**equity**, **equity_option** - Planned

---

## Key Concepts

**Atomic Partition Replacement**: Stages write to tmp dir → atomically replace entire partition. Each partition has exactly one data file after stage completion.

**Contract Enforcement**: Every stage input/output is validated against Avro schema contracts to ensure column names and order match exactly.

**Idempotency**: Stages check for `_SUCCESS` marker. If partition already complete, stage skips. Safe to re-run.

---

## Quickstart

```bash
cd backend

# Process DBN files to Bronze (vectorized for M4 Silicon)
uv run python -c "
from pathlib import Path
from src.data_eng.config import load_config
from src.data_eng.stages.bronze.future import process_mbp10_to_bronze

repo_root = Path('.')
cfg = load_config(repo_root, repo_root / 'src/data_eng/config/datasets.yaml')
process_mbp10_to_bronze(cfg, repo_root, 'ES', '2025-06-05')
"

# Run Silver → Gold pipeline
uv run python -m src.data_eng.runner \
  --product-type future \
  --symbol ESM6 \
  --dt 2025-06-05 \
  --config src/data_eng/config/datasets.yaml
```
