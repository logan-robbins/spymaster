# Market data pipeline (Bronze → Silver → Gold)

Stage-based data pipeline for market data organized by product type.

- **Bronze**: decoded vendor data partitioned by source → product_type → symbol → table
- **Silver**: canonical transformation stages (UTC to EST timezone conversion)
- **Gold**: filtered data (first 3 hours of Regular Trading Hours 09:30–12:30 America/New_York)

Everything is driven by:
- dataset definitions in `config/datasets.yaml`
- Avro contracts in `contracts/{product_type}/*.avsc`
- stages organized by `product_type` in `stages/{product_type}/`

Uses **CSV** for futures demo. **Parquet** for options (production data).

---

## Lake layout 

```
lake/
├── bronze/source=databento/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/
├── silver/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/
└── gold/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/
```

Each partition contains:
- `part-00000.csv` or `hour=HH/*.parquet` (data)
- `_MANIFEST.json` (row count, schema hash, lineage)
- `_SUCCESS` (completion marker)

---

## Product Types

- **future**: Futures (MBP-10 data)
  - Tables: `market_by_price_10`
  - Format: CSV (demo), Parquet (production)
- **future_option**: Future Options (Trades, NBBO, Statistics)
  - Tables: `trades`, `nbbo`, `statistics`
  - Format: Parquet with hour-level partitioning
  - Data migrated from `data/bronze/options/`
- **equity**: Equities (planned)
- **equity_option**: Equity Options (planned)

---

## Quickstart

```bash
# Run the pipeline for ES futures on a specific date
uv run python -m src.data_eng.runner \
  --product-type future \
  --symbol ES \
  --dt 2026-01-02 \
  --config src/data_eng/config/datasets.yaml

# Run the pipeline for ES future options
uv run python -m src.data_eng.runner \
  --product-type future_option \
  --symbol ES \
  --dt 2026-01-02 \
  --config src/data_eng/config/datasets.yaml
```

After running, you will have:
- Silver: `lake/silver/product_type={type}/symbol={symbol}/table={name}_clean/dt={dt}/`
- Gold: `lake/gold/product_type={type}/symbol={symbol}/table={name}_first3h/dt={dt}/`

---

## What the stages do

### Future (MBP-10)

#### Silver stage: `SilverConvertUtcToEst`
- Input: `bronze.future.market_by_price_10`
- Output: `silver.future.market_by_price_10_clean`
- Adds one field: `ts_event_est` (ISO-8601 string with America/New_York offset)

#### Gold stage: `GoldFilterFirst3Hours`
- Input: `silver.future.market_by_price_10_clean`
- Output: `gold.future.market_by_price_10_first3h`
- Keeps rows where `ts_event_est` is between **09:30 (inclusive)** and **12:30 (exclusive)** in America/New_York

### Future Option (Trades, NBBO, Statistics)

#### Silver stage: `SilverConvertUtcToEst`
- Input: `bronze.future_option.{trades,nbbo,statistics}`
- Output: `silver.future_option.{trades,nbbo,statistics}_clean`
- Converts `ts_event_ns` (nanoseconds) to `ts_event_est` (ISO-8601 string with America/New_York offset)
- Bronze data format: Parquet with hour-level partitioning (`dt=YYYY-MM-DD/hour=HH/`)

#### Gold stage: `GoldFilterFirst3Hours`
- Input: `silver.future_option.{trades,nbbo,statistics}_clean`
- Output: `gold.future_option.{trades,nbbo,statistics}_first3h`
- Keeps rows where `ts_event_est` is between **09:30 (inclusive)** and **12:30 (exclusive)** in America/New_York

**Note**: Existing bronze data at `data/bronze/options/` has been migrated to `lake/bronze/source=databento/product_type=future_option/`

---

## Architecture

### Contracts
Organized by product_type and layer:
```
contracts/
├── bronze/{product_type}/{table}.avsc
├── silver/{product_type}/{table}_clean.avsc
└── gold/{product_type}/{table}_first3h.avsc
```

### Stages
Organized by product_type:
```
stages/
├── base.py (Stage base class)
├── future/
│   ├── silver_convert_utc_to_est.py
│   └── gold_filter_first3h.py
└── future_option/
    ├── silver_convert_utc_to_est.py
    └── gold_filter_first3h.py
```

### Pipeline
`pipeline.py` selects stages based on `--product-type`:
```python
def build_pipeline(product_type: str) -> List[Stage]:
    if product_type == "future":
        from .stages.future import ...
    elif product_type == "future_option":
        from .stages.future_option import ...
```

All symbols within a product_type are processed identically.

---

## Multiple Pipelines

data_eng handles the medallion architecture (Bronze → Silver → Gold). 

For feature engineering pipelines (ES Level Pipeline, ES Market/Global Pipeline), those live in `src/pipelines/` and consume the Gold layer from data_eng.
