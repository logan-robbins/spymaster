# Market data pipeline (Bronze → Silver → Gold)

Stage-based data pipeline for market data organized by product type.

- **Raw**: Raw databento data in dbn format partitioned by day
- **Bronze**: decoded vendor data partitioned by source → product_type → symbol → table
- **Silver**: canonical transformation stages (UTC to EST timezone conversion)
- **Gold**: filtered data (first 3 hours of Regular Trading Hours 09:30–12:30 America/New_York)

Everything is driven by:
- dataset definitions in `config/datasets.yaml`
- Avro contracts in `contracts/{layer}/{product_type}/*.avsc`
- stages organized by `product_type` in `stages/{layer}/{product_type}/`

---

## Lake layout 

```
lake/
├── bronze/source=databento/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/hour=HH/
├── silver/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/hour=HH/
└── gold/product_type={future|future_option}/symbol={ES}/table={name}/dt=YYYY-MM-DD/hour=HH/
```

Each partition should contain:
-  `hour=HH/part-{}.parquet` (data)
- `_MANIFEST.json` (row count, schema hash, lineage)
- `_SUCCESS` (completion marker)

---

## Product Types

- **future**: Futures (MBP-10 data)
  - Format: DBN (raw), Parquet (bronze/silver/gold)
- **future_option**: Future Options (Trades, NBBO, Statistics)
  - Format:  DBN (raw), Parquet (bronze/silver/gold)
- **equity**: Equities (planned)
- **equity_option**: Equity Options (planned)

All symbols within a product_type are processed identically.

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
