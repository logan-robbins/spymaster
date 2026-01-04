# Market data pipeline example (Bronze → Silver → Gold)

This repo is a **minimal but real** example of a stage-based data pipeline for Databento **MBP-10** data.

- **Bronze**: decoded vendor data (MBP-10) as partitioned files
- **Silver**: a single transformation stage that converts `ts_event` from UTC to Eastern Time (America/New_York) and writes a canonical table
- **Gold**: a single transformation stage that filters Silver to the first 3 hours of Regular Trading Hours (09:30–12:30 America/New_York)

Everything is driven by:
- dataset definitions in `config/datasets.yaml`
- Avro contracts in `contracts/*/*.avsc`

This example uses **CSV** to stay dependency-free. In production, switch the storage format to Parquet (same table layout / contracts).

---

## Lake layout (end state included)

The `lake/` directory in this zip contains a fully materialized example partition (`dt=2026-01-02`) for:

- `bronze/source=databento/table=market_by_price_10/dt=2026-01-02/`
- `silver/domain=futures/table=market_by_price_10_clean/dt=2026-01-02/`
- `gold/product=futures/table=market_by_price_10_first3h/dt=2026-01-02/`

Each partition contains:
- `part-00000.csv` (data)
- `_MANIFEST.json` (row count, schema hash, lineage)
- `_SUCCESS` (completion marker)

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run the two stages for a given date partition
python -m market_data.runner --dt 2026-01-02
```

After running, you will have:
- Silver: `lake/silver/domain=futures/table=market_by_price_10_clean/dt=2026-01-02/`
- Gold: `lake/gold/product=futures/table=market_by_price_10_first3h/dt=2026-01-02/`

---

## What the stages do

### Silver stage: `SilverConvertUtcToEst`
- Input dataset: `bronze.futures.market_by_price_10`
- Output dataset: `silver.futures.market_by_price_10_clean`
- Adds one field: `ts_event_est` (ISO-8601 string with America/New_York offset)

### Gold stage: `GoldFilterFirst3Hours`
- Input dataset: `silver.futures.market_by_price_10_clean`
- Output dataset: `gold.futures.market_by_price_10_first3h`
- Keeps rows where `ts_event_est` is between **09:30 (inclusive)** and **12:30 (exclusive)** in America/New_York

