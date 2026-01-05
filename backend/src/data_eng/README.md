# PREFACE

We are currently iterating on our data pipeline. We have two types of pipeline:

1) In the *LEVEL* pipeline, ALL features in the output are relative and DIRECTLY related ONLY to the *LEVEL* we are interested in. The day trader is watching their chart and asking "is this price going to bounce/reject or break through this *LEVEL*". The *LEVEL* is explicity Pre-Market High/Low, Opening Range High/Low, SMA 90 (based on 2 min bars). 

2) in *MARKET* Pipeline, it is GENERAL market context based off of the MBP-10 data we have for Futures, and Trades+NBBO+Statistics data we have for the Futures Options. 

Eventually we may combine ALL feature vectors into a single vector, so it is criticial we do not duplicate/mix features in the final output for each *LEVEL* data pipeline, and prepend the *LEVEL*  name to each feature. The Market/global does not need a prefix, and it is important we dont duplicate features. 

The rules are: *WE ONLY USE* industry STANDARD terminology, we call the features EXACTLY WHAT THEY ARE, we name the stages EXACTLY WHAT WE ARE. We follow BEST PRACTICES for data pipelines as "stages"-- meanting we load -> transform -> write. every stage is atomic and idempotent. Every stage focuses on ONE concern. Every stage has a DEFINED input -> output contract.

# Market data pipeline (Bronze → Silver → Gold)

**Bronze**: DBN → Parquet, front-month contract only, filters spreads, extracts flag bits
**Silver**: Add ts_event_est (America/New_York)
**Gold**: Filter to 09:30-12:30 RTH using partition date

## Stage Pattern

All stages extend `Stage` base class with:
- Idempotency: checks `_SUCCESS` marker
- Atomic writes: tmp dir → replace
- Contract enforcement: Avro schema validation
- Lineage: manifest SHA256 tracking

## CLI

```bash
uv run python -m src.data_eng.runner \
  --product-type {future|future_option} \
  --layer {bronze|silver|gold|all} \
  --symbol {ES|ESM5} \
  --dt YYYY-MM-DD \
  [--dates YYYY-MM-DD,YYYY-MM-DD] \
  [--workers N]
```

**Bronze**: `--symbol ES` (prefix) → writes front-month contract (ESU5, ESM5, etc.)
**Silver/Gold**: `--symbol ESU5` (specific contract from Bronze output)

## Lake Structure

```
lake/
├── bronze/source=databento/product_type=future/symbol={contract}/table=market_by_price_10/dt=YYYY-MM-DD/
├── silver/product_type=future/symbol={contract}/table=market_by_price_10_clean/dt=YYYY-MM-DD/
└── gold/product_type=future/symbol={contract}/table=market_by_price_10_first3h/dt=YYYY-MM-DD/
```

Each partition: `part-00000.parquet`, `_MANIFEST.json`, `_SUCCESS`

## Reprocessing

Delete partition dir then rerun:
```bash
rm -rf lake/{layer}/.../dt=YYYY-MM-DD/
```
