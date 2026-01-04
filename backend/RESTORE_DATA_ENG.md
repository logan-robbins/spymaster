# Data_Eng Restoration Checklist

## What Was Built (Before My Mistake)

### 1. Lake Structure Reorganization
- Reorganized from flat structure to: `source=databento/product_type={type}/symbol={sym}/table={tbl}/dt={date}/`
- Migrated 8,546 options parquet files (trades, nbbo, statistics)
- 377 date partitions migrated
- Preserved hour-level partitioning for options

### 2. Contracts Created (All Need Recreation)

**Bronze Future Option:**
- trades.avsc (14 fields: ts_event_ns, ts_recv_ns, source, underlying, option_symbol, exp_date, strike, right, price, size, opt_bid, opt_ask, seq, aggressor)
- nbbo.avsc (13 fields: ts_event_ns, ts_recv_ns, source, underlying, option_symbol, exp_date, strike, right, bid_px, ask_px, bid_sz, ask_sz, seq)
- statistics.avsc (9 fields: ts_event_ns, ts_recv_ns, source, underlying, option_symbol, exp_date, strike, right, open_interest)

**Silver Future Option:**
- trades_clean.avsc (adds ts_event_est)
- nbbo_clean.avsc (adds ts_event_est)
- statistics_clean.avsc (adds ts_event_est)

**Gold Future Option:**
- trades_first3h.avsc
- nbbo_first3h.avsc
- statistics_first3h.avsc

**Bronze/Silver/Gold Future:**
- market_by_price_10.avsc (original from example)
- market_by_price_10_clean.avsc
- market_by_price_10_first3h.avsc

### 3. Stages by Product Type

**stages/future/**
- `__init__.py`
- `silver_convert_utc_to_est.py` (bronze.future.market_by_price_10 → silver.future.market_by_price_10_clean)
- `gold_filter_first3h.py` (silver → gold, 09:30-12:30 NY filter)

**stages/future_option/**
- `__init__.py`
- `silver_convert_utc_to_est.py` (works with trades/nbbo/statistics)
- `gold_filter_first3h.py`

### 4. Core Files Updated

**config.py** - No changes needed (already flexible)

**io.py** - Updated:
- `partition_ref()` now takes `symbol` parameter
- Path templating with `{symbol}` placeholders
- Simplified partition path building

**pipeline.py** - Updated:
- `build_pipeline(product_type)` router
- Returns stages based on product_type

**runner.py** - Updated:
- Added `--product-type` and `--symbol` arguments
- Passes symbol to stages

**config/datasets.yaml** - Complete rewrite with:
- bronze.future.market_by_price_10
- bronze.future_option.{trades,nbbo,statistics}
- silver.future.market_by_price_10_clean
- silver.future_option.{trades,nbbo,statistics}_clean  
- gold.future.market_by_price_10_first3h
- gold.future_option.{trades,nbbo,statistics}_first3h

### 5. Migration Script
- `scripts/migrate_options_bronze_to_lake.py` - Successfully ran, moved all data

## Status

- ✅ Lake data intact (verified at ../../lake/)
- ✅ Migration complete (8,546 files moved)
- ❌ All Python files deleted by mistake
- ❌ All contracts deleted
- ❌ datasets.yaml deleted

## Recovery Plan

Since user provided backup at market-data-example-real/, need to:
1. Use base files from example
2. Recreate all custom contracts (14 total .avsc files)
3. Recreate stages for future_option
4. Update io.py for symbol parameter
5. Update pipeline.py for product_type routing
6. Update runner.py for new args
7. Recreate datasets.yaml with all datasets
8. Update stages/base.py for symbol parameter

