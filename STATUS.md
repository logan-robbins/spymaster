# Data Pipeline Status and Flow

This document summarizes the key files in the data pipeline and what happens at each stage. It follows the Bronze -> Silver -> Gold flow used by VALIDATE.md.

## Stage 0: Source Data and Backfill
- `dbn-data/`: Databento DBN inputs for ES futures trades and MBP-10 depth snapshots.
- `backend/src/ingestor/polygon_flatfiles.py`: Loads SPY options flat files into Bronze.
- `backend/src/ingestor/polygon_historical.py`: Historical options ingestion helper for Bronze.
- `backend/scripts/backfill_bronze_futures.py`: Offline backfill of ES trades and MBP-10 into Bronze from DBN sources.

## Stage 1: Bronze Storage (Raw, Append-Only)
- `backend/src/lake/bronze_writer.py`: Writes NATS market events to Bronze Parquet with partitioning.
- `backend/src/lake/main.py`: Runs the Bronze and Gold writers as services.
- `backend/src/common/schemas/`: Canonical schemas for Bronze event types (futures trades, MBP-10, options trades).
- `backend/data/lake/bronze/`: Bronze storage layout (futures trades, MBP-10, options trades) partitioned by symbol/date/hour.

## Stage 1 Validation and Cleanup
- `backend/scripts/validate_bronze_integrity.py`: File inventory, row count sanity, duplicate sampling checks.
- `backend/scripts/filter_bronze_contracts.py`: Filters futures data in-place to the dominant ES contract per date.
- `backend/scripts/validate_bronze_stats.py`: Statistical checks (trade distribution, MBP-10 quality, options flow, time alignment, ES/SPY correlation).
- `backend/scripts/dedup_bronze_trades.py`: Optional dedup helper if duplicate rates are high.
- `VALIDATE.md`: Authoritative checklist and acceptance criteria for Bronze validation.

## Stage 2: Silver Compaction (Deduped, Sorted, Enriched)
- `backend/src/lake/silver_compactor.py`: Deduplicates and sorts Bronze to Silver; produces enriched options data and stable partitions.
- `backend/data/lake/silver/`: Silver storage location for compacted datasets.
Note: The vectorized training pipeline currently reads directly from Bronze; Silver is available for hygiene and downstream reuse.

## Stage 3: Gold Signal Generation (ML-Ready)
- `backend/src/pipeline/vectorized_pipeline.py`: Core offline pipeline that:
  - Loads Bronze futures trades, MBP-10, and options.
  - Builds OHLCV bars and level universe.
  - Computes barrier/tape/fuel physics metrics and engineered features.
  - Labels outcomes (break/bounce/chop), multi-timeframe targets, and tradeability.
  - Filters to regular session (09:30-16:00 ET) with full forward window.
  - Writes the consolidated research Parquet output.
- `backend/src/lake/gold_writer.py`: Streaming Gold writer for live `levels.signals` output.
- `backend/data/lake/gold/research/signals_vectorized.parquet`: Offline ML training dataset output.
- `backend/data/lake/gold/levels/signals/`: Streaming Gold output from live pipeline.

## Stage 4: Gold Validation (ML-Ready Checks)
- `backend/scripts/validate_data.py`: Validates schema, missingness, ranges, invariants, and label distributions for Gold.
- `backend/features.json`: Authoritative feature schema for ML training and validation.
- `backend/src/common/schemas/levels_signals.py`: Gold signal schema definition (levels + features + labels).

## Stage 5: ML Training and Evaluation
- `backend/src/ml/boosted_tree_train.py`: Trains tradeability, direction, strength, and time-to-threshold models.
- `backend/src/ml/build_retrieval_index.py`: Builds kNN retrieval index for similarity-based probabilities.
- `backend/src/ml/calibration_eval.py`: Evaluates calibration of trained models.
- `backend/src/ml/feature_sets.py`: Feature selection logic for training stages and ablations.
- `backend/src/ml/tracking.py`: MLflow and W&B logging hooks for experiment tracking.
- `backend/src/ml/README.md`: ML training plan, dataset expectations, and run recipes.

## Cross-Cutting Configuration and Contracts
- `backend/src/common/config.py`: Single source of truth for physics windows, bands, and thresholds.
- `backend/src/common/run_manifest_manager.py`: Tracks which Bronze and Gold files were produced in a run.
- `COMPONENTS.md` and `backend/src/common/INTERFACES.md`: Interfaces and contracts for data flow between services.
