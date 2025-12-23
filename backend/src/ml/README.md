# ML Module — PatchTST Training & Sequence Dataset Builder

**Module**: `backend/src/ml/`  
**Audience**: AI Coding Agents  
**Role**: Build sequence datasets from level-touch signals + OHLCV, then train PatchTST for BREAK/BOUNCE classification and strength regression.  
**Scope**: SPY 0DTE options only.

---

## Overview

This module turns the vectorized signal dataset into PatchTST-ready sequences and trains a multi-task model that outputs:
- `p_break` vs `p_bounce` (binary classification)
- `strength_signed` (regression)

It is designed to align with the feature contract in `backend/features.json` and to support MLflow + W&B tracking.

---

## Module Files

`backend/src/ml/` contains:
- `__init__.py`
- `sequence_dataset_builder.py` (builds `sequence_dataset_YYYY-MM-DD.npz`)
- `patchtst_train.py` (multi-task PatchTST trainer, MPS compatible)
- `README.md` (this file)

---

## Data Flow

1. **Signals Parquet** (from vectorized pipeline)
   - Default path from `features.json -> output_path`.
2. **Sequence Dataset Builder**
   - Reads signals for requested dates.
   - Builds 1-min OHLCV (from DBN trades) + 2-min SMA context.
   - Prepends up to `CONFIG.SMA_WARMUP_DAYS` prior weekday sessions for SMA warmup.
   - Emits `sequence_dataset_YYYY-MM-DD.npz`.
3. **PatchTST Trainer**
   - Loads `sequence_dataset_*.npz`.
   - Filters to BREAK/BOUNCE labels.
   - Trains PatchTST with classification + regression heads.
   - Logs to MLflow + W&B, saves best model checkpoint.

---

## Inputs and Outputs

### Inputs (Sequence Builder)
- **Signals parquet** with `event_id`, `ts_ns`, `date`, `direction`, targets, and numeric/bool features.
- **DBN trades** (ES) via `DBNIngestor` for OHLCV reconstruction.
- **CONFIG** values:
  - `MEAN_REVERSION_VOL_WINDOW_MINUTES`
  - `SMA_SLOPE_SHORT_BARS`
  - `SMA_WARMUP_DAYS`

### Outputs (Sequence Builder)
`sequence_dataset_YYYY-MM-DD.npz` containing:
- `X`: `(n_samples, seq_len, n_features)` sequence data
- `mask`: `(n_samples, seq_len)` padding mask
- `static`: `(n_samples, n_static_features)` static event features
- `y_break`: label mapping (BREAK=1, BOUNCE=0, CHOP/UNDEFINED=-1)
- `y_strength`: `strength_signed`
- `event_id`, `ts_ns`
- `seq_feature_names`, `static_feature_names`

### Outputs (Trainer)
`patchtst_multitask.pt` checkpoint with:
- model weights
- PatchTST config
- feature name lists

`run_metadata.json` with:
- schema version
- dataset hash
- train/val file list
- sample counts
- feature names

---

## Configuration Knobs

### Sequence Builder
- `SEQ_FEATURE_COLUMNS` (in `sequence_dataset_builder.py`): sequence channels.
- Sequence length and output paths are configured via CLI arguments (see `sequence_dataset_builder.py`).
- The default signals parquet path comes from `backend/features.json`.

### PatchTST Trainer
- Uses CLI arguments for data splits, hyperparameters, and loss weights (see `patchtst_train.py`).
- Requires a validation split (explicit dates or a ratio with at least two dates).

---

## Tracking (MLflow + W&B)

### MLflow
- Experiment name: `spymaster_patchtst` (override with `MLFLOW_EXPERIMENT_NAME`).
- Logs: params, metrics, `run_metadata.json`, `features.json`, model checkpoint.

### W&B
- Project name: `spymaster_patchtst` (override with `WANDB_PROJECT`).
- Requires `WANDB_API_KEY`, or set `WANDB_MODE=offline`.
- Logs: params, per-epoch metrics, model artifact with metadata.

---

## Entry Points (uv only)

Use `sequence_dataset_builder.py` to build datasets and `patchtst_train.py` to train PatchTST. Both scripts are intended to be executed with `uv run` and document their CLI arguments in-file.

---

## Failure Modes (Intentional)

- Missing `features.json` or `signals` parquet → hard failure.
- Missing DBN trades for date → hard failure.
- No matching dates for split → hard failure.
- Validation files missing → hard failure.
- Missing W&B configuration (unless offline) → hard failure.

---

## Notes for Extensions

- To add new sequence features, update `SEQ_FEATURE_COLUMNS` and ensure they exist in 1-min OHLCV.
- To add new static features, modify `_build_static_features` (currently numeric + bool + `direction_sign`).
- PatchTST is configured for MPS if available; falls back to CUDA/CPU.
- Schema version is read from `backend/features.json` and used in run naming.
- SMA warmup uses `CONFIG.SMA_WARMUP_DAYS` prior weekday sessions to reduce early-session NaNs.
