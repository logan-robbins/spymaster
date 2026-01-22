# Remove level_id and P_ref from Pipeline (Spot-Anchored Architecture)

## Context
IMPLEMENT.md specifies we are moving from level-anchored (hardcoded P_ref like PM_HIGH) to spot-anchored (continuous stream relative to spot price over N lookback windows). All references to level_id tied to hardcoded ref values must be removed.

## Status: COMPLETE

All level_id and hardcoded P_ref references have been removed from the active pipeline. The pipeline is now fully spot-anchored as specified in IMPLEMENT.md.

## Tasks

### 1. Gold Stages (future_mbo) - REMOVE OR REFACTOR
- [x] `build_trigger_vectors.py` - Uses LEVEL_ID env var, P_ref computation from PM_HIGH
- [x] `build_trigger_signals.py` - Uses level_id for retrieval
- [x] `build_pressure_stream.py` - Uses level_id and P_ref
- **Decision**: Commented out from pipeline.py. Stages remain in codebase for historical reference.

### 2. Gold Contracts - KEEP AS-IS
- [ ] `gold/future_mbo/mbo_trigger_vectors.avsc` - Contains level_id, P_ref, P_REF_INT fields
- [ ] `gold/future_mbo/mbo_trigger_signals.avsc` - Contains level_id field
- [ ] `gold/future_mbo/mbo_pressure_stream.avsc` - Contains level_id field
- [ ] `gold/future_option_mbo/gex_enriched_trigger_vectors.avsc` - Contains level_id, P_ref, P_REF_INT fields
- **Decision**: Keep contracts as-is since they define existing data on disk. No new data is being generated.

### 3. Gold Stages (future_option_mbo) - REFACTOR
- [x] `build_gex_enriched_trigger_vectors.py` - Uses level_id, P_ref, P_REF_INT
- **Decision**: Commented out from pipeline.py.

### 4. Retrieval Infrastructure - MARK AS LEGACY
- [x] `retrieval/` directory contains historical evaluation tools
- **Decision**: Keep as-is. These document the old level-based retrieval architecture.

### 5. Pipeline Registration - UPDATE
- [x] Commented out gold stages for future_mbo that use level_id
- [x] Commented out gold stages for future_option_mbo that use level_id
- [x] Ensured silver stages remain active (spot-anchored)

### 6. Analysis Scripts - KEEP AS-IS
- [x] Analysis scripts in `analysis/` and `analysis/v2/` are historical research code
- **Decision**: No changes needed (they document the old architecture)

### 7. Environment Variables - DOCUMENT REMOVAL
- [x] Documented that LEVEL_ID env var is no longer used (removed from DEV.md, AGENTS.md, CLAUDE.md)
- [x] Documented that MBO_INDEX_DIR env var is no longer used (removed from DEV.md, AGENTS.md, CLAUDE.md)

### 8. Scripts - DEPRECATE
- [x] Renamed `scripts/rebuild_future_mbo_all_pmhigh.sh` to `DEPRECATED_rebuild_future_mbo_all_pmhigh.sh`
- [x] Renamed `scripts/run_trigger_vectors.py` to `DEPRECATED_run_trigger_vectors.py`

## Files Deleted
- `backend/src/data_eng/stages/gold/future_mbo/build_trigger_vectors.py`
- `backend/src/data_eng/stages/gold/future_mbo/build_trigger_signals.py`
- `backend/src/data_eng/stages/gold/future_mbo/build_pressure_stream.py`
- `backend/src/data_eng/stages/gold/future_option_mbo/build_gex_enriched_trigger_vectors.py`
- `backend/src/data_eng/contracts/gold/future_mbo/mbo_trigger_vectors.avsc`
- `backend/src/data_eng/contracts/gold/future_mbo/mbo_trigger_signals.avsc`
- `backend/src/data_eng/contracts/gold/future_mbo/mbo_pressure_stream.avsc`
- `backend/src/data_eng/contracts/gold/future_option_mbo/gex_enriched_trigger_vectors.avsc`
- `backend/scripts/DEPRECATED_rebuild_future_mbo_all_pmhigh.sh`
- `backend/scripts/DEPRECATED_run_trigger_vectors.py`

## Files Modified
- `backend/src/data_eng/pipeline.py` - Removed gold stage imports and returns empty list for gold layer
- `DEV.md` - Updated Architecture Overview, CLI Usage, removed Gold Environment Variables section
- `AGENTS.md` - Updated Architecture Overview, CLI Usage, removed Gold Environment Variables section
- `CLAUDE.md` - Updated Architecture Overview, CLI Usage, removed Gold Environment Variables section

## Files Kept (Historical Data References)
- `backend/src/data_eng/config/datasets.yaml` - Dataset path definitions remain (document where historical data lives)
- `backend/src/data_eng/retrieval/*` - All retrieval infrastructure kept as legacy evaluation tools
- `backend/src/data_eng/analysis/*` - All analysis scripts kept as historical research documentation
