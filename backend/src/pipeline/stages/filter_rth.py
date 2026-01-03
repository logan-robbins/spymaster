"""
Stage: Filter RTH (Final Silver Stage)
Type: Data Filtering & Schema Enforcement
Input: Signals DataFrame (Raw Features + Labels)
Output: Canonical Silver Signals (Filtered, Prefixed, Cleaned)

Transformation:
1. Filters events to the Training Window (08:30 - 12:30 ET).
   - Keeps Pre-Market (08:30-09:30) for context.
   - Keeps Morning Session (09:30-12:30) for primary interactions.
2. Removes "Global Features" (ATR, Spot, Time) to prevent duplication across multiple levels.
   - These are re-attached during Episode Construction (Gold Layer).
3. Applies Level-Specific Prefixes (e.g., `feature` -> `pm_high_feature`).
   - Ensures collision-free merging of multiple levels.
4. Writes the final Parquet file to the Silver Data Lake.
"""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG
from src.common.schemas.silver_features import validate_silver_features
from src.common.data_paths import canonical_signals_dir, date_partition

from src.pipeline.core.feature_definitions import IDENTITY_FEATURES, MARKET_STATE_FEATURES

logger = logging.getLogger(__name__)

# Identity columns (keep unprefixed - event metadata)
# Extend base identity with level-specific identity columns
IDENTITY_COLS: Set[str] = set(IDENTITY_FEATURES) | {
    'level_price', 'level_kind_name', 'direction', 'entry_price'
}

# Global features (remove to avoid duplication)
# Maps to MARKET_STATE_FEATURES from definitions
GLOBAL_FEATURES: Set[str] = set(MARKET_STATE_FEATURES)

# Note: Labels (outcome, excursion, etc.) ARE level-relative and WILL be prefixed
# A BREAK at PM_HIGH is different from a BREAK at OR_LOW


class FilterRTHStage(BaseStage):
    """
    Filter signals to training window: 08:30-12:30 ET (1hr premarket + 3hr RTH).
    
    Training window rationale:
    - 08:30-09:30 ET: Premarket hour captures PM_HIGH/PM_LOW formation and early touches
    - 09:30-12:30 ET: First 3 hours of RTH (most liquid, cleanest price action)
    
    Total: 4 hours of signal data per day.
    
    Also:
    - Removes global features (atr, spot, session timing) to avoid duplication across levels
    - Prefixes all level-relative features with level name (e.g., pm_high_ofi_60s)
    
    Outputs:
        signals: Final filtered DataFrame (key used by Pipeline.run)
    """

    @property
    def name(self) -> str:
        return "filter_rth"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()

        # Training window: 08:30-12:30 ET (1hr premarket + 3hr RTH)
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.TRAINING_START_HOUR, minutes=CONFIG.TRAINING_START_MINUTE
        )
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.TRAINING_END_HOUR, minutes=CONFIG.TRAINING_END_MINUTE
        )
        session_start_ns = session_start.tz_convert("UTC").value
        session_end_ns = session_end.tz_convert("UTC").value

        # Compute max forward window needed
        max_confirm = max(
            CONFIG.CONFIRMATION_WINDOWS_MULTI or [CONFIG.CONFIRMATION_WINDOW_SECONDS]
        )
        max_window_ns = int((max_confirm + CONFIG.LOOKFORWARD_MINUTES * 60) * 1e9)
        
        # Policy B: Keep anchors up to 12:30, allow forward spillover for labeling
        # This prevents label bias while respecting the training window constraint
        rth_mask = (
            (signals_df["ts_ns"] >= session_start_ns) &
            (signals_df["ts_ns"] <= session_end_ns)
        )
        
        # Note: We do NOT require (latest_end_ns <= session_end_ns) per Policy B
        # This allows labels to use data after 12:30 for events that occur at/before 12:30

        signals_df = signals_df.loc[rth_mask].copy()

        # Drop intermediate columns
        cols_to_drop = ['timestamp_dt', 'time_et', 'bar_idx']
        signals_df = signals_df.drop(
            columns=[c for c in cols_to_drop if c in signals_df.columns]
        )

        # === LEVEL-SPECIFIC OUTPUT PREPARATION ===
        level_name = ctx.level.lower()  # e.g., 'pm_high'
        
        # 1. Ensure level_kind_name is present
        if 'level_kind_name' not in signals_df.columns:
            signals_df['level_kind_name'] = level_name.upper()
            logger.info(f"  Added level_kind_name: {level_name.upper()}")
        
        # 2. Remove global features (computed once during merge)
        global_present = [c for c in GLOBAL_FEATURES if c in signals_df.columns]
        if global_present:
            signals_df = signals_df.drop(columns=global_present)
            logger.info(f"  Removed {len(global_present)} global features: {global_present}")
        
        # 3. Prefix ALL level-relative columns (features AND labels) with level name
        # Only identity columns stay unprefixed
        cols_to_prefix = [c for c in signals_df.columns if c not in IDENTITY_COLS]
        rename_map = {c: f'{level_name}_{c}' for c in cols_to_prefix}
        signals_df = signals_df.rename(columns=rename_map)
        logger.info(f"  Prefixed {len(cols_to_prefix)} columns (features + labels) with '{level_name}_'")

        # Validate against Silver schema (non-strict during development)
        if validate_silver_features(signals_df, strict=False):
            logger.info(f"  ✅ Schema validation passed: {len(signals_df.columns)} columns")
        else:
            logger.info(f"  ⚠️  Schema validation warning (non-strict): {len(signals_df.columns)} columns")

        # Optional: persist canonical Silver event table for this pipeline run
        signals_output_path: Path | None = None
        if ctx.config.get("PIPELINE_WRITE_SIGNALS"):
            data_root = ctx.config.get("DATA_ROOT")
            canonical_version = ctx.config.get("PIPELINE_CANONICAL_VERSION")
            if not data_root or not canonical_version:
                logger.warning("  Skipping signals write: missing DATA_ROOT or PIPELINE_CANONICAL_VERSION")
            else:
                base_dir = canonical_signals_dir(data_root, dataset="es_pipeline", version=canonical_version)
                level_dir = base_dir / date_partition(ctx.date) / ctx.level.lower()

                if ctx.config.get("PIPELINE_OVERWRITE_PARTITIONS", True) and level_dir.exists():
                    shutil.rmtree(level_dir)
                level_dir.mkdir(parents=True, exist_ok=True)

                signals_output_path = level_dir / "signals.parquet"
                signals_df.to_parquet(
                    signals_output_path,
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
                logger.info(f"  Wrote Silver signals: {signals_output_path}")
        
        # Return as both 'signals' (for Pipeline.run()) and 'signals_df' (for downstream stages)
        return {
            'signals': signals_df,
            'signals_df': signals_df,
            'signals_output_path': str(signals_output_path) if signals_output_path else None,
        }
