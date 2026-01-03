"""Filter to regular trading hours stage."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG
from src.common.schemas.silver_features import validate_silver_features
from src.common.data_paths import canonical_signals_dir, date_partition

logger = logging.getLogger(__name__)


class FilterRTHStage(BaseStage):
    """
    Filter signals to RTH only: first 3 hours (09:30-12:30 ET).
    
    Training window:
    - MBP-10 data includes RTH-1 (08:30-09:30 ET) for barrier context
    - Touch detection only during RTH: 09:30-12:30 ET (first 3 hours)
    
    Note: Touch detection (Stage 5) already filters to RTH, so this stage
    is now redundant but kept for schema validation and canonical Silver output.
    
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

        # Compute RTH bounds: first 3 hours only (09:30-12:30 ET)
        # Note: This should match the filtering done in detect_interaction_zones
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=12, minutes=30)
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

        # Validate against Silver schema
        try:
            validate_silver_features(signals_df)
            logger.info(f"  ✅ Schema validation passed: {len(signals_df.columns)} columns")
        except ValueError as e:
            logger.warning(f"  ⚠️  Schema validation failed: {e}")
            # Log but don't fail - this is informational during development

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
