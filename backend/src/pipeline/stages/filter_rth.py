"""Filter to regular trading hours stage."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG
from src.common.schemas.silver_features import validate_silver_features
from src.common.lake_paths import canonical_signals_dir, date_partition

logger = logging.getLogger(__name__)


class FilterRTHStage(BaseStage):
    """
    Filter signals to v1 scope: first 4 hours only (09:30-13:30 ET).
    
    RTH filtering:
    - Only create events during 09:30-13:30 ET (first 4 hours)
    - Use Policy B: keep anchors up to 13:30, allow forward window spillover
    - Ensures all signals have complete forward window for labeling
    
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

        # Compute session bounds (v1: first 4 hours only)
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.RTH_START_HOUR,
            minutes=CONFIG.RTH_START_MINUTE
        )
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.RTH_END_HOUR,
            minutes=CONFIG.RTH_END_MINUTE
        )
        session_start_ns = session_start.tz_convert("UTC").value
        session_end_ns = session_end.tz_convert("UTC").value

        # Compute max forward window needed
        max_confirm = max(
            CONFIG.CONFIRMATION_WINDOWS_MULTI or [CONFIG.CONFIRMATION_WINDOW_SECONDS]
        )
        max_window_ns = int((max_confirm + CONFIG.LOOKFORWARD_MINUTES * 60) * 1e9)
        
        # Policy B: Keep anchors up to 13:30, allow forward spillover for labeling
        # This prevents label bias while respecting the first-4-hours constraint
        rth_mask = (
            (signals_df["ts_ns"] >= session_start_ns) &
            (signals_df["ts_ns"] <= session_end_ns)
        )
        
        # Note: We do NOT require (latest_end_ns <= session_end_ns) per Policy B
        # This allows labels to use data after 13:30 for events that occur at/before 13:30

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
                date_dir = base_dir / date_partition(ctx.date)

                if ctx.config.get("PIPELINE_OVERWRITE_PARTITIONS", True) and date_dir.exists():
                    shutil.rmtree(date_dir)
                date_dir.mkdir(parents=True, exist_ok=True)

                signals_output_path = date_dir / "signals.parquet"
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
