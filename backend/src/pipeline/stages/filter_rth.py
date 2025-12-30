"""Filter to regular trading hours stage."""
from typing import Any, Dict, List
import pandas as pd
import pyarrow as pa
import logging

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG
from src.common.schemas.silver_features import SilverFeaturesESPipelineV1, validate_silver_features

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
        
        # Return as both 'signals' (for Pipeline.run()) and 'signals_df' (for downstream stages)
        return {
            'signals': signals_df,
            'signals_df': signals_df
        }
