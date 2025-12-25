"""Filter to regular trading hours stage."""
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


class FilterRTHStage(BaseStage):
    """Filter signals to regular trading hours with full forward window.

    Ensures all signals have complete forward window for labeling
    by restricting to 09:30-16:00 ET with buffer for confirmation.

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

        # Compute session bounds
        session_start = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        session_end = pd.Timestamp(ctx.date, tz="America/New_York") + pd.Timedelta(hours=16)
        session_start_ns = session_start.tz_convert("UTC").value
        session_end_ns = session_end.tz_convert("UTC").value

        # Compute max forward window needed
        max_confirm = max(
            CONFIG.CONFIRMATION_WINDOWS_MULTI or [CONFIG.CONFIRMATION_WINDOW_SECONDS]
        )
        max_window_ns = int((max_confirm + CONFIG.LOOKFORWARD_MINUTES * 60) * 1e9)
        latest_end_ns = signals_df["ts_ns"].astype("int64") + max_window_ns

        # Apply RTH filter
        rth_mask = (
            (signals_df["ts_ns"] >= session_start_ns) &
            (signals_df["ts_ns"] <= session_end_ns) &
            (latest_end_ns <= session_end_ns)
        )

        signals_df = signals_df.loc[rth_mask].copy()

        # Drop intermediate columns
        cols_to_drop = ['timestamp_dt', 'time_et', 'bar_idx']
        signals_df = signals_df.drop(
            columns=[c for c in cols_to_drop if c in signals_df.columns]
        )

        # Return as 'signals' for Pipeline.run() to extract
        return {'signals': signals_df}
