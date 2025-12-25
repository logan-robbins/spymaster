"""Compute context features stage."""
from typing import Any, Dict, List
from datetime import time as dt_time
import uuid
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import compute_structural_distances


class ComputeContextFeaturesStage(BaseStage):
    """Add context features to signals.

    Adds:
    - is_first_15m: Whether signal is in first 15 minutes of session
    - date, symbol columns
    - direction_sign: +1 for UP, -1 for DOWN
    - event_id: Unique identifier
    - atr: ATR value at signal time
    - structural distances

    Outputs:
        signals_df: Updated with context features
    """

    @property
    def name(self) -> str:
        return "compute_context_features"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'atr', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df'].copy()
        atr_series = ctx.data['atr']
        ohlcv_df = ctx.data['ohlcv_1min']

        # Add is_first_15m
        signals_df['timestamp_dt'] = pd.to_datetime(signals_df['ts_ns'], unit='ns', utc=True)
        signals_df['time_et'] = signals_df['timestamp_dt'].dt.tz_convert('America/New_York').dt.time
        signals_df['is_first_15m'] = signals_df['time_et'].apply(
            lambda t: dt_time(9, 30) <= t < dt_time(9, 45)
        )

        # Add date and symbol
        signals_df['date'] = ctx.date
        signals_df['symbol'] = 'SPY'

        # Add direction sign
        signals_df['direction_sign'] = np.where(
            signals_df['direction'] == 'UP', 1, -1
        )

        # Generate event IDs
        signals_df['event_id'] = [str(uuid.uuid4()) for _ in range(len(signals_df))]

        # Attach ATR for normalization
        atr_values = atr_series.to_numpy()
        bar_idx_vals = signals_df['bar_idx'].values.astype(np.int64)
        if bar_idx_vals.max(initial=0) >= len(atr_values):
            raise ValueError("Bar index exceeds ATR series length.")
        signals_df['atr'] = atr_values[bar_idx_vals]

        # Compute structural distances
        signals_df = compute_structural_distances(signals_df, ohlcv_df)

        return {'signals_df': signals_df}
