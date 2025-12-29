"""Compute context features stage."""
from typing import Any, Dict, List
from datetime import time as dt_time
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext


def compute_structural_distances(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute signed distances to structural levels (pre-market high/low) relative to target level."""
    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    df = ohlcv_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time

    pm_mask = (df['time_et'] >= dt_time(4, 0)) & (df['time_et'] < dt_time(9, 30))

    pm_high = np.nan
    pm_low = np.nan
    if pm_mask.any():
        pm_data = df[pm_mask]
        pm_high = pm_data['high'].max()
        pm_low = pm_data['low'].min()

    level_prices = signals_df['level_price'].values.astype(np.float64)
    dist_to_pm_high = np.full(len(signals_df), np.nan, dtype=np.float64)
    dist_to_pm_low = np.full(len(signals_df), np.nan, dtype=np.float64)

    if np.isfinite(pm_high):
        dist_to_pm_high = level_prices - pm_high
    if np.isfinite(pm_low):
        dist_to_pm_low = level_prices - pm_low

    result = signals_df.copy()
    result['dist_to_pm_high'] = dist_to_pm_high
    result['dist_to_pm_low'] = dist_to_pm_low

    return result


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
        signals_df['symbol'] = 'ES'

        # Add direction sign
        signals_df['direction_sign'] = np.where(
            signals_df['direction'] == 'UP', 1, -1
        )

        # Generate deterministic event IDs
        # Format: {date}_{level_kind_name}_{level_price}_{ts_ns}_{direction}
        # Reproducible for retrieval across runs
        if 'event_id' not in signals_df.columns:
            signals_df['event_id'] = (
                signals_df['date'].astype(str) + '_' +
                signals_df['level_kind_name'].astype(str) + '_' +
                signals_df['level_price'].astype(str) + '_' +
                signals_df['ts_ns'].astype(str) + '_' +
                signals_df['direction'].astype(str)
            )

        # Attach ATR for normalization
        atr_values = atr_series.to_numpy()
        bar_idx_vals = signals_df['bar_idx'].values.astype(np.int64)
        if bar_idx_vals.max(initial=0) >= len(atr_values):
            raise ValueError("Bar index exceeds ATR series length.")
        signals_df['atr'] = atr_values[bar_idx_vals]

        # Compute structural distances
        signals_df = compute_structural_distances(signals_df, ohlcv_df)

        return {'signals_df': signals_df}
