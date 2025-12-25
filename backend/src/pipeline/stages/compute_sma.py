"""Compute SMA-based mean reversion features stage (v2.0+)."""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def compute_mean_reversion_features(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    ohlcv_2min: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute SMA-based mean reversion features at each touch timestamp.

    Features:
        - sma_200 / sma_400 values at touch
        - dist_to_sma_200 / dist_to_sma_400 (spot - SMA)
        - sma_200_slope / sma_400_slope ($/min)
        - sma_spread (SMA-200 minus SMA-400)
        - mean_reversion_pressure_200 / 400 (distance normalized by volatility)
        - mean_reversion_velocity_200 / 400 (change in distance per minute)
    """
    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    df_2min = ohlcv_2min.copy() if ohlcv_2min is not None else None
    if df_2min is None:
        df_2min = ohlcv_df.set_index('timestamp').resample('2min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

    if df_2min.empty or 'timestamp' not in df_2min.columns:
        return signals_df

    df_2min = df_2min.sort_values('timestamp')
    sma_200_values = df_2min['close'].rolling(200).mean().to_numpy()
    sma_400_values = df_2min['close'].rolling(400).mean().to_numpy()
    sma_ts_ns = df_2min['timestamp'].values.astype('datetime64[ns]').astype(np.int64)

    # Prepare volatility series on 1-min bars
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts_ns = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    price_changes = np.diff(ohlcv_close, prepend=ohlcv_close[0])
    rolling_vol = pd.Series(price_changes).rolling(
        window=CONFIG.MEAN_REVERSION_VOL_WINDOW_MINUTES
    ).std().to_numpy()

    def _index_at_or_before(ts_ns: int, series_ts_ns: np.ndarray) -> int:
        return np.searchsorted(series_ts_ns, ts_ns, side='right') - 1

    def _value_at_or_before(ts_ns: int, series_ts_ns: np.ndarray, series_vals: np.ndarray) -> float:
        idx = _index_at_or_before(ts_ns, series_ts_ns)
        if idx < 0 or idx >= len(series_vals):
            return np.nan
        val = series_vals[idx]
        return val if np.isfinite(val) else np.nan

    n = len(signals_df)
    sma_200 = np.full(n, np.nan, dtype=np.float64)
    sma_400 = np.full(n, np.nan, dtype=np.float64)
    dist_to_sma_200 = np.full(n, np.nan, dtype=np.float64)
    dist_to_sma_400 = np.full(n, np.nan, dtype=np.float64)
    sma_200_slope = np.full(n, np.nan, dtype=np.float64)
    sma_400_slope = np.full(n, np.nan, dtype=np.float64)
    sma_200_slope_5bar = np.full(n, np.nan, dtype=np.float64)
    sma_400_slope_5bar = np.full(n, np.nan, dtype=np.float64)
    sma_spread = np.full(n, np.nan, dtype=np.float64)
    mean_rev_200 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_400 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_vel_200 = np.full(n, np.nan, dtype=np.float64)
    mean_rev_vel_400 = np.full(n, np.nan, dtype=np.float64)

    slope_minutes = max(1, CONFIG.SMA_SLOPE_WINDOW_MINUTES)
    slope_bars = max(1, int(round(slope_minutes / 2)))
    slope_short_bars = max(1, CONFIG.SMA_SLOPE_SHORT_BARS)
    slope_short_minutes = slope_short_bars * 2
    vel_minutes = max(1, CONFIG.MEAN_REVERSION_VELOCITY_WINDOW_MINUTES)
    vel_ns = int(vel_minutes * 60 * 1e9)

    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    level_prices = signals_df['level_price'].values.astype(np.float64)

    for i in range(n):
        ts = signal_ts[i]
        level_price = level_prices[i]

        sma200 = _value_at_or_before(ts, sma_ts_ns, sma_200_values)
        sma400 = _value_at_or_before(ts, sma_ts_ns, sma_400_values)

        sma_200[i] = sma200
        sma_400[i] = sma400

        if np.isfinite(sma200):
            dist_to_sma_200[i] = level_price - sma200
        if np.isfinite(sma400):
            dist_to_sma_400[i] = level_price - sma400

        # SMA slopes
        idx_now = _index_at_or_before(ts, sma_ts_ns)
        idx_prev = idx_now - slope_bars
        if idx_now >= 0 and idx_prev >= 0 and idx_now < len(sma_200_values):
            if np.isfinite(sma_200_values[idx_now]) and np.isfinite(sma_200_values[idx_prev]):
                sma_200_slope[i] = (sma_200_values[idx_now] - sma_200_values[idx_prev]) / slope_minutes
        if idx_now >= 0 and idx_prev >= 0 and idx_now < len(sma_400_values):
            if np.isfinite(sma_400_values[idx_now]) and np.isfinite(sma_400_values[idx_prev]):
                sma_400_slope[i] = (sma_400_values[idx_now] - sma_400_values[idx_prev]) / slope_minutes

        idx_prev_short = idx_now - slope_short_bars
        if idx_now >= 0 and idx_prev_short >= 0 and idx_now < len(sma_200_values):
            if np.isfinite(sma_200_values[idx_now]) and np.isfinite(sma_200_values[idx_prev_short]):
                sma_200_slope_5bar[i] = (sma_200_values[idx_now] - sma_200_values[idx_prev_short]) / slope_short_minutes
        if idx_now >= 0 and idx_prev_short >= 0 and idx_now < len(sma_400_values):
            if np.isfinite(sma_400_values[idx_now]) and np.isfinite(sma_400_values[idx_prev_short]):
                sma_400_slope_5bar[i] = (sma_400_values[idx_now] - sma_400_values[idx_prev_short]) / slope_short_minutes

        if np.isfinite(sma200) and np.isfinite(sma400):
            sma_spread[i] = sma200 - sma400

        vol = _value_at_or_before(ts, ohlcv_ts_ns, rolling_vol)
        if np.isfinite(vol) and vol > 0:
            if np.isfinite(dist_to_sma_200[i]):
                mean_rev_200[i] = dist_to_sma_200[i] / (vol + 1e-6)
            if np.isfinite(dist_to_sma_400[i]):
                mean_rev_400[i] = dist_to_sma_400[i] / (vol + 1e-6)

        # Mean reversion velocity
        prev_ts = ts - vel_ns
        prev_sma200 = _value_at_or_before(prev_ts, sma_ts_ns, sma_200_values)
        prev_sma400 = _value_at_or_before(prev_ts, sma_ts_ns, sma_400_values)

        if np.isfinite(prev_sma200) and np.isfinite(dist_to_sma_200[i]):
            prev_dist_200 = level_price - prev_sma200
            mean_rev_vel_200[i] = (dist_to_sma_200[i] - prev_dist_200) / vel_minutes
        if np.isfinite(prev_sma400) and np.isfinite(dist_to_sma_400[i]):
            prev_dist_400 = level_price - prev_sma400
            mean_rev_vel_400[i] = (dist_to_sma_400[i] - prev_dist_400) / vel_minutes

    result = signals_df.copy()
    result['sma_200'] = sma_200
    result['sma_400'] = sma_400
    result['dist_to_sma_200'] = dist_to_sma_200
    result['dist_to_sma_400'] = dist_to_sma_400
    result['sma_200_slope'] = sma_200_slope
    result['sma_400_slope'] = sma_400_slope
    result['sma_200_slope_5bar'] = sma_200_slope_5bar
    result['sma_400_slope_5bar'] = sma_400_slope_5bar
    result['sma_spread'] = sma_spread
    result['mean_reversion_pressure_200'] = mean_rev_200
    result['mean_reversion_pressure_400'] = mean_rev_400
    result['mean_reversion_velocity_200'] = mean_rev_vel_200
    result['mean_reversion_velocity_400'] = mean_rev_vel_400

    return result


class ComputeSMAFeaturesStage(BaseStage):
    """Compute SMA-based mean reversion features.

    This stage is used in v2.0+ pipelines to add SMA-200 and SMA-400
    distance features for confluence analysis.

    Outputs:
        signals_df: Updated with mean reversion features
    """

    @property
    def name(self) -> str:
        return "compute_sma_features"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']
        ohlcv_2min = ctx.data.get('ohlcv_2min')

        signals_df = compute_mean_reversion_features(
            signals_df, ohlcv_df, ohlcv_2min=ohlcv_2min
        )

        return {'signals_df': signals_df}
