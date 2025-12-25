"""Compute confluence features stage (v2.0+)."""
from typing import Any, Dict, List
import pandas as pd

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import (
    compute_confluence_features_dynamic,
    compute_confluence_alignment,
    compute_dealer_velocity_features,
    compute_pressure_indicators,
    compute_confluence_level_features,
)
from src.pipeline.utils.duckdb_reader import DuckDBReader
from src.common.config import CONFIG
import numpy as np


class ComputeConfluenceStage(BaseStage):
    """Compute confluence and pressure features.

    This stage is used in v2.0+ pipelines to add:
    - Stacked key level confluence
    - Confluence alignment with direction
    - Dealer mechanics velocity
    - Fluid pressure indicators
    - Hierarchical confluence level

    Outputs:
        signals_df: Updated with confluence features
    """

    @property
    def name(self) -> str:
        return "compute_confluence"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'dynamic_levels', 'option_trades_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        dynamic_levels = ctx.data['dynamic_levels']
        option_trades_df = ctx.data['option_trades_df']
        ohlcv_df = ctx.data['ohlcv_1min']

        # Confluence features (stacked key levels)
        signals_df = compute_confluence_features_dynamic(signals_df, dynamic_levels)
        signals_df = compute_confluence_alignment(signals_df)

        # Dealer mechanics velocity features
        signals_df = compute_dealer_velocity_features(signals_df, option_trades_df)

        # Fluid pressure indicators
        signals_df = compute_pressure_indicators(signals_df)

        # Gamma bucket classification
        gamma_exposure = signals_df.get('gamma_exposure')
        if gamma_exposure is not None:
            gamma_vals = gamma_exposure.values.astype(np.float64)
            gamma_bucket = np.where(
                np.isfinite(gamma_vals),
                np.where(gamma_vals < 0, "SHORT_GAMMA", "LONG_GAMMA"),
                "UNKNOWN"
            )
            signals_df['gamma_bucket'] = gamma_bucket

        # Build hourly cumulative volume for hierarchical confluence
        hourly_cumvol = self._build_hourly_cumvol_table(ctx, ohlcv_df)

        # Convert dynamic_levels dict to DataFrame
        dynamic_levels_df = pd.DataFrame(dynamic_levels)
        dynamic_levels_df['timestamp'] = ohlcv_df['timestamp'].values

        signals_df = compute_confluence_level_features(
            signals_df, dynamic_levels_df, hourly_cumvol, ctx.date
        )

        return {'signals_df': signals_df}

    def _build_hourly_cumvol_table(
        self, ctx: StageContext, ohlcv_df: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        """Build hourly cumulative volume table for relative volume."""
        from datetime import time as dt_time

        hourly_cumvol: Dict[str, Dict[int, float]] = {}

        def _compute_hourly_cumvol(ohlcv: pd.DataFrame, date_str: str) -> Dict[int, float]:
            if ohlcv.empty or 'timestamp' not in ohlcv.columns:
                return {}

            df = ohlcv.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York')
            df['hour_et'] = df['time_et'].dt.hour

            # Filter to RTH (9:30-16:00)
            rth_mask = (
                (df['time_et'].dt.time >= dt_time(9, 30)) &
                (df['time_et'].dt.time < dt_time(16, 0))
            )
            df = df[rth_mask]

            if df.empty:
                return {}

            # Compute cumulative volume up to end of each hour
            hourly = {}
            for hour in [9, 10, 11, 12, 13, 14, 15]:
                if hour == 9:
                    hour_end = dt_time(9, 59, 59)
                else:
                    hour_end = dt_time(hour, 59, 59)

                mask = df['time_et'].dt.time <= hour_end
                if mask.any():
                    hourly[hour] = df.loc[mask, 'volume'].sum()

            return hourly

        # Get prior dates for lookback
        reader = ctx.data.get('_reader')
        if reader is None:
            reader = DuckDBReader()

        warmup_days = max(0, CONFIG.VOLUME_LOOKBACK_DAYS)
        prior_dates = reader.get_warmup_dates(ctx.date, warmup_days) if warmup_days > 0 else []

        from src.pipeline.utils.vectorized_ops import (
            futures_trades_from_df,
            build_ohlcv_vectorized,
        )

        for prior_date in prior_dates:
            trades_df = reader.read_futures_trades(symbol='ES', date=prior_date)
            trades = futures_trades_from_df(trades_df)
            if not trades:
                continue
            ohlcv = build_ohlcv_vectorized(trades, convert_to_spy=True, freq='1min')
            if not ohlcv.empty:
                hourly_cumvol[prior_date] = _compute_hourly_cumvol(ohlcv, prior_date)

        # Add current date
        if not ohlcv_df.empty:
            hourly_cumvol[ctx.date] = _compute_hourly_cumvol(ohlcv_df, ctx.date)

        return hourly_cumvol
