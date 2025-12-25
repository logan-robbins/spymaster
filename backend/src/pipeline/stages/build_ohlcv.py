"""Build OHLCV bars stage."""
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import (
    build_ohlcv_vectorized,
    compute_atr_vectorized,
    futures_trades_from_df,
)
from src.common.config import CONFIG


class BuildOHLCVStage(BaseStage):
    """Build OHLCV bars from trades using vectorized pandas.

    Can build different frequencies (1min, 2min) with optional warmup.

    Args:
        freq: Bar frequency ('1min', '2min')
        output_key: Key for context.data (defaults to 'ohlcv_{freq}')
        include_warmup: Whether to include SMA warmup bars (for 2min only)

    Outputs:
        {output_key}: pd.DataFrame with OHLCV columns
        atr: pd.Series (only for 1min bars)
    """

    def __init__(
        self,
        freq: str = '1min',
        output_key: str = None,
        include_warmup: bool = False
    ):
        self.freq = freq
        self.output_key = output_key or f'ohlcv_{freq}'
        self.include_warmup = include_warmup

    @property
    def name(self) -> str:
        return f"build_ohlcv_{self.freq}"

    @property
    def required_inputs(self) -> List[str]:
        return ['trades']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        trades = ctx.data['trades']

        # Build OHLCV bars
        ohlcv_df = build_ohlcv_vectorized(trades, convert_to_spy=True, freq=self.freq)

        result = {self.output_key: ohlcv_df}

        # Compute ATR only for 1min bars
        if self.freq == '1min':
            result['atr'] = compute_atr_vectorized(ohlcv_df)

        # Add warmup for 2min SMA calculation
        if self.include_warmup and self.freq == '2min':
            warmup_df, warmup_dates = self._build_warmup(ctx)
            if not warmup_df.empty:
                ohlcv_df = pd.concat([warmup_df, ohlcv_df], ignore_index=True)
                ohlcv_df = ohlcv_df.sort_values('timestamp')
                result[self.output_key] = ohlcv_df
                result['warmup_dates'] = warmup_dates

        return result

    def _build_warmup(self, ctx: StageContext):
        """Build warmup bars from prior dates for SMA calculation."""
        from datetime import datetime

        reader = ctx.data.get('_reader')
        if reader is None:
            from src.pipeline.utils.duckdb_reader import DuckDBReader
            reader = DuckDBReader()

        warmup_days = max(0, CONFIG.SMA_WARMUP_DAYS)
        if warmup_days == 0:
            return pd.DataFrame(), []

        warmup_dates = reader.get_warmup_dates(ctx.date, warmup_days)
        if not warmup_dates:
            return pd.DataFrame(), []

        frames = []
        for warmup_date in warmup_dates:
            trades_df = reader.read_futures_trades(symbol='ES', date=warmup_date)
            trades = futures_trades_from_df(trades_df)
            if not trades:
                continue
            ohlcv = build_ohlcv_vectorized(trades, convert_to_spy=True, freq=self.freq)
            if not ohlcv.empty:
                frames.append(ohlcv)

        if not frames:
            return pd.DataFrame(), warmup_dates

        warmup_df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
        return warmup_df, warmup_dates
