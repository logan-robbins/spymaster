"""Build OHLCV bars stage."""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import FuturesTrade
from src.common.config import CONFIG


def build_ohlcv(
    trades: List[FuturesTrade],
    convert_to_spx: bool = True,
    freq: str = '1min'
) -> pd.DataFrame:
    """
    Build OHLCV bars using pandas operations optimized for M4 Silicon.

    Args:
        trades: List of FuturesTrade objects
        convert_to_spx: If True, keep ES prices as-is (ES ≈ SPX, same index points)
                        If False, keep raw ES prices
        freq: Bar frequency ('1min', '2min', '5min')

    Returns:
        DataFrame with OHLCV columns (in SPX index points if convert_to_spx=True)
    """
    if not trades:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Extract numpy arrays directly
    n = len(trades)
    ts_ns = np.empty(n, dtype=np.int64)
    prices = np.empty(n, dtype=np.float64)
    sizes = np.empty(n, dtype=np.int64)

    for i, trade in enumerate(trades):
        ts_ns[i] = trade.ts_event_ns
        prices[i] = trade.price
        sizes[i] = trade.size

    # Filter outliers using vectorized operations
    valid_mask = (prices > 3000) & (prices < 10000)
    ts_ns = ts_ns[valid_mask]
    prices = prices[valid_mask]
    sizes = sizes[valid_mask]

    if len(ts_ns) == 0:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Create DataFrame with numpy arrays
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(ts_ns, unit='ns', utc=True),
        'price': prices,
        'size': sizes
    })

    # Set index and resample (pandas optimized)
    df.set_index('timestamp', inplace=True)

    # Vectorized aggregation
    ohlcv = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    ohlcv.columns = ['open', 'high', 'low', 'close']
    ohlcv['volume'] = df['size'].resample(freq).sum()

    # Drop NaN bars
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv = ohlcv.reset_index()

    # ES prices are already in index points; no conversion needed.
    if convert_to_spx:
        # Keep as-is (ES prices are already in SPX-equivalent index points)
        # Small basis spread (1-5 points) handled by PriceConverter if needed
        pass

    return ohlcv


def compute_atr(
    ohlcv_df: pd.DataFrame,
    window_minutes: Optional[int] = None
) -> pd.Series:
    """Compute ATR on 1-minute bars for normalization."""
    if window_minutes is None:
        window_minutes = CONFIG.ATR_WINDOW_MINUTES
    if ohlcv_df.empty:
        return pd.Series(dtype=np.float64)

    df = ohlcv_df.sort_values('timestamp').copy()
    high = df['high'].astype(np.float64).to_numpy()
    low = df['low'].astype(np.float64).to_numpy()
    close = df['close'].astype(np.float64).to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=window_minutes, min_periods=1).mean().to_numpy()

    return pd.Series(atr, index=df.index)


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

        # Build OHLCV bars (ES → SPX, same index points)
        ohlcv_df = build_ohlcv(trades, convert_to_spx=True, freq=self.freq)

        result = {self.output_key: ohlcv_df}

        # Compute ATR only for 1min bars
        if self.freq == '1min':
            result['atr'] = compute_atr(ohlcv_df)

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
        from src.pipeline.stages.load_bronze import futures_trades_from_df

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
            ohlcv = build_ohlcv(trades, convert_to_spx=True, freq=self.freq)
            if not ohlcv.empty:
                frames.append(ohlcv)

        if not frames:
            return pd.DataFrame(), warmup_dates

        warmup_df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
        return warmup_df, warmup_dates
