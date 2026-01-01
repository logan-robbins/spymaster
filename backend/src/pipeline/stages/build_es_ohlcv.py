"""
Build ES OHLCV series from ES futures trades.

ES futures are quoted in index points.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.event_types import FuturesTrade
from src.common.config import CONFIG
from src.common.utils.session_time import filter_rth_only, filter_premarket_only


def build_es_ohlcv_from_es(
    trades: List[FuturesTrade],
    date: str,
    freq: str = '1min',
    rth_only: bool = False
) -> pd.DataFrame:
    """
    Build ES OHLCV from ES futures trades.

    ES futures are quoted in index points.

    Args:
        trades: List of ES FuturesTrade objects
        date: Date string for RTH filtering
        freq: Bar frequency ('1min', '2min')
        rth_only: If True, only include RTH bars (09:30-13:30 ET for v1)

    Returns:
        DataFrame with OHLCV columns in ES index points
    """
    if not trades:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'], 
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    # Extract arrays
    n = len(trades)
    ts_ns = np.empty(n, dtype=np.int64)
    prices = np.empty(n, dtype=np.float64)
    sizes = np.empty(n, dtype=np.int64)
    
    for i, trade in enumerate(trades):
        ts_ns[i] = trade.ts_event_ns
        prices[i] = trade.price
        sizes[i] = trade.size
    
    # Filter outliers (ES typically 3000-10000 range in late 2024/early 2025)
    valid_mask = (prices > 3000) & (prices < 10000)
    ts_ns = ts_ns[valid_mask]
    prices = prices[valid_mask]
    sizes = sizes[valid_mask]
    
    if len(ts_ns) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'],
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(ts_ns, unit='ns', utc=True),
        'price': prices,
        'size': sizes,
        'ts_ns': ts_ns
    })
    
    # Apply RTH filter BEFORE resampling (critical for v1)
    if rth_only:
        df = filter_rth_only(df, date, ts_col='ts_ns')
    
    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'],
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    # Resample to OHLCV
    df_indexed = df.set_index('timestamp')
    ohlcv = df_indexed['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    ohlcv.columns = ['open', 'high', 'low', 'close']
    ohlcv['volume'] = df_indexed['size'].resample(freq).sum()
    
    # Drop NaN bars and keep DatetimeIndex
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv.index.name = 'timestamp'
    
    # Add ts_ns for merging
    ohlcv['ts_ns'] = ohlcv.index.values.astype('datetime64[ns]').astype(np.int64)
    
    # NO CONVERSION - ES prices are already in ES index points.
    
    return ohlcv


def compute_rth_atr(ohlcv_df: pd.DataFrame, window_minutes: int = None) -> pd.Series:
    """
    Compute ATR from RTH-only OHLCV data.
    
    ATR must be RTH-only (no premarket/overnight leakage).
    
    Args:
        ohlcv_df: OHLCV DataFrame (already filtered to RTH)
        window_minutes: ATR window (defaults to CONFIG.ATR_WINDOW_MINUTES)
    
    Returns:
        ATR series indexed by bar
    """
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
    
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).rolling(window=window_minutes, min_periods=1).mean().to_numpy()
    
    return pd.Series(atr, index=df.index)


def compute_rth_volatility(ohlcv_df: pd.DataFrame, window_minutes: int = 20) -> pd.Series:
    """
    Compute realized volatility from RTH-only returns.
    
    Vol baselines must be RTH-only.
    
    Args:
        ohlcv_df: OHLCV DataFrame (RTH-filtered)
        window_minutes: Rolling window
    
    Returns:
        Volatility series
    """
    if ohlcv_df.empty:
        return pd.Series(dtype=np.float64)
    
    df = ohlcv_df.sort_values('timestamp').copy()
    returns = df['close'].diff().fillna(0.0)
    vol = returns.rolling(window=window_minutes, min_periods=1).std().fillna(0.0)
    
    return vol


class BuildOHLCVStage(BaseStage):
    """
    Build ES OHLCV bars from ES futures.

    ES futures are quoted in index points.

    Args:
        freq: Bar frequency ('1min', '2min')
        output_key: Key for context.data
        include_warmup: Whether to include warmup bars (for SMA)
        rth_only: If True, filter to RTH before resampling (for ATR/vol)

    Outputs:
        {output_key}: OHLCV DataFrame
        atr: ATR series (only for 1min bars)
        volatility: Volatility series (only for 1min bars)
    """
    
    def __init__(
        self,
        freq: str = '1min',
        output_key: str = None,
        include_warmup: bool = False,
        rth_only: bool = False
    ):
        self.freq = freq
        self.output_key = output_key or f'ohlcv_{freq}'
        self.include_warmup = include_warmup
        self.rth_only = rth_only
    
    @property
    def name(self) -> str:
        return f"build_ohlcv_{self.freq}"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['trades']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        from src.pipeline.stages.load_bronze import futures_trades_from_df
        from src.pipeline.utils.duckdb_reader import DuckDBReader
        
        trades = ctx.data['trades']
        
        # Build OHLCV
        ohlcv_df = build_es_ohlcv_from_es(
            trades=trades,
            date=ctx.date,
            freq=self.freq,
            rth_only=self.rth_only
        )
        
        result = {self.output_key: ohlcv_df}
        
        # Compute ATR and volatility for 1min bars
        if self.freq == '1min' and not ohlcv_df.empty:
            result['atr'] = compute_rth_atr(ohlcv_df)
            result['volatility'] = compute_rth_volatility(ohlcv_df)
        
        # Add warmup for 2min SMA calculation
        if self.include_warmup and self.freq == '2min':
            import logging
            logger = logging.getLogger(__name__)
            
            reader = ctx.data.get('_reader') or DuckDBReader()
            warmup_dates = reader.get_warmup_dates(ctx.date, CONFIG.SMA_WARMUP_DAYS)
            
            logger.info(f"    Warmup: loading {len(warmup_dates)} dates: {warmup_dates}")
            
            if warmup_dates:
                warmup_frames = []
                for warmup_date in warmup_dates:
                    trades_df = reader.read_futures_trades(
                        symbol='ES',
                        date=warmup_date,
                        front_month_only=False
                    )
                    if trades_df.empty:
                        logger.warning(f"    Warmup: no trades for {warmup_date}")
                        continue
                    
                    from src.pipeline.stages.load_bronze import futures_trades_from_df
                    warmup_trades = futures_trades_from_df(trades_df)
                    if not warmup_trades:
                        logger.warning(f"    Warmup: failed to convert trades for {warmup_date}")
                        continue
                    
                    warmup_ohlcv = build_es_ohlcv_from_es(
                        warmup_trades,
                        warmup_date,
                        freq=self.freq,
                        rth_only=False  # Include all hours for SMA warmup
                    )
                    if not warmup_ohlcv.empty:
                        warmup_frames.append(warmup_ohlcv)
                        logger.info(f"    Warmup: loaded {len(warmup_ohlcv)} bars from {warmup_date}")
                    else:
                        logger.warning(f"    Warmup: empty OHLCV for {warmup_date}")
                
                if warmup_frames:
                    # Concatenate keeping DatetimeIndex
                    warmup_df = pd.concat(warmup_frames, axis=0)
                    logger.info(f"    Warmup: {len(warmup_df)} total warmup bars")
                    ohlcv_df = pd.concat([warmup_df, ohlcv_df], axis=0)
                    ohlcv_df = ohlcv_df.sort_index()
                    result[self.output_key] = ohlcv_df
                    result['warmup_dates'] = warmup_dates
                    logger.info(f"    Warmup: final 2min OHLCV has {len(ohlcv_df)} bars")
                else:
                    logger.warning(f"    Warmup: no warmup frames loaded")
        
        return result
