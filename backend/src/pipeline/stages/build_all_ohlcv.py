"""
Build all OHLCV frequencies hierarchically from trades_df.

trades_df → 10s bars → 1min bars → 2min bars

Fully vectorized, no Python loops.
"""

import logging
from typing import Any, Dict
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


def build_10s_ohlcv_from_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build 10-second OHLCV from trades_df (fully vectorized)."""
    if trades_df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'],
                            index=pd.DatetimeIndex([], name='timestamp'))
    
    # Vectorized extraction
    ts_ns = trades_df['ts_event_ns'].to_numpy()
    prices = trades_df['price'].to_numpy()
    sizes = trades_df['size'].to_numpy()
    
    # Vectorized filter
    valid_mask = (prices > 3000) & (prices < 10000)
    ts_ns = ts_ns[valid_mask]
    prices = prices[valid_mask]
    sizes = sizes[valid_mask]
    
    if len(ts_ns) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'],
                            index=pd.DatetimeIndex([], name='timestamp'))
    
    # Build DataFrame with timestamp index
    df = pd.DataFrame({
        'price': prices,
        'size': sizes,
    }, index=pd.to_datetime(ts_ns, unit='ns', utc=True))
    df.index.name = 'timestamp'
    
    # Vectorized resample
    ohlcv = df['price'].resample('10s').agg(['first', 'max', 'min', 'last'])
    ohlcv.columns = ['open', 'high', 'low', 'close']
    ohlcv['volume'] = df['size'].resample('10s').sum()
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv['ts_ns'] = ohlcv.index.values.astype('datetime64[ns]').astype(np.int64)
    
    return ohlcv


def resample_ohlcv(ohlcv_df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """Resample OHLCV to coarser frequency."""
    if ohlcv_df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'ts_ns'],
                            index=pd.DatetimeIndex([], name='timestamp'))
    
    resampled = ohlcv_df.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    resampled = resampled.dropna(subset=['open'])
    resampled.index.name = 'timestamp'
    resampled['ts_ns'] = resampled.index.values.astype('datetime64[ns]').astype(np.int64)
    
    return resampled


def compute_atr(ohlcv_df: pd.DataFrame, window: int = None) -> pd.Series:
    """Compute ATR from OHLCV."""
    if window is None:
        window = CONFIG.ATR_WINDOW_MINUTES
    
    if ohlcv_df.empty:
        return pd.Series(dtype=np.float64)
    
    df = ohlcv_df.sort_index()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr, index=df.index).rolling(window=window, min_periods=1).mean()
    
    return atr


def compute_volatility(ohlcv_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute realized volatility from returns."""
    if ohlcv_df.empty:
        return pd.Series(dtype=np.float64)
    
    df = ohlcv_df.sort_index()
    returns = df['close'].diff().fillna(0.0)
    vol = returns.rolling(window=window, min_periods=1).std().fillna(0.0)
    
    return vol


class BuildAllOHLCVStage(BaseStage):
    """
    Build all OHLCV frequencies hierarchically (fully vectorized).
    
    trades_df → 10s → 1min → 2min
    
    Outputs:
        ohlcv_10s: 10-second bars
        ohlcv_1min: 1-minute bars (resampled from 10s)
        ohlcv_2min: 2-minute bars (resampled from 1min, with warmup)
        atr: ATR series (from 1min)
        volatility: Volatility series (from 1min)
        warmup_dates: List of warmup dates loaded
    """
    
    @property
    def name(self) -> str:
        return "build_all_ohlcv"
    
    @property
    def required_inputs(self):
        return ['trades_df']
    
    def execute(self, ctx: StageContext):
        from src.pipeline.utils.duckdb_reader import DuckDBReader
        
        trades_df = ctx.data['trades_df']
        
        # Build 10s from trades_df (vectorized)
        logger.info(f"    Building 10s bars from {len(trades_df):,} trades...")
        ohlcv_10s = build_10s_ohlcv_from_df(trades_df)
        logger.info(f"    10s bars: {len(ohlcv_10s):,}")
        
        # Resample 10s → 1min
        ohlcv_1min = resample_ohlcv(ohlcv_10s, '1min')
        logger.info(f"    1min bars: {len(ohlcv_1min):,}")
        
        # Resample 1min → 2min
        ohlcv_2min = resample_ohlcv(ohlcv_1min, '2min')
        logger.info(f"    2min bars: {len(ohlcv_2min):,}")
        
        # Compute ATR and volatility from 1min
        atr = compute_atr(ohlcv_1min)
        volatility = compute_volatility(ohlcv_1min)
        
        result = {
            'ohlcv_10s': ohlcv_10s,
            'ohlcv_1min': ohlcv_1min,
            'ohlcv_2min': ohlcv_2min,
            'atr': atr,
            'volatility': volatility,
        }
        
        # Load warmup days for 2min SMA calculation
        reader = ctx.data.get('_reader') or DuckDBReader()
        warmup_dates = reader.get_warmup_dates(ctx.date, CONFIG.SMA_WARMUP_DAYS)
        
        if warmup_dates:
            logger.info(f"    Warmup: loading {len(warmup_dates)} dates for SMA...")
            warmup_frames = []
            
            for warmup_date in warmup_dates:
                # Extract trades from MBP-10 (unified source)
                warmup_start = pd.Timestamp(warmup_date, tz="America/New_York") + pd.Timedelta(hours=4)
                warmup_end = pd.Timestamp(warmup_date, tz="America/New_York") + pd.Timedelta(hours=16)
                warmup_trades_df = reader.read_futures_trades_from_mbp10(
                    date=warmup_date,
                    start_ns=int(warmup_start.tz_convert("UTC").value),
                    end_ns=int(warmup_end.tz_convert("UTC").value)
                )
                if warmup_trades_df.empty:
                    continue
                
                # Vectorized: trades_df → 10s → 1min → 2min
                warmup_10s = build_10s_ohlcv_from_df(warmup_trades_df)
                warmup_2min = resample_ohlcv(resample_ohlcv(warmup_10s, '1min'), '2min')
                
                if not warmup_2min.empty:
                    warmup_frames.append(warmup_2min)
                    logger.info(f"    Warmup: {warmup_date} → {len(warmup_2min)} 2min bars")
            
            if warmup_frames:
                warmup_df = pd.concat(warmup_frames, axis=0)
                ohlcv_2min = pd.concat([warmup_df, ohlcv_2min], axis=0).sort_index()
                result['ohlcv_2min'] = ohlcv_2min
                result['warmup_dates'] = warmup_dates
                logger.info(f"    Warmup: total 2min bars with warmup: {len(ohlcv_2min):,}")
        
        return result

