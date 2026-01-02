import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import time

from src.pipeline.core.stage import BaseStage, StageContext

logger = logging.getLogger(__name__)

LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
STATE_CADENCE_SECONDS = 30


def materialize_state_table(
    signals_df: pd.DataFrame,
    ohlcv_2min: pd.DataFrame,
    date: pd.Timestamp,
    cadence_seconds: int = STATE_CADENCE_SECONDS
) -> pd.DataFrame:
    """
    Materialize level-relative state table at fixed 30-second cadence using Vectorized Merge Asof.
    
    Optimization:
    - Replaces nested loop (Times x Levels) with pd.merge_asof.
    - Align timestamp grid to signals for each level kind efficiently.
    
    Args:
        signals_df: Event table with all features computed
        ohlcv_2min: 2-minute OHLCV bars for SMA computation
        date: Trading date
        cadence_seconds: Sampling interval (default 30s)
    
    Returns:
        DataFrame with state table schema (Section 4.3)
    """
    if signals_df.empty:
        return pd.DataFrame()
    
    logger.info(f"  Materializing state table (30s cadence) for {date.date()}...")
    
    # 1. Generate Timestamp Grid
    date_str = date.strftime('%Y-%m-%d')
    start_ts = pd.Timestamp(f"{date_str} 09:30:00", tz='America/New_York')
    end_ts = pd.Timestamp(f"{date_str} 12:30:00", tz='America/New_York')
    
    timestamp_grid = pd.date_range(start=start_ts, end=end_ts, freq=f'{cadence_seconds}s')
    grid_df = pd.DataFrame({'timestamp': timestamp_grid})
    
    # 2. Prepare Signals
    signals_sorted = signals_df.sort_values('timestamp').copy()
    if signals_sorted['timestamp'].dt.tz is None:
        signals_sorted['timestamp'] = signals_sorted['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    elif str(signals_sorted['timestamp'].dt.tz) == 'UTC':
         signals_sorted['timestamp'] = signals_sorted['timestamp'].dt.tz_convert('America/New_York')

    # Mapping for level kind integers
    LEVEL_KIND_DECODE = {
        0: 'PM_HIGH', 1: 'PM_LOW', 2: 'OR_HIGH', 3: 'OR_LOW',
        6: 'SMA_90', 12: 'EMA_20'
    }
    if signals_sorted['level_kind'].dtype in [int, 'int64', 'int32', 'int16', 'int8']:
         signals_sorted['level_kind'] = signals_sorted['level_kind'].map(LEVEL_KIND_DECODE).fillna('UNKNOWN')

    # 3. Resolve Dynamic Levels (SMA/EMA) for the grid
    # We need to map grid timestamps to SMA values to fill 'level_price' for SMA/EMA rows
    sma_series = None
    ema_series = None
    if not ohlcv_2min.empty:
        df_2m = ohlcv_2min.sort_index().copy()
        if 'sma_90' not in df_2m.columns:
            df_2m['sma_90'] = df_2m['close'].rolling(window=90, min_periods=90).mean()
        if 'ema_20' not in df_2m.columns:
            df_2m['ema_20'] = df_2m['close'].ewm(span=20, adjust=False).mean()
            
        # AsOf merge grid to OHLCV to get indicators
        # OHLCV index is timestamp, usually left-labeled or right? 
        # Standard: use 'backward' lookup (value known at T)
        # We need temporary grid with tz-naive if ohlcv is naive, or convert.
        # Assuming ohlcv index is same tz as grid (ET or UTC). Check inputs.
        # Assuming standard pipeline: inputs converted to consistent TZ.
        # Let's align zones.
        if df_2m.index.tz is None:
             # Assume ET if naive, or UTC?
             # Standard pipeline uses UTC internally usually. But here we constructed grid in ET.
             # Safest: Convert grid to UTC for lookup if OHLCV is UTC.
             pass
        
        # We will do a merge_asof on the grid
        # grid_df['timestamp'] is ET.
        # df_2m index... let's reset
        df_2m = df_2m.reset_index()
        # Rename index to timestamp if needed
        ts_col = 'timestamp' if 'timestamp' in df_2m.columns else 'index'
        
        # Align TZs
        if df_2m[ts_col].dt.tz is None:
             df_2m[ts_col] = df_2m[ts_col].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
             df_2m[ts_col] = df_2m[ts_col].dt.tz_convert('America/New_York')
             
        df_2m = df_2m.sort_values(ts_col)
        
        # Merge columns
        grid_with_indicators = pd.merge_asof(
            grid_df, 
            df_2m[[ts_col, 'sma_90', 'ema_20']], 
            left_on='timestamp', 
            right_on=ts_col, 
            direction='backward'
        )
        sma_series = grid_with_indicators.set_index('timestamp')['sma_90']
        ema_series = grid_with_indicators.set_index('timestamp')['ema_20']
        
    # 4. Vectorized Construction Per Level Kind
    dfs = []
    
    or_open_ts = pd.Timestamp(f"{date_str} 09:45:00", tz='America/New_York')
    
    # Pre-calculate common columns
    base_grid = grid_df.copy()
    base_grid['ts_ns'] = base_grid['timestamp'].astype('int64') # nanoseconds
    base_grid['date'] = date
    base_grid['minutes_since_open'] = (base_grid['timestamp'] - start_ts).dt.total_seconds() / 60.0
    base_grid['bars_since_open'] = (base_grid['minutes_since_open'] / 2).astype(int)
    
    for kind in LEVEL_KINDS:
        # Filter signals for this kind
        # We need to forward fill: "What was the last state of this level?"
        subset = signals_sorted[signals_sorted['level_kind'] == kind].sort_values('timestamp')
        
        if subset.empty:
            # Maybe level exists but no signals triggered?
            # State table tracks "signals". If no signal, we have no "features" for this level.
            # But we might still want the 'level_price' if it exists (e.g. PM High defined but never touched).
            # The current requirement says "Forward fill features from LAST SIGNAL".
            # If no signal ever, feature values are NaN.
            # We still produce rows for the level kind.
            pass
            
        # Merge AsOf
        # Columns to keep from signals
        keep_cols = [c for c in subset.columns if c not in ['timestamp', 'level_kind']]
        merged = pd.merge_asof(
            base_grid, 
            subset[['timestamp'] + keep_cols], 
            on='timestamp', 
            direction='backward'
        )
        
        merged['level_kind'] = kind
        
        # Logic for 'level_active' and 'level_price'
        # 1. PM Levels: Always active. Price should be backfilled/constant if known?
        # Only known if at least one signal OR from 'generate_levels' output (which we don't have here explicitly, only via signals).
        # Actually, signals have 'level_price'.
        # If merged has NaN, it means NO signal occurred before T.
        # But level might be active.
        # We can attempt to fill 'level_price' if we observed it LATER?
        # No, online-safe means we assume we don't know it until observed?
        # Actually 'level_prices' for PM/OR are known once established.
        # Since this stage receives signals_df, and signals capture interaction...
        # If we rely strictly on signals_df, we only know level_price after first interaction?
        # The previous code extracted level_prices from the entire daily signals_df (peeking future signals to find the constant level price).
        # "PM levels: use first occurrence".
        # This is safe because PM levels are determined Pre-Market (before 9:30).
        # So conceptually we know them at 9:30.
        # Code: Extract canonical price for the day from ALL signals (peeking is valid for static levels).
        
        canonical_price = None
        if kind in ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW']:
            in_signals = signals_sorted[signals_sorted['level_kind'] == kind]
            if not in_signals.empty:
                canonical_price = in_signals['level_price'].iloc[0]
                
        # Set Active/Price
        if kind in ['PM_HIGH', 'PM_LOW']:
            merged['level_active'] = True
            # Fill price if not present from merge (e.g. before first signal)
            if canonical_price is not None:
                merged['level_price'] = merged['level_price'].fillna(canonical_price)
                
        elif kind in ['OR_HIGH', 'OR_LOW']:
            merged['level_active'] = merged['timestamp'] >= or_open_ts
            if canonical_price is not None:
                 merged['level_price'] = merged['level_price'].fillna(canonical_price)
            # Mask out price before open
            merged.loc[~merged['level_active'], 'level_price'] = None

        elif kind == 'SMA_90':
            # Dynamic Price from OHLCV
            if sma_series is not None:
                merged['level_price'] = sma_series.values # aligned by grid index because sma_series comes from grid
                merged['level_active'] = merged['level_price'].notna()
            else:
                merged['level_active'] = False
                
        elif kind == 'EMA_20':
            if ema_series is not None:
                merged['level_price'] = ema_series.values
                merged['level_active'] = merged['level_price'].notna()
            else:
                 merged['level_active'] = False
                 
        dfs.append(merged)
        
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


class MaterializeStateTableStage(BaseStage):
    """Materialize level-relative state table at 30-second cadence.
    
    Per IMPLEMENTATION_READY.md Section 4 (Stage 16):
    - Samples market state every 30 seconds from 09:30-12:30 ET
    - One row per (timestamp, level_kind) pair  
    - Enables consistent window extraction for episode vectors
    - All features are online-safe (forward-filled from events)
    
    Outputs:
        state_df: State table with schema per Section 4.3
    """
    
    @property
    def name(self) -> str:
        return "materialize_state_table"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_2min']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_2min = ctx.data['ohlcv_2min']
        date = pd.Timestamp(ctx.date)
        
        state_df = materialize_state_table(
            signals_df=signals_df,
            ohlcv_2min=ohlcv_2min,
            date=date
        )

        # Optional: persist canonical state table for this pipeline run
        state_output_path: Path | None = None
        if ctx.config.get("PIPELINE_WRITE_STATE_TABLE"):
            data_root = ctx.config.get("DATA_ROOT")
            canonical_version = ctx.config.get("PIPELINE_CANONICAL_VERSION")
            if not data_root or not canonical_version:
                logger.warning("  Skipping state table write: missing DATA_ROOT or PIPELINE_CANONICAL_VERSION")
            else:
                base_dir = canonical_state_dir(data_root, dataset="es_level_state", version=canonical_version)
                date_dir = base_dir / date_partition(ctx.date)

                if ctx.config.get("PIPELINE_OVERWRITE_PARTITIONS", True) and date_dir.exists():
                    shutil.rmtree(date_dir)
                date_dir.mkdir(parents=True, exist_ok=True)

                state_output_path = date_dir / "state.parquet"
                state_df.to_parquet(
                    state_output_path,
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
                logger.info(f"  Wrote Silver state table: {state_output_path}")

        return {
            'state_df': state_df,
            'state_output_path': str(state_output_path) if state_output_path else None,
        }
