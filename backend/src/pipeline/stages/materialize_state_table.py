"""Materialize state table at fixed cadence - IMPLEMENTATION_READY.md Section 4."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from datetime import time

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.lake_paths import canonical_state_dir, date_partition

logger = logging.getLogger(__name__)


LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400']
STATE_CADENCE_SECONDS = 30
RTH_START = time(9, 30, 0)  # 09:30:00 ET
RTH_END = time(12, 30, 0)   # 12:30:00 ET

# Level kind integer encoding (from generate_levels.py)
LEVEL_KIND_DECODE = {
    0: 'PM_HIGH',
    1: 'PM_LOW',
    2: 'OR_HIGH',
    3: 'OR_LOW',
    6: 'SMA_90',
    12: 'EMA_20'
}


def materialize_state_table(
    signals_df: pd.DataFrame,
    ohlcv_2min: pd.DataFrame,
    date: pd.Timestamp,
    cadence_seconds: int = STATE_CADENCE_SECONDS
) -> pd.DataFrame:
    """
    Materialize level-relative state table at fixed 30-second cadence.
    
    Per IMPLEMENTATION_READY.md Section 4:
    - Generate timestamps every 30s from 09:30 to 12:30 ET (360 samples per level per day)
    - One row per (timestamp, level_kind) pair
    - All features are online-safe (use only data from timestamp T and before)
    - Handle OR levels being undefined before 09:45
    
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
    logger.info(f"    Input signals_df shape: {signals_df.shape}")
    
    # Generate timestamp grid: 09:30 to 12:30 ET at 30s intervals
    date_str = date.strftime('%Y-%m-%d')
    start_ts = pd.Timestamp(f"{date_str} 09:30:00", tz='America/New_York')
    end_ts = pd.Timestamp(f"{date_str} 12:30:00", tz='America/New_York')
    
    timestamp_grid = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=f'{cadence_seconds}s'
    )
    
    logger.info(f"    Generated {len(timestamp_grid)} timestamps (09:30-12:30 ET @ {cadence_seconds}s)")
    logger.info(f"    Grid range: {timestamp_grid[0]} to {timestamp_grid[-1]}")
    
    # Prepare signals_df for efficient lookup
    signals_sorted = signals_df.sort_values('timestamp').copy()
    
    logger.info(f"    Signals timestamp dtype before TZ: {signals_sorted['timestamp'].dtype}")
    logger.info(f"    Signals timestamp tz before: {signals_sorted['timestamp'].dt.tz}")
    logger.info(f"    Signals timestamp range before TZ: {signals_sorted['timestamp'].min()} to {signals_sorted['timestamp'].max()}")
    
    # Ensure timestamps are timezone-aware (convert UTC to ET)
    if signals_sorted['timestamp'].dt.tz is None:
        signals_sorted['timestamp'] = signals_sorted['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    elif str(signals_sorted['timestamp'].dt.tz) == 'UTC':
        signals_sorted['timestamp'] = signals_sorted['timestamp'].dt.tz_convert('America/New_York')
    
    logger.info(f"    Signals timestamp range after TZ: {signals_sorted['timestamp'].min()} to {signals_sorted['timestamp'].max()}")
    logger.info(f"    Number of signals in RTH window: {((signals_sorted['timestamp'] >= start_ts) & (signals_sorted['timestamp'] <= end_ts)).sum()}")
    
    logger.info(f"    Level kind distribution (raw): {signals_sorted['level_kind'].value_counts().to_dict()}")
    
    # Decode level_kind from integer to string (if needed)
    if signals_sorted['level_kind'].dtype in [np.int8, np.int16, np.int32, np.int64, int]:
        signals_sorted['level_kind'] = signals_sorted['level_kind'].map(LEVEL_KIND_DECODE).fillna('UNKNOWN')
        logger.info(f"    Level kind distribution (decoded): {signals_sorted['level_kind'].value_counts().to_dict()}")
    
    # Extract level prices from signals_df
    # PM and OR levels are static after establishment
    # SMA levels are dynamic (recomputed each timestamp)
    level_prices = {}
    
    for level_kind in ['PM_HIGH', 'PM_LOW']:
        # PM levels: use first occurrence (should be same for all)
        level_rows = signals_sorted[signals_sorted['level_kind'] == level_kind]
        if not level_rows.empty:
            level_prices[level_kind] = level_rows['level_price'].iloc[0]
    
    # OR levels: established at 09:45, get from first signal after 09:45
    or_establishment = pd.Timestamp(f"{date_str} 09:45:00", tz='America/New_York')
    for level_kind in ['OR_HIGH', 'OR_LOW']:
        level_rows = signals_sorted[
            (signals_sorted['level_kind'] == level_kind) &
            (signals_sorted['timestamp'] >= or_establishment)
        ]
        if not level_rows.empty:
            level_prices[level_kind] = level_rows['level_price'].iloc[0]
    
    logger.info(f"    Static level_prices extracted: {list(level_prices.keys())}")
    
    # Compute SMAs/EMAs for each timestamp from ohlcv_2min
    sma_90_series = {}
    ema_20_series = {}
    if not ohlcv_2min.empty:
        ohlcv_2min_sorted = ohlcv_2min.sort_index().copy()
        ohlcv_2min_sorted['sma_90'] = ohlcv_2min_sorted['close'].rolling(window=90, min_periods=90).mean()
        ohlcv_2min_sorted['ema_20'] = ohlcv_2min_sorted['close'].ewm(span=20, adjust=False).mean()
        
        # Create lookup dict by timestamp (index is timestamp)
        for ts, row in ohlcv_2min_sorted.iterrows():
            if pd.notna(row['sma_90']):
                sma_90_series[ts] = row['sma_90']
            if pd.notna(row['ema_20']):
                ema_20_series[ts] = row['ema_20']
    
    # Build state table rows
    state_rows = []
    rows_added_per_timestamp = []
    
    for ts_idx, ts in enumerate(timestamp_grid):
        ts_ns = ts.value
        minutes_since_open = (ts - start_ts).total_seconds() / 60.0
        bars_since_open = int(minutes_since_open / 2)  # 2-minute bars
        
        # Get most recent signal state at or before this timestamp
        # (forward-fill features from event table)
        recent_signals = signals_sorted[signals_sorted['timestamp'] <= ts]
        
        if recent_signals.empty:
            if ts_idx < 5:  # Log first few empty cases
                logger.info(f"      ts={ts}: no recent signals yet")
            continue
        
        rows_before = len(state_rows)
        
        # For each level kind, create a state row
        for level_kind in LEVEL_KINDS:
            level_active = True
            level_price = None
            
            # Determine level price and active status
            if level_kind in ['PM_HIGH', 'PM_LOW']:
                level_price = level_prices.get(level_kind)
                if level_price is None:
                    continue
            elif level_kind in ['OR_HIGH', 'OR_LOW']:
                if ts < or_establishment:
                    level_active = False
                else:
                    level_price = level_prices.get(level_kind)
                    if level_price is None:
                        level_active = False
            elif level_kind == 'SMA_90':
                # Find closest SMA value at or before ts
                sma_times = [t for t in sma_90_series.keys() if t <= ts]
                if sma_times:
                    closest_time = max(sma_times)
                    level_price = sma_90_series[closest_time]
                else:
                    level_active = False
            elif level_kind == 'EMA_20':
                ema_times = [t for t in ema_20_series.keys() if t <= ts]
                if ema_times:
                    closest_time = max(ema_times)
                    level_price = ema_20_series[closest_time]
                else:
                    level_active = False
            
            # Get most recent signal for this level_kind
            level_signals = recent_signals[recent_signals['level_kind'] == level_kind]
            if level_signals.empty:
                continue
            
            latest_signal = level_signals.iloc[-1]
            
            # Build state row with forward-filled features
            state_row = {
                'timestamp': ts,
                'ts_ns': ts_ns,
                'date': date,
                'minutes_since_open': minutes_since_open,
                'bars_since_open': bars_since_open,
                'level_kind': level_kind,
                'level_price': level_price if level_active else None,
                'level_active': level_active,
            }
            
            # Forward-fill all feature columns from latest signal
            # (All features in signals_df are online-safe by construction)
            feature_cols = [
                'spot', 'atr', 'distance_signed_atr',
                # Distances to all levels
                'dist_to_pm_high_atr', 'dist_to_pm_low_atr',
                'dist_to_or_high_atr', 'dist_to_or_low_atr',
                'dist_to_sma_90_atr', 'dist_to_ema_20_atr',
                # Level stacking
                'level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt',
                # Kinematics
                'velocity_1min', 'velocity_3min', 'velocity_5min', 'velocity_10min', 'velocity_20min',
                'acceleration_1min', 'acceleration_3min', 'acceleration_5min', 'acceleration_10min', 'acceleration_20min',
                'jerk_1min', 'jerk_3min', 'jerk_5min', 'jerk_10min', 'jerk_20min',
                'momentum_trend_3min', 'momentum_trend_5min', 'momentum_trend_10min', 'momentum_trend_20min',
                # Approach
                'approach_velocity', 'approach_bars', 'approach_distance_atr',
                # Order flow
                'ofi_30s', 'ofi_60s', 'ofi_120s', 'ofi_300s',
                'ofi_near_level_30s', 'ofi_near_level_60s', 'ofi_near_level_120s', 'ofi_near_level_300s',
                'ofi_acceleration',
                # Tape
                'tape_imbalance', 'tape_velocity', 'tape_buy_vol', 'tape_sell_vol', 'sweep_detected',
                # Barrier
                'barrier_state', 'barrier_depth_current', 'barrier_delta_liq',
                'barrier_replenishment_ratio', 'wall_ratio',
                'barrier_delta_1min', 'barrier_delta_3min', 'barrier_delta_5min',
                'barrier_pct_change_1min', 'barrier_pct_change_3min', 'barrier_pct_change_5min',
                # GEX
                'gamma_exposure', 'fuel_effect', 'gex_asymmetry', 'gex_ratio',
                'net_gex_2strike', 'gex_above_1strike', 'gex_below_1strike',
                'call_gex_above_2strike', 'put_gex_below_2strike',
                # Physics
                'predicted_accel', 'accel_residual', 'force_mass_ratio',
                # Touch/attempt
                'prior_touches', 'attempt_index', 'time_since_last_touch',
                # Cluster trends
                'barrier_replenishment_trend', 'barrier_delta_liq_trend',
                'tape_velocity_trend', 'tape_imbalance_trend',
            ]
            
            for col in feature_cols:
                if col in latest_signal.index:
                    state_row[col] = latest_signal[col]
                else:
                    # Handle missing columns gracefully
                    state_row[col] = None
            
            state_rows.append(state_row)
        
        rows_added = len(state_rows) - rows_before
        if ts_idx < 5 or rows_added > 0:  # Log first few or when rows are added
            logger.info(f"      ts={ts} ({minutes_since_open:.1f}min): added {rows_added} rows (total: {len(state_rows)})")
        rows_added_per_timestamp.append(rows_added)
    
    logger.info(f"    Total rows added across {len(timestamp_grid)} timestamps: {sum(rows_added_per_timestamp)}")
    logger.info(f"    Timestamps with >0 rows: {sum(1 for x in rows_added_per_timestamp if x > 0)}")
    
    state_df = pd.DataFrame(state_rows)
    
    logger.info(f"    Generated {len(state_df):,} state rows ({len(state_df) // len(LEVEL_KINDS)} timestamps × {len(LEVEL_KINDS)} levels)")
    
    return state_df


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
