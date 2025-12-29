"""Materialize state table at fixed cadence - IMPLEMENTATION_READY.md Section 4."""
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from datetime import time

from src.pipeline.core.stage import BaseStage, StageContext

logger = logging.getLogger(__name__)


LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400']
STATE_CADENCE_SECONDS = 30
RTH_START = time(9, 30, 0)  # 09:30:00 ET
RTH_END = time(12, 30, 0)   # 12:30:00 ET


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
    
    # Generate timestamp grid: 09:30 to 12:30 ET at 30s intervals
    date_str = date.strftime('%Y-%m-%d')
    start_ts = pd.Timestamp(f"{date_str} 09:30:00", tz='America/New_York')
    end_ts = pd.Timestamp(f"{date_str} 12:30:00", tz='America/New_York')
    
    timestamp_grid = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=f'{cadence_seconds}s'
    )
    
    logger.debug(f"    Generated {len(timestamp_grid)} timestamps (09:30-12:30 ET @ {cadence_seconds}s)")
    
    # Prepare signals_df for efficient lookup
    signals_sorted = signals_df.sort_values('timestamp').copy()
    
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
    
    # Compute SMAs for each timestamp from ohlcv_2min
    sma_200_series = {}
    sma_400_series = {}
    if not ohlcv_2min.empty:
        ohlcv_2min_sorted = ohlcv_2min.sort_values('timestamp').copy()
        ohlcv_2min_sorted['sma_200'] = ohlcv_2min_sorted['close'].rolling(window=200, min_periods=200).mean()
        ohlcv_2min_sorted['sma_400'] = ohlcv_2min_sorted['close'].rolling(window=400, min_periods=400).mean()
        
        # Create lookup dict by timestamp
        for _, row in ohlcv_2min_sorted.iterrows():
            ts = row['timestamp']
            if pd.notna(row['sma_200']):
                sma_200_series[ts] = row['sma_200']
            if pd.notna(row['sma_400']):
                sma_400_series[ts] = row['sma_400']
    
    # Build state table rows
    state_rows = []
    
    for ts in timestamp_grid:
        ts_ns = ts.value
        minutes_since_open = (ts - start_ts).total_seconds() / 60.0
        bars_since_open = int(minutes_since_open / 2)  # 2-minute bars
        
        # Get most recent signal state at or before this timestamp
        # (forward-fill features from event table)
        recent_signals = signals_sorted[signals_sorted['timestamp'] <= ts]
        
        if recent_signals.empty:
            continue
        
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
            elif level_kind == 'SMA_200':
                # Find closest SMA value at or before ts
                sma_times = [t for t in sma_200_series.keys() if t <= ts]
                if sma_times:
                    closest_time = max(sma_times)
                    level_price = sma_200_series[closest_time]
                else:
                    level_active = False
            elif level_kind == 'SMA_400':
                sma_times = [t for t in sma_400_series.keys() if t <= ts]
                if sma_times:
                    closest_time = max(sma_times)
                    level_price = sma_400_series[closest_time]
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
                'dist_to_sma_200_atr', 'dist_to_sma_400_atr',
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
        return ['signals_df', 'ohlcv_2min', 'date']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_2min = ctx.data['ohlcv_2min']
        date = ctx.data.get('date', pd.Timestamp.now())
        
        state_df = materialize_state_table(
            signals_df=signals_df,
            ohlcv_2min=ohlcv_2min,
            date=date
        )
        
        return {'state_df': state_df}

