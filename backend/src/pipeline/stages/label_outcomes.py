"""Label outcomes stage - IMPLEMENTATION_READY.md Section 3."""
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


def label_outcomes(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    lookforward_minutes: int = None,
    outcome_threshold: float = None,
    confirmation_seconds: float = None,
    use_multi_timeframe: bool = True
) -> pd.DataFrame:
    """
    Label outcomes using FIRST-CROSSING SEMANTICS with fixed-point thresholds.
    
    Outcome determined by which threshold is crossed first:
    - BREAK: Price trades through level to far side by MONITOR_BAND points
    - REJECT: Price moves away on near side by MONITOR_BAND points before achieving BREAK
    - CHOP: Neither BREAK nor REJECT within 8 minutes
    
    Direction semantics:
    - UP: approaching from below, BREAK = crossed above, REJECT = reversed down
    - DOWN: approaching from above, BREAK = crossed below, REJECT = reversed up
    
    Touch detection (OHLC-based):
    - A touch occurs when any bar's OHLC range intersects TOUCH_BAND
    - Condition: bar_low <= level_price + TOUCH_BAND AND bar_high >= level_price - TOUCH_BAND
    
    Fixed horizons: 2min, 4min, 8min (8min is primary for outcome determination)
    
    CRITICAL: This generates LABELS ONLY. These fields must NEVER appear in feature vectors.

    Args:
        signals_df: DataFrame with signals (event table)
        ohlcv_df: OHLCV DataFrame (ES futures, 1min bars)
        lookforward_minutes: Not used (horizons are fixed)
        outcome_threshold: Not used (uses MONITOR_BAND from CONFIG)
        use_multi_timeframe: Must be True (generates 2/4/8min labels)

    Returns:
        DataFrame with outcome labels, touch flags, and defended touch counts
    """
    # Fixed horizons: 2min, 4min, 8min (8min primary per user spec)
    confirmation_windows = [120, 240, 480]  # seconds
    window_labels = ['2min', '4min', '8min']
    
    # Fixed-point thresholds (from CONFIG)
    break_threshold_points = CONFIG.MONITOR_BAND  # 5.0 points
    reject_threshold_points = CONFIG.MONITOR_BAND  # 5.0 points
    touch_band_points = CONFIG.TOUCH_BAND  # 2.0 points
    
    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    # Prepare OHLCV data for fast lookup
    ohlcv = ohlcv_df.copy()
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv = ohlcv.reset_index()
        if 'timestamp' not in ohlcv.columns:
            ohlcv = ohlcv.rename(columns={'index': 'timestamp'})

    if 'timestamp' not in ohlcv.columns:
        raise ValueError("ohlcv_df must have DatetimeIndex or 'timestamp' column")

    ohlcv_sorted = ohlcv.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_high = ohlcv_sorted['high'].values.astype(np.float64)
    ohlcv_low = ohlcv_sorted['low'].values.astype(np.float64)
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    directions = signals_df['direction'].values
    if 'level_price' not in signals_df.columns:
        raise ValueError("Missing level_price for outcome labeling.")
    level_prices = signals_df['level_price'].values.astype(np.float64)
    
    # Get level identifiers for per-level touch tracking
    if 'level_kind_name' in signals_df.columns:
        level_identifiers = signals_df['level_kind_name'].values
    elif 'level_kind' in signals_df.columns:
        level_kind_map = {0: 'PM_HIGH', 1: 'PM_LOW', 2: 'OR_HIGH', 3: 'OR_LOW', 6: 'SMA_90', 12: 'EMA_20'}
        level_identifiers = np.array([level_kind_map.get(lk, str(lk)) for lk in signals_df['level_kind'].values])
    else:
        level_identifiers = np.array(['UNKNOWN'] * n)
    
    # Storage for multi-timeframe results
    results_by_window = {}
    
    # Global per-level touch tracking (for all 6 levels)
    level_names = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
    per_level_touches_from_above = {level: 0 for level in level_names}
    per_level_touches_from_below = {level: 0 for level in level_names}
    per_level_defended_touches_from_above = {level: 0 for level in level_names}
    per_level_defended_touches_from_below = {level: 0 for level in level_names}
    per_level_last_touch_from_above_ts = {level: None for level in level_names}
    per_level_last_touch_from_below_ts = {level: None for level in level_names}
    
    # Initialize per-event storage for per-level features
    per_level_feature_arrays = {}
    for level_name in level_names:
        per_level_feature_arrays[f'{level_name.lower()}_touches_from_above'] = np.zeros(n, dtype=np.int32)
        per_level_feature_arrays[f'{level_name.lower()}_touches_from_below'] = np.zeros(n, dtype=np.int32)
        per_level_feature_arrays[f'{level_name.lower()}_defended_touches_from_above'] = np.zeros(n, dtype=np.int32)
        per_level_feature_arrays[f'{level_name.lower()}_defended_touches_from_below'] = np.zeros(n, dtype=np.int32)
        per_level_feature_arrays[f'{level_name.lower()}_time_since_touch_from_above_sec'] = np.full(n, np.nan, dtype=np.float64)
        per_level_feature_arrays[f'{level_name.lower()}_time_since_touch_from_below_sec'] = np.full(n, np.nan, dtype=np.float64)
    
    # Process each confirmation window (2min, 4min, 8min)
    for window_sec, label in zip(confirmation_windows, window_labels):
        horizon_ns = int(window_sec * 1e9)
        
        # Initialize arrays for this window
        outcomes = np.empty(n, dtype=object)
        touched_flag = np.zeros(n, dtype=bool)
        excursion_favorable = np.full(n, np.nan, dtype=np.float64)
        excursion_adverse = np.full(n, np.nan, dtype=np.float64)
        excursion_max = np.full(n, np.nan, dtype=np.float64)  # Keep for backward compat
        excursion_min = np.full(n, np.nan, dtype=np.float64)  # Keep for backward compat
        strength_signed = np.full(n, np.nan, dtype=np.float64)
        strength_abs = np.full(n, np.nan, dtype=np.float64)
        time_to_break_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_bounce_1 = np.full(n, np.nan, dtype=np.float64)
        
        # Process each signal using first-crossing semantics with fixed thresholds
        for i in range(n):
            ts = signal_ts[i]
            direction = directions[i]
            level_price = level_prices[i]
            level_id = level_identifiers[i]
            
            # Snapshot current global state for ALL levels at this event (before processing)
            if label == '8min':  # Only track on 8min window (primary)
                for level_name in level_names:
                    per_level_feature_arrays[f'{level_name.lower()}_touches_from_above'][i] = per_level_touches_from_above[level_name]
                    per_level_feature_arrays[f'{level_name.lower()}_touches_from_below'][i] = per_level_touches_from_below[level_name]
                    per_level_feature_arrays[f'{level_name.lower()}_defended_touches_from_above'][i] = per_level_defended_touches_from_above[level_name]
                    per_level_feature_arrays[f'{level_name.lower()}_defended_touches_from_below'][i] = per_level_defended_touches_from_below[level_name]
                    
                    if per_level_last_touch_from_above_ts[level_name] is not None:
                        per_level_feature_arrays[f'{level_name.lower()}_time_since_touch_from_above_sec'][i] = (ts - per_level_last_touch_from_above_ts[level_name]) / 1e9
                    
                    if per_level_last_touch_from_below_ts[level_name] is not None:
                        per_level_feature_arrays[f'{level_name.lower()}_time_since_touch_from_below_sec'][i] = (ts - per_level_last_touch_from_below_ts[level_name]) / 1e9
            
            start_idx = np.searchsorted(ohlcv_ts, ts, side='right') - 1
            if start_idx < 0:
                start_idx = 0
            if start_idx >= len(ohlcv_ts):
                outcomes[i] = 'CHOP'
                continue

            end_idx = np.searchsorted(ohlcv_ts, ts + horizon_ns, side='right')
            if start_idx >= end_idx:
                outcomes[i] = 'CHOP'
                continue

            window_high = ohlcv_high[start_idx:end_idx]
            window_low = ohlcv_low[start_idx:end_idx]
            window_ts = ohlcv_ts[start_idx:end_idx]

            if len(window_high) == 0:
                outcomes[i] = 'CHOP'
                continue

            # Touch detection: OHLC range intersects TOUCH_BAND
            # bar_low <= level_price + TOUCH_BAND AND bar_high >= level_price - TOUCH_BAND
            touch_mask = (window_low <= level_price + touch_band_points) & (window_high >= level_price - touch_band_points)
            touched_during_window = np.any(touch_mask)
            
            if label == '8min':  # Only record touches for 8min window (primary)
                touched_flag[i] = touched_during_window

            # Fixed-point thresholds for BREAK/REJECT
            if direction == 'UP':
                # Approaching from below: break = cross above, reject = reverse down
                break_threshold = level_price + break_threshold_points
                bounce_threshold = level_price - reject_threshold_points
                
                # Find first crossing times
                break_mask = window_high >= break_threshold
                bounce_mask = window_low <= bounce_threshold
            else:  # DOWN
                # Approaching from above: break = cross below, reject = reverse up
                break_threshold = level_price - break_threshold_points
                bounce_threshold = level_price + reject_threshold_points
                
                # Find first crossing times
                break_mask = window_low <= break_threshold
                bounce_mask = window_high >= bounce_threshold
            
            # Get first crossing times (in seconds from signal timestamp)
            t_break = None
            t_bounce = None
            
            if np.any(break_mask):
                first_break_idx = int(np.argmax(break_mask))
                t_break = (window_ts[first_break_idx] - ts) / 1e9
                time_to_break_1[i] = t_break
            
            if np.any(bounce_mask):
                first_bounce_idx = int(np.argmax(bounce_mask))
                t_bounce = (window_ts[first_bounce_idx] - ts) / 1e9
                time_to_bounce_1[i] = t_bounce
            
            # Apply first-crossing semantics
            if t_break is None and t_bounce is None:
                # Neither threshold crossed
                outcomes[i] = 'CHOP'
            elif t_break is not None and t_bounce is None:
                outcomes[i] = 'BREAK'
            elif t_bounce is not None and t_break is None:
                outcomes[i] = 'REJECT'
            elif abs(t_break - t_bounce) < 0.1:  # Crossed at same time (within 0.1s)
                # Tie: both thresholds crossed at same bar
                # Use excursion magnitudes to determine winner
                max_above = window_high.max() - level_price
                max_below = level_price - window_low.min()
                
                if direction == 'UP':
                    # Favorable = above, adverse = below
                    if max_above > max_below:
                        outcomes[i] = 'BREAK'
                    else:
                        outcomes[i] = 'REJECT'
                else:  # DOWN
                    # Favorable = below, adverse = above
                    if max_below > max_above:
                        outcomes[i] = 'BREAK'
                    else:
                        outcomes[i] = 'REJECT'
            elif t_break < t_bounce:
                # Break hit first
                outcomes[i] = 'BREAK'
            elif t_bounce < t_break:
                # Bounce (reject) hit first
                outcomes[i] = 'REJECT'
            else:
                # Shouldn't reach here
                outcomes[i] = 'CHOP'
            
            # Compute continuous outcome variables (in points, not ATR-normalized)
            max_above = window_high.max() - level_price
            max_below = level_price - window_low.min()
            
            if direction == 'UP':
                # Favorable = movement up (break direction), adverse = movement down
                excursion_favorable[i] = max(max_above, 0.0)
                excursion_adverse[i] = max(max_below, 0.0)
                strength_signed[i] = max_above - max_below
                strength_abs[i] = max(max_above, max_below)
                # Backward compat
                excursion_max[i] = max_above
                excursion_min[i] = max_below
            else:  # DOWN
                # Favorable = movement down (break direction), adverse = movement up
                excursion_favorable[i] = max(max_below, 0.0)
                excursion_adverse[i] = max(max_above, 0.0)
                strength_signed[i] = max_below - max_above
                strength_abs[i] = max(max_below, max_above)
                # Backward compat
                excursion_max[i] = max_below
                excursion_min[i] = max_above
            
            # Update global per-level state (only for 8min window)
            if label == '8min' and touched_during_window and level_id in level_names:
                dir_key = 'from_above' if direction == 'DOWN' else 'from_below'
                
                # Increment touch count
                if dir_key == 'from_above':
                    per_level_touches_from_above[level_id] += 1
                    per_level_last_touch_from_above_ts[level_id] = ts
                    # Defended touch = touched but did NOT break
                    if outcomes[i] != 'BREAK':
                        per_level_defended_touches_from_above[level_id] += 1
                else:  # from_below
                    per_level_touches_from_below[level_id] += 1
                    per_level_last_touch_from_below_ts[level_id] = ts
                    # Defended touch = touched but did NOT break
                    if outcomes[i] != 'BREAK':
                        per_level_defended_touches_from_below[level_id] += 1
        
        # Store results for this window
        results_by_window[label] = {
            'outcomes': outcomes,
            'excursion_favorable': excursion_favorable,
            'excursion_adverse': excursion_adverse,
            'excursion_max': excursion_max,  # Backward compat
            'excursion_min': excursion_min,  # Backward compat
            'strength_signed': strength_signed,
            'strength_abs': strength_abs,
            'time_to_break_1': time_to_break_1,
            'time_to_bounce_1': time_to_bounce_1,
        }
    
    # Build result DataFrame with multi-horizon labels (IMPLEMENTATION_READY.md Section 3.3)
    result = signals_df.copy()
    
    # Add columns for each horizon
    for label, data in results_by_window.items():
        result[f'outcome_{label}'] = data['outcomes']
        result[f'time_to_break_1_{label}'] = data['time_to_break_1']
        result[f'time_to_bounce_1_{label}'] = data['time_to_bounce_1']
    
    # Add direction-aware continuous outcomes using 8min as primary
    if '8min' in results_by_window:
        primary_data = results_by_window['8min']
        result['excursion_favorable'] = primary_data['excursion_favorable']
        result['excursion_adverse'] = primary_data['excursion_adverse']
        result['strength_signed'] = primary_data['strength_signed']
        result['strength_abs'] = primary_data['strength_abs']
        
        # Backward compatibility columns
        result['excursion_max'] = primary_data['excursion_max']
        result['excursion_min'] = primary_data['excursion_min']
        result['outcome'] = primary_data['outcomes']  # Primary outcome (8min)
        result['time_to_break_1'] = primary_data['time_to_break_1']
        result['time_to_bounce_1'] = primary_data['time_to_bounce_1']
    
    # Add per-level touch features (48 features: 6 levels × 8 metrics)
    # - touches_from_above/below: count of touches (OHLC range intersects TOUCH_BAND)
    # - defended_touches_from_above/below: touches that did NOT break
    # - time_since_touch_from_above/below_sec: recency of last touch
    for feature_name, feature_values in per_level_feature_arrays.items():
        result[feature_name] = feature_values

    return result


class LabelOutcomesStage(BaseStage):
    """Label outcomes using first-crossing semantics (IMPLEMENTATION_READY.md Section 3).

    Computes forward-looking outcome labels based on which threshold is hit first:
    - BREAK: 1 ATR movement in direction of approach
    - REJECT: 1 ATR movement in opposite direction  
    - CHOP: Neither threshold hit within horizon

    Multi-horizon labels: 2min, 4min, 8min (independent)
    Primary horizon: 4min

    Outputs:
        signals_df: Updated with outcome labels:
            - outcome_2min, outcome_4min, outcome_8min: {BREAK, REJECT, CHOP}
            - excursion_favorable, excursion_adverse: ATR-normalized continuous outcomes
            - time_to_break_1_{horizon}, time_to_bounce_1_{horizon}: First crossing times
    
    CRITICAL: These are LABELS ONLY. Must never appear in feature vectors.
    """

    @property
    def name(self) -> str:
        return "label_outcomes"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']  # Use 1min bars for precise alignment

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']

        n_signals = len(signals_df)
        logger.info(f"  Labeling outcomes (first-crossing, ATR-normalized) for {n_signals:,} signals...")
        logger.debug(f"    OHLCV bars: {len(ohlcv_df):,}")
        logger.debug(f"    Horizons: 2min, 4min, 8min | Threshold: 1.0 ATR")

        signals_df = label_outcomes(signals_df, ohlcv_df)

        # Log outcome distribution for each horizon
        for horizon in ['2min', '4min', '8min']:
            col = f'outcome_{horizon}'
            if col in signals_df.columns:
                outcome_dist = signals_df[col].value_counts().to_dict()
                logger.info(f"    {horizon} outcome distribution: {outcome_dist}")

        return {'signals_df': signals_df}
