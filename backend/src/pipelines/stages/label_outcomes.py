"""
Stage: Label Outcomes
Type: Label Generation (Target Logic)
Input: Signals DataFrame (touches), OHLCV Data (1min)
Output: Signals DataFrame with Outcome Labels (BREAK, REJECT, CHOP)

Transformation:
1. Projects price forward in time (2min, 4min, 8min horizons).
2. Determines the "Outcome" of the interaction using First-Crossing Logic:
   - BREAK: Price moves 12.5pt (50 ticks) THROUGH the level.
   - REJECT: Price moves 12.5pt (50 ticks) AWAY from the level.
   - CHOP: Neither threshold hit within time limit.
3. Computes Continuous Metrics:
   - Excursion Favorable/Adverse: How far did it go right vs wrong?
   - Time to Result: Speed of the move.

Note: These are LABELS. They define what we are trying to predict. NEVER use them as input features.
"""
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
    
    Outcome determined by which threshold (12.5pt / 50 ticks) is crossed first:
    - BREAK: Price moves 12.5pt in direction of approach (through level)
    - REJECT: Price moves 12.5pt opposite to approach (bounced off level)
    - CHOP: Neither threshold crossed within horizon
    
    Direction semantics:
    - UP: approaching from below, BREAK = +12.5pt above level, REJECT = -12.5pt below level
    - DOWN: approaching from above, BREAK = -12.5pt below level, REJECT = +12.5pt above level
    
    Touch detection (OHLC-based):
    - A touch occurs when any bar's OHLC range intersects TOUCH_BAND (±1.5pt)
    - Condition: bar_low <= level + 1.5pt AND bar_high >= level - 1.5pt
    
    Fixed horizons: 2min (scalp), 4min (day trade), 8min (swing)
    Primary horizon: 8min
    
    CRITICAL: This generates LABELS ONLY. These fields must NEVER appear in feature vectors.

    Args:
        signals_df: DataFrame with signals (event table)
        ohlcv_df: OHLCV DataFrame (ES futures, 1min bars)
        lookforward_minutes: Not used (horizons are fixed)
        outcome_threshold: Not used (uses BREAK_REJECT_THRESHOLD from CONFIG)
        use_multi_timeframe: Must be True (generates 2/4/8min labels)

    Returns:
        DataFrame with outcome labels and continuous excursion metrics
    """
    # Fixed horizons: 2min (scalp), 4min (day trade), 8min (swing)
    confirmation_windows = [120, 240, 480]  # seconds: 2min, 4min, 8min
    window_labels = ['2min', '4min', '8min']
    
    # Fixed-point thresholds (from CONFIG)
    break_threshold_points = CONFIG.BREAK_REJECT_THRESHOLD  # 12.5 points (meaningful follow-through)
    reject_threshold_points = CONFIG.BREAK_REJECT_THRESHOLD  # 12.5 points
    touch_band_points = CONFIG.TOUCH_BAND  # 2.0 points (for touch detection)
    
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
    
    # Storage for multi-timeframe results
    results_by_window = {}
    
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
        
        # Process each signal using first-crossing semantics
        for i in range(n):
            ts = signal_ts[i]
            direction = directions[i]
            level_price = level_prices[i]
            
            # Start from NEXT bar after signal to avoid lookahead bias
            # (current bar's OHLC includes price action before signal)
            start_idx = np.searchsorted(ohlcv_ts, ts, side='right')
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
        result['outcome'] = primary_data['outcomes']
        result['time_to_break_1'] = primary_data['time_to_break_1']
        result['time_to_bounce_1'] = primary_data['time_to_bounce_1']

    return result


class LabelOutcomesStage(BaseStage):
    """Label outcomes using first-crossing semantics.

    Computes forward-looking outcome labels based on which threshold is hit first:
    - BREAK: 12.5pt (50 ticks) movement in direction of approach
    - REJECT: 12.5pt (50 ticks) movement in opposite direction  
    - CHOP: Neither threshold hit within horizon

    Fixed-point thresholds (not ATR-scaled) match day trader intuition.
    
    Multi-horizon labels: 2min (scalp), 4min (day trade), 8min (swing)
    Primary horizon: 8min

    Outputs:
        signals_df: Updated with outcome labels:
            - outcome_2min, outcome_4min, outcome_8min: {BREAK, REJECT, CHOP}
            - excursion_favorable, excursion_adverse: Points (direction-aware)
            - time_to_break_1_{horizon}, time_to_bounce_1_{horizon}: First crossing times (seconds)
    
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
        logger.info(f"  Labeling outcomes (first-crossing, fixed 12.5pt) for {n_signals:,} signals...")
        logger.debug(f"    OHLCV bars: {len(ohlcv_df):,}")
        logger.debug(f"    Horizons: 2min, 4min, 8min | Threshold: 12.5pt (50 ticks)")

        signals_df = label_outcomes(signals_df, ohlcv_df)

        # Log outcome distribution for each horizon
        for horizon in ['2min', '4min', '8min']:
            col = f'outcome_{horizon}'
            if col in signals_df.columns:
                outcome_dist = signals_df[col].value_counts().to_dict()
                logger.info(f"    {horizon} outcome distribution: {outcome_dist}")

        return {'signals_df': signals_df}
