"""Label outcomes stage."""
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
    Label outcomes using TRIPLE-BARRIER method (competing risks).
    
    Outcome labeling:
    - Break barrier: level + dir_sign × (+threshold_2)
    - Bounce barrier: level + dir_sign × (-threshold_2)
    - Vertical barrier: lookforward_minutes (if neither break/bounce)
    
    First barrier hit determines label:
    - Hit break first → BREAK
    - Hit bounce first → BOUNCE
    - Hit neither → CHOP (vertical barrier)
    
    Policy B: Anchors up to 13:30 ET, forward window can spillover for labels.
    
    Threshold for ES: 15 points ≈ 3 ATM strikes at 5-pt spacing.
    
    Multi-timeframe mode generates outcomes at 2min, 4min, 8min confirmations
    to enable training models for different trading horizons.

    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame (ES points from ES futures)
        lookforward_minutes: Forward window for labeling (defaults to CONFIG.LOOKFORWARD_MINUTES)
        outcome_threshold: Price move threshold for BREAK/BOUNCE (defaults to CONFIG.OUTCOME_THRESHOLD)
        use_multi_timeframe: If True, label at 2min/4min/8min; if False, use single confirmation

    Returns:
        DataFrame with outcome labels added (multi-timeframe or single)
    """
    if lookforward_minutes is None:
        lookforward_minutes = CONFIG.LOOKFORWARD_MINUTES
    if outcome_threshold is None:
        outcome_threshold = CONFIG.OUTCOME_THRESHOLD

    if signals_df.empty or ohlcv_df.empty:
        return signals_df

    # Prepare OHLCV data for fast lookup
    ohlcv_sorted = ohlcv_df.sort_values('timestamp')
    ohlcv_ts = ohlcv_sorted['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    ohlcv_open = ohlcv_sorted['open'].values.astype(np.float64)
    ohlcv_close = ohlcv_sorted['close'].values.astype(np.float64)
    ohlcv_high = ohlcv_sorted['high'].values.astype(np.float64)
    ohlcv_low = ohlcv_sorted['low'].values.astype(np.float64)

    # Lookforward in nanoseconds
    lookforward_ns = int(lookforward_minutes * 60 * 1e9)
    
    # Multi-timeframe windows or single window
    if use_multi_timeframe:
        confirmation_windows = CONFIG.CONFIRMATION_WINDOWS_MULTI  # [120, 240, 480]
        window_labels = ['2min', '4min', '8min']
    else:
        confirmation_windows = [confirmation_seconds or CONFIG.CONFIRMATION_WINDOW_SECONDS]
        window_labels = ['']
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values
    directions = signals_df['direction'].values
    bar_idx = signals_df['bar_idx'].values if 'bar_idx' in signals_df.columns else None
    if 'level_price' not in signals_df.columns:
        raise ValueError("Missing level_price for outcome labeling.")
    level_prices = signals_df['level_price'].values.astype(np.float64)
    threshold_1 = CONFIG.STRENGTH_THRESHOLD_1
    threshold_2 = CONFIG.STRENGTH_THRESHOLD_2
    
    # Storage for multi-timeframe results
    results_by_window = {}
    
    # Process each confirmation window
    for window_sec, label in zip(confirmation_windows, window_labels):
        suffix = f'_{label}' if label else ''
        confirmation_ns = int(window_sec * 1e9)
        
        # Initialize arrays for this window
        outcomes = np.empty(n, dtype=object)
        future_prices = np.full(n, np.nan, dtype=np.float64)
        excursion_max = np.full(n, np.nan, dtype=np.float64)
        excursion_min = np.full(n, np.nan, dtype=np.float64)
        strength_signed = np.full(n, np.nan, dtype=np.float64)
        strength_abs = np.full(n, np.nan, dtype=np.float64)
        time_to_threshold_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_threshold_2 = np.full(n, np.nan, dtype=np.float64)
        time_to_break_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_break_2 = np.full(n, np.nan, dtype=np.float64)
        time_to_bounce_1 = np.full(n, np.nan, dtype=np.float64)
        time_to_bounce_2 = np.full(n, np.nan, dtype=np.float64)
        tradeable_1 = np.zeros(n, dtype=np.int8)
        tradeable_2 = np.zeros(n, dtype=np.int8)
        confirm_ts_ns = np.full(n, -1, dtype=np.int64)
        anchor_spot = np.full(n, np.nan, dtype=np.float64)
        
        # Vectorized: find indices for each signal's lookforward window
        for i in range(n):
            ts = signal_ts[i]
            anchor_idx = None
            if bar_idx is not None:
                anchor_idx = int(bar_idx[i])
            else:
                anchor_idx = np.searchsorted(ohlcv_ts, ts, side='right') - 1

            if anchor_idx < 0 or anchor_idx >= len(ohlcv_ts):
                outcomes[i] = 'UNDEFINED'
                continue

            anchor_time = ohlcv_ts[anchor_idx] + confirmation_ns
            confirm_ts_ns[i] = anchor_time

            confirm_idx = np.searchsorted(ohlcv_ts, anchor_time, side='left')
            if confirm_idx >= len(ohlcv_ts):
                outcomes[i] = 'UNDEFINED'
                continue

            if confirm_idx < len(ohlcv_ts) and ohlcv_ts[confirm_idx] == anchor_time:
                anchor_spot[i] = ohlcv_open[confirm_idx]
            else:
                prior_idx = confirm_idx - 1
                if prior_idx < 0:
                    outcomes[i] = 'UNDEFINED'
                    continue
                anchor_spot[i] = ohlcv_close[prior_idx]

            start_idx = confirm_idx
            end_idx = np.searchsorted(ohlcv_ts, anchor_time + lookforward_ns, side='right')

            if start_idx >= len(ohlcv_ts) or start_idx >= end_idx:
                outcomes[i] = 'UNDEFINED'
                continue

            future_close = ohlcv_close[start_idx:end_idx]
            future_high = ohlcv_high[start_idx:end_idx]
            future_low = ohlcv_low[start_idx:end_idx]

            if len(future_close) == 0:
                outcomes[i] = 'UNDEFINED'
                continue

            direction = directions[i]
            level_price = level_prices[i]
            future_prices[i] = future_close[-1]

            max_above = max(future_high.max() - level_price, 0.0)
            max_below = max(level_price - future_low.min(), 0.0)

            above_1 = np.where(future_high >= level_price + threshold_1)[0]
            above_2 = np.where(future_high >= level_price + threshold_2)[0]
            below_1 = np.where(future_low <= level_price - threshold_1)[0]
            below_2 = np.where(future_low <= level_price - threshold_2)[0]

            if direction == 'UP':
                break_1 = above_1
                break_2 = above_2
                bounce_1 = below_1
                bounce_2 = below_2
            else:
                break_1 = below_1
                break_2 = below_2
                bounce_1 = above_1
                bounce_2 = above_2

            if len(break_1) > 0:
                idx = start_idx + break_1[0]
                time_to_break_1[i] = (ohlcv_ts[idx] - anchor_time) / 1e9
            if len(bounce_1) > 0:
                idx = start_idx + bounce_1[0]
                time_to_bounce_1[i] = (ohlcv_ts[idx] - anchor_time) / 1e9

            if len(break_2) > 0:
                idx = start_idx + break_2[0]
                time_to_break_2[i] = (ohlcv_ts[idx] - anchor_time) / 1e9
            if len(bounce_2) > 0:
                idx = start_idx + bounce_2[0]
                time_to_bounce_2[i] = (ohlcv_ts[idx] - anchor_time) / 1e9

            t1_candidates = [
                v for v in (time_to_break_1[i], time_to_bounce_1[i]) if np.isfinite(v)
            ]
            if t1_candidates:
                time_to_threshold_1[i] = min(t1_candidates)
                tradeable_1[i] = 1

            t2_candidates = [
                v for v in (time_to_break_2[i], time_to_bounce_2[i]) if np.isfinite(v)
            ]
            if t2_candidates:
                time_to_threshold_2[i] = min(t2_candidates)
                tradeable_2[i] = 1

            if direction == 'UP':
                excursion_max[i] = max_above
                excursion_min[i] = max_below
                strength_signed[i] = max_above - max_below
                strength_abs[i] = max(max_above, max_below)
            else:
                excursion_max[i] = max_below
                excursion_min[i] = max_above
                strength_signed[i] = max_below - max_above
                strength_abs[i] = max(max_below, max_above)
            
            break_t2 = time_to_break_2[i]
            bounce_t2 = time_to_bounce_2[i]
            if np.isfinite(break_t2) and np.isfinite(bounce_t2):
                if break_t2 < bounce_t2:
                    outcomes[i] = 'BREAK'
                elif bounce_t2 < break_t2:
                    outcomes[i] = 'BOUNCE'
                else:
                    if strength_signed[i] > 0:
                        outcomes[i] = 'BREAK'
                    elif strength_signed[i] < 0:
                        outcomes[i] = 'BOUNCE'
                    else:
                        outcomes[i] = 'CHOP'
            elif np.isfinite(break_t2):
                outcomes[i] = 'BREAK'
            elif np.isfinite(bounce_t2):
                outcomes[i] = 'BOUNCE'
            else:
                outcomes[i] = 'CHOP'
        
        # Store results for this window
        results_by_window[label] = {
            'outcomes': outcomes,
            'future_prices': future_prices,
            'excursion_max': excursion_max,
            'excursion_min': excursion_min,
            'strength_signed': strength_signed,
            'strength_abs': strength_abs,
            'time_to_threshold_1': time_to_threshold_1,
            'time_to_threshold_2': time_to_threshold_2,
            'time_to_break_1': time_to_break_1,
            'time_to_break_2': time_to_break_2,
            'time_to_bounce_1': time_to_bounce_1,
            'time_to_bounce_2': time_to_bounce_2,
            'tradeable_1': tradeable_1,
            'tradeable_2': tradeable_2,
            'confirm_ts_ns': confirm_ts_ns,
            'anchor_spot': anchor_spot,
        }
    
    # Build result DataFrame
    result = signals_df.copy()
    
    # Add columns for each timeframe
    for label, data in results_by_window.items():
        suffix = f'_{label}' if label else ''
        
        result[f'outcome{suffix}'] = data['outcomes']
        result[f'excursion_max{suffix}'] = data['excursion_max']
        result[f'excursion_min{suffix}'] = data['excursion_min']
        result[f'strength_signed{suffix}'] = data['strength_signed']
        result[f'strength_abs{suffix}'] = data['strength_abs']
        result[f'time_to_threshold_1{suffix}'] = data['time_to_threshold_1']
        result[f'time_to_threshold_2{suffix}'] = data['time_to_threshold_2']
        result[f'time_to_break_1{suffix}'] = data['time_to_break_1']
        result[f'time_to_break_2{suffix}'] = data['time_to_break_2']
        result[f'time_to_bounce_1{suffix}'] = data['time_to_bounce_1']
        result[f'time_to_bounce_2{suffix}'] = data['time_to_bounce_2']
        result[f'tradeable_1{suffix}'] = data['tradeable_1']
        result[f'tradeable_2{suffix}'] = data['tradeable_2']
        confirm_series = pd.Series(data['confirm_ts_ns']).mask(
            pd.Series(data['confirm_ts_ns']) < 0, pd.NA
        ).astype("Int64")
        result[f'confirm_ts_ns{suffix}'] = confirm_series
        result[f'anchor_spot{suffix}'] = data['anchor_spot']
        result[f'future_price{suffix}'] = data['future_prices']
    
    # For primary results, use 4min window (standard confirmation)
    if use_multi_timeframe and '4min' in results_by_window:
        primary_data = results_by_window['4min']
        result['outcome'] = primary_data['outcomes']
        result['excursion_max'] = primary_data['excursion_max']
        result['excursion_min'] = primary_data['excursion_min']
        result['strength_signed'] = primary_data['strength_signed']
        result['strength_abs'] = primary_data['strength_abs']
        result['time_to_threshold_1'] = primary_data['time_to_threshold_1']
        result['time_to_threshold_2'] = primary_data['time_to_threshold_2']
        result['time_to_break_1'] = primary_data['time_to_break_1']
        result['time_to_break_2'] = primary_data['time_to_break_2']
        result['time_to_bounce_1'] = primary_data['time_to_bounce_1']
        result['time_to_bounce_2'] = primary_data['time_to_bounce_2']
        result['tradeable_1'] = primary_data['tradeable_1']
        result['tradeable_2'] = primary_data['tradeable_2']
        result['confirm_ts_ns'] = pd.Series(primary_data['confirm_ts_ns']).mask(
            pd.Series(primary_data['confirm_ts_ns']) < 0, pd.NA
        ).astype("Int64")
        result['anchor_spot'] = primary_data['anchor_spot']
        result['future_price'] = primary_data['future_prices']

    return result


class LabelOutcomesStage(BaseStage):
    """Label outcomes using competing risks methodology.

    Computes forward-looking outcome labels based on whether
    price breaks through or bounces from the level.

    Outputs:
        signals_df: Updated with outcome labels:
            - outcome: BREAK, BOUNCE, CHOP
            - strength_signed: Signed magnitude of move
            - t1_60, t1_120: Confirmation timestamps
            - tradeable_1, tradeable_2: Tradeable flags
    """

    @property
    def name(self) -> str:
        return "label_outcomes"

    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_1min']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_1min']

        n_signals = len(signals_df)
        logger.info(f"  Labeling outcomes for {n_signals:,} signals...")
        logger.debug(f"    OHLCV bars: {len(ohlcv_df):,}")
        logger.debug(f"    Lookforward: {CONFIG.LOOKFORWARD_MINUTES} min, Threshold: ${CONFIG.OUTCOME_THRESHOLD}")

        signals_df = label_outcomes(signals_df, ohlcv_df)

        # Log outcome distribution
        if 'outcome' in signals_df.columns:
            outcome_dist = signals_df['outcome'].value_counts().to_dict()
            logger.info(f"    Outcome distribution: {outcome_dist}")

        # Log tradeable counts
        if 'tradeable_2' in signals_df.columns:
            tradeable_count = signals_df['tradeable_2'].sum()
            tradeable_pct = 100 * tradeable_count / n_signals if n_signals > 0 else 0
            logger.info(f"    Tradeable signals: {tradeable_count:,} ({tradeable_pct:.1f}%)")

        return {'signals_df': signals_df}
