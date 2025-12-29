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
    
    Outcome labeling (triple-barrier, anchored at level):
    - Price must touch the level within the horizon window
    - After touch, the first barrier hit determines outcome
    - Barriers are volatility-scaled (dynamic) using near-term realized vol
    
    First barrier hit determines label:
    - Hit break first → BREAK
    - Hit bounce first → BOUNCE
    - Hit neither → CHOP (vertical barrier)
    
    Policy B: Anchors up to 13:30 ET, forward window can spillover for labels.
    
    Thresholds are dynamic, derived from recent realized volatility and
    scaled by horizon length. Static thresholds are fallback only.
    
    Multi-timeframe mode generates outcomes at 2min, 4min, 8min horizons
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

    # Multi-timeframe windows or single window
    if use_multi_timeframe:
        confirmation_windows = CONFIG.CONFIRMATION_WINDOWS_MULTI  # [120, 240, 480]
        window_labels = ['2min', '4min', '8min']
    else:
        confirmation_windows = [confirmation_seconds or (lookforward_minutes * 60)]
        window_labels = ['']
    
    n = len(signals_df)
    signal_ts = signals_df['ts_ns'].values.astype(np.int64)
    directions = signals_df['direction'].values
    if 'level_price' not in signals_df.columns:
        raise ValueError("Missing level_price for outcome labeling.")
    level_prices = signals_df['level_price'].values.astype(np.float64)
    entry_prices = signals_df['entry_price'].values.astype(np.float64) if 'entry_price' in signals_df.columns else None

    vol_window_minutes = max(1, int(round(CONFIG.LABEL_VOL_WINDOW_SECONDS / 60)))
    barrier_min = CONFIG.LABEL_BARRIER_MIN_POINTS
    barrier_max = CONFIG.LABEL_BARRIER_MAX_POINTS
    barrier_scale = CONFIG.LABEL_BARRIER_SCALE
    t1_fraction = CONFIG.LABEL_T1_FRACTION
    monitor_band = CONFIG.MONITOR_BAND

    def _sigma_per_minute(anchor_idx: int) -> Optional[float]:
        if anchor_idx <= 0:
            return None
        start_idx = max(1, anchor_idx - vol_window_minutes)
        returns = np.diff(ohlcv_close[start_idx:anchor_idx + 1])
        if len(returns) < 2:
            return None
        return float(np.std(returns))

    def _atr_fallback(anchor_idx: int) -> Optional[float]:
        if anchor_idx <= 0:
            return None
        window = max(1, CONFIG.ATR_WINDOW_MINUTES)
        start_idx = max(1, anchor_idx - window)
        highs = ohlcv_high[start_idx:anchor_idx + 1]
        lows = ohlcv_low[start_idx:anchor_idx + 1]
        closes = ohlcv_close[start_idx:anchor_idx + 1]
        if len(closes) == 0:
            return None
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
        if len(tr) == 0:
            return None
        return float(np.mean(tr))
    
    # Storage for multi-timeframe results
    results_by_window = {}
    
    # Process each confirmation window
    for window_sec, label in zip(confirmation_windows, window_labels):
        suffix = f'_{label}' if label else ''
        horizon_minutes = window_sec / 60.0
        horizon_ns = int(window_sec * 1e9)
        
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
            if entry_prices is not None:
                if abs(entry_prices[i] - level_prices[i]) > monitor_band:
                    outcomes[i] = 'UNDEFINED'
                    continue

            start_idx = np.searchsorted(ohlcv_ts, ts, side='left')
            if start_idx < 0 or start_idx >= len(ohlcv_ts):
                outcomes[i] = 'UNDEFINED'
                continue

            end_idx = np.searchsorted(ohlcv_ts, ts + horizon_ns, side='right')
            if start_idx >= end_idx:
                outcomes[i] = 'UNDEFINED'
                continue

            direction = directions[i]
            level_price = level_prices[i]

            sigma_per_min = _sigma_per_minute(start_idx)
            if sigma_per_min is None or sigma_per_min <= 0:
                sigma_per_min = _atr_fallback(start_idx)
            if sigma_per_min is None or sigma_per_min <= 0:
                barrier = outcome_threshold
            else:
                barrier = sigma_per_min * np.sqrt(horizon_minutes) * barrier_scale
                barrier = max(barrier_min, min(barrier, barrier_max))

            threshold_2 = barrier
            threshold_1 = max(1e-6, threshold_2 * t1_fraction)

            window_high = ohlcv_high[start_idx:end_idx]
            window_low = ohlcv_low[start_idx:end_idx]
            window_close = ohlcv_close[start_idx:end_idx]
            window_ts = ohlcv_ts[start_idx:end_idx]

            if len(window_close) == 0:
                outcomes[i] = 'UNDEFINED'
                continue

            # Require level touch before evaluating barrier outcomes
            touch_mask = (window_low <= level_price) & (window_high >= level_price)
            if not np.any(touch_mask):
                outcomes[i] = 'CHOP'
                future_prices[i] = window_close[-1]
                continue

            touch_offset = int(np.argmax(touch_mask))
            touch_ts = window_ts[touch_offset]
            confirm_ts_ns[i] = touch_ts
            anchor_spot[i] = window_close[touch_offset]

            post_high = window_high[touch_offset:]
            post_low = window_low[touch_offset:]
            post_ts = window_ts[touch_offset:]

            if direction == 'UP':
                break_1 = np.where(post_high >= level_price + threshold_1)[0]
                break_2 = np.where(post_high >= level_price + threshold_2)[0]
                bounce_1 = np.where(post_low <= level_price - threshold_1)[0]
                bounce_2 = np.where(post_low <= level_price - threshold_2)[0]
            else:
                break_1 = np.where(post_low <= level_price - threshold_1)[0]
                break_2 = np.where(post_low <= level_price - threshold_2)[0]
                bounce_1 = np.where(post_high >= level_price + threshold_1)[0]
                bounce_2 = np.where(post_high >= level_price + threshold_2)[0]

            if len(break_1) > 0:
                idx = break_1[0]
                time_to_break_1[i] = (post_ts[idx] - touch_ts) / 1e9
            if len(bounce_1) > 0:
                idx = bounce_1[0]
                time_to_bounce_1[i] = (post_ts[idx] - touch_ts) / 1e9

            if len(break_2) > 0:
                idx = break_2[0]
                time_to_break_2[i] = (post_ts[idx] - touch_ts) / 1e9
            if len(bounce_2) > 0:
                idx = bounce_2[0]
                time_to_bounce_2[i] = (post_ts[idx] - touch_ts) / 1e9

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

            max_above = max(post_high.max() - level_price, 0.0)
            max_below = max(level_price - post_low.min(), 0.0)

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

            future_prices[i] = window_close[-1]
        
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
