"""Episode vector construction - Updated to 144D with DCT basis per Analyst Opinion."""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import dct

from src.ml.normalization import normalize_vector
from src.ml.constants import (
    VECTOR_DIMENSION, 
    VECTOR_SECTIONS,
    STATE_CADENCE_SEC,
    LOOKBACK_WINDOW_MIN
)

logger = logging.getLogger(__name__)


# Vector dimensions: 144D per analyst opinion
# Section boundaries# Check dimension consistency at module load
assert VECTOR_DIMENSION == 152, "Vector dimension must be 152"


def encode_fuel_effect(fuel_effect: str) -> float:
    """Encode fuel_effect."""
    mapping = {'AMPLIFY': 1.0, 'NEUTRAL': 0.0, 'DAMPEN': -1.0}
    return mapping.get(fuel_effect, 0.0)


def encode_barrier_state(barrier_state: str) -> float:
    """
    Encode barrier_state on monotone scale.
    Analyst opinion: map to support/resistance intensity.
    """
    mapping = {
        'STRONG_SUPPORT': 2.0, 'WEAK_SUPPORT': 1.0,
        'NEUTRAL': 0.0,
        'WEAK_RESISTANCE': -1.0, 'STRONG_RESISTANCE': -2.0
    }
    return mapping.get(barrier_state, 0.0)


def compute_dct_coefficients(series: np.ndarray, n_coeffs: int = 8) -> np.ndarray:
    """
    Compute DCT-II coefficients for a time series.
    
    Analyst opinion Section F: Use DCT to encode 20-minute trajectory shape
    in frequency domain (compact representation).
    
    Args:
        series: Time series array (e.g., 40 samples @ 30s = 20 minutes)
        n_coeffs: Number of DCT coefficients to keep (default 8)
    
    Returns:
        First n_coeffs DCT-II coefficients
    """
    if len(series) == 0:
        return np.zeros(n_coeffs, dtype=np.float32)
    
    # Handle NaN/inf
    series_clean = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute DCT-II (most common, used in JPEG/MP3)
    coeffs = dct(series_clean, type=2, norm='ortho')
    
    # Return first n_coeffs
    if len(coeffs) >= n_coeffs:
        return coeffs[:n_coeffs].astype(np.float32)
    else:
        # Pad with zeros if series shorter than n_coeffs
        padded = np.zeros(n_coeffs, dtype=np.float32)
        padded[:len(coeffs)] = coeffs
        return padded


def construct_episode_vector(
    current_bar: Dict[str, Any],
    history_buffer: List[Dict[str, Any]],
    trajectory_window: List[Dict[str, Any]],
    level_price: float
) -> np.ndarray:
    """
    Construct 144-dimensional episode vector with DCT trajectory basis.
    
    Per Analyst Opinion:
    - Section A: Context + Regime (25 dims) - removed redundant level_kind/direction
    - Section B: Multi-Scale Dynamics (37 dims) - multi-window kinematics at T=0
    - Section C: Micro-History (35 dims) - 5-bar history with LOG-TRANSFORMED features
    - Section D: Derived Physics (11 dims) - added mass_proxy, force_proxy, flow_alignment
    - Section E: Online Trends (4 dims) - rolling trends
    - Section F: Trajectory Basis (32 dims) - 4 series × 8 DCT coefficients
    
    Args:
        current_bar: Feature dict at anchor timestamp T=0
        history_buffer: Last 5 bars (T-4 to T=0) at 30s cadence (2.5 min)
        trajectory_window: Full 20-minute lookback (40 samples @ 30s cadence)
        level_price: Level price being tested
    
    Returns:
        Raw (unnormalized) 144-dimensional vector
    """
    vector = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    idx = 0
    
    # ─── SECTION A: Context + Regime (25 dims) ───
    # Note: level_kind and direction are partition keys (redundant to encode)
    vector[idx] = current_bar.get('minutes_since_open', 0.0)
    idx += 1
    vector[idx] = current_bar.get('bars_since_open', 0)
    idx += 1
    vector[idx] = current_bar.get('atr', 1.0)
    idx += 1
    
    # OR active flag (0/1) - analyst addition
    or_active = 1.0 if current_bar.get('minutes_since_open', 0) >= 15 else 0.0
    vector[idx] = or_active
    idx += 1
    
    # Level stacking (3)
    for f in ['level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt']:
        vector[idx] = current_bar.get(f, 0)
        idx += 1
    
    # Distances to all levels (6)
    for f in ['dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr',
              'dist_to_or_low_atr', 'dist_to_sma_90_atr', 'dist_to_ema_20_atr']:
        val = current_bar.get(f, 0.0)
        # Set OR distances to 0 if OR not active
        if 'or_' in f and or_active == 0.0:
            vector[idx] = 0.0
        else:
            vector[idx] = val if val is not None else 0.0
        idx += 1
    
    # Touch/attempt (3)
    vector[idx] = current_bar.get('prior_touches', 0)
    idx += 1
    vector[idx] = current_bar.get('attempt_index', 0)
    idx += 1
    # Handle time_since_last_touch (may be null in state table)
    time_since_last = current_bar.get('time_since_last_touch', 0.0)
    if time_since_last is None or (isinstance(time_since_last, float) and np.isnan(time_since_last)):
        time_since_last = 0.0  # Default: no prior touch (or very first touch)
    vector[idx] = float(time_since_last)
    idx += 1
    
    # GEX features (8)
    vector[idx] = current_bar.get('gamma_exposure', 0.0)
    idx += 1
    vector[idx] = encode_fuel_effect(current_bar.get('fuel_effect', 'NEUTRAL'))
    idx += 1
    for f in ['gex_ratio', 'gex_asymmetry', 'net_gex_2strike',
              'gex_above_1strike', 'gex_below_1strike', 
              'call_gex_above_2strike', 'put_gex_below_2strike']:
        vector[idx] = current_bar.get(f, 0.0)
        idx += 1
    
    # ─── SECTION B: Multi-Scale Dynamics (37 dims) ───
    # Velocity (6)
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'velocity_{scale}', 0.0)
        idx += 1
    
    # Acceleration (6)
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'acceleration_{scale}', 0.0)
        idx += 1
    
    # Jerk (6)
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'jerk_{scale}', 0.0)
        idx += 1
    
    # Momentum trend (4)
    for scale in ['3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'momentum_trend_{scale}', 0.0)
        idx += 1
    
    # OFI (4)
    for scale in ['30s', '60s', '120s', '300s']:
        vector[idx] = current_bar.get(f'ofi_{scale}', 0.0)
        idx += 1
    
    # OFI near level (4)
    for scale in ['30s', '60s', '120s', '300s']:
        vector[idx] = current_bar.get(f'ofi_near_level_{scale}', 0.0)
        idx += 1
    
    vector[idx] = current_bar.get('ofi_acceleration', 0.0)
    idx += 1
    
    # Barrier delta (3)
    for scale in ['1min', '3min', '5min']:
        vector[idx] = current_bar.get(f'barrier_delta_{scale}', 0.0)
        idx += 1
    
    # Barrier pct change (3)
    for scale in ['1min', '3min', '5min']:
        vector[idx] = current_bar.get(f'barrier_pct_change_{scale}', 0.0)
        idx += 1
    
    # Approach dynamics (3)
    vector[idx] = current_bar.get('dist_to_or_low_atr', 0.0)
    idx += 1
    vector[idx] = current_bar.get('dist_to_sma_90_atr', 0.0)
    idx += 1
    vector[idx] = current_bar.get('dist_to_ema_20_atr', 0.0)
    idx += 1
    
    # ─── SECTION C: Micro-History (35 dims) ───
    # Pad history if less than 5 bars
    history = list(history_buffer)
    while len(history) < 5:
        history.insert(0, history[0] if len(history) > 0 else current_bar)
    
    # Take last 5 bars (oldest first)
    history = history[-5:]
    
    # Analyst: Use LOG-TRANSFORMED barrier and wall features
    micro_features = [
        'd_atr',                    # distance_signed_atr (renamed for clarity)
        'tape_imbalance',
        'tape_velocity',
        'ofi_60s',
        'barrier_delta_liq_log',    # LOG-TRANSFORMED
        'wall_ratio_log',           # LOG-TRANSFORMED
        'gamma_exposure'
    ]
    
    for feature in micro_features:
        for bar in history:
            # Handle log features: compute on the fly if not pre-computed
            if feature == 'barrier_delta_liq_log':
                val = bar.get('barrier_delta_liq', 0.0)
                vector[idx] = np.log1p(abs(val)) * np.sign(val)  # signed log
            elif feature == 'wall_ratio_log':
                val = bar.get('wall_ratio', 1.0)
                vector[idx] = np.log(max(val, 1e-6))
            elif feature == 'd_atr':
                vector[idx] = bar.get('distance_signed_atr', 0.0)
            else:
                vector[idx] = bar.get(feature, 0.0)
            idx += 1
    
    # ─── SECTION D: Derived Physics (11 dims) ───
    vector[idx] = current_bar.get('predicted_accel', 0.0)
    idx += 1
    vector[idx] = current_bar.get('accel_residual', 0.0)
    idx += 1
    vector[idx] = current_bar.get('force_mass_ratio', 0.0)
    idx += 1
    
    # Analyst additions: mass_proxy and force_proxy
    barrier_depth = current_bar.get('barrier_depth_current', 0.0)
    mass_proxy = np.log1p(abs(barrier_depth))
    vector[idx] = mass_proxy
    idx += 1
    
    ofi_60s = current_bar.get('ofi_60s', 0.0)
    force_proxy = ofi_60s / (mass_proxy + 1e-6)
    vector[idx] = force_proxy
    idx += 1
    
    # Market Tide (Net Premium Flow) - Log Normalization
    vector[idx] = np.clip(current_bar.get("fuel_yield", 0.0), -10.0, 10.0)
    idx += 1
    vector[idx] = np.clip(current_bar.get("fuel_call_tide_log", 0.0), -10.0, 10.0)
    idx += 1
    vector[idx] = np.clip(current_bar.get("fuel_put_tide_log", 0.0), -10.0, 10.0)
    idx += 1
    vector[idx] = np.clip(current_bar.get("gamma_exposure", 0.0), -5.0, 5.0)
    idx += 1

    feature_val = current_bar.get('fuel_effect_encoded', 0.0)
    if not isinstance(feature_val, (int, float)):
        feature_val = 0.0 # Default fallback
    vector[idx] = feature_val
    idx += 1
    
    # Missing Feature Fix: Barrier State Encoded
    vector[idx] = encode_barrier_state(current_bar.get('barrier_state', 'NEUTRAL'))
    idx += 1
    
    vector[idx] = current_bar.get('barrier_replenishment_ratio', 0.0)
    idx += 1
    vector[idx] = 1.0 if current_bar.get('sweep_detected', False) else 0.0
    idx += 1
    
    # Tape log features
    tape_buy = current_bar.get('tape_buy_vol', 0.0) + 1.0
    tape_sell = current_bar.get('tape_sell_vol', 0.0) + 1.0
    vector[idx] = np.log(tape_buy / tape_sell)
    idx += 1
    vector[idx] = np.log(tape_buy + tape_sell)
    idx += 1
    
    # Analyst addition: flow_alignment
    d_atr = current_bar.get('distance_signed_atr', 0.0)
    flow_alignment = ofi_60s * (-np.sign(d_atr))  # positive = flow aligned with approach
    vector[idx] = flow_alignment
    idx += 1
    
    # ─── SECTION E: Online Trends (4 dims) ───
    vector[idx] = current_bar.get('barrier_replenishment_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('barrier_delta_liq_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('tape_velocity_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('tape_imbalance_trend', 0.0)
    idx += 1
    
    # ─── SECTION F: 20-Minute Trajectory Basis (32 dims) ───
    # 4 series × 8 DCT coefficients = 32 dims
    # Analyst: Explicitly encode "approach shape over time" using frequency domain
    
    # Extract time series from trajectory_window (40 samples @ 30s = 20 minutes)
    def extract_series(window: List[Dict[str, Any]], key: str) -> np.ndarray:
        """Extract time series from window, handle missing values."""
        series = []
        for bar in window:
            val = bar.get(key, 0.0)
            if key == 'barrier_delta_liq_log':
                # Compute log transform
                raw_val = bar.get('barrier_delta_liq', 0.0)
                val = np.log1p(abs(raw_val)) * np.sign(raw_val)
            series.append(val if val is not None else 0.0)
        return np.array(series, dtype=np.float32)
    
    # Series 1: d_atr trajectory (distance path)
    d_atr_series = extract_series(trajectory_window, 'distance_signed_atr')
    dct_d_atr = compute_dct_coefficients(d_atr_series, n_coeffs=8)
    vector[idx:idx+8] = dct_d_atr
    idx += 8
    
    # Series 2: ofi_60s trajectory (flow path)
    ofi_series = extract_series(trajectory_window, 'ofi_60s')
    dct_ofi = compute_dct_coefficients(ofi_series, n_coeffs=8)
    vector[idx:idx+8] = dct_ofi
    idx += 8
    
    # Series 3: barrier_delta_liq_log trajectory (liquidity path)
    barrier_series = extract_series(trajectory_window, 'barrier_delta_liq_log')
    dct_barrier = compute_dct_coefficients(barrier_series, n_coeffs=8)
    vector[idx:idx+8] = dct_barrier
    idx += 8
    
    # Series 4: tape_imbalance trajectory (aggression path)
    tape_series = extract_series(trajectory_window, 'tape_imbalance')
    dct_tape = compute_dct_coefficients(tape_series, n_coeffs=8)
    vector[idx:idx+8] = dct_tape
    idx += 8
    
    assert idx == VECTOR_DIMENSION, f"Vector dimension mismatch: {idx} != {VECTOR_DIMENSION}"
    
    return vector


def get_feature_names() -> List[str]:
    """
    Get ordered list of feature names matching vector indices.
    
    Returns list of 144 feature names for normalization.
    """
    names = []
    
    # Section A: Context + Regime (25) - removed level_kind/direction encodings
    names.extend([
        'minutes_since_open', 'bars_since_open', 'atr', 'or_active',
        'level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt',
        'dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr',
        'dist_to_or_low_atr', 'dist_to_sma_90_atr', 'dist_to_ema_20_atr',
        'prior_touches', 'attempt_index', 'time_since_last_touch_sec',
        'gamma_exposure', 'fuel_effect_encoded',
        'gex_ratio', 'gex_asymmetry', 'net_gex_2strike',
        'gex_above_1strike', 'gex_below_1strike',
        'call_gex_above_2strike', 'put_gex_below_2strike'
    ])
    
    # Section B: Multi-Scale Dynamics (37)
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        names.append(f'velocity_{scale}')
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        names.append(f'acceleration_{scale}')
    for scale in ['1min', '2min', '3min', '5min', '10min', '20min']:
        names.append(f'jerk_{scale}')
    for scale in ['3min', '5min', '10min', '20min']:
        names.append(f'momentum_trend_{scale}')
    for scale in ['30s', '60s', '120s', '300s']:
        names.append(f'ofi_{scale}')
    for scale in ['30s', '60s', '120s', '300s']:
        names.append(f'ofi_near_level_{scale}')
    names.append('ofi_acceleration')
    for scale in ['1min', '3min', '5min']:
        names.append(f'barrier_delta_{scale}')
    for scale in ['1min', '3min', '5min']:
        names.append(f'barrier_pct_change_{scale}')
    names.extend(['approach_velocity', 'approach_bars', 'approach_distance_atr'])
    
    # Section C: Micro-History (35) - 7 features × 5 bars with LOG transforms
    micro_features = [
        'd_atr', 'tape_imbalance', 'tape_velocity',
        'ofi_60s', 'barrier_delta_liq_log', 'wall_ratio_log', 'gamma_exposure'
    ]
    for feature in micro_features:
        for t in range(5):
            names.append(f'{feature}_t{t}')
    
    # Section D: Derived Physics (11) - added mass_proxy, force_proxy, flow_alignment
    names.extend([
        'predicted_accel', 'accel_residual', 'force_mass_ratio',
        'mass_proxy', 'force_proxy',
        "fuel_yield", "fuel_call_tide_log", "fuel_put_tide_log",
        "gamma_exposure", "fuel_effect_encoded",
        'barrier_state_encoded', 'barrier_replenishment_ratio',
        'sweep_detected', 'tape_log_ratio', 'tape_log_total',
        'flow_alignment'
    ])
    
    # Section E: Online Trends (4)
    names.extend([
        'barrier_replenishment_trend', 'barrier_delta_liq_trend',
        'tape_velocity_trend', 'tape_imbalance_trend'
    ])
    
    # Section F: Trajectory Basis (32) - 4 series × 8 DCT coefficients
    for series_name in ['d_atr', 'ofi_60s', 'barrier_delta_liq_log', 'tape_imbalance']:
        for k in range(8):
            names.append(f'dct_{series_name}_c{k}')
    
    assert len(names) == VECTOR_DIMENSION, f"Feature names count mismatch: {len(names)} != {VECTOR_DIMENSION}"
    
    return names


def assign_time_bucket(minutes_since_open: float) -> str:
    """
    Assign time bucket per Analyst Opinion (5 buckets).
    
    Split first 30 min to separate OR formation (0-15) from post-OR (15-30).
    """
    if minutes_since_open < 15:
        return 'T0_15'
    elif minutes_since_open < 30:
        return 'T15_30'
    elif minutes_since_open < 60:
        return 'T30_60'
    elif minutes_since_open < 120:
        return 'T60_120'
    else:
        return 'T120_180'


def compute_emission_weight(
    spot: float,
    level_price: float,
    atr: float,
    approach_velocity: float,
    ofi_60s: float
) -> float:
    """
    Compute episode quality weight per IMPLEMENTATION_READY.md Section 5.3.
    
    Quality weight based on:
    - Proximity to level (closer = higher weight)
    - Approach velocity (faster = more decisive)
    - OFI alignment with approach direction
    
    Returns:
        Weight in [0, 1], higher = better quality
    """
    distance_atr = abs(spot - level_price) / max(atr, 1e-6)
    
    # Proximity weight: 1.0 at level, 0.5 at 1 ATR, 0.1 at 3.5 ATR
    proximity_w = np.exp(-distance_atr / 1.5)
    
    # Velocity weight: clip to [0.2, 1.0]
    velocity_w = np.clip(abs(approach_velocity) / 2.0, 0.2, 1.0)
    
    # OFI alignment
    ofi_sign = np.sign(ofi_60s)
    approach_sign = np.sign(level_price - spot)  # positive if approaching from below
    ofi_aligned = (ofi_sign == approach_sign) or (ofi_sign == 0)
    ofi_w = 1.0 if ofi_aligned else 0.6
    
    return float(proximity_w * velocity_w * ofi_w)


def build_raw_vectors_from_state(
    state_df: pd.DataFrame,
    history_length: int = 5,
    trajectory_length: int = 40
) -> np.ndarray:
    """
    Build raw (unnormalized) episode vectors from state table rows.

    Uses each state row as an anchor and derives micro-history/trajectory windows
    within the same date and level_kind.
    """
    if state_df.empty:
        return np.array([])

    required_cols = {'date', 'timestamp', 'level_kind', 'level_price'}
    missing = required_cols - set(state_df.columns)
    if missing:
        raise ValueError(f"state_df missing required columns: {sorted(missing)}")

    state_sorted = state_df.sort_values(['date', 'level_kind', 'timestamp']).copy()

    vectors = []
    grouped = state_sorted.groupby(['date', 'level_kind'], sort=False)
    for _, group in grouped:
        records = group.to_dict('records')
        for idx, row in enumerate(records):
            if row.get('level_active') is False:
                continue
            level_price = row.get('level_price')
            if level_price is None or (isinstance(level_price, float) and np.isnan(level_price)):
                continue

            history_buffer = records[max(0, idx - (history_length - 1)):idx + 1]
            trajectory_window = records[max(0, idx - (trajectory_length - 1)):idx + 1]

            while len(history_buffer) < history_length:
                history_buffer.insert(0, history_buffer[0] if history_buffer else row)
            while len(trajectory_window) < trajectory_length:
                trajectory_window.insert(0, trajectory_window[0] if trajectory_window else row)

            vector = construct_episode_vector(
                current_bar=row,
                history_buffer=history_buffer,
                trajectory_window=trajectory_window,
                level_price=level_price
            )
            vectors.append(vector)

    return np.array(vectors, dtype=np.float32)


def construct_episodes_from_events(
    events_df: pd.DataFrame,
    state_df: pd.DataFrame,
    normalization_stats: Dict[str, Any],
    cadence_seconds: int = 30
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Construct episode vectors and metadata from event table and state table.
    
    Updated for 144D with DCT trajectory basis:
    - For each event (anchor), extract 5-bar micro-history from state table
    - Extract 40-bar (20-minute) trajectory window for DCT computation
    - Construct raw 144-dim vector
    - Normalize vector
    - Compute labels and emission weight
    - Return vectors array and metadata DataFrame
    
    Args:
        events_df: Event table (signals_df with all features)
        state_df: State table (30s cadence)
        normalization_stats: Normalization statistics dict
        cadence_seconds: State table cadence (default 30s)
    
    Returns:
        Tuple of (
            vectors array [N × 144], 
            metadata DataFrame [N rows],
            sequences array [N × 40 × 4]  (Raw trajectory for Transformer)
        )
    """
    if events_df.empty:
        return np.array([]), pd.DataFrame(), np.array([])
    
    logger.info(f"Constructing 144D episode vectors from {len(events_df):,} events...")
    
    # Sort state table by timestamp and level_kind for efficient lookup
    state_sorted = state_df.sort_values(['level_kind', 'timestamp']).copy()
    
    # Ensure events_df timestamps are timezone-aware (match state_df)
    events_sorted = events_df.copy()
    if events_sorted['timestamp'].dt.tz is None and state_sorted['timestamp'].dt.tz is not None:
        # Convert events to match state table timezone
        events_sorted['timestamp'] = events_sorted['timestamp'].dt.tz_localize('UTC').dt.tz_convert(state_sorted['timestamp'].dt.tz)
    elif events_sorted['timestamp'].dt.tz != state_sorted['timestamp'].dt.tz:
        # Convert to state table timezone
        events_sorted['timestamp'] = events_sorted['timestamp'].dt.tz_convert(state_sorted['timestamp'].dt.tz)
    
    # Decode events level_kind if integer (to match state table strings)
    level_kind_decode = {
        0: 'PM_HIGH', 1: 'PM_LOW', 2: 'OR_HIGH', 3: 'OR_LOW', 6: 'SMA_90', 12: 'EMA_20'
    }
    if events_sorted['level_kind'].dtype in [np.int8, np.int16, np.int32, np.int64, int]:
        events_sorted['level_kind'] = events_sorted['level_kind'].map(level_kind_decode).fillna('UNKNOWN')
        logger.info(f"  Decoded event level_kinds: {events_sorted['level_kind'].value_counts().to_dict()}")
    
    feature_names = get_feature_names()
    
    vectors = []
    metadata_rows = []
    
    sequences = []
    skipped_reasons = {'no_history': 0, 'vector_construction_failed': 0, 'normalization_failed': 0}
    
    for i, event in events_sorted.iterrows():
        event_ts = event['timestamp']
        level_kind = event['level_kind']
        level_price = event['level_price']
        direction = event.get('direction', 'UP')
        
        # Get state history for this level_kind
        level_state = state_sorted[state_sorted['level_kind'] == level_kind]
        
        # Get state rows at or before event timestamp
        history_states = level_state[level_state['timestamp'] <= event_ts]
        
        if len(history_states) < 1:
            skipped_reasons['no_history'] += 1
            if skipped_reasons['no_history'] <= 3:  # Log first few
                logger.info(f"  Event {i}: no state history for {level_kind} at {event_ts}")
            continue
        
        # Micro-history: Take last 5 state samples (2.5 minutes)
        history_buffer = history_states.tail(5).to_dict('records')
        current_bar = history_buffer[-1] if history_buffer else event.to_dict()
        
        # Trajectory window: Take last 40 state samples (20 minutes @ 30s cadence)
        trajectory_window = history_states.tail(40).to_dict('records')
        
        # Pad trajectory window if insufficient history (early in session)
        while len(trajectory_window) < 40:
            # Pad with first available state if needed
            first_state = trajectory_window[0] if trajectory_window else current_bar
            trajectory_window.insert(0, first_state)
        
        # Construct raw vector
        try:
            raw_vector = construct_episode_vector(
                current_bar=current_bar,
                history_buffer=history_buffer,
                trajectory_window=trajectory_window,
                level_price=level_price
            )
        except Exception as e:
            skipped_reasons['vector_construction_failed'] += 1
            if skipped_reasons['vector_construction_failed'] <= 3:
                logger.warning(f"  Event {i}: vector construction failed: {e}")
            continue
        
        # Normalize vector
        normalized_vector = normalize_vector(
            raw_vector=raw_vector,
            feature_names=feature_names,
            stats=normalization_stats
        )
        
        vectors.append(normalized_vector)
        
        # Extract raw sequence (40, 4)
        seq_array = np.zeros((40, 4), dtype=np.float32)
        for t, bar in enumerate(trajectory_window):
            # 0: d_atr
            seq_array[t, 0] = bar.get('distance_signed_atr', 0.0) or 0.0
            # 1: ofi_60s
            seq_array[t, 1] = bar.get('ofi_60s', 0.0) or 0.0
            # 2: barrier_delta_liq_log
            bl = bar.get('barrier_delta_liq', 0.0) or 0.0
            seq_array[t, 2] = np.log1p(abs(bl)) * np.sign(bl)
            # 3: tape_imbalance
            seq_array[t, 3] = bar.get('tape_imbalance', 0.0) or 0.0
        sequences.append(seq_array)
        
        # Build metadata row
        minutes_since_open = event.get('minutes_since_open', 0.0)
        time_bucket = assign_time_bucket(minutes_since_open)
        
        emission_weight = compute_emission_weight(
            spot=event.get('spot', level_price),
            level_price=level_price,
            atr=event.get('atr', 1.0),
            approach_velocity=event.get('approach_velocity', 0.0),
            ofi_60s=event.get('ofi_60s', 0.0)
        )
        
        metadata_row = {
            'event_id': event.get('event_id', f'evt_{i}'),
            'date': event.get('date'),
            'timestamp': event_ts,
            'ts_ns': event.get('ts_ns'),
            'level_kind': level_kind,
            'level_price': level_price,
            'direction': direction,
            'spot': event.get('spot', level_price),
            'atr': event.get('atr', 1.0),
            'minutes_since_open': minutes_since_open,
            'time_bucket': time_bucket,
            # Labels (multi-horizon)
            'outcome_2min': event.get('outcome_2min', 'CHOP'),
            'outcome_4min': event.get('outcome_4min', 'CHOP'),
            'outcome_8min': event.get('outcome_8min', 'CHOP'),
            # Continuous outcomes
            'excursion_favorable': event.get('excursion_favorable', 0.0),
            'excursion_adverse': event.get('excursion_adverse', 0.0),
            'strength_signed': event.get('strength_signed', 0.0),
            'strength_abs': event.get('strength_abs', 0.0),
            # Quality
            'emission_weight': emission_weight,
            # Attempt context
            'prior_touches': event.get('prior_touches', 0),
            'attempt_index': event.get('attempt_index', 0),
        }
        
        metadata_rows.append(metadata_row)
    
    vectors_array = np.array(vectors, dtype=np.float32)
    sequences_array = np.array(sequences, dtype=np.float32)
    metadata_df = pd.DataFrame(metadata_rows)
    
    logger.info(f"  Constructed {len(vectors_array):,} episode vectors (144 dims)")
    logger.info(f"  Constructed {len(sequences_array):,} raw sequences (40x4)")
    logger.info(f"  Skipped: no_history={skipped_reasons['no_history']}, vector_failed={skipped_reasons['vector_construction_failed']}, norm_failed={skipped_reasons['normalization_failed']}")
    
    return vectors_array, metadata_df, sequences_array


def save_episodes(
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    date: pd.Timestamp,
    sequences: np.ndarray = None
) -> Dict[str, Path]:
    """
    Save episode vectors and metadata.
    
    Output structure:
        gold/episodes/es_level_episodes/
        ├── vectors/date=YYYY-MM-DD/episodes.npy (144D)
        ├── sequences/date=YYYY-MM-DD/sequences.npy (40x4 Raw)
        └── metadata/date=YYYY-MM-DD/metadata.parquet
    
    Args:
        vectors: Episode vectors array [N × 144]
        metadata: Episode metadata DataFrame
        output_dir: Base output directory (gold/episodes/es_level_episodes/)
        date: Trading date
        sequences: Optional raw sequences array [N × 40 × 4]
    
    Returns:
        Dict with paths to saved files
    """
    output_dir = Path(output_dir)
    date_str = date.strftime('%Y-%m-%d')
    
    # Create date-partitioned directories
    vectors_dir = output_dir / 'vectors' / f'date={date_str}'
    metadata_dir = output_dir / 'metadata' / f'date={date_str}'
    sequences_dir = output_dir / 'sequences' / f'date={date_str}'
    
    vectors_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if sequences is not None:
        sequences_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vectors as numpy array
    vectors_path = vectors_dir / 'episodes.npy'
    np.save(vectors_path, vectors)
    
    # Save metadata as parquet
    metadata_path = metadata_dir / 'metadata.parquet'
    metadata.to_parquet(metadata_path, index=False)
    
    logger.info(f"Saved {len(vectors):,} episodes to {output_dir}")
    logger.info(f"  Vectors: {vectors_path}")
    logger.info(f"  Metadata: {metadata_path}")
    
    result = {
        'vectors': vectors_path,
        'metadata': metadata_path
    }

    # Save sequences if provided
    if sequences is not None:
        sequences_path = sequences_dir / 'sequences.npy'
        np.save(sequences_path, sequences)
        logger.info(f"  Sequences: {sequences_path}")
        result['sequences'] = sequences_path
    
    return result
