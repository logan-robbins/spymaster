"""Episode vector construction - IMPLEMENTATION_READY.md Section 6."""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from src.ml.normalization import normalize_vector

logger = logging.getLogger(__name__)


# Vector dimensions per IMPLEMENTATION_READY.md Section 6.8
VECTOR_SECTIONS = {
    'context_state': (0, 26),
    'multiscale_trajectory': (26, 63),
    'micro_history': (63, 98),
    'derived_physics': (98, 107),
    'cluster_trends': (107, 111),
}

VECTOR_DIMENSION = 111


def encode_level_kind(level_kind: str) -> float:
    """Encode level_kind as ordinal (Section 6.10)."""
    mapping = {
        'PM_HIGH': 0.0, 'PM_LOW': 1.0,
        'OR_HIGH': 2.0, 'OR_LOW': 3.0,
        'SMA_200': 4.0, 'SMA_400': 5.0
    }
    return mapping.get(level_kind, -1.0)


def encode_direction(direction: str) -> float:
    """Encode direction: UP=1, DOWN=-1 (Section 6.10)."""
    return 1.0 if direction == 'UP' else -1.0


def encode_fuel_effect(fuel_effect: str) -> float:
    """Encode fuel_effect (Section 6.10)."""
    mapping = {'AMPLIFY': 1.0, 'NEUTRAL': 0.0, 'DAMPEN': -1.0}
    return mapping.get(fuel_effect, 0.0)


def encode_barrier_state(barrier_state: str) -> float:
    """Encode barrier_state (Section 6.10)."""
    mapping = {
        'STRONG_SUPPORT': 2.0, 'WEAK_SUPPORT': 1.0,
        'NEUTRAL': 0.0,
        'WEAK_RESISTANCE': -1.0, 'STRONG_RESISTANCE': -2.0
    }
    return mapping.get(barrier_state, 0.0)


def construct_episode_vector(
    current_bar: Dict[str, Any],
    history_buffer: List[Dict[str, Any]],
    level_price: float
) -> np.ndarray:
    """
    Construct 111-dimensional episode vector from current state and history.
    
    Per IMPLEMENTATION_READY.md Section 6.9:
    - Section A: Context State (26 dims) - snapshot at T=0
    - Section B: Multi-Scale Trajectory (37 dims) - multi-window kinematics at T=0
    - Section C: Micro-History (35 dims) - 5-bar history of fast features
    - Section D: Derived Physics (9 dims) - force model, barrier state
    - Section E: Cluster Trends (4 dims) - rolling trends
    
    Args:
        current_bar: Feature dict at anchor timestamp T=0
        history_buffer: Last 5 bars (T-4 to T=0), oldest first
        level_price: Level price being tested
    
    Returns:
        Raw (unnormalized) 111-dimensional vector
    """
    vector = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    idx = 0
    
    # ─── SECTION A: Context State (26 dims) ───
    vector[idx] = encode_level_kind(current_bar.get('level_kind', ''))
    idx += 1
    vector[idx] = encode_direction(current_bar.get('direction', 'UP'))
    idx += 1
    vector[idx] = current_bar.get('minutes_since_open', 0.0)
    idx += 1
    vector[idx] = current_bar.get('bars_since_open', 0)
    idx += 1
    vector[idx] = current_bar.get('atr', 1.0)
    idx += 1
    
    # GEX features (8)
    for f in ['gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
              'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike',
              'put_gex_below_2strike']:
        vector[idx] = current_bar.get(f, 0.0)
        idx += 1
    
    vector[idx] = encode_fuel_effect(current_bar.get('fuel_effect', 'NEUTRAL'))
    idx += 1
    
    # Level stacking (3)
    for f in ['level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt']:
        vector[idx] = current_bar.get(f, 0)
        idx += 1
    
    # Distances to all levels (6)
    for f in ['dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr',
              'dist_to_or_low_atr', 'dist_to_sma_200_atr', 'dist_to_sma_400_atr']:
        val = current_bar.get(f, 0.0)
        vector[idx] = val if val is not None else 0.0
        idx += 1
    
    # Touch/attempt (3)
    vector[idx] = current_bar.get('prior_touches', 0)
    idx += 1
    vector[idx] = current_bar.get('attempt_index', 0)
    idx += 1
    vector[idx] = current_bar.get('attempt_cluster_id', 0) % 1000
    idx += 1
    
    # ─── SECTION B: Multi-Scale Trajectory (37 dims) ───
    # Velocity (5)
    for scale in ['1min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'velocity_{scale}', 0.0)
        idx += 1
    
    # Acceleration (5)
    for scale in ['1min', '3min', '5min', '10min', '20min']:
        vector[idx] = current_bar.get(f'acceleration_{scale}', 0.0)
        idx += 1
    
    # Jerk (5)
    for scale in ['1min', '3min', '5min', '10min', '20min']:
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
    vector[idx] = current_bar.get('approach_velocity', 0.0)
    idx += 1
    vector[idx] = current_bar.get('approach_bars', 0)
    idx += 1
    vector[idx] = current_bar.get('approach_distance_atr', 0.0)
    idx += 1
    
    # ─── SECTION C: Micro-History (35 dims) ───
    # Pad history if less than 5 bars
    history = list(history_buffer)
    while len(history) < 5:
        history.insert(0, history[0] if len(history) > 0 else current_bar)
    
    # Take last 5 bars (oldest first)
    history = history[-5:]
    
    micro_features = [
        'distance_signed_atr', 'tape_imbalance', 'tape_velocity',
        'ofi_60s', 'barrier_delta_liq', 'wall_ratio', 'gamma_exposure'
    ]
    
    for feature in micro_features:
        for bar in history:
            vector[idx] = bar.get(feature, 0.0)
            idx += 1
    
    # ─── SECTION D: Derived Physics (9 dims) ───
    vector[idx] = current_bar.get('predicted_accel', 0.0)
    idx += 1
    vector[idx] = current_bar.get('accel_residual', 0.0)
    idx += 1
    vector[idx] = current_bar.get('force_mass_ratio', 0.0)
    idx += 1
    vector[idx] = encode_barrier_state(current_bar.get('barrier_state', 'NEUTRAL'))
    idx += 1
    vector[idx] = current_bar.get('barrier_depth_current', 0.0)
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
    
    # ─── SECTION E: Cluster Trends (4 dims) ───
    vector[idx] = current_bar.get('barrier_replenishment_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('barrier_delta_liq_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('tape_velocity_trend', 0.0)
    idx += 1
    vector[idx] = current_bar.get('tape_imbalance_trend', 0.0)
    idx += 1
    
    assert idx == VECTOR_DIMENSION, f"Vector dimension mismatch: {idx} != {VECTOR_DIMENSION}"
    
    return vector


def get_feature_names() -> List[str]:
    """
    Get ordered list of feature names matching vector indices.
    
    Returns list of 111 feature names for normalization.
    """
    names = []
    
    # Section A: Context State (26)
    names.extend([
        'level_kind_encoded', 'direction_encoded', 'minutes_since_open', 'bars_since_open', 'atr',
        'gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
        'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike', 'put_gex_below_2strike',
        'fuel_effect_encoded',
        'level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt',
        'dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr',
        'dist_to_or_low_atr', 'dist_to_sma_200_atr', 'dist_to_sma_400_atr',
        'prior_touches', 'attempt_index', 'attempt_cluster_id_mod'
    ])
    
    # Section B: Multi-Scale Trajectory (37)
    for scale in ['1min', '3min', '5min', '10min', '20min']:
        names.append(f'velocity_{scale}')
    for scale in ['1min', '3min', '5min', '10min', '20min']:
        names.append(f'acceleration_{scale}')
    for scale in ['1min', '3min', '5min', '10min', '20min']:
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
    
    # Section C: Micro-History (35) - 7 features × 5 bars
    micro_features = [
        'distance_signed_atr', 'tape_imbalance', 'tape_velocity',
        'ofi_60s', 'barrier_delta_liq', 'wall_ratio', 'gamma_exposure'
    ]
    for feature in micro_features:
        for t in range(5):
            names.append(f'{feature}_t{t}')
    
    # Section D: Derived Physics (9)
    names.extend([
        'predicted_accel', 'accel_residual', 'force_mass_ratio',
        'barrier_state_encoded', 'barrier_depth_current', 'barrier_replenishment_ratio',
        'sweep_detected', 'tape_log_ratio', 'tape_log_total'
    ])
    
    # Section E: Cluster Trends (4)
    names.extend([
        'barrier_replenishment_trend', 'barrier_delta_liq_trend',
        'tape_velocity_trend', 'tape_imbalance_trend'
    ])
    
    assert len(names) == VECTOR_DIMENSION, f"Feature names count mismatch: {len(names)} != {VECTOR_DIMENSION}"
    
    return names


def assign_time_bucket(minutes_since_open: float) -> str:
    """Assign time bucket per IMPLEMENTATION_READY.md Section 5.2."""
    if minutes_since_open < 30:
        return 'T0_30'
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


def construct_episodes_from_events(
    events_df: pd.DataFrame,
    state_df: pd.DataFrame,
    normalization_stats: Dict[str, Any],
    cadence_seconds: int = 30
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Construct episode vectors and metadata from event table and state table.
    
    Per IMPLEMENTATION_READY.md Section 6 and Stage 18:
    - For each event (anchor), extract 5-bar history window from state table
    - Construct raw 111-dim vector
    - Normalize vector
    - Compute labels and emission weight
    - Return vectors array and metadata DataFrame
    
    Args:
        events_df: Event table (signals_df with all features)
        state_df: State table (30s cadence)
        normalization_stats: Normalization statistics dict
        cadence_seconds: State table cadence (default 30s)
    
    Returns:
        Tuple of (vectors array [N × 111], metadata DataFrame [N rows])
    """
    if events_df.empty:
        return np.array([]), pd.DataFrame()
    
    logger.info(f"Constructing episode vectors from {len(events_df):,} events...")
    
    # Sort state table by timestamp and level_kind for efficient lookup
    state_sorted = state_df.sort_values(['level_kind', 'timestamp']).copy()
    
    feature_names = get_feature_names()
    
    vectors = []
    metadata_rows = []
    
    for i, event in events_df.iterrows():
        event_ts = event['timestamp']
        level_kind = event['level_kind']
        level_price = event['level_price']
        direction = event.get('direction', 'UP')
        
        # Get state history for this level_kind
        level_state = state_sorted[state_sorted['level_kind'] == level_kind]
        
        # Get state rows at or before event timestamp
        history_states = level_state[level_state['timestamp'] <= event_ts]
        
        if len(history_states) < 1:
            continue
        
        # Take last 5 state samples (30s cadence = 2.5 minutes of history)
        history_buffer = history_states.tail(5).to_dict('records')
        current_bar = history_buffer[-1] if history_buffer else event.to_dict()
        
        # Construct raw vector
        try:
            raw_vector = construct_episode_vector(
                current_bar=current_bar,
                history_buffer=history_buffer,
                level_price=level_price
            )
        except Exception as e:
            logger.warning(f"Failed to construct vector for event {event.get('event_id', i)}: {e}")
            continue
        
        # Normalize vector
        normalized_vector = normalize_vector(
            raw_vector=raw_vector,
            feature_names=feature_names,
            stats=normalization_stats
        )
        
        vectors.append(normalized_vector)
        
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
    metadata_df = pd.DataFrame(metadata_rows)
    
    logger.info(f"  Constructed {len(vectors_array):,} episode vectors (111 dims)")
    
    return vectors_array, metadata_df


def save_episodes(
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    date: pd.Timestamp
) -> Dict[str, Path]:
    """
    Save episode vectors and metadata per IMPLEMENTATION_READY.md Section 2.2.
    
    Output structure:
        gold/episodes/es_level_episodes/
        ├── vectors/date=YYYY-MM-DD/episodes.npy
        └── metadata/date=YYYY-MM-DD/metadata.parquet
    
    Args:
        vectors: Episode vectors array [N × 111]
        metadata: Episode metadata DataFrame
        output_dir: Base output directory (gold/episodes/es_level_episodes/)
        date: Trading date
    
    Returns:
        Dict with paths to saved files
    """
    output_dir = Path(output_dir)
    date_str = date.strftime('%Y-%m-%d')
    
    # Create date-partitioned directories
    vectors_dir = output_dir / 'vectors' / f'date={date_str}'
    metadata_dir = output_dir / 'metadata' / f'date={date_str}'
    
    vectors_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vectors as numpy array
    vectors_path = vectors_dir / 'episodes.npy'
    np.save(vectors_path, vectors)
    
    # Save metadata as parquet
    metadata_path = metadata_dir / 'metadata.parquet'
    metadata.to_parquet(metadata_path, index=False)
    
    logger.info(f"Saved {len(vectors):,} episodes to {output_dir}")
    logger.info(f"  Vectors: {vectors_path}")
    logger.info(f"  Metadata: {metadata_path}")
    
    return {
        'vectors': vectors_path,
        'metadata': metadata_path
    }

