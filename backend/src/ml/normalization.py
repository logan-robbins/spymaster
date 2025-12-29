"""Normalization statistics computation - IMPLEMENTATION_READY.md Section 7."""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Feature classification per IMPLEMENTATION_READY.md Section 7.3
PASSTHROUGH_FEATURES = {
    'level_kind', 'direction', 'fuel_effect', 'barrier_state', 'sweep_detected'
}

ROBUST_FEATURES = {
    'tape_velocity', 'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol',
    'barrier_delta_liq', 'barrier_delta_1min', 'barrier_delta_3min', 'barrier_delta_5min',
    'ofi_30s', 'ofi_60s', 'ofi_120s', 'ofi_300s',
    'ofi_near_level_30s', 'ofi_near_level_60s', 'ofi_near_level_120s', 'ofi_near_level_300s',
    'wall_ratio', 'accel_residual', 'force_mass_ratio',
    'barrier_pct_change_1min', 'barrier_pct_change_3min', 'barrier_pct_change_5min',
    'tape_log_ratio', 'tape_log_total',
    'gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
    'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike', 'put_gex_below_2strike',
    'barrier_depth_current', 'barrier_replenishment_ratio',
    'barrier_replenishment_trend', 'barrier_delta_liq_trend', 'tape_velocity_trend', 'tape_imbalance_trend'
}

ZSCORE_FEATURES = {
    'velocity_1min', 'velocity_3min', 'velocity_5min', 'velocity_10min', 'velocity_20min',
    'acceleration_1min', 'acceleration_3min', 'acceleration_5min', 'acceleration_10min', 'acceleration_20min',
    'jerk_1min', 'jerk_3min', 'jerk_5min', 'jerk_10min', 'jerk_20min',
    'momentum_trend_3min', 'momentum_trend_5min', 'momentum_trend_10min', 'momentum_trend_20min',
    'distance_signed_atr', 'approach_velocity', 'approach_distance_atr',
    'predicted_accel', 'atr',
    'dist_to_pm_high_atr', 'dist_to_pm_low_atr', 'dist_to_or_high_atr', 'dist_to_or_low_atr',
    'dist_to_sma_200_atr', 'dist_to_sma_400_atr'
}

MINMAX_FEATURES = {
    'minutes_since_open', 'bars_since_open',
    'level_stacking_2pt', 'level_stacking_5pt', 'level_stacking_10pt',
    'prior_touches', 'attempt_index', 'approach_bars', 'attempt_cluster_id_mod'
}


def compute_normalization_stats(
    state_data: pd.DataFrame,
    feature_list: List[str],
    lookback_days: int = 60
) -> Dict[str, Any]:
    """
    Compute normalization statistics from historical state table data.
    
    Per IMPLEMENTATION_READY.md Section 7.2:
    - Robust: (x - median) / IQR, clip ±4σ
    - Z-Score: (x - mean) / std, clip ±4σ  
    - MinMax: (x - min) / (max - min), clip [0, 1]
    - Passthrough: No transformation
    
    Args:
        state_data: Historical state table (60+ days)
        feature_list: Features to compute stats for
        lookback_days: Days of history to use
    
    Returns:
        Dict with normalization statistics per feature
    """
    logger.info(f"Computing normalization statistics from {len(state_data):,} state samples...")
    
    stats = {
        'version': 1,
        'computed_date': datetime.now().strftime('%Y-%m-%d'),
        'lookback_days': lookback_days,
        'n_samples': len(state_data),
        'features': {}
    }
    
    for feature in feature_list:
        if feature not in state_data.columns:
            logger.warning(f"  Feature {feature} not found in state data, skipping")
            continue
        
        values = state_data[feature].dropna()
        
        if len(values) == 0:
            logger.warning(f"  Feature {feature} has no valid values, skipping")
            continue
        
        # Determine normalization method
        if feature in PASSTHROUGH_FEATURES:
            stats['features'][feature] = {'method': 'passthrough'}
            continue
        
        if feature in ROBUST_FEATURES:
            # Robust normalization: (x - median) / IQR
            median = float(values.median())
            q75 = float(values.quantile(0.75))
            q25 = float(values.quantile(0.25))
            iqr = q75 - q25
            
            stats['features'][feature] = {
                'method': 'robust',
                'center': median,
                'scale': iqr if iqr > 1e-6 else 1.0,
                'q25': q25,
                'q75': q75
            }
        
        elif feature in ZSCORE_FEATURES:
            # Z-score normalization: (x - mean) / std
            mean = float(values.mean())
            std = float(values.std())
            
            stats['features'][feature] = {
                'method': 'zscore',
                'center': mean,
                'scale': std if std > 1e-6 else 1.0
            }
        
        elif feature in MINMAX_FEATURES:
            # MinMax normalization: (x - min) / (max - min)
            min_val = float(values.min())
            max_val = float(values.max())
            
            stats['features'][feature] = {
                'method': 'minmax',
                'min': min_val,
                'max': max_val if max_val > min_val else min_val + 1.0
            }
        
        else:
            # Default to robust for unknown features
            logger.warning(f"  Feature {feature} not in any classification, using robust")
            median = float(values.median())
            iqr = float(values.quantile(0.75) - values.quantile(0.25))
            
            stats['features'][feature] = {
                'method': 'robust',
                'center': median,
                'scale': iqr if iqr > 1e-6 else 1.0
            }
    
    logger.info(f"  Computed stats for {len(stats['features'])} features")
    logger.info(f"    Robust: {sum(1 for s in stats['features'].values() if s.get('method') == 'robust')}")
    logger.info(f"    Z-Score: {sum(1 for s in stats['features'].values() if s.get('method') == 'zscore')}")
    logger.info(f"    MinMax: {sum(1 for s in stats['features'].values() if s.get('method') == 'minmax')}")
    logger.info(f"    Passthrough: {sum(1 for s in stats['features'].values() if s.get('method') == 'passthrough')}")
    
    return stats


def normalize_value(
    value: float,
    feature: str,
    stats: Dict[str, Any],
    clip_sigma: float = 4.0
) -> float:
    """
    Normalize a single feature value using computed statistics.
    
    Per IMPLEMENTATION_READY.md Section 7.4
    
    Args:
        value: Raw feature value
        feature: Feature name
        stats: Normalization statistics dict
        clip_sigma: Clipping threshold (±4σ for robust/zscore, [0,1] for minmax)
    
    Returns:
        Normalized value
    """
    if feature not in stats['features']:
        # No stats available, return as-is
        return value
    
    stat = stats['features'][feature]
    method = stat['method']
    
    if method == 'passthrough':
        return value
    
    elif method == 'robust':
        normalized = (value - stat['center']) / stat['scale']
        return np.clip(normalized, -clip_sigma, clip_sigma)
    
    elif method == 'zscore':
        normalized = (value - stat['center']) / stat['scale']
        return np.clip(normalized, -clip_sigma, clip_sigma)
    
    elif method == 'minmax':
        normalized = (value - stat['min']) / (stat['max'] - stat['min'])
        return np.clip(normalized, 0.0, 1.0)
    
    else:
        return value


def normalize_vector(
    raw_vector: np.ndarray,
    feature_names: List[str],
    stats: Dict[str, Any],
    clip_sigma: float = 4.0
) -> np.ndarray:
    """
    Normalize an entire feature vector.
    
    Args:
        raw_vector: Raw feature values (1D array)
        feature_names: Ordered feature names matching vector indices
        stats: Normalization statistics dict
        clip_sigma: Clipping threshold
    
    Returns:
        Normalized vector (same shape as input)
    """
    normalized = np.zeros_like(raw_vector, dtype=np.float32)
    
    for i, feature_name in enumerate(feature_names):
        normalized[i] = normalize_value(
            raw_vector[i],
            feature_name,
            stats,
            clip_sigma
        )
    
    return normalized


def save_normalization_stats(
    stats: Dict[str, Any],
    output_path: Path,
    create_symlink: bool = True
) -> Path:
    """
    Save normalization statistics to JSON with version number.
    
    Per IMPLEMENTATION_READY.md Section 7.5:
    Location: gold/normalization/stats_v{version}.json
    
    Args:
        stats: Normalization statistics dict
        output_path: Output directory
        create_symlink: Create 'current' symlink to latest version
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find next version number
    existing_files = list(output_path.glob('stats_v*.json'))
    if existing_files:
        versions = []
        for f in existing_files:
            try:
                version = int(f.stem.split('_v')[1])
                versions.append(version)
            except (IndexError, ValueError):
                pass
        next_version = max(versions, default=0) + 1
    else:
        next_version = 1
    
    stats['version'] = next_version
    
    # Save versioned file
    output_file = output_path / f'stats_v{next_version:03d}.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved normalization stats v{next_version} to {output_file}")
    
    # Create/update symlink to current version
    if create_symlink:
        symlink_path = output_path / 'current.json'
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(output_file.name)
        logger.info(f"  Updated 'current' symlink to v{next_version}")
    
    return output_file


def load_normalization_stats(stats_path: Path) -> Dict[str, Any]:
    """
    Load normalization statistics from JSON file.
    
    Args:
        stats_path: Path to stats file (or directory with 'current.json' symlink)
    
    Returns:
        Normalization statistics dict
    """
    stats_path = Path(stats_path)
    
    # If directory provided, use 'current' symlink
    if stats_path.is_dir():
        stats_path = stats_path / 'current.json'
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    logger.info(f"Loaded normalization stats v{stats['version']} from {stats_path}")
    logger.info(f"  Computed: {stats['computed_date']}, Lookback: {stats['lookback_days']} days, Samples: {stats['n_samples']:,}")
    
    return stats


class ComputeNormalizationStage:
    """
    Compute normalization statistics from historical state table data.
    
    Per IMPLEMENTATION_READY.md Section 7 (Stage 17):
    - Loads 60 days of state table data
    - Computes robust/zscore/minmax statistics per feature
    - Saves versioned JSON with statistics
    - Runs daily at 05:00 ET before market open
    
    This is typically run offline/scheduled, not in the main pipeline.
    """
    
    def __init__(
        self,
        state_table_dir: Path,
        output_dir: Path,
        lookback_days: int = 60
    ):
        self.state_table_dir = Path(state_table_dir)
        self.output_dir = Path(output_dir)
        self.lookback_days = lookback_days
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute normalization statistics computation.
        
        Returns:
            Dict with stats and output path
        """
        logger.info(f"Computing normalization statistics (lookback: {self.lookback_days} days)...")
        
        # Load historical state table data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        logger.info(f"  Loading state table from {start_date.date()} to {end_date.date()}...")
        
        # Load all state table files in date range
        state_dfs = []
        for date in pd.date_range(start=start_date, end=end_date, freq='D'):
            date_str = date.strftime('%Y-%m-%d')
            date_dir = self.state_table_dir / f'date={date_str}'
            
            if not date_dir.exists():
                continue
            
            parquet_files = list(date_dir.glob('*.parquet'))
            if parquet_files:
                for pq_file in parquet_files:
                    try:
                        df = pd.read_parquet(pq_file)
                        state_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"  Failed to load {pq_file}: {e}")
        
        if not state_dfs:
            raise ValueError(f"No state table data found in {self.state_table_dir}")
        
        state_data = pd.concat(state_dfs, ignore_index=True)
        logger.info(f"  Loaded {len(state_data):,} state samples from {len(state_dfs)} files")
        
        # Get all numeric feature columns (exclude identifiers and labels)
        exclude_cols = {
            'timestamp', 'ts_ns', 'date', 'event_id', 'level_price', 'level_active',
            'spot', 'entry_price',  # Exclude raw prices
            # Exclude all outcome/label columns
            'outcome', 'outcome_2min', 'outcome_4min', 'outcome_8min',
            'time_to_break_1', 'time_to_bounce_1',
            'time_to_break_1_2min', 'time_to_bounce_1_2min',
            'time_to_break_1_4min', 'time_to_bounce_1_4min',
            'time_to_break_1_8min', 'time_to_bounce_1_8min',
            'excursion_max', 'excursion_min', 'excursion_favorable', 'excursion_adverse',
            'strength_signed', 'strength_abs',
            'future_price', 'tradeable_1', 'tradeable_2'
        }
        
        feature_list = [
            col for col in state_data.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(state_data[col])
        ]
        
        logger.info(f"  Computing stats for {len(feature_list)} features...")
        
        # Compute statistics
        stats = compute_normalization_stats(
            state_data=state_data,
            feature_list=feature_list,
            lookback_days=self.lookback_days
        )
        
        # Save statistics
        output_file = save_normalization_stats(
            stats=stats,
            output_path=self.output_dir,
            create_symlink=True
        )
        
        return {
            'stats': stats,
            'output_file': str(output_file),
            'n_features': len(stats['features']),
            'n_samples': stats['n_samples']
        }

