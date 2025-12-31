"""Stream normalization for Pentaview - STREAMS.md Section 2."""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Normalization method classification per STREAMS.md Section 2.1
STREAM_ROBUST_FEATURES = {
    # Tape/flow (heavy-tailed)
    'tape_velocity', 'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol',
    'tape_log_ratio', 'tape_log_total',
    # OFI
    'ofi_30s', 'ofi_60s', 'ofi_120s', 'ofi_300s',
    'ofi_near_level_30s', 'ofi_near_level_60s', 'ofi_near_level_120s', 'ofi_near_level_300s',
    'ofi_acceleration',
    # Barrier/liquidity (heavy-tailed, log-transformed)
    'barrier_delta_liq', 'barrier_delta_liq_log',
    'barrier_delta_1min', 'barrier_delta_3min', 'barrier_delta_5min',
    'barrier_pct_change_1min', 'barrier_pct_change_3min', 'barrier_pct_change_5min',
    'barrier_depth_current', 'barrier_replenishment_ratio',
    'wall_ratio', 'wall_ratio_log',
    # Physics proxies
    'accel_residual', 'force_mass_ratio', 'mass_proxy', 'force_proxy', 'flow_alignment',
    # GEX
    'gex_asymmetry', 'gex_ratio', 'net_gex_2strike', 'gamma_exposure',
    'gex_above_1strike', 'gex_below_1strike', 'call_gex_above_2strike', 'put_gex_below_2strike',
    # Trends
    'barrier_replenishment_trend', 'barrier_delta_liq_trend', 'tape_velocity_trend', 'tape_imbalance_trend'
}

STREAM_ZSCORE_FEATURES = {
    # Kinematics (approximately symmetric)
    'velocity_1min', 'velocity_3min', 'velocity_5min', 'velocity_10min', 'velocity_20min',
    'acceleration_1min', 'acceleration_3min', 'acceleration_5min', 'acceleration_10min', 'acceleration_20min',
    'jerk_1min', 'jerk_3min', 'jerk_5min', 'jerk_10min', 'jerk_20min',
    'momentum_trend_3min', 'momentum_trend_5min', 'momentum_trend_10min', 'momentum_trend_20min',
    # Approach
    'approach_velocity', 'approach_distance_atr',
    # DCT coefficients for trend/chop scoring
}

STREAM_PASSTHROUGH_FEATURES = {
    'fuel_effect', 'fuel_effect_encoded', 'barrier_state', 'barrier_state_encoded',
    'sweep_detected', 'or_active'
}


def compute_stream_normalization_stats(
    state_df: pd.DataFrame,
    stratify_by: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute robust normalization statistics for stream features.
    
    Per STREAMS.md Section 2.2:
    - Stratify by time_bucket, or_active, optionally vol_bucket
    - Compute median and MAD for robust normalization
    - Fall back to global stats if stratification bucket missing
    
    Args:
        state_df: State table DataFrame with all features
        stratify_by: Columns to stratify by (default: ['time_bucket'])
        output_path: Optional path to save stats JSON
    
    Returns:
        Dictionary with normalization statistics per feature per stratum
    """
    if state_df.empty:
        raise ValueError("Cannot compute stats from empty DataFrame")
    
    stratify_by = stratify_by or ['time_bucket']
    logger.info(f"Computing stream normalization stats with stratification: {stratify_by}")
    logger.info(f"  State table shape: {state_df.shape}")
    
    # Compute global stats (fallback)
    global_stats = {}
    for feature in STREAM_ROBUST_FEATURES | STREAM_ZSCORE_FEATURES:
        if feature not in state_df.columns:
            continue
        
        values = state_df[feature].dropna()
        if len(values) == 0:
            continue
        
        method = 'robust' if feature in STREAM_ROBUST_FEATURES else 'zscore'
        
        if method == 'robust':
            median = float(values.median())
            mad = float(np.median(np.abs(values - median)))
            global_stats[feature] = {
                'method': 'robust',
                'median': median,
                'mad': max(mad, 1e-8)  # Avoid division by zero
            }
        else:  # zscore
            mean = float(values.mean())
            std = float(values.std())
            global_stats[feature] = {
                'method': 'zscore',
                'mean': mean,
                'std': max(std, 1e-8)
            }
    
    # Compute stratified stats if requested
    stratified_stats = {}
    if stratify_by and all(col in state_df.columns for col in stratify_by):
        for stratum_key, group in state_df.groupby(stratify_by, dropna=False):
            if not isinstance(stratum_key, tuple):
                stratum_key = (stratum_key,)
            stratum_name = '_'.join(str(k) for k in stratum_key)
            
            stratified_stats[stratum_name] = {}
            for feature in STREAM_ROBUST_FEATURES | STREAM_ZSCORE_FEATURES:
                if feature not in group.columns:
                    continue
                
                values = group[feature].dropna()
                if len(values) < 30:  # Require minimum sample size
                    continue
                
                method = 'robust' if feature in STREAM_ROBUST_FEATURES else 'zscore'
                
                if method == 'robust':
                    median = float(values.median())
                    mad = float(np.median(np.abs(values - median)))
                    stratified_stats[stratum_name][feature] = {
                        'method': 'robust',
                        'median': median,
                        'mad': max(mad, 1e-8),
                        'n_samples': len(values)
                    }
                else:  # zscore
                    mean = float(values.mean())
                    std = float(values.std())
                    stratified_stats[stratum_name][feature] = {
                        'method': 'zscore',
                        'mean': mean,
                        'std': max(std, 1e-8),
                        'n_samples': len(values)
                    }
    
    # Package results
    stats = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'n_samples': len(state_df),
        'stratify_by': stratify_by,
        'global_stats': global_stats,
        'stratified_stats': stratified_stats
    }
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Saved stream normalization stats to {output_path}")
    
    logger.info(f"  Computed stats for {len(global_stats)} features")
    logger.info(f"  Stratified stats: {len(stratified_stats)} strata")
    
    return stats


def load_stream_normalization_stats(stats_path: Path) -> Dict[str, Any]:
    """Load stream normalization statistics from JSON file."""
    if not stats_path.exists():
        raise FileNotFoundError(f"Stream normalization stats not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    logger.info(f"Loaded stream normalization stats from {stats_path}")
    logger.info(f"  Version: {stats.get('version')}")
    logger.info(f"  Created: {stats.get('created_at')}")
    
    return stats


def normalize_feature_robust(
    value: float,
    median: float,
    mad: float,
    z_clip: float = 6.0,
    z_scale: float = 2.0
) -> float:
    """
    Robust normalization with tanh squashing per STREAMS.md Section 2.1.
    
    Formula:
        robust_z = (x - median) / (1.4826 * mad)
        clip_z = clamp(robust_z, -z_clip, +z_clip)
        norm = tanh(clip_z / z_scale)
    
    Args:
        value: Raw feature value
        median: Median from training set
        mad: Median absolute deviation from training set
        z_clip: Clip threshold (default 6.0)
        z_scale: Scale before tanh (default 2.0)
    
    Returns:
        Normalized value in (-1, +1)
    """
    if np.isnan(value):
        return 0.0
    
    robust_z = (value - median) / (1.4826 * mad + 1e-8)
    clip_z = np.clip(robust_z, -z_clip, z_clip)
    return float(np.tanh(clip_z / z_scale))


def normalize_feature_zscore(
    value: float,
    mean: float,
    std: float,
    z_clip: float = 6.0,
    z_scale: float = 2.0
) -> float:
    """
    Z-score normalization with tanh squashing per STREAMS.md Section 2.1.
    
    Formula:
        z = (x - mean) / std
        clip_z = clamp(z, -z_clip, +z_clip)
        norm = tanh(clip_z / z_scale)
    
    Args:
        value: Raw feature value
        mean: Mean from training set
        std: Standard deviation from training set
        z_clip: Clip threshold (default 6.0)
        z_scale: Scale before tanh (default 2.0)
    
    Returns:
        Normalized value in (-1, +1)
    """
    if np.isnan(value):
        return 0.0
    
    z = (value - mean) / (std + 1e-8)
    clip_z = np.clip(z, -z_clip, z_clip)
    return float(np.tanh(clip_z / z_scale))


def normalize_feature(
    feature_name: str,
    value: float,
    stats: Dict[str, Any],
    stratum: Optional[str] = None
) -> float:
    """
    Normalize a single feature value using precomputed statistics.
    
    Args:
        feature_name: Name of feature to normalize
        value: Raw feature value
        stats: Normalization statistics dictionary
        stratum: Optional stratum key for stratified stats
    
    Returns:
        Normalized value in (-1, +1) or passthrough
    """
    if np.isnan(value):
        return 0.0
    
    # Passthrough features
    if feature_name in STREAM_PASSTHROUGH_FEATURES:
        return float(value)
    
    # Try stratified stats first
    feature_stats = None
    if stratum and stratum in stats.get('stratified_stats', {}):
        feature_stats = stats['stratified_stats'][stratum].get(feature_name)
    
    # Fall back to global stats
    if feature_stats is None:
        feature_stats = stats.get('global_stats', {}).get(feature_name)
    
    if feature_stats is None:
        logger.warning(f"No normalization stats for feature {feature_name}, returning 0.0")
        return 0.0
    
    method = feature_stats['method']
    if method == 'robust':
        return normalize_feature_robust(value, feature_stats['median'], feature_stats['mad'])
    elif method == 'zscore':
        return normalize_feature_zscore(value, feature_stats['mean'], feature_stats['std'])
    else:
        return 0.0

