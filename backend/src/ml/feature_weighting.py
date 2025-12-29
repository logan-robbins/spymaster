"""
Feature weighting utilities for ML models.

Applies domain knowledge to weight features based on their empirical importance
and theoretical reliability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from src.common.config import CONFIG


def get_feature_weights(columns: List[str]) -> Dict[str, float]:
    """
    Get feature weights for ML model training.
    
    Based on empirical evidence and domain knowledge:
    - Gamma features: 0.3x (small relative to ES volume, pinning only)
    - Liquidity features: 1.0x (primary driver - order book depth)
    - Tape features: 1.0x (directional flow, validated driver)
    - Kinematic features: 1.0x (setup encoding for kNN)
    - OFI features: 1.0x (order flow pressure)
    
    Args:
        columns: List of feature column names
        
    Returns:
        Dict mapping column name to weight multiplier (0.0-1.0)
    """
    weights = {}
    
    for col in columns:
        # Default weight
        weight = 1.0
        
        # Downweight gamma features (evidence: 0.04-0.17% of ES volume)
        if any(prefix in col for prefix in ['gamma_', 'gex_', 'fuel_', 'call_gex', 'put_gex']):
            weight = CONFIG.GAMMA_FEATURE_WEIGHT  # Default 0.3
        
        # Liquidity features at full weight (primary driver)
        elif any(prefix in col for prefix in ['barrier_', 'wall_ratio']):
            weight = 1.0
        
        # Tape features at full weight (directional flow)
        elif any(prefix in col for prefix in ['tape_', 'sweep_']):
            weight = 1.0
        
        # Kinematic features at full weight (setup shape for kNN)
        elif any(prefix in col for prefix in ['velocity_', 'acceleration_', 'jerk_', 'momentum_']):
            weight = 1.0
        
        # OFI features at full weight (order flow pressure)
        elif any(prefix in col for prefix in ['ofi_', 'integrated_ofi']):
            weight = 1.0
        
        # Structural context at full weight
        elif any(prefix in col for prefix in ['dist_to_', 'distance_', 'level_stacking']):
            weight = 1.0
        
        # All others at full weight
        else:
            weight = 1.0
        
        weights[col] = weight
    
    return weights


def apply_feature_weights(
    X: pd.DataFrame,
    columns: List[str] = None
) -> pd.DataFrame:
    """
    Apply feature weights by scaling columns.
    
    This downweights gamma features during training to reduce their
    influence in the learned model.
    
    Args:
        X: Feature matrix
        columns: Columns to weight (defaults to all X columns)
        
    Returns:
        Weighted feature matrix
    """
    if columns is None:
        columns = list(X.columns)
    
    weights = get_feature_weights(columns)
    
    X_weighted = X.copy()
    for col in columns:
        if col in weights and col in X_weighted.columns:
            X_weighted[col] = X_weighted[col] * weights[col]
    
    return X_weighted


def get_sample_weights_from_gamma(
    df: pd.DataFrame,
    gamma_col: str = 'gamma_exposure'
) -> np.ndarray:
    """
    Generate sample weights that downweight high-gamma events.
    
    Alternative to feature weighting: reduce importance of training examples
    where gamma dominates the signal (to prevent model overfitting to gamma).
    
    Args:
        df: Training dataframe
        gamma_col: Column name for gamma exposure
        
    Returns:
        Array of sample weights (1.0 for low gamma, 0.5 for high gamma)
    """
    if gamma_col not in df.columns:
        return np.ones(len(df))
    
    gamma = df[gamma_col].abs().values
    gamma_percentile = np.percentile(gamma, 75)  # 75th percentile
    
    # Downweight samples with very high gamma (top 25%)
    weights = np.where(
        gamma > gamma_percentile,
        0.5,  # Half weight for high-gamma events
        1.0   # Full weight for normal events
    )
    
    return weights

