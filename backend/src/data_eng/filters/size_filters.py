"""
Size-based outlier filters for detecting anomalous trade/order sizes.

Thresholds:
- 99.9th percentile per product/side as upper bound
- 10x median as secondary check
- CME ES futures max: 30,000 contracts per order

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

# Flag column name
FLAG_SIZE_OUTLIER = "size_outlier_flag"

# Product-specific maximum sizes (contracts/shares)
MAX_SIZES = {
    "ES": 30_000,      # CME ES futures max order size
    "NQ": 20_000,      # CME NQ futures max order size
    "QQQ": 1_000_000,  # Reasonable equity order cap
    "SPY": 1_000_000,  # Reasonable equity order cap
}

# Percentile threshold
SIZE_PERCENTILE_THRESHOLD = 99.9

# Multiple of median threshold
SIZE_MEDIAN_MULTIPLIER = 10.0


def compute_size_percentiles(
    df: pd.DataFrame,
    size_col: str = "size",
    groupby_cols: Optional[list] = None,
) -> Dict[str, float]:
    """Compute size percentiles for outlier detection.
    
    Args:
        df: DataFrame with size data
        size_col: Column name for size
        groupby_cols: Optional columns to group by (e.g., ['side', 'symbol'])
        
    Returns:
        Dict with percentile thresholds
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    if size_col not in df.columns:
        return {"p50": 0, "p99": 0, "p999": 0}
    
    size = df[size_col].astype(float)
    valid_size = size[size > 0]
    
    if len(valid_size) == 0:
        return {"p50": 0, "p99": 0, "p999": 0}
    
    return {
        "p50": float(valid_size.quantile(0.50)),
        "p99": float(valid_size.quantile(0.99)),
        "p999": float(valid_size.quantile(0.999)),
    }


def apply_size_outlier_filter(
    df: pd.DataFrame,
    size_col: str = "size",
    product_symbol: Optional[str] = None,
    percentile_threshold: float = SIZE_PERCENTILE_THRESHOLD,
    median_multiplier: float = SIZE_MEDIAN_MULTIPLIER,
) -> pd.DataFrame:
    """Flag rows with anomalously large sizes.
    
    Flags if size exceeds:
    - Product-specific maximum (if defined)
    - percentile_threshold (default 99.9th percentile)
    - median_multiplier × median size (default 10x)
    
    Args:
        df: DataFrame with size data
        size_col: Column name for size
        product_symbol: Symbol for product-specific limits (e.g., 'ES', 'QQQ')
        percentile_threshold: Upper percentile to flag
        median_multiplier: Flag if > N × median
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_SIZE_OUTLIER column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if size_col not in df.columns:
        df[FLAG_SIZE_OUTLIER] = False
        return df
    
    size = df[size_col].astype(float)
    valid_size = size > 0
    
    # Get thresholds
    percentile_val = size[valid_size].quantile(percentile_threshold / 100.0)
    median_val = size[valid_size].median()
    
    # Product-specific max
    if product_symbol:
        # Extract root symbol (ESH6 -> ES, QQQ240119C00100000 -> QQQ)
        root = product_symbol[:2] if len(product_symbol) >= 2 else product_symbol
        if root in MAX_SIZES:
            max_size = MAX_SIZES[root]
        elif product_symbol[:3] in MAX_SIZES:
            max_size = MAX_SIZES[product_symbol[:3]]
        else:
            max_size = float('inf')
    else:
        max_size = float('inf')
    
    # Flag outliers
    above_percentile = size > percentile_val
    above_median_mult = size > (median_multiplier * median_val)
    above_max = size > max_size
    
    df[FLAG_SIZE_OUTLIER] = valid_size & (above_percentile | above_median_mult | above_max)
    
    return df


def apply_zero_size_filter(
    df: pd.DataFrame,
    size_col: str = "size",
    action_col: Optional[str] = "action",
) -> pd.DataFrame:
    """Filter rows with zero or negative size where size is required.
    
    For add/modify actions, size must be positive.
    For fills/cancels, size can be legitimately zero in some cases.
    
    Args:
        df: DataFrame with size data
        size_col: Column name for size
        action_col: Column name for action type (optional)
        
    Returns:
        pd.DataFrame: Filtered DataFrame (hard reject)
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    if size_col not in df.columns:
        return df
    
    size = df[size_col].astype(float)
    
    if action_col and action_col in df.columns:
        # Only require positive size for Add/Modify actions
        needs_size = df[action_col].isin(["A", "M"])
        bad_rows = needs_size & (size <= 0)
        return df[~bad_rows]
    else:
        # Reject all zero/negative sizes
        return df[size > 0]
