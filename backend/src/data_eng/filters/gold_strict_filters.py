"""
Gold layer strict filters for clean feature engineering.

These filters remove rows that have been soft-flagged in earlier layers.
The gold layer requires clean data for ML features.

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Optional

from .price_filters import FLAG_PRICE_OUTLIER, FLAG_CROSSED_MARKET, FLAG_SPREAD_ANOMALY
from .size_filters import FLAG_SIZE_OUTLIER
from .options_filters import FLAG_IV_ANOMALY, FLAG_ARBITRAGE_VIOLATION

# All possible flag columns
ALL_FLAG_COLUMNS = [
    FLAG_PRICE_OUTLIER,
    FLAG_CROSSED_MARKET,
    FLAG_SPREAD_ANOMALY,
    FLAG_SIZE_OUTLIER,
    FLAG_IV_ANOMALY,
    FLAG_ARBITRAGE_VIOLATION,
]


def filter_flagged_rows(
    df: pd.DataFrame,
    flag_columns: Optional[List[str]] = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, int]]:
    """Remove rows that have been flagged as suspicious.
    
    Args:
        df: DataFrame with flag columns from silver layer
        flag_columns: List of flag columns to check (default: all)
        return_stats: If True, return (df, stats_dict)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with flagged rows removed
        Dict[str, int]: (optional) Counts of rows removed per flag
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    original_len = len(df)
    stats = {"original_count": original_len}
    
    if df.empty:
        stats["total_filtered"] = 0
        if return_stats:
            return df, stats
        return df
    
    if flag_columns is None:
        flag_columns = ALL_FLAG_COLUMNS
    
    # Track which flags are present and their counts
    mask_keep = pd.Series(True, index=df.index)
    
    for flag_col in flag_columns:
        if flag_col in df.columns:
            flag_vals = df[flag_col].fillna(False).astype(bool)
            stats[flag_col] = int(flag_vals.sum())
            mask_keep &= ~flag_vals
        else:
            stats[flag_col] = 0
    
    df_filtered = df[mask_keep].copy()
    stats["total_filtered"] = original_len - len(df_filtered)
    
    if return_stats:
        return df_filtered, stats
    return df_filtered


def apply_gold_strict_filters(
    df: pd.DataFrame,
    product_type: str = "futures",
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, int]]:
    """Apply all strict filters appropriate for gold layer.
    
    This is the main entry point for gold-layer filtering.
    
    Args:
        df: DataFrame from silver layer (should have flag columns if applicable)
        product_type: 'futures', 'equities', 'futures_options', 'equity_options'
        return_stats: If True, return (df, stats_dict)
        
    Returns:
        pd.DataFrame: Clean DataFrame for feature engineering
        Dict[str, int]: (optional) Filtering statistics
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    original_len = len(df)
    all_stats = {"original_count": original_len}
    
    if df.empty:
        all_stats["total_filtered"] = 0
        if return_stats:
            return df, all_stats
        return df
    
    # Select appropriate flags based on product type
    if product_type in ("futures", "equities"):
        flags_to_check = [
            FLAG_PRICE_OUTLIER,
            FLAG_CROSSED_MARKET,
            FLAG_SPREAD_ANOMALY,
            FLAG_SIZE_OUTLIER,
        ]
    elif product_type in ("futures_options", "equity_options"):
        flags_to_check = [
            FLAG_PRICE_OUTLIER,
            FLAG_CROSSED_MARKET,
            FLAG_SPREAD_ANOMALY,
            FLAG_SIZE_OUTLIER,
            FLAG_IV_ANOMALY,
            FLAG_ARBITRAGE_VIOLATION,
        ]
    else:
        flags_to_check = ALL_FLAG_COLUMNS
    
    # Apply flag-based filtering
    df_filtered, flag_stats = filter_flagged_rows(
        df, flag_columns=flags_to_check, return_stats=True
    )
    all_stats.update(flag_stats)
    
    # Additional sanity checks (even without flags)
    # These catch issues that might not have been flagged in silver
    
    # 1. Remove rows with null/NaN critical columns
    critical_cols_by_type = {
        "futures": ["window_end_ts_ns", "spot_ref_price_int"],
        "equities": ["window_end_ts_ns", "spot_ref_price_int"],
        "futures_options": ["window_end_ts_ns", "spot_ref_price_int"],
        "equity_options": ["window_end_ts_ns", "spot_ref_price_int"],
    }
    
    critical_cols = critical_cols_by_type.get(product_type, ["window_end_ts_ns"])
    for col in critical_cols:
        if col in df_filtered.columns:
            pre_len = len(df_filtered)
            df_filtered = df_filtered.dropna(subset=[col])
            all_stats[f"null_{col}"] = pre_len - len(df_filtered)
    
    # 2. Remove rows with zero spot reference (invalid book state)
    if "spot_ref_price_int" in df_filtered.columns:
        pre_len = len(df_filtered)
        df_filtered = df_filtered[df_filtered["spot_ref_price_int"] > 0]
        all_stats["zero_spot_ref"] = pre_len - len(df_filtered)
    
    all_stats["final_count"] = len(df_filtered)
    all_stats["total_filtered"] = original_len - len(df_filtered)
    
    if return_stats:
        return df_filtered, all_stats
    return df_filtered


def drop_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all flag columns from DataFrame.
    
    Call this after filtering if you don't want flags in final output.
    
    Args:
        df: DataFrame with flag columns
        
    Returns:
        pd.DataFrame: DataFrame without flag columns
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    cols_to_drop = [col for col in ALL_FLAG_COLUMNS if col in df.columns]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df
