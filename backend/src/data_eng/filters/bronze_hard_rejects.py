"""
Bronze layer hard reject filters for impossible values.

These filters remove data that is clearly invalid and should never
enter the pipeline. Unlike soft flags, these result in row deletion.

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Rejection reason codes
REJECT_REASON_ZERO_PRICE = "zero_price"
REJECT_REASON_ZERO_SIZE = "zero_size"
REJECT_REASON_CROSSED_BOOK = "crossed_book"
REJECT_REASON_NULL_TIMESTAMP = "null_timestamp"
REJECT_REASON_NEGATIVE_VALUES = "negative_values"

# Rejection statistics column
REJECT_STATS_KEY = "_bronze_reject_stats"


def apply_bronze_hard_rejects(
    df: pd.DataFrame,
    price_col: str = "price",
    size_col: str = "size",
    action_col: str = "action",
    ts_col: str = "ts_event",
    return_stats: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]] | pd.DataFrame:
    """Apply hard rejection filters for bronze layer ingest.
    
    Removes rows with:
    - Zero/negative price for add/modify actions
    - Zero/negative size for add/modify actions
    - Null timestamps
    
    Args:
        df: Raw DataFrame from data source
        price_col: Column name for price
        size_col: Column name for size
        action_col: Column name for action type
        ts_col: Column name for timestamp
        return_stats: If True, return (df, stats_dict)
        
    Returns:
        pd.DataFrame: Filtered DataFrame
        Dict[str, int]: (optional) Rejection counts by reason
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    original_len = len(df)
    stats = {
        REJECT_REASON_ZERO_PRICE: 0,
        REJECT_REASON_ZERO_SIZE: 0,
        REJECT_REASON_NULL_TIMESTAMP: 0,
        REJECT_REASON_NEGATIVE_VALUES: 0,
        "total_rejected": 0,
        "original_count": original_len,
    }
    
    if df.empty:
        if return_stats:
            return df, stats
        return df
    
    mask_keep = pd.Series(True, index=df.index)
    
    # Reject null timestamps
    if ts_col in df.columns:
        null_ts = df[ts_col].isna()
        stats[REJECT_REASON_NULL_TIMESTAMP] = int(null_ts.sum())
        mask_keep &= ~null_ts
    
    # For add/modify actions, price must be positive
    if price_col in df.columns and action_col in df.columns:
        price = df[price_col].astype(float)
        needs_price = df[action_col].isin(["A", "M"])  # Add, Modify
        zero_price = needs_price & (price <= 0)
        stats[REJECT_REASON_ZERO_PRICE] = int(zero_price.sum())
        mask_keep &= ~zero_price
    
    # For add/modify actions, size must be positive
    if size_col in df.columns and action_col in df.columns:
        size = df[size_col].astype(float)
        needs_size = df[action_col].isin(["A", "M"])  # Add, Modify
        zero_size = needs_size & (size <= 0)
        stats[REJECT_REASON_ZERO_SIZE] = int(zero_size.sum())
        mask_keep &= ~zero_size
    
    # General negative value check (beyond what's already caught)
    if size_col in df.columns:
        size = df[size_col].astype(float)
        negative_size = size < 0
        negative_count = int((negative_size & mask_keep).sum())  # Only count new rejects
        if negative_count > 0:
            stats[REJECT_REASON_NEGATIVE_VALUES] = negative_count
            mask_keep &= ~negative_size
    
    df_filtered = df[mask_keep].copy()
    stats["total_rejected"] = original_len - len(df_filtered)
    
    if return_stats:
        return df_filtered, stats
    return df_filtered


def apply_bronze_crossed_book_filter(
    df: pd.DataFrame,
    bid_col: str = "bid_px",
    ask_col: str = "ask_px",
    return_stats: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]] | pd.DataFrame:
    """Hard reject crossed book quotes at bronze level.
    
    This is for quote-level data where bid >= ask is impossible.
    
    Args:
        df: DataFrame with bid/ask columns
        bid_col: Column name for bid price
        ask_col: Column name for ask price
        return_stats: If True, return (df, stats_dict)
        
    Returns:
        pd.DataFrame: Filtered DataFrame
        Dict[str, int]: (optional) Rejection counts
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    original_len = len(df)
    stats = {
        REJECT_REASON_CROSSED_BOOK: 0,
        "total_rejected": 0,
        "original_count": original_len,
    }
    
    if df.empty:
        if return_stats:
            return df, stats
        return df
    
    if bid_col not in df.columns or ask_col not in df.columns:
        if return_stats:
            return df, stats
        return df
    
    bid = df[bid_col].astype(float)
    ask = df[ask_col].astype(float)
    
    # Only check where both are valid (positive)
    valid_quotes = (bid > 0) & (ask > 0)
    crossed = valid_quotes & (bid >= ask)
    
    stats[REJECT_REASON_CROSSED_BOOK] = int(crossed.sum())
    df_filtered = df[~crossed].copy()
    stats["total_rejected"] = original_len - len(df_filtered)
    
    if return_stats:
        return df_filtered, stats
    return df_filtered
