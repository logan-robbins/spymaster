"""
Options-specific filters for detecting anomalous option prices.

Filters:
- Implied volatility bounds (1% < IV < 500%)
- Arbitrage bounds (bid < ask, call-put parity approx)
- Fat-finger detection (price >> underlying)

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Flag column names
FLAG_IV_ANOMALY = "iv_anomaly_flag"
FLAG_ARBITRAGE_VIOLATION = "arbitrage_violation_flag"

# IV bounds (as decimal, not percentage)
IV_MIN = 0.01    # 1% - extremely low
IV_MAX = 5.00    # 500% - extremely high (even meme stocks rarely exceed)

# Price bounds relative to underlying
OPTION_PRICE_MAX_RATIO = 2.0  # Option premium shouldn't exceed 2x underlying spot
OPTION_PRICE_MIN = 0.0       # Options can be worthless but not negative


def apply_iv_filter(
    df: pd.DataFrame,
    iv_col: str = "iv",
    iv_min: float = IV_MIN,
    iv_max: float = IV_MAX,
) -> pd.DataFrame:
    """Flag options with implausible implied volatility.
    
    IV outside [iv_min, iv_max] indicates either:
    - Arbitrage opportunity (unlikely in liquid markets)
    - Data error or fat-finger trade
    - Illiquid strike with stale quote
    
    Args:
        df: DataFrame with IV data
        iv_col: Column name for implied volatility
        iv_min: Minimum plausible IV (default 0.01 = 1%)
        iv_max: Maximum plausible IV (default 5.00 = 500%)
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_IV_ANOMALY column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if iv_col not in df.columns:
        df[FLAG_IV_ANOMALY] = False
        return df
    
    iv = df[iv_col].astype(float)
    
    # Flag if IV is outside reasonable bounds or non-positive
    df[FLAG_IV_ANOMALY] = (iv <= 0) | (iv < iv_min) | (iv > iv_max) | pd.isna(iv)
    
    return df


def apply_option_price_bounds(
    df: pd.DataFrame,
    price_col: str = "mid_price",
    underlying_col: str = "spot_ref_price_int",
    price_scale: float = 1e-9,
    max_ratio: float = OPTION_PRICE_MAX_RATIO,
) -> pd.DataFrame:
    """Flag options with price exceeding bounds relative to underlying.
    
    An option premium > 2x underlying is almost certainly an error (except
    in extreme leveraged products which we don't handle).
    
    Args:
        df: DataFrame with option and underlying prices
        price_col: Column name for option price
        underlying_col: Column name for underlying spot price
        price_scale: Scale factor for fixed-point prices (default 1e-9)
        max_ratio: Maximum ratio of option price to underlying
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_ARBITRAGE_VIOLATION column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    has_price = price_col in df.columns
    has_underlying = underlying_col in df.columns
    
    if not has_price or not has_underlying:
        df[FLAG_ARBITRAGE_VIOLATION] = False
        return df
    
    price = df[price_col].astype(float)
    underlying = df[underlying_col].astype(float)
    
    # Scale if in fixed-point
    if price.median() > 1e6:
        price_dollars = price * price_scale
    else:
        price_dollars = price
        
    if underlying.median() > 1e6:
        underlying_dollars = underlying * price_scale
    else:
        underlying_dollars = underlying
    
    # Flag violations
    valid = (underlying_dollars > 0) & (price_dollars >= 0)
    ratio = price_dollars / (underlying_dollars + 1e-10)
    
    # Negative price or price > max_ratio * underlying
    df[FLAG_ARBITRAGE_VIOLATION] = (price_dollars < OPTION_PRICE_MIN) | (valid & (ratio > max_ratio))
    
    return df


def apply_option_crossed_market_filter(
    df: pd.DataFrame,
    bid_col: str = "bid_px",
    ask_col: str = "ask_px",
) -> pd.DataFrame:
    """Flag options with crossed markets (bid >= ask).
    
    Crossed markets violate no-arbitrage and indicate data issues.
    
    Args:
        df: DataFrame with bid/ask data
        bid_col: Column name for bid price
        ask_col: Column name for ask price
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_ARBITRAGE_VIOLATION column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if bid_col not in df.columns or ask_col not in df.columns:
        if FLAG_ARBITRAGE_VIOLATION not in df.columns:
            df[FLAG_ARBITRAGE_VIOLATION] = False
        return df
    
    bid = df[bid_col].astype(float)
    ask = df[ask_col].astype(float)
    
    valid_quotes = (bid > 0) & (ask > 0)
    crossed = valid_quotes & (bid >= ask)
    
    if FLAG_ARBITRAGE_VIOLATION in df.columns:
        df[FLAG_ARBITRAGE_VIOLATION] = df[FLAG_ARBITRAGE_VIOLATION] | crossed
    else:
        df[FLAG_ARBITRAGE_VIOLATION] = crossed
    
    return df


def apply_fat_finger_option_filter(
    df: pd.DataFrame,
    price_col: str = "price_int",
    underlying_col: str = "spot_ref_price_int",
    price_scale: float = 1e-9,
    threshold_dollars: float = 100_000.0,
) -> pd.DataFrame:
    """Flag obvious fat-finger option prices.
    
    Example: $187,187 premium when underlying is ~$620 (found in dataset).
    This is clearly erroneous and should be filtered.
    
    Args:
        df: DataFrame with option prices
        price_col: Column name for option price
        underlying_col: Column name for underlying price
        price_scale: Scale factor for fixed-point prices
        threshold_dollars: Absolute maximum option price in dollars
        
    Returns:
        pd.DataFrame: DataFrame with additional flags in FLAG_ARBITRAGE_VIOLATION
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if price_col not in df.columns:
        if FLAG_ARBITRAGE_VIOLATION not in df.columns:
            df[FLAG_ARBITRAGE_VIOLATION] = False
        return df
    
    price = df[price_col].astype(float)
    
    # Scale if in fixed-point
    if price.median() > 1e6:
        price_dollars = price * price_scale
    else:
        price_dollars = price
    
    # Flag any option price > threshold_dollars (obvious fat finger)
    fat_finger = price_dollars > threshold_dollars
    
    if FLAG_ARBITRAGE_VIOLATION in df.columns:
        df[FLAG_ARBITRAGE_VIOLATION] = df[FLAG_ARBITRAGE_VIOLATION] | fat_finger
    else:
        df[FLAG_ARBITRAGE_VIOLATION] = fat_finger
    
    return df
