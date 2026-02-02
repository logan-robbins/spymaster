"""
Price deviation filters based on institutional standards.

Thresholds:
- NYSE Rule 128: 3/5/10% by price tier
- CME Circuit Breakers: 7% regular trading hours
- Academic: 3σ z-score from rolling median

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Flag column names
FLAG_PRICE_OUTLIER = "price_outlier_flag"
FLAG_CROSSED_MARKET = "crossed_market_flag"
FLAG_SPREAD_ANOMALY = "spread_anomaly_flag"

# NYSE Rule 128 thresholds (regular trading hours)
NYSE_THRESHOLDS = [
    (0, 25, 0.10),      # $0-$25: 10%
    (25, 50, 0.05),     # $25-$50: 5%
    (50, float('inf'), 0.03),  # >$50: 3%
]

# CME futures thresholds
CME_RTH_THRESHOLD = 0.07   # 7% during regular trading hours
CME_OVERNIGHT_THRESHOLD = 0.035  # 3.5% overnight

# Z-score threshold for outlier detection
ZSCORE_THRESHOLD = 3.0

# Spread anomaly multiplier (flag if spread > N * median spread)
SPREAD_ANOMALY_MULTIPLIER = 10.0


def get_nyse_threshold(price: float) -> float:
    """Get NYSE Rule 128 threshold for given price level.
    
    Args:
        price: Current price in dollars
        
    Returns:
        float: Threshold as decimal (e.g., 0.03 for 3%)
    """
    for low, high, threshold in NYSE_THRESHOLDS:
        if low <= price < high:
            return threshold
    return 0.03  # Default to strictest


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 10,
) -> pd.Series:
    """Compute rolling z-score for outlier detection.
    
    Uses Median Absolute Deviation (MAD) method which is robust to outliers.
    Z-score = (x - median) / (MAD * 1.4826)
    
    Args:
        series: Price or value series
        window: Rolling window size in periods
        min_periods: Minimum periods for valid calculation
        
    Returns:
        pd.Series: Z-scores (absolute value)
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    rolling_median = series.rolling(window=window, min_periods=min_periods, center=False).median()
    rolling_mad = series.rolling(window=window, min_periods=min_periods, center=False).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    # Scale factor for MAD to match standard deviation for normal distribution
    mad_scale = 1.4826
    zscore = np.abs(series - rolling_median) / (rolling_mad * mad_scale + 1e-10)
    return zscore.fillna(0.0)


def apply_price_deviation_filter(
    df: pd.DataFrame,
    price_col: str = "mid_price",
    reference_col: Optional[str] = None,
    product_type: str = "futures",
    price_scale: float = 1e-9,
) -> pd.DataFrame:
    """Apply price deviation filter and add flag column.
    
    Flags rows where price deviates too far from reference (rolling median or explicit).
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price to check
        reference_col: Column for reference price (optional, uses rolling median if None)
        product_type: 'futures' or 'equities' (determines threshold selection)
        price_scale: Scale factor to convert price_int to dollars (default 1e-9)
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_PRICE_OUTLIER column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if price_col not in df.columns:
        df[FLAG_PRICE_OUTLIER] = False
        return df
    
    price = df[price_col].astype(float)
    
    # Convert to dollars if needed
    if price.median() > 1e6:  # Likely in fixed-point format
        price_dollars = price * price_scale
    else:
        price_dollars = price
    
    # Compute z-score based outliers
    zscore = compute_rolling_zscore(price)
    zscore_outlier = zscore > ZSCORE_THRESHOLD
    
    # Compute deviation-based outliers
    if reference_col and reference_col in df.columns:
        ref_price = df[reference_col].astype(float)
        if ref_price.median() > 1e6:
            ref_price_dollars = ref_price * price_scale
        else:
            ref_price_dollars = ref_price
    else:
        ref_price_dollars = price_dollars.rolling(window=60, min_periods=1).median()
    
    deviation = np.abs(price_dollars - ref_price_dollars) / (ref_price_dollars + 1e-10)
    
    # Apply threshold based on product type and price level
    if product_type == "futures":
        threshold = CME_RTH_THRESHOLD
    else:
        # Use NYSE tiered thresholds for equities
        median_price = price_dollars.median()
        threshold = get_nyse_threshold(median_price)
    
    deviation_outlier = deviation > threshold
    
    # Flag if either condition triggered
    df[FLAG_PRICE_OUTLIER] = zscore_outlier | deviation_outlier
    
    return df


def apply_crossed_market_filter(
    df: pd.DataFrame,
    bid_col: str = "best_bid_price_int",
    ask_col: str = "best_ask_price_int",
) -> pd.DataFrame:
    """Flag rows where bid >= ask (crossed or locked market).
    
    A crossed market is a clear data quality issue and should be flagged.
    
    Args:
        df: DataFrame with bid/ask prices
        bid_col: Column name for bid price
        ask_col: Column name for ask price
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_CROSSED_MARKET column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if bid_col not in df.columns or ask_col not in df.columns:
        df[FLAG_CROSSED_MARKET] = False
        return df
    
    bid = df[bid_col].astype(float)
    ask = df[ask_col].astype(float)
    
    # Crossed: bid >= ask (and both are valid/non-zero)
    valid_quotes = (bid > 0) & (ask > 0)
    df[FLAG_CROSSED_MARKET] = valid_quotes & (bid >= ask)
    
    return df


def compute_spread_anomaly_flag(
    df: pd.DataFrame,
    bid_col: str = "best_bid_price_int",
    ask_col: str = "best_ask_price_int",
    multiplier: float = SPREAD_ANOMALY_MULTIPLIER,
) -> pd.DataFrame:
    """Flag rows with abnormally wide spreads.
    
    Wide spreads can indicate low liquidity or data issues.
    
    Args:
        df: DataFrame with bid/ask prices
        bid_col: Column name for bid price  
        ask_col: Column name for ask price
        multiplier: Flag if spread > multiplier * median_spread
        
    Returns:
        pd.DataFrame: DataFrame with FLAG_SPREAD_ANOMALY column added
        
    Last Grunted: 02/01/2026 10:00:00 AM PST
    """
    df = df.copy()
    
    if bid_col not in df.columns or ask_col not in df.columns:
        df[FLAG_SPREAD_ANOMALY] = False
        return df
    
    bid = df[bid_col].astype(float)
    ask = df[ask_col].astype(float)
    
    spread = ask - bid
    valid_spread = (spread > 0) & (bid > 0) & (ask > 0)
    
    median_spread = spread[valid_spread].median()
    if pd.isna(median_spread) or median_spread <= 0:
        df[FLAG_SPREAD_ANOMALY] = False
    else:
        df[FLAG_SPREAD_ANOMALY] = valid_spread & (spread > multiplier * median_spread)
    
    return df
