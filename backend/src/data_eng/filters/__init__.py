"""
Institutional-grade data filters for market microstructure data.

This module implements a 3-layer filtering strategy:
- Bronze layer: Hard rejects (impossible values)
- Silver layer: Soft flags (mark suspicious, preserve data)
- Gold layer: Strict filters (clean features for ML)

Thresholds based on:
- NYSE Rule 128: 3/5/10% deviation by price tier
- CME Circuit Breakers: 7/13/20% limits
- Academic consensus: 3σ z-score, MAD filtering
"""

from .price_filters import (
    FLAG_PRICE_OUTLIER,
    FLAG_CROSSED_MARKET,
    FLAG_SPREAD_ANOMALY,
    apply_price_deviation_filter,
    apply_crossed_market_filter,
    compute_rolling_zscore,
)
from .size_filters import (
    FLAG_SIZE_OUTLIER,
    apply_size_outlier_filter,
    compute_size_percentiles,
)
from .options_filters import (
    FLAG_IV_ANOMALY,
    FLAG_ARBITRAGE_VIOLATION,
    apply_option_price_bounds,
    apply_iv_filter,
    apply_fat_finger_option_filter,
)
from .bronze_hard_rejects import (
    apply_bronze_hard_rejects,
    REJECT_REASON_ZERO_PRICE,
    REJECT_REASON_ZERO_SIZE,
    REJECT_REASON_CROSSED_BOOK,
)
from .gold_strict_filters import (
    apply_gold_strict_filters,
    filter_flagged_rows,
)

__all__ = [
    # Price filters
    "FLAG_PRICE_OUTLIER",
    "FLAG_CROSSED_MARKET", 
    "FLAG_SPREAD_ANOMALY",
    "apply_price_deviation_filter",
    "apply_crossed_market_filter",
    "compute_rolling_zscore",
    # Size filters
    "FLAG_SIZE_OUTLIER",
    "apply_size_outlier_filter",
    "compute_size_percentiles",
    # Options filters
    "FLAG_IV_ANOMALY",
    "FLAG_ARBITRAGE_VIOLATION",
    "apply_option_price_bounds",
    "apply_iv_filter",
    "apply_fat_finger_option_filter",
    # Bronze hard rejects
    "apply_bronze_hard_rejects",
    "REJECT_REASON_ZERO_PRICE",
    "REJECT_REASON_ZERO_SIZE",
    "REJECT_REASON_CROSSED_BOOK",
    # Gold strict filters
    "apply_gold_strict_filters",
    "filter_flagged_rows",
]
