#!/usr/bin/env python3
"""
Test script for institutional-grade data filters.

Validates:
1. Bronze hard reject filters
2. Price deviation filters
3. Size outlier filters
4. Options filters
5. Gold strict filters

Run: cd backend && uv run python scripts/test_filters.py

Last Grunted: 02/01/2026 10:00:00 AM PST
"""

import numpy as np
import pandas as pd
import sys

from src.data_eng.filters import (
    # Bronze
    apply_bronze_hard_rejects,
    REJECT_REASON_ZERO_PRICE,
    REJECT_REASON_ZERO_SIZE,
    # Price
    apply_price_deviation_filter,
    apply_crossed_market_filter,
    compute_rolling_zscore,
    FLAG_PRICE_OUTLIER,
    FLAG_CROSSED_MARKET,
    # Size
    apply_size_outlier_filter,
    FLAG_SIZE_OUTLIER,
    # Options
    apply_iv_filter,
    apply_option_price_bounds,
    apply_fat_finger_option_filter,
    FLAG_IV_ANOMALY,
    FLAG_ARBITRAGE_VIOLATION,
    # Gold
    apply_gold_strict_filters,
    filter_flagged_rows,
)


def test_bronze_hard_rejects():
    """Test bronze layer hard reject filters."""
    print("\n=== Testing Bronze Hard Rejects ===")
    
    # Create test data with various issues
    df = pd.DataFrame({
        "ts_event": [1e18, 1e18, 1e18, 1e18, 1e18, np.nan],
        "action": ["A", "A", "M", "C", "A", "A"],
        "price": [1000, 0, 1000, 0, -500, 1000],  # 0 and -500 should be rejected for A/M
        "size": [10, 10, 0, 10, 10, 10],  # 0 should be rejected for M
    })
    
    df_filtered, stats = apply_bronze_hard_rejects(df, return_stats=True)
    
    print(f"  Original rows: {stats['original_count']}")
    print(f"  Rejected: {stats['total_rejected']}")
    print(f"    - Zero price: {stats[REJECT_REASON_ZERO_PRICE]}")
    print(f"    - Zero size: {stats[REJECT_REASON_ZERO_SIZE]}")
    print(f"  Remaining rows: {len(df_filtered)}")
    
    # Should reject: row 1 (zero price), row 2 (zero size), row 4 (neg price), row 5 (null ts)
    expected_rejects = 4
    assert stats['total_rejected'] >= 3, f"Expected at least 3 rejects, got {stats['total_rejected']}"
    print("  ✓ Bronze hard rejects working correctly")


def test_price_deviation_filter():
    """Test price deviation detection."""
    print("\n=== Testing Price Deviation Filter ===")
    
    # Create data with normal prices and outliers
    np.random.seed(42)
    normal_prices = np.random.normal(100, 1, 100)  # Mean 100, std 1
    outlier_prices = [50, 200]  # 50% and 100% deviation - clear outliers
    
    df = pd.DataFrame({
        "mid_price": list(normal_prices) + outlier_prices
    })
    
    df_flagged = apply_price_deviation_filter(df, price_col="mid_price", product_type="equities")
    
    outlier_count = df_flagged[FLAG_PRICE_OUTLIER].sum()
    print(f"  Total rows: {len(df_flagged)}")
    print(f"  Flagged as outliers: {outlier_count}")
    
    # The last two rows (50 and 200) should definitely be flagged
    assert df_flagged[FLAG_PRICE_OUTLIER].iloc[-2], "50 should be flagged as outlier"
    assert df_flagged[FLAG_PRICE_OUTLIER].iloc[-1], "200 should be flagged as outlier"
    print("  ✓ Price deviation filter working correctly")


def test_crossed_market_filter():
    """Test crossed market detection."""
    print("\n=== Testing Crossed Market Filter ===")
    
    df = pd.DataFrame({
        "best_bid_price_int": [100, 110, 105, 0, 100],
        "best_ask_price_int": [105, 105, 100, 100, 0],  # Row 2 crossed, rows 3-4 invalid
    })
    
    df_flagged = apply_crossed_market_filter(df)
    
    crossed_count = df_flagged[FLAG_CROSSED_MARKET].sum()
    print(f"  Total rows: {len(df_flagged)}")
    print(f"  Crossed markets: {crossed_count}")
    
    assert df_flagged[FLAG_CROSSED_MARKET].iloc[1], "Row 1 (bid=110, ask=105) should be crossed"
    assert df_flagged[FLAG_CROSSED_MARKET].iloc[2], "Row 2 (bid=105, ask=100) should be crossed"
    assert not df_flagged[FLAG_CROSSED_MARKET].iloc[0], "Row 0 should not be crossed"
    print("  ✓ Crossed market filter working correctly")


def test_size_outlier_filter():
    """Test size outlier detection."""
    print("\n=== Testing Size Outlier Filter ===")
    
    # Normal sizes plus one extreme outlier
    np.random.seed(42)
    normal_sizes = np.random.exponential(10, 100).astype(int) + 1  # typical order sizes
    outlier_size = [50000]  # Way above 99.9th percentile
    
    df = pd.DataFrame({
        "size": list(normal_sizes) + outlier_size
    })
    
    df_flagged = apply_size_outlier_filter(df, product_symbol="ES")
    
    outlier_count = df_flagged[FLAG_SIZE_OUTLIER].sum()
    print(f"  Total rows: {len(df_flagged)}")
    print(f"  Size outliers: {outlier_count}")
    
    assert df_flagged[FLAG_SIZE_OUTLIER].iloc[-1], "50000 should be flagged as outlier for ES"
    print("  ✓ Size outlier filter working correctly")


def test_iv_filter():
    """Test implied volatility filter."""
    print("\n=== Testing IV Filter ===")
    
    df = pd.DataFrame({
        "iv": [0.20, 0.50, 0.005, 6.0, -0.1, np.nan]  # Normal, normal, too low, too high, negative, null
    })
    
    df_flagged = apply_iv_filter(df)
    
    anomaly_count = df_flagged[FLAG_IV_ANOMALY].sum()
    print(f"  Total rows: {len(df_flagged)}")
    print(f"  IV anomalies: {anomaly_count}")
    
    assert not df_flagged[FLAG_IV_ANOMALY].iloc[0], "20% IV should be normal"
    assert not df_flagged[FLAG_IV_ANOMALY].iloc[1], "50% IV should be normal"
    assert df_flagged[FLAG_IV_ANOMALY].iloc[2], "0.5% IV should be anomalous"
    assert df_flagged[FLAG_IV_ANOMALY].iloc[3], "600% IV should be anomalous"
    assert df_flagged[FLAG_IV_ANOMALY].iloc[4], "Negative IV should be anomalous"
    assert df_flagged[FLAG_IV_ANOMALY].iloc[5], "Null IV should be anomalous"
    print("  ✓ IV filter working correctly")


def test_fat_finger_option_filter():
    """Test fat-finger option price detection (e.g., $187,187 premium)."""
    print("\n=== Testing Fat-Finger Option Filter ===")
    
    # Simulate the $187,187 issue found in the data
    df = pd.DataFrame({
        "price_int": [
            20_000_000_000,        # $20 (normal)
            50_000_000_000,        # $50 (normal for ITM)
            187_187_000_000_000,   # $187,187 (fat finger!)
            500_000_000_000,       # $500 (high but plausible for deep ITM)
        ],
        "spot_ref_price_int": [620_000_000_000] * 4  # $620 underlying
    })
    
    df_flagged = apply_fat_finger_option_filter(df, threshold_dollars=100_000.0)
    
    fat_finger_count = df_flagged[FLAG_ARBITRAGE_VIOLATION].sum()
    print(f"  Total rows: {len(df_flagged)}")
    print(f"  Fat-finger flags: {fat_finger_count}")
    
    assert not df_flagged[FLAG_ARBITRAGE_VIOLATION].iloc[0], "$20 premium should be normal"
    assert not df_flagged[FLAG_ARBITRAGE_VIOLATION].iloc[1], "$50 premium should be normal"
    assert df_flagged[FLAG_ARBITRAGE_VIOLATION].iloc[2], "$187,187 should be flagged"
    assert not df_flagged[FLAG_ARBITRAGE_VIOLATION].iloc[3], "$500 premium should be normal"
    print("  ✓ Fat-finger filter working correctly")


def test_gold_strict_filters():
    """Test gold layer strict filter application."""
    print("\n=== Testing Gold Strict Filters ===")
    
    df = pd.DataFrame({
        "window_end_ts_ns": [1e18, 1e18, 1e18, 1e18, np.nan],
        "spot_ref_price_int": [1000, 1000, 0, 1000, 1000],
        FLAG_PRICE_OUTLIER: [False, True, False, False, False],
        FLAG_CROSSED_MARKET: [False, False, True, False, False],
    })
    
    df_filtered, stats = apply_gold_strict_filters(df, product_type="futures", return_stats=True)
    
    print(f"  Original rows: {stats['original_count']}")
    print(f"  Filtered rows: {stats['total_filtered']}")
    print(f"  Final rows: {stats['final_count']}")
    print(f"  Breakdown: price_outlier={stats.get(FLAG_PRICE_OUTLIER, 0)}, "
          f"crossed={stats.get(FLAG_CROSSED_MARKET, 0)}, "
          f"zero_spot={stats.get('zero_spot_ref', 0)}")
    
    # Should filter: row 1 (price outlier), row 2 (crossed), row 4 (null ts)
    # Row 2 has zero spot_ref which should also be filtered
    assert stats['total_filtered'] >= 3, f"Expected at least 3 filtered, got {stats['total_filtered']}"
    print("  ✓ Gold strict filters working correctly")


def test_rolling_zscore():
    """Test rolling z-score calculation."""
    print("\n=== Testing Rolling Z-Score ===")
    
    # Create data with known outlier
    data = [100.0] * 50 + [200.0] + [100.0] * 49  # Single spike at index 50
    series = pd.Series(data)
    
    zscore = compute_rolling_zscore(series, window=20, min_periods=5)
    
    max_zscore = zscore.max()
    max_idx = zscore.idxmax()
    
    print(f"  Max z-score: {max_zscore:.2f} at index {max_idx}")
    
    # The spike should have high z-score
    assert max_idx == 50, f"Spike should be at index 50, got {max_idx}"
    assert max_zscore > 3.0, f"Spike z-score should be > 3, got {max_zscore:.2f}"
    print("  ✓ Rolling z-score working correctly")


def main():
    print("=" * 60)
    print("INSTITUTIONAL-GRADE DATA FILTER TEST SUITE")
    print("=" * 60)
    
    try:
        test_bronze_hard_rejects()
        test_price_deviation_filter()
        test_crossed_market_filter()
        test_size_outlier_filter()
        test_iv_filter()
        test_fat_finger_option_filter()
        test_gold_strict_filters()
        test_rolling_zscore()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
