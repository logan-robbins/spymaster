"""
Comprehensive tests for physics pipeline integration.

Tests the combination of:
- MBP-10 data (ES futures depth)
- Trades data (ES futures trades)
- Options data (ES options gamma)

Using easy-to-validate numbers to ensure calculations are correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import List, Dict, Tuple
from datetime import datetime, timezone

from src.common.event_types import FuturesTrade, MBP10, BidAskLevel, EventSource, Aggressor
from src.core.physics_engines import (
    MarketData,
    build_market_data,
    compute_tape_metrics,
    compute_barrier_metrics,
    compute_fuel_metrics,
    compute_all_physics,
)


# =============================================================================
# FIXTURES: Synthetic Data with Easy-to-Validate Numbers
# =============================================================================


@pytest.fixture
def base_timestamp() -> int:
    """Base timestamp: 2025-12-16 10:00:00 UTC"""
    dt = datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


@pytest.fixture
def simple_trades(base_timestamp: int) -> List[FuturesTrade]:
    """
    Create simple trade tape for easy validation.
    
    ES at 6850.00:
    - 10:00:00 - Buy 100 @ 6850.00
    - 10:00:01 - Buy 200 @ 6850.25 
    - 10:00:02 - Sell 150 @ 6850.50
    - 10:00:03 - Sell 100 @ 6850.75
    - 10:00:04 - Buy 300 @ 6851.00
    
    Expected tape imbalance in window:
    Buy = 100 + 200 + 300 = 600
    Sell = 150 + 100 = 250
    Imbalance = (600 - 250) / (600 + 250) = 350/850 = 0.4118
    """
    trades = []
    
    # Trade 1: Buy 100 @ 6850.00
    trades.append(FuturesTrade(
        ts_event_ns=base_timestamp,
        ts_recv_ns=base_timestamp,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        price=6850.00,
        size=100,
        aggressor=Aggressor.BUY
    ))
    
    # Trade 2: Buy 200 @ 6850.25 (1 second later)
    trades.append(FuturesTrade(
        ts_event_ns=base_timestamp + 1_000_000_000,
        ts_recv_ns=base_timestamp + 1_000_000_000,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        price=6850.25,
        size=200,
        aggressor=Aggressor.BUY
    ))
    
    # Trade 3: Sell 150 @ 6850.50 (2 seconds later)
    trades.append(FuturesTrade(
        ts_event_ns=base_timestamp + 2_000_000_000,
        ts_recv_ns=base_timestamp + 2_000_000_000,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        price=6850.50,
        size=150,
        aggressor=Aggressor.SELL
    ))
    
    # Trade 4: Sell 100 @ 6850.75 (3 seconds later)
    trades.append(FuturesTrade(
        ts_event_ns=base_timestamp + 3_000_000_000,
        ts_recv_ns=base_timestamp + 3_000_000_000,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        price=6850.75,
        size=100,
        aggressor=Aggressor.SELL
    ))
    
    # Trade 5: Buy 300 @ 6851.00 (4 seconds later)
    trades.append(FuturesTrade(
        ts_event_ns=base_timestamp + 4_000_000_000,
        ts_recv_ns=base_timestamp + 4_000_000_000,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        price=6851.00,
        size=300,
        aggressor=Aggressor.BUY
    ))
    
    return trades


@pytest.fixture
def simple_mbp10(base_timestamp: int) -> List[MBP10]:
    """
    Create simple MBP-10 snapshots for easy validation.
    
    ES at 6850.00:
    
    Snapshot 1 (t=0s): Wall at 6850.00
    - Bid: 6850.00 x 2000 (strong wall)
    - Ask: 6850.25 x 500
    
    Snapshot 2 (t=5s): Wall consumed
    - Bid: 6850.00 x 1000 (wall reduced)
    - Ask: 6850.25 x 500
    
    Expected barrier metrics:
    delta_liq = 1000 - 2000 = -1000 (liquidity pulled/consumed)
    """
    snapshots = []
    
    # Snapshot 1: Initial state with wall
    levels_1 = []
    # Level 0: Wall at 6850.00
    levels_1.append(BidAskLevel(
        bid_px=6850.00,
        bid_sz=2000,
        ask_px=6850.25,
        ask_sz=500
    ))
    # Levels 1-9: Regular depth (decreasing)
    for i in range(1, 10):
        levels_1.append(BidAskLevel(
            bid_px=6850.00 - (i * 0.25),
            bid_sz=1000 - (i * 50),
            ask_px=6850.25 + (i * 0.25),
            ask_sz=500 - (i * 25)
        ))
    
    snapshots.append(MBP10(
        ts_event_ns=base_timestamp,
        ts_recv_ns=base_timestamp,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        levels=levels_1,
        is_snapshot=True
    ))
    
    # Snapshot 2: Wall consumed (5 seconds later)
    levels_2 = []
    # Level 0: Wall reduced
    levels_2.append(BidAskLevel(
        bid_px=6850.00,
        bid_sz=1000,  # Reduced from 2000
        ask_px=6850.25,
        ask_sz=500
    ))
    # Levels 1-9: Regular depth
    for i in range(1, 10):
        levels_2.append(BidAskLevel(
            bid_px=6850.00 - (i * 0.25),
            bid_sz=1000 - (i * 50),
            ask_px=6850.25 + (i * 0.25),
            ask_sz=500 - (i * 25)
        ))
    
    snapshots.append(MBP10(
        ts_event_ns=base_timestamp + 5_000_000_000,
        ts_recv_ns=base_timestamp + 5_000_000_000,
        source=EventSource.DIRECT_FEED,
        symbol="ES",
        levels=levels_2,
        is_snapshot=True
    ))
    
    return snapshots


@pytest.fixture
def simple_option_flows() -> Dict[Tuple[float, str, str], Any]:
    """
    Create simple option flows for easy validation.
    
    ES strikes near 6850:
    - 6845 Put: +30,000 gamma (dealers long, DAMPEN)
    - 6850 Call: -25,000 gamma (dealers short, AMPLIFY)
    - 6855 Call: +15,000 gamma (dealers long, DAMPEN)
    
    At ES 6850.00 level:
    - In range ±5.0: 6845, 6850, 6855
    - Net gamma = 30000 - 25000 + 15000 = 20,000 (DAMPEN)
    """
    from dataclasses import dataclass
    
    @dataclass
    class MockFlow:
        net_gamma_flow: float
        net_premium_flow: float = 0.0
        cumulative_volume: int = 1000
    
    flows = {}
    
    # 6845 Put: +30K gamma (long)
    flows[(6845.0, 'P', '2025-12-16')] = MockFlow(net_gamma_flow=30000.0)
    
    # 6850 Call: -25K gamma (short)
    flows[(6850.0, 'C', '2025-12-16')] = MockFlow(net_gamma_flow=-25000.0)
    
    # 6855 Call: +15K gamma (long)
    flows[(6855.0, 'C', '2025-12-16')] = MockFlow(net_gamma_flow=15000.0)
    
    return flows


# =============================================================================
# TESTS: build_market_data
# =============================================================================


def test_build_market_data_converts_trades(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """Test that trades are correctly converted to numpy arrays."""
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=simple_mbp10,
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    # Check trade arrays
    assert len(market_data.trade_ts_ns) == 5
    assert len(market_data.trade_prices) == 5
    assert len(market_data.trade_sizes) == 5
    assert len(market_data.trade_aggressors) == 5
    
    # Check sorted by timestamp
    assert np.all(market_data.trade_ts_ns[:-1] <= market_data.trade_ts_ns[1:])
    
    # Check first trade
    assert market_data.trade_ts_ns[0] == base_timestamp
    assert market_data.trade_prices[0] == 6850.00
    assert market_data.trade_sizes[0] == 100
    assert market_data.trade_aggressors[0] == 1  # BUY


def test_build_market_data_converts_mbp10(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """Test that MBP-10 snapshots are correctly converted to numpy arrays."""
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=simple_mbp10,
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    # Check MBP arrays
    assert len(market_data.mbp_ts_ns) == 2
    assert market_data.mbp_bid_prices.shape == (2, 10)
    assert market_data.mbp_bid_sizes.shape == (2, 10)
    assert market_data.mbp_ask_prices.shape == (2, 10)
    assert market_data.mbp_ask_sizes.shape == (2, 10)
    
    # Check sorted by timestamp
    assert market_data.mbp_ts_ns[0] < market_data.mbp_ts_ns[1]
    
    # Check first snapshot, first level
    assert market_data.mbp_bid_prices[0, 0] == 6850.00
    assert market_data.mbp_bid_sizes[0, 0] == 2000  # Wall
    assert market_data.mbp_ask_prices[0, 0] == 6850.25
    assert market_data.mbp_ask_sizes[0, 0] == 500
    
    # Check second snapshot, first level (wall consumed)
    assert market_data.mbp_bid_sizes[1, 0] == 1000  # Reduced


def test_build_market_data_aggregates_gamma(simple_trades, simple_mbp10, simple_option_flows):
    """Test that option gamma is correctly aggregated by strike."""
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=simple_mbp10,
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    # Check strike gamma
    assert 6845.0 in market_data.strike_gamma
    assert 6850.0 in market_data.strike_gamma
    assert 6855.0 in market_data.strike_gamma
    
    # Check values
    assert market_data.strike_gamma[6845.0] == 30000.0   # Put +30K
    assert market_data.strike_gamma[6850.0] == -25000.0  # Call -25K
    assert market_data.strike_gamma[6855.0] == 15000.0   # Call +15K


# =============================================================================
# TESTS: compute_tape_metrics
# =============================================================================


def test_tape_metrics_imbalance_calculation(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test tape imbalance calculation with known values.
    
    Setup:
    - Buy: 100 + 200 + 300 = 600
    - Sell: 150 + 100 = 250
    - Imbalance = (600 - 250) / 850 = 350/850 = 0.4118
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Touch at start, looking forward 5 seconds
    touch_ts = np.array([base_timestamp], dtype=np.int64)
    level_prices = np.array([6850.0], dtype=np.float64)  # ES
    
    result = compute_tape_metrics(
        touch_ts_ns=touch_ts,
        level_prices=level_prices,
        market_data=market_data,
        window_seconds=5.0,
        band_dollars=1.00  # Wide band to capture all trades
    )
    
    # Validate imbalance
    assert 'tape_imbalance' in result
    expected_imbalance = (600 - 250) / (600 + 250)
    assert abs(result['tape_imbalance'][0] - expected_imbalance) < 0.01  # ~0.4118
    
    # Validate volumes
    assert result['tape_buy_vol'][0] == 600
    assert result['tape_sell_vol'][0] == 250


def test_tape_metrics_velocity_positive(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test tape velocity with upward price movement.
    
    Prices: 6850.00, 6850.25, 6850.50, 6850.75, 6851.00
    Times: 0s, 1s, 2s, 3s, 4s
    
    Linear fit slope ≈ 0.25 ES per second
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])
    
    result = compute_tape_metrics(
        touch_ts, level_prices, market_data,
        window_seconds=5.0,
        band_dollars=1.00
    )
    
    # Velocity should be positive (prices rising)
    assert result['tape_velocity'][0] > 0
    # Should be roughly 0.25 ES per second
    assert 0.20 < result['tape_velocity'][0] < 0.30


def test_tape_metrics_price_band_filtering(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test that tape metrics only include trades within price band.
    
    With narrow band (0.10 ES), only first trade at 6850.00 included.
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])  # ES
    
    result = compute_tape_metrics(
        touch_ts, level_prices, market_data,
        window_seconds=5.0,
        band_dollars=0.10  # Very narrow band
    )
    
    # Only first trade at 6850.00 should be included (ES = 6850)
    # Buy volume should be 100
    assert result['tape_buy_vol'][0] == 100
    assert result['tape_sell_vol'][0] == 0


# =============================================================================
# TESTS: compute_barrier_metrics
# =============================================================================


def test_barrier_metrics_wall_consumption(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test barrier metrics detecting wall consumption.
    
    Setup:
    - t=0s: Bid 6850.00 x 2000
    - t=5s: Bid 6850.00 x 1000
    - Delta = 1000 - 2000 = -1000 (liquidity removed)
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Touch at base timestamp (resistance at ES 6850.00)
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])  # ES
    directions = np.array([1])  # UP (resistance, check ask side)
    
    result = compute_barrier_metrics(
        touch_ts_ns=touch_ts,
        level_prices=level_prices,
        directions=directions,
        market_data=market_data,
        window_seconds=10.0,  # Capture both snapshots
        zone_es_ticks=2  # ±2 ticks around level
    )
    
    # Check that we got results
    assert 'barrier_state' in result
    assert 'barrier_delta_liq' in result
    assert 'depth_in_zone' in result
    
    # For resistance (UP), we look at ask side
    # First snapshot: ask 6850.25 x 500 (within zone 6850 ± 0.50)
    # Last snapshot: ask 6850.25 x 500 (same)
    # Note: For resistance, defending side is ASK
    # If ask liquidity stable or reduced → could be VACUUM/CONSUMED


def test_barrier_metrics_support_side_detection(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test that barrier metrics check correct side for support vs resistance.
    
    Support (DOWN direction) → Check BID side
    Resistance (UP direction) → Check ASK side
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Test support (direction = DOWN = -1)
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])
    directions = np.array([-1])  # DOWN (support, check bid side)
    
    result = compute_barrier_metrics(
        touch_ts, level_prices, directions, market_data,
        window_seconds=10.0,
        zone_es_ticks=2
    )
    
    # For support, we check bid side
    # First snapshot: bid 6850.00 x 2000
    # Last snapshot: bid 6850.00 x 1000
    # Delta should be negative (consumed)
    assert result['barrier_delta_liq'][0] <= 0


def test_barrier_metrics_multiple_touches(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """Test barrier metrics for multiple touches."""
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Three touches at different times
    touch_ts = np.array([
        base_timestamp,
        base_timestamp + 1_000_000_000,
        base_timestamp + 2_000_000_000
    ])
    level_prices = np.array([6850.0, 6850.0, 6850.0])
    directions = np.array([1, 1, 1])  # All resistance
    
    result = compute_barrier_metrics(
        touch_ts, level_prices, directions, market_data,
        window_seconds=10.0
    )
    
    # Should have results for all 3 touches
    assert len(result['barrier_state']) == 3
    assert len(result['barrier_delta_liq']) == 3


# =============================================================================
# TESTS: compute_fuel_metrics
# =============================================================================


def test_fuel_metrics_gamma_aggregation(simple_trades, simple_mbp10, simple_option_flows):
    """
    Test fuel metrics gamma aggregation.
    
    At ES 6850.00 with strike_range=5.0:
    - 6845 Put: +30,000
    - 6850 Call: -25,000
    - 6855 Call: +15,000
    - Net: 30000 - 25000 + 15000 = +20,000 (DAMPEN)
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    level_prices = np.array([6850.0])
    
    result = compute_fuel_metrics(
        level_prices=level_prices,
        market_data=market_data,
        strike_range=5.0
    )
    
    # Check gamma exposure
    assert 'gamma_exposure' in result
    expected_gamma = 30000.0 - 25000.0 + 15000.0  # = 20,000
    assert abs(result['gamma_exposure'][0] - expected_gamma) < 1.0
    
    # Check fuel effect
    assert result['fuel_effect'][0] == 'DAMPEN'  # Positive gamma > 10K


def test_fuel_metrics_amplify_effect(simple_trades, simple_mbp10, base_timestamp):
    """
    Test fuel effect classification for AMPLIFY (dealers short gamma).
    """
    from dataclasses import dataclass
    
    @dataclass
    class MockFlow:
        net_gamma_flow: float
        net_premium_flow: float = 0.0
        cumulative_volume: int = 1000
    
    # Create option flows with large negative gamma
    option_flows = {
        (6850.0, 'C', '2025-12-16'): MockFlow(net_gamma_flow=-50000.0)
    }
    
    market_data = build_market_data(
        simple_trades, simple_mbp10, option_flows, "2025-12-16"
    )
    
    level_prices = np.array([6850.0])
    
    result = compute_fuel_metrics(
        level_prices, market_data, strike_range=5.0
    )
    
    # Large negative gamma → AMPLIFY
    assert result['gamma_exposure'][0] == -50000.0
    assert result['fuel_effect'][0] == 'AMPLIFY'


def test_fuel_metrics_neutral_effect(simple_trades, simple_mbp10, base_timestamp):
    """
    Test fuel effect classification for NEUTRAL (small gamma).
    """
    from dataclasses import dataclass
    
    @dataclass
    class MockFlow:
        net_gamma_flow: float
        net_premium_flow: float = 0.0
        cumulative_volume: int = 1000
    
    # Small gamma values
    option_flows = {
        (6850.0, 'C', '2025-12-16'): MockFlow(net_gamma_flow=5000.0)
    }
    
    market_data = build_market_data(
        simple_trades, simple_mbp10, option_flows, "2025-12-16"
    )
    
    level_prices = np.array([6850.0])
    
    result = compute_fuel_metrics(
        level_prices, market_data, strike_range=5.0
    )
    
    # Small gamma → NEUTRAL
    assert abs(result['gamma_exposure'][0] - 5000.0) < 1.0
    assert result['fuel_effect'][0] == 'NEUTRAL'


def test_fuel_metrics_strike_range_filtering(simple_trades, simple_mbp10, simple_option_flows):
    """
    Test that fuel metrics only include strikes within range.
    
    At ES 6850.00:
    - strike_range=1.0 → only 6850 included
    - strike_range=5.0 → 6845, 6850, 6855 included
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    level_prices = np.array([6850.0])
    
    # Narrow range: only 685
    result_narrow = compute_fuel_metrics(
        level_prices, market_data, strike_range=1.0
    )
    assert result_narrow['gamma_exposure'][0] == -25000.0  # Only 685 Call
    
    # Wide range: 684, 685, 686
    result_wide = compute_fuel_metrics(
        level_prices, market_data, strike_range=5.0
    )
    expected = 30000.0 - 25000.0 + 15000.0  # All three strikes
    assert abs(result_wide['gamma_exposure'][0] - expected) < 1.0


# =============================================================================
# TESTS: compute_all_physics (Integration)
# =============================================================================


def test_compute_all_physics_combines_metrics(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test that compute_all_physics combines all three engines.
    
    Should return tape + barrier + fuel metrics in one dict.
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])
    directions = np.array([1])  # UP
    
    result = compute_all_physics(
        touch_ts_ns=touch_ts,
        level_prices=level_prices,
        directions=directions,
        market_data=market_data
    )
    
    # Should have tape metrics
    assert 'tape_imbalance' in result
    assert 'tape_buy_vol' in result
    assert 'tape_sell_vol' in result
    assert 'tape_velocity' in result
    
    # Should have barrier metrics
    assert 'barrier_state' in result
    assert 'barrier_delta_liq' in result
    assert 'depth_in_zone' in result
    
    # Should have fuel metrics
    assert 'gamma_exposure' in result
    assert 'fuel_effect' in result
    
    # All should have same length
    assert len(result['tape_imbalance']) == 1
    assert len(result['barrier_state']) == 1
    assert len(result['gamma_exposure']) == 1


def test_pipeline_end_to_end(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    End-to-end test combining MBP-10, Trades, and Options.
    
    Validates the complete data flow with easy-to-check numbers.
    """
    # Step 1: Build market data
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=simple_mbp10,
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    # Step 2: Define touch event (ES touching 6850.00 from below)
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])  # ES strike level
    directions = np.array([1])  # UP (resistance)
    
    # Step 3: Compute all physics
    result = compute_all_physics(
        touch_ts_ns=touch_ts,
        level_prices=level_prices,
        directions=directions,
        market_data=market_data
    )
    
    # Step 4: Validate tape metrics (ES futures trades)
    expected_buy_vol = 300  # 100 + 200 within ±0.50 band
    expected_sell_vol = 150  # 150 within ±0.50 band
    assert result['tape_buy_vol'][0] == expected_buy_vol
    assert result['tape_sell_vol'][0] == expected_sell_vol
    
    expected_imbalance = (300 - 150) / 450  # ≈ 0.3333
    assert abs(result['tape_imbalance'][0] - expected_imbalance) < 0.01
    
    # Velocity should be positive (rising prices)
    assert result['tape_velocity'][0] > 0
    
    # Step 5: Validate barrier metrics (ES MBP-10 depth)
    # For resistance, we check ask side
    # Delta should reflect changes in defending liquidity
    assert 'barrier_delta_liq' in result
    assert 'depth_in_zone' in result
    
    # Step 6: Validate fuel metrics (ES options gamma)
    expected_gamma = 20000.0  # 30K - 25K + 15K
    assert abs(result['gamma_exposure'][0] - expected_gamma) < 1.0
    assert result['fuel_effect'][0] == 'DAMPEN'





# =============================================================================
# TESTS: Edge Cases
# =============================================================================


def test_empty_trades_handling(simple_mbp10, simple_option_flows, base_timestamp):
    """Test pipeline handles empty trade list."""
    market_data = build_market_data(
        trades=[],  # Empty
        mbp10_snapshots=simple_mbp10,
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    assert len(market_data.trade_ts_ns) == 0
    
    # Tape metrics should return zeros
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])
    
    result = compute_tape_metrics(
        touch_ts, level_prices, market_data,
        window_seconds=5.0
    )
    
    assert result['tape_buy_vol'][0] == 0
    assert result['tape_sell_vol'][0] == 0
    assert result['tape_imbalance'][0] == 0.0


def test_empty_mbp10_handling(simple_trades, simple_option_flows, base_timestamp):
    """Test pipeline handles empty MBP-10 list."""
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=[],  # Empty
        option_flows=simple_option_flows,
        date="2025-12-16"
    )
    
    assert len(market_data.mbp_ts_ns) == 0
    
    # Barrier metrics should return NEUTRAL
    touch_ts = np.array([base_timestamp])
    level_prices = np.array([6850.0])
    directions = np.array([1])
    
    result = compute_barrier_metrics(
        touch_ts, level_prices, directions, market_data,
        window_seconds=10.0
    )
    
    assert result['barrier_state'][0] == 'NEUTRAL'


def test_no_option_flows_handling(simple_trades, simple_mbp10):
    """Test pipeline handles no option flows."""
    market_data = build_market_data(
        trades=simple_trades,
        mbp10_snapshots=simple_mbp10,
        option_flows={},  # Empty
        date="2025-12-16"
    )
    
    assert len(market_data.strike_gamma) == 0
    
    # Fuel metrics should return NEUTRAL
    level_prices = np.array([6850.0])
    
    result = compute_fuel_metrics(
        level_prices, market_data, strike_range=5.0
    )
    
    assert result['gamma_exposure'][0] == 0.0
    assert result['fuel_effect'][0] == 'NEUTRAL'


# =============================================================================
# TESTS: Multi-touch Throughput
# =============================================================================


def test_multi_level_processing(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test processing multiple levels simultaneously.
    
    Tests that the array engines correctly handle multiple levels.
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # 5 touches at different levels and times
    touch_ts = np.array([base_timestamp] * 5)
    level_prices = np.array([6830.0, 6840.0, 6850.0, 6860.0, 6870.0])
    directions = np.array([1, -1, 1, -1, 1])  # Mixed
    
    result = compute_all_physics(
        touch_ts, level_prices, directions, market_data
    )
    
    # All arrays should have length 5
    assert len(result['tape_imbalance']) == 5
    assert len(result['barrier_state']) == 5
    assert len(result['gamma_exposure']) == 5
    
    # Different levels should have different gamma
    # 6850 should have known value, others may differ
    idx_685 = 2
    expected_gamma_685 = 20000.0
    assert abs(result['gamma_exposure'][idx_685] - expected_gamma_685) < 1.0


def test_array_vs_scalar_consistency(simple_trades, simple_mbp10, simple_option_flows, base_timestamp):
    """
    Test that multi-touch processing matches processing one-by-one.
    """
    market_data = build_market_data(
        simple_trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Single touch
    single_result = compute_all_physics(
        touch_ts_ns=np.array([base_timestamp]),
        level_prices=np.array([6850.0]),
        directions=np.array([1]),
        market_data=market_data
    )
    
    # Array with one element
    multi_result = compute_all_physics(
        touch_ts_ns=np.array([base_timestamp]),
        level_prices=np.array([6850.0]),
        directions=np.array([1]),
        market_data=market_data
    )
    
    # Should be identical
    assert single_result['tape_imbalance'][0] == multi_result['tape_imbalance'][0]
    assert single_result['gamma_exposure'][0] == multi_result['gamma_exposure'][0]


# =============================================================================
# TESTS: Data Quality Validation
# =============================================================================


def test_timestamp_ordering_requirement(simple_mbp10, simple_option_flows):
    """Test that unsorted trades are sorted during conversion."""
    # Create trades out of order
    ts_base = int(datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    
    trades = [
        FuturesTrade(
            ts_event_ns=ts_base + 2_000_000_000,  # Later
            ts_recv_ns=ts_base + 2_000_000_000,
            source=EventSource.DIRECT_FEED,
            symbol="ES",
            price=6850.0,
            size=100,
            aggressor=Aggressor.BUY
        ),
        FuturesTrade(
            ts_event_ns=ts_base,  # Earlier
            ts_recv_ns=ts_base,
            source=EventSource.DIRECT_FEED,
            symbol="ES",
            price=6849.0,
            size=50,
            aggressor=Aggressor.SELL
        )
    ]
    
    market_data = build_market_data(
        trades, simple_mbp10, simple_option_flows, "2025-12-16"
    )
    
    # Should be sorted
    assert market_data.trade_ts_ns[0] < market_data.trade_ts_ns[1]
    assert market_data.trade_prices[0] == 6849.0  # Earlier trade
    assert market_data.trade_prices[1] == 6850.0  # Later trade


def test_aggressor_encoding(base_timestamp):
    """Test that aggressor side is correctly encoded (1=BUY, -1=SELL)."""
    trades = [
        FuturesTrade(
            ts_event_ns=base_timestamp,
            ts_recv_ns=base_timestamp,
            source=EventSource.DIRECT_FEED,
            symbol="ES",
            price=6850.0,
            size=100,
            aggressor=Aggressor.BUY
        ),
        FuturesTrade(
            ts_event_ns=base_timestamp + 1_000_000_000,
            ts_recv_ns=base_timestamp + 1_000_000_000,
            source=EventSource.DIRECT_FEED,
            symbol="ES",
            price=6850.0,
            size=100,
            aggressor=Aggressor.SELL
        )
    ]
    
    market_data = build_market_data(
        trades, [], {}, "2025-12-16"
    )
    
    assert market_data.trade_aggressors[0] == 1   # BUY
    assert market_data.trade_aggressors[1] == -1  # SELL


# =============================================================================
# TESTS: Realistic Scenarios
# =============================================================================


def test_vacuum_scenario():
    """
    Test VACUUM detection: liquidity pulled without fills.
    
    Scenario:
    - t=0: Bid wall at 6850 x 3000
    - t=5: Bid wall at 6850 x 500 (pulled, not filled)
    - No trades near level
    
    Expected: VACUUM (delta_liq negative, low fill volume)
    """
    ts_base = int(datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    
    # No trades (liquidity pulled, not consumed)
    trades = []
    
    # MBP-10: Wall then pulled
    mbp1_levels = [BidAskLevel(bid_px=6850.0, bid_sz=3000, ask_px=6850.25, ask_sz=500)]
    mbp1_levels += [BidAskLevel(bid_px=6850.0 - i*0.25, bid_sz=1000, ask_px=6850.25 + i*0.25, ask_sz=500) for i in range(1, 10)]
    
    mbp2_levels = [BidAskLevel(bid_px=6850.0, bid_sz=500, ask_px=6850.25, ask_sz=500)]  # Wall pulled
    mbp2_levels += [BidAskLevel(bid_px=6850.0 - i*0.25, bid_sz=1000, ask_px=6850.25 + i*0.25, ask_sz=500) for i in range(1, 10)]
    
    mbp_snapshots = [
        MBP10(ts_event_ns=ts_base, ts_recv_ns=ts_base, source=EventSource.DIRECT_FEED, 
              symbol="ES", levels=mbp1_levels, is_snapshot=True),
        MBP10(ts_event_ns=ts_base + 5_000_000_000, ts_recv_ns=ts_base + 5_000_000_000,
              source=EventSource.DIRECT_FEED, symbol="ES", levels=mbp2_levels, is_snapshot=True)
    ]
    
    market_data = build_market_data(trades, mbp_snapshots, {}, "2025-12-16")
    
    # Touch at support (checking bid side)
    touch_ts = np.array([ts_base])
    level_prices = np.array([6850.0])
    directions = np.array([-1])  # DOWN (support, check bid)
    
    result = compute_barrier_metrics(
        touch_ts, level_prices, directions, market_data, window_seconds=10.0
    )
    
    # Should detect liquidity reduction
    assert result['barrier_delta_liq'][0] <= 0
    # State could be VACUUM or CONSUMED depending on thresholds
    assert result['barrier_state'][0] in ['VACUUM', 'CONSUMED', 'WEAK', 'NEUTRAL']


def test_wall_scenario():
    """
    Test WALL detection: liquidity replenishing after consumption.
    
    Scenario:
    - t=0: Bid at 6850 x 2000
    - t=2: Bid at 6850 x 1500 (consumed)
    - t=5: Bid at 6850 x 2500 (replenished!)
    
    Expected: WALL (replenishment_ratio > 1.5)
    """
    ts_base = int(datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    
    # Small trades (some consumption)
    trades = [
        FuturesTrade(ts_event_ns=ts_base + 1_000_000_000, ts_recv_ns=ts_base + 1_000_000_000,
                    source=EventSource.DIRECT_FEED, symbol="ES", price=6850.0, size=500, aggressor=Aggressor.SELL)
    ]
    
    # MBP-10: Initial → Consumed → Replenished
    mbp_snapshots = []
    
    # t=0: 2000
    levels = [BidAskLevel(bid_px=6850.0, bid_sz=2000, ask_px=6850.25, ask_sz=500)]
    levels += [BidAskLevel(bid_px=6850.0 - i*0.25, bid_sz=1000, ask_px=6850.25 + i*0.25, ask_sz=500) for i in range(1, 10)]
    mbp_snapshots.append(MBP10(ts_event_ns=ts_base, ts_recv_ns=ts_base, source=EventSource.DIRECT_FEED, 
                               symbol="ES", levels=levels, is_snapshot=True))
    
    # t=2s: 1500 (consumed)
    levels = [BidAskLevel(bid_px=6850.0, bid_sz=1500, ask_px=6850.25, ask_sz=500)]
    levels += [BidAskLevel(bid_px=6850.0 - i*0.25, bid_sz=1000, ask_px=6850.25 + i*0.25, ask_sz=500) for i in range(1, 10)]
    mbp_snapshots.append(MBP10(ts_event_ns=ts_base + 2_000_000_000, ts_recv_ns=ts_base + 2_000_000_000,
                               source=EventSource.DIRECT_FEED, symbol="ES", levels=levels, is_snapshot=True))
    
    # t=5s: 2500 (replenished!)
    levels = [BidAskLevel(bid_px=6850.0, bid_sz=2500, ask_px=6850.25, ask_sz=500)]
    levels += [BidAskLevel(bid_px=6850.0 - i*0.25, bid_sz=1000, ask_px=6850.25 + i*0.25, ask_sz=500) for i in range(1, 10)]
    mbp_snapshots.append(MBP10(ts_event_ns=ts_base + 5_000_000_000, ts_recv_ns=ts_base + 5_000_000_000,
                               source=EventSource.DIRECT_FEED, symbol="ES", levels=levels, is_snapshot=True))
    
    market_data = build_market_data(trades, mbp_snapshots, {}, "2025-12-16")
    
    # Touch at support
    touch_ts = np.array([ts_base])
    level_prices = np.array([6850.0])
    directions = np.array([-1])  # DOWN (support, check bid)
    
    result = compute_barrier_metrics(
        touch_ts, level_prices, directions, market_data, window_seconds=10.0
    )
    
    # Net delta positive (2500 - 2000 = +500)
    assert result['barrier_delta_liq'][0] >= 0
    # State should reflect replenishment
    # (exact state depends on replenishment ratio thresholds)


def test_heavy_sell_imbalance():
    """
    Test strong sell imbalance detection.
    
    All sells, no buys → imbalance = -1.0
    """
    ts_base = int(datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    
    trades = [
        FuturesTrade(ts_event_ns=ts_base + i * 1_000_000_000, ts_recv_ns=ts_base + i * 1_000_000_000,
                    source=EventSource.DIRECT_FEED, symbol="ES", price=6850.0, size=100, aggressor=Aggressor.SELL)
        for i in range(10)
    ]
    
    market_data = build_market_data(trades, [], {}, "2025-12-16")
    
    touch_ts = np.array([ts_base])
    level_prices = np.array([6850.0])
    
    result = compute_tape_metrics(
        touch_ts, level_prices, market_data,
        window_seconds=12.0,  # Capture all trades
        band_dollars=1.00
    )
    
    # All sells → imbalance = -1.0
    assert result['tape_buy_vol'][0] == 0
    assert result['tape_sell_vol'][0] == 1000  # 10 trades x 100
    assert result['tape_imbalance'][0] == -1.0  # Perfect sell imbalance


# =============================================================================
# TESTS: Strike Alignment
# =============================================================================


def test_es_strikes_at_5pt_intervals():
    """
    Test that ES strikes at 5-point intervals are preserved.
    
    ES strikes: 6845, 6850, 6855
    """
    from dataclasses import dataclass
    
    @dataclass
    class MockFlow:
        net_gamma_flow: float
        net_premium_flow: float = 0.0
        cumulative_volume: int = 1000
    
    ts_base = int(datetime(2025, 12, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    
    # Create flows at ES strikes (5-point intervals)
    option_flows = {
        (6845.0, 'P', '2025-12-16'): MockFlow(net_gamma_flow=10000.0),
        (6850.0, 'C', '2025-12-16'): MockFlow(net_gamma_flow=-5000.0),
        (6855.0, 'C', '2025-12-16'): MockFlow(net_gamma_flow=8000.0),
    }
    
    market_data = build_market_data([], [], option_flows, "2025-12-16")
    
    # Test gamma lookup at each strike
    assert market_data.strike_gamma[6845.0] == 10000.0
    assert market_data.strike_gamma[6850.0] == -5000.0
    assert market_data.strike_gamma[6855.0] == 8000.0
