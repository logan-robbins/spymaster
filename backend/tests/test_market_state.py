#!/usr/bin/env python3
"""
Test script for MarketState functionality.

Verifies:
- Option flow aggregation and queries
- ES trade and MBP-10 updates are tested in test_e2e_replay.py
"""

import time
from src.common.event_types import OptionTrade, EventSource, Aggressor
from src.core.market_state import MarketState


def test_option_flow_aggregation():
    """Test option flow updates."""
    print("\n=== Test: Option Flow Aggregation ===")

    state = MarketState()
    ts_base = int(time.time() * 1e9)

    # Add option trades
    opt_trade = OptionTrade(
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        source=EventSource.DIRECT_FEED,
        underlying="ES",
        option_symbol="O:ES251216C06850000",
        exp_date="2025-12-16",
        strike=6850.0,
        right="C",
        price=2.50,
        size=10,
        aggressor=Aggressor.BUY  # customer buys
    )

    # Update with greeks
    state.update_option_trade(opt_trade, delta=0.50, gamma=0.05)

    # Verify aggregate
    key = (6850.0, "C", "2025-12-16")
    assert key in state.option_flows, f"Expected option flow for {key}"

    agg = state.option_flows[key]
    print(f"✅ Option flow created for strike={agg.strike} {agg.right}")
    print(f"   Volume: {agg.cumulative_volume}")
    print(f"   Premium: ${agg.cumulative_premium:,.0f}")
    print(f"   Net Delta Flow: {agg.net_delta_flow:,.0f}")
    print(f"   Net Dealer Gamma: {agg.net_gamma_flow:,.0f}")

    assert agg.cumulative_volume == 10
    from src.common.config import CONFIG
    assert agg.cumulative_premium == 2.50 * 10 * CONFIG.OPTION_CONTRACT_MULTIPLIER
    # Customer buys gamma -> dealer sells gamma (negative)
    assert agg.net_gamma_flow == -10 * 0.05 * CONFIG.OPTION_CONTRACT_MULTIPLIER


def test_option_flows_near_level():
    """Test querying option flows near a level."""
    print("\n=== Test: Option Flows Near Level ===")

    state = MarketState()
    ts_base = int(time.time() * 1e9)

    # Add option trades at different strikes
    strikes = [6840.0, 6850.0, 6860.0, 6880.0]
    for strike in strikes:
        opt_trade = OptionTrade(
            ts_event_ns=ts_base,
            ts_recv_ns=ts_base,
            source=EventSource.DIRECT_FEED,
            underlying="ES",
            option_symbol=f"O:ES251216C0{int(strike*100):05d}000",
            exp_date="2025-12-16",
            strike=strike,
            right="C",
            price=2.00,
            size=5,
            aggressor=Aggressor.BUY
        )
        state.update_option_trade(opt_trade, delta=0.50, gamma=0.05)

    # Query near 6850.0 within ±20.0
    flows = state.get_option_flows_near_level(
        level_price=6850.0,
        strike_range=20.0
    )
    print(f"✅ Option flows near 6850.0 (±20.0): {len(flows)} (expected 3)")
    print(f"   Strikes: {[f.strike for f in flows]}")
    assert len(flows) == 3, f"Expected 3 flows, got {len(flows)}"
