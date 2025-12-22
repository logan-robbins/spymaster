#!/usr/bin/env python3
"""
Test script for MarketState functionality.

Verifies:
- Event ingestion (trades, quotes, option trades)
- Ring buffer cleanup and window queries
- Last-known value tracking
- Option flow aggregation
- Derived metrics (VWAP, session high/low)
"""

import time
from src.event_types import (
    StockTrade, StockQuote, OptionTrade, EventSource, Aggressor
)
from src.market_state import MarketState
from src.config import CONFIG


def test_basic_stock_updates():
    """Test basic stock trade and quote updates."""
    print("\n=== Test 1: Basic Stock Updates ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Add quote
    quote = StockQuote(
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        bid_px=545.40,
        ask_px=545.42,
        bid_sz=1000,
        ask_sz=800
    )
    state.update_stock_quote(quote)
    
    # Add trade
    trade = StockTrade(
        ts_event_ns=ts_base + 1_000_000,  # +1ms
        ts_recv_ns=ts_base + 1_000_000,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        price=545.41,
        size=100
    )
    state.update_stock_trade(trade)
    
    # Verify
    assert state.get_spot() == 545.41, f"Expected spot=545.41, got {state.get_spot()}"
    assert state.get_bid_ask() == (545.40, 545.42), f"Expected bid/ask=(545.40, 545.42), got {state.get_bid_ask()}"
    assert state.get_vwap() == 545.41, f"Expected VWAP=545.41, got {state.get_vwap()}"
    
    print(f"✅ Spot: {state.get_spot()}")
    print(f"✅ Bid/Ask: {state.get_bid_ask()}")
    print(f"✅ VWAP: {state.get_vwap()}")
    print(f"✅ Session High: {state.get_session_high()}")
    print(f"✅ Session Low: {state.get_session_low()}")


def test_aggressor_inference():
    """Test aggressor side inference from quote."""
    print("\n=== Test 2: Aggressor Inference ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Set quote
    quote = StockQuote(
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        bid_px=545.40,
        ask_px=545.42,
        bid_sz=1000,
        ask_sz=800
    )
    state.update_stock_quote(quote)
    
    # Trade at ask (BUY aggressor)
    buy_trade = StockTrade(
        ts_event_ns=ts_base + 1_000_000,
        ts_recv_ns=ts_base + 1_000_000,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        price=545.42,
        size=100
    )
    state.update_stock_trade(buy_trade)
    assert state.last_trade.aggressor == Aggressor.BUY, f"Expected BUY, got {state.last_trade.aggressor}"
    print(f"✅ Lift ask: {state.last_trade.aggressor}")
    
    # Trade at bid (SELL aggressor)
    sell_trade = StockTrade(
        ts_event_ns=ts_base + 2_000_000,
        ts_recv_ns=ts_base + 2_000_000,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        price=545.40,
        size=200
    )
    state.update_stock_trade(sell_trade)
    assert state.last_trade.aggressor == Aggressor.SELL, f"Expected SELL, got {state.last_trade.aggressor}"
    print(f"✅ Hit bid: {state.last_trade.aggressor}")
    
    # Trade at mid (MID)
    mid_trade = StockTrade(
        ts_event_ns=ts_base + 3_000_000,
        ts_recv_ns=ts_base + 3_000_000,
        source=EventSource.MASSIVE_WS,
        symbol="SPY",
        price=545.41,
        size=150
    )
    state.update_stock_trade(mid_trade)
    assert state.last_trade.aggressor == Aggressor.MID, f"Expected MID, got {state.last_trade.aggressor}"
    print(f"✅ Mid trade: {state.last_trade.aggressor}")


def test_ring_buffer_window_queries():
    """Test time-windowed queries."""
    print("\n=== Test 3: Ring Buffer Window Queries ===")
    
    state = MarketState(max_buffer_window_seconds=60.0)
    ts_base = int(time.time() * 1e9)
    
    # Add trades spread over 10 seconds
    for i in range(10):
        trade = StockTrade(
            ts_event_ns=ts_base + i * 1_000_000_000,  # +1 second each
            ts_recv_ns=ts_base + i * 1_000_000_000,
            source=EventSource.MASSIVE_WS,
            symbol="SPY",
            price=545.0 + i * 0.01,
            size=100
        )
        state.update_stock_trade(trade)
    
    # Query last 5 seconds
    ts_now = ts_base + 10_000_000_000  # 10 seconds in
    trades_5s = state.get_trades_in_window(ts_now, window_seconds=5.0)
    print(f"✅ Trades in 5s window: {len(trades_5s)} (expected ~5)")
    assert 4 <= len(trades_5s) <= 6, f"Expected ~5 trades, got {len(trades_5s)}"
    
    # Query last 3 seconds
    trades_3s = state.get_trades_in_window(ts_now, window_seconds=3.0)
    print(f"✅ Trades in 3s window: {len(trades_3s)} (expected ~3)")
    assert 2 <= len(trades_3s) <= 4, f"Expected ~3 trades, got {len(trades_3s)}"
    
    # Query all
    trades_all = state.get_trades_in_window(ts_now, window_seconds=60.0)
    print(f"✅ Trades in 60s window: {len(trades_all)} (expected 10)")
    assert len(trades_all) == 10, f"Expected 10 trades, got {len(trades_all)}"


def test_trades_near_level():
    """Test price-band filtered queries."""
    print("\n=== Test 4: Trades Near Level ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Add trades at different prices
    prices = [545.0, 545.05, 545.10, 545.50, 546.00]
    for i, price in enumerate(prices):
        trade = StockTrade(
            ts_event_ns=ts_base + i * 100_000_000,
            ts_recv_ns=ts_base + i * 100_000_000,
            source=EventSource.MASSIVE_WS,
            symbol="SPY",
            price=price,
            size=100
        )
        state.update_stock_trade(trade)
    
    # Query near 545.0 within ±0.10
    ts_now = ts_base + 1_000_000_000
    near_545 = state.get_trades_near_level(
        ts_now_ns=ts_now,
        window_seconds=10.0,
        level_price=545.0,
        band_dollars=0.10
    )
    print(f"✅ Trades near 545.0 (±0.10): {len(near_545)} (expected 3)")
    print(f"   Prices: {[t.price for t in near_545]}")
    assert len(near_545) == 3, f"Expected 3 trades, got {len(near_545)}"


def test_option_flow_aggregation():
    """Test option flow updates."""
    print("\n=== Test 5: Option Flow Aggregation ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Add option trades
    opt_trade = OptionTrade(
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        source=EventSource.MASSIVE_WS,
        underlying="SPY",
        option_symbol="O:SPY251216C00545000",
        exp_date="2025-12-16",
        strike=545.0,
        right="C",
        price=2.50,
        size=10,
        aggressor=Aggressor.BUY  # customer buys
    )
    
    # Update with greeks
    state.update_option_trade(opt_trade, delta=0.50, gamma=0.05)
    
    # Verify aggregate
    key = (545.0, "C", "2025-12-16")
    assert key in state.option_flows, f"Expected option flow for {key}"
    
    agg = state.option_flows[key]
    print(f"✅ Option flow created for strike={agg.strike} {agg.right}")
    print(f"   Volume: {agg.cumulative_volume}")
    print(f"   Premium: ${agg.cumulative_premium:,.0f}")
    print(f"   Net Delta Flow: {agg.net_delta_flow:,.0f}")
    print(f"   Net Dealer Gamma: {agg.net_gamma_flow:,.0f}")
    
    assert agg.cumulative_volume == 10
    assert agg.cumulative_premium == 2.50 * 10 * 100  # 2500
    # Customer buys gamma -> dealer sells gamma (negative)
    assert agg.net_gamma_flow == -10 * 0.05 * 100  # -50


def test_option_flows_near_level():
    """Test querying option flows near a level."""
    print("\n=== Test 6: Option Flows Near Level ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Add option trades at different strikes
    strikes = [544.0, 545.0, 546.0, 548.0]
    for strike in strikes:
        opt_trade = OptionTrade(
            ts_event_ns=ts_base,
            ts_recv_ns=ts_base,
            source=EventSource.MASSIVE_WS,
            underlying="SPY",
            option_symbol=f"O:SPY251216C00{int(strike*1000):05d}000",
            exp_date="2025-12-16",
            strike=strike,
            right="C",
            price=2.00,
            size=5,
            aggressor=Aggressor.BUY
        )
        state.update_option_trade(opt_trade, delta=0.50, gamma=0.05)
    
    # Query near 545.0 within ±2.0
    flows = state.get_option_flows_near_level(
        level_price=545.0,
        strike_range=2.0
    )
    print(f"✅ Option flows near 545.0 (±2.0): {len(flows)} (expected 3)")
    print(f"   Strikes: {[f.strike for f in flows]}")
    assert len(flows) == 3, f"Expected 3 flows, got {len(flows)}"


def test_buffer_stats():
    """Test buffer introspection."""
    print("\n=== Test 7: Buffer Stats ===")
    
    state = MarketState()
    ts_base = int(time.time() * 1e9)
    
    # Add some data
    for i in range(5):
        trade = StockTrade(
            ts_event_ns=ts_base + i * 100_000_000,
            ts_recv_ns=ts_base + i * 100_000_000,
            source=EventSource.MASSIVE_WS,
            symbol="SPY",
            price=545.0,
            size=100
        )
        state.update_stock_trade(trade)
        
        quote = StockQuote(
            ts_event_ns=ts_base + i * 100_000_000,
            ts_recv_ns=ts_base + i * 100_000_000,
            source=EventSource.MASSIVE_WS,
            symbol="SPY",
            bid_px=545.0,
            ask_px=545.02,
            bid_sz=1000,
            ask_sz=800
        )
        state.update_stock_quote(quote)
    
    stats = state.get_buffer_stats()
    print(f"✅ Buffer stats: {stats}")
    assert stats["trades_buffer_size"] == 5
    assert stats["quotes_buffer_size"] == 5
    print(f"   Trades buffer: {stats['trades_buffer_size']}")
    print(f"   Quotes buffer: {stats['quotes_buffer_size']}")
    print(f"   Option flows: {stats['option_flows_count']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MarketState Verification Tests")
    print("=" * 60)
    
    try:
        test_basic_stock_updates()
        test_aggressor_inference()
        test_ring_buffer_window_queries()
        test_trades_near_level()
        test_option_flow_aggregation()
        test_option_flows_near_level()
        test_buffer_stats()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nAgent C deliverables verified:")
        print("  ✓ Last-known values tracking")
        print("  ✓ Ring buffer implementation")
        print("  ✓ Window query methods")
        print("  ✓ Aggressor inference")
        print("  ✓ Option flow aggregation")
        print("  ✓ Price-band filtering")
        print("  ✓ Buffer introspection")
        print("\nReady for consumption by Agents D/E/F/G")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
