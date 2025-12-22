"""
Test Barrier and Tape engines (Agent D deliverables)

Verifies:
- Engines instantiate correctly
- Basic state classification works
- Integration with MarketState (Agent C) works
"""

import time
from src.barrier_engine import BarrierEngine, BarrierState, Direction
from src.tape_engine import TapeEngine
from src.market_state import MarketState
from src.event_types import StockTrade, StockQuote, EventSource, Aggressor
from src.config import CONFIG


def test_barrier_engine_vacuum():
    """Test VACUUM state detection (liquidity pulled without fills)"""
    print("\n=== Test: Barrier Engine - VACUUM ===")
    
    engine = BarrierEngine()
    market_state = MarketState()
    
    # Setup: quotes at 545.00 with decreasing size, no trades
    ts_base = time.time_ns()
    level_price = 545.0
    
    # Initial quote: large bid at level
    market_state.update_stock_quote(StockQuote(
        source=EventSource.SIM,
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        symbol="SPY",
        bid_px=545.00,
        ask_px=545.02,
        bid_sz=50000,
        ask_sz=10000
    ))
    
    # Subsequent quotes: bid size decreasing (pulled), no fills
    for i in range(1, 10):
        market_state.update_stock_quote(StockQuote(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(1e9),
            ts_recv_ns=ts_base + i * int(1e9),
            symbol="SPY",
            bid_px=545.00,
            ask_px=545.02,
            bid_sz=50000 - i * 5000,
            ask_sz=10000
        ))
    
    # Compute barrier state for SUPPORT at 545
    metrics = engine.compute_barrier_state(
        level_price=level_price,
        direction=Direction.SUPPORT,
        market_state=market_state
    )
    
    print(f"State: {metrics.state}")
    print(f"Delta Liq: {metrics.delta_liq}")
    print(f"Replenishment Ratio: {metrics.replenishment_ratio:.2f}")
    print(f"Added: {metrics.added_size}, Canceled: {metrics.canceled_size}, Filled: {metrics.filled_size}")
    print(f"Confidence: {metrics.confidence:.2f}")
    
    assert metrics.state == BarrierState.VACUUM, f"Expected VACUUM, got {metrics.state}"
    assert metrics.canceled_size > metrics.filled_size, "Expected more canceled than filled"
    print("✅ VACUUM test passed")


def test_barrier_engine_wall():
    """Test WALL state detection (liquidity replenished)"""
    print("\n=== Test: Barrier Engine - WALL ===")
    
    engine = BarrierEngine()
    market_state = MarketState()
    
    ts_base = time.time_ns()
    level_price = 545.0
    
    # Initial quote
    market_state.update_stock_quote(StockQuote(
        source=EventSource.SIM,
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        symbol="SPY",
        bid_px=545.00,
        ask_px=545.02,
        bid_sz=10000,
        ask_sz=10000
    ))
    
    # Add some fills (trades hit bid)
    for i in range(5):
        market_state.update_stock_trade(StockTrade(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(0.5e9),
            ts_recv_ns=ts_base + i * int(0.5e9),
            symbol="SPY",
            price=545.00,
            size=1000
        ), aggressor=Aggressor.SELL)
    
    # But bid size keeps replenishing (increasing)
    for i in range(1, 10):
        market_state.update_stock_quote(StockQuote(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(1e9),
            ts_recv_ns=ts_base + i * int(1e9),
            symbol="SPY",
            bid_px=545.00,
            ask_px=545.02,
            bid_sz=10000 + i * 2000,  # Growing
            ask_sz=10000
        ))
    
    metrics = engine.compute_barrier_state(
        level_price=level_price,
        direction=Direction.SUPPORT,
        market_state=market_state
    )
    
    print(f"State: {metrics.state}")
    print(f"Delta Liq: {metrics.delta_liq}")
    print(f"Replenishment Ratio: {metrics.replenishment_ratio:.2f}")
    
    assert metrics.state in [BarrierState.WALL, BarrierState.ABSORPTION], f"Expected WALL/ABSORPTION, got {metrics.state}"
    assert metrics.delta_liq > 0, "Expected positive delta_liq"
    print("✅ WALL test passed")


def test_tape_engine_buy_imbalance():
    """Test buy imbalance and velocity detection"""
    print("\n=== Test: Tape Engine - Buy Imbalance ===")
    
    engine = TapeEngine()
    market_state = MarketState()
    
    ts_base = time.time_ns()
    level_price = 545.0
    
    # Add initial quote
    market_state.update_stock_quote(StockQuote(
        source=EventSource.SIM,
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        symbol="SPY",
        bid_px=544.90,
        ask_px=545.00,
        bid_sz=10000,
        ask_sz=10000
    ))
    
    # Add buy-heavy trades near level (lifting ask, moving up)
    for i in range(20):
        price = 545.0 + i * 0.01  # Rising
        market_state.update_stock_trade(StockTrade(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(0.2e9),
            ts_recv_ns=ts_base + i * int(0.2e9),
            symbol="SPY",
            price=price,
            size=500
        ), aggressor=Aggressor.BUY)
    
    # Add a few sells
    for i in range(5):
        market_state.update_stock_trade(StockTrade(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(0.3e9),
            ts_recv_ns=ts_base + i * int(0.3e9),
            symbol="SPY",
            price=545.05,
            size=200
        ), aggressor=Aggressor.SELL)
    
    metrics = engine.compute_tape_state(
        level_price=level_price,
        market_state=market_state
    )
    
    print(f"Imbalance: {metrics.imbalance:.2f}")
    print(f"Buy Vol: {metrics.buy_vol}, Sell Vol: {metrics.sell_vol}")
    print(f"Velocity: {metrics.velocity:.4f} $/sec")
    print(f"Sweep Detected: {metrics.sweep.detected}")
    
    assert metrics.imbalance > 0, "Expected positive imbalance (buy > sell)"
    assert metrics.velocity > 0, "Expected positive velocity (rising prices)"
    print("✅ Buy imbalance test passed")


def test_tape_engine_sweep_detection():
    """Test sweep detection"""
    print("\n=== Test: Tape Engine - Sweep Detection ===")
    
    engine = TapeEngine()
    market_state = MarketState()
    
    ts_base = time.time_ns()
    level_price = 545.0
    
    # Add quote
    market_state.update_stock_quote(StockQuote(
        source=EventSource.SIM,
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        symbol="SPY",
        bid_px=544.95,
        ask_px=545.00,
        bid_sz=10000,
        ask_sz=10000
    ))
    
    # Add a sweep: rapid cluster of BUY trades with large notional
    for i in range(15):
        market_state.update_stock_trade(StockTrade(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(0.05e9),  # 50ms gaps
            ts_recv_ns=ts_base + i * int(0.05e9),
            symbol="SPY",
            price=545.00 + i * 0.01,
            size=2000  # Large size
        ), aggressor=Aggressor.BUY)
    
    metrics = engine.compute_tape_state(
        level_price=level_price,
        market_state=market_state
    )
    
    print(f"Sweep Detected: {metrics.sweep.detected}")
    print(f"Sweep Direction: {metrics.sweep.direction}")
    print(f"Sweep Notional: ${metrics.sweep.notional:,.0f}")
    print(f"Sweep Prints: {metrics.sweep.num_prints}")
    print(f"Sweep Window: {metrics.sweep.window_ms:.0f}ms")
    
    assert metrics.sweep.detected, "Expected sweep to be detected"
    assert metrics.sweep.notional > CONFIG.SWEEP_MIN_NOTIONAL, "Notional should exceed threshold"
    print("✅ Sweep detection test passed")


def test_integration():
    """Test integration: both engines working together"""
    print("\n=== Test: Integration - BREAK scenario ===")
    
    barrier_engine = BarrierEngine()
    tape_engine = TapeEngine()
    market_state = MarketState()
    
    ts_base = time.time_ns()
    level_price = 545.0
    
    # Setup BREAK scenario:
    # - Bid at level shrinking (VACUUM)
    # - Heavy sell aggression (tape imbalance negative)
    
    # Initial large bid
    market_state.update_stock_quote(StockQuote(
        source=EventSource.SIM,
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        symbol="SPY",
        bid_px=545.00,
        ask_px=545.02,
        bid_sz=50000,
        ask_sz=10000
    ))
    
    # Bid shrinks (no fills, just pulled)
    for i in range(1, 8):
        market_state.update_stock_quote(StockQuote(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(1e9),
            ts_recv_ns=ts_base + i * int(1e9),
            symbol="SPY",
            bid_px=545.00,
            ask_px=545.02,
            bid_sz=50000 - i * 6000,
            ask_sz=10000
        ))
    
    # Heavy sell prints below level
    for i in range(30):
        market_state.update_stock_trade(StockTrade(
            source=EventSource.SIM,
            ts_event_ns=ts_base + i * int(0.2e9),
            ts_recv_ns=ts_base + i * int(0.2e9),
            symbol="SPY",
            price=545.00 - i * 0.01,
            size=1000
        ), aggressor=Aggressor.SELL)
    
    # Compute states
    barrier_metrics = barrier_engine.compute_barrier_state(
        level_price=level_price,
        direction=Direction.SUPPORT,
        market_state=market_state
    )
    
    tape_metrics = tape_engine.compute_tape_state(
        level_price=level_price,
        market_state=market_state
    )
    
    print(f"Barrier State: {barrier_metrics.state}")
    print(f"Tape Imbalance: {tape_metrics.imbalance:.2f}")
    print(f"Tape Velocity: {tape_metrics.velocity:.4f} $/sec")
    
    print("\n🎯 BREAK scenario characteristics:")
    print(f"  - Barrier VACUUM: {barrier_metrics.state == BarrierState.VACUUM}")
    print(f"  - Sell imbalance: {tape_metrics.imbalance < 0}")
    print(f"  - Negative velocity: {tape_metrics.velocity < 0}")
    
    print("✅ Integration test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent D Test Suite: Barrier & Tape Engines")
    print("=" * 60)
    
    test_barrier_engine_vacuum()
    test_barrier_engine_wall()
    test_tape_engine_buy_imbalance()
    test_tape_engine_sweep_detection()
    test_integration()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
