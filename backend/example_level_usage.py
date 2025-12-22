"""
Example usage of Level Universe and Room to Run modules.

This demonstrates how Agent G (Score Engine) and other downstream
consumers should use the level generation and runway computation logic.
"""

from src.core.market_state import MarketState, OptionFlowAggregate
from src.core.level_universe import LevelUniverse, LevelKind
from src.core.room_to_run import RoomToRun, Direction, get_break_direction, get_reject_direction
from src.common.event_types import StockTrade, StockQuote, EventSource


def main():
    print("=" * 80)
    print("Level Universe & Room to Run Example")
    print("=" * 80)
    
    # ========== Setup Market State ==========
    market_state = MarketState()
    
    # Add SPY trade to establish spot price
    trade = StockTrade(
        ts_event_ns=1000000000000,
        ts_recv_ns=1000000000000,
        source=EventSource.SIM,
        symbol="SPY",
        price=545.42,
        size=100
    )
    market_state.update_stock_trade(trade)
    
    # Add SPY quote
    quote = StockQuote(
        ts_event_ns=1000000000000,
        ts_recv_ns=1000000000000,
        source=EventSource.SIM,
        symbol="SPY",
        bid_px=545.41,
        ask_px=545.43,
        bid_sz=1000,
        ask_sz=800
    )
    market_state.update_stock_quote(quote)
    
    # Add some option flows to create interesting levels
    market_state.option_flows[(545.0, 'C', '2025-12-20')] = OptionFlowAggregate(
        strike=545.0,
        right='C',
        exp_date='2025-12-20',
        cumulative_volume=1000,
        net_gamma_flow=-8000.0
    )
    market_state.option_flows[(548.0, 'C', '2025-12-20')] = OptionFlowAggregate(
        strike=548.0,
        right='C',
        exp_date='2025-12-20',
        cumulative_volume=5000,
        net_gamma_flow=-25000.0  # Strong call wall
    )
    market_state.option_flows[(542.0, 'P', '2025-12-20')] = OptionFlowAggregate(
        strike=542.0,
        right='P',
        exp_date='2025-12-20',
        cumulative_volume=3000,
        net_gamma_flow=-15000.0  # Strong put wall
    )
    
    print(f"\nSPY Spot: ${market_state.get_spot():.2f}")
    print(f"SPY Bid/Ask: ${quote.bid_px:.2f} / ${quote.ask_px:.2f}")
    print(f"SPY VWAP: ${market_state.get_vwap():.2f}")
    
    # ========== Generate Levels ==========
    universe = LevelUniverse(user_hotzones=[550.0])
    levels = universe.get_levels(market_state)
    
    print(f"\n{'='*80}")
    print(f"Generated {len(levels)} Levels:")
    print(f"{'='*80}")
    
    # Group by kind
    by_kind = {}
    for level in levels:
        kind = level.kind.value
        if kind not in by_kind:
            by_kind[kind] = []
        by_kind[kind].append(level)
    
    for kind in sorted(by_kind.keys()):
        print(f"\n{kind}:")
        for level in sorted(by_kind[kind], key=lambda x: x.price):
            print(f"  {level.id:20s} @ ${level.price:7.2f}")
            if level.metadata:
                print(f"    Metadata: {level.metadata}")
    
    # ========== Compute Runway for Key Levels ==========
    print(f"\n{'='*80}")
    print("Runway Analysis:")
    print(f"{'='*80}")
    
    rtr = RoomToRun()
    spot = market_state.get_spot()
    
    # Find interesting levels to analyze
    strike_545 = next((l for l in levels if l.id == "STRIKE_545"), None)
    call_wall = next((l for l in levels if l.kind == LevelKind.CALL_WALL), None)
    put_wall = next((l for l in levels if l.kind == LevelKind.PUT_WALL), None)
    
    for level in [strike_545, call_wall, put_wall]:
        if level is None:
            continue
        
        print(f"\n--- Level: {level.id} @ ${level.price:.2f} ---")
        
        # Determine if it's support or resistance
        if spot > level.price:
            context = "SUPPORT (spot > level)"
            break_dir = Direction.DOWN
            reject_dir = Direction.UP
        else:
            context = "RESISTANCE (spot < level)"
            break_dir = Direction.UP
            reject_dir = Direction.DOWN
        
        print(f"Context: {context}")
        
        # Compute runway in break direction
        break_runway = rtr.compute_runway(level, break_dir, levels, spot)
        print(f"\nBreak Direction ({break_dir.value}):")
        if break_runway.next_obstacle:
            print(f"  Next Obstacle: {break_runway.next_obstacle.id} @ ${break_runway.next_obstacle.price:.2f}")
            print(f"  Distance: ${break_runway.distance:.2f}")
            print(f"  Quality: {break_runway.quality.value}")
            if break_runway.intermediate_levels:
                print(f"  Intermediate Levels: {len(break_runway.intermediate_levels)}")
                for inter in break_runway.intermediate_levels:
                    print(f"    - {inter.id} @ ${inter.price:.2f}")
        else:
            print(f"  No obstacles (infinite runway)")
        
        # Compute runway in reject direction
        reject_runway = rtr.compute_runway(level, reject_dir, levels, spot)
        print(f"\nReject Direction ({reject_dir.value}):")
        if reject_runway.next_obstacle:
            print(f"  Next Obstacle: {reject_runway.next_obstacle.id} @ ${reject_runway.next_obstacle.price:.2f}")
            print(f"  Distance: ${reject_runway.distance:.2f}")
            print(f"  Quality: {reject_runway.quality.value}")
        else:
            print(f"  No obstacles (infinite runway)")
    
    print(f"\n{'='*80}")
    print("Example Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
