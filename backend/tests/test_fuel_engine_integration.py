"""
Integration example for Fuel Engine

Demonstrates how to use FuelEngine with MarketState to compute
dealer gamma exposure and identify gamma walls.

This shows Agent G (score engine) how to integrate fuel metrics.
"""

import time
from src.core.fuel_engine import FuelEngine, FuelEffect
from src.core.market_state import MarketState, OptionFlowAggregate
from src.common.event_types import StockTrade, OptionTrade, EventSource, Aggressor


def example_fuel_computation():
    """
    Example: Compute fuel state for a level with option flow data.
    
    Scenario:
    - SPY trading at 680
    - Testing level 680 (ATM strike)
    - Customers have been buying 680 calls (dealers short gamma)
    - Expect AMPLIFY effect (dealers will chase moves)
    """
    print("\n" + "="*60)
    print("FUEL ENGINE INTEGRATION EXAMPLE")
    print("="*60)
    
    # Initialize
    market_state = MarketState()
    fuel_engine = FuelEngine()
    
    # 1. Update stock state (set spot price)
    ts_base = time.time_ns()
    trade = StockTrade(
        ts_event_ns=ts_base,
        ts_recv_ns=ts_base,
        source=EventSource.SIM,
        symbol='SPY',
        price=680.0,
        size=100
    )
    market_state.update_stock_trade(trade, Aggressor.BUY)
    
    print(f"\n📊 SPY Spot: ${market_state.get_spot()}")
    
    # 2. Simulate option trades (customers buying calls at multiple strikes)
    print("\n📈 Simulating option flow...")
    
    option_trades = [
        # 679 strike: moderate buying
        OptionTrade(
            ts_event_ns=ts_base + int(1e9),
            ts_recv_ns=ts_base + int(1e9),
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol='O:SPY251220C00679000',
            exp_date='2025-12-20',
            strike=679.0,
            right='C',
            price=1.50,
            size=100,
            aggressor=Aggressor.BUY  # Customer buys → dealer sells gamma
        ),
        # 680 strike: heavy buying (ATM)
        OptionTrade(
            ts_event_ns=ts_base + int(2e9),
            ts_recv_ns=ts_base + int(2e9),
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol='O:SPY251220C00680000',
            exp_date='2025-12-20',
            strike=680.0,
            right='C',
            price=1.75,
            size=300,
            aggressor=Aggressor.BUY  # Customer buys → dealer sells gamma
        ),
        # 681 strike: light buying
        OptionTrade(
            ts_event_ns=ts_base + int(3e9),
            ts_recv_ns=ts_base + int(3e9),
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol='O:SPY251220C00681000',
            exp_date='2025-12-20',
            strike=681.0,
            right='C',
            price=1.25,
            size=50,
            aggressor=Aggressor.BUY  # Customer buys → dealer sells gamma
        ),
        # 682 strike: customers selling (dealers buying = long gamma = wall)
        OptionTrade(
            ts_event_ns=ts_base + int(4e9),
            ts_recv_ns=ts_base + int(4e9),
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol='O:SPY251220C00682000',
            exp_date='2025-12-20',
            strike=682.0,
            right='C',
            price=1.00,
            size=500,
            aggressor=Aggressor.SELL  # Customer sells → dealer buys gamma → WALL
        ),
        # 678 put: customers buying (dealers selling gamma)
        OptionTrade(
            ts_event_ns=ts_base + int(5e9),
            ts_recv_ns=ts_base + int(5e9),
            source=EventSource.SIM,
            underlying='SPY',
            option_symbol='O:SPY251220P00678000',
            exp_date='2025-12-20',
            strike=678.0,
            right='P',
            price=1.20,
            size=200,
            aggressor=Aggressor.BUY  # Customer buys → dealer sells gamma
        ),
    ]
    
    # Process option trades through market state
    for opt_trade in option_trades:
        # Simulate Greeks (in reality these come from greek_enricher)
        # ATM delta ~0.5, gamma ~0.10 for 0DTE
        # OTM delta/gamma decay
        moneyness = abs(opt_trade.strike - 680.0)
        if moneyness < 1:
            delta = 0.50 if opt_trade.right == 'C' else -0.50
            gamma = 0.10
        elif moneyness < 2:
            delta = 0.35 if opt_trade.right == 'C' else -0.35
            gamma = 0.08
        else:
            delta = 0.20 if opt_trade.right == 'C' else -0.20
            gamma = 0.05
        
        market_state.update_option_trade(opt_trade, delta=delta, gamma=gamma)
        
        # Print flow summary
        sign = "BUY" if opt_trade.aggressor == Aggressor.BUY else "SELL"
        print(f"  {opt_trade.strike} {opt_trade.right}: {sign} {opt_trade.size} @ ${opt_trade.price:.2f}")
    
    # 3. Compute fuel state at level 680
    print(f"\n🔥 Computing fuel state at level 680...")
    metrics = fuel_engine.compute_fuel_state(
        level_price=680.0,
        market_state=market_state,
        exp_date_filter='2025-12-20'
    )
    
    # 4. Display results
    print("\n" + "-"*60)
    print("FUEL METRICS")
    print("-"*60)
    print(f"Effect:              {metrics.effect.value}")
    print(f"Net Dealer Gamma:    {metrics.net_dealer_gamma:,.0f}")
    print(f"Confidence:          {metrics.confidence:.2f}")
    
    if metrics.call_wall:
        print(f"\n📍 Call Wall:")
        print(f"  Strike:            ${metrics.call_wall.strike:.2f}")
        print(f"  Net Gamma:         {metrics.call_wall.net_gamma:,.0f}")
        print(f"  Strength:          {metrics.call_wall.strength:.2f}x")
    else:
        print("\n📍 Call Wall:        None")
    
    if metrics.put_wall:
        print(f"\n📍 Put Wall:")
        print(f"  Strike:            ${metrics.put_wall.strike:.2f}")
        print(f"  Net Gamma:         {metrics.put_wall.net_gamma:,.0f}")
        print(f"  Strength:          {metrics.put_wall.strength:.2f}x")
    else:
        print("\n📍 Put Wall:        None")
    
    if metrics.hvl:
        print(f"\n⚡ HVL (Gamma Flip): ${metrics.hvl:.2f}")
    else:
        print("\n⚡ HVL (Gamma Flip): Not detected")
    
    print("\n📊 Gamma by Strike:")
    for strike in sorted(metrics.gamma_by_strike.keys()):
        gamma = metrics.gamma_by_strike[strike]
        print(f"  ${strike:.2f}: {gamma:>12,.0f}")
    
    # 5. Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if metrics.effect == FuelEffect.AMPLIFY:
        print("⚠️  AMPLIFY: Dealers are short gamma near this level.")
        print("   They will need to chase moves (buy as price rises, sell as it falls).")
        print("   This creates positive feedback → trend acceleration.")
    elif metrics.effect == FuelEffect.DAMPEN:
        print("🛡️  DAMPEN: Dealers are long gamma near this level.")
        print("   They will fade moves (sell as price rises, buy as it falls).")
        print("   This creates negative feedback → mean reversion.")
    else:
        print("⚖️  NEUTRAL: Minimal gamma exposure near this level.")
    
    if metrics.call_wall:
        print(f"\n🚧 Call wall at ${metrics.call_wall.strike:.2f} acts as resistance.")
    if metrics.put_wall:
        print(f"🛡️  Put wall at ${metrics.put_wall.strike:.2f} acts as support.")
    
    print("\n" + "="*60)
    
    return metrics


def example_scoring_integration():
    """
    Example: How Agent G (score engine) would use fuel metrics.
    
    This shows the expected interface for score composition.
    """
    print("\n" + "="*60)
    print("SCORE ENGINE INTEGRATION (Agent G)")
    print("="*60)
    
    # Get fuel metrics (from above example)
    market_state = MarketState()
    fuel_engine = FuelEngine()
    
    # Minimal setup
    trade = StockTrade(
        ts_event_ns=time.time_ns(),
        ts_recv_ns=time.time_ns(),
        source=EventSource.SIM,
        symbol='SPY',
        price=680.0,
        size=100
    )
    market_state.update_stock_trade(trade, Aggressor.BUY)
    
    # Simulate option flow that creates AMPLIFY effect
    market_state.option_flows[(680.0, 'C', '2025-12-20')] = OptionFlowAggregate(
        strike=680.0,
        right='C',
        exp_date='2025-12-20',
        net_gamma_flow=-50000.0,  # Dealers short gamma
        cumulative_volume=500
    )
    
    metrics = fuel_engine.compute_fuel_state(680.0, market_state, '2025-12-20')
    
    # Score engine mapping (per PLAN.md §5.4.1)
    print("\n📊 Hedge Score Computation:")
    print(f"   Effect: {metrics.effect.value}")
    
    if metrics.effect == FuelEffect.AMPLIFY:
        hedge_score = 100  # Maximum amplification
        print(f"   → S_H = 100 (AMPLIFY in break direction)")
    elif metrics.effect == FuelEffect.DAMPEN:
        hedge_score = 0    # Maximum dampening
        print(f"   → S_H = 0 (DAMPEN)")
    else:
        hedge_score = 50   # Neutral
        print(f"   → S_H = 50 (NEUTRAL)")
    
    print(f"\n   Confidence weight: {metrics.confidence:.2f}")
    print(f"   Net dealer gamma: {metrics.net_dealer_gamma:,.0f}")
    
    # Composite score example (per PLAN.md §5.4.2)
    # S = w_L * S_L + w_H * S_H + w_T * S_T
    # Default weights: w_L=0.45, w_H=0.35, w_T=0.20
    
    print("\n📈 Composite Break Score:")
    print(f"   S = w_L * S_L + w_H * S_H + w_T * S_T")
    print(f"   S = 0.45 * S_L + 0.35 * {hedge_score} + 0.20 * S_T")
    print(f"   (Hedge contributes: 0.35 * {hedge_score} = {0.35 * hedge_score:.1f} points)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run examples
    metrics = example_fuel_computation()
    print("\n")
    example_scoring_integration()
    
    print("\n✅ Fuel Engine integration complete!")
    print("   Agent G can now consume FuelMetrics for score composition.")

