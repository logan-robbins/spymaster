"""
Run Pipeline - Agent D Integration

Complete research pipeline that integrates:
- Agent A: Physics Engine (order book microstructure)
- Agent B: Context Engine (level identification + timing)
- Agent C: Labeler (outcome classification) + Experiment Runner (analysis)

Pipeline Flow:
1. Generate synthetic OHLCV data (1 day)
2. Scan for level touches (whole numbers, SMA-200, PM high/low)
3. Calculate physics metrics at each touch
4. Create LevelSignalV1 objects
5. Label outcomes with forward price data
6. Run statistical analysis and print reports

Usage:
    cd backend/
    uv run python -m src.pipeline.run_pipeline
"""

import sys
import uuid
from typing import List, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Import Agent A, B, C components
from src.features.physics_engine import PhysicsEngine
from src.features.context_engine import ContextEngine
from src.research.labeler import get_outcome
from src.research.experiment_runner import ExperimentRunner

# Import schemas
from src.common.schemas.levels_signals import LevelSignalV1, LevelKind, Direction, OutcomeLabel
from src.common.event_types import MBP10, FuturesTrade


def generate_synthetic_price_path(
    num_minutes: int = 480,
    base_price: float = 400.0,
    volatility: float = 0.5,
    trend: float = 0.02
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data with realistic price action.
    
    Creates a full trading day (pre-market through regular session) with:
    - Random walk with drift
    - Intraday volatility patterns (higher at open/close)
    - Realistic OHLC relationships
    
    Args:
        num_minutes: Number of 1-minute bars to generate (default 480 = 8 hours)
        base_price: Starting price (default 400.0 for SPY)
        volatility: Price volatility in dollars (default 0.5)
        trend: Drift per minute in dollars (default 0.02 = slight uptrend)
    
    Returns:
        DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    # Start at 04:00 ET (pre-market start)
    # Note: Using a fixed date for reproducibility
    start_time = pd.Timestamp('2025-12-22 04:00:00', tz='America/New_York')
    
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(num_minutes)]
    
    # Generate close prices with random walk + drift
    np.random.seed(42)  # Reproducible results
    returns = np.random.normal(trend, volatility, num_minutes)
    close_prices = [base_price]
    
    for ret in returns[1:]:
        new_price = close_prices[-1] + ret
        close_prices.append(new_price)
    
    # Generate OHLC from close prices with realistic spreads
    data = []
    for i, ts in enumerate(timestamps):
        close = close_prices[i]
        
        # High/Low spread increases with volatility
        spread = np.random.uniform(0.05, 0.30)  # Realistic intrabar range
        high = close + np.random.uniform(0, spread)
        low = close - np.random.uniform(0, spread)
        
        # Open is close of previous bar (with small gap)
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1] + np.random.uniform(-0.05, 0.05)
        
        # Volume increases during open and close
        hour = ts.hour
        if 9 <= hour <= 10 or 15 <= hour <= 16:
            volume = np.random.randint(100000, 500000)  # High volume
        else:
            volume = np.random.randint(50000, 200000)   # Normal volume
        
        data.append({
            'timestamp': ts.tz_convert('UTC'),  # Convert to UTC for consistency
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


def detect_level_touches(
    ohlcv_df: pd.DataFrame,
    context_engine: ContextEngine,
    touch_tolerance: float = 0.05
) -> List[Tuple[int, float, float, List[LevelKind]]]:
    """
    Scan price path for level touches.
    
    Returns list of (timestamp_ns, price, current_price, [level_kinds]) tuples
    where price touched a significant level (whole number, SMA, PM high/low).
    
    Args:
        ohlcv_df: OHLCV DataFrame
        context_engine: Initialized ContextEngine with level data
        touch_tolerance: How close to level counts as a "touch" ($0.05)
    
    Returns:
        List of touch events: (ts_ns, level_price, spot_price, [LevelKind, ...])
    """
    touches = []
    
    for idx, row in ohlcv_df.iterrows():
        ts_ns = int(row['timestamp'].value)  # Unix nanoseconds
        low = row['low']
        high = row['high']
        close = row['close']
        
        # Check for whole number touches (STRIKE levels)
        # Look for round strikes near the price action
        for strike in range(int(low) - 1, int(high) + 2):
            if low <= strike <= high:
                # Price touched this whole number
                touches.append((
                    ts_ns,
                    float(strike),
                    close,
                    [LevelKind.STRIKE]
                ))
        
        # Check for context-based levels (PM high/low, SMA-200)
        active_levels = context_engine.get_active_levels(close, ts_ns)
        
        for level_info in active_levels:
            level_price = level_info['level_price']
            level_kind = level_info['level_kind']
            
            # Did price touch this level?
            if low - touch_tolerance <= level_price <= high + touch_tolerance:
                touches.append((
                    ts_ns,
                    level_price,
                    close,
                    [level_kind]
                ))
    
    return touches


def get_future_prices(
    ohlcv_df: pd.DataFrame,
    current_ts_ns: int,
    lookforward_minutes: int = 5
) -> List[float]:
    """
    Get future prices for outcome labeling.
    
    Returns all close prices from current_ts forward for next N minutes.
    
    Args:
        ohlcv_df: OHLCV DataFrame
        current_ts_ns: Current timestamp in nanoseconds
        lookforward_minutes: How many minutes to look forward (default 5)
    
    Returns:
        List of future close prices
    """
    # Convert to pandas timestamp
    current_ts = pd.Timestamp(current_ts_ns, unit='ns', tz='UTC')
    end_ts = current_ts + pd.Timedelta(minutes=lookforward_minutes)
    
    # Filter future data
    future_mask = (
        (ohlcv_df['timestamp'] > current_ts) & 
        (ohlcv_df['timestamp'] <= end_ts)
    )
    
    future_prices = ohlcv_df[future_mask]['close'].tolist()
    return future_prices


def main():
    """
    Main pipeline execution.
    
    Generates synthetic data, detects level touches, calculates physics,
    labels outcomes, and runs experiments.
    """
    print("\n" + "="*70)
    print("🚀 SPYMASTER RESEARCH PIPELINE - AGENT D INTEGRATION")
    print("="*70)
    print()
    
    # ========== Step 1: Generate Synthetic Data ==========
    print("📊 Step 1: Generating synthetic OHLCV data...")
    ohlcv_df = generate_synthetic_price_path(
        num_minutes=480,      # 8 hours (04:00 - 12:00 ET)
        base_price=400.0,     # SPY-like price
        volatility=0.5,       # Moderate volatility
        trend=0.02            # Slight uptrend
    )
    print(f"   Generated {len(ohlcv_df)} 1-minute bars")
    print(f"   Price range: ${ohlcv_df['low'].min():.2f} - ${ohlcv_df['high'].max():.2f}")
    print()
    
    # ========== Step 2: Initialize Engines ==========
    print("🔧 Step 2: Initializing Context and Physics Engines...")
    context_engine = ContextEngine(ohlcv_df=ohlcv_df)
    physics_engine = PhysicsEngine()
    
    # Print detected context levels
    pm_high = context_engine.get_premarket_high()
    pm_low = context_engine.get_premarket_low()
    if pm_high and pm_low:
        print(f"   Pre-Market Range: ${pm_low:.2f} - ${pm_high:.2f}")
    print()
    
    # ========== Step 3: Detect Level Touches ==========
    print("🎯 Step 3: Scanning for level touches...")
    touches = detect_level_touches(ohlcv_df, context_engine)
    print(f"   Detected {len(touches)} level touch events")
    print()
    
    # ========== Step 4: Create Signals with Physics ==========
    print("⚛️  Step 4: Calculating physics metrics and creating signals...")
    signals: List[LevelSignalV1] = []
    
    for ts_ns, level_price, spot_price, level_kinds in touches:
        # Generate mock physics data (since we don't have real order book)
        # In production, this would use real MBP-10 snapshots
        # Note: ES prices are ~10x SPY (e.g., SPY 400.0 = ES 4000.0)
        es_price = level_price * 10.0  # Convert SPY price to ES equivalent
        
        mock_mbp10 = physics_engine.generate_mock_mbp10(
            timestamp_ns=ts_ns,
            level_price=es_price,
            symbol="ES"
        )
        
        mock_trades = physics_engine.generate_mock_trades(
            start_time_ns=ts_ns - 5_000_000_000,  # 5 seconds before
            num_trades=50,
            price_level=es_price,
            symbol="ES"
        )
        
        # Calculate physics metrics (using ES prices)
        wall_ratio = physics_engine.calculate_wall_ratio(
            mbp10=mock_mbp10,
            level_price=es_price,
            tolerance=0.25  # ES tick size
        )
        
        replenishment = physics_engine.detect_replenishment(
            trade_tape=mock_trades,
            mbp10_snapshots=[mock_mbp10],  # Simplified: single snapshot
            level_price=es_price,
            tolerance=0.25
        )
        
        tape_velocity = physics_engine.calculate_tape_velocity(
            trade_tape=mock_trades,
            current_time_ns=ts_ns,
            window_s=5.0
        )
        
        # Determine direction (UP if price is below level, DOWN if above)
        direction = Direction.UP if spot_price < level_price else Direction.DOWN
        
        # Check if first 15 minutes
        is_first_15m = context_engine.is_first_15m(ts_ns)
        
        # Get distance to SMA-200
        sma_200 = context_engine.get_sma_200_at_time(ts_ns)
        dist_to_sma = (spot_price - sma_200) if sma_200 else None
        
        # Create signal for each level kind at this touch
        for level_kind in level_kinds:
            signal = LevelSignalV1(
                event_id=str(uuid.uuid4()),
                ts_event_ns=ts_ns,
                symbol="SPY",
                spot=spot_price,
                level_price=level_price,
                level_kind=level_kind,
                direction=direction,
                distance=abs(spot_price - level_price),
                is_first_15m=is_first_15m,
                dist_to_sma_200=dist_to_sma,
                wall_ratio=wall_ratio,
                replenishment_speed_ms=replenishment,
                gamma_exposure=np.random.uniform(-50000, 50000),  # Mock GEX
                tape_velocity=tape_velocity,
            )
            
            signals.append(signal)
    
    print(f"   Created {len(signals)} level signals with physics metrics")
    print()
    
    # ========== Step 5: Label Outcomes ==========
    print("🏷️  Step 5: Labeling outcomes with forward price data...")
    labeled_count = 0
    
    for signal in signals:
        # Get future prices (next 5 minutes)
        future_prices = get_future_prices(
            ohlcv_df=ohlcv_df,
            current_ts_ns=signal.ts_event_ns,
            lookforward_minutes=5
        )
        
        if not future_prices:
            # No future data available (end of dataset)
            signal.outcome = OutcomeLabel.UNDEFINED
            continue
        
        # Determine direction string for labeler
        direction_str = "UP" if signal.direction == Direction.UP else "DOWN"
        
        # Label the outcome
        outcome = get_outcome(
            signal_price=signal.level_price,
            future_prices=future_prices,
            direction=direction_str
        )
        
        signal.outcome = outcome
        signal.future_price_5min = future_prices[-1] if future_prices else None
        labeled_count += 1
    
    print(f"   Labeled {labeled_count} signals")
    outcome_counts = {
        "BOUNCE": sum(1 for s in signals if s.outcome == OutcomeLabel.BOUNCE),
        "BREAK": sum(1 for s in signals if s.outcome == OutcomeLabel.BREAK),
        "CHOP": sum(1 for s in signals if s.outcome == OutcomeLabel.CHOP),
        "UNDEFINED": sum(1 for s in signals if s.outcome == OutcomeLabel.UNDEFINED),
    }
    print(f"   Distribution: {outcome_counts}")
    print()
    
    # ========== Step 6: Run Experiments ==========
    print("🔬 Step 6: Running experiments and statistical analysis...")
    print()
    
    experiment_runner = ExperimentRunner(signals=signals)
    
    # Run simple backtest by level kind
    backtest_results = experiment_runner.run_simple_backtest(print_report=True)
    
    # Run physics correlation analysis
    correlation_results = experiment_runner.run_physics_correlation(print_report=True)
    
    # Run time-based analysis
    time_results = experiment_runner.run_time_based_analysis(print_report=True)
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print()
    print(f"📌 Total Signals Generated: {len(signals)}")
    print(f"📌 Labeled Outcomes: {labeled_count}")
    print(f"📌 Unique Level Kinds: {len(set(s.level_kind for s in signals))}")
    print()
    print("🎯 Key Insights:")
    
    # Find best performing level kind
    if backtest_results:
        best_level = max(backtest_results.items(), key=lambda x: x[1]['bounce_rate'])
        print(f"   • Best performing level: {best_level[0]} ({best_level[1]['bounce_rate']*100:.1f}% bounce rate)")
    
    if correlation_results and correlation_results.get('sample_size', 0) > 0:
        print(f"   • wall_ratio correlation: {correlation_results['wall_ratio_correlation']:+.3f}")
    
    print()
    print("📂 Next Steps:")
    print("   • Adjust synthetic data parameters to test different scenarios")
    print("   • Integrate real DBN data from dbn-data/ directory")
    print("   • Export signals to Parquet for further analysis")
    print("   • Build ML models using labeled features")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

