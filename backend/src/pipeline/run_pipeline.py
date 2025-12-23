"""
Run Pipeline - Real DBN Data + Production Engines

Complete research pipeline that integrates:
- Agent A: Physics Engine (now using production BarrierEngine + TapeEngine)
- Agent B: Context Engine (level identification + timing)
- Agent C: Labeler (outcome classification) + Experiment Runner (analysis)
- Production FuelEngine (gamma exposure from real option flow)

Pipeline Flow (with Real Data per PLAN.md §7):
1. Load real ES futures data from DBN files (MBP-10 + trades)
2. Load SPY option trades from Bronze tier (via Polygon API download)
3. Build OHLCV from ES trades, convert ES → SPY prices
4. Initialize MarketState with all data sources
5. Initialize production engines (BarrierEngine, TapeEngine, FuelEngine)
6. Detect level touches using ContextEngine
7. Calculate real physics metrics at each touch
8. Create LevelSignalV1 objects with complete features
9. Label outcomes with forward price data
10. Run statistical analysis

Usage:
    cd backend/

    # List available dates
    uv run python -m src.pipeline.run_pipeline --list-dates

    # Run pipeline for specific date
    uv run python -m src.pipeline.run_pipeline --date 2025-12-18

    # Run pipeline for most recent date
    uv run python -m src.pipeline.run_pipeline
"""

import argparse
import sys
import uuid
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Import data loading
from src.ingestor.dbn_ingestor import DBNIngestor
from src.lake.bronze_writer import BronzeReader

# Import production engines
from src.core.market_state import MarketState
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine

# Import Agent B: Context Engine
from src.features.context_engine import ContextEngine

# Import Agent C: Labeler + Experiment Runner
from src.research.labeler import get_outcome
from src.research.experiment_runner import ExperimentRunner

# Import schemas and event types
from src.common.schemas.levels_signals import LevelSignalV1, LevelKind, Direction, OutcomeLabel
from src.common.event_types import MBP10, FuturesTrade, OptionTrade, Aggressor

# Import Black-Scholes calculator for real greeks (we NEVER estimate)
from src.core.black_scholes import BlackScholesCalculator, compute_greeks_for_dataframe


def build_ohlcv_from_trades(
    trades: List[FuturesTrade],
    convert_to_spy: bool = True
) -> pd.DataFrame:
    """
    Build 1-minute OHLCV bars from ES futures trades.

    Args:
        trades: List of FuturesTrade objects
        convert_to_spy: If True, divide prices by 10 to get SPY equivalent

    Returns:
        DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if not trades:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert to DataFrame for aggregation
    # Filter out ES price outliers (reasonable range: 3000-10000)
    data = []
    for trade in trades:
        if 3000 < trade.price < 10000:  # Filter outliers
            data.append({
                'ts_event_ns': trade.ts_event_ns,
                'price': trade.price,
                'size': trade.size
            })

    if not data:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(data)

    # Convert timestamp to pandas datetime (UTC)
    df['timestamp'] = pd.to_datetime(df['ts_event_ns'], unit='ns', utc=True)

    # Set timestamp as index for resampling
    df.set_index('timestamp', inplace=True)

    # Resample to 1-minute bars
    ohlcv = df['price'].resample('1min').agg(['first', 'max', 'min', 'last'])
    ohlcv.columns = ['open', 'high', 'low', 'close']

    # Sum volume
    ohlcv['volume'] = df['size'].resample('1min').sum()

    # Drop bars with no data
    ohlcv = ohlcv.dropna(subset=['open'])

    # Reset index to get timestamp column
    ohlcv = ohlcv.reset_index()

    # Convert ES prices to SPY equivalent (ES ≈ SPY × 10)
    if convert_to_spy:
        for col in ['open', 'high', 'low', 'close']:
            ohlcv[col] = ohlcv[col] / 10.0

    return ohlcv


def load_real_data(
    date: str,
    dbn_ingestor: DBNIngestor,
    bronze_reader: BronzeReader,
    max_mbp10: int = 100000  # Limit MBP-10 to avoid loading 9GB into memory
) -> Tuple[List[FuturesTrade], List[MBP10], pd.DataFrame]:
    """
    Load real data from DBN files and Bronze tier.

    Args:
        date: Date string in YYYY-MM-DD format
        dbn_ingestor: DBNIngestor instance
        bronze_reader: BronzeReader instance
        max_mbp10: Maximum MBP-10 snapshots to load (default 100k, ~1% of full day)

    Returns:
        Tuple of (trades, mbp10_snapshots, option_trades_df)
    """
    import itertools

    print(f"   Loading ES futures trades for {date}...", flush=True)
    trades = list(dbn_ingestor.read_trades(date=date))
    print(f"   Loaded {len(trades):,} ES trades", flush=True)

    print(f"   Loading ES MBP-10 data for {date} (max {max_mbp10:,})...", flush=True)
    # Stream MBP-10 with limit to avoid loading 9GB into memory
    mbp10_iter = dbn_ingestor.read_mbp10(date=date)
    mbp10_snapshots = list(itertools.islice(mbp10_iter, max_mbp10))
    print(f"   Loaded {len(mbp10_snapshots):,} MBP-10 snapshots", flush=True)

    print(f"   Loading SPY option trades for {date}...", flush=True)
    option_trades_df = bronze_reader.read_option_trades(underlying='SPY', date=date)
    print(f"   Loaded {len(option_trades_df):,} option trades", flush=True)

    return trades, mbp10_snapshots, option_trades_df


def initialize_market_state(
    trades: List[FuturesTrade],
    mbp10_snapshots: List[MBP10],
    option_trades_df: pd.DataFrame,
    trading_date: str,
    max_mbp10: int = 50000  # Limit MBP-10 to avoid memory issues
) -> MarketState:
    """
    Initialize MarketState with real data and computed Black-Scholes greeks.

    Uses vectorized numpy operations for ~100x speedup on millions of options.

    Args:
        trades: List of ES FuturesTrade objects
        mbp10_snapshots: List of ES MBP10 objects
        option_trades_df: DataFrame of SPY option trades
        trading_date: Trading date (YYYY-MM-DD) for 0DTE expiration
        max_mbp10: Maximum MBP-10 snapshots to load

    Returns:
        Initialized MarketState
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    market_state = MarketState(max_buffer_window_seconds=120.0)

    # Load trades into market state and get current spot price
    log.info(f"Loading {len(trades):,} ES trades...")
    spot_price = None
    for trade in trades:
        market_state.update_es_trade(trade)
        if 3000 < trade.price < 10000:  # Valid ES range
            spot_price = trade.price / 10.0  # Convert to SPY

    if spot_price is None:
        log.warning("No valid ES trades found for spot price, using fallback")
        spot_price = 600.0  # Fallback

    # Load MBP-10 snapshots (limited to avoid memory issues)
    snapshots_to_load = mbp10_snapshots[:max_mbp10] if len(mbp10_snapshots) > max_mbp10 else mbp10_snapshots
    log.info(f"Loading {len(snapshots_to_load):,} MBP-10 snapshots...")
    for mbp in snapshots_to_load:
        market_state.update_es_mbp10(mbp)

    # Load option trades with VECTORIZED Black-Scholes greeks (fast!)
    if not option_trades_df.empty:
        log.info(f"Computing Black-Scholes greeks for {len(option_trades_df):,} options (vectorized)...")

        # Compute all greeks at once using numpy vectorization
        delta_arr, gamma_arr = compute_greeks_for_dataframe(
            df=option_trades_df,
            spot=spot_price,
            exp_date=trading_date
        )

        # Add greeks as columns for fast access
        option_trades_df = option_trades_df.copy()
        option_trades_df['delta'] = delta_arr
        option_trades_df['gamma'] = gamma_arr

        log.info(f"Loading options into MarketState...")
        option_count = 0
        for idx in range(len(option_trades_df)):
            try:
                row = option_trades_df.iloc[idx]
                # Convert aggressor int to Aggressor enum
                aggressor_val = row.get('aggressor', 0)
                if hasattr(aggressor_val, 'value'):
                    aggressor_enum = aggressor_val  # Already an enum
                else:
                    aggressor_enum = Aggressor(int(aggressor_val) if aggressor_val and aggressor_val != '<NA>' else 0)

                trade = OptionTrade(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=row.get('source', 'polygon_rest'),
                    underlying=row.get('underlying', 'SPY'),
                    option_symbol=row['option_symbol'],
                    exp_date=str(row['exp_date']),
                    strike=float(row['strike']),
                    right=row['right'],
                    price=float(row['price']),
                    size=int(row['size']),
                    opt_bid=row.get('opt_bid'),
                    opt_ask=row.get('opt_ask'),
                    aggressor=aggressor_enum,
                    conditions=None,
                    seq=row.get('seq')
                )
                market_state.update_option_trade(
                    trade,
                    delta=row['delta'],
                    gamma=row['gamma']
                )
                option_count += 1

                # Progress logging every 100k
                if option_count % 100000 == 0:
                    log.info(f"  Processed {option_count:,} options...")

            except Exception as e:
                continue  # Skip malformed rows

        log.info(f"Loaded {option_count:,} options with Black-Scholes greeks")

    return market_state


def detect_level_touches(
    ohlcv_df: pd.DataFrame,
    context_engine: ContextEngine,
    touch_tolerance: float = 0.05
) -> List[Tuple[int, float, float, List[LevelKind]]]:
    """
    Scan price path for level touches.

    Returns list of (timestamp_ns, level_price, spot_price, [level_kinds]) tuples.

    Args:
        ohlcv_df: OHLCV DataFrame (SPY prices)
        context_engine: Initialized ContextEngine
        touch_tolerance: How close to level counts as a touch ($0.05)

    Returns:
        List of touch events
    """
    touches = []

    for idx, row in ohlcv_df.iterrows():
        ts_ns = int(row['timestamp'].value)
        low = row['low']
        high = row['high']
        close = row['close']

        # Check for whole number touches (STRIKE levels)
        for strike in range(int(low) - 1, int(high) + 2):
            if low <= strike <= high:
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

            if low - touch_tolerance <= level_price <= high + touch_tolerance:
                touches.append((
                    ts_ns,
                    level_price,
                    close,
                    [level_kind]
                ))

    return touches


def calculate_real_physics(
    ts_ns: int,
    level_price: float,
    spot_price: float,
    direction: Direction,
    market_state: MarketState,
    barrier_engine: BarrierEngine,
    tape_engine: TapeEngine,
    fuel_engine: FuelEngine,
    exp_date: str
) -> Dict[str, Any]:
    """
    Calculate real physics metrics using production engines.

    Args:
        ts_ns: Touch timestamp in nanoseconds
        level_price: Level price in SPY terms
        spot_price: Current spot price in SPY terms
        direction: Direction (UP or DOWN)
        market_state: Initialized MarketState
        barrier_engine: BarrierEngine instance
        tape_engine: TapeEngine instance
        fuel_engine: FuelEngine instance
        exp_date: Expiration date filter for options (YYYY-MM-DD)

    Returns:
        Dict with all physics metrics
    """
    # Convert direction to barrier engine format
    barrier_direction = (
        BarrierDirection.RESISTANCE if direction == Direction.UP
        else BarrierDirection.SUPPORT
    )

    # Compute barrier state (ES MBP-10 based)
    barrier_metrics = barrier_engine.compute_barrier_state(
        level_price=level_price,
        direction=barrier_direction,
        market_state=market_state
    )

    # Compute tape state (ES trades based)
    tape_metrics = tape_engine.compute_tape_state(
        level_price=level_price,
        market_state=market_state
    )

    # Compute fuel state (SPY options based)
    fuel_metrics = fuel_engine.compute_fuel_state(
        level_price=level_price,
        market_state=market_state,
        exp_date_filter=exp_date
    )

    return {
        # Barrier metrics
        'barrier_state': barrier_metrics.state.value,
        'barrier_delta_liq': barrier_metrics.delta_liq,
        'barrier_replenishment_ratio': barrier_metrics.replenishment_ratio,
        'barrier_added': barrier_metrics.added_size,
        'barrier_canceled': barrier_metrics.canceled_size,
        'barrier_filled': barrier_metrics.filled_size,
        'defending_quote_price': barrier_metrics.defending_quote.get('price', 0.0),
        'defending_quote_size': barrier_metrics.defending_quote.get('size', 0),
        'wall_ratio': barrier_metrics.depth_in_zone / 5000.0 if barrier_metrics.depth_in_zone else 0.0,

        # Tape metrics
        'tape_imbalance': tape_metrics.imbalance,
        'tape_buy_vol': tape_metrics.buy_vol,
        'tape_sell_vol': tape_metrics.sell_vol,
        'tape_velocity': tape_metrics.velocity,
        'sweep_detected': tape_metrics.sweep.detected,
        'sweep_direction': tape_metrics.sweep.direction,
        'sweep_notional': tape_metrics.sweep.notional,

        # Fuel metrics
        'fuel_effect': fuel_metrics.effect.value,
        'net_dealer_gamma': fuel_metrics.net_dealer_gamma,
        'fuel_call_wall': fuel_metrics.call_wall.strike if fuel_metrics.call_wall else None,
        'fuel_put_wall': fuel_metrics.put_wall.strike if fuel_metrics.put_wall else None,
        'fuel_hvl': fuel_metrics.hvl,
        'gamma_exposure': fuel_metrics.net_dealer_gamma,

        # Confidence scores
        'barrier_confidence': barrier_metrics.confidence,
        'tape_confidence': tape_metrics.confidence,
        'fuel_confidence': fuel_metrics.confidence,
    }


def get_future_prices(
    ohlcv_df: pd.DataFrame,
    current_ts_ns: int,
    lookforward_minutes: int = 5
) -> List[float]:
    """
    Get future prices for outcome labeling.

    Args:
        ohlcv_df: OHLCV DataFrame
        current_ts_ns: Current timestamp in nanoseconds
        lookforward_minutes: How many minutes to look forward

    Returns:
        List of future close prices
    """
    current_ts = pd.Timestamp(current_ts_ns, unit='ns', tz='UTC')
    end_ts = current_ts + pd.Timedelta(minutes=lookforward_minutes)

    future_mask = (
        (ohlcv_df['timestamp'] > current_ts) &
        (ohlcv_df['timestamp'] <= end_ts)
    )

    return ohlcv_df[future_mask]['close'].tolist()


def main(date: Optional[str] = None):
    """
    Main pipeline execution with real data.

    Args:
        date: Trading date (YYYY-MM-DD), or None for most recent
    """
    print("\n" + "="*70)
    print("SPYMASTER RESEARCH PIPELINE - REAL DATA + PRODUCTION ENGINES")
    print("="*70)
    print()

    # Initialize data sources
    dbn_ingestor = DBNIngestor()
    bronze_reader = BronzeReader()

    # Get available dates
    available_dates = dbn_ingestor.get_available_dates('trades')
    weekday_dates = [d for d in available_dates
                     if datetime.strptime(d, '%Y-%m-%d').weekday() < 5]

    if not weekday_dates:
        print("ERROR: No DBN data found. Check dbn-data/ directory.")
        return

    # Select date
    if date is None:
        date = weekday_dates[-1]  # Most recent weekday
    elif date not in available_dates:
        print(f"ERROR: Date {date} not found in DBN data.")
        print(f"Available dates: {', '.join(available_dates)}")
        return

    print(f"Processing date: {date}")
    print()

    # ========== Step 1: Load Real Data ==========
    print("Step 1: Loading real data...")
    trades, mbp10_snapshots, option_trades_df = load_real_data(
        date=date,
        dbn_ingestor=dbn_ingestor,
        bronze_reader=bronze_reader
    )

    if not trades:
        print("ERROR: No ES trades found for this date.")
        return

    # ========== Step 2: Build OHLCV ==========
    print("\nStep 2: Building OHLCV from ES trades...")
    ohlcv_df = build_ohlcv_from_trades(trades, convert_to_spy=True)
    print(f"   Generated {len(ohlcv_df)} 1-minute bars")
    print(f"   SPY price range: ${ohlcv_df['low'].min():.2f} - ${ohlcv_df['high'].max():.2f}")
    print()

    # ========== Step 3: Initialize Engines ==========
    print("Step 3: Initializing engines...")

    # Initialize MarketState with real data and Black-Scholes greeks
    print("   Initializing MarketState with Black-Scholes greeks...")
    market_state = initialize_market_state(
        trades=trades,
        mbp10_snapshots=mbp10_snapshots,
        option_trades_df=option_trades_df,
        trading_date=date,  # For 0DTE expiration calculation
        max_mbp10=50000  # Limit for memory
    )
    print(f"   Buffer stats: {market_state.get_buffer_stats()}")

    # Initialize production engines
    barrier_engine = BarrierEngine()
    tape_engine = TapeEngine()
    fuel_engine = FuelEngine()

    # Initialize context engine
    context_engine = ContextEngine(ohlcv_df=ohlcv_df)
    pm_high = context_engine.get_premarket_high()
    pm_low = context_engine.get_premarket_low()
    if pm_high and pm_low:
        print(f"   Pre-Market Range: ${pm_low:.2f} - ${pm_high:.2f}")
    print()

    # ========== Step 4: Detect Level Touches ==========
    print("Step 4: Scanning for level touches...")
    touches = detect_level_touches(ohlcv_df, context_engine)
    print(f"   Detected {len(touches)} level touch events")

    # Limit touches for performance
    max_touches = 200
    if len(touches) > max_touches:
        print(f"   (Limiting to {max_touches} touches for performance)")
        touches = touches[:max_touches]
    print()

    # ========== Step 5: Calculate Physics and Create Signals ==========
    print("Step 5: Calculating real physics metrics...")
    signals: List[LevelSignalV1] = []

    for i, (ts_ns, level_price, spot_price, level_kinds) in enumerate(touches):
        if (i + 1) % 50 == 0:
            print(f"   Processing touch {i+1}/{len(touches)}...")

        # Determine direction
        direction = Direction.UP if spot_price < level_price else Direction.DOWN

        # Calculate real physics
        physics = calculate_real_physics(
            ts_ns=ts_ns,
            level_price=level_price,
            spot_price=spot_price,
            direction=direction,
            market_state=market_state,
            barrier_engine=barrier_engine,
            tape_engine=tape_engine,
            fuel_engine=fuel_engine,
            exp_date=date
        )

        # Get context features
        is_first_15m = context_engine.is_first_15m(ts_ns)
        sma_200 = context_engine.get_sma_200_at_time(ts_ns)
        dist_to_sma = (spot_price - sma_200) if sma_200 else None

        # Create signal for each level kind
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
                wall_ratio=physics['wall_ratio'],
                replenishment_speed_ms=physics['barrier_replenishment_ratio'] * 100,  # Scaled estimate
                gamma_exposure=physics['gamma_exposure'],
                tape_velocity=physics['tape_velocity'],
                # Additional physics fields
                barrier_state=physics['barrier_state'],
                barrier_delta_liq=physics['barrier_delta_liq'],
                tape_imbalance=physics['tape_imbalance'],
                fuel_effect=physics['fuel_effect'],
            )
            signals.append(signal)

    print(f"   Created {len(signals)} level signals with real physics")
    print()

    # ========== Step 6: Label Outcomes ==========
    print("Step 6: Labeling outcomes...")
    labeled_count = 0

    for signal in signals:
        future_prices = get_future_prices(
            ohlcv_df=ohlcv_df,
            current_ts_ns=signal.ts_event_ns,
            lookforward_minutes=5
        )

        if not future_prices:
            signal.outcome = OutcomeLabel.UNDEFINED
            continue

        direction_str = "UP" if signal.direction == Direction.UP else "DOWN"
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

    # ========== Step 7: Run Experiments ==========
    print("Step 7: Running experiments...")
    print()

    experiment_runner = ExperimentRunner(signals=signals)

    # Run analyses
    backtest_results = experiment_runner.run_simple_backtest(print_report=True)
    correlation_results = experiment_runner.run_physics_correlation(print_report=True)
    time_results = experiment_runner.run_time_based_analysis(print_report=True)

    # ========== Summary ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print()
    print(f"Date Processed: {date}")
    print(f"Total Signals: {len(signals)}")
    print(f"Labeled Outcomes: {labeled_count}")
    print(f"Unique Level Kinds: {len(set(s.level_kind for s in signals))}")
    print()

    if backtest_results:
        best_level = max(backtest_results.items(), key=lambda x: x[1]['bounce_rate'])
        print(f"Best performing level: {best_level[0]} ({best_level[1]['bounce_rate']*100:.1f}% bounce rate)")

    if correlation_results and correlation_results.get('sample_size', 0) > 0:
        print(f"wall_ratio correlation: {correlation_results['wall_ratio_correlation']:+.3f}")

    print()


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run Spymaster research pipeline with real DBN data'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Trading date to process (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available DBN dates'
    )

    args = parser.parse_args()

    # List dates mode
    if args.list_dates:
        dbn_ingestor = DBNIngestor()
        dates = dbn_ingestor.get_available_dates('trades')
        print(f"Available DBN dates ({len(dates)}):")
        for d in dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            day_name = dt.strftime('%a')
            print(f"  {d} ({day_name})")
        return 0

    # Run pipeline
    try:
        main(date=args.date)
        return 0
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(cli())
