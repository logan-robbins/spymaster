"""
Batch Process - Multi-Day Pipeline Execution

Runs the research pipeline across all available DBN dates and aggregates results.

Features:
- Discovers all available DBN dates
- Downloads missing options data from Polygon API
- Runs pipeline for each date
- Aggregates signals into combined dataset
- Exports to Parquet for ML training

Usage:
    cd backend/

    # Process all available dates
    uv run python -m src.pipeline.batch_process

    # Process specific dates only
    uv run python -m src.pipeline.batch_process --dates 2025-12-18,2025-12-19

    # Skip options download (assume already downloaded)
    uv run python -m src.pipeline.batch_process --skip-download

    # Specify output path
    uv run python -m src.pipeline.batch_process --output /path/to/output.parquet
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestor.dbn_ingestor import DBNIngestor
from src.ingestor.polygon_historical import PolygonHistoricalDownloader
from src.pipeline.run_pipeline import main as run_pipeline_main
from src.common.schemas.levels_signals import LevelSignalV1


# Default output path
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / 'data' / 'lake' / 'gold' / 'research'


def discover_available_dates() -> List[str]:
    """
    Discover all available weekday dates from DBN data.

    Returns:
        List of date strings in YYYY-MM-DD format (weekdays only)
    """
    dbn_ingestor = DBNIngestor()
    all_dates = dbn_ingestor.get_available_dates('trades')

    # Filter to weekdays only
    weekday_dates = []
    for date_str in all_dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        if dt.weekday() < 5:  # Monday=0 to Friday=4
            weekday_dates.append(date_str)

    return sorted(weekday_dates)


def check_options_downloaded(date: str) -> bool:
    """
    Check if SPY options data already exists for a date.

    Args:
        date: Date in YYYY-MM-DD format

    Returns:
        True if options data exists in Bronze tier
    """
    bronze_path = (
        Path(__file__).parent.parent.parent / 'data' / 'lake' / 'bronze' /
        'options' / 'trades' / 'underlying=SPY' / f'date={date}'
    )

    if not bronze_path.exists():
        return False

    # Check for Parquet files
    parquet_files = list(bronze_path.glob('*.parquet'))
    return len(parquet_files) > 0


def download_missing_options(
    dates: List[str],
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Download SPY options data for dates that are missing.

    Args:
        dates: List of dates to check and download
        force: If True, re-download even if data exists
        dry_run: If True, show what would be downloaded

    Returns:
        Dict of {date: trade_count} for downloaded dates
    """
    try:
        downloader = PolygonHistoricalDownloader()
    except ValueError as e:
        print(f"ERROR: Cannot initialize Polygon downloader: {e}")
        return {}

    results = {}

    for date in dates:
        if not force and check_options_downloaded(date):
            print(f"  {date}: Skipping (already downloaded)")
            results[date] = -1
            continue

        if dry_run:
            print(f"  {date}: Would download options")
            results[date] = 0
            continue

        try:
            count = downloader.download_options_for_date(date)
            results[date] = count
            print(f"  {date}: Downloaded {count:,} trades")
        except Exception as e:
            print(f"  {date}: ERROR - {e}")
            results[date] = -2

    return results


def run_pipeline_for_date(date: str) -> Optional[List[Dict[str, Any]]]:
    """
    Run the research pipeline for a single date.

    Args:
        date: Date in YYYY-MM-DD format

    Returns:
        List of signal dictionaries, or None if failed
    """
    try:
        # Import here to avoid circular imports and capture signals
        from src.pipeline.run_pipeline import (
            build_ohlcv_from_trades,
            load_real_data,
            initialize_market_state,
            detect_level_touches,
            calculate_real_physics,
            get_anchor_and_future_prices,
        )
        from src.ingestor.dbn_ingestor import DBNIngestor
        from src.lake.bronze_writer import BronzeReader
        from src.core.market_state import MarketState
        from src.core.barrier_engine import BarrierEngine
        from src.core.tape_engine import TapeEngine
        from src.core.fuel_engine import FuelEngine
        from src.features.context_engine import ContextEngine
        from src.research.labeler import get_outcome
        from src.common.schemas.levels_signals import Direction, OutcomeLabel
        from src.common.config import CONFIG
        import uuid

        print(f"\n--- Processing {date} ---")

        # Initialize data sources
        dbn_ingestor = DBNIngestor()
        bronze_reader = BronzeReader()

        # Load data
        trades, mbp10_snapshots, option_trades_df = load_real_data(
            date=date,
            dbn_ingestor=dbn_ingestor,
            bronze_reader=bronze_reader
        )

        if not trades:
            print(f"  No ES trades found for {date}")
            return None

        # Build OHLCV
        ohlcv_df = build_ohlcv_from_trades(trades, convert_to_spy=True)
        print(f"  OHLCV: {len(ohlcv_df)} bars, ${ohlcv_df['low'].min():.2f}-${ohlcv_df['high'].max():.2f}")

        # Initialize engines with Black-Scholes greeks (we NEVER estimate)
        market_state = initialize_market_state(
            trades=trades,
            mbp10_snapshots=mbp10_snapshots,
            option_trades_df=option_trades_df,
            trading_date=date,  # For 0DTE expiration calculation
            max_mbp10=50000
        )

        barrier_engine = BarrierEngine()
        tape_engine = TapeEngine()
        fuel_engine = FuelEngine()
        context_engine = ContextEngine(ohlcv_df=ohlcv_df)

        # Detect touches
        touches = detect_level_touches(ohlcv_df, context_engine)
        print(f"  Detected {len(touches)} touches")

        # Limit for performance
        max_touches = 200
        if len(touches) > max_touches:
            touches = touches[:max_touches]

        # Create signals
        signals = []
        for ts_ns, level_price, spot_price, level_kinds in touches:
            direction = Direction.UP if spot_price < level_price else Direction.DOWN

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

            is_first_15m = context_engine.is_first_15m(ts_ns)
            sma_200 = context_engine.get_sma_200_at_time(ts_ns)
            dist_to_sma = (spot_price - sma_200) if sma_200 else None

            for level_kind in level_kinds:
                # Get future prices for labeling
                anchor_price, future_prices, confirm_ts_ns = get_anchor_and_future_prices(
                    ohlcv_df,
                    ts_ns,
                    confirmation_seconds=CONFIG.CONFIRMATION_WINDOW_SECONDS,
                    lookforward_minutes=5
                )
                direction_str = "UP" if direction == Direction.UP else "DOWN"

                if future_prices and anchor_price is not None:
                    outcome = get_outcome(
                        level_price,
                        future_prices,
                        direction_str,
                        threshold=CONFIG.OUTCOME_THRESHOLD
                    )
                    future_price_5min = future_prices[-1]
                else:
                    outcome = OutcomeLabel.UNDEFINED
                    future_price_5min = None

                # Create signal dict
                signal_dict = {
                    'date': date,
                    'event_id': str(uuid.uuid4()),
                    'ts_event_ns': ts_ns,
                    'symbol': 'SPY',
                    'spot': spot_price,
                    'level_price': level_price,
                    'level_kind': level_kind.value,
                    'direction': direction.value,
                    'distance': abs(spot_price - level_price),
                    'is_first_15m': is_first_15m,
                    'dist_to_sma_200': dist_to_sma,
                    'wall_ratio': physics['wall_ratio'],
                    'gamma_exposure': physics['gamma_exposure'],
                    'tape_velocity': physics['tape_velocity'],
                    'barrier_state': physics['barrier_state'],
                    'barrier_delta_liq': physics['barrier_delta_liq'],
                    'tape_imbalance': physics['tape_imbalance'],
                    'fuel_effect': physics['fuel_effect'],
                    'outcome': outcome.value,
                    'future_price_5min': future_price_5min,
                    'confirm_ts_ns': confirm_ts_ns,
                    'anchor_spot': anchor_price,
                }
                signals.append(signal_dict)

        print(f"  Created {len(signals)} signals")
        return signals

    except Exception as e:
        print(f"  ERROR processing {date}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(all_signals: List[List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Aggregate signals from multiple dates into a single DataFrame.

    Args:
        all_signals: List of signal lists (one per date)

    Returns:
        Combined DataFrame
    """
    # Flatten
    flat_signals = []
    for date_signals in all_signals:
        if date_signals:
            flat_signals.extend(date_signals)

    if not flat_signals:
        return pd.DataFrame()

    df = pd.DataFrame(flat_signals)
    return df


def export_to_parquet(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Export DataFrame to Parquet with ZSTD compression.

    Args:
        df: DataFrame to export
        output_path: Path for output file
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to Arrow table
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write with ZSTD compression
    pq.write_table(
        table,
        output_path,
        compression='zstd',
        compression_level=3
    )

    print(f"\nExported {len(df):,} signals to {output_path}")


def main(
    dates: Optional[List[str]] = None,
    skip_download: bool = False,
    output_path: Optional[Path] = None,
    dry_run: bool = False
):
    """
    Main batch processing function.

    Args:
        dates: List of dates to process (None = all available)
        skip_download: If True, skip options download
        output_path: Path for output Parquet file
        dry_run: If True, show what would be done without executing
    """
    print("\n" + "="*70)
    print("SPYMASTER BATCH PROCESSOR")
    print("="*70)
    print()

    # Discover available dates
    available_dates = discover_available_dates()
    print(f"Available DBN dates: {len(available_dates)}")
    for d in available_dates:
        status = "DOWNLOADED" if check_options_downloaded(d) else "MISSING"
        print(f"  {d}: {status}")
    print()

    # Filter to requested dates
    if dates:
        process_dates = [d for d in dates if d in available_dates]
        if len(process_dates) != len(dates):
            missing = set(dates) - set(process_dates)
            print(f"WARNING: Some dates not found in DBN data: {missing}")
    else:
        process_dates = available_dates

    if not process_dates:
        print("ERROR: No dates to process")
        return 1

    print(f"Will process {len(process_dates)} dates: {', '.join(process_dates)}")
    print()

    # Download missing options
    if not skip_download:
        print("Step 1: Checking/downloading options data...")
        download_results = download_missing_options(process_dates, dry_run=dry_run)
        downloaded = sum(1 for v in download_results.values() if v > 0)
        print(f"  Downloaded {downloaded} dates")
        print()

    if dry_run:
        print("[DRY RUN] Would process dates and export to Parquet")
        return 0

    # Run pipeline for each date
    print("Step 2: Running pipeline for each date...")
    all_signals = []

    for date in process_dates:
        try:
            signals = run_pipeline_for_date(date)
            if signals:
                all_signals.append(signals)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"  {date}: FAILED - {e}")
            continue

    print()

    # Aggregate results
    print("Step 3: Aggregating results...")
    df = aggregate_results(all_signals)

    if df.empty:
        print("ERROR: No signals generated")
        return 1

    print(f"  Total signals: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique dates: {df['date'].nunique()}")

    # Export to Parquet
    print("\nStep 4: Exporting to Parquet...")
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH / 'signals_multi_day.parquet'

    export_to_parquet(df, output_path)

    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print()
    print(f"Total signals: {len(df):,}")
    print(f"Dates processed: {df['date'].nunique()}")
    print()

    # Outcome distribution
    if 'outcome' in df.columns:
        print("Outcome distribution:")
        for outcome, count in df['outcome'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {outcome}: {count:,} ({pct:.1f}%)")
    print()

    # Level kind distribution
    if 'level_kind' in df.columns:
        print("Level kind distribution:")
        for kind, count in df['level_kind'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {kind}: {count:,} ({pct:.1f}%)")
    print()

    print(f"Output: {output_path}")
    return 0


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch process research pipeline across multiple dates'
    )
    parser.add_argument(
        '--dates',
        type=str,
        help='Comma-separated list of dates to process (default: all)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading options data'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output Parquet file path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )

    args = parser.parse_args()

    # Parse dates
    dates = None
    if args.dates:
        dates = [d.strip() for d in args.dates.split(',')]

    # Parse output path
    output_path = None
    if args.output:
        output_path = Path(args.output)

    try:
        return main(
            dates=dates,
            skip_download=args.skip_download,
            output_path=output_path,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted")
        return 1
    except Exception as e:
        print(f"\n\nBatch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(cli())
