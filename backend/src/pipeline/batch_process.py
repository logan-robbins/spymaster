"""
Batch Process - Multi-Day Vectorized Pipeline Execution

Runs VectorizedPipeline across multiple dates and aggregates results.

Features:
- Discovers all available DBN dates
- Runs VectorizedPipeline for each date
- Aggregates signals into combined dataset
- Exports to Parquet for ML training

Usage:
    cd backend/

    # Process all available dates (with sufficient warmup)
    uv run python -m src.pipeline.batch_process

    # Process specific date range
    uv run python -m src.pipeline.batch_process --start-date 2025-12-10 --end-date 2025-12-19

    # Process specific dates only
    uv run python -m src.pipeline.batch_process --dates 2025-12-18,2025-12-19

    # Specify output path
    uv run python -m src.pipeline.batch_process --output /path/to/output.parquet

    # Dry run (show what would be processed)
    uv run python -m src.pipeline.batch_process --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestor.dbn_ingestor import DBNIngestor
from src.pipeline.vectorized_pipeline import VectorizedPipeline
from src.common.config import CONFIG


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default output path
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / 'data' / 'lake' / 'gold' / 'vectorized'


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


def get_dates_with_warmup(available_dates: List[str]) -> List[str]:
    """
    Filter dates to only those with sufficient warmup data.

    Requires:
    - SMA_WARMUP_DAYS (3) prior days for SMA-200/400
    - VOLUME_LOOKBACK_DAYS (7) prior days for relative volume

    Returns:
        Dates that have at least VOLUME_LOOKBACK_DAYS prior dates available
    """
    warmup_required = max(CONFIG.SMA_WARMUP_DAYS, CONFIG.VOLUME_LOOKBACK_DAYS)

    if len(available_dates) <= warmup_required:
        logger.warning(f"Not enough dates for warmup (need {warmup_required})")
        return []

    # Skip first warmup_required dates
    return available_dates[warmup_required:]


def filter_dates_by_range(
    dates: List[str],
    start_date: Optional[str],
    end_date: Optional[str]
) -> List[str]:
    """Filter dates to specified range."""
    result = dates

    if start_date:
        result = [d for d in result if d >= start_date]
    if end_date:
        result = [d for d in result if d <= end_date]

    return result


def run_pipeline_for_date(
    pipeline: VectorizedPipeline,
    date: str
) -> Optional[pd.DataFrame]:
    """
    Run VectorizedPipeline for a single date.

    Args:
        pipeline: Initialized VectorizedPipeline instance
        date: Date in YYYY-MM-DD format

    Returns:
        DataFrame of signals, or None if failed
    """
    try:
        signals_df = pipeline.run(date=date)

        if signals_df is not None and len(signals_df) > 0:
            logger.info(f"  Generated {len(signals_df):,} signals")
            return signals_df
        else:
            logger.warning(f"  No signals generated")
            return None

    except Exception as e:
        logger.error(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(all_signals: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate signals from multiple dates into a single DataFrame.

    Args:
        all_signals: List of DataFrames (one per date)

    Returns:
        Combined DataFrame
    """
    if not all_signals:
        return pd.DataFrame()

    return pd.concat(all_signals, ignore_index=True)


def export_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export DataFrame to Parquet with ZSTD compression.

    Args:
        df: DataFrame to export
        output_path: Path for output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        output_path,
        compression='zstd',
        compression_level=3
    )

    logger.info(f"Exported {len(df):,} signals to {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)

    logger.info(f"Total signals: {len(df):,}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique dates: {df['date'].nunique()}")

    # Outcome distribution
    if 'outcome' in df.columns:
        logger.info("\nOutcome distribution:")
        for outcome, count in df['outcome'].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"  {outcome}: {count:,} ({pct:.1f}%)")

    # Level kind distribution
    if 'level_kind' in df.columns:
        logger.info("\nLevel kind distribution:")
        for kind, count in df['level_kind'].value_counts().head(10).items():
            pct = count / len(df) * 100
            logger.info(f"  {kind}: {count:,} ({pct:.1f}%)")

    # Confluence level distribution (per REPORT.md Section 11)
    if 'confluence_level' in df.columns:
        logger.info("\nConfluence level distribution:")
        for level in sorted(df['confluence_level'].unique()):
            count = (df['confluence_level'] == level).sum()
            pct = count / len(df) * 100
            logger.info(f"  Level {level}: {count:,} ({pct:.1f}%)")

    # Check for required confluence features
    confluence_cols = ['confluence_level', 'breakout_state', 'gex_alignment', 'rel_vol_ratio']
    logger.info("\nConfluence features check:")
    for col in confluence_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"  {col}: PRESENT ({non_null:,} non-null)")
        else:
            logger.warning(f"  {col}: MISSING")


def main(
    dates: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[Path] = None,
    dry_run: bool = False
) -> int:
    """
    Main batch processing function.

    Args:
        dates: Explicit list of dates to process
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        output_path: Path for output Parquet file
        dry_run: If True, show what would be done without executing

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    logger.info("\n" + "=" * 70)
    logger.info("VECTORIZED PIPELINE BATCH PROCESSOR")
    logger.info("=" * 70)

    # Discover available dates
    available_dates = discover_available_dates()
    logger.info(f"\nAvailable DBN dates: {len(available_dates)}")
    if available_dates:
        logger.info(f"  Range: {available_dates[0]} to {available_dates[-1]}")

    # Determine dates to process
    if dates:
        # Explicit date list provided
        process_dates = [d for d in dates if d in available_dates]
        if len(process_dates) != len(dates):
            missing = set(dates) - set(process_dates)
            logger.warning(f"Dates not found in DBN data: {missing}")
    else:
        # Use date range with warmup filtering
        process_dates = get_dates_with_warmup(available_dates)
        process_dates = filter_dates_by_range(process_dates, start_date, end_date)

    if not process_dates:
        logger.error("No dates to process")
        return 1

    logger.info(f"\nWill process {len(process_dates)} dates:")
    logger.info(f"  {process_dates[0]} to {process_dates[-1]}")

    if dry_run:
        logger.info("\n[DRY RUN] Would process these dates:")
        for d in process_dates:
            logger.info(f"  - {d}")
        return 0

    # Initialize pipeline (reused across dates)
    pipeline = VectorizedPipeline()

    all_signals = []
    successful_dates = []
    failed_dates = []

    for i, date in enumerate(process_dates, 1):
        logger.info(f"\n[{i}/{len(process_dates)}] Processing {date}")

        try:
            signals_df = run_pipeline_for_date(pipeline, date)

            if signals_df is not None:
                all_signals.append(signals_df)
                successful_dates.append(date)
            else:
                failed_dates.append(date)

        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            failed_dates.append(date)

    # Aggregate results
    if not all_signals:
        logger.error("No signals generated")
        return 1

    logger.info("\nAggregating results...")
    df = aggregate_results(all_signals)

    # Export to Parquet
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH / 'signals_combined.parquet'

    export_to_parquet(df, output_path)

    # Print summary
    print_summary(df)

    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Successful: {len(successful_dates)} dates")
    logger.info(f"Failed: {len(failed_dates)} dates")
    logger.info(f"Output: {output_path}")

    return 0


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch process VectorizedPipeline across multiple dates'
    )
    parser.add_argument(
        '--dates',
        type=str,
        help='Comma-separated list of specific dates to process'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for range processing (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for range processing (YYYY-MM-DD)'
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
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=output_path,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        logger.warning("\nBatch processing interrupted")
        return 1
    except Exception as e:
        logger.error(f"\nBatch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(cli())
