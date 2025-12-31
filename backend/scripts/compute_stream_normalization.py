"""Compute stream normalization statistics from state table."""
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from src.ml.stream_normalization import compute_stream_normalization_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Compute stream normalization statistics')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Data root directory (default: data)')
    parser.add_argument('--canonical-version', type=str, default='3.1.0',
                        help='Canonical version (default: 3.1.0)')
    parser.add_argument('--lookback-days', type=int, default=60,
                        help='Number of days of history to use (default: 60)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--stratify-by', type=str, nargs='+', default=['time_bucket'],
                        help='Columns to stratify by (default: time_bucket)')
    parser.add_argument('--output-name', type=str, default='current',
                        help='Output filename stem (default: current)')
    
    args = parser.parse_args()
    
    # Parse dates
    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
    else:
        end_date = pd.Timestamp.now().normalize()
    
    start_date = end_date - timedelta(days=args.lookback_days)
    
    logger.info(f"Computing stream normalization statistics")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Lookback: {args.lookback_days} days")
    logger.info(f"  Stratify by: {args.stratify_by}")
    
    # Load state tables for date range
    data_root = Path(args.data_root)
    state_base_versioned = data_root / "silver" / "state" / "es_level_state" / f"version={args.canonical_version}"
    state_base_unversioned = data_root / "silver" / "state" / "es_level_state"
    
    # Try versioned first, fall back to unversioned
    if state_base_versioned.exists():
        state_base = state_base_versioned
        logger.info(f"Using versioned state table: {state_base}")
    elif state_base_unversioned.exists():
        state_base = state_base_unversioned
        logger.info(f"Using unversioned state table: {state_base}")
    else:
        logger.error(f"State table directory not found: {state_base_versioned} or {state_base_unversioned}")
        logger.error("Run es_pipeline Stage 16 first to generate state tables")
        return 1
    
    # Collect all state tables in date range
    all_state_dfs = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        state_dir = state_base / f"date={date_str}"
        
        if state_dir.exists():
            parquet_files = list(state_dir.glob("*.parquet"))
            if parquet_files:
                logger.info(f"  Loading {date_str}...")
                for pq_file in parquet_files:
                    df = pd.read_parquet(pq_file)
                    all_state_dfs.append(df)
        
        current_date += timedelta(days=1)
    
    if not all_state_dfs:
        logger.error("No state tables found in date range")
        return 1
    
    # Concatenate all state data
    logger.info(f"Concatenating {len(all_state_dfs)} state table files...")
    state_df = pd.concat(all_state_dfs, ignore_index=True)
    logger.info(f"  Total samples: {len(state_df):,}")
    
    # Add time_bucket column if not present (for stratification)
    if 'time_bucket' in args.stratify_by and 'time_bucket' not in state_df.columns:
        logger.info("  Adding time_bucket column...")
        
        def assign_time_bucket(minutes_since_open):
            if pd.isna(minutes_since_open):
                return 'UNKNOWN'
            elif minutes_since_open < 15:
                return 'T0_15'
            elif minutes_since_open < 30:
                return 'T15_30'
            elif minutes_since_open < 60:
                return 'T30_60'
            elif minutes_since_open < 120:
                return 'T60_120'
            else:
                return 'T120_180'
        
        state_df['time_bucket'] = state_df['minutes_since_open'].apply(assign_time_bucket)
    
    # Compute statistics
    output_dir = data_root / "gold" / "streams" / "normalization"
    output_path = output_dir / f"{args.output_name}.json"
    
    logger.info("Computing normalization statistics...")
    stats = compute_stream_normalization_stats(
        state_df=state_df,
        stratify_by=args.stratify_by,
        output_path=output_path
    )
    
    logger.info(f"\nStream normalization statistics saved to: {output_path}")
    logger.info(f"  Global features: {len(stats['global_stats'])}")
    logger.info(f"  Stratified buckets: {len(stats['stratified_stats'])}")
    
    return 0


if __name__ == '__main__':
    exit(main())

