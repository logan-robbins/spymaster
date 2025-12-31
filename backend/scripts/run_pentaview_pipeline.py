"""Run Pentaview pipeline to compute streams from state table."""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.pipeline.pipelines.pentaview_pipeline import build_pentaview_pipeline
from src.pipeline.core.stage import StageContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Pentaview stream pipeline')
    parser.add_argument('--date', type=str, required=True,
                        help='Date to process (YYYY-MM-DD)')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Data root directory (default: data)')
    parser.add_argument('--canonical-version', type=str, default='3.1.0',
                        help='Canonical version (default: 3.1.0)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date for batch processing (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date for batch processing (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_date = pd.Timestamp(args.start)
        end_date = pd.Timestamp(args.end)
        dates = pd.date_range(start_date, end_date, freq='D')
    else:
        dates = [pd.Timestamp(args.date)]
    
    logger.info(f"Running Pentaview pipeline for {len(dates)} date(s)")
    logger.info(f"  Canonical version: {args.canonical_version}")
    logger.info(f"  Data root: {args.data_root}")
    
    # Build pipeline
    pipeline = build_pentaview_pipeline()
    
    # Process each date
    success_count = 0
    error_count = 0
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {date_str}")
        logger.info(f"{'='*80}")
        
        try:
            # Run pipeline (pass date string, not context)
            # Note: pipeline.run() returns a DataFrame (last stage output)
            # For pentaview, this is the streams_df
            streams_df = pipeline.run(
                date=date_str,
                data_root=args.data_root,
                canonical_version=args.canonical_version
            )
            
            if streams_df is not None and not streams_df.empty:
                n_bars = len(streams_df)
                n_levels = streams_df['level_kind'].nunique() if 'level_kind' in streams_df.columns else 0
                logger.info(f"\n✓ Success: {date_str}")
                logger.info(f"  Generated {n_bars:,} stream bars across {n_levels} levels")
                success_count += 1
            else:
                logger.warning(f"\n⚠ Warning: {date_str} - No streams generated")
                error_count += 1
                
        except Exception as e:
            logger.error(f"\n✗ Error: {date_str}")
            logger.error(f"  {type(e).__name__}: {e}")
            error_count += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Pipeline Summary")
    logger.info(f"{'='*80}")
    logger.info(f"  Total dates: {len(dates)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())

