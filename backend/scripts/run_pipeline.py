"""
Run ES pipeline with optional checkpointing support.

Provides a single canonical entrypoint for full or incremental runs.

Usage:
    # Run full pipeline with checkpointing
    uv run python -m scripts.run_pipeline \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints
    
    # Run first 3 stages only
    uv run python -m scripts.run_pipeline \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints \
      --stop-at-stage 2
    
    # Resume from stage 3 (loads stage 2 checkpoint)
    uv run python -m scripts.run_pipeline \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints \
      --resume-from-stage 3
    
    # List available checkpoints
    uv run python -m scripts.run_pipeline \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints \
      --list
    
    # Clear checkpoints for date
    uv run python -m scripts.run_pipeline \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints \
      --clear

    # Run full pipeline for a date range (weekdays only by default)
    uv run python -m scripts.run_pipeline \
      --start 2025-12-16 \
      --end 2025-12-20 \
      --checkpoint-dir data/checkpoints
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.pipeline.core.checkpoint import CheckpointManager


def list_checkpoints(checkpoint_dir: str, date: str):
    """List available checkpoints for a date."""
    manager = CheckpointManager(checkpoint_dir)
    checkpoints = manager.list_checkpoints("es_pipeline", date)
    
    if not checkpoints:
        print(f"No checkpoints found for {date}")
        return
    
    print(f"\nAvailable checkpoints for {date}:")
    print(f"{'='*80}")
    print(f"{'Idx':<5} {'Stage Name':<30} {'Time':<10} {'Outputs':<10}")
    print(f"{'-'*80}")
    
    for cp in checkpoints:
        idx = cp['stage_idx']
        name = cp['stage_name']
        elapsed = f"{cp['elapsed_time']:.2f}s"
        outputs = len(cp['outputs'])
        print(f"{idx:<5} {name:<30} {elapsed:<10} {outputs:<10}")
    
    print(f"{'='*80}\n")


def clear_checkpoints(checkpoint_dir: str, date: str):
    """Clear checkpoints for a date."""
    manager = CheckpointManager(checkpoint_dir)
    
    response = input(f"Clear all checkpoints for {date}? [y/N]: ")
    if response.lower() == 'y':
        manager.clear_checkpoints("es_pipeline", date)
        print(f"Checkpoints cleared for {date}")
    else:
        print("Cancelled")


def inspect_stage_output(checkpoint_dir: str, date: str, stage_idx: int):
    """Load and inspect a stage's output."""
    import pandas as pd
    
    manager = CheckpointManager(checkpoint_dir)
    ctx = manager.load_checkpoint("es_pipeline", date, stage_idx)
    
    if ctx is None:
        print(f"Checkpoint not found for stage {stage_idx}")
        return
    
    print(f"\n{'='*80}")
    print(f"Stage {stage_idx} Outputs:")
    print(f"{'='*80}\n")
    
    for key, value in ctx.data.items():
        if isinstance(value, pd.DataFrame):
            print(f"{key}: DataFrame")
            print(f"  Shape: {value.shape}")
            print(f"  Columns: {list(value.columns)[:10]}")
            if len(value.columns) > 10:
                print(f"    ... and {len(value.columns)-10} more")
            print(f"  Head:\n{value.head(3)}\n")
        
        elif isinstance(value, pd.Series):
            print(f"{key}: Series")
            print(f"  Length: {len(value)}")
            print(f"  Head:\n{value.head(3)}\n")
        
        elif isinstance(value, list):
            print(f"{key}: list")
            print(f"  Length: {len(value)}")
            if value:
                print(f"  First item type: {type(value[0]).__name__}")
        
        elif isinstance(value, dict):
            print(f"{key}: dict")
            print(f"  Keys: {list(value.keys())[:10]}")
        
        else:
            print(f"{key}: {type(value).__name__}")
    
    print(f"{'='*80}\n")


def build_date_list(
    date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    include_weekends: bool
) -> List[str]:
    """Resolve a single date or a date range into a list of dates."""
    if date and (start_date or end_date):
        raise ValueError("Use --date or --start/--end, not both")

    if date:
        return [date]

    if not start_date or not end_date:
        raise ValueError("Provide --date or both --start and --end")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start > end:
        raise ValueError("--start must be on or before --end")

    dates = []
    current = start
    while current <= end:
        if include_weekends or current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not dates:
        raise ValueError("Date range produced no dates to process")

    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Run ES pipeline (optional checkpointing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date to process (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--start',
        dest='start_date',
        type=str,
        help='Start date for batch run (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        dest='end_date',
        type=str,
        help='End date for batch run (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--include-weekends',
        action='store_true',
        help='Include weekends when running a date range'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for checkpoints (default: data/checkpoints)'
    )
    
    parser.add_argument(
        '--resume-from-stage',
        type=int,
        default=None,
        help='Resume from stage N (0-based, loads stage N-1 checkpoint)'
    )
    
    parser.add_argument(
        '--stop-at-stage',
        type=int,
        default=None,
        help='Stop after stage N (0-based, for debugging)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available checkpoints for date'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear checkpoints for date'
    )
    
    parser.add_argument(
        '--inspect',
        type=int,
        default=None,
        metavar='STAGE_IDX',
        help='Inspect outputs from stage N'
    )
    
    args = parser.parse_args()
    
    # Set default checkpoint dir
    if args.checkpoint_dir is None and not (args.list or args.clear or args.inspect):
        backend_dir = Path(__file__).parent.parent
        args.checkpoint_dir = str(backend_dir / 'data' / 'checkpoints')
    
    # Handle management commands
    if args.list:
        if not args.checkpoint_dir:
            print("Error: --checkpoint-dir required for --list")
            return 1
        if not args.date:
            print("Error: --date required for --list")
            return 1
        list_checkpoints(args.checkpoint_dir, args.date)
        return 0
    
    if args.clear:
        if not args.checkpoint_dir:
            print("Error: --checkpoint-dir required for --clear")
            return 1
        if not args.date:
            print("Error: --date required for --clear")
            return 1
        clear_checkpoints(args.checkpoint_dir, args.date)
        return 0
    
    if args.inspect is not None:
        if not args.checkpoint_dir:
            print("Error: --checkpoint-dir required for --inspect")
            return 1
        if not args.date:
            print("Error: --date required for --inspect")
            return 1
        inspect_stage_output(args.checkpoint_dir, args.date, args.inspect)
        return 0

    try:
        dates = build_date_list(
            args.date,
            args.start_date,
            args.end_date,
            args.include_weekends
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    
    # Run pipeline
    if len(dates) == 1:
        print(f"Running ES pipeline for {dates[0]}")
    else:
        print(f"Running ES pipeline for {len(dates)} dates: {dates[0]} -> {dates[-1]}")
    if args.checkpoint_dir:
        print(f"Checkpoints: {args.checkpoint_dir}")
    if args.resume_from_stage is not None:
        print(f"Resuming from stage_idx: {args.resume_from_stage}")
    if args.stop_at_stage is not None:
        print(f"Stopping at stage_idx: {args.stop_at_stage}")
    print()
    
    pipeline = build_es_pipeline()
    processed = 0

    for idx, run_date in enumerate(dates, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(dates)}] Running pipeline for {run_date}")
        print(f"{'='*60}\n")
        try:
            signals_df = pipeline.run(
                date=run_date,
                checkpoint_dir=args.checkpoint_dir,
                resume_from_stage=args.resume_from_stage,
                stop_at_stage=args.stop_at_stage
            )
        except Exception as e:
            print(f"\nPipeline failed for {run_date}: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print(f"\nPipeline completed successfully for {run_date}")
        print(f"Final signals: {len(signals_df):,} rows")
        processed += 1

    if processed != len(dates):
        print(f"\nProcessed {processed}/{len(dates)} dates")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
