"""
Run ES pipelines with optional checkpointing support.

Supports both Bronze→Silver and Silver→Gold pipelines.

Usage:
    # Run Bronze→Silver (feature engineering)
    uv run python -m scripts.run_pipeline \
      --pipeline bronze_to_silver \
      --date 2025-12-16 \
      --checkpoint-dir data/checkpoints \
      --write-outputs
    
    # Run Silver→Gold (episode construction)  
    uv run python -m scripts.run_pipeline \
      --pipeline silver_to_gold \
      --date 2025-12-16 \
      --write-outputs
    
    # Run date range (weekdays only)
    uv run python -m scripts.run_pipeline \
      --pipeline bronze_to_silver \
      --start 2025-12-16 \
      --end 2025-12-20 \
      --workers 4 \
      --write-outputs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from src.pipeline.pipelines.registry import get_pipeline, list_available_pipelines
from src.pipeline.core.checkpoint import CheckpointManager


def build_date_list(
    single_date: str = None,
    start_date: str = None,
    end_date: str = None,
    include_weekends: bool = False
) -> List[str]:
    """Build list of dates to process."""
    if single_date:
        return [single_date]
    
    if not (start_date and end_date):
        raise ValueError("Must provide either --date or --start/--end")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    curr = start
    while curr <= end:
        if include_weekends or curr.weekday() < 5:
            dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    
    return dates


def run_single_date(
    pipeline_name: str,
    date: str,
    checkpoint_dir: str,
    canonical_version: str,
    data_root: str,
    write_outputs: bool,
    overwrite_partitions: bool,
    resume_from_stage: int = None,
    stop_at_stage: int = None
) -> dict:
    """Run pipeline for a single date (worker function for parallel execution)."""
    try:
        pipeline = get_pipeline(pipeline_name)
        start_t = time.time()
        
        result_df = pipeline.run(
            date=date,
            checkpoint_dir=checkpoint_dir,
            canonical_version=canonical_version,
            data_root=data_root,
            write_outputs=write_outputs,
            overwrite_partitions=overwrite_partitions,
            resume_from_stage=resume_from_stage,
            stop_at_stage=stop_at_stage,
            log_level=40  # ERROR only to reduce noise in parallel runs
        )
        
        elapsed = time.time() - start_t
        return {
            "date": date,
            "success": True,
            "rows": len(result_df) if result_df is not None else 0,
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "date": date,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run ES Pipeline")
    parser.add_argument("--pipeline", default="bronze_to_silver", 
                       help="Pipeline name (bronze_to_silver, silver_to_gold, pentaview)")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--canonical-version", default="4.5.0", help="Canonical version")
    parser.add_argument("--data-root", default=None, help="Data root override")
    parser.add_argument("--write-outputs", action="store_true", help="Write to Silver/Gold layers")
    parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite existing partitions")
    parser.add_argument("--resume-from-stage", type=int, default=None, help="Resume from stage_idx")
    parser.add_argument("--stop-at-stage", type=int, default=None, help="Stop after stage_idx")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (enables parallel mode)")
    parser.add_argument("--include-weekends", action="store_true", help="Include weekends in date range")
    parser.add_argument("--list-pipelines", action="store_true", help="List available pipelines")
    
    args = parser.parse_args()
    
    if args.list_pipelines:
        pipelines = list_available_pipelines()
        print("Available pipelines:")
        for p in pipelines:
            print(f"  - {p}")
        return 0
    
    # Build date list
    try:
        dates = build_date_list(args.date, args.start, args.end, args.include_weekends)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Parallel mode
    if args.workers and len(dates) > 1:
        print(f"Running {args.pipeline} pipeline for {len(dates)} dates with {args.workers} workers...")
        
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_date,
                    args.pipeline,
                    d,
                    args.checkpoint_dir,
                    args.canonical_version,
                    args.data_root,
                    args.write_outputs,
                    not args.no_overwrite,
                    args.resume_from_stage,
                    args.stop_at_stage
                ): d for d in dates
            }
            
            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    print(f"✅ {res['date']}: {res['rows']} rows ({res['elapsed']:.1f}s)")
                else:
                    print(f"❌ {res['date']}: FAILED - {res['error']}")
                results.append(res)
        
        failed = sum(1 for r in results if not r["success"])
        return 1 if failed > 0 else 0
    
    # Sequential mode
    pipeline = get_pipeline(args.pipeline)
    
    if len(dates) == 1:
        print(f"Running {args.pipeline} pipeline for {dates[0]}")
    else:
        print(f"Running {args.pipeline} pipeline for {len(dates)} dates: {dates[0]} -> {dates[-1]}")
    
    if args.checkpoint_dir:
        print(f"Checkpoints: {args.checkpoint_dir}")
    if args.resume_from_stage is not None:
        print(f"Resuming from stage_idx: {args.resume_from_stage}")
    if args.stop_at_stage is not None:
        print(f"Stopping at stage_idx: {args.stop_at_stage}")
    print()
    
    for idx, run_date in enumerate(dates, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(dates)}] Running {args.pipeline} for {run_date}")
        print(f"{'='*60}\n")
        
        try:
            result_df = pipeline.run(
                date=run_date,
                checkpoint_dir=args.checkpoint_dir,
                resume_from_stage=args.resume_from_stage,
                stop_at_stage=args.stop_at_stage,
                canonical_version=args.canonical_version,
                data_root=args.data_root,
                write_outputs=args.write_outputs,
                overwrite_partitions=not args.no_overwrite,
            )
            
            print(f"\n✅ Pipeline completed for {run_date}")
            print(f"Result: {len(result_df):,} rows")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed for {run_date}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
