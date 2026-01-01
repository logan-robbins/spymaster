
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.common.config import CONFIG

def run_single_day(
    date: str,
    checkpoint_dir: str,
    canonical_version: str,
    data_root: str,
    write_outputs: bool,
    overwrite_partitions: bool
) -> dict:
    """Run pipeline for a single day (worker function)."""
    try:
        pipeline = build_es_pipeline()
        start_t = time.time()
        signals = pipeline.run(
            date=date,
            checkpoint_dir=checkpoint_dir,
            canonical_version=canonical_version,
            data_root=data_root,
            write_outputs=write_outputs,
            overwrite_partitions=overwrite_partitions,
            log_level=40 # ERROR only to reduce noise
        )
        elapsed = time.time() - start_t
        return {
            "date": date,
            "success": True,
            "rows": len(signals),
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "date": date,
            "success": False,
            "error": str(e)
        }

def build_date_list(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    curr = s
    while curr <= e:
        if curr.weekday() < 5:
            dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return dates

def main():
    parser = argparse.ArgumentParser(description="Run ES Pipeline in Parallel")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--canonical-version", default="3.1.0")
    
    args = parser.parse_args()
    
    dates = build_date_list(args.start, args.end)
    print(f"Running parallel pipeline for {len(dates)} dates with {args.workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_day, 
                d, 
                args.checkpoint_dir,
                args.canonical_version,
                None,
                True, # write_outputs
                True  # overwrite
            ): d for d in dates
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                print(f"✅ {res['date']}: {res['rows']} signals ({res['elapsed']:.1f}s)")
            else:
                print(f"❌ {res['date']}: FAILED - {res['error']}")
            results.append(res)

if __name__ == "__main__":
    main()
