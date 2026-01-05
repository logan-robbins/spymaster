from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .config import load_config
from .pipeline import build_pipeline
from .utils import expand_date_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the data pipeline for a specific product_type, symbol, and dt partition(s)"
    )
    p.add_argument(
        "--product-type",
        required=True,
        choices=["future", "future_option", "equity", "equity_option"],
        help="Product type",
    )
    p.add_argument(
        "--layer",
        required=True,
        choices=["bronze", "silver", "gold", "all"],
        help="Pipeline layer to run (bronze, silver, gold, or all)",
    )
    p.add_argument(
        "--symbol",
        required=True,
        help="Symbol to process (e.g., ES for Bronze, ESM6 for Silver/Gold)",
    )
    p.add_argument(
        "--dt",
        help="Single partition date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--dates",
        help="Comma-separated (2025-06-01,2025-06-02) or range (2025-06-01:2025-06-10)",
    )
    p.add_argument(
        "--start-date",
        help="Start date for range (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        help="End date for range (YYYY-MM-DD)",
    )
    p.add_argument(
        "--config",
        default="src/data_eng/config/datasets.yaml",
        help="Path to dataset config YAML (default: src/data_eng/config/datasets.yaml)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    return p.parse_args()


def run_single_date(product_type: str, layer: str, symbol: str, dt: str, config_path: Path) -> dict:
    """Run pipeline for single date (for parallel execution)."""
    try:
        repo_root = Path.cwd()
        cfg = load_config(repo_root=repo_root, config_path=config_path)
        
        stages = build_pipeline(product_type, layer)
        
        for stage in stages:
            stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt)
        
        return {'dt': dt, 'status': 'success'}
    except Exception as e:
        return {'dt': dt, 'status': 'error', 'error': str(e)}


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    config_path = repo_root / args.config
    
    if args.dt:
        dates = [args.dt]
    else:
        dates = expand_date_range(
            dates=args.dates,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    if not dates:
        raise ValueError("Must provide --dt, --dates, or --start-date/--end-date")
    
    print(f"Product Type: {args.product_type}")
    print(f"Layer:        {args.layer}")
    print(f"Symbol:       {args.symbol}")
    print(f"Dates:        {len(dates)}")
    print(f"Workers:      {args.workers}")
    print()
    
    if args.workers > 1 and len(dates) > 1:
        print(f"Processing {len(dates)} dates with {args.workers} workers\n")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_single_date, args.product_type, args.layer, args.symbol, dt, config_path): dt
                for dt in dates
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    print(f"✅ {result['dt']}")
                else:
                    print(f"❌ {result['dt']}: {result['error']}")
    else:
        cfg = load_config(repo_root=repo_root, config_path=config_path)
        
        for dt in dates:
            print(f"Processing {dt}...")
            
            stages = build_pipeline(args.product_type, args.layer)
            
            for stage in stages:
                stage.run(cfg=cfg, repo_root=repo_root, symbol=args.symbol, dt=dt)
            
            print(f"✅ {dt}\n")
    
    print(f"DONE")
    print(f"Lake Root: {repo_root / 'lake'}")


if __name__ == "__main__":
    main()
