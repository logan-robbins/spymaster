"""
Parallel Bronze backfill wrapper (OOM-safe).

Launches separate processes for each date to avoid pickling issues
and control memory usage.

Usage:
    cd backend
    uv run python scripts/backfill_bronze_parallel.py --all --workers 3
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set


def get_available_dates() -> Set[str]:
    """Get dates that have DBN files."""
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    trades_dir = raw_dir / 'trades'
    dates = set()
    
    for path in trades_dir.glob('*.dbn'):
        # Extract date from filename: glbx-mdp3-20250602.trades.dbn
        parts = path.stem.split('.')
        if parts:
            date_part = parts[0].split('-')[-1]
            if len(date_part) == 8:
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                dates.add(date_str)
    
    return dates


def get_processed_dates() -> Set[str]:
    """Get dates already in Bronze layer."""
    bronze_dir = Path(__file__).parent.parent / 'data' / 'lake' / 'bronze' / 'futures' / 'trades' / 'symbol=ES'
    
    if not bronze_dir.exists():
        return set()
    
    dates = set()
    for date_dir in bronze_dir.glob('date=*'):
        date_str = date_dir.name.replace('date=', '')
        # Check if it has parquet files
        if list(date_dir.rglob('*.parquet')):
            dates.add(date_str)
    
    return dates


def process_date(date: str, force: bool) -> dict:
    """
    Process a single date using subprocess.
    
    Args:
        date: Date to process (YYYY-MM-DD)
        force: Force reprocessing
    
    Returns:
        Result dict with status
    """
    start_time = time.time()
    
    try:
        cmd = [
            'uv', 'run', 'python', '-m',
            'scripts.backfill_bronze_futures',
            '--date', date
        ]
        
        if force:
            cmd.append('--force')
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout per date
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Extract counts from output
            output = result.stdout
            return {
                'date': date,
                'status': 'success',
                'elapsed': elapsed,
                'output': output
            }
        else:
            return {
                'date': date,
                'status': 'error',
                'elapsed': elapsed,
                'error': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            'date': date,
            'status': 'timeout',
            'elapsed': 1800
        }
    except Exception as e:
        return {
            'date': date,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Bronze backfill (OOM-safe)'
    )
    parser.add_argument('--all', action='store_true', help='Process all available dates')
    parser.add_argument('--date', type=str, help='Single date')
    parser.add_argument('--dates', type=str, help='Comma-separated dates')
    parser.add_argument('--workers', type=int, default=3, help='Parallel workers (default: 3, max: 4 recommended)')
    parser.add_argument('--force', action='store_true', help='Reprocess existing dates')
    
    args = parser.parse_args()
    
    if args.workers > 4:
        print(f"⚠️  Warning: {args.workers} workers may cause OOM. Recommended max: 4")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Determine dates to process
    available = get_available_dates()
    
    if args.all:
        dates = sorted(available)
    elif args.date:
        dates = [args.date]
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(',')]
    else:
        print("Provide --all, --date, or --dates")
        return 1
    
    # Filter to available dates
    dates = [d for d in dates if d in available]
    
    if not dates:
        print("No dates to process")
        return 1
    
    # Check what's already done
    processed = get_processed_dates()
    
    if not args.force:
        dates = [d for d in dates if d not in processed]
    
    if not dates:
        print("✅ All dates already processed")
        return 0
    
    print(f"\n{'='*70}")
    print(f"Bronze Backfill (Parallel, OOM-Safe)")
    print(f"{'='*70}")
    print(f"  Total dates: {len(dates)}")
    print(f"  Workers: {args.workers}")
    print(f"  Memory per worker: ~10-15 GB")
    print(f"  Estimated total RAM: ~{args.workers * 15} GB")
    print(f"{'='*70}\n")
    
    # Process in parallel
    start_time = time.time()
    success_count = 0
    error_count = 0
    timeout_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_date = {
            executor.submit(process_date, date, args.force): date
            for date in dates
        }
        
        for future in as_completed(future_to_date):
            result = future.result()
            
            if result['status'] == 'success':
                success_count += 1
                print(f"✅ {result['date']}: {result['elapsed']:.1f}s")
            elif result['status'] == 'timeout':
                timeout_count += 1
                print(f"⏱️  {result['date']}: timeout")
            else:
                error_count += 1
                error_msg = result.get('error', 'unknown')[:100]
                print(f"❌ {result['date']}: {error_msg}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}/{len(dates)}")
    print(f"  Errors: {error_count}")
    print(f"  Timeouts: {timeout_count}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

