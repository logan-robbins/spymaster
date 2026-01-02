"""
Recover ALL missing ES Options data (Trades + NBBO).

This script downloads raw DBN files for dates where we're missing data:
1. June 1 - October 31, 2025: Re-download trades + NBBO (raw files were deleted by mistake)
2. November 1 - December 31, 2025: Download trades + NBBO (both raw and Bronze missing)

Total: 153 trading days (June 1 - Dec 31, 2025)
Schemas: trades, mbp-1 (NBBO)
Output: data/raw/es_options/*.dbn

After this completes, run: uv run python scripts/convert_options_dbn_to_bronze.py --all

Usage:
    cd backend
    uv run python scripts/recover_missing_es_options_data.py --workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
from dotenv import load_dotenv

_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
load_dotenv(_backend_dir / '.env')


def download_single_file(args_tuple):
    """Download a single date/schema."""
    date_str, schema, api_key, output_dir = args_tuple
    
    date_compact = date_str.replace('-', '')
    output_file = output_dir / f"es-opt-{date_compact}.{schema}.dbn"
    
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ {date_str} {schema}: exists ({size_mb:.1f} MB)")
        return {'date': date_str, 'schema': schema, 'status': 'exists', 'size_mb': size_mb}
    
    try:
        print(f"  ⬇️  {date_str} {schema}: downloading...", flush=True)
        start_time = time.time()
        
        client = db.Historical(key=api_key)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        end_date_str = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        
        client.timeseries.get_range(
            dataset='GLBX.MDP3',
            schema=schema,
            symbols=['ES.OPT'],
            start=date_str,
            end=end_date_str,
            stype_in='parent',
            path=str(output_file)
        )
        
        elapsed = time.time() - start_time
        size_mb = output_file.stat().st_size / (1024 * 1024)
        rate = size_mb / elapsed if elapsed > 0 else 0
        
        print(f"  ✅ {date_str} {schema}: {size_mb:.1f} MB in {elapsed:.1f}s ({rate:.1f} MB/s)")
        return {'date': date_str, 'schema': schema, 'status': 'downloaded', 'size_mb': size_mb, 'elapsed': elapsed}
    
    except Exception as e:
        print(f"  ❌ {date_str} {schema}: {e}")
        if output_file.exists():
            output_file.unlink()
        return {'date': date_str, 'schema': schema, 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Recover ALL missing ES Options data (Trades + NBBO)'
    )
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers (default: 8)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    
    args = parser.parse_args()
    
    api_key = os.getenv('DATABENTO_API_KEY')
    if not api_key:
        print("❌ ERROR: DATABENTO_API_KEY not found in .env")
        return 1
    
    output_dir = _backend_dir / 'data' / 'raw' / 'es_options'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all trading days June 1 - Dec 31, 2025
    dates = []
    current = datetime(2025, 6, 1)
    end = datetime(2025, 12, 31)
    
    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    schemas = ['trades', 'mbp-1']
    
    print("="*70)
    print("ES OPTIONS DATA RECOVERY")
    print("="*70)
    print(f"Date range: 2025-06-01 to 2025-12-31")
    print(f"Trading days: {len(dates)}")
    print(f"Schemas: {', '.join(schemas)}")
    print(f"Total downloads: {len(dates) * len(schemas)} files")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    if args.dry_run:
        print("DRY RUN - Would download:")
        for schema in schemas:
            print(f"\n{schema}:")
            for date in dates[:5]:
                print(f"  - {date}")
            print(f"  ... ({len(dates)} total)")
        return 0
    
    # Create download jobs
    jobs = []
    for date in dates:
        for schema in schemas:
            jobs.append((date, schema, api_key, output_dir))
    
    print(f"Starting parallel download of {len(jobs)} files...\n")
    
    start_time = time.time()
    stats = {'downloaded': 0, 'exists': 0, 'errors': 0}
    total_mb = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_single_file, job) for job in jobs]
        
        for future in as_completed(futures):
            result = future.result()
            stats[result['status']] += 1
            if 'size_mb' in result:
                total_mb += result['size_mb']
    
    elapsed = time.time() - start_time
    
    print()
    print("="*70)
    print(f"COMPLETE: {elapsed/60:.1f} minutes")
    print("="*70)
    print(f"  Downloaded: {stats['downloaded']} files ({total_mb:.1f} MB)")
    print(f"  Already existed: {stats['exists']} files")
    print(f"  Errors: {stats['errors']} files")
    print()
    print("Next step:")
    print("  cd backend")
    print("  uv run python scripts/convert_options_dbn_to_bronze.py --all")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

