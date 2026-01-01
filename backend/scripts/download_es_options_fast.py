"""
Fast ES Options downloader - writes directly to DBN (no DataFrame overhead).

Downloads ES options (trades, NBBO, statistics) directly to DBN format,
bypassing slow DataFrame conversion. Then converts to Bronze Parquet in a separate step.

Usage:
    cd backend
    # Download all schemas
    uv run python scripts/download_es_options_fast.py --start 2025-10-29 --end 2025-10-31 --workers 8
    
    # Download only statistics
    uv run python scripts/download_es_options_fast.py --start 2025-06-01 --end 2025-12-31 --workers 8 --schemas statistics
    
    # Download specific schemas
    uv run python scripts/download_es_options_fast.py --start 2025-06-01 --end 2025-12-31 --workers 8 --schemas "mbp-1,statistics"
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import databento as db
from dotenv import load_dotenv

_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
load_dotenv(_backend_dir / '.env')


class FastESOptionsDownloader:
    """Fast ES options downloader using DBN format."""
    
    DATASET = 'GLBX.MDP3'
    SYMBOL = 'ES.OPT'
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY not found")
        
        self.raw_dir = _backend_dir / 'data' / 'raw' / 'es_options'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.create_client = lambda: db.Historical(key=self.api_key)
    
    def download_date_schema(self, date_str: str, schema: str, client: db.Historical) -> bool:
        """Download single date/schema directly to DBN."""
        
        date_compact = date_str.replace('-', '')
        output_file = self.raw_dir / f"es-opt-{date_compact}.{schema}.dbn"
        
        if output_file.exists():
            print(f"  ⏭️  {date_str} {schema}: exists")
            return True
        
        try:
            print(f"  ⬇️  {date_str} {schema}: downloading...", flush=True)
            start_time = time.time()
            
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            end_date_str = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Download directly to DBN file (no DataFrame conversion!)
            client.timeseries.get_range(
                dataset=self.DATASET,
                schema=schema,
                symbols=[self.SYMBOL],
                start=date_str,
                end=end_date_str,
                stype_in='parent',
                path=str(output_file)  # Write directly
            )
            
            elapsed = time.time() - start_time
            size_mb = output_file.stat().st_size / (1024 * 1024)
            
            print(f"  ✅ {date_str} {schema}: {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/elapsed if elapsed > 0 else 0:.1f} MB/s)")
            return True
        
        except Exception as e:
            print(f"  ❌ {date_str} {schema}: {e}")
            if output_file.exists():
                output_file.unlink()
            return False
    
    def download_dates(self, dates: list, schema: str, max_workers: int = 8):
        """Download multiple dates in parallel."""
        print(f"\n{'='*70}")
        print(f"Downloading {schema} for {len(dates)} dates ({max_workers} workers)")
        print(f"{'='*70}")
        
        success = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_date_schema, date, schema, self.create_client()): date
                for date in dates
            }
            
            for future in as_completed(futures):
                if future.result():
                    success += 1
        
        print(f"\n✅ {schema}: {success}/{len(dates)} dates")
        return success


def main():
    parser = argparse.ArgumentParser(description='Fast ES Options download to DBN')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--schemas', type=str, default='trades,mbp-1,statistics',
                        help='Comma-separated schemas to download (default: all)')
    
    args = parser.parse_args()
    
    # Parse schemas
    requested_schemas = [s.strip() for s in args.schemas.split(',')]
    valid_schemas = {'trades', 'mbp-1', 'statistics'}
    schemas_to_download = [s for s in requested_schemas if s in valid_schemas]
    
    if not schemas_to_download:
        print(f"❌ No valid schemas specified. Valid options: {valid_schemas}")
        return 1
    
    # Generate dates
    dates = []
    current = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    
    while current <= end:
        if current.weekday() < 5:
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"Downloading {len(dates)} dates to DBN format...")
    print(f"Schemas: {', '.join(schemas_to_download)}\n")
    
    downloader = FastESOptionsDownloader()
    
    # Download requested schemas
    for schema in schemas_to_download:
        downloader.download_dates(dates, schema, args.workers)
    
    print(f"\n✅ DBN files saved to: {downloader.raw_dir}")
    print(f"\nNext: Convert DBN to Bronze Parquet")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

