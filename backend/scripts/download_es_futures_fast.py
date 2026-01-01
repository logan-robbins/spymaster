"""
Download ES Futures (Trades + MBP-10) using optimized streaming API.

This is a fallback for when batch API is not available. Uses parallel downloads
with daily chunks to maximize throughput.

Usage:
    cd backend
    uv run python scripts/download_es_futures_fast.py \
        --start 2025-06-01 \
        --end 2025-11-01 \
        --workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

import databento as db
from dotenv import load_dotenv

# Load environment
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
load_dotenv(_backend_dir / '.env')


class ESFuturesFastDownloader:
    """Fast ES Futures downloader using streaming API with parallelization."""
    
    DATASET = 'GLBX.MDP3'
    SYMBOLS = ['ES.FUT']  # Use parent symbol for all ES futures contracts
    
    def __init__(self, api_key: Optional[str] = None, data_root: Optional[Path] = None):
        self.api_key = api_key or os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY not found in .env")
        
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = _backend_dir / 'data' / 'raw'
        
        self.trades_dir = self.data_root / 'trades'
        self.mbp10_dir = self.data_root / 'MBP-10'
        
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.mbp10_dir.mkdir(parents=True, exist_ok=True)
        
        # Each worker gets its own client (thread-safe)
        self.create_client = lambda: db.Historical(key=self.api_key)
    
    def get_existing_dates(self, schema: str) -> Set[str]:
        """Get dates already downloaded."""
        schema_dir = self.trades_dir if schema == 'trades' else self.mbp10_dir
        
        existing = set()
        for path in schema_dir.glob('*.dbn*'):
            parts = path.stem.split('.')
            if parts:
                date_part = parts[0].split('-')[-1]
                if len(date_part) == 8:
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    existing.add(date_str)
        
        return existing
    
    def generate_date_range(self, start: str, end: str) -> List[str]:
        """Generate list of trading days (exclude weekends)."""
        dates = []
        current = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        
        while current <= end_dt:
            if current.weekday() < 5:  # Mon-Fri
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    
    def download_single_date(
        self,
        date_str: str,
        schema: str,
        client: db.Historical
    ) -> bool:
        """Download single date for a schema."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        end_date_str = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Output filename
        date_compact = date_str.replace('-', '')
        output_dir = self.trades_dir if schema == 'trades' else self.mbp10_dir
        output_file = output_dir / f"glbx-mdp3-{date_compact}.{schema}.dbn"
        
        if output_file.exists():
            print(f"  ⏭️  {date_str} {schema}: exists")
            return True
        
        try:
            print(f"  ⬇️  {date_str} {schema}: downloading...")
            start_time = time.time()
            
            # Download data to file directly
            # databento 0.68.1 doesn't have to_dbn(), write raw bytes
            client.timeseries.get_range(
                dataset=self.DATASET,
                schema=schema,
                symbols=self.SYMBOLS,
                start=date_str,
                end=end_date_str,
                stype_in='parent',
                path=str(output_file)  # Write directly to path
            )
            
            elapsed = time.time() - start_time
            size_mb = output_file.stat().st_size / (1024 * 1024)
            rate_mbps = size_mb / elapsed if elapsed > 0 else 0
            
            print(f"  ✅ {date_str} {schema}: {size_mb:.1f} MB in {elapsed:.1f}s ({rate_mbps:.1f} MB/s)")
            return True
        
        except Exception as e:
            print(f"  ❌ {date_str} {schema}: {e}")
            if output_file.exists():
                output_file.unlink()
            return False
    
    def download_schema_parallel(
        self,
        dates: List[str],
        schema: str,
        max_workers: int = 4
    ) -> int:
        """Download all dates for a schema in parallel."""
        print(f"\n{'='*70}")
        print(f"Downloading {schema} for {len(dates)} dates ({max_workers} workers)")
        print(f"{'='*70}")
        
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Each worker gets its own client
            future_to_date = {}
            for date in dates:
                client = self.create_client()
                future = executor.submit(self.download_single_date, date, schema, client)
                future_to_date[future] = date
            
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  ❌ {date} {schema}: Unexpected error - {e}")
        
        print(f"\n✅ {schema}: Downloaded {success_count}/{len(dates)} dates")
        return success_count
    
    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        skip_existing: bool = True,
        max_workers: int = 4
    ):
        """Download date range for both schemas."""
        all_dates = self.generate_date_range(start_date, end_date)
        
        existing_trades = self.get_existing_dates('trades')
        existing_mbp10 = self.get_existing_dates('mbp-10')
        
        trades_dates = [d for d in all_dates if d not in existing_trades] if skip_existing else all_dates
        mbp10_dates = [d for d in all_dates if d not in existing_mbp10] if skip_existing else all_dates
        
        print(f"\n{'='*70}")
        print(f"ES Futures Download Summary")
        print(f"{'='*70}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Total trading days: {len(all_dates)}")
        print(f"\nTrades:")
        print(f"  Already have: {len(existing_trades)}")
        print(f"  Need: {len(trades_dates)}")
        print(f"\nMBP-10:")
        print(f"  Already have: {len(existing_mbp10)}")
        print(f"  Need: {len(mbp10_dates)}")
        print(f"\nParallel workers: {max_workers}")
        print(f"{'='*70}")
        
        if not trades_dates and not mbp10_dates:
            print(f"\n✅ All data already downloaded!")
            return
        
        start_time = time.time()
        
        # Download trades
        if trades_dates:
            self.download_schema_parallel(trades_dates, 'trades', max_workers)
        
        # Download MBP-10
        if mbp10_dates:
            self.download_schema_parallel(mbp10_dates, 'mbp-10', max_workers)
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {elapsed_min:.1f} minutes total")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Fast ES Futures download (optimized streaming)'
    )
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers (default: 4, max: 8)')
    parser.add_argument('--no-skip', action='store_true', help='Download even if exists')
    
    args = parser.parse_args()
    
    if args.workers > 8:
        print("⚠️  Warning: More than 8 workers may hit API rate limits")
        args.workers = 8
    
    try:
        downloader = ESFuturesFastDownloader()
        downloader.download_date_range(
            start_date=args.start,
            end_date=args.end,
            skip_existing=not args.no_skip,
            max_workers=args.workers
        )
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

