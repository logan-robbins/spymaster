"""
Download SPY stock trades from Polygon API and write to Bronze.

This script downloads SPY equity trades (not options) to build the SPY spot series
required for v1 feature engineering (kinematics, ATR, volatility features).

Usage:
    uv run python scripts/download_spy_trades.py --date 2025-12-16
    uv run python scripts/download_spy_trades.py --start 2025-11-02 --end 2025-12-31
"""

import argparse
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

# Load environment
_backend_dir = Path(__file__).parent.parent
load_dotenv(_backend_dir / '.env')


class SPYTradesDownloader:
    """Download SPY stock trades from Polygon REST API."""
    
    BASE_URL = "https://api.polygon.io/v3/trades"
    SYMBOL = "SPY"
    
    # Rate limiting (conservative)
    REQUESTS_PER_MINUTE = 5
    REQUEST_DELAY_S = 60.0 / REQUESTS_PER_MINUTE
    
    def __init__(self, api_key: Optional[str] = None, data_root: Optional[str] = None):
        """
        Initialize downloader.
        
        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
            data_root: Root directory for data lake
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = _backend_dir / 'data'
        
        self.bronze_root = self.data_root / 'bronze'
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY_S:
            time.sleep(self.REQUEST_DELAY_S - elapsed)
        self._last_request_time = time.time()
    
    def download_date(self, date_str: str, force: bool = False) -> int:
        """
        Download SPY trades for a single date.
        
        Args:
            date_str: Date (YYYY-MM-DD)
            force: Re-download if exists
        
        Returns:
            Number of trades downloaded
        """
        # Parse date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() >= 5:
            print(f"{date_str}: Weekend, skipping")
            return 0
        
        # Check if exists
        output_dir = self.bronze_root / 'stocks' / 'trades' / 'symbol=SPY' / f'date={date_str}'
        if output_dir.exists() and not force:
            existing_files = list(output_dir.glob('*.parquet'))
            if existing_files:
                print(f"{date_str}: Already exists (use --force to re-download)")
                return -1
        
        print(f"\n{date_str}: Downloading SPY trades...")
        
        # Time range: 04:00-20:00 ET (includes pre/post market for PM levels)
        # Convert to UTC timestamps
        start_dt = datetime(
            date_obj.year, date_obj.month, date_obj.day,
            9, 0, tzinfo=timezone.utc  # 04:00 ET ≈ 09:00 UTC (approximate, ignoring DST)
        )
        end_dt = datetime(
            date_obj.year, date_obj.month, date_obj.day,
            21, 0, tzinfo=timezone.utc  # 20:00 ET ≈ 01:00 UTC next day
        )
        
        start_ns = int(start_dt.timestamp() * 1e9)
        end_ns = int(end_dt.timestamp() * 1e9)
        
        # Fetch trades
        all_trades = []
        next_url = None
        page = 0
        
        while True:
            page += 1
            self._rate_limit()
            
            if next_url:
                url = next_url
                params = {}
            else:
                url = f"{self.BASE_URL}/{self.SYMBOL}"
                params = {
                    'timestamp.gte': start_ns,
                    'timestamp.lt': end_ns,
                    'order': 'asc',
                    'limit': 50000,
                    'sort': 'timestamp'
                }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                if not results:
                    break
                
                all_trades.extend(results)
                print(f"  Page {page}: {len(results)} trades (total: {len(all_trades)})")
                
                next_url = data.get('next_url')
                if not next_url:
                    break
                
            except Exception as e:
                print(f"  Error fetching page {page}: {e}")
                break
        
        if not all_trades:
            print(f"  No trades found")
            return 0
        
        # Transform to Bronze schema
        df = self._transform_to_schema(all_trades, date_str)
        
        # Write to Parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Partition by hour for consistency with other Bronze data
        df['hour'] = pd.to_datetime(df['ts_event_ns'], unit='ns', utc=True).dt.hour
        
        for hour, hour_df in df.groupby('hour'):
            hour_dir = output_dir / f'hour={hour:02d}'
            hour_dir.mkdir(parents=True, exist_ok=True)
            output_path = hour_dir / f'trades_{date_str}_h{hour:02d}.parquet'
            
            hour_df_out = hour_df.drop(columns=['hour'])
            table = pa.Table.from_pandas(hour_df_out, preserve_index=False)
            pq.write_table(table, output_path, compression='zstd')
            
            print(f"  Wrote {len(hour_df_out):,} trades to hour={hour:02d}/")
        
        print(f"  Total: {len(df):,} SPY trades")
        return len(df)
    
    def _transform_to_schema(self, trades: List[Dict[str, Any]], date_str: str) -> pd.DataFrame:
        """Transform Polygon API response to Bronze schema."""
        records = []
        
        for trade in trades:
            # Polygon timestamps are in nanoseconds
            ts_event_ns = trade.get('sip_timestamp', 0)
            if ts_event_ns == 0:
                ts_event_ns = trade.get('participant_timestamp', 0)
            
            records.append({
                'ts_event_ns': int(ts_event_ns),
                'ts_recv_ns': int(ts_event_ns),  # Use same for historical
                'source': 'POLYGON_API',
                'symbol': 'SPY',
                'price': float(trade.get('price', 0)),
                'size': int(trade.get('size', 0)),
                'exchange': trade.get('exchange', None),
                'conditions': str(trade.get('conditions', [])) if trade.get('conditions') else None,
                'seq': trade.get('sequence_number', None)
            })
        
        df = pd.DataFrame(records)
        
        # Sort by event time
        df = df.sort_values('ts_event_ns').reset_index(drop=True)
        
        # Filter outliers
        df = df[(df['price'] > 300) & (df['price'] < 800)]
        df = df[df['size'] > 0]
        
        return df
    
    def download_range(self, start_date: str, end_date: str, force: bool = False) -> Dict[str, int]:
        """
        Download SPY trades for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Re-download if exists
        
        Returns:
            Dict mapping date -> trade count
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        results = {}
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            count = self.download_date(date_str, force=force)
            results[date_str] = count
            current += timedelta(days=1)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Download SPY stock trades from Polygon')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='Re-download if exists')
    
    args = parser.parse_args()
    
    try:
        downloader = SPYTradesDownloader()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    if args.date:
        count = downloader.download_date(args.date, force=args.force)
        if count > 0:
            print(f"\n✅ Downloaded {count:,} SPY trades for {args.date}")
        elif count == -1:
            print(f"\n⏭️  Skipped {args.date} (already exists)")
        return 0
    
    if args.start and args.end:
        results = downloader.download_range(args.start, args.end, force=args.force)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total = 0
        for date_str, count in results.items():
            if count == -1:
                print(f"  {date_str}: skipped (exists)")
            elif count == 0:
                print(f"  {date_str}: no data / weekend")
            else:
                print(f"  {date_str}: {count:,} trades")
                total += count
        print(f"\nTotal: {total:,} SPY trades downloaded")
        return 0
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

