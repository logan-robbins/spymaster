"""
Download ES options data from Databento.

Downloads both trades and NBBO (MBP-1) for ES futures options.

ES options advantages:
- PERFECT alignment: ES options on ES futures (same underlying!)
- Cash-settled, same as ES futures
- Same contract specs, same venue (CME)
- Zero conversion/basis spread issues
- 0DTE available (EW options)

Usage:
    uv run python scripts/download_es_options.py --start 2025-11-02 --end 2025-12-28
    uv run python scripts/download_es_options.py --start 2025-11-02 --end 2025-12-28 --workers 8
    uv run python scripts/download_es_options.py --date 2025-12-16
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import databento as db
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Load environment
_backend_dir = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(_backend_dir))
load_dotenv(_backend_dir / '.env')


class ESOptionsDownloader:
    """Download ES options trades + NBBO from Databento."""
    
    # ES options trade on CME (not OPRA)
    DATASET = 'GLBX.MDP3'  # CME Globex dataset
    SYMBOL = 'ES.OPT'  # E-mini S&P 500 options (Databento parent symbology)
    
    def __init__(self, api_key: Optional[str] = None, data_root: Optional[str] = None):
        """
        Initialize downloader.
        
        Args:
            api_key: Databento API key (defaults to DATABENTO_API_KEY env var)
            data_root: Root directory for data lake
        """
        self.api_key = api_key or os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DATABENTO_API_KEY not found. Set it in .env or pass to constructor."
            )
        
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = _backend_dir / 'data' / 'lake'
        
        self.bronze_root = self.data_root / 'bronze'
        self.raw_root = _backend_dir / 'data' / 'raw' / 'databento' / 'spx_options'
        self.raw_root.mkdir(parents=True, exist_ok=True)
        
        self.client = db.Historical(key=self.api_key)
    
    def download_date(self, date_str: str, force: bool = False) -> dict:
        """
        Download SPX options trades + NBBO for a single date.
        
        Args:
            date_str: Date (YYYY-MM-DD)
            force: Re-download if exists
        
        Returns:
            Dict with trades_count and nbbo_count
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() >= 5:
            print(f"{date_str}: Weekend, skipping")
            return {'trades': 0, 'nbbo': 0}
        
        # Check if exists
        trades_output_dir = self.bronze_root / 'options' / 'trades' / 'underlying=ES' / f'date={date_str}'
        nbbo_output_dir = self.bronze_root / 'options' / 'nbbo' / 'underlying=ES' / f'date={date_str}'
        
        if not force:
            trades_exists = trades_output_dir.exists() and list(trades_output_dir.rglob('*.parquet'))
            nbbo_exists = nbbo_output_dir.exists() and list(nbbo_output_dir.rglob('*.parquet'))
            if trades_exists and nbbo_exists:
                print(f"{date_str}: Already exists (use --force to re-download)")
                return {'trades': -1, 'nbbo': -1}
        
        print(f"\n{'='*60}")
        print(f"{date_str}: Downloading ES options")
        print(f"{'='*60}")
        
        # Download trades
        trades_count = self._download_schema(date_str, 'trades', trades_output_dir)
        
        # Download NBBO (MBP-1)
        nbbo_count = self._download_schema(date_str, 'mbp-1', nbbo_output_dir)

        # Download Statistics (Open Interest)
        stats_output_dir = self.bronze_root / 'options' / 'statistics' / 'underlying=ES' / f'date={date_str}'
        stats_count = self._download_schema(date_str, 'statistics', stats_output_dir)
        
        return {'trades': trades_count, 'nbbo': nbbo_count, 'stats': stats_count}
    
        return {'trades': trades_count, 'nbbo': nbbo_count, 'stats': stats_count}
    
    def _download_schema(self, date_str: str, schema: str, output_dir: Path) -> int:
        """Download specific schema and write to Bronze."""
        print(f"\n  Downloading {schema}...")

        # For large schemas (mbp-1), download in hourly chunks to avoid >5GB streaming requests
        # CME Globex hours: 17:00 CT previous day to 16:00 CT (Sunday-Friday)
        # In UTC: ~23:00 to ~21:00 next day
        use_chunked = schema == 'mbp-1'

        if use_chunked:
            return self._download_schema_chunked(date_str, schema, output_dir)
        else:
            return self._download_schema_single(date_str, schema, output_dir)

    def _download_schema_single(self, date_str: str, schema: str, output_dir: Path) -> int:
        """Download schema in single request (for smaller datasets like trades)."""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            end_date_str = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

            data = self.client.timeseries.get_range(
                dataset=self.DATASET,
                schema=schema,
                symbols=[self.SYMBOL],
                start=date_str,
                end=end_date_str,
                stype_in='parent'
            )

            # Note: to_df() uses ts_recv as the index, so reset_index() to make it a column
            df = data.to_df().reset_index()

            if df.empty:
                print(f"    No {schema} data found")
                return 0

            print(f"    Retrieved {len(df):,} {schema} records")
            return self._transform_and_write(df, date_str, schema, output_dir)

        except Exception as e:
            print(f"    Error downloading {schema}: {e}")
            return 0

    def _download_schema_chunked(self, date_str: str, schema: str, output_dir: Path) -> int:
        """Download schema in hourly chunks to avoid >5GB streaming limit."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Download in 4-hour chunks (6 chunks per day)
        # This keeps each request well under the 5GB limit
        chunks = [
            (0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)
        ]

        all_dfs = []

        for start_hour, end_hour in chunks:
            chunk_start = date_obj + timedelta(hours=start_hour)
            chunk_end = date_obj + timedelta(hours=end_hour)

            try:
                print(f"    Chunk {start_hour:02d}:00-{end_hour:02d}:00 UTC...", end=" ", flush=True)

                data = self.client.timeseries.get_range(
                    dataset=self.DATASET,
                    schema=schema,
                    symbols=[self.SYMBOL],
                    start=chunk_start.strftime('%Y-%m-%dT%H:%M:%S'),
                    end=chunk_end.strftime('%Y-%m-%dT%H:%M:%S'),
                    stype_in='parent'
                )

                df = data.to_df().reset_index()

                if df.empty:
                    print("no data")
                    continue

                print(f"{len(df):,} records")
                all_dfs.append(df)

            except Exception as e:
                print(f"error: {e}")
                continue

        if not all_dfs:
            print(f"    No {schema} data found")
            return 0

        # Combine all chunks
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"    Total retrieved: {len(df):,} {schema} records")

        return self._transform_and_write(df, date_str, schema, output_dir)

    def _transform_and_write(self, df: pd.DataFrame, date_str: str, schema: str, output_dir: Path) -> int:
        """Transform DataFrame and write to partitioned Parquet."""
        # Transform to Bronze schema
        if schema == 'trades':
            df_bronze = self._transform_trades(df, date_str)
        elif schema == 'mbp-1':
            df_bronze = self._transform_nbbo(df, date_str)
        elif schema == 'statistics':
            df_bronze = self._transform_statistics(df, date_str)
        else:
            raise ValueError(f"Unknown schema: {schema}")

        # Write to Parquet partitioned by hour
        df_bronze['hour'] = pd.to_datetime(df_bronze['ts_event_ns'], unit='ns', utc=True).dt.hour

        for hour, hour_df in df_bronze.groupby('hour'):
            hour_dir = output_dir / f'hour={hour:02d}'
            hour_dir.mkdir(parents=True, exist_ok=True)

            output_path = hour_dir / f'{schema}_{date_str}_h{hour:02d}.parquet'
            hour_df_out = hour_df.drop(columns=['hour'])

            table = pa.Table.from_pandas(hour_df_out, preserve_index=False)
            pq.write_table(table, output_path, compression='zstd')

            print(f"      Wrote {len(hour_df_out):,} records to hour={hour:02d}/")

        print(f"    Total: {len(df_bronze):,} {schema} records")
        return len(df_bronze)
    
    def _transform_trades(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Transform Databento trades to Bronze schema."""
        # Databento columns: ts_event, ts_recv, symbol, price, size, side, ...
        
        # Parse ES option symbol
        # Actual format from Databento CME: "ESZ5 P6800" or "ESZ5 C7275"
        # Format: {FUTURES_CONTRACT} {C|P}{STRIKE}
        
        df['option_symbol'] = df['symbol'].astype(str)
        
        # Filter out non-ES option symbols (spreads, etc.)
        # Valid ES options: Start with ES followed by month code and year
        es_mask = df['option_symbol'].str.match(r'^ES[FGHJKMNQUVXZ]\d\s+[CP]\d+$')
        df = df[es_mask].copy()
        
        if df.empty:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'ts_event_ns', 'ts_recv_ns', 'source', 'underlying', 'option_symbol',
                'exp_date', 'strike', 'right', 'price', 'size', 'opt_bid', 'opt_ask',
                'seq', 'aggressor'
            ])
        
        # Parse components from "ESZ5 P6800" format
        # Split on space: ["ESZ5", "P6800"]
        split_parts = df['option_symbol'].str.split(' ', expand=True)
        df['futures_contract'] = split_parts[0]  # ESZ5
        df['right_strike'] = split_parts[1]      # P6800
        
        # CRITICAL: Filter to ACTIVE futures contracts (supports Rollover overlap)
        # Instead of generic "Front Month", get all active liquid contracts
        from src.common.utils.es_contract_calendar import get_active_contracts
        
        active_contracts = get_active_contracts(date_str)
        
        print(f"    Active contracts: {active_contracts}")
        print(f"    Before filter: {len(df):,} records across all contracts")
        
        # Show distribution before filtering
        contract_counts = df['futures_contract'].value_counts()
        print(f"    Contract distribution:")
        for contract, count in contract_counts.items():
            marker = "✓" if contract in active_contracts else "✗"
            print(f"      {marker} {contract}: {count:,}")
        
        # Filter to options on active contracts only
        df = df[df['futures_contract'].isin(active_contracts)].copy()
        
        if df.empty:
            print(f"    WARNING: No options on {active_contracts} found")
            return pd.DataFrame(columns=[
                'ts_event_ns', 'ts_recv_ns', 'source', 'underlying', 'option_symbol',
                'exp_date', 'strike', 'right', 'price', 'size', 'opt_bid', 'opt_ask',
                'seq', 'aggressor'
            ])
        
        print(f"    After active contracts filter: {len(df):,} records")
        
        # Extract right (C/P)
        df['right'] = df['right_strike'].str[0]  # First char: C or P
        
        # Extract strike (rest of string after C/P)
        df['strike'] = df['right_strike'].str[1:].astype(float)
        
        # Expiration date: For 0DTE, assume session date
        # For longer-dated, would need to parse from futures contract month code
        # Since we're focusing on 0DTE, use session date
        df['exp_date'] = date_str
        
        # Map Databento aggressor to our enum
        # Databento side: 'A' = ask (sell aggressor), 'B' = bid (buy aggressor), 'N' = none
        aggressor_map = {'A': 2, 'B': 1, 'N': 0}  # SELL=2, BUY=1, MID=0
        df['aggressor'] = df.get('side', 'N').map(aggressor_map).fillna(0).astype(int)
        
        return pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['option_symbol'],
            'exp_date': df['exp_date'],
            'strike': df['strike'],
            'right': df['right'],
            'price': df['price'].astype('float64'),
            'size': df['size'].astype('int64'),
            'opt_bid': None,  # Not in trades schema
            'opt_ask': None,
            'seq': df.get('sequence', 0).astype('int64'),
            'aggressor': df['aggressor']
        })
    
    def _transform_nbbo(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Transform Databento MBP-1 (NBBO) to Bronze schema."""
        # Databento MBP-1 columns: ts_event, ts_recv, symbol, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00, ...
        
        # Parse ES option symbol (same format as trades: "ESZ5 P6800")
        df['option_symbol'] = df['symbol'].astype(str)
        
        # Filter to valid ES options only
        es_mask = df['option_symbol'].str.match(r'^ES[FGHJKMNQUVXZ]\d\s+[CP]\d+$')
        df = df[es_mask].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                'ts_event_ns', 'ts_recv_ns', 'source', 'underlying', 'option_symbol',
                'exp_date', 'strike', 'right', 'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'seq'
            ])
        
        # Parse symbol: "ESZ5 P6800"
        split_parts = df['option_symbol'].str.split(' ', expand=True)
        df['futures_contract'] = split_parts[0]
        df['right_strike'] = split_parts[1]
        
        # CRITICAL: Filter to ACTIVE contracts
        from src.common.utils.es_contract_calendar import get_active_contracts
        
        active_contracts = get_active_contracts(date_str)
        
        # Filter to options on active contracts
        df = df[df['futures_contract'].isin(active_contracts)].copy()
        
        if df.empty:
            print(f"    WARNING: No NBBO on {active_contracts}")
            return pd.DataFrame(columns=[
                'ts_event_ns', 'ts_recv_ns', 'source', 'underlying', 'option_symbol',
                'exp_date', 'strike', 'right', 'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'seq'
            ])
        
        df['right'] = df['right_strike'].str[0]
        df['strike'] = df['right_strike'].str[1:].astype(float)
        df['exp_date'] = date_str  # 0DTE assumption
        
        return pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['option_symbol'],
            'exp_date': df['exp_date'],
            'strike': df['strike'],
            'right': df['right'],
            'bid_px': df.get('bid_px_00', 0).astype('float64'),
            'ask_px': df.get('ask_px_00', 0).astype('float64'),
            'bid_sz': df.get('bid_sz_00', 0).astype('int64'),
            'ask_sz': df.get('ask_sz_00', 0).astype('int64'),
            'seq': df.get('sequence', 0).astype('int64')
        })

    def _transform_statistics(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Transform Databento Statistics (Open Interest) to Bronze schema."""
        # Databento stats columns: ts_event, ts_recv, symbol, stat_type, quantity, ...
        # stat_type 3 = Open Interest (Tag 269=C in FIX, usually mapped to 3 or specific code in databento)
        # Check databento docs: stat_type 3 is Open Interest.
        
        # Filter for Open Interest only (stat_type=3)
        # Note: Depending on databento version, it might be an enum or int.
        # Assuming int 3 for now, or check column context.
        df = df[df['stat_type'] == 3].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                'ts_event_ns', 'ts_recv_ns', 'source', 'underlying', 'option_symbol',
                'exp_date', 'strike', 'right', 'open_interest'
            ])

        # Parse ES option symbol
        df['option_symbol'] = df['symbol'].astype(str)
        
        # Filter to valid ES options only
        es_mask = df['option_symbol'].str.match(r'^ES[FGHJKMNQUVXZ]\d\s+[CP]\d+$')
        df = df[es_mask].copy()

        # Parse symbol
        split_parts = df['option_symbol'].str.split(' ', expand=True)
        df['futures_contract'] = split_parts[0]
        df['right_strike'] = split_parts[1]
        
        # Filter to ACTIVE contracts
        from src.common.utils.es_contract_calendar import get_active_contracts
        active_contracts = get_active_contracts(date_str)
        df = df[df['futures_contract'].isin(active_contracts)].copy()
        
        df['right'] = df['right_strike'].str[0]
        df['strike'] = df['right_strike'].str[1:].astype(float)
        df['exp_date'] = date_str
        
        return pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['option_symbol'],
            'exp_date': df['exp_date'],
            'strike': df['strike'],
            'right': df['right'],
            'open_interest': df.get('quantity', 0).astype('float64') # OI is quantity
        })
    
    def download_date(self, date_str: str, force: bool = False, only_stats: bool = False) -> dict:
        """
        Download SPX options trades + NBBO + Stats for a single date.
        
        Args:
            date_str: Date (YYYY-MM-DD)
            force: Re-download if exists
            only_stats: Download only statistics (Open Interest)
        """
        # ... existing prologue ...
        if only_stats:
            stats_output_dir = self.bronze_root / 'options' / 'statistics' / 'underlying=ES' / f'date={date_str}'
            print(f"\n{'='*60}")
            print(f"{date_str}: Downloading ES options (Statistics ONLY)")
            print(f"{'='*60}")
            stats_count = self._download_schema(date_str, 'statistics', stats_output_dir)
            return {'trades': 0, 'nbbo': 0, 'stats': stats_count}

        return self._download_date_full(date_str, force)

    def _download_date_full(self, date_str: str, force: bool = False) -> dict:
        """Original download_date logic split out."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() >= 5:
            print(f"{date_str}: Weekend, skipping")
            return {'trades': 0, 'nbbo': 0, 'stats': 0}
        
        # Check if exists
        trades_output_dir = self.bronze_root / 'options' / 'trades' / 'underlying=ES' / f'date={date_str}'
        nbbo_output_dir = self.bronze_root / 'options' / 'nbbo' / 'underlying=ES' / f'date={date_str}'
        
        if not force:
            trades_exists = trades_output_dir.exists() and list(trades_output_dir.rglob('*.parquet'))
            nbbo_exists = nbbo_output_dir.exists() and list(nbbo_output_dir.rglob('*.parquet'))
            if trades_exists and nbbo_exists:
                print(f"{date_str}: Already exists (use --force to re-download)")
                # still check stats? assume if trades exist, we might skip unless forced
                return {'trades': -1, 'nbbo': -1, 'stats': -1}
        
        print(f"\n{'='*60}")
        print(f"{date_str}: Downloading ES options")
        print(f"{'='*60}")
        
        # Download trades
        trades_count = self._download_schema(date_str, 'trades', trades_output_dir)
        
        # Download NBBO (MBP-1)
        nbbo_count = self._download_schema(date_str, 'mbp-1', nbbo_output_dir)

        # Download Statistics (Open Interest)
        stats_output_dir = self.bronze_root / 'options' / 'statistics' / 'underlying=ES' / f'date={date_str}'
        stats_count = self._download_schema(date_str, 'statistics', stats_output_dir)
        
        return {'trades': trades_count, 'nbbo': nbbo_count, 'stats': stats_count}

    def download_range(
        self, start_date: str, end_date: str, force: bool = False, max_workers: int = 4, only_stats: bool = False
    ) -> list:
        """
        Download ES options for date range in parallel.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Re-download if exists
            max_workers: Max parallel downloads
            only_stats: Download only statistics
        """
        # Build list of dates
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        print(f"\nDownloading {len(dates)} dates with {max_workers} parallel workers (Stats Only: {only_stats})...\n")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self.download_date, date, force, only_stats): date
                for date in dates
            }

            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    result['date'] = date
                    results.append(result)
                except Exception as e:
                    print(f"{date}: Error - {e}")
                    results.append({'date': date, 'trades': 0, 'nbbo': 0, 'stats': 0, 'error': str(e)})

        return sorted(results, key=lambda x: x['date'])


def main():
    parser = argparse.ArgumentParser(
        description='Download ES options (trades + NBBO) from Databento CME'
    )
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='Re-download if exists')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers (default: 4)')
    parser.add_argument('--only-stats', action='store_true', help='Download ONLY statistics (Open Interest)')
    
    args = parser.parse_args()
    
    try:
        downloader = ESOptionsDownloader()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nSet DATABENTO_API_KEY in backend/.env")
        return 1
    
    if args.date:
        if args.only_stats:
            print(f"\n{args.date}: Downloading ONLY statistics...")
            stats_output_dir = downloader.bronze_root / 'options' / 'statistics' / 'underlying=ES' / f'date={args.date}'
            stats_count = downloader._download_schema(args.date, 'statistics', stats_output_dir)
            result = {'trades': 0, 'nbbo': 0, 'stats': stats_count}
        else:
            result = downloader.download_date(args.date, force=args.force)
            
        print(f"\n✅ Downloaded ES options for {args.date}")
        print(f"   Trades: {result['trades']:,}")
        print(f"   NBBO: {result['nbbo']:,}")
        print(f"   Stats: {result['stats']:,}")
        return 0
    
    if args.start and args.end:
        results = downloader.download_range(
            args.start, args.end, force=args.force, max_workers=args.workers, only_stats=args.only_stats
        )
        
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        total_trades = 0
        total_nbbo = 0
        
        for result in results:
            date = result['date']
            trades = result['trades']
            nbbo = result['nbbo']
            stats = result.get('stats', 0)
            
            if trades == -1:
                print(f"{date}: Skipped (already exists)")
            elif trades == 0:
                print(f"{date}: No data / weekend")
            else:
                print(f"{date}: {trades:,} trades, {nbbo:,} NBBO, {stats:,} Stats")
                total_trades += trades
                total_nbbo += nbbo
        
        print(f"\nTotal: {total_trades:,} trades, {total_nbbo:,} NBBO records")
        return 0
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

