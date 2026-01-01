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
    uv run python scripts/download_es_options.py --date 2025-12-16
"""

import argparse
import os
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
            self.data_root = _backend_dir / 'data'
        
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
        
        return {'trades': trades_count, 'nbbo': nbbo_count}
    
    def _download_schema(self, date_str: str, schema: str, output_dir: Path) -> int:
        """Download specific schema and write to Bronze."""
        print(f"\n  Downloading {schema}...")

        try:
            # Download from Databento
            # Note: Using 'parent' stype_in requires format 'SPX.OPT' (not just 'SPX')
            # End date must be exclusive (next day) for Databento API
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            end_date_str = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

            data = self.client.timeseries.get_range(
                dataset=self.DATASET,
                schema=schema,
                symbols=[self.SYMBOL],
                start=date_str,
                end=end_date_str,
                stype_in='parent'  # Get all expirations for SPX.OPT parent
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                print(f"    No {schema} data found")
                return 0
            
            print(f"    Retrieved {len(df):,} {schema} records")
            
            # Transform to Bronze schema
            if schema == 'trades':
                df_bronze = self._transform_trades(df, date_str)
            else:  # mbp-1
                df_bronze = self._transform_nbbo(df, date_str)
            
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
            
        except Exception as e:
            print(f"    Error downloading {schema}: {e}")
            return 0
    
    def _transform_trades(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Transform Databento trades to Bronze schema."""
        # Databento columns: ts_event, ts_recv, symbol, price, size, side, ...
        
        # Parse ES option symbol
        # ES options from CME: Format varies, examples:
        # - EW1H25 C5750 (weekly, month H = March, year 25, Call 5750)
        # - LO prefix for weekly options on ES
        # Databento may provide in different formats
        
        df['option_symbol'] = df['symbol'].astype(str)
        
        # For ES options, symbol metadata should be in the record
        # If not available, we'll parse from symbol string
        
        # Try to extract from symbol or use provided columns
        if 'expiration' in df.columns:
            df['exp_date'] = pd.to_datetime(df['expiration']).dt.strftime('%Y-%m-%d')
        elif 'expiry' in df.columns:
            df['exp_date'] = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d')
        else:
            # Parse from symbol - this may need adjustment based on actual data format
            # For now, default to session date (0DTE assumption)
            df['exp_date'] = date_str
        
        # Right (call/put)
        if 'option_type' in df.columns:
            df['right'] = df['option_type'].map({'C': 'C', 'P': 'P', 'CALL': 'C', 'PUT': 'P'})
        elif 'right' in df.columns:
            df['right'] = df['right']
        else:
            # Try to parse from symbol - may contain C/P indicator
            df['right'] = df['option_symbol'].str.extract(r'([CP])', expand=False).fillna('C')
        
        # Strike price
        if 'strike_price' in df.columns:
            df['strike'] = df['strike_price'].astype(float)
        elif 'strike' not in df.columns:
            # Extract from symbol - this format is dataset-specific
            # ES options strikes are in index points (e.g., 5750.0)
            # May need to parse based on actual Databento format
            df['strike'] = 0.0  # Placeholder - will need actual data to determine format
        
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
        
        # Parse ES option symbol (same logic as trades)
        df['option_symbol'] = df['symbol'].astype(str)
        
        # Extract metadata (same as trades)
        if 'expiration' in df.columns:
            df['exp_date'] = pd.to_datetime(df['expiration']).dt.strftime('%Y-%m-%d')
        elif 'expiry' in df.columns:
            df['exp_date'] = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d')
        else:
            df['exp_date'] = date_str
        
        if 'option_type' in df.columns:
            df['right'] = df['option_type'].map({'C': 'C', 'P': 'P', 'CALL': 'C', 'PUT': 'P'})
        elif 'right' in df.columns:
            df['right'] = df['right']
        else:
            df['right'] = df['option_symbol'].str.extract(r'([CP])', expand=False).fillna('C')
        
        if 'strike_price' in df.columns:
            df['strike'] = df['strike_price'].astype(float)
        else:
            df['strike'] = 0.0  # Placeholder
        
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
    
    def download_range(self, start_date: str, end_date: str, force: bool = False) -> dict:
        """
        Download SPX options for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Re-download if exists
        
        Returns:
            Dict with summary stats
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        results = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            result = self.download_date(date_str, force=force)
            result['date'] = date_str
            results.append(result)
            current += timedelta(days=1)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Download ES options (trades + NBBO) from Databento CME'
    )
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='Re-download if exists')
    
    args = parser.parse_args()
    
    try:
        downloader = ESOptionsDownloader()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nSet DATABENTO_API_KEY in backend/.env")
        return 1
    
    if args.date:
        result = downloader.download_date(args.date, force=args.force)
        print(f"\n✅ Downloaded ES options for {args.date}")
        print(f"   Trades: {result['trades']:,}")
        print(f"   NBBO: {result['nbbo']:,}")
        return 0
    
    if args.start and args.end:
        results = downloader.download_range(args.start, args.end, force=args.force)
        
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        total_trades = 0
        total_nbbo = 0
        
        for result in results:
            date = result['date']
            trades = result['trades']
            nbbo = result['nbbo']
            
            if trades == -1:
                print(f"{date}: Skipped (already exists)")
            elif trades == 0:
                print(f"{date}: No data / weekend")
            else:
                print(f"{date}: {trades:,} trades, {nbbo:,} NBBO")
                total_trades += trades
                total_nbbo += nbbo
        
        print(f"\nTotal: {total_trades:,} trades, {total_nbbo:,} NBBO records")
        return 0
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

