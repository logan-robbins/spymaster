"""
Polygon/Massive S3 Flat Files Downloader for SPY Options.

Downloads historical option trades from S3 flat files - MUCH faster than REST API.
Single CSV.gz file per day instead of thousands of API calls.

S3 Configuration:
- Endpoint: https://files.massive.com (or https://files.polygon.io)
- Bucket: flatfiles
- Options path: us_options_opra/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz

Usage:
    # Download single date
    uv run python -m src.ingestor.polygon_flatfiles --date 2025-12-18

    # Download all DBN dates
    uv run python -m src.ingestor.polygon_flatfiles --download-all

    # List available files
    uv run python -m src.ingestor.polygon_flatfiles --list

Requirements:
    pip install boto3
"""

import argparse
import gzip
import os
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import boto3
from botocore.config import Config
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from src.ingestor.dbn_ingestor import DBNIngestor


# Load .env
_backend_dir = Path(__file__).parent.parent.parent
load_dotenv(_backend_dir / '.env')


class PolygonFlatFileDownloader:
    """
    Fast S3-based downloader for Polygon/Massive options flat files.

    Downloads entire day's options trades in one file (~50-200MB compressed).
    """

    # S3 configuration - try both endpoints
    S3_ENDPOINTS = [
        "https://files.polygon.io",
        "https://files.massive.com",
    ]
    BUCKET = "flatfiles"
    OPTIONS_PREFIX = "us_options_opra/trades_v1"

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        data_root: Optional[Path] = None
    ):
        """
        Initialize S3 flat file downloader.

        Args:
            access_key: S3 access key (defaults to POLYGON_S3_ACCESS_KEY or POLYGON_API_KEY)
            secret_key: S3 secret key (defaults to POLYGON_S3_SECRET_KEY or POLYGON_API_KEY)
            data_root: Root directory for data lake
        """
        # Try S3-specific keys first, then fall back to API key
        self.access_key = (
            access_key or
            os.getenv('POLYGON_S3_ACCESS_KEY') or
            os.getenv('POLYGON_API_KEY')
        )
        self.secret_key = (
            secret_key or
            os.getenv('POLYGON_S3_SECRET_KEY') or
            os.getenv('POLYGON_API_KEY')
        )

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "S3 credentials not found. Set POLYGON_S3_ACCESS_KEY/SECRET_KEY or POLYGON_API_KEY in .env"
            )

        # Data root
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = _backend_dir / 'data' / 'lake'

        self.bronze_root = self.data_root / 'bronze'

        # DBN ingestor for date discovery
        self.dbn_ingestor = DBNIngestor()

        # Initialize S3 client (will try endpoints)
        self.s3_client = None
        self.working_endpoint = None

    def _init_s3_client(self, endpoint: str) -> boto3.client:
        """Create S3 client for given endpoint."""
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3}
            )
        )

    def _get_s3_client(self):
        """Get working S3 client, trying multiple endpoints."""
        if self.s3_client:
            return self.s3_client

        for endpoint in self.S3_ENDPOINTS:
            try:
                client = self._init_s3_client(endpoint)
                # Test connection by listing bucket
                client.list_objects_v2(Bucket=self.BUCKET, MaxKeys=1)
                self.s3_client = client
                self.working_endpoint = endpoint
                print(f"Connected to S3: {endpoint}")
                return client
            except Exception as e:
                print(f"Failed to connect to {endpoint}: {e}")
                continue

        raise RuntimeError("Could not connect to any S3 endpoint")

    def list_available_dates(self, year: int = 2025, month: int = 12) -> List[str]:
        """List available flat file dates for a given month."""
        client = self._get_s3_client()

        prefix = f"{self.OPTIONS_PREFIX}/{year}/{month:02d}/"

        try:
            response = client.list_objects_v2(
                Bucket=self.BUCKET,
                Prefix=prefix
            )

            dates = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                # Extract date from path like us_options_opra/trades_v1/2025/12/2025-12-18.csv.gz
                filename = key.split('/')[-1]
                if filename.endswith('.csv.gz'):
                    date_str = filename.replace('.csv.gz', '')
                    dates.append(date_str)

            return sorted(dates)

        except Exception as e:
            print(f"Error listing dates: {e}")
            return []

    def download_date(self, date_str: str, force: bool = False) -> int:
        """
        Download options trades for a single date from S3 flat files.

        Args:
            date_str: Date in YYYY-MM-DD format
            force: Re-download even if exists

        Returns:
            Number of SPY trades downloaded
        """
        # Check if already downloaded
        output_dir = self.bronze_root / 'options' / 'trades' / 'underlying=SPY' / f'date={date_str}'
        if output_dir.exists() and not force:
            existing = list(output_dir.glob('*.parquet'))
            if existing:
                print(f"  {date_str}: Already exists ({len(existing)} files), skipping")
                return -1

        # Parse date for S3 path
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        s3_key = f"{self.OPTIONS_PREFIX}/{dt.year}/{dt.month:02d}/{date_str}.csv.gz"

        print(f"  Downloading s3://{self.BUCKET}/{s3_key}...")

        client = self._get_s3_client()

        try:
            # Download file to memory
            response = client.get_object(Bucket=self.BUCKET, Key=s3_key)
            compressed_data = response['Body'].read()

            print(f"  Downloaded {len(compressed_data) / 1024 / 1024:.1f} MB compressed")

            # Decompress and parse CSV
            with gzip.open(io.BytesIO(compressed_data), 'rt') as f:
                df = pd.read_csv(f)

            print(f"  Parsed {len(df):,} total option trades")

            # Filter to SPY only
            if 'ticker' in df.columns:
                spy_mask = df['ticker'].str.startswith('O:SPY')
                df = df[spy_mask]
            elif 'underlying' in df.columns:
                df = df[df['underlying'] == 'SPY']

            if df.empty:
                print(f"  No SPY trades found in file")
                return 0

            print(f"  Filtered to {len(df):,} SPY trades")

            # Transform to our schema
            df_out = self._transform_to_schema(df, date_str)

            # Write to Bronze Parquet
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'trades_{date_str}.parquet'

            table = pa.Table.from_pandas(df_out, preserve_index=False)
            pq.write_table(table, output_path, compression='zstd')

            print(f"  Wrote {len(df_out):,} trades to {output_path}")

            return len(df_out)

        except client.exceptions.NoSuchKey:
            print(f"  File not found: {s3_key}")
            return 0
        except Exception as e:
            print(f"  Error downloading {date_str}: {e}")
            return 0

    def _transform_to_schema(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Transform Polygon flat file schema to our Bronze schema."""
        # Map Polygon columns to our schema
        # Polygon flat file columns vary, handle flexibly

        out = pd.DataFrame()

        # Timestamp (Polygon uses nanoseconds)
        if 'sip_timestamp' in df.columns:
            out['ts_event_ns'] = df['sip_timestamp'].astype('int64')
        elif 'participant_timestamp' in df.columns:
            out['ts_event_ns'] = df['participant_timestamp'].astype('int64')
        else:
            # Fallback to current time
            import time
            out['ts_event_ns'] = int(time.time_ns())

        out['ts_recv_ns'] = out['ts_event_ns']  # Use same for historical
        out['source'] = 'polygon_flatfile'
        out['underlying'] = 'SPY'

        # Option symbol
        if 'ticker' in df.columns:
            out['option_symbol'] = df['ticker']
        else:
            out['option_symbol'] = 'UNKNOWN'

        # Parse option details from ticker: O:SPY251218C00590000
        # Format: O:SPY + YYMMDD + C/P + 8-digit strike
        def parse_ticker(ticker):
            try:
                if not ticker.startswith('O:SPY'):
                    return None, None, None
                suffix = ticker[5:]  # Remove O:SPY
                exp_yymmdd = suffix[:6]
                right = suffix[6]
                strike_raw = suffix[7:]

                exp_date = f"20{exp_yymmdd[:2]}-{exp_yymmdd[2:4]}-{exp_yymmdd[4:6]}"
                strike = float(strike_raw) / 1000.0

                return exp_date, strike, right
            except:
                return None, None, None

        parsed = out['option_symbol'].apply(parse_ticker)
        out['exp_date'] = parsed.apply(lambda x: x[0] if x else None)
        out['strike'] = parsed.apply(lambda x: x[1] if x else None)
        out['right'] = parsed.apply(lambda x: x[2] if x else None)

        # Price and size
        out['price'] = df.get('price', 0.0).astype('float64')
        out['size'] = df.get('size', 0).astype('int64')

        # Optional fields
        out['opt_bid'] = None
        out['opt_ask'] = None
        out['seq'] = df.get('sequence_number', range(len(df)))

        # Infer aggressor using Tick Rule (Lee-Ready fallback)
        # Sort by timestamp first, then compute price changes per symbol
        out = out.sort_values('ts_event_ns').reset_index(drop=True)

        # Compute previous price per option symbol
        out['prev_price'] = out.groupby('option_symbol')['price'].shift(1)

        # Tick Rule: uptick = BUY (+1), downtick = SELL (-1), no change = MID (0)
        out['aggressor'] = 0  # Default to MID
        out.loc[out['price'] > out['prev_price'], 'aggressor'] = 1   # BUY
        out.loc[out['price'] < out['prev_price'], 'aggressor'] = -1  # SELL

        # Drop helper column
        out = out.drop(columns=['prev_price'])

        # Filter rows with valid data
        out = out.dropna(subset=['exp_date', 'strike', 'right'])

        # Filter to 0DTE only (exp_date == trading date)
        out = out[out['exp_date'] == date_str]

        return out

    def download_all_dbn_dates(self, force: bool = False) -> Dict[str, int]:
        """Download options for all available DBN dates."""
        dates = self.dbn_ingestor.get_available_dates('trades')

        # Filter to weekdays
        weekday_dates = []
        for d in dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            if dt.weekday() < 5:
                weekday_dates.append(d)

        print(f"Found {len(weekday_dates)} weekday dates with DBN data")

        results = {}
        for date_str in weekday_dates:
            print(f"\n{'='*60}")
            print(f"Processing {date_str}")
            print('='*60)

            count = self.download_date(date_str, force=force)
            results[date_str] = count

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Download SPY options from Polygon S3 flat files (fast!)'
    )
    parser.add_argument('--date', type=str, help='Single date to download (YYYY-MM-DD)')
    parser.add_argument('--download-all', action='store_true', help='Download all DBN dates')
    parser.add_argument('--list', action='store_true', help='List available dates')
    parser.add_argument('--force', action='store_true', help='Re-download even if exists')

    args = parser.parse_args()

    try:
        downloader = PolygonFlatFileDownloader()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    if args.list:
        print("Checking available flat files...")
        dates = downloader.list_available_dates(2025, 12)
        print(f"Available dates in 2025-12: {dates}")
        return 0

    if args.download_all:
        results = downloader.download_all_dbn_dates(force=args.force)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for date_str, count in results.items():
            if count == -1:
                print(f"  {date_str}: skipped (exists)")
            elif count == 0:
                print(f"  {date_str}: no data")
            else:
                print(f"  {date_str}: {count:,} trades")
        return 0

    if args.date:
        count = downloader.download_date(args.date, force=args.force)
        if count > 0:
            print(f"Downloaded {count:,} SPY trades for {args.date}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
