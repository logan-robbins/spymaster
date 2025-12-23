"""
Polygon REST API historical downloader for SPY options data.

Downloads historical option trades for dates matching available DBN data
to calculate real gamma exposure metrics using production FuelEngine.

Per PLAN.md §6: Polygon API Integration (REQUIRED - Options Data)

Usage:
    # Download options for a trading day (one-time per date)
    uv run python -m src.ingestor.polygon_historical --date 2025-12-18

    # Download all available DBN dates
    uv run python -m src.ingestor.polygon_historical --download-all

    # List available dates without downloading
    uv run python -m src.ingestor.polygon_historical --list-dates

    # Dry run (show what would be downloaded)
    uv run python -m src.ingestor.polygon_historical --date 2025-12-18 --dry-run
"""

import argparse
import os
import time
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
import json

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from src.common.event_types import OptionTrade, EventSource, Aggressor
from src.ingestor.dbn_ingestor import DBNIngestor


# Load .env from backend directory
_backend_dir = Path(__file__).parent.parent.parent
load_dotenv(_backend_dir / '.env')


class PolygonHistoricalDownloader:
    """
    Download historical SPY options data from Polygon REST API.

    Writes to Bronze tier matching existing schema at:
    backend/data/lake/bronze/options/trades/underlying=SPY/date={date}/
    """

    BASE_URL = "https://api.polygon.io/v3/trades"

    # Rate limiting
    REQUESTS_PER_MINUTE = 5  # Conservative for free/starter tier
    REQUEST_DELAY_S = 60.0 / REQUESTS_PER_MINUTE

    # Market hours in ET
    MARKET_OPEN_ET = "09:30:00"
    MARKET_CLOSE_ET = "16:00:00"

    def __init__(self, api_key: Optional[str] = None, data_root: Optional[str] = None):
        """
        Initialize Polygon downloader.

        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
            data_root: Root directory for data lake (defaults to backend/data/lake)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY not found. Set it in .env or pass to constructor."
            )

        # Data root setup
        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = _backend_dir / 'data' / 'lake'

        self.bronze_root = self.data_root / 'bronze'

        # DBN ingestor for discovering available dates
        self.dbn_ingestor = DBNIngestor()

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {self.api_key}'
        })

        # Rate limiting state
        self._last_request_time = 0.0
        self._request_count = 0

    def _rate_limit(self):
        """Apply rate limiting to respect API limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY_S:
            sleep_time = self.REQUEST_DELAY_S - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()
        self._request_count += 1

    def build_option_ticker(
        self,
        strike: float,
        right: str,
        exp_date: str
    ) -> str:
        """
        Construct Polygon option ticker symbol.

        Format: O:SPY{YYMMDD}{C|P}{00000000}
        Example: O:SPY251216C00687000 (Dec 16, 2025 $687 Call)

        Args:
            strike: Strike price (e.g., 687.0)
            right: 'C' for call, 'P' for put
            exp_date: Expiration date in YYYY-MM-DD format

        Returns:
            Polygon option ticker string
        """
        # Parse expiration date
        exp = datetime.strptime(exp_date, '%Y-%m-%d')
        exp_str = exp.strftime('%y%m%d')  # YYMMDD

        # Strike encoding: multiply by 1000, pad to 8 digits
        # Example: 687.0 -> 687000 -> 00687000
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"

        return f"O:SPY{exp_str}{right}{strike_str}"

    def parse_option_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Parse Polygon option ticker to extract components.

        Args:
            ticker: Polygon option ticker (e.g., O:SPY251216C00687000)

        Returns:
            Dict with underlying, exp_date, right, strike
        """
        # Remove O: prefix
        symbol = ticker.replace('O:', '')

        # SPY251216C00687000
        underlying = symbol[:3]  # SPY
        exp_yy = symbol[3:5]     # 25
        exp_mm = symbol[5:7]     # 12
        exp_dd = symbol[7:9]     # 16
        right = symbol[9]        # C or P
        strike_str = symbol[10:] # 00687000

        # Construct expiration date
        exp_date = f"20{exp_yy}-{exp_mm}-{exp_dd}"

        # Parse strike (divide by 1000)
        strike = int(strike_str) / 1000

        return {
            'underlying': underlying,
            'exp_date': exp_date,
            'right': right,
            'strike': strike
        }

    def _get_market_timestamps(self, date_str: str) -> Tuple[int, int]:
        """
        Get market open/close timestamps in nanoseconds for a given date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            Tuple of (open_ns, close_ns) in Unix nanoseconds
        """
        # Parse date
        dt = datetime.strptime(date_str, '%Y-%m-%d')

        # Market hours are in ET (Eastern Time)
        # For December dates, EST is active (UTC-5)
        # Market open: 09:30 ET = 14:30 UTC
        # Market close: 16:00 ET = 21:00 UTC

        # Create UTC datetime explicitly
        open_dt = datetime(
            dt.year, dt.month, dt.day,
            hour=14, minute=30, second=0, microsecond=0,
            tzinfo=timezone.utc
        )
        open_ns = int(open_dt.timestamp() * 1e9)

        close_dt = datetime(
            dt.year, dt.month, dt.day,
            hour=21, minute=0, second=0, microsecond=0,
            tzinfo=timezone.utc
        )
        close_ns = int(close_dt.timestamp() * 1e9)

        return open_ns, close_ns

    def fetch_trades_for_ticker(
        self,
        ticker: str,
        start_ns: int,
        end_ns: int,
        limit: int = 50000
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch trades for a single option ticker from Polygon API.

        Handles pagination via next_url.

        Args:
            ticker: Option ticker (e.g., O:SPY251216C00687000)
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            limit: Max results per request

        Yields:
            Raw trade dictionaries from Polygon API
        """
        # Build initial URL
        url = f"{self.BASE_URL}/{ticker}"
        params = {
            'timestamp.gte': start_ns,
            'timestamp.lte': end_ns,
            'limit': limit,
            'sort': 'timestamp',
            'order': 'asc'
        }

        while url:
            self._rate_limit()

            try:
                if '?' in url:
                    # Pagination URL already has params
                    response = self._session.get(url)
                else:
                    response = self._session.get(url, params=params)

                response.raise_for_status()
                data = response.json()

                # Yield results
                for trade in data.get('results', []):
                    yield trade

                # Check for next page
                url = data.get('next_url')
                if url:
                    # Add API key to next_url
                    url = f"{url}&apiKey={self.api_key}"
                    params = None  # next_url has all params

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    print(f"  Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                elif e.response.status_code == 404:
                    # No trades for this ticker
                    break
                else:
                    print(f"  HTTP Error for {ticker}: {e}")
                    break
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                break

    def normalize_option_trade(
        self,
        raw_trade: Dict[str, Any],
        ticker: str
    ) -> OptionTrade:
        """
        Convert Polygon raw trade to normalized OptionTrade event.

        Args:
            raw_trade: Raw trade dict from Polygon API
            ticker: Option ticker for parsing metadata

        Returns:
            Normalized OptionTrade event
        """
        # Parse ticker
        parsed = self.parse_option_ticker(ticker)

        # Polygon timestamp is in nanoseconds
        ts_event_ns = raw_trade.get('sip_timestamp') or raw_trade.get('participant_timestamp')
        ts_recv_ns = int(time.time_ns())  # Download time

        # Infer aggressor from conditions if available
        # Polygon condition codes: https://polygon.io/glossary/us/stocks/conditions-indicators
        conditions = raw_trade.get('conditions', [])
        aggressor = Aggressor.MID

        # Parse expiration date
        exp_date = datetime.strptime(parsed['exp_date'], '%Y-%m-%d').date()

        return OptionTrade(
            ts_event_ns=ts_event_ns,
            ts_recv_ns=ts_recv_ns,
            source=EventSource.POLYGON_REST,
            underlying=parsed['underlying'],
            option_symbol=ticker,
            exp_date=parsed['exp_date'],
            strike=parsed['strike'],
            right=parsed['right'],
            price=raw_trade.get('price', 0.0),
            size=raw_trade.get('size', 0),
            opt_bid=None,  # Not available in trade data
            opt_ask=None,
            aggressor=aggressor,
            conditions=conditions if conditions else None,
            seq=raw_trade.get('sequence_number')
        )

    def _get_spy_price_range(self, date_str: str) -> Tuple[float, float]:
        """
        Get SPY price range for a date from ES futures data.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            Tuple of (min_price, max_price) in SPY terms
        """
        # Read trades from DBN
        trades = list(self.dbn_ingestor.read_trades(date=date_str))

        if not trades:
            raise ValueError(f"No ES trades found for {date_str}")

        # Filter to reasonable ES prices (3000 - 10000 range)
        # This removes any outliers or bad data points
        es_prices = [t.price for t in trades if 3000 < t.price < 10000]

        if not es_prices:
            raise ValueError(f"No valid ES prices found for {date_str} in range 3000-10000")

        # Use percentiles to avoid outliers
        es_prices_sorted = sorted(es_prices)
        # 1st percentile for min, 99th percentile for max
        idx_min = max(0, int(len(es_prices_sorted) * 0.01))
        idx_max = min(len(es_prices_sorted) - 1, int(len(es_prices_sorted) * 0.99))

        es_min = es_prices_sorted[idx_min]
        es_max = es_prices_sorted[idx_max]

        # Convert to SPY (ES ≈ SPY × 10)
        spy_min = es_min / 10
        spy_max = es_max / 10

        print(f"  ES range: [{es_min:.2f}, {es_max:.2f}] -> SPY range: [{spy_min:.2f}, {spy_max:.2f}]")

        return spy_min, spy_max

    def download_options_for_date(
        self,
        date_str: str,
        spy_min: Optional[float] = None,
        spy_max: Optional[float] = None,
        exp_dates: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> int:
        """
        Download all option trades for a trading date.

        Args:
            date_str: Trading date in YYYY-MM-DD format
            spy_min: Minimum SPY price (if None, calculated from ES data)
            spy_max: Maximum SPY price (if None, calculated from ES data)
            exp_dates: List of expiration dates to download (if None, uses trading date for 0DTE)
            dry_run: If True, show what would be downloaded without actually downloading

        Returns:
            Total number of trades downloaded
        """
        print(f"\n{'='*60}")
        print(f"Downloading SPY options for {date_str}")
        print(f"{'='*60}")

        # Get price range if not provided
        if spy_min is None or spy_max is None:
            try:
                spy_min, spy_max = self._get_spy_price_range(date_str)
            except ValueError as e:
                print(f"  ERROR: {e}")
                return 0

        # Calculate strike range (±$10 from extremes)
        strike_min = int(spy_min - 10)
        strike_max = int(spy_max + 10)

        # Generate whole dollar strikes
        strikes = list(range(strike_min, strike_max + 1))

        # Use trading date as expiration for 0DTE if not specified
        if exp_dates is None:
            exp_dates = [date_str]

        # Get market timestamps
        start_ns, end_ns = self._get_market_timestamps(date_str)

        print(f"  Strike range: ${strike_min} - ${strike_max} ({len(strikes)} strikes)")
        print(f"  Expirations: {exp_dates}")
        print(f"  Time range: {datetime.fromtimestamp(start_ns/1e9, tz=timezone.utc)} to "
              f"{datetime.fromtimestamp(end_ns/1e9, tz=timezone.utc)}")

        if dry_run:
            print(f"\n  [DRY RUN] Would download {len(strikes) * len(exp_dates) * 2} option contracts")
            return 0

        # Prepare output directory
        output_dir = self.bronze_root / 'options' / 'trades' / 'underlying=SPY' / f'date={date_str}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download trades
        all_trades = []
        total_count = 0
        ticker_count = 0

        for exp_date in exp_dates:
            for strike in strikes:
                for right in ['C', 'P']:
                    ticker = self.build_option_ticker(strike, right, exp_date)
                    ticker_count += 1

                    # Progress indicator
                    if ticker_count % 10 == 0:
                        print(f"  Progress: {ticker_count}/{len(strikes) * len(exp_dates) * 2} tickers, "
                              f"{total_count} trades so far")

                    # Fetch trades
                    trade_count = 0
                    for raw_trade in self.fetch_trades_for_ticker(ticker, start_ns, end_ns):
                        trade = self.normalize_option_trade(raw_trade, ticker)
                        all_trades.append(self._trade_to_dict(trade))
                        trade_count += 1
                        total_count += 1

                        # Batch write every 50000 trades
                        if len(all_trades) >= 50000:
                            self._write_trades(all_trades, output_dir, date_str)
                            all_trades = []

                    if trade_count > 0:
                        print(f"    {ticker}: {trade_count} trades")

        # Write remaining trades
        if all_trades:
            self._write_trades(all_trades, output_dir, date_str)

        print(f"\n  Total: {total_count} trades downloaded for {date_str}")
        return total_count

    def _trade_to_dict(self, trade: OptionTrade) -> Dict[str, Any]:
        """Convert OptionTrade to dict for Parquet storage."""
        return {
            'ts_event_ns': trade.ts_event_ns,
            'ts_recv_ns': trade.ts_recv_ns,
            'source': trade.source.value,
            'underlying': trade.underlying,
            'option_symbol': trade.option_symbol,
            'exp_date': trade.exp_date,
            'strike': trade.strike,
            'right': trade.right,
            'price': trade.price,
            'size': trade.size,
            'opt_bid': trade.opt_bid,
            'opt_ask': trade.opt_ask,
            'aggressor': trade.aggressor.value,
            'conditions': str(trade.conditions) if trade.conditions else None,
            'seq': trade.seq
        }

    def _write_trades(
        self,
        trades: List[Dict[str, Any]],
        output_dir: Path,
        date_str: str
    ):
        """Write trades batch to Parquet file."""
        if not trades:
            return

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Sort by event time
        df = df.sort_values('ts_event_ns')

        # Generate unique filename
        timestamp_str = datetime.now(timezone.utc).strftime('%H%M%S_%f')
        file_name = f'part-{timestamp_str}.parquet'
        file_path = output_dir / file_name

        # Write with ZSTD compression
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            file_path,
            compression='zstd',
            compression_level=3
        )

        print(f"  Wrote {len(df)} trades -> {file_path}")

    def get_available_dbn_dates(self) -> List[str]:
        """Get list of available weekday DBN dates."""
        dates = self.dbn_ingestor.get_available_dates('trades')

        # Filter to weekdays only (Mon=0 to Fri=4)
        weekday_dates = []
        for date_str in dates:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            if dt.weekday() < 5:  # Monday to Friday
                weekday_dates.append(date_str)

        return weekday_dates

    def check_options_downloaded(self, date_str: str) -> bool:
        """Check if options data already exists for a date."""
        options_dir = self.bronze_root / 'options' / 'trades' / 'underlying=SPY' / f'date={date_str}'
        if not options_dir.exists():
            return False

        # Check if there are any Parquet files
        parquet_files = list(options_dir.glob('*.parquet'))
        return len(parquet_files) > 0

    def download_all_dbn_dates(self, dry_run: bool = False, force: bool = False) -> Dict[str, int]:
        """
        Download options data for all available DBN dates.

        Args:
            dry_run: If True, show what would be downloaded
            force: If True, re-download even if data exists

        Returns:
            Dict of {date: trade_count}
        """
        dates = self.get_available_dbn_dates()

        print(f"Found {len(dates)} weekday dates with DBN data:")
        for d in dates:
            status = "EXISTS" if self.check_options_downloaded(d) else "MISSING"
            print(f"  {d}: {status}")

        results = {}

        for date_str in dates:
            if not force and self.check_options_downloaded(date_str):
                print(f"\nSkipping {date_str} (already downloaded)")
                results[date_str] = -1  # Already exists
                continue

            count = self.download_options_for_date(date_str, dry_run=dry_run)
            results[date_str] = count

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download historical SPY options data from Polygon API'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Trading date to download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download options for all available DBN dates'
    )
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available DBN dates without downloading'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if data already exists'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Polygon API key (defaults to POLYGON_API_KEY env var)'
    )

    args = parser.parse_args()

    # Initialize downloader
    try:
        downloader = PolygonHistoricalDownloader(api_key=args.api_key)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # List dates mode
    if args.list_dates:
        dates = downloader.get_available_dbn_dates()
        print(f"Available DBN dates ({len(dates)} weekdays):")
        for d in dates:
            status = "DOWNLOADED" if downloader.check_options_downloaded(d) else "NOT DOWNLOADED"
            print(f"  {d}: {status}")
        return 0

    # Download all mode
    if args.download_all:
        results = downloader.download_all_dbn_dates(
            dry_run=args.dry_run,
            force=args.force
        )
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for date_str, count in results.items():
            if count == -1:
                print(f"  {date_str}: skipped (already exists)")
            else:
                print(f"  {date_str}: {count} trades")
        return 0

    # Single date mode
    if args.date:
        count = downloader.download_options_for_date(
            args.date,
            dry_run=args.dry_run
        )
        print(f"\nDownloaded {count} trades for {args.date}")
        return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    exit(main())
