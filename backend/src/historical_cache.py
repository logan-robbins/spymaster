import os
import duckdb
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
from polygon import RESTClient


class HistoricalDataCache:
    """
    Local cache for raw historical trade data.
    Fetch once from Polygon, store in Parquet, reuse forever.
    """
    
    def __init__(self, cache_dir: str = "data/historical/trades"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # DuckDB connection for efficient querying
        self.db = duckdb.connect(":memory:")
        
        print(f"📦 HistoricalDataCache: Using {self.cache_dir.absolute()}")
    
    def _get_cache_path(self, ticker: str, trade_date: date) -> Path:
        """Generate cache file path: data/historical/trades/YYYY-MM-DD/{ticker}.parquet"""
        date_dir = self.cache_dir / trade_date.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean ticker for filename (remove special chars)
        safe_ticker = ticker.replace(":", "_")
        return date_dir / f"{safe_ticker}.parquet"
    
    def has_cached_data(self, ticker: str, trade_date: date) -> bool:
        """Check if we have cached data for this ticker and date."""
        cache_path = self._get_cache_path(ticker, trade_date)
        return cache_path.exists()
    
    def get_cached_trades(
        self, 
        ticker: str, 
        trade_date: date, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Read cached trades from Parquet.
        Optionally filter by time range.
        """
        cache_path = self._get_cache_path(ticker, trade_date)
        
        if not cache_path.exists():
            return []
        
        try:
            # Read directly with pandas (DuckDB renames 't' to 't_1')
            df = pd.read_parquet(cache_path)
            
            # Filter by time if specified
            if start_time or end_time:
                # Convert timestamp column (ms) to datetime for filtering
                df['dt'] = pd.to_datetime(df['t'], unit='ms')
                
                if start_time:
                    df = df[df['dt'] >= start_time]
                if end_time:
                    df = df[df['dt'] <= end_time]
                
                df = df.drop(columns=['dt'])
            
            # Convert to list of dicts
            return df.to_dict('records')
            
        except Exception as e:
            print(f"⚠️ Error reading cache for {ticker}: {e}")
            return []
    
    def save_trades(self, ticker: str, trade_date: date, trades: List[Dict[str, Any]]):
        """
        Save trades to Parquet cache.
        Only saves if there's data to save.
        """
        if not trades:
            return
        
        cache_path = self._get_cache_path(ticker, trade_date)
        
        try:
            df = pd.DataFrame(trades)
            
            # Ensure we have the required columns
            required_cols = ['T', 'p', 's', 't']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Missing required columns for {ticker}, skipping cache")
                return
            
            # Write to Parquet
            df.to_parquet(cache_path, engine='pyarrow', index=False)
            print(f"💾 Cached {len(trades)} trades for {ticker} on {trade_date}")
            
        except Exception as e:
            print(f"❌ Error caching trades for {ticker}: {e}")
    
    def fetch_or_get_trades(
        self,
        client: RESTClient,
        ticker: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Smart fetch: Check cache first, fetch from Polygon only if needed.
        Handles multi-day ranges by caching each day separately.
        
        Intelligence:
        - If cache exists for a day, use it (already has full trading session)
        - If no cache, fetch ONLY the exact time range requested
        - This avoids fetching future data during live trading hours
        """
        # Determine which days we need
        current_date = start_time.date()
        end_date = end_time.date()
        
        all_trades = []
        
        while current_date <= end_date:
            # Determine the time boundaries we need for this specific day
            if current_date == start_time.date():
                day_start = start_time
            else:
                # For days after the first, start from market open
                day_start = datetime.combine(current_date, datetime.min.time())
                if start_time.tzinfo:
                    day_start = day_start.replace(tzinfo=start_time.tzinfo)
                # Set to market open (9:30 AM ET)
                day_start = day_start.replace(hour=9, minute=30)
            
            if current_date == end_time.date():
                day_end = end_time
            else:
                # For days before the last, end at market close
                day_end = datetime.combine(current_date, datetime.max.time())
                if end_time.tzinfo:
                    day_end = day_end.replace(tzinfo=end_time.tzinfo)
                # Set to market close (4:00 PM ET)
                day_end = day_end.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if we have this day cached
            if self.has_cached_data(ticker, current_date):
                print(f"✓ Using cached data for {ticker} on {current_date}")
                
                cached_trades = self.get_cached_trades(ticker, current_date, day_start, day_end)
                all_trades.extend(cached_trades)
            else:
                # Fetch from Polygon for the EXACT time range requested
                # This prevents fetching future data during live trading
                print(f"⬇️ Fetching {ticker} from Polygon: {day_start.strftime('%Y-%m-%d %H:%M')} to {day_end.strftime('%H:%M')}")
                
                ts_start = int(day_start.timestamp() * 1_000_000_000)
                ts_end = int(day_end.timestamp() * 1_000_000_000)
                
                day_trades = []
                
                try:
                    resp = client.list_trades(
                        ticker, 
                        timestamp_gte=ts_start, 
                        timestamp_lte=ts_end, 
                        limit=50000
                    )
                    
                    for t in resp:
                        ts = t.sip_timestamp
                        if ts > 10**14:  # Convert ns to ms
                            ts = int(ts / 1000000)
                        
                        day_trades.append({
                            'T': ticker,
                            'p': t.price,
                            's': t.size,
                            't': ts
                        })
                    
                    # Only cache if this represents a complete trading session
                    # Complete = ends at or after market close (4:00 PM)
                    is_complete_session = day_end.hour >= 16
                    
                    if day_trades and is_complete_session:
                        self.save_trades(ticker, current_date, day_trades)
                        print(f"💾 Cached {len(day_trades)} trades (complete session)")
                    elif day_trades and not is_complete_session:
                        print(f"⚠️ Not caching {len(day_trades)} trades (incomplete session, market still open)")
                    
                    all_trades.extend(day_trades)
                    
                except Exception as e:
                    print(f"Error fetching {ticker} for {current_date}: {e}")
            
            # Move to next day
            current_date = datetime.combine(current_date, datetime.min.time()) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        return all_trades
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        total_files = 0
        total_size = 0
        dates = set()
        
        for date_dir in self.cache_dir.iterdir():
            if date_dir.is_dir():
                dates.add(date_dir.name)
                for file in date_dir.glob("*.parquet"):
                    total_files += 1
                    total_size += file.stat().st_size
        
        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "cached_dates": sorted(dates),
            "cache_dir": str(self.cache_dir.absolute())
        }

