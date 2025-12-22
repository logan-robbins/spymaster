import os
import duckdb
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
from polygon import RESTClient

class HistoricalDataCache:
    """
    Unified Data Lake Access Layer.
    Reads/Writes to `data/raw/flow` using Hive Partitioning.
    Schema Standard: ticker, price, size, timestamp (ms), [greeks...]
    """
    
    def __init__(self, base_dir: str = "data/raw/flow"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # DuckDB connection for efficient querying across partitions
        self.db = duckdb.connect(":memory:")
        print(f"📦 Data Lake Access: {self.base_dir.absolute()}")
    
    def _get_partition_path(self, trade_date: date) -> Path:
        """
        Get the hive partition directory for a date.
        Format: year=YYYY/month=MM/day=DD
        """
        return self.base_dir / f"year={trade_date.year}/month={trade_date.month}/day={trade_date.day}"
    
    def has_cached_data(self, trade_date: date) -> bool:
        """
        Check if data exists for this DATE partition.
        Note: We define 'cached' as 'the directory exists and contains parquet files'.
        We cannot easily check for a specific ticker without opening the files, 
        so we assume if the partition exists, we have data.
        """
        partition_path = self._get_partition_path(trade_date)
        if not partition_path.exists():
            return False
        
        # Check for any parquet files
        files = list(partition_path.glob("*.parquet"))
        return len(files) > 0
    
    def get_cached_trades(
        self, 
        ticker: Optional[str], 
        trade_date: date, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the Data Lake for trades.
        Uses DuckDB to scan all parquet files in the daily partition effectively.
        """
        partition_path = self._get_partition_path(trade_date)
        if not partition_path.exists():
            return []
            
        # Construct Query
        # We query ALL parquet files in the partition
        # DuckDB requires string path
        partition_glob = str(partition_path / "*.parquet")
        
        # Note: Timestamps in parquet are stored as naive datetime64[ns]
        # We need to compare them directly as timestamps in WHERE clause
        # But return epoch_ms for downstream consumers
        query = f"""
        SELECT ticker, price, size,
               epoch_ms(timestamp) as timestamp,
               delta, gamma, premium, aggressor_side, net_delta_impact
        FROM read_parquet('{partition_glob}') 
        WHERE 1=1
        """
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
            
        if start_time:
            # Convert timezone-aware datetime to naive for comparison
            # Parquet timestamps are naive but represent ET times
            start_naive = start_time.replace(tzinfo=None)
            query += " AND timestamp >= ?"
            params.append(start_naive)
            
        if end_time:
            end_naive = end_time.replace(tzinfo=None)
            query += " AND timestamp <= ?"
            params.append(end_naive)
            
        # Sort by time
        query += " ORDER BY timestamp ASC"
        
        try:
            # Execute
            df = self.db.execute(query, params).df()
            return df.to_dict('records')
        except Exception as e:
            print(f"⚠️ Error querying data lake: {e}")
            return []
            
    def save_trades(self, trade_date: date, trades: List[Dict[str, Any]], source_tag: str = "backfill"):
        """
        Save a batch of trades to the Data Lake.
        Auto-normalizes schema to: ticker, price, size, timestamp.
        """
        if not trades:
            return
            
        partition_path = self._get_partition_path(trade_date)
        partition_path.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.DataFrame(trades)
            
            # Normalize Columns if coming from Polygon API directly (T, p, s, t)
            rename_map = {
                'T': 'ticker',
                'p': 'price',
                's': 'size',
                't': 'timestamp'
            }
            # Only rename if columns exist
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            
            # Ensure timestamp is int (ms)
            if 'timestamp' in df.columns:
                 df['timestamp'] = df['timestamp'].astype('int64')

            # Required columns validation
            required = {'ticker', 'price', 'size', 'timestamp'}
            if not required.issubset(df.columns):
                print(f"❌ Schema alignment failed. Missing: {required - set(df.columns)}")
                return

            # Filename: flow_{source_tag}_{timestamp}.parquet
            ts_str = datetime.now().strftime("%H%M%S_%f")
            file_path = partition_path / f"flow_{source_tag}_{ts_str}.parquet"
            
            df.to_parquet(file_path, engine='pyarrow', index=False)
            print(f"💾 Saved {len(df)} records to {file_path}")
            
        except Exception as e:
             print(f"❌ Error saving to Data Lake: {e}")

    def fetch_backfill(
        self,
        client: RESTClient,
        ticker: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch from Polygon and write to Data Lake.
        Checks existence largely by DATE, not granular ticker/time, 
        so manual backfills might duplicate if repeated.
        """
        # Simple Date Loop
        current_date = start_time.date()
        end_date = end_time.date()
        
        all_trades = []
        
        while current_date <= end_date:
            # Define day constraints
            if current_date == start_time.date():
                t_start = start_time
            else:
                t_start = datetime.combine(current_date, datetime.min.time()).replace(hour=9, minute=30)
                
            if current_date == end_time.date():
                t_end = end_time
            else:
                 t_end = datetime.combine(current_date, datetime.min.time()).replace(hour=16, minute=0)

            # Check if this DATE partition exists. 
            # If so, we assume we might have data. 
            # BUT, since we are requesting a specific TICKER, 
            # and our partition check is generic, we might miss.
            # For robustness: We Query first.
            existing = self.get_cached_trades(ticker, current_date, t_start, t_end)
            if existing:
                print(f"✓ Found {len(existing)} cached trades for {ticker} on {current_date}")
                all_trades.extend(existing)
            else:
                # Fetch
                print(f"⬇️ Fetching {ticker} from Polygon for {current_date}")
                ts_start_ns = int(t_start.timestamp() * 1e9)
                ts_end_ns = int(t_end.timestamp() * 1e9)
                
                new_trades = []
                try:
                    for t in client.list_trades(ticker, timestamp_gte=ts_start_ns, timestamp_lte=ts_end_ns, limit=50000):
                        ts_ms = int(t.sip_timestamp / 1e6)
                        new_trades.append({
                            'ticker': ticker,
                            'price': t.price,
                            'size': t.size,
                            'timestamp': ts_ms
                        })
                    
                    if new_trades:
                        self.save_trades(current_date, new_trades, source_tag=f"backfill_{ticker}")
                        all_trades.extend(new_trades)
                        
                except Exception as e:
                    print(f"Error fetching: {e}")
            
            # Next day
            current_date = (datetime.combine(current_date, datetime.min.time()) + pd.Timedelta(days=1)).date()
            
        return all_trades

    def get_latest_available_date(self) -> Optional[date]:
        """Find the most recent populated date partition."""
        # Walk directory
        # Structure: base/year=Y/month=M/day=D
        # We can implement a smart finder or just recursive glob?
        # recursive glob is easy.
        
        try:
            # Find all 'day=DD' folders
            days = list(self.base_dir.glob("year=*/month=*/day=*"))
            if not days:
                return None
                
            # Parse dates
            found_dates = []
            for d in days:
                try:
                    # path parts
                    parts = d.parts
                    # Expecting .../year=2025/month=12/day=16
                    # Use string searching to be safer than index if path varies
                    y_str = next(p for p in parts if p.startswith("year=")).split('=')[1]
                    m_str = next(p for p in parts if p.startswith("month=")).split('=')[1]
                    d_str = next(p for p in parts if p.startswith("day=")).split('=')[1]
                    dt = date(int(y_str), int(m_str), int(d_str))
                    found_dates.append(dt)
                except:
                    continue
            
            if not found_dates:
                return None
                
            return max(found_dates)
            
        except Exception:
            return None
