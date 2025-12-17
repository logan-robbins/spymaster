import duckdb
import pandas as pd
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import os

class PersistenceEngine:
    """
    High-throughput asynchronous logging of data to Parquet via DuckDB.
    """
    def __init__(self, db_path: str = ":memory:"):
        # We use DuckDB for its Parquet writing capabilities.
        # Although we might write directly using pandas to parquet,
        # APP.md specifies DuckDB.
        
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_limit = 5000
        self.flush_interval = 60 # seconds
        self.last_flush_time = datetime.now()
        
        # S3 Configuration
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_endpoint = os.getenv("S3_ENDPOINT")
        self.s3_key = os.getenv("S3_ACCESS_KEY")
        self.s3_secret = os.getenv("S3_SECRET_KEY")
        
        if self.s3_bucket and self.s3_key:
             print(f"💾 PersistenceEngine: Configured for S3 ({self.s3_bucket})")
             self.base_path = f"s3://{self.s3_bucket}/data/raw/flow"
             self.storage_options = {
                 "key": self.s3_key,
                 "secret": self.s3_secret,
                 "client_kwargs": {
                     "endpoint_url": self.s3_endpoint
                 }
             }
        else:
             print("💾 PersistenceEngine: Configured for Local Disk")
             self.base_path = "data/raw/flow"
             self.storage_options = None

        self._lock = asyncio.Lock()
        
    async def process_trade(self, trade_data: Dict[str, Any]):
        """
        Ingest a trade record.
        """
        async with self._lock:
            self.buffer.append(trade_data)
        
        # Check triggers
        if len(self.buffer) >= self.buffer_limit or \
           (datetime.now() - self.last_flush_time).total_seconds() > self.flush_interval:
            await self.flush()

    async def flush(self):
        """
        Write buffer to Parquet.
        """
        async with self._lock:
            if not self.buffer:
                return
            
            # Swap buffer
            data_to_write = self.buffer
            self.buffer = []
            self.last_flush_time = datetime.now()

        # Write in executor to avoid blocking loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_to_parquet, data_to_write)

    def _write_to_parquet(self, data: List[Dict[str, Any]]):
        try:
            df = pd.DataFrame(data)
            
            # Determine path based on current time
            now = datetime.utcnow()
            year = now.year
            month = now.month
            day = now.day
            
            # Construct path: data/raw/flow/year=YYYY/month=MM/day=DD/
            dir_path = f"{self.base_path}/year={year}/month={month}/day={day}"
            os.makedirs(dir_path, exist_ok=True)
            
            # Filename: flow_part_{timestamp}.parquet
            timestamp_str = now.strftime("%H%M%S_%f")
            file_path = f"{dir_path}/flow_part_{timestamp_str}.parquet"
            
            # Write
            if self.storage_options:
                df.to_parquet(file_path, engine='pyarrow', index=False, storage_options=self.storage_options)
            else:
                os.makedirs(dir_path, exist_ok=True)
                df.to_parquet(file_path, engine='pyarrow', index=False)
            
            # print(f"💾 Flushed {len(df)} rows to {file_path}")
            
        except Exception as e:
            print(f"❌ Persistence Error: {e}") 
