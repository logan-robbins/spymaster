
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

# Path to Bronze MBO
bronze_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-06")

if not bronze_path.exists():
    print(f"Path not found: {bronze_path}")
    exit(1)

parquet_files = sorted(list(bronze_path.glob("*.parquet")))
if not parquet_files:
    print("No parquet files found.")
    exit(1)

first_file = parquet_files[0]
print(f"Loading metadata from {first_file}...")

meta = pq.read_metadata(first_file)
print(f"Rows: {meta.num_rows}")

# Read first few rows for ts_event
df = pd.read_parquet(first_file).head(10)
print("\nFirst 10 Rows:")
print(df[["ts_event", "action", "flags"]])

start_ts = df["ts_event"].min()
print(f"\nMin TS: {start_ts}")
print(f"Time (UTC): {pd.to_datetime(start_ts, unit='ns')}")
print(f"Time (ET): {pd.to_datetime(start_ts, unit='ns').tz_localize('UTC').tz_convert('US/Eastern')}")
