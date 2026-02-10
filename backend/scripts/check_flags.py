
import pandas as pd
from pathlib import Path

bronze_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-06")
parquet_file = list(bronze_path.glob("*.parquet"))[0]

print(f"Scanning flags in {parquet_file}...")
df = pd.read_parquet(parquet_file, columns=["flags", "action", "ts_event"])

print("\nValue Counts for 'flags':")
print(df["flags"].value_counts())

print("\nValue Counts for 'flags' bitwise AND 32 (F_SNAPSHOT, bit 5):")
print((df["flags"] & 32).value_counts())

print("\nSample of rows with Snapshot flag (F_SNAPSHOT=32):")
print(df[df["flags"] & 32 != 0].head(10))

print("\nValue Counts for 'action':")
print(df["action"].value_counts())

print("\nValue Counts for 'flags' (Is Snapshot?) vs 'action':")
print(df.groupby(["action", df["flags"] & 32 != 0]).size())
