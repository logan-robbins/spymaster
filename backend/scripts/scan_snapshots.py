
import pandas as pd
from pathlib import Path

# Path to Bronze MBO
bronze_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-06")
parquet_file = list(bronze_path.glob("*.parquet"))[0]

print(f"Scanning {parquet_file} for snapshots...")

# snapshot flag = 128
# We'll stick to pandas for simplicity, or pyarrow if large.
# Reading 'flags' and 'ts_event' columns only.

df = pd.read_parquet(parquet_file, columns=["ts_event", "flags"])

# Filter for snapshots
# F_SNAPSHOT = 32 (bit 5). F_LAST = 128 (bit 7).
snapshots = df[df["flags"] & 32 != 0]

print(f"Total Rows: {len(df)}")
print(f"Snapshot Rows: {len(snapshots)}")

if not snapshots.empty:
    print("\nSnapshot Timestamps (UTC & ET):")
    # Convert unique timestamps to readable format
    unique_snaps = snapshots["ts_event"].unique()
    for ts in unique_snaps[:20]: # Show first 20
        dt_utc = pd.to_datetime(ts, unit="ns")
        dt_et = dt_utc.tz_localize("UTC").tz_convert("US/Eastern")
        print(f"{ts} | {dt_utc} | {dt_et}")
    
    if len(unique_snaps) > 20:
        print(f"... and {len(unique_snaps) - 20} more.")
        
    # Check if there's one near 07:30 ET
    # 07:30 ET is roughly 12:30 UTC
    print("\nChecking for snapshots after 07:00 ET:")
    cutoff = pd.Timestamp("2026-01-06 07:00:00", tz="US/Eastern").tz_convert("UTC").value
    late_snaps = snapshots[snapshots["ts_event"] >= cutoff]
    if not late_snaps.empty:
        print(f"Found {late_snaps['ts_event'].nunique()} snapshot times after 07:00 ET.")
        first_late = late_snaps["ts_event"].min()
        print(f"First late snapshot: {pd.to_datetime(first_late, unit='ns').tz_localize('UTC').tz_convert('US/Eastern')}")
    else:
        print("No snapshots found after 07:00 ET.")

else:
    print("NO SNAPSHOTS FOUND IN FILE.")
