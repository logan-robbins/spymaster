
import os
from pathlib import Path
import time

DATA_ROOT = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/data/silver/features/es_pipeline")

def find_recent_files():
    print(f"Scanning {DATA_ROOT} for recent updates...")
    for version_dir in DATA_ROOT.iterdir():
        if version_dir.is_dir():
            print(f"Checking {version_dir.name}")
            aug20 = version_dir / "date=2025-08-20"
            if aug20.exists():
                parquet = aug20 / "signals.parquet"
                if parquet.exists():
                    mtime = parquet.stat().st_mtime
                    print(f"  Found: {parquet}")
                    print(f"  Modified: {time.ctime(mtime)}")
                    # Check if modified in last 5 mins
                    if time.time() - mtime < 300:
                        print("  !!! JUST UPDATED !!!")
                else:
                     print(f"  Found directory but no parquet: {aug20}")
            else:
                print(f"  Date dir not found: {aug20}")

if __name__ == "__main__":
    find_recent_files()
