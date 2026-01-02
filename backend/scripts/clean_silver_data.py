
import os
import shutil
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/data/silver/features/es_pipeline")
CUTOFF_DATE = datetime(2025, 9, 30).date()

def verify_data_coverage():
    versions = [v for v in DATA_ROOT.iterdir() if v.is_dir() and v.name.startswith("version=")]
    
    # Ranges
    train_start = datetime(2025, 6, 5).date()
    train_end = datetime(2025, 9, 10).date()
    val_start = datetime(2025, 9, 15).date()
    val_end = datetime(2025, 9, 30).date()
    
    for version_dir in versions:
        print(f"\nVerifying content of {version_dir.name}...")
        dates = []
        for d in version_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                try:
                    date_str = d.name.split("=")[1]
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                except:
                    pass
        
        dates.sort()
        train_count = sum(1 for d in dates if train_start <= d <= train_end)
        val_count = sum(1 for d in dates if val_start <= d <= val_end)
        
        print(f"  Train ({train_start} to {train_end}): Found {train_count} dates")
        print(f"  Val   ({val_start} to {val_end}): Found {val_count} dates")
        
        # Identify missing business days? (Approximate check if counts seem low)
        # Just printing range for now
        if dates:
            print(f"  Range Present: {min(dates)} to {max(dates)}")

if __name__ == "__main__":
    # clean_silver_data() # Already run
    verify_data_coverage()
