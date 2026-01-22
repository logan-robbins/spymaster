
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import databento as db
from dotenv import load_dotenv

# Setup environment
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / '.env')


def is_third_friday(d: datetime) -> bool:
    # Check if Friday
    if d.weekday() != 4:
        return False
    # Check if 3rd occurrence
    # Day 15-21 is range for 3rd Friday
    return 15 <= d.day <= 21

def get_parents_for_date(d: datetime) -> List[str]:
    """
    Smart 0DTE Selection:
    - 3rd Friday: Standard (ES)
    - Other Fridays: Weekly (EW, EW1-4)
    - Mon-Thu: Daily (E1-E5, E1A-E5E)
    """
    if is_third_friday(d):
        print(f"  [Smart Filter] {d.date()} is 3rd Friday -> Standard (ES)")
        return ["ES.OPT"]
    
    if d.weekday() == 4: # Friday but not 3rd
        print(f"  [Smart Filter] {d.date()} is Friday -> Weekly (EW)")
        return ["EW.OPT", "EW1.OPT", "EW2.OPT", "EW3.OPT", "EW4.OPT"]
        
    # Mon-Thu -> Daily
    print(f"  [Smart Filter] {d.date()} is Mon-Thu -> Daily (E1-E5)")
    dailies = []
    # Generic Dailies
    for i in range(1, 6):
        dailies.append(f"E{i}.OPT")
        # Add permutations E1A..E5E just in case parents specific
        for char in ['A', 'B', 'C', 'D', 'E']:
             dailies.append(f"E{i}{char}.OPT")
    return dailies

def download_missing_options(start_date: str):
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("Error: DATABENTO_API_KEY not found.")
        sys.exit(1)

    client = db.Historical(key=api_key)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.now()
    
    current = start
    while current <= end:
        # Skip Saturday
        if current.weekday() == 5: 
            current += timedelta(days=1)
            continue
            
        date_str = current.strftime("%Y-%m-%d")
        date_compact = date_str.replace("-", "")
        
        # Determine Parents
        parents = get_parents_for_date(current)
        if not parents:
            print(f"Skipping {date_str} (No parents configured)")
            current += timedelta(days=1)
            continue

        # Paths
        base_raw = backend_dir / "lake" / "raw" / "source=databento"
        def_dir = base_raw / "dataset=definition"
        def_file = def_dir / f"glbx-mdp3-{date_compact}.definition.dbn"
        stats_dir = base_raw / "product_type=future_option_mbo" / "symbol=ES" / "table=statistics"
        stats_file = stats_dir / f"glbx-mdp3-{date_compact}.statistics.dbn"
        
        if def_file.exists() and stats_file.exists():
            print(f"Skipping {date_str} (Files exist)")
            current += timedelta(days=1)
            continue
            
        print(f"\nProcessing {date_str}...")
        end_date_str = (current + timedelta(days=1)).strftime("%Y-%m-%d") 
        
        # 1. Download Definitions
        if not def_file.exists():
            print(f"  Downloading Definitions -> {def_file}")
            def_dir.mkdir(parents=True, exist_ok=True)
            try:
                client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=parents,
                    schema="definition",
                    start=date_str,
                    end=end_date_str,
                    stype_in="parent",
                    path=def_file
                )
            except Exception as e:
                print(f"  Error downloading definition for {date_str}: {e}")
                current += timedelta(days=1)
                continue

        # 2. Download Statistics
        if not stats_file.exists():
            print(f"  Downloading Statistics -> {stats_file}")
            stats_dir.mkdir(parents=True, exist_ok=True)
            try:
                client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=parents,
                    schema="statistics",
                    start=date_str,
                    end=end_date_str,
                    stype_in="parent",
                    path=stats_file
                )
            except Exception as e:
                print(f"  Error downloading statistics for {date_str}: {e}")
        
        current += timedelta(days=1)

if __name__ == "__main__":
    download_missing_options("2025-12-21")
