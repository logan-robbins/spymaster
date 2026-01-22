
import os
import sys
from pathlib import Path
import databento as db
from dotenv import load_dotenv

# Setup environment
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / '.env')

def fix_download():
    api_key = os.getenv("DATABENTO_API_KEY")
    client = db.Historical(key=api_key)
    
    date_str = "2026-01-06"
    date_compact = "20260106"
    
    # Comprehensive list to ensure we catch the 0DTE
    parents = ["ES.OPT", "EW.OPT", "EW1.OPT", "EW2.OPT", "EW3.OPT", "EW4.OPT"]
    for i in range(1, 6):
        parents.append(f"E{i}.OPT")
        for char in ['A', 'B', 'C', 'D', 'E']:
            parents.append(f"E{i}{char}.OPT")
            
    print(f"Downloading MBO for {date_str} with {len(parents)} potential parents...")
    
    base_raw = backend_dir / "lake" / "raw" / "source=databento"
    mbo_dir = base_raw / "product_type=future_option_mbo" / "symbol=ES" / "table=market_by_order_dbn"
    mbo_file = mbo_dir / f"glbx-mdp3-{date_compact}.mbo.dbn"
    
    mbo_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=parents,
            schema="mbo",
            start=date_str,
            end="2026-01-07",
            stype_in="parent",
            path=mbo_file
        )
        print("Download complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_download()
