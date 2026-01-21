
import pandas as pd
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_eng.stages.silver.future_mbo.compute_level_vacuum_5s import _load_p_ref

def run():
    dt = "2026-01-05"
    symbol = "ESH6"
    path = f"lake/bronze/source=databento/product_type=future_mbo/symbol={symbol}/table=mbo/dt={dt}"
    
    print(f"Loading {path}...")
    try:
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} rows.")
        
        # Calculate PM High
        p_ref = _load_p_ref(df, dt)
        print(f"PM High for {dt}: {p_ref}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
