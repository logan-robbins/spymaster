from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.stages.silver.future_mbo.mbo_batches import iter_mbo_batches

def main():
    dt = "2026-01-06"
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    symbol = "ESH6"
    
    print(f"Scanning MBO for flags on {dt}...")
    
    F_SNAPSHOT = 128
    F_LAST = 256
    
    found_snapshot = False
    found_last = False
    
    count = 0
    max_batches = 500
    
    unique_values = set()
    
    for batch in iter_mbo_batches(cfg, repo_root, symbol, dt):
        if batch.empty: continue
        
        flags = batch["flags"].unique()
        unique_values.update(flags)
        
        if (batch["flags"] & F_LAST).any():
            print(f"FOUND F_LAST in batch {count}!")
            found_last = True
            break
            
        count += 1
        if count % 50 == 0:
            print(f"Scanned {count} batches...")
        if count >= max_batches:
            break
            
    print(f"Unique flag values found: {sorted(list(unique_values))}")
    if not found_last:
        print("WARNING: No F_LAST found in scanned batches.")

if __name__ == "__main__":
    main()
