
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "src"))

from data_eng.config import load_config
from data_eng.io import read_partition, partition_ref

def verify_stage_1(cfg, symbol, dt):
    print(f"Verifying Stage 1 (Snapshot/Wall) for {symbol} {dt}...")
    try:
        ref = partition_ref(cfg, "silver.future_mbo.wall_surface_1s", symbol, dt)
        df = read_partition(ref)
        print(f"Loaded wall surface: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Missing wall surface: {e}")
        return False
        
    # Check Conservation of Depth
    # d_end = d_start + add - pull - fill
    # Allow small float error?
    cols = ["depth_start", "depth_end", "add_vol", "pull_vol", "fill_vol"]
    # Map from actual cols:
    # depth_qty_start -> depth_start
    # depth_qty_end -> depth_end
    # add_qty -> add_vol
    # pull_qty_total -> pull_vol
    # fill_qty -> fill_vol
    
    expected = df["depth_qty_end"]
    calculated = df["depth_qty_start"] + df["add_qty"] - df["pull_qty_total"] - df["fill_qty"]
    
    diff = (expected - calculated).fillna(0.0).abs()
    bad = diff > 1e-9
    if bad.any():
        print(f"FAILED: Conservation of Depth violated in {bad.sum()} rows")
        print(df[bad].head())
        # Inspect a failed row
        fail_idx = bad.idxmax()
        print("Fail Example:")
        print(df.loc[fail_idx][["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty_total", "fill_qty"]])
        return False
    else:
        print("PASSED: Conservation of Depth")
        
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", default="2026-01-06")
    parser.add_argument("--symbol", default="ESH6")
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    
    verify_stage_1(cfg, args.symbol, args.dt)

if __name__ == "__main__":
    main()
