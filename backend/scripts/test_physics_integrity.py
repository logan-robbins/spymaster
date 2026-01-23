from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref

def check_physics(dt: str) -> None:
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    
    # 1. Resolve symbol
    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection map: {selection_path}")
    df_sel = pd.read_parquet(selection_path)
    row = df_sel.loc[df_sel["session_date"].astype(str) == dt]
    if row.empty:
        raise ValueError(f"No selection entry for {dt}")
    symbol = str(row.iloc[0]["selected_symbol"]).strip()
    print(f"Checking physics for {symbol} on {dt}")

    # 2. Load Datasets
    snap_ref = partition_ref(cfg, "silver.future_mbo.book_snapshot_1s", symbol, dt)
    vac_ref = partition_ref(cfg, "silver.future_mbo.vacuum_surface_1s", symbol, dt)
    bands_ref = partition_ref(cfg, "silver.future_mbo.physics_bands_1s", symbol, dt)

    if not snap_ref.dir.exists():
        print("Snapshot partition missing.")
        sys.exit(1)
    if not vac_ref.dir.exists():
        print("Vacuum partition missing.")
        sys.exit(1)
    if not bands_ref.dir.exists():
        print("Bands partition missing.")
        sys.exit(1)

    print("Loading data...")
    df_snap = pd.read_parquet(snap_ref.dir)
    df_vac = pd.read_parquet(vac_ref.dir)
    df_bands = pd.read_parquet(bands_ref.dir)

    # 3. Counts
    print(f"Snapshot rows: {len(df_snap)}")
    print(f"Vacuum rows: {len(df_vac)}")
    print(f"Bands rows: {len(df_bands)}")
    
    # Bands should match snapshot rows (1 row per window)
    if len(df_bands) != len(df_snap):
        print(f"WARNING: Bands count {len(df_bands)} != Snapshot count {len(df_snap)}")
        
    # 4. Vacuum Surface Checks
    # vacuum_score should be 0..1
    if not df_vac.empty:
        min_v = df_vac["vacuum_score"].min()
        max_v = df_vac["vacuum_score"].max()
        print(f"Vacuum Surface Score range: {min_v} - {max_v}")
        if min_v < 0.0 or max_v > 1.0 + 1e-9:
            print("FAIL: Vacuum score outside 0..1 range")
            sys.exit(1)
            
        # Check calibration effect - if all exactly 0 or 1, maybe suspicious but potential.
        
    # 5. Physics Bands Checks
    # above_score, below_score, vacuum_total_score should be 0..1
    if not df_bands.empty:
        cols = ["above_score", "below_score", "vacuum_total_score"]
        for col in cols:
            min_v = df_bands[col].min()
            max_v = df_bands[col].max()
            print(f"Bands {col} range: {min_v} - {max_v}")
            if min_v < 0.0 or max_v > 1.0 + 1e-9:
                print(f"FAIL: {col} outside 0..1 range")
                sys.exit(1)

    # 6. Spot Ref Consistency
    # Check join
    merged = pd.merge(df_snap, df_bands, on="window_start_ts_ns", how="inner", suffixes=("_snap", "_bands"))
    if not merged.empty:
        mismatched = merged[merged["spot_ref_price_int_snap"] != merged["spot_ref_price_int_bands"]]
        if not mismatched.empty:
            print(f"FAIL: Spot ref mismatch between Snapshot and Bands for {len(mismatched)} rows")
            sys.exit(1)

    print("PASS: Physics checks passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", default="2026-01-06")
    args = parser.parse_args()
    check_physics(args.dt)
