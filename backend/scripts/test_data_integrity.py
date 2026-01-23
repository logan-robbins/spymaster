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

def check_integrity(dt: str) -> None:
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    
    # 1. Resolve symbol (same logic as run_pipeline)
    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection map: {selection_path}")
    df_sel = pd.read_parquet(selection_path)
    row = df_sel.loc[df_sel["session_date"].astype(str) == dt]
    if row.empty:
        raise ValueError(f"No selection entry for {dt}")
    symbol = str(row.iloc[0]["selected_symbol"]).strip()
    print(f"Checking integrity for {symbol} on {dt}")

    # 2. Load Datasets
    snap_ref = partition_ref(cfg, "silver.future_mbo.book_snapshot_1s", symbol, dt)
    wall_ref = partition_ref(cfg, "silver.future_mbo.wall_surface_1s", symbol, dt)
    radar_ref = partition_ref(cfg, "silver.future_mbo.radar_vacuum_1s", symbol, dt)

    if not snap_ref.dir.exists():
        print("Snapshot partition missing. Run pipeline first.")
        sys.exit(1)
    if not radar_ref.dir.exists():
        print("Radar partition missing (or pipeline failed to write it).")
        sys.exit(1)

    print("Loading data...")
    df_snap = pd.read_parquet(snap_ref.dir)
    df_radar = pd.read_parquet(radar_ref.dir)
    
    # Optional wall load (might be huge)
    # df_wall = pd.read_parquet(wall_ref.dir) 

    # 3. Sort
    df_snap = df_snap.sort_values("window_start_ts_ns").reset_index(drop=True)
    df_radar = df_radar.sort_values("window_start_ts_ns").reset_index(drop=True)

    print(f"Snapshot rows: {len(df_snap)}")
    print(f"Radar rows: {len(df_radar)}")

    # 4. Snapshot Integrity
    # Best Bid < Best Ask when valid
    valid_snap = df_snap[df_snap["book_valid"]]
    print(f"Valid snapshot rows: {len(valid_snap)}")
    
    crossed = valid_snap[valid_snap["best_bid_price_int"] >= valid_snap["best_ask_price_int"]]
    if not crossed.empty and valid_snap["best_bid_price_int"].max() > 0:
        # It's possible to have 0 if book is empty but valid? No, usually valid means we have data.
        # But if book is cleared, prices might be 0.
        # Check non-zero crossed
        real_cross = crossed[(crossed["best_bid_price_int"] > 0) & (crossed["best_ask_price_int"] > 0)]
        if not real_cross.empty:
            print(f"FAIL: Found {len(real_cross)} rows with crossed book (Bid >= Ask).")
            print(real_cross[["window_start_ts_ns", "best_bid_price_int", "best_ask_price_int"]].head())
            sys.exit(1)

    # 5. Radar Alignment
    # Join on window_start_ts_ns
    merged = pd.merge(df_snap, df_radar, on="window_start_ts_ns", how="inner", suffixes=("_snap", "_radar"))
    print(f"Merged rows (Snap + Radar): {len(merged)}")

    # Check Spot Ref match
    # Snap has spot_ref_price_int, Radar has spot_ref_price_int
    mismatched_spot = merged[merged["spot_ref_price_int_snap"] != merged["spot_ref_price_int_radar"]]
    if not mismatched_spot.empty:
        print(f"FAIL: Found {len(mismatched_spot)} rows where spot_ref mismatches between Snapshot and Radar.")
        print(mismatched_spot[["window_start_ts_ns", "spot_ref_price_int_snap", "spot_ref_price_int_radar"]].head())
        sys.exit(1)

    # Check Radar Validity Logic
    # Radar should only be emitted if spot_ref > 0.
    # Check if there are radar rows with spot_ref_int == 0
    zero_radar = df_radar[df_radar["spot_ref_price_int"] == 0]
    if not zero_radar.empty:
         print(f"FAIL: Found {len(zero_radar)} radar rows with spot_ref=0. Radar should not emit for 0 spot.")
         sys.exit(1)

    # 6. Check Vacuum Features Range
    # vacuum_score is not in radar (that's vacuum_surface), but radar has base features.
    # Check for NaNs in radar features
    if df_radar.isnull().any().any():
        print("WARNING: NaNs found in radar dataframe.")
        print(df_radar.columns[df_radar.isnull().any()].tolist())
        # Not necessarily fatal, but suspicious for computed features.

    print("PASS: Integrity checks passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", default="2026-01-06")
    args = parser.parse_args()
    check_integrity(args.dt)
