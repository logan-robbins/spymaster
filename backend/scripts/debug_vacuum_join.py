import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_eng.io import read_partition, partition_ref
from src.data_eng.config import load_config

def check_join():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    
    symbol = "ESH6"
    dt = "2026-01-06"
    
    print("Reading Snap...")
    snap_ref = partition_ref(cfg, "silver.future_mbo.book_snapshot_1s", symbol, dt)
    df_snap = read_partition(snap_ref)[["window_end_ts_ns"]]
    
    print("Reading Vacuum...")
    vac_ref = partition_ref(cfg, "silver.future_mbo.vacuum_surface_1s", symbol, dt)
    df_vac = read_partition(vac_ref)[["window_end_ts_ns"]]
    
    print(f"Snap Rows: {len(df_snap)}")
    print(f"Vacuum Rows: {len(df_vac)}")
    
    snap_ids = sorted(list(set(df_snap["window_end_ts_ns"].unique())))
    vac_ids = sorted(list(set(df_vac["window_end_ts_ns"].unique())))
    
    print(f"Snap First 5 IDs: {[str(x) for x in snap_ids[:5]]}")
    print(f"Vac  First 5 IDs: {[str(x) for x in vac_ids[:5]]}")
    
    common = set(snap_ids).intersection(set(vac_ids))
    print(f"Common Window IDs: {len(common)}")
    
    if len(common) < len(snap_ids):
        print("WARN: Missing vacuum coverage for some snapshots.")
        missing = list(snap_ids - vac_ids)[:5]
        print(f"Sample Missing IDs: {missing}")
    
    # Check first common ID grouping
    if common:
        wid = list(common)[0]
        print(f"Checking grouping for ID {wid}...")
        sample_grp = df_vac[df_vac["window_end_ts_ns"] == wid]
        print(f"Vacuum rows for {wid}: {len(sample_grp)}")

if __name__ == "__main__":
    check_join()
