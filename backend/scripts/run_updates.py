from __future__ import annotations
import sys
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref
from src.data_eng.stages.silver.future_mbo.compute_snapshot_and_wall_1s import SilverComputeSnapshotAndWall1s
from src.data_eng.stages.silver.future_mbo.compute_radar_vacuum_1s import SilverComputeRadarVacuum1s

def clear_partition(cfg, dataset_key: str, symbol: str, dt: str) -> None:
    ref = partition_ref(cfg, dataset_key, symbol, dt)
    if ref.dir.exists():
        shutil.rmtree(ref.dir)
        print(f"cleared {dataset_key} {symbol} {dt}")

def main():
    dt = "2026-01-06"
    print(f"Running updates for {dt}...")
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    symbol = "ESH6" 
    
    # 1. Clear partitions
    targets = [
        "silver.future_mbo.book_snapshot_1s", 
        "silver.future_mbo.wall_surface_1s",
        "silver.future_mbo.radar_vacuum_1s"
    ]
    for key in targets:
        clear_partition(cfg, key, symbol, dt)
        
    # 2. Run stages
    stages = [
        SilverComputeSnapshotAndWall1s(),
        SilverComputeRadarVacuum1s()
    ]
    
    for stage in stages:
        print(f"Running {stage.name}...")
        stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt)
        print(f"Finished {stage.name}")
        
    print("Updates done.")

if __name__ == "__main__":
    main()
