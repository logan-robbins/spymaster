from __future__ import annotations
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.stages.silver.future_mbo.compute_radar_vacuum_1s import SilverComputeRadarVacuum1s

def main():
    dt = "2026-01-06"
    print(f"Running Radar Vacuum 1s for {dt}...")
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    # Resolve symbol hack (hardcoded ESH6 based on logs)
    symbol = "ESH6" 
    
    stage = SilverComputeRadarVacuum1s()
    stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt)
    print("Radar Vacuum 1s Done.")

if __name__ == "__main__":
    main()
