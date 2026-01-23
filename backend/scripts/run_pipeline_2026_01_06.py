from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref
from src.data_eng.stages.gold.hud.build_physics_norm_calibration import GoldBuildHudPhysicsNormCalibration
from src.data_eng.stages.silver.future_mbo.compute_physics_bands_1s import SilverComputePhysicsBands1s
from src.data_eng.stages.silver.future_mbo.compute_radar_vacuum_1s import SilverComputeRadarVacuum1s
from src.data_eng.stages.silver.future_mbo.compute_snapshot_and_wall_1s import SilverComputeSnapshotAndWall1s
from src.data_eng.stages.silver.future_mbo.compute_vacuum_surface_1s import SilverComputeVacuumSurface1s
from src.data_eng.stages.silver.future_option.compute_statistics_clean import SilverComputeStatisticsClean
from src.data_eng.stages.silver.future_option_mbo.compute_gex_surface_1s import SilverComputeGexSurface1s


SILVER_GOLD_KEYS = [
#    "silver.future_mbo.book_snapshot_1s",
#    "silver.future_mbo.wall_surface_1s",
    "silver.future_mbo.vacuum_surface_1s",
    "silver.future_mbo.radar_vacuum_1s",
    "silver.future_mbo.physics_bands_1s",
#    "silver.future_option.statistics_clean",
    "silver.future_option_mbo.gex_surface_1s",
    "gold.hud.physics_norm_calibration",
]


def resolve_symbol(dt: str) -> str:
    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection map: {selection_path}")
    df = pd.read_parquet(selection_path)
    row = df.loc[df["session_date"].astype(str) == dt]
    if row.empty:
        raise ValueError(f"No selection entry for {dt}")
    symbol = str(row.iloc[0]["selected_symbol"]).strip()
    if not symbol:
        raise ValueError(f"Empty selected_symbol for {dt}")
    return symbol


def clear_partition(cfg, dataset_key: str, symbol: str, dt: str) -> None:
    ref = partition_ref(cfg, dataset_key, symbol, dt)
    if ref.dir.exists():
        shutil.rmtree(ref.dir)
        print(f"cleared {dataset_key} {symbol} {dt}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", default="2026-01-06")
    args = parser.parse_args()

    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")

    symbol = resolve_symbol(args.dt)
    print(f"using symbol {symbol} for dt {args.dt}", flush=True)

    for key in SILVER_GOLD_KEYS:
        clear_partition(cfg, key, symbol, args.dt)

    stages = [
        SilverComputeSnapshotAndWall1s(),
        SilverComputeStatisticsClean(),
        SilverComputeGexSurface1s(),
        GoldBuildHudPhysicsNormCalibration(),
        SilverComputeVacuumSurface1s(),
        SilverComputeRadarVacuum1s(),
        SilverComputePhysicsBands1s(),
    ]

    for stage in stages:
        print(f"stage_start {stage.name}", flush=True)
        stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=args.dt)
        print(f"stage_end {stage.name}", flush=True)


if __name__ == "__main__":
    main()
