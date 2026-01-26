from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    write_partition,
)
from .book_engine import (
    RADAR_COLUMNS,
    SNAP_COLUMNS,
    WALL_COLUMNS,
    compute_futures_surfaces_1s_from_batches,
)
from .mbo_batches import first_hour_window_ns, iter_mbo_batches


class SilverComputeSnapshotAndWall1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_snapshot_and_wall_1s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output="silver.future_mbo.book_snapshot_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_key_snap = "silver.future_mbo.book_snapshot_1s"
        out_key_wall = "silver.future_mbo.wall_surface_1s"
        out_key_radar = "silver.future_mbo.radar_vacuum_1s"

        ref_snap = partition_ref(cfg, out_key_snap, symbol, dt)
        ref_wall = partition_ref(cfg, out_key_wall, symbol, dt)
        ref_radar = partition_ref(cfg, out_key_radar, symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_wall) and is_partition_complete(ref_radar):
            return

        # Warmup: Read enough history to catch the initial snapshot (starts at 05:00 ET)
        WARMUP_NS = 3600_000_000_000 * 6  # 6 hours (covering 05:00 start for 09:30 window)

        df_snap, df_wall, df_radar = compute_futures_surfaces_1s_from_batches(
            iter_mbo_batches(cfg, repo_root, symbol, dt, start_buffer_ns=WARMUP_NS)
        )

        # Filter out warmup data from output
        start_ns, _ = first_hour_window_ns(dt)
        df_snap = df_snap[df_snap["window_end_ts_ns"] >= start_ns]
        df_wall = df_wall[df_wall["window_end_ts_ns"] >= start_ns]
        df_radar = df_radar[df_radar["window_end_ts_ns"] >= start_ns]

        contract_snap_path = repo_root / cfg.dataset(out_key_snap).contract
        contract_wall_path = repo_root / cfg.dataset(out_key_wall).contract
        contract_radar_path = repo_root / cfg.dataset(out_key_radar).contract

        contract_snap = load_avro_contract(contract_snap_path)
        contract_wall = load_avro_contract(contract_wall_path)
        contract_radar = load_avro_contract(contract_radar_path)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_wall = enforce_contract(df_wall, contract_wall)
        df_radar = enforce_contract(df_radar, contract_radar)

        mbo_ref = partition_ref(cfg, "bronze.future_mbo.mbo", symbol, dt)
        lineage = [{"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)}]

        if not is_partition_complete(ref_snap):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_snap,
                symbol=symbol,
                dt=dt,
                df=df_snap,
                contract_path=contract_snap_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_wall):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_wall,
                symbol=symbol,
                dt=dt,
                df=df_wall,
                contract_path=contract_wall_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_radar):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_radar,
                symbol=symbol,
                dt=dt,
                df=df_radar,
                contract_path=contract_radar_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_snap, df_wall, df_radar = compute_futures_surfaces_1s_from_batches([df])

        if df_snap.empty:
            df_snap = pd.DataFrame(columns=SNAP_COLUMNS)
        if df_wall.empty:
            df_wall = pd.DataFrame(columns=WALL_COLUMNS)
        if df_radar.empty:
            df_radar = pd.DataFrame(columns=RADAR_COLUMNS)

        return df_snap, df_wall, df_radar
