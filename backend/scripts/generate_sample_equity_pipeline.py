from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from src.data_eng.config import load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
import shutil

from src.data_eng.io import partition_ref, write_partition
from src.data_eng.utils import session_window_ns
from src.data_eng.stages.silver.equity_mbo.compute_snapshot_and_wall_1s import (
    SilverComputeEquitySnapshotAndWall1s,
)
from src.data_eng.stages.silver.equity_mbo.compute_vacuum_surface_1s import (
    SilverComputeEquityVacuumSurface1s,
)
from src.data_eng.stages.silver.equity_mbo.compute_radar_vacuum_1s import (
    SilverComputeEquityRadarVacuum1s,
)
from src.data_eng.stages.silver.equity_mbo.compute_physics_bands_1s import (
    SilverComputeEquityPhysicsBands1s,
)
from src.data_eng.stages.gold.equity_mbo.build_physics_norm_calibration import (
    GoldBuildEquityPhysicsNormCalibration,
)

PRICE_SCALE = 1e-9
RTYPE_MBO = 160
FLAGS_LAST = 128      # bit 7 (0x80): last record in event for instrument_id
FLAGS_SNAPSHOT = 32   # bit 5 (0x20): sourced from replay/snapshot server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample equity pipeline data.")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--dt", default="2026-01-02")
    parser.add_argument("--prev-days", type=int, default=1)
    return parser.parse_args()


def _ts_ns(date_str: str, time_str: str) -> int:
    ts = pd.Timestamp(f"{date_str} {time_str}", tz="Etc/GMT+5")
    return int(ts.tz_convert("UTC").value)


def _price_int(price: float) -> int:
    return int(round(price / PRICE_SCALE))


def _equity_mbo_rows(symbol: str, dt: str) -> pd.DataFrame:
    base = _ts_ns(dt, "03:05:00")  # XNAS session starts ~03:05 ET with Clear
    day_shift = int(pd.Timestamp(dt).dayofyear % 3)
    size_shift = 20 * day_shift
    bid_px = _price_int(399.98 + 0.01 * day_shift)
    ask_px = _price_int(400.02 + 0.01 * day_shift)
    trade_px = _price_int(400.00 + 0.01 * day_shift)
    bid_size = 180 + size_shift
    ask_size = 200 + size_shift

    rows = [
        {
            "ts_recv": base + 100_000_100,
            "size": 0,
            "ts_event": base + 100_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 0,
            "publisher_id": 1,
            "flags": FLAGS_SNAPSHOT,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "R",
            "sequence": 1,
            "side": "N",
            "symbol": symbol,
            "price": 0,
        },
        {
            "ts_recv": base + 200_000_100,
            "size": ask_size,
            "ts_event": base + 200_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 10,
            "publisher_id": 1,
            "flags": FLAGS_LAST | FLAGS_SNAPSHOT,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "A",
            "sequence": 2,
            "side": "A",
            "symbol": symbol,
            "price": ask_px,
        },
        {
            "ts_recv": base + 300_000_100,
            "size": bid_size,
            "ts_event": base + 300_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 11,
            "publisher_id": 1,
            "flags": 0,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "A",
            "sequence": 3,
            "side": "B",
            "symbol": symbol,
            "price": bid_px,
        },
        {
            "ts_recv": base + 400_000_100,
            "size": 50,
            "ts_event": base + 400_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 0,
            "publisher_id": 1,
            "flags": 0,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "T",
            "sequence": 4,
            "side": "B",
            "symbol": symbol,
            "price": trade_px,
        },
        {
            "ts_recv": base + 1_200_000_100,
            "size": bid_size - 60,
            "ts_event": base + 1_200_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 11,
            "publisher_id": 1,
            "flags": 0,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "M",
            "sequence": 5,
            "side": "B",
            "symbol": symbol,
            "price": bid_px,
        },
        {
            "ts_recv": base + 1_400_000_100,
            "size": 60,
            "ts_event": base + 1_400_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 0,
            "publisher_id": 1,
            "flags": 0,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "T",
            "sequence": 6,
            "side": "A",
            "symbol": symbol,
            "price": trade_px + _price_int(0.01),
        },
        {
            "ts_recv": base + 2_200_000_100,
            "size": bid_size + 40,
            "ts_event": base + 2_200_000_000,
            "channel_id": 0,
            "rtype": RTYPE_MBO,
            "order_id": 11,
            "publisher_id": 1,
            "flags": 0,
            "instrument_id": 1001,
            "ts_in_delta": 0,
            "action": "M",
            "sequence": 7,
            "side": "B",
            "symbol": symbol,
            "price": bid_px,
        },
    ]
    return pd.DataFrame(rows)


def _equity_option_cmbp_rows(symbol: str, dt: str) -> pd.DataFrame:
    base = _ts_ns(dt, "03:10:00")  # Options slightly after equity session start
    bid_px = _price_int(2.50)
    ask_px = _price_int(2.60)

    rows = [
        {
            "ts_recv": base + 100_000_100,
            "flags": 0,
            "ts_event": base + 100_000_000,
            "ts_in_delta": 0,
            "rtype": 0,
            "bid_px_00": bid_px,
            "publisher_id": 1,
            "ask_px_00": ask_px,
            "instrument_id": 2001,
            "bid_sz_00": 100,
            "action": "A",
            "ask_sz_00": 120,
            "side": "N",
            "bid_pb_00": 1,
            "price": 0,
            "ask_pb_00": 1,
            "size": 0,
            "symbol": symbol,
        },
        {
            "ts_recv": base + 200_000_100,
            "flags": 0,
            "ts_event": base + 200_000_000,
            "ts_in_delta": 0,
            "rtype": 0,
            "bid_px_00": bid_px,
            "publisher_id": 1,
            "ask_px_00": ask_px + _price_int(0.01),
            "instrument_id": 2001,
            "bid_sz_00": 110,
            "action": "A",
            "ask_sz_00": 130,
            "side": "N",
            "bid_pb_00": 1,
            "price": 0,
            "ask_pb_00": 1,
            "size": 0,
            "symbol": symbol,
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(backend_dir, backend_dir / "src/data_eng/config/datasets.yaml")

    base_date = datetime.strptime(args.dt, "%Y-%m-%d").date()
    dates = [(base_date - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(args.prev_days, -1, -1)]

    mbo_contract_path = backend_dir / cfg.dataset("bronze.equity_mbo.mbo").contract
    mbo_contract = load_avro_contract(mbo_contract_path)

    cmbp_contract_path = backend_dir / cfg.dataset("bronze.equity_option_cmbp_1.cmbp_1").contract
    cmbp_contract = load_avro_contract(cmbp_contract_path)

    for dt in dates:
        df_mbo = enforce_contract(_equity_mbo_rows(args.symbol, dt), mbo_contract)
        write_partition(
            cfg=cfg,
            dataset_key="bronze.equity_mbo.mbo",
            symbol=args.symbol,
            dt=dt,
            df=df_mbo,
            contract_path=mbo_contract_path,
            inputs=[],
            stage="generate_sample_equity_pipeline",
        )

        df_cmbp = enforce_contract(_equity_option_cmbp_rows(args.symbol, dt), cmbp_contract)
        write_partition(
            cfg=cfg,
            dataset_key="bronze.equity_option_cmbp_1.cmbp_1",
            symbol=args.symbol,
            dt=dt,
            df=df_cmbp,
            contract_path=cmbp_contract_path,
            inputs=[],
            stage="generate_sample_equity_pipeline",
        )

        _clear_partition(cfg, "silver.equity_mbo.book_snapshot_1s", args.symbol, dt)
        _clear_partition(cfg, "silver.equity_mbo.wall_surface_1s", args.symbol, dt)
        _clear_partition(cfg, "silver.equity_mbo.vacuum_surface_1s", args.symbol, dt)
        _clear_partition(cfg, "silver.equity_mbo.radar_vacuum_1s", args.symbol, dt)
        _clear_partition(cfg, "silver.equity_mbo.physics_bands_1s", args.symbol, dt)
        _clear_partition(cfg, "gold.equity_mbo.physics_norm_calibration", args.symbol, dt)

    stage_snap = SilverComputeEquitySnapshotAndWall1s()
    stage_cal = GoldBuildEquityPhysicsNormCalibration()
    stage_vac = SilverComputeEquityVacuumSurface1s()
    stage_radar = SilverComputeEquityRadarVacuum1s()
    stage_bands = SilverComputeEquityPhysicsBands1s()

    for dt in dates:
        stage_snap.run(cfg=cfg, repo_root=backend_dir, symbol=args.symbol, dt=dt)

    stage_cal.run(cfg=cfg, repo_root=backend_dir, symbol=args.symbol, dt=dates[-1])
    stage_vac.run(cfg=cfg, repo_root=backend_dir, symbol=args.symbol, dt=dates[-1])
    stage_radar.run(cfg=cfg, repo_root=backend_dir, symbol=args.symbol, dt=dates[-1])
    stage_bands.run(cfg=cfg, repo_root=backend_dir, symbol=args.symbol, dt=dates[-1])

    start_ns, end_ns = session_window_ns(dates[-1])
    if end_ns <= start_ns:
        raise ValueError("Invalid session window")

    print(f"Sample equity pipeline complete for {args.symbol} {dates[-1]}")
    return 0


def _clear_partition(cfg, dataset_key: str, symbol: str, dt: str) -> None:
    ref = partition_ref(cfg, dataset_key, symbol, dt)
    if ref.dir.exists():
        shutil.rmtree(ref.dir)


if __name__ == "__main__":
    raise SystemExit(main())
