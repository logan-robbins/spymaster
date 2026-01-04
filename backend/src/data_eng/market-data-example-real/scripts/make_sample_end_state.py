"""Generate a tiny end-to-end example partition under `lake/`.

This script:
1) creates a small Bronze MBP-10 partition (CSV)
2) runs the pipeline to generate Silver + Gold partitions

Run from the repository root:

    python scripts/make_sample_end_state.py --dt 2026-01-02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_data.config import load_config
from market_data.contracts import enforce_contract, load_avro_contract
from market_data.io import write_partition_csv
from market_data.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dt", required=True)
    return p.parse_args()


def build_bronze_sample_df(contract_fields: list[str], dt: str) -> pd.DataFrame:
    """Create a tiny MBP-10 dataset with a few rows inside and outside 1st-3h RTH."""

    # NY RTH open 09:30 = 14:30 UTC (standard time) for 2026-01-02
    ts_utc = pd.to_datetime(
        [
            "2026-01-02 14:30:00+00:00",  # 09:30 NY
            "2026-01-02 15:00:00+00:00",  # 10:00 NY
            "2026-01-02 16:29:00+00:00",  # 11:29 NY
            "2026-01-02 17:29:00+00:00",  # 12:29 NY
            "2026-01-02 17:30:00+00:00",  # 12:30 NY (excluded)
            "2026-01-02 18:00:00+00:00",  # 13:00 NY (excluded)
        ]
    )

    ts_event_ns = (ts_utc.view("int64")).astype("int64")
    ts_recv_ns = (ts_event_ns + 1_000).astype("int64")

    base_px = int(4500.25 * 1_000_000_000)  # 4500.25 in 1e-9 units

    rows = []
    for i in range(len(ts_event_ns)):
        row = {
            "ts_recv": int(ts_recv_ns[i]),
            "flags": 0,
            "ts_event": int(ts_event_ns[i]),
            "ts_in_delta": 1000,
            "rtype": 10,
            "sequence": i + 1,
            "publisher_id": 1,
            "instrument_id": 12345,
            "action": "Add",
            "side": "Bid",
            "depth": 0,
            "price": base_px,
            "size": 1,
        }

        for lvl in range(10):
            bid_px = base_px - int(lvl * 0.25 * 1_000_000_000)
            ask_px = base_px + int((lvl + 1) * 0.25 * 1_000_000_000)
            row[f"bid_px_{lvl:02d}"] = bid_px
            row[f"ask_px_{lvl:02d}"] = ask_px
            row[f"bid_sz_{lvl:02d}"] = 10 + lvl
            row[f"ask_sz_{lvl:02d}"] = 12 + lvl
            row[f"bid_ct_{lvl:02d}"] = 1 + lvl
            row[f"ask_ct_{lvl:02d}"] = 2 + lvl

        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder + validate to match the bronze contract
    df = df.loc[:, contract_fields]
    return df


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    cfg = load_config(repo_root, repo_root / "config/datasets.yaml")

    bronze_key = "bronze.futures.market_by_price_10"
    bronze_contract_path = repo_root / cfg.dataset(bronze_key).contract
    bronze_contract = load_avro_contract(bronze_contract_path)

    df_bronze = build_bronze_sample_df(bronze_contract.fields, args.dt)
    df_bronze = enforce_contract(df_bronze, bronze_contract)

    write_partition_csv(
        cfg=cfg,
        dataset_key=bronze_key,
        dt=args.dt,
        df=df_bronze,
        contract_path=bronze_contract_path,
        inputs=[],
        stage="bootstrap_bronze_sample",
    )

    for stage in build_pipeline():
        stage.run(cfg=cfg, repo_root=repo_root, dt=args.dt)

    print("Sample end state written under:")
    print(cfg.lake_root)


if __name__ == "__main__":
    main()
