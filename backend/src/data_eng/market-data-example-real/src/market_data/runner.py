from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the demo MBP-10 pipeline for a dt partition")
    p.add_argument("--dt", required=True, help="Partition date in YYYY-MM-DD format")
    p.add_argument(
        "--config",
        default="config/datasets.yaml",
        help="Path to the dataset config YAML (default: config/datasets.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    config_path = repo_root / args.config

    cfg = load_config(repo_root=repo_root, config_path=config_path)

    for stage in build_pipeline():
        stage.run(cfg=cfg, repo_root=repo_root, dt=args.dt)

    print("DONE")
    print(f"Silver: {cfg.lake_root}/silver/domain=futures/table=market_by_price_10_clean/dt={args.dt}")
    print(f"Gold:   {cfg.lake_root}/gold/product=futures/table=market_by_price_10_first3h/dt={args.dt}")


if __name__ == "__main__":
    main()
