from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the data pipeline for a specific product_type, symbol, and dt partition"
    )
    p.add_argument(
        "--product-type",
        required=True,
        choices=["future", "future_option", "equity", "equity_option"],
        help="Product type (future, future_option, equity, equity_option)",
    )
    p.add_argument(
        "--symbol",
        required=True,
        help="Symbol to process (e.g., ES, NQ, ESZ24_C6000)",
    )
    p.add_argument(
        "--dt",
        required=True,
        help="Partition date in YYYY-MM-DD format",
    )
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

    for stage in build_pipeline(args.product_type):
        stage.run(cfg=cfg, repo_root=repo_root, symbol=args.symbol, dt=args.dt)

    print("DONE")
    print(f"Product Type: {args.product_type}")
    print(f"Symbol:       {args.symbol}")
    print(f"Date:         {args.dt}")
    print(f"Lake Root:    {cfg.lake_root}")


if __name__ == "__main__":
    main()
