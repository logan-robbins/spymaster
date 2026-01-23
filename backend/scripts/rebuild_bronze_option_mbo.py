from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.stages.bronze.future_option_mbo.ingest import BronzeIngestFutureOptionMbo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--dt", required=True)
    args = parser.parse_args()

    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")

    stage = BronzeIngestFutureOptionMbo()
    stage.run(cfg=cfg, repo_root=repo_root, symbol=args.symbol, dt=args.dt)
    print(f"bronze mbo rebuilt {args.symbol} {args.dt}")


if __name__ == "__main__":
    main()
