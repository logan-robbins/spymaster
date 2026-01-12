from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_eng.config import load_config
from src.data_eng.stages.gold.future_mbo.build_trigger_vectors import GoldBuildMboTriggerVectors
from src.data_eng.utils import expand_date_range


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gold trigger vector stage for selected dates.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--dates")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _load_dates(selection_path: Path, symbol: str, dates: list[str]) -> list[str]:
    df = pd.read_parquet(selection_path)
    df = df.loc[df["session_date"].isin(dates)]
    df = df.loc[df["selected_symbol"] == symbol]
    return sorted(df["session_date"].unique().tolist())


def _run_single_date(
    dt: str,
    symbol: str,
    selection_path: Path,
    level_id: str,
    config_path: Path,
) -> dict:
    os.environ["MBO_SELECTION_PATH"] = str(selection_path)
    os.environ["LEVEL_ID"] = level_id
    repo_root = Path.cwd()
    cfg = load_config(repo_root=repo_root, config_path=config_path)
    stage = GoldBuildMboTriggerVectors()
    stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt)
    return {"dt": dt, "status": "success"}


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT
    config_path = repo_root / "src/data_eng/config/datasets.yaml"
    selection_path = Path(args.selection_path)

    dates = expand_date_range(dates=args.dates, start_date=args.start_date, end_date=args.end_date)
    if not dates:
        raise ValueError("No dates provided")

    run_dates = _load_dates(selection_path, args.symbol, dates)
    if not run_dates:
        raise ValueError("No selection dates for symbol")

    print(f"Symbol:  {args.symbol}")
    print(f"Dates:   {len(run_dates)}")
    print(f"Workers: {args.workers}")
    print()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_single_date,
                dt,
                args.symbol,
                selection_path,
                args.level_id,
                config_path,
            ): dt
            for dt in run_dates
        }
        for future in as_completed(futures):
            result = future.result()
            print(f"✅ {result['dt']}")


if __name__ == "__main__":
    main()
