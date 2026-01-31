from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.mbo_contract_day_selector import load_selection
from src.serving.forecast_calibration import (
    HORIZONS,
    evaluate_forecasts_across_days,
    filter_session,
    forecast_day,
    prepare_futures,
    prepare_options,
)

FUT_KEY = "gold.future_mbo.physics_surface_1s"
OPT_KEY = "gold.future_option_mbo.physics_surface_1s"


def _load_params(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Physics params not found: {path}")
    return json.loads(path.read_text())


def _load_day_frames(cfg, symbol: str, session_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fut_ref = partition_ref(cfg, FUT_KEY, symbol, session_date)
    opt_ref = partition_ref(cfg, OPT_KEY, "ES", session_date)
    if not (is_partition_complete(fut_ref) and is_partition_complete(opt_ref)):
        raise FileNotFoundError(f"Missing gold partitions for {session_date} ({symbol})")
    df_fut = read_partition(fut_ref)
    df_opt = read_partition(opt_ref)
    df_fut = prepare_futures(filter_session(df_fut, session_date))
    df_opt = prepare_options(filter_session(df_opt, session_date))
    return df_fut, df_opt


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    params_path = repo_root / "data" / "physics" / "physics_beta_gamma.json"

    params = _load_params(params_path)
    beta = float(params.get("beta", 0.0))
    gamma = float(params.get("gamma", 0.0))
    holdout_dates = params.get("holdout_dates", [])
    if not holdout_dates:
        raise ValueError("No holdout_dates found in physics params.")

    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    selection = load_selection(selection_path)
    selection["session_date"] = selection["session_date"].astype(str)
    selection["selected_symbol"] = selection["selected_symbol"].astype(str)
    symbol_by_date = dict(
        zip(selection["session_date"].tolist(), selection["selected_symbol"].tolist())
    )

    eval_inputs = []
    for session_date in holdout_dates:
        symbol = symbol_by_date.get(session_date)
        if not symbol:
            raise ValueError(f"No selected_symbol for holdout date {session_date}")
        df_fut, df_opt = _load_day_frames(cfg, symbol, session_date)
        eval_inputs.append(forecast_day(df_fut, df_opt, beta=beta, gamma=gamma, horizons=HORIZONS))

    metrics = evaluate_forecasts_across_days(eval_inputs, horizons=HORIZONS)

    output_path = repo_root / "data" / "physics" / "physics_beta_gamma_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "beta": beta,
        "gamma": gamma,
        "holdout_dates": holdout_dates,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params_path": str(params_path),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote evaluation report: {output_path}")


if __name__ == "__main__":
    main()
