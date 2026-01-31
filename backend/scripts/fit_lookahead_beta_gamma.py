from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data_eng.config import load_config
from src.data_eng.io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
)
from src.data_eng.mbo_contract_day_selector import load_selection
from src.serving.forecast_calibration import (
    HORIZONS,
    build_regression_stats,
    evaluate_forecasts_across_days,
    filter_session,
    forecast_day,
    map_coeffs_to_params,
    prepare_futures,
    prepare_options,
    RegressionStats,
)

FUT_KEY = "gold.future_mbo.physics_surface_1s"
OPT_KEY = "gold.future_option_mbo.physics_surface_1s"
TRAIN_DAYS = 20
HOLDOUT_DAYS = 5


def _select_available_days(
    cfg, repo_root: Path, selection_path: Path
) -> Tuple[List[str], Dict[str, str]]:
    selection = load_selection(selection_path)
    selection["session_date"] = selection["session_date"].astype(str)
    selection["selected_symbol"] = selection["selected_symbol"].astype(str)
    symbol_by_date = dict(
        zip(selection["session_date"].tolist(), selection["selected_symbol"].tolist())
    )
    available: list[str] = []
    for session_date in sorted(symbol_by_date.keys()):
        symbol = symbol_by_date[session_date]
        fut_ref = partition_ref(cfg, FUT_KEY, symbol, session_date)
        opt_ref = partition_ref(cfg, OPT_KEY, "ES", session_date)
        if not (is_partition_complete(fut_ref) and is_partition_complete(opt_ref)):
            continue
        available.append(session_date)
    return available, symbol_by_date


def _load_day_frames(
    cfg,
    symbol: str,
    session_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    fut_ref = partition_ref(cfg, FUT_KEY, symbol, session_date)
    opt_ref = partition_ref(cfg, OPT_KEY, "ES", session_date)
    df_fut = read_partition(fut_ref)
    df_opt = read_partition(opt_ref)

    df_fut = prepare_futures(filter_session(df_fut, session_date))
    df_opt = prepare_options(filter_session(df_opt, session_date))

    lineage = {
        "futures_manifest": read_manifest_hash(fut_ref),
        "options_manifest": read_manifest_hash(opt_ref),
    }
    return df_fut, df_opt, lineage


def _accumulate_stats(total: RegressionStats, stats: RegressionStats) -> RegressionStats:
    return RegressionStats(
        s11=total.s11 + stats.s11,
        s12=total.s12 + stats.s12,
        s22=total.s22 + stats.s22,
        t1=total.t1 + stats.t1,
        t2=total.t2 + stats.t2,
        count=total.count + stats.count,
    )


def _solve_coeffs(stats: RegressionStats) -> Tuple[float, float]:
    if stats.count == 0:
        raise ValueError("No regression samples available for calibration.")
    xtx = np.array([[stats.s11, stats.s12], [stats.s12, stats.s22]], dtype=float)
    xty = np.array([stats.t1, stats.t2], dtype=float)
    try:
        coeffs = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return float(coeffs[0]), float(coeffs[1])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"

    available_dates, symbol_by_date = _select_available_days(cfg, repo_root, selection_path)
    if len(available_dates) < TRAIN_DAYS:
        raise ValueError(
            f"Need {TRAIN_DAYS} trading days for calibration, found {len(available_dates)}."
        )

    selected_dates = available_dates[-TRAIN_DAYS:]
    train_dates = selected_dates[: TRAIN_DAYS - HOLDOUT_DAYS]
    holdout_dates = selected_dates[TRAIN_DAYS - HOLDOUT_DAYS :]

    total_stats = RegressionStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    lineage: list[Dict[str, str]] = []

    for session_date in train_dates:
        symbol = symbol_by_date[session_date]
        df_fut, df_opt, day_lineage = _load_day_frames(cfg, symbol, session_date)
        stats = build_regression_stats(df_fut, df_opt)
        total_stats = _accumulate_stats(total_stats, stats)
        lineage.append(
            {
                "session_date": session_date,
                "symbol": symbol,
                **day_lineage,
            }
        )

    a, b = _solve_coeffs(total_stats)
    beta, gamma = map_coeffs_to_params(a, b)

    eval_inputs = []
    for session_date in holdout_dates:
        symbol = symbol_by_date[session_date]
        df_fut, df_opt, day_lineage = _load_day_frames(cfg, symbol, session_date)
        eval_inputs.append(forecast_day(df_fut, df_opt, beta=beta, gamma=gamma, horizons=HORIZONS))
        lineage.append(
            {
                "session_date": session_date,
                "symbol": symbol,
                **day_lineage,
            }
        )

    metrics = evaluate_forecasts_across_days(eval_inputs, horizons=HORIZONS)

    output_dir = repo_root / "data" / "physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "physics_beta_gamma.json"

    payload = {
        "beta": beta,
        "gamma": gamma,
        "train_dates": train_dates,
        "holdout_dates": holdout_dates,
        "train_samples": total_stats.count,
        "metrics": metrics,
        "lineage": lineage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_path": str(selection_path),
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote physics params: {output_path}")


if __name__ == "__main__":
    main()
