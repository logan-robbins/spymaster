from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.serving.forecast_math import (
    build_window_fields,
    force_and_damping,
    run_forecast,
    tick_index_from_price,
)

HORIZONS = (1, 2, 5, 10, 20, 30)

SESSION_START_HOUR = 9
SESSION_START_MINUTE = 30
SESSION_END_HOUR = 12
SESSION_END_MINUTE = 30


@dataclass(frozen=True)
class RegressionStats:
    s11: float
    s12: float
    s22: float
    t1: float
    t2: float
    count: int


def map_coeffs_to_params(a: float, b: float) -> Tuple[float, float]:
    gamma = 1.0 - float(np.clip(a, 0.0, 1.0))
    beta = float(max(0.0, b))
    return beta, gamma


def session_window_ns(date: str) -> Tuple[int, int]:
    start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
        hours=SESSION_START_HOUR, minutes=SESSION_START_MINUTE
    )
    end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
        hours=SESSION_END_HOUR, minutes=SESSION_END_MINUTE
    )
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


def filter_session(df: pd.DataFrame, date: str) -> pd.DataFrame:
    if df.empty:
        return df
    start_ns, end_ns = session_window_ns(date)
    return df.loc[
        (df["window_end_ts_ns"] >= start_ns) & (df["window_end_ts_ns"] < end_ns)
    ].copy()


def prepare_futures(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["pressure_grad", "Omega", "u_near", "u_p_slow"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
    return df


def prepare_options(df: pd.DataFrame) -> pd.DataFrame:
    if "Omega_opt" not in df.columns:
        df["Omega_opt"] = 0.0
    return df


def iter_window_fields(
    df_fut: pd.DataFrame, df_opt: pd.DataFrame
) -> Iterable[Tuple[int, float, object]]:
    if df_fut.empty:
        return []
    options_by_window = {k: g for k, g in df_opt.groupby("window_end_ts_ns")}
    window_ids = sorted(df_fut["window_end_ts_ns"].unique())
    for window_id in window_ids:
        fut_group = df_fut.loc[df_fut["window_end_ts_ns"] == window_id]
        if fut_group.empty:
            continue
        opt_group = options_by_window.get(window_id, pd.DataFrame())
        fields = build_window_fields(fut_group, opt_group)
        spot_ref = fut_group["spot_ref_price_int"].iloc[0]
        spot_ticks = tick_index_from_price(spot_ref)
        yield int(window_id), float(spot_ticks), fields


def build_regression_stats(df_fut: pd.DataFrame, df_opt: pd.DataFrame) -> RegressionStats:
    spot_ticks: list[float] = []
    x2_values: list[float] = []

    for _, spot_tick, fields in iter_window_fields(df_fut, df_opt):
        spot_ticks.append(spot_tick)
        force_0, nu_0 = force_and_damping(fields, delta=0.0)
        x2_values.append(force_0 / (1.0 + nu_0))

    if len(spot_ticks) < 3:
        return RegressionStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    spot_arr = np.asarray(spot_ticks, dtype=float)
    v = np.diff(spot_arr)
    x1 = v[:-1]
    y = v[1:]
    x2 = np.asarray(x2_values, dtype=float)[1:-1]

    s11 = float(np.sum(x1 * x1))
    s12 = float(np.sum(x1 * x2))
    s22 = float(np.sum(x2 * x2))
    t1 = float(np.sum(x1 * y))
    t2 = float(np.sum(x2 * y))
    return RegressionStats(s11, s12, s22, t1, t2, int(len(y)))


def forecast_day(
    df_fut: pd.DataFrame,
    df_opt: pd.DataFrame,
    beta: float,
    gamma: float,
    horizons: Iterable[int] = HORIZONS,
) -> Tuple[np.ndarray, Dict[int, List[int]], Dict[int, List[float]]]:
    horizons = tuple(sorted(set(horizons)))
    preds_by_h = {h: [] for h in horizons}
    conf_by_h = {h: [] for h in horizons}
    spot_ticks: list[float] = []

    last_spot: float | None = None
    for _, spot_tick, fields in iter_window_fields(df_fut, df_opt):
        v0 = 0.0 if last_spot is None else (spot_tick - last_spot)
        last_spot = spot_tick
        spot_ticks.append(spot_tick)

        forecast_rows, _ = run_forecast(
            fields=fields,
            spot_ticks=spot_tick,
            v0=v0,
            beta=beta,
            gamma=gamma,
            horizon_s=max(horizons),
        )
        for row in forecast_rows:
            h = int(row["horizon_s"])
            if h in preds_by_h:
                preds_by_h[h].append(int(row["predicted_tick_delta"]))
                conf_by_h[h].append(float(row["confidence"]))

    return np.asarray(spot_ticks, dtype=float), preds_by_h, conf_by_h


def confidence_calibration(
    confidences: Iterable[float], hits: Iterable[bool], n_bins: int = 10
) -> Dict[str, object]:
    conf_arr = np.asarray(list(confidences), dtype=float)
    hit_arr = np.asarray(list(hits), dtype=float)
    if conf_arr.size == 0:
        return {"bins": [], "monotonic": False}

    order = np.argsort(conf_arr)
    conf_sorted = conf_arr[order]
    hit_sorted = hit_arr[order]

    n_bins = int(min(n_bins, conf_sorted.size))
    indices = np.array_split(np.arange(conf_sorted.size), n_bins)

    bins: list[Dict[str, float]] = []
    hit_rates: list[float] = []
    for idx in indices:
        bin_conf = conf_sorted[idx]
        bin_hit = hit_sorted[idx]
        hit_rate = float(np.mean(bin_hit)) if bin_hit.size else 0.0
        bins.append(
            {
                "min_conf": float(np.min(bin_conf)),
                "max_conf": float(np.max(bin_conf)),
                "mean_conf": float(np.mean(bin_conf)),
                "hit_rate": hit_rate,
                "count": int(bin_hit.size),
            }
        )
        hit_rates.append(hit_rate)

    monotonic = all(
        hit_rates[i] <= hit_rates[i + 1] + 1e-12 for i in range(len(hit_rates) - 1)
    )
    return {"bins": bins, "monotonic": monotonic}


def evaluate_forecasts(
    spot_ticks: np.ndarray,
    preds_by_h: Dict[int, List[int]],
    conf_by_h: Dict[int, List[float]],
    horizons: Iterable[int] = HORIZONS,
) -> Dict[str, object]:
    horizons = tuple(sorted(set(horizons)))
    metrics: Dict[str, object] = {"horizons": {}}
    conf_all: list[float] = []
    hit_all: list[bool] = []

    for h in horizons:
        preds = np.asarray(preds_by_h.get(h, []), dtype=float)
        confs = np.asarray(conf_by_h.get(h, []), dtype=float)
        if spot_ticks.size <= h or preds.size == 0:
            metrics["horizons"][str(h)] = {"samples": 0}
            continue
        usable = min(preds.size, spot_ticks.size - h)
        preds = preds[:usable]
        confs = confs[:usable]
        actual = spot_ticks[h : h + usable] - spot_ticks[:usable]

        hits = np.sign(preds) == np.sign(actual)
        mae = float(np.mean(np.abs(preds - actual)))
        hit_rate = float(np.mean(hits))

        metrics["horizons"][str(h)] = {
            "samples": int(usable),
            "sign_hit_rate": hit_rate,
            "mae_ticks": mae,
        }

        conf_all.extend(confs.tolist())
        hit_all.extend(hits.tolist())

    metrics["confidence"] = confidence_calibration(conf_all, hit_all, n_bins=10)
    return metrics


def evaluate_forecasts_across_days(
    day_results: Iterable[Tuple[np.ndarray, Dict[int, List[int]], Dict[int, List[float]]]],
    horizons: Iterable[int] = HORIZONS,
) -> Dict[str, object]:
    horizons = tuple(sorted(set(horizons)))
    preds_all: Dict[int, list[float]] = {h: [] for h in horizons}
    actual_all: Dict[int, list[float]] = {h: [] for h in horizons}
    conf_all: list[float] = []
    hit_all: list[bool] = []

    for spot_ticks, preds_by_h, conf_by_h in day_results:
        for h in horizons:
            preds = np.asarray(preds_by_h.get(h, []), dtype=float)
            confs = np.asarray(conf_by_h.get(h, []), dtype=float)
            if spot_ticks.size <= h or preds.size == 0:
                continue
            usable = min(preds.size, spot_ticks.size - h)
            preds = preds[:usable]
            confs = confs[:usable]
            actual = spot_ticks[h : h + usable] - spot_ticks[:usable]
            hits = np.sign(preds) == np.sign(actual)

            preds_all[h].extend(preds.tolist())
            actual_all[h].extend(actual.tolist())
            conf_all.extend(confs.tolist())
            hit_all.extend(hits.tolist())

    metrics: Dict[str, object] = {"horizons": {}}
    for h in horizons:
        preds = np.asarray(preds_all[h], dtype=float)
        actual = np.asarray(actual_all[h], dtype=float)
        if preds.size == 0:
            metrics["horizons"][str(h)] = {"samples": 0}
            continue
        hits = np.sign(preds) == np.sign(actual)
        metrics["horizons"][str(h)] = {
            "samples": int(preds.size),
            "sign_hit_rate": float(np.mean(hits)),
            "mae_ticks": float(np.mean(np.abs(preds - actual))),
        }

    metrics["confidence"] = confidence_calibration(conf_all, hit_all, n_bins=10)
    return metrics
