from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_config
from ..contracts import enforce_contract, load_avro_contract
from ..io import is_partition_complete, partition_ref, read_partition
from ..utils import expand_date_range
from .query import TriggerVectorRetriever
from .trigger_engine import (
    COOLDOWN_WINDOWS,
    H_FIRE,
    K,
    K_RAW,
    LONG_CLASSES,
    MAX_WHIPSAW_RATE,
    MIN_GAP_WINDOWS,
    MIN_RESOLVE_RATE,
    P_CHOP_MAX,
    SHORT_CLASSES,
    STOP_TICKS,
    EpisodeGate,
    TriggerEngine,
    TriggerMetrics,
    TriggerThresholds,
    apply_fire_rule,
)

TRIGGER_DATASET = "gold.future_mbo.mbo_trigger_vectors"
VACUUM_DATASET = "silver.future_mbo.mbo_level_vacuum_5s"

P_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MARGIN_GRID = [0.10, 0.15, 0.20, 0.25]
HORIZONS = list(range(7))


def _load_dataset(repo_root: Path, dataset_key: str, symbol: str, dates: List[str]) -> pd.DataFrame:
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    contract_path = repo_root / cfg.dataset(dataset_key).contract
    contract = load_avro_contract(contract_path)

    frames = []
    for dt in dates:
        ref = partition_ref(cfg, dataset_key, symbol, dt)
        if not is_partition_complete(ref):
            raise FileNotFoundError(f"Missing partition: {dataset_key} dt={dt}")
        df = read_partition(ref)
        if len(df) > 0:
            df = enforce_contract(df, contract)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=contract.fields)
    return pd.concat(frames, ignore_index=True)


def _build_metric_cache(
    retriever: TriggerVectorRetriever,
    df_triggers: pd.DataFrame,
) -> Dict[int, TriggerMetrics]:
    engine = TriggerEngine(retriever=retriever, k=K, k_raw=K_RAW)
    cache: Dict[int, TriggerMetrics] = {}
    for row in df_triggers.itertuples(index=False):
        ts_end = int(getattr(row, "ts_end_ns"))
        vector = np.array(getattr(row, "vector"), dtype=np.float64)
        metrics = engine.score_vector(
            level_id=str(getattr(row, "level_id")),
            approach_dir=str(getattr(row, "approach_dir")),
            vector=vector,
            session_date=str(getattr(row, "session_date")),
            exclude_session_date=True,
        )
        cache[ts_end] = metrics
    return cache


def _evaluate_thresholds(
    df_vacuum: pd.DataFrame,
    df_triggers: pd.DataFrame,
    metrics_by_ts: Dict[int, TriggerMetrics],
    thresholds: TriggerThresholds,
) -> Dict[str, float]:
    df_vacuum = df_vacuum.sort_values("window_end_ts_ns").reset_index(drop=True)
    trigger_by_ts: Dict[int, pd.Series] = {}
    for row in df_triggers.itertuples(index=False):
        trigger_by_ts[int(getattr(row, "ts_end_ns"))] = row

    total_eligible = 0
    total_fires = 0
    correct_counts = {h: 0 for h in HORIZONS}
    chop_false = 0
    whipsaw_hit = 0
    stop_violation = 0
    resolve_bar1 = 0
    resolve_bar2 = 0

    gate = EpisodeGate(
        cooldown_windows=COOLDOWN_WINDOWS,
        min_gap_windows=MIN_GAP_WINDOWS,
    )

    for idx, vac in enumerate(df_vacuum.itertuples(index=False)):
        approach_dir = str(getattr(vac, "approach_dir"))
        ts_end = int(getattr(vac, "window_end_ts_ns"))
        episode_id, blocked = gate.step(idx, approach_dir)

        trigger = trigger_by_ts.get(ts_end)
        if trigger is None:
            continue
        total_eligible += 1

        metrics = metrics_by_ts.get(ts_end)
        if metrics is None:
            raise ValueError(f"Missing metrics for trigger window: {ts_end}")

        fire_flag, _ = apply_fire_rule(metrics, thresholds)
        if blocked:
            fire_flag = 0

        if fire_flag == 1:
            gate.register_fire(episode_id)
            total_fires += 1

            for h in HORIZONS:
                true_label = str(getattr(trigger, f"true_outcome_h{h}"))
                if true_label != "WHIPSAW" and metrics.c_top1 == true_label:
                    correct_counts[h] += 1

            true_h1 = str(getattr(trigger, "true_outcome_h1"))
            if true_h1 == "CHOP":
                chop_false += 1
            if true_h1 == "WHIPSAW":
                whipsaw_hit += 1

            if metrics.c_top1 in LONG_CLASSES:
                mae = float(getattr(trigger, "mae_before_upper_ticks"))
            elif metrics.c_top1 in SHORT_CLASSES:
                mae = float(getattr(trigger, "mae_before_lower_ticks"))
            else:
                mae = 0.0
            if isinstance(mae, float) and np.isnan(mae):
                mae = 0.0
            if mae > STOP_TICKS:
                stop_violation += 1

            offset = getattr(trigger, "first_hit_bar_offset")
            if offset is not None and not (isinstance(offset, float) and np.isnan(offset)):
                if int(offset) <= 1:
                    resolve_bar1 += 1
                if int(offset) <= 2:
                    resolve_bar2 += 1

    precision_h = {}
    coverage_h = {}
    for h in HORIZONS:
        precision_h[h] = (correct_counts[h] / total_fires) if total_fires else 0.0
        coverage_h[h] = (total_fires / total_eligible) if total_eligible else 0.0

    precision = precision_h[H_FIRE]
    fire_rate = coverage_h[H_FIRE]
    chop_false_rate = (chop_false / total_fires) if total_fires else 0.0
    whipsaw_hit_rate = (whipsaw_hit / total_fires) if total_fires else 0.0
    stop_violation_rate = (stop_violation / total_fires) if total_fires else 0.0
    resolve_by_bar1_rate = (resolve_bar1 / total_fires) if total_fires else 0.0
    resolve_by_bar2_rate = (resolve_bar2 / total_fires) if total_fires else 0.0

    result: Dict[str, float] = {
        "precision": precision,
        "fire_rate": fire_rate,
        "chop_false_rate": chop_false_rate,
        "whipsaw_hit_rate": whipsaw_hit_rate,
        "stop_violation_rate": stop_violation_rate,
        "resolve_by_bar1_rate": resolve_by_bar1_rate,
        "resolve_by_bar2_rate": resolve_by_bar2_rate,
    }
    for h in HORIZONS:
        result[f"precision_h{h}"] = precision_h[h]
        result[f"coverage_h{h}"] = coverage_h[h]
    return result


def _select_best(results: List[Dict[str, float]]) -> Dict[str, float]:
    filtered = []
    for row in results:
        if row["precision"] < 0.60:
            continue
        if row["chop_false_rate"] > 0.15:
            continue
        if row["stop_violation_rate"] > 0.20:
            continue
        if row["fire_rate"] < 0.01 or row["fire_rate"] > 0.20:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No threshold pair meets constraints")

    def score(r: Dict[str, float]) -> tuple[float, float, float]:
        return (
            r["fire_rate"] * r["precision"],
            r["precision"],
            r["resolve_by_bar1_rate"],
        )

    best = max(filtered, key=score)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune trigger thresholds.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--indices-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dates", required=True)
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")

    df_triggers = _load_dataset(args.repo_root, TRIGGER_DATASET, args.symbol, dates)
    if len(df_triggers) == 0:
        raise ValueError("No trigger vectors loaded from lake")

    df_vacuum = _load_dataset(args.repo_root, VACUUM_DATASET, args.symbol, dates)
    if len(df_vacuum) == 0:
        raise ValueError("No vacuum windows loaded from lake")

    retriever = TriggerVectorRetriever(args.indices_dir)
    metrics_by_ts = _build_metric_cache(retriever, df_triggers)

    results: List[Dict[str, float]] = []
    for p_min in P_GRID:
        for margin_min in MARGIN_GRID:
            thresholds = TriggerThresholds(
                p_min=p_min,
                margin_min=margin_min,
                p_chop_max=P_CHOP_MAX,
                stop_ticks=STOP_TICKS,
                min_resolve_rate=MIN_RESOLVE_RATE,
                max_whipsaw_rate=MAX_WHIPSAW_RATE,
            )
            metrics = _evaluate_thresholds(
                df_vacuum=df_vacuum,
                df_triggers=df_triggers,
                metrics_by_ts=metrics_by_ts,
                thresholds=thresholds,
            )
            metrics["p_min"] = p_min
            metrics["margin_min"] = margin_min
            results.append(metrics)

    best = _select_best(results)
    payload = {"best": best, "results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
