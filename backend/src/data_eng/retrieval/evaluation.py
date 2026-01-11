from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..contracts import enforce_contract, load_avro_contract
from ..io import is_partition_complete, partition_ref, read_partition
from ..utils import expand_date_range
from .query import TriggerVectorRetriever

DATASET_KEY = "gold.future_mbo.mbo_trigger_vectors"


def evaluate_retrieval(
    df: pd.DataFrame,
    retriever: TriggerVectorRetriever,
    k: int = 20,
    horizons: List[int] | None = None,
) -> Dict[str, object]:
    if horizons is None:
        horizons = list(range(7))

    counters = {
        h: {
            "total": 0,
            "correct": 0,
            "hit_break": 0,
            "hit_reject": 0,
            "false_break": 0,
            "false_reject": 0,
            "overtrade": 0,
            "offsets": Counter(),
        }
        for h in horizons
    }

    for row in df.itertuples(index=False):
        level_id = getattr(row, "level_id")
        approach_dir = getattr(row, "approach_dir")
        vector = np.array(getattr(row, "vector"), dtype=np.float64)
        vector_id = getattr(row, "vector_id")

        neighbors = retriever.find_similar(level_id, approach_dir, vector, k=k + 1)
        filtered = [n for n in neighbors if n.metadata.get("vector_id") != vector_id]
        if not filtered:
            continue

        break_label, reject_label = _labels_for_approach(approach_dir)

        for h in horizons:
            true_label = getattr(row, f"true_outcome_h{h}")
            dist = retriever.outcome_distribution(filtered, approach_dir, horizon=h)
            pred = dist.predicted

            counters[h]["total"] += 1
            if true_label == "WHIPSAW":
                continue

            if pred == true_label:
                counters[h]["correct"] += 1
                if true_label in (break_label, reject_label):
                    offset = getattr(row, "first_hit_bar_offset")
                    if offset is not None:
                        counters[h]["offsets"][int(offset)] += 1

            if pred == break_label:
                if true_label == break_label:
                    counters[h]["hit_break"] += 1
                else:
                    counters[h]["false_break"] += 1
            elif pred == reject_label:
                if true_label == reject_label:
                    counters[h]["hit_reject"] += 1
                else:
                    counters[h]["false_reject"] += 1

            if pred in (break_label, reject_label) and true_label == "CHOP":
                counters[h]["overtrade"] += 1

    results: Dict[str, object] = {}
    for h in horizons:
        total = counters[h]["total"]
        if total == 0:
            results[f"acc_H{h}"] = 0.0
            results[f"hit_break_H{h}"] = 0.0
            results[f"hit_reject_H{h}"] = 0.0
            results[f"false_break_H{h}"] = 0.0
            results[f"false_reject_H{h}"] = 0.0
            results[f"overtrade_H{h}"] = 0.0
            results[f"first_hit_bar_offset_H{h}"] = {}
            continue

        results[f"acc_H{h}"] = counters[h]["correct"] / total
        results[f"hit_break_H{h}"] = counters[h]["hit_break"] / total
        results[f"hit_reject_H{h}"] = counters[h]["hit_reject"] / total
        results[f"false_break_H{h}"] = counters[h]["false_break"] / total
        results[f"false_reject_H{h}"] = counters[h]["false_reject"] / total
        results[f"overtrade_H{h}"] = counters[h]["overtrade"] / total
        results[f"first_hit_bar_offset_H{h}"] = dict(counters[h]["offsets"])

    return results


def _labels_for_approach(approach_dir: str) -> Tuple[str, str]:
    if approach_dir == "approach_up":
        return "BREAK_UP", "REJECT_DOWN"
    if approach_dir == "approach_down":
        return "BREAK_DOWN", "REJECT_UP"
    raise ValueError(f"Unexpected approach_dir: {approach_dir}")


def _load_vectors(repo_root: Path, symbol: str, dates: List[str]) -> pd.DataFrame:
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    contract_path = repo_root / cfg.dataset(DATASET_KEY).contract
    contract = load_avro_contract(contract_path)

    frames = []
    for dt in dates:
        ref = partition_ref(cfg, DATASET_KEY, symbol, dt)
        if not is_partition_complete(ref):
            raise FileNotFoundError(f"Missing partition: {DATASET_KEY} dt={dt}")
        df = read_partition(ref)
        if len(df) > 0:
            df = enforce_contract(df, contract)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=contract.fields)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy by horizon.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--indices-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")

    df = _load_vectors(args.repo_root, args.symbol, dates)
    if len(df) == 0:
        raise ValueError("No vectors loaded from lake")

    retriever = TriggerVectorRetriever(args.indices_dir)
    results = evaluate_retrieval(df, retriever, k=args.k)
    for key in sorted(results.keys()):
        print(f"{key}: {results[key]}")


if __name__ == "__main__":
    main()
