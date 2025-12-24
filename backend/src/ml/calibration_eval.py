"""
Calibration evaluation for boosted-tree outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml.tree_inference import TreeModelBundle


DEFAULT_HORIZONS = [60, 120, 180, 300]


def load_features_json() -> Dict:
    features_path = Path(__file__).resolve().parents[2] / "features.json"
    with open(features_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_by_date(df: pd.DataFrame, val_size: int, test_size: int) -> Dict[str, List[str]]:
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < (val_size + test_size + 1):
        raise ValueError("Not enough dates for walk-forward split.")

    test_dates = dates[-test_size:]
    val_dates = dates[-(test_size + val_size) : -test_size]
    train_dates = dates[: -(test_size + val_size)]

    return {
        "train": train_dates,
        "val": val_dates,
        "test": test_dates
    }


def _calibration_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    if len(np.unique(y_true)) < 2:
        return []
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )
    return [
        {"mean_pred": float(mp), "frac_pos": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="data/ml/boosted_trees")
    parser.add_argument("--output-path", type=str, default="data/ml/calibration_eval.json")
    parser.add_argument("--stage", type=str, choices=["stage_a", "stage_b"], default="stage_b")
    parser.add_argument("--ablation", type=str, choices=["full", "ta", "mechanics"], default="full")
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="*", default=DEFAULT_HORIZONS)
    args = parser.parse_args()

    features_json = load_features_json()
    data_path = args.data_path or features_json["output_path"]
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Signals parquet missing: {data_path}")

    df = pd.read_parquet(data_path)
    if df.empty:
        raise ValueError("Signals dataset is empty.")

    df = df[df["outcome"].isin(["BREAK", "BOUNCE", "CHOP"])].copy()
    if df.empty:
        raise ValueError("No labeled outcomes available for calibration.")

    splits = split_by_date(df, args.val_size, args.test_size)
    test_df = df[df["date"].isin(splits["test"])].copy()
    if test_df.empty:
        raise ValueError("No test samples found for calibration.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    bundle = TreeModelBundle(model_dir=model_dir, stage=args.stage, ablation=args.ablation, horizons=args.horizons)
    preds = bundle.predict(test_df)

    tradeable_true = test_df["tradeable_2"].fillna(0).astype(int).to_numpy()
    tradeable_pred = preds.tradeable_2

    results = {
        "stage": args.stage,
        "ablation": args.ablation,
        "splits": splits,
        "metrics": {}
    }

    results["metrics"]["tradeable_2"] = {
        "brier": float(brier_score_loss(tradeable_true, tradeable_pred)),
        "calibration": _calibration_points(tradeable_true, tradeable_pred),
        "samples": int(len(tradeable_true))
    }

    direction_mask = tradeable_true == 1
    if np.any(direction_mask):
        direction_true = (test_df["outcome"].to_numpy()[direction_mask] == "BREAK").astype(int)
        direction_pred = preds.p_break[direction_mask]
        results["metrics"]["p_break"] = {
            "brier": float(brier_score_loss(direction_true, direction_pred)),
            "calibration": _calibration_points(direction_true, direction_pred),
            "samples": int(len(direction_true))
        }
    else:
        results["metrics"]["p_break"] = {
            "brier": float("nan"),
            "calibration": [],
            "samples": 0
        }

    t1_metrics = {}
    t2_metrics = {}
    time_to_1 = test_df.get("time_to_threshold_1")
    time_to_2 = test_df.get("time_to_threshold_2")
    if time_to_1 is None or time_to_2 is None:
        raise ValueError("Missing time-to-threshold columns for calibration.")

    time_to_1_vals = time_to_1.to_numpy(dtype=float)
    time_to_2_vals = time_to_2.to_numpy(dtype=float)

    for horizon in args.horizons:
        t1_true = np.nan_to_num((time_to_1_vals <= horizon).astype(int), nan=0)
        t2_true = np.nan_to_num((time_to_2_vals <= horizon).astype(int), nan=0)
        t1_pred = preds.t1_probs[horizon]
        t2_pred = preds.t2_probs[horizon]

        t1_metrics[horizon] = {
            "brier": float(brier_score_loss(t1_true, t1_pred)),
            "calibration": _calibration_points(t1_true, t1_pred),
            "samples": int(len(t1_true))
        }
        t2_metrics[horizon] = {
            "brier": float(brier_score_loss(t2_true, t2_pred)),
            "calibration": _calibration_points(t2_true, t2_pred),
            "samples": int(len(t2_true))
        }

    results["metrics"]["t1_reach"] = t1_metrics
    results["metrics"]["t2_reach"] = t2_metrics

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
