"""
Calibration evaluation for boosted-tree outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml.data_filters import filter_rth_signals
from src.ml.tree_inference import TreeModelBundle
from src.ml.tracking import (
    hash_file,
    log_artifacts,
    log_metrics,
    resolve_git_sha,
    resolve_repo_root,
    tracking_run,
)


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


def _extract_metrics(results: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    block = results.get("metrics", {})

    tradeable = block.get("tradeable_2", {})
    metrics["tradeable_2_brier"] = tradeable.get("brier")
    metrics["tradeable_2_samples"] = tradeable.get("samples")

    p_break = block.get("p_break", {})
    metrics["p_break_brier"] = p_break.get("brier")
    metrics["p_break_samples"] = p_break.get("samples")

    t1 = block.get("t1_reach", {})
    for horizon, payload in t1.items():
        metrics[f"t1_{horizon}_brier"] = payload.get("brier")
        metrics[f"t1_{horizon}_samples"] = payload.get("samples")

    t2 = block.get("t2_reach", {})
    for horizon, payload in t2.items():
        metrics[f"t2_{horizon}_brier"] = payload.get("brier")
        metrics[f"t2_{horizon}_samples"] = payload.get("samples")

    for key in ("t1_break_reach", "t1_bounce_reach", "t2_break_reach", "t2_bounce_reach"):
        block_metrics = block.get(key, {})
        for horizon, payload in block_metrics.items():
            metrics[f"{key}_{horizon}_brier"] = payload.get("brier")
            metrics[f"{key}_{horizon}_samples"] = payload.get("samples")

    return metrics


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
    features_version = features_json.get("version")
    if not features_version:
        raise ValueError("features.json missing version field")
    data_path = args.data_path or features_json["output_path"]
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Signals parquet missing: {data_path}")

    df = pd.read_parquet(data_path)
    if df.empty:
        raise ValueError("Signals dataset is empty.")

    df = filter_rth_signals(df)
    if df.empty:
        raise ValueError("No signals left after RTH filter.")

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

    repo_root = resolve_repo_root()
    git_sha = resolve_git_sha(repo_root)
    dataset_hash = hash_file(data_path)

    test_dates = sorted(splits["test"])
    run_name = f"calibration_eval-v{features_version}-{args.stage}-{args.ablation}-{test_dates[0]}_{test_dates[-1]}"
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "spymaster_calibration")
    project = os.getenv("WANDB_PROJECT", "spymaster")
    params = {
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "output_path": args.output_path,
        "stage": args.stage,
        "ablation": args.ablation,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "horizons": args.horizons,
        "rows": len(df),
        "test_rows": len(test_df),
        "dataset_hash": dataset_hash,
        "features_version": features_version,
        "git_sha": git_sha,
        "test_dates": test_dates,
    }
    tags = {
        "stage": args.stage,
        "ablation": args.ablation,
        "dataset_hash": dataset_hash,
        "git_sha": git_sha,
        "features_version": features_version,
    }
    wandb_tags = ["calibration_eval", args.stage, args.ablation]

    with tracking_run(
        run_name=run_name,
        experiment=experiment,
        params=params,
        tags=tags,
        wandb_tags=wandb_tags,
        project=project,
        repo_root=repo_root,
    ) as tracking:
        bundle = TreeModelBundle(
            model_dir=model_dir,
            stage=args.stage,
            ablation=args.ablation,
            horizons=args.horizons,
        )
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

        directional_cols = {
            "t1_break_reach": ("time_to_break_1", preds.t1_break_probs),
            "t1_bounce_reach": ("time_to_bounce_1", preds.t1_bounce_probs),
            "t2_break_reach": ("time_to_break_2", preds.t2_break_probs),
            "t2_bounce_reach": ("time_to_bounce_2", preds.t2_bounce_probs),
        }
        for metric_key, (col_name, pred_map) in directional_cols.items():
            col = test_df.get(col_name)
            if col is None:
                continue
            col_vals = col.to_numpy(dtype=float)
            metric_block = {}
            for horizon in args.horizons:
                true_vals = np.nan_to_num((col_vals <= horizon).astype(int), nan=0)
                pred_vals = pred_map[horizon]
                metric_block[horizon] = {
                    "brier": float(brier_score_loss(true_vals, pred_vals)),
                    "calibration": _calibration_points(true_vals, pred_vals),
                    "samples": int(len(true_vals))
                }
            results["metrics"][metric_key] = metric_block

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        log_metrics(_extract_metrics(results), tracking.wandb_run)

        features_path = Path(__file__).resolve().parents[2] / "features.json"
        log_artifacts(
            [output_path, features_path],
            name=f"calibration_eval_{args.stage}_{args.ablation}",
            artifact_type="metrics",
            wandb_run=tracking.wandb_run,
        )


if __name__ == "__main__":
    main()
