"""
Build and persist a kNN retrieval index from the signals parquet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src.ml.feature_sets import select_features
from src.ml.retrieval_engine import RetrievalIndex
from src.ml.tracking import (
    hash_file,
    log_artifacts,
    resolve_git_sha,
    resolve_repo_root,
    tracking_run,
)


def _resolve_signals_path(signals_path: str | None) -> Path:
    if signals_path:
        return Path(signals_path)

    features_path = Path(__file__).resolve().parents[2] / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found at {features_path}")

    with features_path.open("r") as fh:
        features = json.load(fh)

    output_path = features.get("output_path")
    if not output_path:
        raise ValueError("features.json missing output_path")

    return features_path.parent / output_path


def _load_features_version(backend_root: Path) -> str:
    features_path = backend_root / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found at {features_path}")
    payload = json.loads(features_path.read_text())
    version = payload.get("version")
    if not version:
        raise ValueError("features.json missing version field")
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="data/ml/retrieval_index.joblib")
    parser.add_argument("--stage", type=str, choices=["stage_a", "stage_b"], default="stage_b")
    parser.add_argument("--ablation", type=str, choices=["full", "ta", "mechanics"], default="full")
    args = parser.parse_args()

    signals_path = _resolve_signals_path(args.data_path)
    if not signals_path.exists():
        raise FileNotFoundError(f"Signals parquet not found: {signals_path}")

    df = pd.read_parquet(signals_path)
    if df.empty:
        raise ValueError("Signals dataset is empty.")

    df = df[df["outcome"].isin(["BREAK", "BOUNCE", "CHOP"])].copy()
    if df.empty:
        raise ValueError("No labeled outcomes found for retrieval index.")

    feature_set = select_features(df, stage=args.stage, ablation=args.ablation)
    if not feature_set.numeric:
        raise ValueError("No numeric features available for retrieval index.")

    backend_root = Path(__file__).resolve().parents[2]
    repo_root = resolve_repo_root()
    git_sha = resolve_git_sha(repo_root)
    features_version = _load_features_version(backend_root)
    dataset_hash = hash_file(signals_path)

    dates = sorted(df["date"].dropna().unique())
    if not dates:
        raise ValueError("No date values found for retrieval index.")

    run_name = f"retrieval_index-v{features_version}-{args.stage}-{args.ablation}-{dates[0]}_{dates[-1]}"
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "spymaster_retrieval")
    project = os.getenv("WANDB_PROJECT", "spymaster")
    params = {
        "data_path": str(signals_path),
        "output_path": args.output_path,
        "stage": args.stage,
        "ablation": args.ablation,
        "rows": len(df),
        "feature_count": len(feature_set.numeric),
        "dataset_hash": dataset_hash,
        "features_version": features_version,
        "git_sha": git_sha,
        "date_range": [dates[0], dates[-1]],
    }
    tags = {
        "stage": args.stage,
        "ablation": args.ablation,
        "dataset_hash": dataset_hash,
        "git_sha": git_sha,
        "features_version": features_version,
    }
    wandb_tags = ["retrieval_index", args.stage, args.ablation]

    with tracking_run(
        run_name=run_name,
        experiment=experiment,
        params=params,
        tags=tags,
        wandb_tags=wandb_tags,
        project=project,
        repo_root=repo_root,
    ) as tracking:
        index = RetrievalIndex(feature_cols=feature_set.numeric)
        index.fit(df)

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(index, output_path)

        meta = {
            "data_path": str(signals_path),
            "rows": int(len(df)),
            "stage": args.stage,
            "ablation": args.ablation,
            "feature_cols": feature_set.numeric,
            "features_version": features_version,
            "dataset_hash": dataset_hash,
            "git_sha": git_sha,
        }
        meta_path = output_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        features_path = backend_root / "features.json"
        log_artifacts(
            [output_path, meta_path, features_path],
            name=f"retrieval_index_{args.stage}_{args.ablation}",
            artifact_type="model",
            wandb_run=tracking.wandb_run,
        )


if __name__ == "__main__":
    main()
