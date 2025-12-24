"""
Build and persist a kNN retrieval index from the signals parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.ml.feature_sets import select_features
from src.ml.retrieval_engine import RetrievalIndex


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
        "feature_cols": feature_set.numeric
    }
    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
