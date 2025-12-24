"""
Boosted tree training for Phase 3 multi-head models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

from src.ml.feature_sets import select_features


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


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )


def train_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_set, label_col: str) -> Dict:
    y_train = train_df[label_col].astype(int)
    y_val = val_df[label_col].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(f"Not enough label diversity for {label_col}")

    preprocessor = build_preprocessor(feature_set.numeric, feature_set.categorical)
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    clf = Pipeline([("pre", preprocessor), ("model", model)])
    clf.fit(train_df, y_train)

    val_proba = clf.predict_proba(val_df)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "roc_auc": float(roc_auc_score(y_val, val_proba)),
        "brier": float(brier_score_loss(y_val, val_proba))
    }
    return {"model": clf, "metrics": metrics}


def train_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_set, label_col: str) -> Dict:
    y_train = train_df[label_col].astype(float)
    y_val = val_df[label_col].astype(float)

    preprocessor = build_preprocessor(feature_set.numeric, feature_set.categorical)
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    reg = Pipeline([("pre", preprocessor), ("model", model)])
    reg.fit(train_df, y_train)

    val_pred = reg.predict(val_df)
    metrics = {
        "mae": float(mean_absolute_error(y_val, val_pred))
    }
    return {"model": reg, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/ml/boosted_trees")
    parser.add_argument("--stage", type=str, choices=["stage_a", "stage_b"], default="stage_b")
    parser.add_argument("--ablation", type=str, choices=["full", "ta", "mechanics", "all"], default="full")
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

    valid_mask = df["outcome"].isin(["BREAK", "BOUNCE", "CHOP"])
    df = df[valid_mask].copy()
    splits = split_by_date(df, args.val_size, args.test_size)

    train_df = df[df["date"].isin(splits["train"])]
    val_df = df[df["date"].isin(splits["val"])]
    test_df = df[df["date"].isin(splits["test"])]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations = ["full", "ta", "mechanics"] if args.ablation == "all" else [args.ablation]

    for ablation in ablations:
        feature_set = select_features(train_df, stage=args.stage, ablation=ablation)
        results = {
            "stage": args.stage,
            "ablation": ablation,
            "splits": splits,
            "features": {
                "numeric": feature_set.numeric,
                "categorical": feature_set.categorical
            },
            "metrics": {}
        }

        # Tradeable_2 classifier
        # Multi-timeframe models: Train separate models for 2min, 4min, 8min
        timeframes = ['2min', '4min', '8min']
        
        for tf in timeframes:
            print(f"\n  Training {tf} horizon models...")
            
            # Tradeable classifier for this timeframe
            tradeable_col = f"tradeable_2_{tf}"
            outcome_col = f"outcome_{tf}"
            
            if tradeable_col not in train_df.columns:
                print(f"    WARNING: {tradeable_col} not found, skipping {tf} models")
                continue
            
            tradeable_df = train_df.copy()
            tradeable_val = val_df.copy()
            tradeable_df[tradeable_col] = tradeable_df[tradeable_col].fillna(0).astype(int)
            tradeable_val[tradeable_col] = tradeable_val[tradeable_col].fillna(0).astype(int)
            
            tradeable_fit = train_classifier(tradeable_df, tradeable_val, feature_set, tradeable_col)
            joblib.dump(tradeable_fit["model"], output_dir / f"tradeable_2_{tf}_{args.stage}_{ablation}.joblib")
            results["metrics"][f"tradeable_2_{tf}"] = tradeable_fit["metrics"]
            
            # Direction classifier (only tradeable events at this timeframe)
            dir_train = train_df[train_df[tradeable_col] == 1].copy()
            dir_val = val_df[val_df[tradeable_col] == 1].copy()
            
            if dir_train.empty or dir_val.empty:
                print(f"    WARNING: Not enough tradeable samples for {tf} direction model")
                continue
            
            dir_train["break_label"] = (dir_train[outcome_col] == "BREAK").astype(int)
            dir_val["break_label"] = (dir_val[outcome_col] == "BREAK").astype(int)
            
            direction_fit = train_classifier(dir_train, dir_val, feature_set, "break_label")
            joblib.dump(direction_fit["model"], output_dir / f"direction_{tf}_{args.stage}_{ablation}.joblib")
            results["metrics"][f"direction_{tf}"] = direction_fit["metrics"]
            
            print(f"    {tf} - Tradeable: {tradeable_fit['metrics']['roc_auc']:.3f} AUC")
            print(f"    {tf} - Direction: {direction_fit['metrics']['roc_auc']:.3f} AUC")
        
        # Backward compatibility: Also train legacy models using 4min as primary
        print(f"\n  Training legacy single-timeframe models (4min)...")
        tradeable_df = train_df.copy()
        tradeable_val = val_df.copy()
        tradeable_df["tradeable_2"] = tradeable_df["tradeable_2_4min"].fillna(0).astype(int)
        tradeable_val["tradeable_2"] = tradeable_val["tradeable_2_4min"].fillna(0).astype(int)
        tradeable_fit = train_classifier(tradeable_df, tradeable_val, feature_set, "tradeable_2")
        joblib.dump(tradeable_fit["model"], output_dir / f"tradeable_2_{args.stage}_{ablation}.joblib")
        results["metrics"]["tradeable_2"] = tradeable_fit["metrics"]

        # Direction classifier (4min outcomes)
        dir_train = train_df[train_df["tradeable_2_4min"] == 1].copy()
        dir_val = val_df[val_df["tradeable_2_4min"] == 1].copy()
        if not dir_train.empty and not dir_val.empty:
            dir_train["break_label"] = (dir_train["outcome_4min"] == "BREAK").astype(int)
            dir_val["break_label"] = (dir_val["outcome_4min"] == "BREAK").astype(int)
            direction_fit = train_classifier(dir_train, dir_val, feature_set, "break_label")
            joblib.dump(direction_fit["model"], output_dir / f"direction_{args.stage}_{ablation}.joblib")
            results["metrics"]["direction"] = direction_fit["metrics"]

        # Multi-timeframe strength regressors
        for tf in timeframes:
            strength_col = f"strength_signed_{tf}"
            
            if strength_col not in train_df.columns:
                print(f"    WARNING: {strength_col} not found, skipping {tf} strength model")
                continue
            
            strength_fit = train_regressor(train_df, val_df, feature_set, strength_col)
            joblib.dump(strength_fit["model"], output_dir / f"strength_{tf}_{args.stage}_{ablation}.joblib")
            results["metrics"][f"strength_{tf}"] = strength_fit["metrics"]
            print(f"    {tf} - Strength MAE: {strength_fit['metrics']['mae']:.3f}")
        
        # Legacy strength model (4min)
        strength_fit = train_regressor(train_df, val_df, feature_set, "strength_signed_4min")
        joblib.dump(strength_fit["model"], output_dir / f"strength_{args.stage}_{ablation}.joblib")
        results["metrics"]["strength_signed"] = strength_fit["metrics"]

        # Multi-timeframe time-to-threshold reach probabilities
        for tf in timeframes:
            for threshold_col_base, label_prefix in [("time_to_threshold_1", "t1"), ("time_to_threshold_2", "t2")]:
                threshold_col = f"{threshold_col_base}_{tf}"
                
                if threshold_col not in train_df.columns:
                    continue
                
                for horizon in args.horizons:
                    col = f"{label_prefix}_{horizon}s_{tf}"
                    train_h = train_df.copy()
                    val_h = val_df.copy()
                    train_h[col] = (train_h[threshold_col] <= horizon).astype(int).fillna(0)
                    val_h[col] = (val_h[threshold_col] <= horizon).astype(int).fillna(0)
                    fit = train_classifier(train_h, val_h, feature_set, col)
                joblib.dump(
                    fit["model"],
                    output_dir / f"{label_prefix}_{horizon}s_{args.stage}_{ablation}.joblib"
                )
                results["metrics"][col] = fit["metrics"]

        meta_path = output_dir / f"metadata_{args.stage}_{ablation}.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    print(f"Saved models and metadata to {output_dir}")


if __name__ == "__main__":
    main()
