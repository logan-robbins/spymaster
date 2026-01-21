from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--max_evals", type=int, default=10)
    parser.add_argument("--model_out", required=True)
    return parser.parse_args()


def load_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    if "feature_vector" not in df.columns or "label" not in df.columns:
        raise ValueError("Input data missing feature_vector or label")
    df = df[df["label"].notna()].copy()
    if df.empty:
        raise ValueError("Input data has no labeled rows")
    features = np.vstack(df["feature_vector"].to_list()).astype(np.float32)
    labels = df["label"].astype(int).to_numpy()
    return features, labels


def main() -> None:
    args = parse_args()

    X_train, y_train = load_split(args.train_dir)
    X_val, y_val = load_split(args.val_dir)
    X_test, y_test = load_split(args.test_dir)

    space = {
        "C": hp.loguniform("C", np.log(1e-3), np.log(10.0)),
        "max_iter": hp.quniform("max_iter", 50, 300, 25),
    }

    trial_index = {"i": 0}

    def objective(params: dict) -> dict:
        trial_index["i"] += 1
        model = LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        mlflow.log_metric("val_accuracy", val_acc, step=trial_index["i"])
        return {"loss": 1.0 - val_acc, "status": STATUS_OK}

    mlflow.set_experiment("spymaster_hyperopt")
    with mlflow.start_run(run_name="hyperopt_run"):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
            rstate=np.random.default_rng(1337),
        )

        best_model = LogisticRegression(
            C=float(best["C"]),
            max_iter=int(best["max_iter"]),
            solver="lbfgs",
        )
        best_model.fit(X_train, y_train)
        test_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_param("best_C", float(best["C"]))
        mlflow.log_param("best_max_iter", int(best["max_iter"]))

        mlflow.sklearn.log_model(best_model, artifact_path="model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_version = mlflow.register_model(model_uri, "es_logreg_model")
        mlflow.log_param("registered_model_name", "es_logreg_model")
        mlflow.log_param("registered_model_version", model_version.version)

        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(best_model, handle)
        mlflow.log_param("model_output_path", str(model_path))


if __name__ == "__main__":
    main()
