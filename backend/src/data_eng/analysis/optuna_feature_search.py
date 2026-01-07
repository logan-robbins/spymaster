from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from numba import njit

from src.data_eng.config import load_config
from src.data_eng.io import is_partition_complete, partition_ref, read_partition

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]
ES_CONTRACTS = ["ESH5", "ESM5", "ESU5", "ESZ5", "ESH6"]
MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
EPSILON = 1e-9

METADATA_COLS = [
    "symbol", "bar_ts", "episode_id", "touch_id", "level_type", "level_price",
    "approach_direction", "dist_to_level_pts", "signed_dist_pts",
    "bar_index_in_episode", "bar_index_in_touch", "bars_to_trigger",
    "is_pre_trigger", "is_trigger_bar", "is_first_trigger",
    "is_extended_forward", "outcome", "outcome_score",
]


class FeatureType(Enum):
    EOB = "eob"
    TWA = "twa"
    SUM = "sum"
    CUMUL = "cumul"
    DERIV = "deriv"
    RVOL = "rvol"
    OTHER = "other"


class AggMethod(Enum):
    EXCLUDE = "exclude"
    LAST = "last"
    MEAN = "mean"
    WEIGHTED_RECENT = "weighted_recent"
    SUM = "sum"
    STD = "std"


VALID_AGGS: Dict[FeatureType, List[AggMethod]] = {
    FeatureType.EOB: [AggMethod.EXCLUDE, AggMethod.LAST, AggMethod.MEAN, AggMethod.WEIGHTED_RECENT],
    FeatureType.TWA: [AggMethod.EXCLUDE, AggMethod.LAST, AggMethod.MEAN, AggMethod.WEIGHTED_RECENT],
    FeatureType.SUM: [AggMethod.EXCLUDE, AggMethod.SUM, AggMethod.LAST, AggMethod.MEAN],
    FeatureType.CUMUL: [AggMethod.EXCLUDE, AggMethod.LAST],
    FeatureType.DERIV: [AggMethod.EXCLUDE, AggMethod.WEIGHTED_RECENT],
    FeatureType.RVOL: [AggMethod.EXCLUDE, AggMethod.LAST, AggMethod.MEAN, AggMethod.WEIGHTED_RECENT],
    FeatureType.OTHER: [AggMethod.EXCLUDE, AggMethod.LAST, AggMethod.MEAN],
}


def classify_feature(name: str) -> FeatureType:
    if "_d1_" in name or "_d2_" in name:
        return FeatureType.DERIV
    if "cumul_" in name:
        return FeatureType.CUMUL
    if name.startswith("rvol_"):
        return FeatureType.RVOL
    if name.endswith("_eob"):
        return FeatureType.EOB
    if name.endswith("_twa"):
        return FeatureType.TWA
    if name.endswith("_sum"):
        return FeatureType.SUM
    return FeatureType.OTHER


def get_front_month_contract(dt: str) -> str:
    from datetime import datetime
    date = datetime.strptime(dt, "%Y-%m-%d")
    for contract in ES_CONTRACTS:
        month_code = contract[2]
        year_digit = contract[3]
        month = MONTH_CODES[month_code]
        year = 2020 + int(year_digit)
        expiry = datetime(year, month, 1)
        if expiry >= date:
            return contract
    raise ValueError(f"No contract for {dt}")


@dataclass
class Episode:
    episode_id: str
    dt: str
    level_type: str
    outcome: str
    outcome_score: float
    trigger_bar: pd.Series
    lookback_bars: pd.DataFrame

    @property
    def directional_outcome(self) -> str:
        if "BOUNCE" in self.outcome:
            return "BOUNCE"
        if "BREAK" in self.outcome:
            return "BREAK"
        return "CHOP"


def weighted_recent(vals: np.ndarray, alpha: float = 0.1) -> float:
    if len(vals) == 0:
        return 0.0
    n = len(vals)
    weights = np.exp(-alpha * np.arange(n - 1, -1, -1))
    weights /= weights.sum()
    return float(np.sum(weights * vals))


def aggregate(vals: np.ndarray, method: AggMethod) -> float:
    if len(vals) == 0:
        return 0.0
    if method == AggMethod.LAST:
        return float(vals[-1])
    if method == AggMethod.MEAN:
        return float(np.mean(vals))
    if method == AggMethod.SUM:
        return float(np.sum(vals))
    if method == AggMethod.STD:
        return float(np.std(vals)) if len(vals) > 1 else 0.0
    if method == AggMethod.WEIGHTED_RECENT:
        return weighted_recent(vals)
    return 0.0


def safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    return df[col].fillna(0).values


def load_episodes(repo_root: Path, dates: List[str]) -> List[Episode]:
    cfg = load_config(repo_root, repo_root / "src/data_eng/config/datasets.yaml")
    episodes = []

    for dt in dates:
        if pd.to_datetime(dt).weekday() >= 5:
            continue
        symbol = get_front_month_contract(dt)

        for level_type in LEVEL_TYPES:
            dataset_key = f"silver.future.market_by_price_10_{level_type}_approach"
            ref = partition_ref(cfg, dataset_key, symbol, dt)
            if not is_partition_complete(ref):
                continue

            df = read_partition(ref)
            if len(df) == 0:
                continue

            for episode_id in df["episode_id"].unique():
                ep_df = df[df["episode_id"] == episode_id].sort_values("bar_ts")
                trigger_mask = ep_df["is_trigger_bar"] == True
                if not trigger_mask.any():
                    continue

                trigger_idx = trigger_mask.idxmax()
                trigger_bar = ep_df.loc[trigger_idx]
                lookback_bars = ep_df[ep_df["is_pre_trigger"] == True]
                if len(lookback_bars) < 12:
                    continue

                episodes.append(Episode(
                    episode_id=episode_id,
                    dt=dt,
                    level_type=level_type.upper(),
                    outcome=str(trigger_bar.get("outcome", "UNKNOWN")),
                    outcome_score=float(trigger_bar.get("outcome_score", 0)),
                    trigger_bar=trigger_bar,
                    lookback_bars=lookback_bars,
                ))

    return episodes


def discover_features(episodes: List[Episode]) -> List[str]:
    sample = episodes[0].trigger_bar
    all_cols = sample.index.tolist()
    return [c for c in all_cols if c not in METADATA_COLS]


@njit
def knn_accuracy_numba(
    train_vecs: np.ndarray,
    train_outcomes: np.ndarray,
    test_vecs: np.ndarray,
    test_outcomes: np.ndarray,
    k: int,
    n_outcomes: int,
) -> float:
    n_test = len(test_vecs)
    n_train = len(train_vecs)

    train_norms = np.sqrt(np.sum(train_vecs * train_vecs, axis=1)) + 1e-9
    test_norms = np.sqrt(np.sum(test_vecs * test_vecs, axis=1)) + 1e-9

    correct = 0
    for i in range(n_test):
        sims = np.zeros(n_train)
        for j in range(n_train):
            dot = 0.0
            for d in range(train_vecs.shape[1]):
                dot += test_vecs[i, d] * train_vecs[j, d]
            sims[j] = dot / (test_norms[i] * train_norms[j])

        idx = np.argsort(-sims)[:k]
        outcome_counts = np.zeros(n_outcomes)
        for j in idx:
            outcome_counts[train_outcomes[j]] += 1

        pred = np.argmax(outcome_counts)
        if pred == test_outcomes[i]:
            correct += 1

    return correct / n_test


def compress_episode(
    ep: Episode,
    features: List[str],
    config: Dict[str, AggMethod],
) -> Optional[np.ndarray]:
    vec = []
    for feat in features:
        method = config.get(feat, AggMethod.EXCLUDE)
        if method == AggMethod.EXCLUDE:
            continue
        vals = safe_col(ep.lookback_bars, feat)
        vec.append(aggregate(vals, method))
    if len(vec) == 0:
        return None
    return np.array(vec, dtype=np.float64)


def evaluate_config(
    config: Dict[str, AggMethod],
    features: List[str],
    train_episodes: List[Episode],
    test_episodes: List[Episode],
    k: int = 5,
) -> Tuple[float, int]:
    train_vectors = []
    train_outcomes = []

    for ep in train_episodes:
        vec = compress_episode(ep, features, config)
        if vec is not None:
            train_vectors.append(vec)
            train_outcomes.append(ep.directional_outcome)

    test_vectors = []
    test_outcomes = []

    for ep in test_episodes:
        vec = compress_episode(ep, features, config)
        if vec is not None:
            test_vectors.append(vec)
            test_outcomes.append(ep.directional_outcome)

    if len(train_vectors) < 10 or len(test_vectors) < 5:
        return 0.0, 0

    train_vecs = np.array(train_vectors)
    test_vecs = np.array(test_vectors)
    dim = train_vecs.shape[1]

    all_vecs = np.vstack([train_vecs, test_vecs])
    mean = np.nanmean(all_vecs, axis=0)
    std = np.nanstd(all_vecs, axis=0)
    std = np.where(std < EPSILON, 1.0, std)

    train_norm = (train_vecs - mean) / std
    test_norm = (test_vecs - mean) / std
    train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    test_norm = np.nan_to_num(test_norm, nan=0.0, posinf=0.0, neginf=0.0)
    train_norm = np.clip(train_norm, -10, 10)
    test_norm = np.clip(test_norm, -10, 10)

    all_outcomes = train_outcomes + test_outcomes
    outcome_labels = sorted(set(all_outcomes))
    outcome_map = {o: i for i, o in enumerate(outcome_labels)}

    train_outcome_idx = np.array([outcome_map[o] for o in train_outcomes], dtype=np.int32)
    test_outcome_idx = np.array([outcome_map[o] for o in test_outcomes], dtype=np.int32)

    accuracy = knn_accuracy_numba(
        train_norm.astype(np.float64),
        train_outcome_idx,
        test_norm.astype(np.float64),
        test_outcome_idx,
        k,
        len(outcome_labels),
    )

    return float(accuracy), dim


class FeatureSearchObjective:
    def __init__(
        self,
        features: List[str],
        feature_types: Dict[str, FeatureType],
        train_episodes: List[Episode],
        test_episodes: List[Episode],
        k: int = 5,
    ):
        self.features = features
        self.feature_types = feature_types
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.k = k

    def __call__(self, trial: optuna.Trial) -> float:
        config: Dict[str, AggMethod] = {}

        for feat in self.features:
            ftype = self.feature_types[feat]
            valid_aggs = VALID_AGGS[ftype]
            agg_names = [a.value for a in valid_aggs]
            chosen = trial.suggest_categorical(feat, agg_names)
            config[feat] = AggMethod(chosen)

        accuracy, dim = evaluate_config(
            config,
            self.features,
            self.train_episodes,
            self.test_episodes,
            self.k,
        )

        trial.set_user_attr("dim", dim)

        n_included = sum(1 for m in config.values() if m != AggMethod.EXCLUDE)
        trial.set_user_attr("n_features", n_included)

        return accuracy


def run_search(
    repo_root: Path,
    dates: List[str],
    n_trials: int,
    output_dir: Path,
    k: int = 5,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    t0 = time.time()

    print("Loading episodes...", flush=True)
    episodes = load_episodes(repo_root, dates)
    print(f"Loaded {len(episodes)} episodes in {time.time() - t0:.1f}s", flush=True)

    if len(episodes) < 50:
        raise ValueError(f"Not enough episodes: {len(episodes)}")

    episodes_by_date = {}
    for ep in episodes:
        episodes_by_date.setdefault(ep.dt, []).append(ep)

    sorted_dates = sorted(episodes_by_date.keys())
    split_idx = int(len(sorted_dates) * 0.8)
    train_dates = set(sorted_dates[:split_idx])
    test_dates = set(sorted_dates[split_idx:])

    train_episodes = [ep for ep in episodes if ep.dt in train_dates]
    test_episodes = [ep for ep in episodes if ep.dt in test_dates]

    print(f"Train: {len(train_episodes)} episodes ({len(train_dates)} days)", flush=True)
    print(f"Test: {len(test_episodes)} episodes ({len(test_dates)} days)", flush=True)

    features = discover_features(episodes)
    feature_types = {f: classify_feature(f) for f in features}

    type_counts = {}
    for f, t in feature_types.items():
        type_counts[t.value] = type_counts.get(t.value, 0) + 1

    print(f"Feature types: {type_counts}", flush=True)
    print(f"Total features in search space: {len(features)}", flush=True)

    objective = FeatureSearchObjective(
        features=features,
        feature_types=feature_types,
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        k=k,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=100),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=0),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number % 50 == 0 or trial.value == study.best_value:
            print(
                f"Trial {trial.number}: acc={trial.value:.3f} "
                f"dim={trial.user_attrs.get('dim', 0)} "
                f"n_feats={trial.user_attrs.get('n_features', 0)} "
                f"best={study.best_value:.3f}",
                flush=True,
            )

    print(f"\nStarting {n_trials} trials...", flush=True)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback],
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_config = {}
    included_features = []

    for feat in features:
        agg = AggMethod(best_trial.params[feat])
        if agg != AggMethod.EXCLUDE:
            best_config[feat] = agg.value
            included_features.append({"feature": feat, "aggregation": agg.value})

    results = {
        "n_trials": n_trials,
        "n_episodes": len(episodes),
        "n_train": len(train_episodes),
        "n_test": len(test_episodes),
        "train_dates": sorted(train_dates),
        "test_dates": sorted(test_dates),
        "best_accuracy": best_trial.value,
        "best_dim": best_trial.user_attrs.get("dim", 0),
        "best_n_features": best_trial.user_attrs.get("n_features", 0),
        "best_trial_number": best_trial.number,
        "included_features": included_features,
        "feature_config": best_config,
        "total_time_s": time.time() - t0,
    }

    (output_dir / "best_config.json").write_text(json.dumps(results, indent=2))

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:20]
    top_results = [
        {
            "rank": i + 1,
            "trial": t.number,
            "accuracy": t.value,
            "dim": t.user_attrs.get("dim", 0),
            "n_features": t.user_attrs.get("n_features", 0),
        }
        for i, t in enumerate(top_trials)
    ]
    (output_dir / "top_trials.json").write_text(json.dumps(top_results, indent=2))

    feature_importance = {}
    for feat in features:
        include_count = sum(
            1 for t in study.trials
            if t.params.get(feat) != "exclude"
        )
        feature_importance[feat] = include_count / len(study.trials)

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    (output_dir / "feature_importance.json").write_text(
        json.dumps([{"feature": f, "inclusion_rate": r} for f, r in sorted_importance], indent=2)
    )

    print(f"\n=== BEST RESULT ===", flush=True)
    print(f"Accuracy: {best_trial.value:.3f}", flush=True)
    print(f"Dimensions: {best_trial.user_attrs.get('dim', 0)}", flush=True)
    print(f"Features: {best_trial.user_attrs.get('n_features', 0)}", flush=True)
    print(f"Trial: {best_trial.number}", flush=True)
    print(f"\nTop features:", flush=True)
    for item in included_features[:20]:
        print(f"  {item['feature']}: {item['aggregation']}", flush=True)

    print(f"\nResults saved to {output_dir}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna feature search")
    parser.add_argument("--dates", required=True, help="Date range (YYYY-MM-DD:YYYY-MM-DD)")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs")
    args = parser.parse_args()

    from src.data_eng.utils import expand_date_range
    dates = expand_date_range(dates=args.dates)

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "ablation_results" / "optuna_feature_search"

    run_search(
        repo_root=repo_root,
        dates=dates,
        n_trials=args.n_trials,
        output_dir=output_dir,
        k=args.k,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
