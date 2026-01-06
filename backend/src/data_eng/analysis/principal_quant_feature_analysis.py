from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif, mutual_info_classif

from src.data_eng.config import load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.utils import expand_date_range

LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
ES_CONTRACTS = ["ESH5", "ESM5", "ESU5", "ESZ5", "ESH6", "ESM6", "ESU6", "ESZ6"]
MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
EPSILON = 1e-9


def get_front_month_contract(contracts: List[str], dt: str) -> str:
    """Determine front-month contract for date. Matches Bronze stage logic."""
    from datetime import datetime

    date = datetime.strptime(dt, "%Y-%m-%d")
    contract_dates = []

    for contract in contracts:
        if len(contract) < 4:
            continue
        month_code = contract[2]
        year_digit = contract[3]
        if month_code not in MONTH_CODES:
            continue
        month = MONTH_CODES[month_code]
        year = 2020 + int(year_digit)
        expiry = datetime(year, month, 1)
        if expiry >= date:
            contract_dates.append((contract, expiry))

    if not contract_dates:
        raise ValueError(f"No valid front month contract found for {dt} in {contracts}")

    contract_dates.sort(key=lambda x: x[1])
    return contract_dates[0][0]


RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_WINDOW_MINUTES = 180


@dataclass
class DataSplit:
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_dates: List[str]
    test_dates: List[str]


def _load_contract_fields(contract_path: Path) -> List[Dict[str, Any]]:
    contract = json.loads(contract_path.read_text())
    return contract["fields"]


def _is_string_field(field: Dict[str, Any]) -> bool:
    ftype = field["type"]
    if isinstance(ftype, list):
        return "string" in ftype
    return ftype == "string"


def _build_feature_names(fields: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    string_cols = [f["name"] for f in fields if _is_string_field(f)]
    label_cols = ["outcome", "outcome_score"]
    excluded = set(string_cols + label_cols)
    feature_names = [f["name"] for f in fields if f["name"] not in excluded]
    return feature_names, string_cols


def _load_silver_approach_data(
    repo_root: Path,
    root_symbol: str,
    dates: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    cfg = load_config(repo_root, repo_root / "src/data_eng/config/datasets.yaml")
    frames = []
    missing = []
    skipped = []

    for dt in dates:
        if pd.to_datetime(dt).weekday() >= 5:
            skipped.append(dt)
            continue

        front_month_symbol = get_front_month_contract(ES_CONTRACTS, dt)

        for level_type in LEVEL_TYPES:
            dataset_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach"
            ref = partition_ref(cfg, dataset_key, front_month_symbol, dt)
            if not is_partition_complete(ref):
                missing.append(f"{dataset_key} symbol={front_month_symbol} dt={dt}")
                continue

            contract_path = repo_root / cfg.dataset(dataset_key).contract
            contract = load_avro_contract(contract_path)
            df = read_partition(ref)
            if len(df) == 0:
                continue

            df = enforce_contract(df, contract)
            df["dt"] = dt
            df["contract_symbol"] = front_month_symbol
            frames.append(df)

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing partitions:\n{missing_text}")

    if not frames:
        raise ValueError("No silver approach data found")

    df = pd.concat(frames, ignore_index=True)
    return df, skipped


def _filter_trigger_and_rth(df: pd.DataFrame) -> pd.DataFrame:
    if "is_trigger_bar" not in df.columns:
        raise ValueError("is_trigger_bar column missing")

    trigger_mask = df["is_trigger_bar"].astype(bool).values
    df = df.loc[trigger_mask].copy()

    ts = pd.to_datetime(df["bar_ts"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    minutes_since_open = (ts.dt.hour - RTH_START_HOUR) * 60 + ts.dt.minute - RTH_START_MINUTE
    rth_mask = (minutes_since_open >= 0) & (minutes_since_open <= RTH_WINDOW_MINUTES)
    df = df.loc[rth_mask].copy()

    if len(df) == 0:
        raise ValueError("No trigger bars in RTH window")

    return df


def _time_split(metadata: pd.DataFrame, test_ratio: float = 0.2) -> DataSplit:
    dates = sorted(metadata["dt"].unique())
    if len(dates) < 2:
        raise ValueError("Need at least 2 dates for split")

    split_idx = int(len(dates) * (1 - test_ratio))
    split_idx = max(1, min(split_idx, len(dates) - 1))
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train_mask = metadata["dt"].isin(train_dates).values
    test_mask = metadata["dt"].isin(test_dates).values

    return DataSplit(train_mask=train_mask, test_mask=test_mask, train_dates=train_dates, test_dates=test_dates)


def _fit_normalization(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < EPSILON, 1.0, std)

    params = {
        name: {"mean": float(m), "std": float(s)}
        for name, m, s in zip(feature_names, mean, std)
    }

    return mean, std, params


def _apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xn = (X - mean) / (std + EPSILON)
    Xn = np.clip(Xn, -10, 10)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=10.0, neginf=-10.0)
    return Xn


def variance_analysis(X: np.ndarray, feature_names: List[str], threshold: float = 1e-6) -> Dict[str, List[Dict[str, Any]]]:
    variances = np.var(X, axis=0)
    results = {"zero_variance": [], "low_variance": [], "normal_variance": []}

    for i, (var, name) in enumerate(zip(variances, feature_names)):
        entry = {"index": i, "name": name, "variance": float(var)}
        if var < threshold:
            entry["action"] = "REMOVE"
            results["zero_variance"].append(entry)
        elif var < 0.01:
            entry["action"] = "FLAG"
            results["low_variance"].append(entry)
        else:
            entry["action"] = "KEEP"
            results["normal_variance"].append(entry)

    return results


def univariate_analysis(
    X: np.ndarray,
    outcome_score: np.ndarray,
    outcome_class: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    y = outcome_score.astype(float)
    y_mean = y.mean()
    y_std = y.std() + EPSILON

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + EPSILON
    cov = ((X - x_mean) * (y - y_mean)[:, None]).mean(axis=0)
    pearson = cov / (x_std * y_std)
    pearson = np.nan_to_num(pearson, nan=0.0)

    x_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    y_rank_mean = y_rank.mean()
    y_rank_std = y_rank.std() + EPSILON
    x_rank_mean = x_rank.mean(axis=0)
    x_rank_std = x_rank.std(axis=0) + EPSILON
    cov_rank = ((x_rank - x_rank_mean) * (y_rank - y_rank_mean)[:, None]).mean(axis=0)
    spearman = cov_rank / (x_rank_std * y_rank_std)
    spearman = np.nan_to_num(spearman, nan=0.0)

    f_stat, p_val = f_classif(X, outcome_class)
    mi = mutual_info_classif(X, outcome_class, random_state=42)

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "index": i,
            "name": name,
            "pearson_r": float(pearson[i]),
            "abs_pearson": float(abs(pearson[i])),
            "spearman_r": float(spearman[i]),
            "f_stat": float(f_stat[i]) if i < len(f_stat) else 0.0,
            "p_value": float(p_val[i]) if i < len(p_val) else 1.0,
            "mutual_info": float(mi[i]) if i < len(mi) else 0.0,
        })

    df = pd.DataFrame(rows)
    df["univariate_rank"] = df["abs_pearson"].rank(ascending=False)
    return df.sort_values("abs_pearson", ascending=False)


def redundancy_analysis(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.85,
) -> Dict[str, Any]:
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    linkage_matrix = linkage(squareform(dist_matrix), method="average")
    clusters = fcluster(linkage_matrix, t=1 - threshold, criterion="distance")

    cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, (cluster_id, name) in enumerate(zip(clusters, feature_names)):
        cluster_groups.setdefault(cluster_id, []).append({"index": i, "name": name})

    redundant_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr_matrix[i, j]
            if abs(r) > 0.95:
                redundant_pairs.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "correlation": float(r),
                })

    return {
        "n_clusters": len(cluster_groups),
        "clusters": cluster_groups,
        "redundant_pairs": redundant_pairs,
        "correlation_matrix": corr_matrix,
    }


def select_cluster_representatives(
    clusters: Dict[int, List[Dict[str, Any]]],
    univariate_scores: pd.DataFrame,
) -> List[Dict[str, Any]]:
    reps = []

    for cluster_id, members in clusters.items():
        member_names = [m["name"] for m in members]
        member_scores = univariate_scores[univariate_scores["name"].isin(member_names)]
        best = member_scores.loc[member_scores["abs_pearson"].idxmax()]
        reps.append({
            "cluster_id": int(cluster_id),
            "representative": best["name"],
            "score": float(best["abs_pearson"]),
            "cluster_size": len(members),
            "dropped": [m["name"] for m in members if m["name"] != best["name"]],
        })

    return reps


def run_backtest(
    train_vectors: np.ndarray,
    train_meta: pd.DataFrame,
    test_vectors: np.ndarray,
    test_meta: pd.DataFrame,
    k: int = 10,
    max_per_day: int = 2,
) -> Dict[str, float]:
    if len(train_vectors) == 0 or len(test_vectors) == 0:
        return {"accuracy": 0.0, "score_correlation": 0.0, "top2_accuracy": 0.0, "n_test": 0}
    if train_vectors.shape[1] == 0 or test_vectors.shape[1] == 0:
        return {"accuracy": 0.0, "score_correlation": 0.0, "top2_accuracy": 0.0, "n_test": int(len(test_vectors))}

    train = train_vectors.astype(np.float64)
    test = test_vectors.astype(np.float64)

    train_norm = train / (np.linalg.norm(train, axis=1, keepdims=True) + EPSILON)
    test_norm = test / (np.linalg.norm(test, axis=1, keepdims=True) + EPSILON)

    sim = test_norm @ train_norm.T
    k_fetch = min(k * 5, train_norm.shape[0])
    idx_part = np.argpartition(-sim, kth=k_fetch - 1, axis=1)[:, :k_fetch]
    row_idx = np.arange(sim.shape[0])[:, None]
    sorted_idx = idx_part[row_idx, np.argsort(-sim[row_idx, idx_part], axis=1)]

    correct = 0
    top2_correct = 0
    predicted_scores = []
    actual_scores = []

    for q_idx in range(len(test)):
        actual_outcome = test_meta.iloc[q_idx]["outcome"]
        actual_score = float(test_meta.iloc[q_idx]["outcome_score"])

        neighbor_indices = sorted_idx[q_idx]
        date_counts: Dict[str, int] = {}
        chosen = []

        for n_idx in neighbor_indices:
            if n_idx < 0:
                continue
            n_date = train_meta.iloc[n_idx]["dt"]
            date_counts[n_date] = date_counts.get(n_date, 0) + 1
            if date_counts[n_date] <= max_per_day:
                chosen.append(n_idx)
            if len(chosen) >= k:
                break

        if not chosen:
            continue

        neighbor_outcomes = [train_meta.iloc[idx]["outcome"] for idx in chosen]
        outcome_counts = pd.Series(neighbor_outcomes).value_counts()
        predicted_outcome = outcome_counts.index[0]

        neighbor_scores = [float(train_meta.iloc[idx]["outcome_score"]) for idx in chosen]
        predicted_score = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0

        if predicted_outcome == actual_outcome:
            correct += 1

        top2 = outcome_counts.index[:2].tolist()
        if actual_outcome in top2:
            top2_correct += 1

        predicted_scores.append(predicted_score)
        actual_scores.append(actual_score)

    n_test = len(test)
    accuracy = correct / n_test if n_test > 0 else 0.0
    top2_accuracy = top2_correct / n_test if n_test > 0 else 0.0

    if len(predicted_scores) >= 2:
        score_corr = np.corrcoef(predicted_scores, actual_scores)[0, 1]
        if np.isnan(score_corr):
            score_corr = 0.0
    else:
        score_corr = 0.0

    return {
        "accuracy": float(accuracy),
        "score_correlation": float(score_corr),
        "top2_accuracy": float(top2_accuracy),
        "n_test": int(n_test),
    }


def _encode_dates(dates: pd.Series) -> Tuple[np.ndarray, int]:
    unique = sorted(dates.unique())
    mapping = {d: i for i, d in enumerate(unique)}
    codes = dates.map(mapping).astype(int).values
    return codes.astype(np.int32), len(unique)


def _encode_outcomes(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    labels = sorted(pd.concat([train_meta["outcome"], test_meta["outcome"]]).unique())
    mapping = {label: i for i, label in enumerate(labels)}
    train_codes = train_meta["outcome"].map(mapping).astype(int).values
    test_codes = test_meta["outcome"].map(mapping).astype(int).values
    return train_codes.astype(np.int32), test_codes.astype(np.int32), len(labels)


@njit
def _accuracy_from_similarity(
    sim: np.ndarray,
    train_dates: np.ndarray,
    train_outcomes: np.ndarray,
    test_outcomes: np.ndarray,
    k: int,
    max_per_day: int,
    n_dates: int,
    n_outcomes: int,
    k_fetch: int,
) -> Tuple[float, float]:
    n_test, n_train = sim.shape
    correct = 0
    top2_correct = 0
    if k_fetch > n_train:
        k_fetch = n_train

    for i in range(n_test):
        idx = np.argsort(sim[i])
        date_counts = np.zeros(n_dates, dtype=np.int32)
        outcome_counts = np.zeros(n_outcomes, dtype=np.int32)
        chosen = 0
        start = n_train - 1
        stop = n_train - k_fetch
        if stop < 0:
            stop = 0

        for j_idx in range(start, stop - 1, -1):
            j = idx[j_idx]
            d = train_dates[j]
            if date_counts[d] < max_per_day:
                date_counts[d] += 1
                outcome_counts[train_outcomes[j]] += 1
                chosen += 1
                if chosen >= k:
                    break

        if chosen == 0:
            continue

        first = -1
        second = -1
        for o in range(n_outcomes):
            if first == -1 or outcome_counts[o] > outcome_counts[first]:
                second = first
                first = o
            elif second == -1 or outcome_counts[o] > outcome_counts[second]:
                second = o

        if first == test_outcomes[i]:
            correct += 1
        if test_outcomes[i] == first or test_outcomes[i] == second:
            top2_correct += 1

    n_test_float = float(n_test) if n_test > 0 else 1.0
    acc = correct / n_test_float
    top2 = top2_correct / n_test_float
    return acc, top2


def forward_selection(
    train_vectors: np.ndarray,
    train_meta: pd.DataFrame,
    test_vectors: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: List[str],
    candidate_indices: List[int],
    max_features: int = 60,
    patience: int = 5,
) -> Dict[str, Any]:
    if not candidate_indices:
        return {
            "selected_indices": [],
            "selected_names": [],
            "final_accuracy": 0.0,
            "history": [],
        }

    train = train_vectors.astype(np.float64)
    test = test_vectors.astype(np.float64)
    n_test, n_train = test.shape[0], train.shape[0]

    train_dates, n_dates = _encode_dates(train_meta["dt"])
    train_outcomes, test_outcomes, n_outcomes = _encode_outcomes(train_meta, test_meta)
    k = 10
    max_per_day = 2
    k_fetch = min(k * 5, n_train)

    selected: List[int] = []
    remaining = set(candidate_indices)
    history = []
    best_score = 0.0
    rounds_no_gain = 0

    dot_base = np.zeros((n_test, n_train), dtype=np.float64)
    test_norm_sq = np.zeros(n_test, dtype=np.float64)
    train_norm_sq = np.zeros(n_train, dtype=np.float64)

    while len(selected) < max_features and remaining and rounds_no_gain < patience:
        best_candidate = None
        best_candidate_score = best_score
        best_dot = None
        best_test_norm_sq = None
        best_train_norm_sq = None

        for candidate in list(remaining):
            test_feat = test[:, candidate]
            train_feat = train[:, candidate]
            dot = dot_base + np.outer(test_feat, train_feat)
            test_norm_sq_cand = test_norm_sq + test_feat * test_feat
            train_norm_sq_cand = train_norm_sq + train_feat * train_feat
            denom = (np.sqrt(test_norm_sq_cand) + EPSILON)[:, None] * (np.sqrt(train_norm_sq_cand) + EPSILON)[None, :]
            sim = dot / denom
            acc, _ = _accuracy_from_similarity(
                sim,
                train_dates,
                train_outcomes,
                test_outcomes,
                k,
                max_per_day,
                n_dates,
                n_outcomes,
                k_fetch,
            )

            if acc > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = acc
                best_dot = dot
                best_test_norm_sq = test_norm_sq_cand
                best_train_norm_sq = train_norm_sq_cand

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

        dot_base = best_dot
        test_norm_sq = best_test_norm_sq
        train_norm_sq = best_train_norm_sq

        improvement = best_candidate_score - best_score
        best_score = best_candidate_score
        if improvement < 0.001:
            rounds_no_gain += 1
        else:
            rounds_no_gain = 0

        history.append({
            "step": len(selected),
            "added_feature": feature_names[best_candidate],
            "accuracy": float(best_score),
            "improvement": float(improvement),
        })

    return {
        "selected_indices": selected,
        "selected_names": [feature_names[i] for i in selected],
        "final_accuracy": float(best_score),
        "history": history,
    }


def backward_elimination(
    train_vectors: np.ndarray,
    train_meta: pd.DataFrame,
    test_vectors: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: List[str],
    selected_indices: List[int],
    tolerance: float = 0.005,
) -> Dict[str, Any]:
    if not selected_indices:
        return {
            "final_set": [],
            "final_names": [],
            "eliminated": [],
            "final_accuracy": 0.0,
        }

    current_set = list(selected_indices)

    baseline = run_backtest(
        train_vectors[:, current_set],
        train_meta,
        test_vectors[:, current_set],
        test_meta,
    )["accuracy"]

    eliminated = []

    for idx in list(selected_indices):
        test_set = [i for i in current_set if i != idx]
        score = run_backtest(
            train_vectors[:, test_set],
            train_meta,
            test_vectors[:, test_set],
            test_meta,
        )["accuracy"]

        drop = baseline - score
        if drop < tolerance:
            eliminated.append({
                "feature": feature_names[idx],
                "drop": float(drop),
                "action": "REMOVE" if drop < 0 else "REMOVE (marginal)",
            })
            current_set = test_set
            baseline = score

    return {
        "final_set": current_set,
        "final_names": [feature_names[i] for i in current_set],
        "eliminated": eliminated,
        "final_accuracy": float(baseline),
    }


def interaction_analysis(
    train_vectors: np.ndarray,
    train_meta: pd.DataFrame,
    test_vectors: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: List[str],
    base_set: List[int],
    candidates: List[int],
) -> pd.DataFrame:
    train = train_vectors.astype(np.float64)
    test = test_vectors.astype(np.float64)
    n_test, n_train = test.shape[0], train.shape[0]

    train_dates, n_dates = _encode_dates(train_meta["dt"])
    train_outcomes, test_outcomes, n_outcomes = _encode_outcomes(train_meta, test_meta)
    k = 10
    max_per_day = 2
    k_fetch = min(k * 5, n_train)

    if base_set:
        base_train = train[:, base_set]
        base_test = test[:, base_set]
        base_dot = base_test @ base_train.T
        base_test_norm_sq = np.sum(base_test * base_test, axis=1)
        base_train_norm_sq = np.sum(base_train * base_train, axis=1)
        base_denom = (np.sqrt(base_test_norm_sq) + EPSILON)[:, None] * (np.sqrt(base_train_norm_sq) + EPSILON)[None, :]
        base_sim = base_dot / base_denom
        base_score, _ = _accuracy_from_similarity(
            base_sim,
            train_dates,
            train_outcomes,
            test_outcomes,
            k,
            max_per_day,
            n_dates,
            n_outcomes,
            k_fetch,
        )
        base_test_feats = base_test
        base_train_feats = base_train
    else:
        base_dot = np.zeros((n_test, n_train), dtype=np.float64)
        base_test_norm_sq = np.zeros(n_test, dtype=np.float64)
        base_train_norm_sq = np.zeros(n_train, dtype=np.float64)
        base_score = 0.0
        base_test_feats = np.zeros((n_test, 0), dtype=np.float64)
        base_train_feats = np.zeros((n_train, 0), dtype=np.float64)

    rows = []

    for candidate in candidates:
        test_feat = test[:, candidate]
        train_feat = train[:, candidate]
        dot_single = np.outer(test_feat, train_feat)
        test_norm_sq = test_feat * test_feat
        train_norm_sq = train_feat * train_feat

        denom_single = (np.sqrt(test_norm_sq) + EPSILON)[:, None] * (np.sqrt(train_norm_sq) + EPSILON)[None, :]
        sim_single = dot_single / denom_single
        single_score, _ = _accuracy_from_similarity(
            sim_single,
            train_dates,
            train_outcomes,
            test_outcomes,
            k,
            max_per_day,
            n_dates,
            n_outcomes,
            k_fetch,
        )

        dot_combined = base_dot + dot_single
        test_norm_sq_combined = base_test_norm_sq + test_norm_sq
        train_norm_sq_combined = base_train_norm_sq + train_norm_sq
        denom_combined = (np.sqrt(test_norm_sq_combined) + EPSILON)[:, None] * (np.sqrt(train_norm_sq_combined) + EPSILON)[None, :]
        sim_combined = dot_combined / denom_combined
        combined_score, _ = _accuracy_from_similarity(
            sim_combined,
            train_dates,
            train_outcomes,
            test_outcomes,
            k,
            max_per_day,
            n_dates,
            n_outcomes,
            k_fetch,
        )

        best_pair = None
        best_pair_score = 0.0
        for base_pos, base_idx in enumerate(base_set):
            base_test_feat = base_test_feats[:, base_pos]
            base_train_feat = base_train_feats[:, base_pos]
            dot_pair = np.outer(base_test_feat, base_train_feat) + dot_single
            test_norm_pair = base_test_feat * base_test_feat + test_norm_sq
            train_norm_pair = base_train_feat * base_train_feat + train_norm_sq
            denom_pair = (np.sqrt(test_norm_pair) + EPSILON)[:, None] * (np.sqrt(train_norm_pair) + EPSILON)[None, :]
            sim_pair = dot_pair / denom_pair
            pair_score, _ = _accuracy_from_similarity(
                sim_pair,
                train_dates,
                train_outcomes,
                test_outcomes,
                k,
                max_per_day,
                n_dates,
                n_outcomes,
                k_fetch,
            )
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = feature_names[base_idx]

        rows.append({
            "candidate": feature_names[candidate],
            "single_score": float(single_score),
            "with_base_set": float(combined_score),
            "improvement": float(combined_score - base_score),
            "best_pair_with": best_pair,
            "best_pair_score": float(best_pair_score),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("improvement", ascending=False)


def stability_analysis(
    X: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: List[str],
    candidate_indices: List[int],
    n_splits: int = 5,
) -> Dict[str, Any]:
    start_time = time.time()
    dates = sorted(metadata["dt"].unique())
    if len(dates) < 3:
        raise ValueError("Need at least 3 dates for stability analysis")

    n_splits = min(n_splits, len(dates) - 1)
    split_size = max(1, len(dates) // n_splits)

    fold_results = []

    for i in range(n_splits - 1):
        fold_start = time.time()
        train_end = dates[(i + 1) * split_size]
        test_start = dates[(i + 1) * split_size]
        test_end = dates[min((i + 2) * split_size, len(dates) - 1)]

        train_mask = metadata["dt"] < train_end
        test_mask = (metadata["dt"] >= test_start) & (metadata["dt"] <= test_end)

        train_vectors = X[train_mask]
        test_vectors = X[test_mask]
        train_meta = metadata[train_mask].reset_index(drop=True)
        test_meta = metadata[test_mask].reset_index(drop=True)

        fs = forward_selection(
            train_vectors,
            train_meta,
            test_vectors,
            test_meta,
            feature_names,
            candidate_indices,
        )

        selected_idx = fs["selected_indices"]
        test_score = run_backtest(
            train_vectors[:, selected_idx],
            train_meta,
            test_vectors[:, selected_idx],
            test_meta,
        )

        fold_results.append({
            "fold": i,
            "train_end": train_end,
            "test_period": f"{test_start} to {test_end}",
            "n_features_selected": len(selected_idx),
            "selected_features": fs["selected_names"],
            "test_accuracy": float(test_score["accuracy"]),
        })
        print(f"Stability fold {i} done in {time.time() - fold_start:.2f}s", flush=True)

    all_selected = [set(r["selected_features"]) for r in fold_results if r["selected_features"]]
    if all_selected:
        intersection = set.intersection(*all_selected)
        union = set.union(*all_selected)
    else:
        intersection = set()
        union = set()

    stable_min = max(1, int(np.ceil(len(fold_results) * 0.6)))
    counts: Dict[str, int] = {}
    for r in fold_results:
        for name in r["selected_features"]:
            counts[name] = counts.get(name, 0) + 1

    stable_features = [name for name, cnt in counts.items() if cnt >= stable_min]
    unstable_features = [name for name, cnt in counts.items() if cnt < stable_min]

    return {
        "fold_results": fold_results,
        "stable_features": stable_features,
        "unstable_features": unstable_features,
        "stability_ratio": float(len(intersection) / len(union)) if union else 0.0,
        "mean_test_accuracy": float(np.mean([r["test_accuracy"] for r in fold_results])) if fold_results else 0.0,
        "std_test_accuracy": float(np.std([r["test_accuracy"] for r in fold_results])) if fold_results else 0.0,
    }


def _feature_category(name: str) -> str:
    if name.startswith("bar5s_"):
        parts = name.split("_")
        if len(parts) > 1:
            return f"bar5s_{parts[1]}"
        return "bar5s"
    if name.startswith("rvol_"):
        return "rvol"
    if name.startswith("is_"):
        return "flags"
    if name.startswith("bars_") or name.startswith("bar_index"):
        return "episode_timing"
    if name.startswith("dist_") or name.startswith("signed_dist") or name.startswith("approach_"):
        return "approach"
    if name.startswith("bar_"):
        return "bar_meta"
    if name.startswith("level_"):
        return "level"
    if name.startswith("trigger_"):
        return "trigger"
    if name.startswith("extension_"):
        return "extension"
    return "other"


def category_analysis(
    train_vectors: np.ndarray,
    train_meta: pd.DataFrame,
    test_vectors: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    baseline = run_backtest(train_vectors, train_meta, test_vectors, test_meta)["accuracy"]

    categories: Dict[str, List[int]] = {}
    for idx, name in enumerate(feature_names):
        categories.setdefault(_feature_category(name), []).append(idx)

    rows = []
    all_indices = list(range(len(feature_names)))

    for category, indices in categories.items():
        if not indices:
            continue

        category_score = run_backtest(
            train_vectors[:, indices],
            train_meta,
            test_vectors[:, indices],
            test_meta,
        )["accuracy"]

        other_indices = [i for i in all_indices if i not in indices]
        without_score = run_backtest(
            train_vectors[:, other_indices],
            train_meta,
            test_vectors[:, other_indices],
            test_meta,
        )["accuracy"]

        rows.append({
            "category": category,
            "n_features": len(indices),
            "category_only_acc": float(category_score),
            "without_category_acc": float(without_score),
            "contribution": float(baseline - without_score),
            "standalone_vs_random": float(category_score - 0.20),
        })

    return pd.DataFrame(rows).sort_values("contribution", ascending=False)


def final_recommendations(
    feature_names: List[str],
    variance_results: Dict[str, List[Dict[str, Any]]],
    univariate_df: pd.DataFrame,
    reps: List[Dict[str, Any]],
    forward_results: Dict[str, Any],
    backward_results: Dict[str, Any],
    interaction_df: pd.DataFrame,
    stability_results: Dict[str, Any],
) -> Dict[str, Any]:
    zero_vars = {f["name"] for f in variance_results["zero_variance"]}
    rep_set = {r["representative"] for r in reps}
    forward_set = set(forward_results["selected_names"])
    backward_set = set(backward_results["final_names"])
    stable_set = set(stability_results["stable_features"])

    interaction_pos = set(
        interaction_df[interaction_df["improvement"] > 0]["candidate"].tolist()
    )

    univariate_scores = {row["name"]: row["abs_pearson"] for _, row in univariate_df.iterrows()}

    tiers = {"GOLD": [], "SILVER": [], "BRONZE": [], "DISCARD": []}

    for name in feature_names:
        if name in zero_vars:
            tiers["DISCARD"].append(name)
            continue

        high_uni = univariate_scores.get(name, 0.0) >= 0.10

        if name in backward_set and name in stable_set:
            tiers["GOLD"].append(name)
        elif (name in forward_set and name in stable_set) or (high_uni and name in stable_set):
            tiers["SILVER"].append(name)
        elif name in interaction_pos or (name in forward_set and name not in stable_set):
            tiers["BRONZE"].append(name)
        else:
            if name in rep_set:
                tiers["BRONZE"].append(name)
            else:
                tiers["DISCARD"].append(name)

    return tiers


def run_analysis(
    repo_root: Path,
    root_symbol: str,
    dates: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    t0 = time.time()
    contract_path = repo_root / "src/data_eng/contracts/silver/future/market_by_price_10_level_approach.avsc"
    fields = _load_contract_fields(contract_path)
    feature_names, string_cols = _build_feature_names(fields)

    df, skipped_dates = _load_silver_approach_data(repo_root, root_symbol, dates)
    df = _filter_trigger_and_rth(df)
    print(f"Loaded data: {len(df)} rows, {len(feature_names)} features in {time.time() - t0:.2f}s", flush=True)

    if "outcome_score" not in df.columns or "outcome" not in df.columns:
        raise ValueError("Outcome columns missing")

    outcome_score = df["outcome_score"].fillna(0.0).astype(float).values
    outcomes = df["outcome"].astype(str).values
    outcome_labels = sorted(set(outcomes))
    outcome_map = {name: i for i, name in enumerate(outcome_labels)}
    outcome_class = np.array([outcome_map[o] for o in outcomes])

    X = df[feature_names].fillna(0.0).astype(float).values

    split = _time_split(df)
    train_X = X[split.train_mask]
    test_X = X[split.test_mask]

    mean, std, norm_params = _fit_normalization(train_X, feature_names)
    train_Xn = _apply_normalization(train_X, mean, std)
    test_Xn = _apply_normalization(test_X, mean, std)
    Xn = _apply_normalization(X, mean, std)

    variance_results = variance_analysis(train_Xn, feature_names)
    print(f"Variance analysis done in {time.time() - t0:.2f}s", flush=True)

    univariate_df = univariate_analysis(train_Xn, outcome_score[split.train_mask], outcome_class[split.train_mask], feature_names)
    print(f"Univariate analysis done in {time.time() - t0:.2f}s", flush=True)

    keep_mask = np.array([name not in {f["name"] for f in variance_results["zero_variance"]} for name in feature_names])
    kept_names = [name for name, keep in zip(feature_names, keep_mask) if keep]
    kept_indices = [i for i, keep in enumerate(keep_mask) if keep]

    redundancy = redundancy_analysis(train_Xn[:, kept_indices], kept_names, threshold=0.85)
    print(f"Redundancy analysis done in {time.time() - t0:.2f}s", flush=True)
    reps = select_cluster_representatives(redundancy["clusters"], univariate_df)
    reps_sorted = sorted(reps, key=lambda r: r["score"], reverse=True)
    reps_used = reps_sorted

    rep_indices = [feature_names.index(r["representative"]) for r in reps_used]
    print(f"Representatives selected: {len(reps_used)} in {time.time() - t0:.2f}s", flush=True)

    train_meta = df[split.train_mask].reset_index(drop=True)
    test_meta = df[split.test_mask].reset_index(drop=True)

    rep_feature_names = [feature_names[i] for i in rep_indices]
    rep_train = train_Xn[:, rep_indices]
    rep_test = test_Xn[:, rep_indices]
    rep_all = Xn[:, rep_indices]
    rep_candidates = list(range(len(rep_indices)))

    forward_rep = forward_selection(
        rep_train,
        train_meta,
        rep_test,
        test_meta,
        rep_feature_names,
        rep_candidates,
    )
    forward_indices = [rep_indices[i] for i in forward_rep["selected_indices"]]
    forward_results = {
        "selected_indices": forward_indices,
        "selected_names": [feature_names[i] for i in forward_indices],
        "final_accuracy": forward_rep["final_accuracy"],
        "history": forward_rep["history"],
    }
    print(f"Forward selection done in {time.time() - t0:.2f}s", flush=True)

    backward_rep = backward_elimination(
        rep_train,
        train_meta,
        rep_test,
        test_meta,
        rep_feature_names,
        forward_rep["selected_indices"],
    )
    backward_indices = [rep_indices[i] for i in backward_rep["final_set"]]
    backward_results = {
        "final_set": backward_indices,
        "final_names": [feature_names[i] for i in backward_indices],
        "eliminated": backward_rep["eliminated"],
        "final_accuracy": backward_rep["final_accuracy"],
    }
    print(f"Backward elimination done in {time.time() - t0:.2f}s", flush=True)

    interaction_candidates = [i for i in rep_candidates if i not in forward_rep["selected_indices"]]
    interaction_df = interaction_analysis(
        rep_train,
        train_meta,
        rep_test,
        test_meta,
        rep_feature_names,
        backward_rep["final_set"],
        interaction_candidates,
    )
    print(f"Interaction analysis done in {time.time() - t0:.2f}s", flush=True)

    stability_results = stability_analysis(
        rep_all,
        df,
        rep_feature_names,
        rep_candidates,
        n_splits=5,
    )
    print(f"Stability analysis done in {time.time() - t0:.2f}s", flush=True)

    category_df = category_analysis(
        train_Xn,
        train_meta,
        test_Xn,
        test_meta,
        feature_names,
    )
    print(f"Category analysis done in {time.time() - t0:.2f}s", flush=True)

    tiers = final_recommendations(
        feature_names,
        variance_results,
        univariate_df,
        reps,
        forward_results,
        backward_results,
        interaction_df,
        stability_results,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "feature_names.json").write_text(json.dumps({
        "feature_names": feature_names,
        "string_columns": string_cols,
        "label_columns": ["outcome", "outcome_score"],
    }, indent=2))

    (output_dir / "normalization_params.json").write_text(json.dumps(norm_params, indent=2))

    (output_dir / "variance_analysis.json").write_text(json.dumps(variance_results, indent=2))

    univariate_df.to_csv(output_dir / "univariate_analysis.csv", index=False)

    clusters_serialized = {int(k): v for k, v in redundancy["clusters"].items()}
    redundancy_summary = {
        "n_clusters": redundancy["n_clusters"],
        "clusters": clusters_serialized,
        "redundant_pairs": redundancy["redundant_pairs"],
    }
    (output_dir / "redundancy_analysis.json").write_text(json.dumps(redundancy_summary, indent=2))

    (output_dir / "cluster_representatives.json").write_text(json.dumps(reps, indent=2))
    (output_dir / "cluster_representatives_used.json").write_text(json.dumps(reps_used, indent=2))

    (output_dir / "forward_selection.json").write_text(json.dumps(forward_results, indent=2))

    (output_dir / "backward_elimination.json").write_text(json.dumps(backward_results, indent=2))

    interaction_df.to_csv(output_dir / "interaction_analysis.csv", index=False)

    (output_dir / "stability_analysis.json").write_text(json.dumps(stability_results, indent=2))

    category_df.to_csv(output_dir / "category_analysis.csv", index=False)

    (output_dir / "final_recommendations.json").write_text(json.dumps(tiers, indent=2))

    contract_symbols = sorted(df["contract_symbol"].unique().tolist()) if "contract_symbol" in df.columns else []

    summary = {
        "root_symbol": root_symbol,
        "contract_symbols": contract_symbols,
        "dates": dates,
        "skipped_dates": skipped_dates,
        "n_rows": int(len(df)),
        "n_features": len(feature_names),
        "n_representatives": len(reps),
        "n_representatives_used": len(reps_used),
        "train_dates": split.train_dates,
        "test_dates": split.test_dates,
        "outcome_map": outcome_map,
        "output_dir": str(output_dir),
    }

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    return {
        "summary": summary,
        "variance": variance_results,
        "univariate": univariate_df,
        "redundancy": redundancy,
        "representatives": reps,
        "forward": forward_results,
        "backward": backward_results,
        "interaction": interaction_df,
        "stability": stability_results,
        "category": category_df,
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Principal quant feature analysis")
    parser.add_argument("--root-symbol", required=True, help="Root symbol (e.g., ES)")
    parser.add_argument("--dates", required=True)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates supplied")

    output_dir = repo_root / "ablation_results" / "principal_quant_feature_analysis"

    result = run_analysis(repo_root, args.root_symbol, dates, output_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
