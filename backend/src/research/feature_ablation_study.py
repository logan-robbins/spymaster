"""Feature Ablation Study - FEATURE_ANALYSIS_ABLATION.md Implementation."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import faiss
except ImportError:
    faiss = None

REPO_ROOT = Path(__file__).resolve().parents[2]
INDICES_DIR = REPO_ROOT / "databases" / "indices"
TARGET_DIM = 256
RAW_FEATURE_COUNT = 148

FEATURE_NAMES = [
    'approach_dist_to_level_pts_eob',
    'approach_side_of_level_eob',
    'approach_alignment_eob',
    'approach_level_polarity',
    'is_standard_approach',
    'state_obi0_eob',
    'state_obi10_eob',
    'state_spread_pts_eob',
    'state_cdi_p0_1_eob',
    'state_cdi_p1_2_eob',
    'state_cdi_p2_3_eob',
    'lvl_depth_imbal_eob',
    'lvl_cdi_p0_1_eob',
    'lvl_cdi_p1_2_eob',
    'depth_bid10_qty_eob',
    'depth_ask10_qty_eob',
    'lvl_depth_above_qty_eob',
    'lvl_depth_below_qty_eob',
    'wall_bid_maxz_eob',
    'wall_ask_maxz_eob',
    'wall_bid_maxz_levelidx_eob',
    'wall_ask_maxz_levelidx_eob',
    'wall_bid_nearest_strong_dist_pts_eob',
    'wall_ask_nearest_strong_dist_pts_eob',
    'wall_bid_nearest_strong_levelidx_eob',
    'wall_ask_nearest_strong_levelidx_eob',
    'cumul_signed_trade_vol',
    'cumul_flow_imbal',
    'cumul_flow_net_bid',
    'cumul_flow_net_ask',
    'lvl_flow_toward_net_sum',
    'lvl_flow_away_net_sum',
    'lvl_flow_toward_away_imbal_sum',
    'trade_signed_vol_sum',
    'trade_aggbuy_vol_sum',
    'trade_aggsell_vol_sum',
    'deriv_dist_d1_w3',
    'deriv_dist_d1_w12',
    'deriv_dist_d1_w36',
    'deriv_dist_d1_w72',
    'deriv_dist_d2_w3',
    'deriv_dist_d2_w12',
    'deriv_dist_d2_w36',
    'deriv_dist_d2_w72',
    'deriv_obi0_d1_w12',
    'deriv_obi0_d1_w36',
    'deriv_obi10_d1_w12',
    'deriv_obi10_d1_w36',
    'deriv_cdi01_d1_w12',
    'deriv_cdi01_d1_w36',
    'deriv_cdi12_d1_w12',
    'deriv_cdi12_d1_w36',
    'deriv_obi0_d2_w12',
    'deriv_obi0_d2_w36',
    'deriv_obi10_d2_w12',
    'deriv_obi10_d2_w36',
    'deriv_cdi01_d2_w12',
    'deriv_cdi01_d2_w36',
    'deriv_cdi12_d2_w12',
    'deriv_cdi12_d2_w36',
    'deriv_dbid10_d1_w12',
    'deriv_dbid10_d1_w36',
    'deriv_dask10_d1_w12',
    'deriv_dask10_d1_w36',
    'deriv_dbelow01_d1_w12',
    'deriv_dbelow01_d1_w36',
    'deriv_dabove01_d1_w12',
    'deriv_dabove01_d1_w36',
    'deriv_wbidz_d1_w12',
    'deriv_wbidz_d1_w36',
    'deriv_waskz_d1_w12',
    'deriv_waskz_d1_w36',
    'deriv_wbidz_d2_w12',
    'deriv_wbidz_d2_w36',
    'deriv_waskz_d2_w12',
    'deriv_waskz_d2_w36',
    'setup_start_dist_pts',
    'setup_min_dist_pts',
    'setup_max_dist_pts',
    'setup_dist_range_pts',
    'setup_approach_bars',
    'setup_retreat_bars',
    'setup_approach_ratio',
    'setup_early_velocity',
    'setup_mid_velocity',
    'setup_late_velocity',
    'setup_velocity_trend',
    'setup_obi0_start',
    'setup_obi0_end',
    'setup_obi0_delta',
    'setup_obi0_min',
    'setup_obi0_max',
    'setup_obi10_start',
    'setup_obi10_end',
    'setup_obi10_delta',
    'setup_obi10_min',
    'setup_obi10_max',
    'setup_total_trade_vol',
    'setup_total_signed_vol',
    'setup_trade_imbal_pct',
    'setup_flow_imbal_total',
    'setup_bid_wall_max_z',
    'setup_ask_wall_max_z',
    'setup_bid_wall_bars',
    'setup_ask_wall_bars',
    'setup_wall_imbal',
    'setup_velocity_std',
    'setup_obi0_mean',
    'setup_obi0_std',
    'setup_obi10_mean',
    'setup_obi10_std',
    'setup_cdi01_mean',
    'setup_cdi01_std',
    'setup_lvl_depth_imbal_mean',
    'setup_lvl_depth_imbal_std',
    'setup_lvl_depth_imbal_trend',
    'setup_spread_mean',
    'setup_flow_toward_total',
    'setup_flow_away_total',
    'setup_flow_toward_away_ratio',
    'setup_trade_vol_early',
    'setup_trade_vol_mid',
    'setup_trade_vol_late',
    'setup_trade_vol_trend',
    'setup_signed_vol_early',
    'setup_signed_vol_mid',
    'setup_signed_vol_late',
    'setup_bid_wall_mean_z',
    'setup_ask_wall_mean_z',
    'setup_bid_wall_closest_dist_min',
    'setup_ask_wall_closest_dist_min',
    'setup_wall_appeared_bid',
    'setup_wall_appeared_ask',
    'setup_wall_disappeared_bid',
    'setup_flow_net_bid_total',
    'setup_flow_net_ask_total',
    'recent_dist_delta',
    'recent_obi0_delta',
    'recent_obi10_delta',
    'recent_cdi01_delta',
    'recent_trade_vol',
    'recent_signed_vol',
    'recent_flow_toward',
    'recent_flow_away',
    'recent_aggbuy_vol',
    'recent_aggsell_vol',
    'recent_bid_depth_delta',
    'recent_ask_depth_delta',
]

assert len(FEATURE_NAMES) == 148

FEATURE_GROUPS = {
    'position': list(range(0, 5)),
    'book_state': list(range(5, 18)),
    'walls': list(range(18, 26)),
    'flow_snapshot': list(range(26, 36)),
    'deriv_dist': list(range(36, 44)),
    'deriv_imbal': list(range(44, 60)),
    'deriv_depth': list(range(60, 68)),
    'deriv_wall': list(range(68, 76)),
    'profile_traj': list(range(76, 87)),
    'profile_book': list(range(87, 106)),
    'profile_flow': list(range(106, 124)),
    'profile_wall': list(range(124, 136)),
    'recent': list(range(136, 148)),
    'padding': list(range(148, 256)),
}

SEMANTIC_GROUPS = {
    'WHERE': FEATURE_GROUPS['position'],
    'TRAJECTORY': FEATURE_GROUPS['deriv_dist'] + FEATURE_GROUPS['profile_traj'] + [136, 137, 138, 139],
    'BOOK_PHYSICS': (FEATURE_GROUPS['book_state'] + FEATURE_GROUPS['deriv_imbal'] +
                     FEATURE_GROUPS['deriv_depth'] + FEATURE_GROUPS['profile_book']),
    'FLOW_PHYSICS': (FEATURE_GROUPS['flow_snapshot'] + FEATURE_GROUPS['profile_flow'] +
                     [140, 141, 142, 143, 144, 145]),
    'WALLS': FEATURE_GROUPS['walls'] + FEATURE_GROUPS['deriv_wall'] + FEATURE_GROUPS['profile_wall'],
}

OUTCOME_MAP = {
    'STRONG_BOUNCE': 0, 'WEAK_BOUNCE': 1, 'CHOP': 2, 'WEAK_BREAK': 3, 'STRONG_BREAK': 4
}
OUTCOME_SCORE_MAP = {
    'STRONG_BOUNCE': 2.0, 'WEAK_BOUNCE': 1.0, 'CHOP': 0.0, 'WEAK_BREAK': -1.0, 'STRONG_BREAK': -2.0
}


def load_vectors_and_metadata(
    level_type: str = "pm_high",
    indices_dir: Path = INDICES_DIR,
) -> Tuple[np.ndarray, pd.DataFrame]:
    vectors_path = indices_dir / f"{level_type}_vectors.npy"
    metadata_path = indices_dir / f"{level_type}_metadata.db"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    vectors = np.load(str(vectors_path))

    conn = sqlite3.connect(str(metadata_path))
    metadata = pd.read_sql("SELECT * FROM setup_metadata ORDER BY vector_id", conn)
    conn.close()

    return vectors, metadata


def load_all_level_vectors(
    indices_dir: Path = INDICES_DIR,
) -> Tuple[np.ndarray, pd.DataFrame]:
    all_vectors = []
    all_metadata = []

    for level_type in ["pm_high", "pm_low", "or_high", "or_low"]:
        try:
            vectors, metadata = load_vectors_and_metadata(level_type, indices_dir)
            metadata["level_type_source"] = level_type
            all_vectors.append(vectors)
            all_metadata.append(metadata)
        except FileNotFoundError:
            print(f"Skipping {level_type}: not found")
            continue

    if not all_vectors:
        raise ValueError("No vectors found")

    combined_vectors = np.vstack(all_vectors)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    combined_metadata["global_vector_id"] = range(len(combined_metadata))

    return combined_vectors, combined_metadata


def temporal_train_test_split(
    metadata: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    dates = sorted(metadata["dt"].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train_mask = metadata["dt"].isin(train_dates)
    test_mask = metadata["dt"].isin(test_dates)

    return train_mask.values, test_mask.values


def run_backtest(
    train_vectors: np.ndarray,
    train_metadata: pd.DataFrame,
    test_vectors: np.ndarray,
    test_metadata: pd.DataFrame,
    k: int = 10,
    max_per_day: int = 2,
) -> Dict[str, float]:
    if faiss is None:
        raise ImportError("faiss-cpu required")

    if len(train_vectors) == 0 or len(test_vectors) == 0:
        return {"accuracy": 0.0, "score_correlation": 0.0, "top2_accuracy": 0.0, "n_test": 0}

    train_vectors = np.ascontiguousarray(train_vectors.astype(np.float32))
    test_vectors = np.ascontiguousarray(test_vectors.astype(np.float32))

    faiss.normalize_L2(train_vectors)
    faiss.normalize_L2(test_vectors)

    d = train_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_vectors)

    _, indices = index.search(test_vectors, min(k * 5, len(train_vectors)))

    correct = 0
    top2_correct = 0
    predicted_scores = []
    actual_scores = []

    for q_idx in range(len(test_vectors)):
        actual_outcome = test_metadata.iloc[q_idx]["outcome"]
        actual_score = OUTCOME_SCORE_MAP.get(actual_outcome, 0.0)

        neighbor_indices = indices[q_idx]
        neighbor_indices = neighbor_indices[neighbor_indices >= 0]

        date_counts = Counter()
        selected_neighbors = []
        for n_idx in neighbor_indices:
            if n_idx >= len(train_metadata):
                continue
            n_date = train_metadata.iloc[n_idx]["dt"]
            if date_counts[n_date] < max_per_day:
                selected_neighbors.append(n_idx)
                date_counts[n_date] += 1
                if len(selected_neighbors) >= k:
                    break

        if not selected_neighbors:
            continue

        neighbor_outcomes = [train_metadata.iloc[idx]["outcome"] for idx in selected_neighbors]
        outcome_counts = Counter(neighbor_outcomes)
        predicted_outcome = outcome_counts.most_common(1)[0][0]

        neighbor_scores = [OUTCOME_SCORE_MAP.get(o, 0.0) for o in neighbor_outcomes]
        predicted_score = np.mean(neighbor_scores)

        if predicted_outcome == actual_outcome:
            correct += 1

        top2_outcomes = [o for o, _ in outcome_counts.most_common(2)]
        if actual_outcome in top2_outcomes:
            top2_correct += 1

        predicted_scores.append(predicted_score)
        actual_scores.append(actual_score)

    n_test = len(test_vectors)
    accuracy = correct / n_test if n_test > 0 else 0.0
    top2_accuracy = top2_correct / n_test if n_test > 0 else 0.0

    if len(predicted_scores) >= 2:
        score_corr = np.corrcoef(predicted_scores, actual_scores)[0, 1]
        if np.isnan(score_corr):
            score_corr = 0.0
    else:
        score_corr = 0.0

    return {
        "accuracy": accuracy,
        "score_correlation": score_corr,
        "top2_accuracy": top2_accuracy,
        "n_test": n_test,
    }


class FeatureAblationStudy:
    def __init__(self, indices_dir: Path = INDICES_DIR):
        self.indices_dir = indices_dir
        self.vectors = None
        self.metadata = None
        self.train_mask = None
        self.test_mask = None
        self.baseline_metrics = None

    def load_data(self):
        self.vectors, self.metadata = load_all_level_vectors(self.indices_dir)
        self.train_mask, self.test_mask = temporal_train_test_split(self.metadata)
        print(f"Loaded {len(self.vectors)} vectors ({self.train_mask.sum()} train, {self.test_mask.sum()} test)")

    def compute_baseline(self) -> Dict[str, float]:
        train_v = self.vectors[self.train_mask]
        train_m = self.metadata[self.train_mask].reset_index(drop=True)
        test_v = self.vectors[self.test_mask]
        test_m = self.metadata[self.test_mask].reset_index(drop=True)

        self.baseline_metrics = run_backtest(train_v, train_m, test_v, test_m)
        print(f"Baseline: Accuracy={self.baseline_metrics['accuracy']:.3f}, "
              f"Score Corr={self.baseline_metrics['score_correlation']:.3f}")
        return self.baseline_metrics

    def run_pca_variance_analysis(self) -> Dict[str, Any]:
        X = self.vectors[:, :RAW_FEATURE_COUNT]

        pca = PCA(n_components=RAW_FEATURE_COUNT)
        pca.fit(X)

        cumvar = np.cumsum(pca.explained_variance_ratio_)

        n_90 = int(np.argmax(cumvar >= 0.90)) + 1
        n_95 = int(np.argmax(cumvar >= 0.95)) + 1
        n_99 = int(np.argmax(cumvar >= 0.99)) + 1

        milestones = {}
        for n in [10, 20, 30, 50, 75, 100, 151]:
            if n <= len(cumvar):
                milestones[n] = float(cumvar[n-1])

        return {
            "cumulative_variance": cumvar.tolist(),
            "n_for_90": n_90,
            "n_for_95": n_95,
            "n_for_99": n_99,
            "milestones": milestones,
            "pca_model": pca,
        }

    def run_feature_correlation_analysis(self) -> Dict[str, Any]:
        X = self.vectors[:, :RAW_FEATURE_COUNT]

        corr_matrix = np.corrcoef(X.T)

        redundant_pairs = []
        for i in range(RAW_FEATURE_COUNT):
            for j in range(i+1, RAW_FEATURE_COUNT):
                r = corr_matrix[i, j]
                if not np.isnan(r) and abs(r) > 0.9:
                    redundant_pairs.append({
                        "feature1": FEATURE_NAMES[i],
                        "feature2": FEATURE_NAMES[j],
                        "correlation": float(r),
                        "idx1": i,
                        "idx2": j,
                    })

        redundant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "correlation_matrix": corr_matrix,
            "redundant_pairs": redundant_pairs,
            "n_redundant": len(redundant_pairs),
        }

    def run_feature_variance_analysis(self) -> Dict[str, Any]:
        X = self.vectors[:, :RAW_FEATURE_COUNT]

        variances = np.var(X, axis=0)

        low_var_threshold = 0.01
        low_var_features = []
        for i, var in enumerate(variances):
            if var < low_var_threshold:
                low_var_features.append({
                    "feature": FEATURE_NAMES[i],
                    "variance": float(var),
                    "idx": i,
                })

        variance_ranking = sorted(
            [(FEATURE_NAMES[i], float(var), i) for i, var in enumerate(variances)],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "variances": variances.tolist(),
            "low_variance_features": low_var_features,
            "n_low_variance": len(low_var_features),
            "variance_ranking": variance_ranking[:20],
        }

    def run_leave_one_group_out(self) -> pd.DataFrame:
        results = []

        for group_name, indices in FEATURE_GROUPS.items():
            if group_name == "padding":
                continue

            ablated = self.vectors.copy()
            ablated[:, indices] = 0

            train_v = ablated[self.train_mask]
            train_m = self.metadata[self.train_mask].reset_index(drop=True)
            test_v = ablated[self.test_mask]
            test_m = self.metadata[self.test_mask].reset_index(drop=True)

            metrics = run_backtest(train_v, train_m, test_v, test_m)

            results.append({
                "group": group_name,
                "dims_removed": len(indices),
                "accuracy": metrics["accuracy"],
                "delta_accuracy": self.baseline_metrics["accuracy"] - metrics["accuracy"],
                "score_corr": metrics["score_correlation"],
                "delta_score_corr": self.baseline_metrics["score_correlation"] - metrics["score_correlation"],
            })

        df = pd.DataFrame(results)
        df["importance_rank"] = df["delta_accuracy"].rank(ascending=False)
        return df.sort_values("delta_accuracy", ascending=False)

    def run_leave_one_group_in(self) -> pd.DataFrame:
        results = []

        for group_name, indices in FEATURE_GROUPS.items():
            if group_name == "padding":
                continue

            isolated = np.zeros_like(self.vectors)
            isolated[:, indices] = self.vectors[:, indices]

            train_v = isolated[self.train_mask]
            train_m = self.metadata[self.train_mask].reset_index(drop=True)
            test_v = isolated[self.test_mask]
            test_m = self.metadata[self.test_mask].reset_index(drop=True)

            metrics = run_backtest(train_v, train_m, test_v, test_m)

            results.append({
                "group": group_name,
                "dims": len(indices),
                "accuracy": metrics["accuracy"],
                "score_corr": metrics["score_correlation"],
                "pct_of_baseline": (metrics["accuracy"] / self.baseline_metrics["accuracy"] * 100
                                    if self.baseline_metrics["accuracy"] > 0 else 0),
            })

        return pd.DataFrame(results).sort_values("accuracy", ascending=False)

    def run_semantic_group_ablation(self) -> pd.DataFrame:
        configs = [
            ("Full baseline", list(range(RAW_FEATURE_COUNT))),
            ("WHERE only", SEMANTIC_GROUPS["WHERE"]),
            ("TRAJECTORY only", SEMANTIC_GROUPS["TRAJECTORY"]),
            ("BOOK_PHYSICS only", SEMANTIC_GROUPS["BOOK_PHYSICS"]),
            ("FLOW_PHYSICS only", SEMANTIC_GROUPS["FLOW_PHYSICS"]),
            ("WALLS only", SEMANTIC_GROUPS["WALLS"]),
            ("WHERE + TRAJECTORY", SEMANTIC_GROUPS["WHERE"] + SEMANTIC_GROUPS["TRAJECTORY"]),
            ("WHERE + TRAJECTORY + BOOK",
             SEMANTIC_GROUPS["WHERE"] + SEMANTIC_GROUPS["TRAJECTORY"] + SEMANTIC_GROUPS["BOOK_PHYSICS"]),
            ("WHERE + TRAJECTORY + FLOW",
             SEMANTIC_GROUPS["WHERE"] + SEMANTIC_GROUPS["TRAJECTORY"] + SEMANTIC_GROUPS["FLOW_PHYSICS"]),
            ("All except WALLS", [i for i in range(RAW_FEATURE_COUNT) if i not in SEMANTIC_GROUPS["WALLS"]]),
            ("All except BOOK", [i for i in range(RAW_FEATURE_COUNT) if i not in SEMANTIC_GROUPS["BOOK_PHYSICS"]]),
        ]

        results = []
        for config_name, indices in configs:
            isolated = np.zeros_like(self.vectors)
            isolated[:, indices] = self.vectors[:, indices]

            train_v = isolated[self.train_mask]
            train_m = self.metadata[self.train_mask].reset_index(drop=True)
            test_v = isolated[self.test_mask]
            test_m = self.metadata[self.test_mask].reset_index(drop=True)

            metrics = run_backtest(train_v, train_m, test_v, test_m)

            results.append({
                "configuration": config_name,
                "dims": len(indices),
                "accuracy": metrics["accuracy"],
                "score_corr": metrics["score_correlation"],
                "efficiency": metrics["accuracy"] / len(indices) * 100 if len(indices) > 0 else 0,
            })

        return pd.DataFrame(results)

    def run_predictive_correlations(self) -> pd.DataFrame:
        X = self.vectors[:, :RAW_FEATURE_COUNT]
        y_score = self.metadata["outcome_score"].values

        correlations = []
        for i in range(RAW_FEATURE_COUNT):
            r = np.corrcoef(X[:, i], y_score)[0, 1]
            if np.isnan(r):
                r = 0.0
            correlations.append({
                "feature_idx": i,
                "feature_name": FEATURE_NAMES[i],
                "correlation": r,
                "abs_correlation": abs(r),
            })

        df = pd.DataFrame(correlations)
        df["rank"] = df["abs_correlation"].rank(ascending=False)
        return df.sort_values("abs_correlation", ascending=False)

    def run_distance_contribution_analysis(self, n_pairs: int = 1000) -> pd.DataFrame:
        X = self.vectors[:, :RAW_FEATURE_COUNT]
        n = len(X)

        contributions = np.zeros(RAW_FEATURE_COUNT)

        np.random.seed(42)
        for _ in range(n_pairs):
            i, j = np.random.choice(n, size=2, replace=False)
            squared_diff = (X[i] - X[j]) ** 2
            contributions += squared_diff

        contributions /= contributions.sum()

        results = []
        for i in range(RAW_FEATURE_COUNT):
            results.append({
                "feature_idx": i,
                "feature_name": FEATURE_NAMES[i],
                "pct_distance_contribution": contributions[i] * 100,
            })

        df = pd.DataFrame(results)
        df["rank"] = df["pct_distance_contribution"].rank(ascending=False)
        return df.sort_values("pct_distance_contribution", ascending=False)

    def run_pca_retrieval_performance(self) -> pd.DataFrame:
        X = self.vectors[:, :RAW_FEATURE_COUNT]

        pca_results = self.run_pca_variance_analysis()
        cumvar = pca_results["cumulative_variance"]

        results = []
        for n_comp in [10, 20, 30, 50, 75, 100, 151]:
            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(X)

            X_padded = np.zeros((len(X_reduced), TARGET_DIM))
            X_padded[:, :n_comp] = X_reduced

            train_v = X_padded[self.train_mask]
            train_m = self.metadata[self.train_mask].reset_index(drop=True)
            test_v = X_padded[self.test_mask]
            test_m = self.metadata[self.test_mask].reset_index(drop=True)

            metrics = run_backtest(train_v, train_m, test_v, test_m)

            results.append({
                "n_components": n_comp,
                "variance_explained": cumvar[n_comp - 1] if n_comp <= len(cumvar) else 1.0,
                "accuracy": metrics["accuracy"],
                "score_corr": metrics["score_correlation"],
                "vs_baseline": (metrics["accuracy"] / self.baseline_metrics["accuracy"] * 100
                               if self.baseline_metrics["accuracy"] > 0 else 0),
            })

        return pd.DataFrame(results)

    def run_cosine_vs_l2_comparison(self) -> Dict[str, Dict[str, float]]:
        train_v = self.vectors[self.train_mask].copy()
        train_m = self.metadata[self.train_mask].reset_index(drop=True)
        test_v = self.vectors[self.test_mask].copy()
        test_m = self.metadata[self.test_mask].reset_index(drop=True)

        l2_metrics = run_backtest(train_v.copy(), train_m, test_v.copy(), test_m)

        return {
            "L2": l2_metrics,
            "Cosine (current)": self.baseline_metrics,
        }

    def run_progressive_feature_addition(self) -> pd.DataFrame:
        groups_to_add = [g for g in FEATURE_GROUPS.keys() if g != "padding"]

        results = []
        current_indices = []
        remaining_groups = list(groups_to_add)

        while remaining_groups:
            best_group = None
            best_accuracy = 0
            best_metrics = None

            for group in remaining_groups:
                test_indices = current_indices + FEATURE_GROUPS[group]

                isolated = np.zeros_like(self.vectors)
                isolated[:, test_indices] = self.vectors[:, test_indices]

                train_v = isolated[self.train_mask]
                train_m = self.metadata[self.train_mask].reset_index(drop=True)
                test_v = isolated[self.test_mask]
                test_m = self.metadata[self.test_mask].reset_index(drop=True)

                metrics = run_backtest(train_v, train_m, test_v, test_m)

                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_group = group
                    best_metrics = metrics

            if best_group:
                current_indices.extend(FEATURE_GROUPS[best_group])
                remaining_groups.remove(best_group)

                prev_accuracy = results[-1]["accuracy"] if results else 0.0

                results.append({
                    "step": len(results) + 1,
                    "group_added": best_group,
                    "total_dims": len(current_indices),
                    "accuracy": best_metrics["accuracy"],
                    "marginal_gain": best_metrics["accuracy"] - prev_accuracy,
                    "score_corr": best_metrics["score_correlation"],
                })

        return pd.DataFrame(results)

    def run_all_phase1(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 1: Basic Analysis")
        print("="*60)

        print("\n>>> PCA Variance Analysis...")
        pca_results = self.run_pca_variance_analysis()
        print(f"  Components for 90% variance: {pca_results['n_for_90']}")
        print(f"  Components for 95% variance: {pca_results['n_for_95']}")
        print(f"  Components for 99% variance: {pca_results['n_for_99']}")

        print("\n>>> Feature Correlation Analysis...")
        corr_results = self.run_feature_correlation_analysis()
        print(f"  Found {corr_results['n_redundant']} redundant pairs (|r| > 0.9)")

        print("\n>>> Feature Variance Analysis...")
        var_results = self.run_feature_variance_analysis()
        print(f"  Found {var_results['n_low_variance']} low-variance features (var < 0.01)")

        return {
            "pca": pca_results,
            "correlations": corr_results,
            "variance": var_results,
        }

    def run_all_phase2(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 2: Ablation Studies")
        print("="*60)

        print("\n>>> Leave-One-Group-Out Ablation...")
        loo_results = self.run_leave_one_group_out()
        print(loo_results.to_string())

        print("\n>>> Per-Feature Predictive Correlations...")
        pred_corr = self.run_predictive_correlations()
        print("Top 10 features by |correlation with outcome|:")
        print(pred_corr.head(10).to_string())

        return {
            "leave_one_out": loo_results,
            "predictive_correlations": pred_corr,
        }

    def run_all_phase3(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 3: PCA & Leave-One-In")
        print("="*60)

        print("\n>>> PCA Retrieval Performance...")
        pca_perf = self.run_pca_retrieval_performance()
        print(pca_perf.to_string())

        print("\n>>> Leave-One-Group-In Ablation...")
        loi_results = self.run_leave_one_group_in()
        print(loi_results.to_string())

        return {
            "pca_retrieval": pca_perf,
            "leave_one_in": loi_results,
        }

    def run_all_phase4(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 4: Semantic Groups & Progressive Addition")
        print("="*60)

        print("\n>>> Semantic Group Ablation...")
        semantic_results = self.run_semantic_group_ablation()
        print(semantic_results.to_string())

        print("\n>>> Progressive Feature Addition...")
        progressive_results = self.run_progressive_feature_addition()
        print(progressive_results.to_string())

        return {
            "semantic_groups": semantic_results,
            "progressive_addition": progressive_results,
        }

    def run_all_phase5(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 5: Distance Metrics & Feature Analysis")
        print("="*60)

        print("\n>>> Distance Metric Comparison...")
        metric_comparison = self.run_cosine_vs_l2_comparison()
        for metric, results in metric_comparison.items():
            print(f"  {metric}: Acc={results['accuracy']:.3f}, Corr={results['score_correlation']:.3f}")

        print("\n>>> Distance Contribution Analysis...")
        dist_contrib = self.run_distance_contribution_analysis()
        print("Top 10 features by distance contribution:")
        print(dist_contrib.head(10).to_string())

        return {
            "distance_metrics": metric_comparison,
            "distance_contribution": dist_contrib,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Feature Ablation Studies")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], default=None,
                       help="Run specific phase (default: all)")
    parser.add_argument("--indices-dir", type=str, default=None,
                       help="Path to indices directory")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                       help="Output directory for results")

    args = parser.parse_args()

    indices_dir = Path(args.indices_dir) if args.indices_dir else INDICES_DIR
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    study = FeatureAblationStudy(indices_dir)

    print("Loading data...")
    study.load_data()

    print("\nComputing baseline...")
    baseline = study.compute_baseline()

    all_results = {"baseline": baseline}

    phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4, 5]

    if 1 in phases_to_run:
        all_results["phase1"] = study.run_all_phase1()

    if 2 in phases_to_run:
        all_results["phase2"] = study.run_all_phase2()

    if 3 in phases_to_run:
        all_results["phase3"] = study.run_all_phase3()

    if 4 in phases_to_run:
        all_results["phase4"] = study.run_all_phase4()

    if 5 in phases_to_run:
        all_results["phase5"] = study.run_all_phase5()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline Accuracy: {baseline['accuracy']:.3f}")
    print(f"Baseline Score Correlation: {baseline['score_correlation']:.3f}")
    print(f"Total Episodes: {baseline['n_test'] + len(study.vectors) - baseline['n_test']}")

    return all_results


if __name__ == "__main__":
    main()
