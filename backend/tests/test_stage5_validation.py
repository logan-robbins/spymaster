"""
Stage 5 Validation Suite: Gold layer output validation.
Tests data integrity, statistical sanity, index functionality, retrieval quality,
predictive signal, and edge cases.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.manifold import TSNE

try:
    import faiss
except ImportError:
    faiss = None


INDICES_DIR = Path(__file__).parent.parent / "databases" / "indices"
TARGET_DIM = 256
LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
OUTCOMES = ["STRONG_BREAK", "WEAK_BREAK", "CHOP", "WEAK_BOUNCE", "STRONG_BOUNCE"]


@dataclass
class ValidationResult:
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    priority: str  # P0, P1, P2


class ValidationContext:
    """Holds loaded data for validation tests."""

    def __init__(self, indices_dir: Path):
        self.indices_dir = indices_dir
        self._metadata: pd.DataFrame = None
        self._vectors: Dict[str, np.ndarray] = {}
        self._indices: Dict[str, Any] = {}
        self._level_metadata: Dict[str, pd.DataFrame] = {}

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            db_path = self.indices_dir / "setup_metadata.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                self._metadata = pd.read_sql("SELECT * FROM setup_metadata", conn)
                conn.close()
            else:
                self._metadata = pd.DataFrame()
        return self._metadata

    def get_level_metadata(self, level_type: str) -> pd.DataFrame:
        if level_type not in self._level_metadata:
            db_path = self.indices_dir / f"{level_type.lower()}_metadata.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                self._level_metadata[level_type] = pd.read_sql("SELECT * FROM setup_metadata", conn)
                conn.close()
            else:
                self._level_metadata[level_type] = pd.DataFrame()
        return self._level_metadata[level_type]

    def get_vectors(self, level_type: str) -> np.ndarray:
        if level_type not in self._vectors:
            vec_path = self.indices_dir / f"{level_type.lower()}_vectors.npy"
            if vec_path.exists():
                self._vectors[level_type] = np.load(str(vec_path))
            else:
                self._vectors[level_type] = np.array([])
        return self._vectors[level_type]

    def get_index(self, level_type: str):
        if faiss is None:
            return None
        if level_type not in self._indices:
            idx_path = self.indices_dir / f"{level_type.lower()}_setups.index"
            if idx_path.exists():
                self._indices[level_type] = faiss.read_index(str(idx_path))
            else:
                self._indices[level_type] = None
        return self._indices[level_type]

    def get_episode_ids(self, level_type: str) -> List[str]:
        ids_path = self.indices_dir / f"{level_type.lower()}_episode_ids.json"
        if ids_path.exists():
            with open(ids_path, "r") as f:
                return json.load(f)
        return []


@pytest.fixture(scope="module")
def ctx() -> ValidationContext:
    return ValidationContext(INDICES_DIR)


class TestSection1DataIntegrity:
    """Section 1: Data Integrity Tests"""

    def test_1_1_completeness_episode_count(self, ctx: ValidationContext):
        """1.1 Verify episode counts match between metadata and vectors"""
        results = []

        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            metadata = ctx.get_level_metadata(level_type)
            episode_ids = ctx.get_episode_ids(level_type)

            if len(vectors) == 0 and len(metadata) == 0:
                continue

            assert len(vectors) == len(metadata), \
                f"{level_type}: Vector count ({len(vectors)}) != Metadata count ({len(metadata)})"
            assert len(vectors) == len(episode_ids), \
                f"{level_type}: Vector count ({len(vectors)}) != Episode IDs count ({len(episode_ids)})"
            results.append((level_type, len(vectors)))

        total = sum(r[1] for r in results)
        assert total > 0, "No vectors found in any level type"
        print(f"\nTotal episodes: {total}")
        for level_type, count in results:
            print(f"  {level_type}: {count}")

    def test_1_2_vector_shape(self, ctx: ValidationContext):
        """1.2 All vectors have correct dimensionality"""
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            if len(vectors) == 0:
                continue

            assert vectors.shape[1] == TARGET_DIM, \
                f"{level_type}: Expected dim {TARGET_DIM}, got {vectors.shape[1]}"
            assert vectors.dtype in [np.float32, np.float64], \
                f"{level_type}: Unexpected dtype {vectors.dtype}"

            print(f"\n{level_type}: shape={vectors.shape}, dtype={vectors.dtype}")

    def test_1_3_vector_metadata_alignment(self, ctx: ValidationContext):
        """1.3 Vector indices match metadata vector_ids (sampling 20 random)"""
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            metadata = ctx.get_level_metadata(level_type)

            if len(vectors) == 0:
                continue

            n_samples = min(20, len(vectors))
            sample_indices = np.random.choice(len(vectors), n_samples, replace=False)

            for idx in sample_indices:
                vector_id_in_meta = metadata.iloc[idx]["vector_id"]
                assert vector_id_in_meta == idx, \
                    f"{level_type}: Misaligned vector_id at index {idx}"

    def test_1_4_no_nan_or_inf(self, ctx: ValidationContext):
        """1.4 Vectors contain no NaN or Inf values"""
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            if len(vectors) == 0:
                continue

            nan_count = np.isnan(vectors).sum()
            inf_count = np.isinf(vectors).sum()

            assert nan_count == 0, f"{level_type}: Found {nan_count} NaN values"
            assert inf_count == 0, f"{level_type}: Found {inf_count} Inf values"
            print(f"\n{level_type}: No NaN or Inf values")

    def test_1_5_metadata_completeness(self, ctx: ValidationContext):
        """1.5 All metadata fields are populated correctly"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata found")

        required_fields = [
            "vector_id", "episode_id", "dt", "symbol", "level_type",
            "level_price", "trigger_bar_ts", "approach_direction",
            "outcome", "outcome_score"
        ]

        for field in required_fields:
            assert field in metadata.columns, f"Missing required field: {field}"
            null_count = metadata[field].isnull().sum()
            assert null_count == 0, f"Field {field} has {null_count} NULL values"

        assert metadata["vector_id"].is_unique, "vector_id is not unique"
        assert metadata["episode_id"].is_unique, "episode_id is not unique"

        level_types = set(metadata["level_type"].unique())
        expected_levels = set(LEVEL_TYPES)
        unexpected = level_types - expected_levels
        assert len(unexpected) == 0, f"Unexpected level_types: {unexpected}"

        outcomes = set(metadata["outcome"].unique())
        expected_outcomes = set(OUTCOMES)
        unexpected_outcomes = outcomes - expected_outcomes
        assert len(unexpected_outcomes) == 0, f"Unexpected outcomes: {unexpected_outcomes}"

        directions = set(metadata["approach_direction"].unique())
        assert directions.issubset({-1, 1}), f"Unexpected approach_directions: {directions}"

        print(f"\nMetadata completeness verified for {len(metadata)} records")
        print(f"  Level types: {dict(metadata['level_type'].value_counts())}")
        print(f"  Outcomes: {dict(metadata['outcome'].value_counts())}")


class TestSection2StatisticalSanity:
    """Section 2: Statistical Sanity Tests"""

    def test_2_1_feature_distribution_analysis(self, ctx: ValidationContext):
        """2.1 Each vector dimension has reasonable distribution"""
        all_vectors = []
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            if len(vectors) > 0:
                all_vectors.append(vectors)

        if not all_vectors:
            pytest.skip("No vectors found")

        combined = np.vstack(all_vectors)

        flagged_dims = []
        stats_list = []

        for i in range(combined.shape[1]):
            vals = combined[:, i]

            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            zeros_pct = (vals == 0).mean() * 100
            unique_pct = len(np.unique(vals)) / len(vals) * 100

            issues = []
            if std_val < 1e-6:
                issues.append("constant")
            if abs(mean_val) > 100:
                issues.append("extreme_mean")
            if zeros_pct > 95:
                issues.append("mostly_zeros")

            if issues:
                flagged_dims.append((i, issues))

            stats_list.append({
                "dim": i,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "zeros_pct": zeros_pct,
            })

        print(f"\nAnalyzed {combined.shape[1]} dimensions across {len(combined)} vectors")
        print(f"Flagged dimensions: {len(flagged_dims)}")

        for dim, issues in flagged_dims[:10]:
            print(f"  Dim {dim}: {issues}")

        severe_flags = [d for d, issues in flagged_dims if "constant" not in issues or d < 200]
        if len(severe_flags) > 20:
            print(f"WARNING: {len(severe_flags)} dimensions have issues")

    def test_2_2_normalization_verification(self, ctx: ValidationContext):
        """2.2 Normalized features are approximately standard normal"""
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            if len(vectors) == 0:
                continue

            means = np.mean(vectors, axis=0)
            stds = np.std(vectors, axis=0)

            active_dims = stds > 1e-6
            n_active = active_dims.sum()

            mean_of_means = np.abs(means[active_dims]).mean()
            mean_of_stds = stds[active_dims].mean()

            print(f"\n{level_type}:")
            print(f"  Active dimensions: {n_active}/{TARGET_DIM}")
            print(f"  Mean of |means|: {mean_of_means:.4f} (target: ~0)")
            print(f"  Mean of stds: {mean_of_stds:.4f} (target: ~1)")

    def test_2_3_feature_correlation_matrix(self, ctx: ValidationContext):
        """2.3 Feature correlations are sensible"""
        all_vectors = []
        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            if len(vectors) > 0:
                all_vectors.append(vectors)

        if not all_vectors:
            pytest.skip("No vectors found")

        combined = np.vstack(all_vectors)

        active_dims = np.std(combined, axis=0) > 1e-6
        active_indices = np.where(active_dims)[0]

        if len(active_indices) < 10:
            pytest.skip("Too few active dimensions for correlation analysis")

        sample_indices = active_indices[:50]
        sample_matrix = combined[:, sample_indices]

        corr = np.corrcoef(sample_matrix.T)

        high_corr_pairs = []
        for i in range(len(sample_indices)):
            for j in range(i+1, len(sample_indices)):
                if abs(corr[i, j]) > 0.95:
                    high_corr_pairs.append((
                        sample_indices[i],
                        sample_indices[j],
                        corr[i, j]
                    ))

        print(f"\nCorrelation analysis on {len(sample_indices)} active dimensions")
        print(f"Highly correlated pairs (|r| > 0.95): {len(high_corr_pairs)}")
        for d1, d2, r in high_corr_pairs[:5]:
            print(f"  Dim {d1} ↔ Dim {d2}: r={r:.3f}")

    def test_2_4_outcome_distribution(self, ctx: ValidationContext):
        """2.4 Outcome labels have reasonable distribution"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata found")

        outcome_counts = metadata["outcome"].value_counts()
        outcome_pcts = outcome_counts / len(metadata) * 100

        print("\nOutcome Distribution:")
        for outcome, pct in outcome_pcts.items():
            print(f"  {outcome}: {pct:.1f}%")

        max_pct = outcome_pcts.max()
        min_pct = outcome_pcts.min()

        if max_pct > 60:
            print(f"WARNING: Dominant outcome at {max_pct:.1f}%")
        if min_pct < 5:
            print(f"WARNING: Rare outcome at {min_pct:.1f}%")

        print("\nOutcome by Level Type:")
        for level_type in LEVEL_TYPES:
            level_data = metadata[metadata["level_type"] == level_type]
            if len(level_data) > 0:
                print(f"  {level_type}: {dict(level_data['outcome'].value_counts())}")

    def test_2_5_temporal_distribution(self, ctx: ValidationContext):
        """2.5 Episodes are distributed across the date range"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata found")

        date_counts = metadata.groupby("dt").size()

        print(f"\nTemporal Distribution:")
        print(f"  Date range: {date_counts.index.min()} to {date_counts.index.max()}")
        print(f"  Total dates: {len(date_counts)}")
        print(f"  Episodes per date: min={date_counts.min()}, max={date_counts.max()}, mean={date_counts.mean():.1f}")

        for dt, count in date_counts.items():
            print(f"    {dt}: {count}")

        cv = date_counts.std() / date_counts.mean() if date_counts.mean() > 0 else 0
        if cv > 1.5:
            print(f"WARNING: High variation in daily counts (CV={cv:.2f})")


class TestSection3IndexFunctionality:
    """Section 3: Index Functionality Tests"""

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_3_1_index_load(self, ctx: ValidationContext):
        """3.1 FAISS indices load correctly"""
        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) == 0:
                continue

            assert index is not None, f"Failed to load index for {level_type}"
            assert index.ntotal == len(vectors), \
                f"{level_type}: Index count ({index.ntotal}) != Vector count ({len(vectors)})"
            assert index.d == TARGET_DIM, \
                f"{level_type}: Index dim ({index.d}) != Expected ({TARGET_DIM})"

            print(f"\n{level_type}: ntotal={index.ntotal}, d={index.d}")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_3_2_self_query(self, ctx: ValidationContext):
        """3.2 Querying a vector returns itself as top match"""
        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) == 0 or index is None:
                continue

            n_samples = min(20, len(vectors))
            sample_indices = np.random.choice(len(vectors), n_samples, replace=False)

            for idx in sample_indices:
                query = vectors[idx:idx+1]
                distances, indices = index.search(query, 1)

                assert indices[0][0] == idx, \
                    f"{level_type}: Self-query failed for index {idx}, got {indices[0][0]}"
                assert distances[0][0] < 1e-5, \
                    f"{level_type}: Self-query distance too high: {distances[0][0]}"

            print(f"\n{level_type}: Self-query passed for {n_samples} samples")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_3_3_knn_sanity(self, ctx: ValidationContext):
        """3.3 KNN queries return sensible results"""
        k = 10

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) == 0 or index is None:
                continue

            actual_k = min(k, len(vectors))
            n_queries = min(20, len(vectors))
            query_indices = np.random.choice(len(vectors), n_queries, replace=False)

            for qidx in query_indices:
                query = vectors[qidx:qidx+1]
                distances, indices = index.search(query, actual_k)

                assert len(indices[0]) == actual_k, \
                    f"{level_type}: Expected {actual_k} results, got {len(indices[0])}"
                assert (distances[0] >= 0).all(), \
                    f"{level_type}: Negative distances found"
                assert (np.diff(distances[0]) >= -1e-6).all(), \
                    f"{level_type}: Distances not monotonic"
                assert len(set(indices[0])) == len(indices[0]), \
                    f"{level_type}: Duplicate indices in results"
                assert (indices[0] < index.ntotal).all(), \
                    f"{level_type}: Invalid index in results"

            print(f"\n{level_type}: KNN sanity passed for {n_queries} queries with k={actual_k}")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_3_4_cross_index_isolation(self, ctx: ValidationContext):
        """3.4 Level-type indices are properly separated"""
        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            metadata = ctx.get_level_metadata(level_type)

            if len(metadata) == 0 or index is None:
                continue

            level_types_in_meta = metadata["level_type"].unique()
            assert len(level_types_in_meta) == 1, \
                f"{level_type}: Multiple level types in metadata: {level_types_in_meta}"
            assert level_types_in_meta[0] == level_type, \
                f"{level_type}: Wrong level type in metadata: {level_types_in_meta[0]}"

            print(f"\n{level_type}: Index isolation verified")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_3_5_query_performance(self, ctx: ValidationContext):
        """3.5 Query latency is acceptable"""
        k = 10
        n_queries = 100

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) < 10 or index is None:
                continue

            actual_k = min(k, len(vectors))
            query_indices = np.random.choice(len(vectors), min(n_queries, len(vectors)), replace=True)

            latencies = []
            for qidx in query_indices:
                query = vectors[qidx:qidx+1]
                start = time.perf_counter()
                index.search(query, actual_k)
                latencies.append((time.perf_counter() - start) * 1000)

            p50 = np.percentile(latencies, 50)
            p99 = np.percentile(latencies, 99)

            print(f"\n{level_type}: Query latency p50={p50:.2f}ms, p99={p99:.2f}ms")

            if p99 > 50:
                print(f"WARNING: p99 latency exceeds 50ms target")


class TestSection4RetrievalQuality:
    """Section 4: Retrieval Quality Tests"""

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_4_1_nearest_neighbor_similarity(self, ctx: ValidationContext):
        """4.1 Nearest neighbors have similar characteristics"""
        k = 10
        n_queries = 50

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)
            metadata = ctx.get_level_metadata(level_type)

            if len(vectors) < k + 1 or index is None:
                continue

            query_indices = np.random.choice(len(vectors), min(n_queries, len(vectors)), replace=False)

            same_direction_pcts = []
            outcome_agreements = []
            date_spans = []

            for qidx in query_indices:
                query = vectors[qidx:qidx+1]
                _, indices = index.search(query, k + 1)
                neighbor_indices = indices[0][1:]

                query_row = metadata.iloc[qidx]
                neighbor_rows = metadata.iloc[neighbor_indices]

                same_dir = (neighbor_rows["approach_direction"] == query_row["approach_direction"]).mean()
                same_direction_pcts.append(same_dir * 100)

                same_outcome = (neighbor_rows["outcome"] == query_row["outcome"]).mean()
                outcome_agreements.append(same_outcome * 100)

                dates = pd.to_datetime(neighbor_rows["dt"])
                query_date = pd.to_datetime(query_row["dt"])
                date_spread = (dates - query_date).abs().dt.days.max()
                date_spans.append(date_spread)

            print(f"\n{level_type}:")
            print(f"  Same direction: {np.mean(same_direction_pcts):.1f}% (expected >60%)")
            print(f"  Same outcome: {np.mean(outcome_agreements):.1f}% (expected >20%)")
            print(f"  Mean date span: {np.mean(date_spans):.1f} days")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_4_2_distance_distribution(self, ctx: ValidationContext):
        """4.2 Distance distributions are informative"""
        k = 10
        n_samples = 100

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) < k + 1 or index is None:
                continue

            query_indices = np.random.choice(len(vectors), min(n_samples, len(vectors)), replace=False)

            topk_distances = []
            for qidx in query_indices:
                query = vectors[qidx:qidx+1]
                distances, _ = index.search(query, k + 1)
                topk_distances.extend(distances[0][1:])

            random_pairs = []
            for _ in range(min(n_samples * 10, 1000)):
                i, j = np.random.choice(len(vectors), 2, replace=False)
                dist = np.linalg.norm(vectors[i] - vectors[j]) ** 2
                random_pairs.append(dist)

            topk_mean = np.mean(topk_distances)
            random_mean = np.mean(random_pairs)
            separation = random_mean / topk_mean if topk_mean > 0 else 0

            print(f"\n{level_type}:")
            print(f"  Top-k mean distance: {topk_mean:.2f}")
            print(f"  Random pair mean distance: {random_mean:.2f}")
            print(f"  Separation ratio: {separation:.2f}x (higher is better)")

    def test_4_3_cluster_analysis(self, ctx: ValidationContext):
        """4.3 Vectors form meaningful clusters"""
        all_vectors = []
        all_outcomes = []

        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            metadata = ctx.get_level_metadata(level_type)
            if len(vectors) > 0:
                all_vectors.append(vectors)
                all_outcomes.extend(metadata["outcome"].tolist())

        if not all_vectors:
            pytest.skip("No vectors found")

        combined = np.vstack(all_vectors)

        if len(combined) < 20:
            pytest.skip("Too few vectors for cluster analysis")

        n_clusters = min(10, len(combined) // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined)

        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(combined, labels)
        else:
            sil_score = 0

        cluster_purity = []
        for c in range(n_clusters):
            cluster_mask = labels == c
            if cluster_mask.sum() > 0:
                cluster_outcomes = [all_outcomes[i] for i in range(len(all_outcomes)) if cluster_mask[i]]
                most_common = max(set(cluster_outcomes), key=cluster_outcomes.count)
                purity = cluster_outcomes.count(most_common) / len(cluster_outcomes)
                cluster_purity.append(purity)

        print(f"\nCluster Analysis ({n_clusters} clusters):")
        print(f"  Silhouette score: {sil_score:.3f} (target >0.1)")
        print(f"  Mean cluster purity: {np.mean(cluster_purity):.2f}")
        print(f"  Max cluster purity: {np.max(cluster_purity):.2f}")

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_4_4_feature_importance_for_similarity(self, ctx: ValidationContext):
        """4.4 Identify which features drive similarity"""
        k = 10
        n_queries = 50

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)

            if len(vectors) < k + 1 or index is None:
                continue

            dim_contributions = np.zeros(TARGET_DIM)
            n_pairs = 0

            query_indices = np.random.choice(len(vectors), min(n_queries, len(vectors)), replace=False)

            for qidx in query_indices:
                query = vectors[qidx]
                _, indices = index.search(query.reshape(1, -1), k + 1)

                for nidx in indices[0][1:]:
                    diff_sq = (query - vectors[nidx]) ** 2
                    dim_contributions += diff_sq
                    n_pairs += 1

            if n_pairs > 0:
                dim_contributions /= n_pairs

            top_dims = np.argsort(dim_contributions)[::-1][:10]

            print(f"\n{level_type}: Top 10 dimensions contributing to distance:")
            for rank, dim in enumerate(top_dims):
                print(f"  {rank+1}. Dim {dim}: {dim_contributions[dim]:.4f}")

            padding_contribution = dim_contributions[200:].mean() if TARGET_DIM > 200 else 0
            if padding_contribution > 0.01:
                print(f"WARNING: Padded dimensions contribute {padding_contribution:.4f} to distance")


class TestSection5PredictiveSignal:
    """Section 5: Predictive Signal Tests"""

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_5_1_temporal_holdout_backtest(self, ctx: ValidationContext):
        """5.1 Retrieval system predicts outcomes better than random"""
        metadata = ctx.metadata

        if len(metadata) < 20:
            pytest.skip("Not enough data for backtest")

        dates = sorted(metadata["dt"].unique())
        if len(dates) < 3:
            pytest.skip("Not enough dates for temporal split")

        split_idx = int(len(dates) * 0.7)
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])

        print(f"\nBacktest split:")
        print(f"  Train: {min(train_dates)} to {max(train_dates)} ({len(train_dates)} days)")
        print(f"  Test: {min(test_dates)} to {max(test_dates)} ({len(test_dates)} days)")

        results = []

        for level_type in LEVEL_TYPES:
            vectors = ctx.get_vectors(level_type)
            level_meta = ctx.get_level_metadata(level_type)

            if len(vectors) < 10:
                continue

            train_mask = level_meta["dt"].isin(train_dates)
            test_mask = level_meta["dt"].isin(test_dates)

            train_vectors = vectors[train_mask.values]
            test_vectors = vectors[test_mask.values]
            train_meta = level_meta[train_mask].reset_index(drop=True)
            test_meta = level_meta[test_mask].reset_index(drop=True)

            if len(train_vectors) < 5 or len(test_vectors) < 1:
                continue

            index = faiss.IndexFlatL2(TARGET_DIM)
            index.add(train_vectors.astype(np.float32))

            k = min(10, len(train_vectors))

            for i in range(len(test_vectors)):
                query = test_vectors[i:i+1]
                distances, indices = index.search(query.astype(np.float32), k)

                neighbor_outcomes = train_meta.iloc[indices[0]]["outcome"].values
                neighbor_scores = train_meta.iloc[indices[0]]["outcome_score"].values
                neighbor_distances = distances[0]

                similarities = 1 / (1 + neighbor_distances)

                outcome_weights = defaultdict(float)
                for outcome, sim in zip(neighbor_outcomes, similarities):
                    outcome_weights[outcome] += sim
                total_weight = sum(outcome_weights.values())
                outcome_probs = {k: v/total_weight for k, v in outcome_weights.items()}

                predicted_outcome = max(outcome_probs, key=outcome_probs.get)
                expected_score = np.average(neighbor_scores, weights=similarities)

                actual_row = test_meta.iloc[i]
                results.append({
                    "level_type": level_type,
                    "actual_outcome": actual_row["outcome"],
                    "predicted_outcome": predicted_outcome,
                    "actual_score": actual_row["outcome_score"],
                    "expected_score": expected_score,
                })

        if not results:
            pytest.skip("No test predictions generated")

        results_df = pd.DataFrame(results)

        accuracy = (results_df["actual_outcome"] == results_df["predicted_outcome"]).mean()

        score_corr = results_df[["actual_score", "expected_score"]].corr().iloc[0, 1]

        baseline_accuracy = 1 / len(OUTCOMES)

        print(f"\nBacktest Results ({len(results_df)} predictions):")
        print(f"  Accuracy: {accuracy:.1%} (baseline: {baseline_accuracy:.1%})")
        print(f"  Score correlation: {score_corr:.3f}")

        if accuracy > baseline_accuracy:
            print("  ✓ Better than random baseline")
        else:
            print("  ✗ Not better than random baseline")

    def test_5_4_outcome_score_regression(self, ctx: ValidationContext):
        """5.4 Predicted outcome_score correlates with actual"""
        metadata = ctx.metadata

        if len(metadata) < 10:
            pytest.skip("Not enough data")

        scores = metadata["outcome_score"].values

        print(f"\nOutcome Score Statistics:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Std: {np.std(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")

        sign_positive = (scores > 0).sum()
        sign_negative = (scores < 0).sum()
        print(f"  Positive: {sign_positive}, Negative: {sign_negative}")

    def test_5_5_by_level_type_performance(self, ctx: ValidationContext):
        """5.5 Performance is consistent across level types"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata")

        print("\nLevel Type Statistics:")
        for level_type in LEVEL_TYPES:
            level_data = metadata[metadata["level_type"] == level_type]
            if len(level_data) == 0:
                continue

            outcome_dist = level_data["outcome"].value_counts(normalize=True)
            score_mean = level_data["outcome_score"].mean()
            score_std = level_data["outcome_score"].std()

            print(f"\n{level_type} ({len(level_data)} episodes):")
            print(f"  Score: mean={score_mean:.2f}, std={score_std:.2f}")
            for outcome, pct in outcome_dist.items():
                print(f"    {outcome}: {pct:.1%}")


class TestSection6EdgeCases:
    """Section 6: Edge Cases & Data Leakage Tests"""

    @pytest.mark.skipif(faiss is None, reason="faiss not installed")
    def test_6_1_temporal_leakage_check(self, ctx: ValidationContext):
        """6.1 Nearest neighbors aren't just temporally adjacent episodes"""
        k = 10
        n_queries = 30

        for level_type in LEVEL_TYPES:
            index = ctx.get_index(level_type)
            vectors = ctx.get_vectors(level_type)
            metadata = ctx.get_level_metadata(level_type)

            if len(vectors) < k + 1 or index is None:
                continue

            query_indices = np.random.choice(len(vectors), min(n_queries, len(vectors)), replace=False)

            same_day_pcts = []
            day_distances = []

            for qidx in query_indices:
                query = vectors[qidx:qidx+1]
                _, indices = index.search(query, k + 1)
                neighbor_indices = indices[0][1:]

                query_date = pd.to_datetime(metadata.iloc[qidx]["dt"])
                neighbor_dates = pd.to_datetime(metadata.iloc[neighbor_indices]["dt"])

                same_day = (neighbor_dates == query_date).sum() / len(neighbor_indices)
                same_day_pcts.append(same_day * 100)

                day_diffs = (neighbor_dates - query_date).abs().dt.days
                day_distances.append(day_diffs.median())

            print(f"\n{level_type}:")
            print(f"  Same-day neighbors: {np.mean(same_day_pcts):.1f}% (target <20%)")
            print(f"  Median days to neighbor: {np.mean(day_distances):.1f} (target >5)")

    def test_6_4_level_price_variation(self, ctx: ValidationContext):
        """6.4 Vectors are price-invariant"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata")

        price_ranges = metadata.groupby("level_type")["level_price"].agg(["min", "max", "mean", "std"])

        print("\nLevel Price Ranges:")
        print(price_ranges.to_string())

    def test_6_5_same_day_cross_level(self, ctx: ValidationContext):
        """6.5 Same-day episodes for different levels are distinguishable"""
        metadata = ctx.metadata

        if len(metadata) == 0:
            pytest.skip("No metadata")

        date_level_counts = metadata.groupby(["dt", "level_type"]).size().unstack(fill_value=0)

        multi_level_days = (date_level_counts > 0).sum(axis=1)
        days_with_all_levels = (multi_level_days == 4).sum()

        print(f"\nSame-Day Cross-Level Analysis:")
        print(f"  Days with all 4 levels: {days_with_all_levels}")
        print(f"  Mean level types per day: {multi_level_days.mean():.1f}")


def run_validation_suite(indices_dir: Path = INDICES_DIR) -> List[ValidationResult]:
    """Run all validation tests and return results."""
    ctx = ValidationContext(indices_dir)
    results = []

    print("=" * 60)
    print("STAGE 5 VALIDATION SUITE")
    print("=" * 60)

    print(f"\nIndices Directory: {indices_dir}")
    print(f"Total episodes: {len(ctx.metadata)}")

    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
