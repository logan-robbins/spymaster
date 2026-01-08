"""Proper evaluation of retrieval quality with precision@k, per-level breakdown."""
from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from data_eng.stages.gold.future.episode_embeddings import (
    load_episodes_from_lake,
    extract_all_episode_tensors,
    create_temporal_embeddings,
    fit_robust_pca,
    apply_robust_pca,
)

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: uv add faiss-cpu")
    sys.exit(1)


# Directional grouping
def get_direction(outcome: str) -> str:
    if outcome in ("STRONG_BREAK", "WEAK_BREAK"):
        return "BREAK"
    elif outcome in ("STRONG_BOUNCE", "WEAK_BOUNCE"):
        return "BOUNCE"
    else:
        return "CHOP"


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def compute_precision_at_k(
    embeddings: np.ndarray,
    metadata: list,
    k_values: list = [5, 10, 20],
    sample_size: int = None,
    match_field: str = "outcome",  # or "direction"
) -> dict:
    """Compute precision@k for retrieval."""
    n = len(embeddings)
    if sample_size is None:
        sample_size = n

    sample_indices = np.random.choice(n, min(sample_size, n), replace=False)

    # Build index
    emb_f32 = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(emb_f32)
    index = faiss.IndexFlatIP(emb_f32.shape[1])
    index.add(emb_f32)

    max_k = max(k_values) + 1

    results = {k: [] for k in k_values}

    for query_idx in sample_indices:
        query = emb_f32[query_idx:query_idx+1].copy()
        sims, indices = index.search(query, max_k)
        indices = indices[0]

        # Exclude self
        indices = [i for i in indices if i != query_idx][:max(k_values)]

        if match_field == "direction":
            query_label = get_direction(metadata[query_idx]["outcome"])
            neighbor_labels = [get_direction(metadata[i]["outcome"]) for i in indices]
        else:
            query_label = metadata[query_idx][match_field]
            neighbor_labels = [metadata[i][match_field] for i in indices]

        for k in k_values:
            matches = sum(1 for lbl in neighbor_labels[:k] if lbl == query_label)
            precision = matches / k
            results[k].append(precision)

    return {k: np.mean(v) for k, v in results.items()}


def compute_random_baseline(metadata: list, k_values: list, match_field: str = "outcome") -> dict:
    """Expected precision if retrieval were random."""
    if match_field == "direction":
        labels = [get_direction(m["outcome"]) for m in metadata]
    else:
        labels = [m[match_field] for m in metadata]

    counts = Counter(labels)
    n = len(labels)

    # Expected precision = sum of (p_i)^2 where p_i is class proportion
    # This is because P(random neighbor matches) = sum_i P(query=i) * P(neighbor=i)
    expected = sum((c/n)**2 for c in counts.values())

    return {k: expected for k in k_values}


def load_all_episodes(lake_path: Path, level_type: str) -> tuple:
    """Load episodes for a specific level type across all symbols."""
    symbols = ["ESU5", "ESZ5", "ESH6"]
    all_dfs = []

    for symbol in symbols:
        table_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type.lower()}_episodes"
        if table_path.exists():
            dates = sorted([d.name.replace("dt=", "") for d in table_path.iterdir() if d.name.startswith("dt=")])
            df = load_episodes_from_lake(lake_path, symbol, level_type, dates)
            if len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        return None, None, None

    df = pd.concat(all_dfs, ignore_index=True)
    tensors, metadata, feature_cols = extract_all_episode_tensors(df)

    return tensors, metadata, feature_cols


def evaluate_level(
    lake_path: Path,
    level_type: str,
    k_values: list = [5, 10, 20],
) -> dict:
    """Full evaluation for one level type."""
    print(f"\n{'='*60}")
    print(f"LEVEL: {level_type}")
    print(f"{'='*60}")

    tensors, metadata, feature_cols = load_all_episodes(lake_path, level_type)

    if tensors is None or len(tensors) == 0:
        print(f"  No data for {level_type}")
        return None

    n_episodes = len(tensors)
    print(f"Episodes: {n_episodes}")

    # Outcome distribution
    outcomes = [m["outcome"] for m in metadata]
    directions = [get_direction(o) for o in outcomes]

    print(f"\nOutcome distribution:")
    for outcome, count in sorted(Counter(outcomes).items()):
        print(f"  {outcome}: {count} ({count/n_episodes:.1%})")

    print(f"\nDirection distribution:")
    for direction, count in sorted(Counter(directions).items()):
        print(f"  {direction}: {count} ({count/n_episodes:.1%})")

    # Create embeddings
    temporal_emb, _ = create_temporal_embeddings(tensors, n_segments=6, add_derivatives=True)

    # Use RobustScaler only (no PCA) for maximum information retention
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    emb_scaled = scaler.fit_transform(
        np.nan_to_num(temporal_emb, nan=0.0, posinf=0.0, neginf=0.0)
    )

    print(f"\nEmbedding dim: {emb_scaled.shape[1]}")

    # Compute precision@k for outcomes
    print(f"\n--- OUTCOME MATCHING (5 classes) ---")
    outcome_precision = compute_precision_at_k(emb_scaled, metadata, k_values, match_field="outcome")
    outcome_baseline = compute_random_baseline(metadata, k_values, match_field="outcome")

    print(f"{'k':<6} {'Precision':<12} {'Random':<12} {'Lift':<12}")
    print("-" * 42)
    for k in k_values:
        lift = outcome_precision[k] / outcome_baseline[k] if outcome_baseline[k] > 0 else 0
        print(f"{k:<6} {outcome_precision[k]:<12.1%} {outcome_baseline[k]:<12.1%} {lift:<12.2f}x")

    # Compute precision@k for direction
    print(f"\n--- DIRECTION MATCHING (BREAK/BOUNCE/CHOP) ---")
    direction_precision = compute_precision_at_k(emb_scaled, metadata, k_values, match_field="direction")
    direction_baseline = compute_random_baseline(metadata, k_values, match_field="direction")

    print(f"{'k':<6} {'Precision':<12} {'Random':<12} {'Lift':<12}")
    print("-" * 42)
    for k in k_values:
        lift = direction_precision[k] / direction_baseline[k] if direction_baseline[k] > 0 else 0
        print(f"{k:<6} {direction_precision[k]:<12.1%} {direction_baseline[k]:<12.1%} {lift:<12.2f}x")

    # Per-outcome precision breakdown
    print(f"\n--- PRECISION@10 BY QUERY OUTCOME ---")

    # Group by query outcome
    emb_f32 = emb_scaled.astype(np.float32).copy()
    faiss.normalize_L2(emb_f32)
    index = faiss.IndexFlatIP(emb_f32.shape[1])
    index.add(emb_f32)

    outcome_precisions = defaultdict(list)

    for query_idx in range(len(metadata)):
        query = emb_f32[query_idx:query_idx+1].copy()
        sims, indices = index.search(query, 11)
        indices = [i for i in indices[0] if i != query_idx][:10]

        query_outcome = metadata[query_idx]["outcome"]
        matches = sum(1 for i in indices if metadata[i]["outcome"] == query_outcome)
        outcome_precisions[query_outcome].append(matches / 10)

    # Get baseline per outcome
    outcome_counts = Counter(outcomes)

    print(f"{'Outcome':<15} {'N':<8} {'P@10':<10} {'Baseline':<10} {'Lift':<8}")
    print("-" * 51)
    for outcome in ["STRONG_BREAK", "WEAK_BREAK", "CHOP", "WEAK_BOUNCE", "STRONG_BOUNCE"]:
        if outcome in outcome_precisions:
            p10 = np.mean(outcome_precisions[outcome])
            baseline = outcome_counts[outcome] / n_episodes
            lift = p10 / baseline if baseline > 0 else 0
            print(f"{outcome:<15} {len(outcome_precisions[outcome]):<8} {p10:<10.1%} {baseline:<10.1%} {lift:<8.2f}x")

    return {
        "level_type": level_type,
        "n_episodes": n_episodes,
        "outcome_precision": outcome_precision,
        "direction_precision": direction_precision,
        "outcome_baseline": outcome_baseline,
        "direction_baseline": direction_baseline,
    }


def main():
    lake_path = Path(__file__).parents[4] / "lake"

    k_values = [5, 10, 20]

    all_results = {}

    for level_type in ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]:
        result = evaluate_level(lake_path, level_type, k_values)
        if result:
            all_results[level_type] = result

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: DIRECTION P@10 LIFT BY LEVEL")
    print("="*60)
    print(f"{'Level':<12} {'Episodes':<10} {'P@10':<10} {'Baseline':<10} {'Lift':<8}")
    print("-" * 50)

    for level_type, result in all_results.items():
        p10 = result["direction_precision"][10]
        baseline = result["direction_baseline"][10]
        lift = p10 / baseline if baseline > 0 else 0
        print(f"{level_type:<12} {result['n_episodes']:<10} {p10:<10.1%} {baseline:<10.1%} {lift:<8.2f}x")

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    np.random.seed(42)
    main()
