"""Test FAISS similarity retrieval with PCA episode embeddings."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from data_eng.stages.gold.future.episode_embeddings import (
    load_episodes_from_lake,
    extract_all_episode_tensors,
    flatten_tensors,
    fit_scaler_pca,
    create_embeddings_scaled,
)

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: uv add faiss-cpu")
    sys.exit(1)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    return index


def query_similar(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 10,
) -> tuple:
    query = query_embedding.copy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    similarities, indices = index.search(query, k)
    return similarities[0], indices[0]


def analyze_retrieval_quality(
    metadata: list,
    query_idx: int,
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
) -> dict:
    query_meta = metadata[query_idx]
    query_outcome = query_meta["outcome"]

    neighbor_outcomes = [metadata[i]["outcome"] for i in neighbor_indices if i != query_idx]
    neighbor_sims = [similarities[j] for j, i in enumerate(neighbor_indices) if i != query_idx]

    outcome_counts = Counter(neighbor_outcomes)
    total = len(neighbor_outcomes)

    return {
        "query_outcome": query_outcome,
        "query_episode_id": query_meta["episode_id"],
        "avg_similarity": np.mean(neighbor_sims) if neighbor_sims else 0,
        "outcome_distribution": {k: v / total for k, v in outcome_counts.items()} if total > 0 else {},
        "same_outcome_rate": outcome_counts.get(query_outcome, 0) / total if total > 0 else 0,
        "n_neighbors": len(neighbor_outcomes),
    }


def run_leave_one_out_test(
    embeddings: np.ndarray,
    metadata: list,
    k: int = 20,
    sample_size: int = 100,
) -> dict:
    n = len(embeddings)

    sample_indices = np.random.choice(n, min(sample_size, n), replace=False)

    index = build_faiss_index(embeddings)

    results = []
    for query_idx in sample_indices:
        sims, indices = query_similar(index, embeddings[query_idx], k=k + 1)

        mask = indices != query_idx
        indices = indices[mask][:k]
        sims = sims[mask][:k]

        result = analyze_retrieval_quality(metadata, query_idx, indices, sims)
        results.append(result)

    avg_similarity = np.mean([r["avg_similarity"] for r in results])
    avg_same_outcome_rate = np.mean([r["same_outcome_rate"] for r in results])

    outcome_rates_by_query = {}
    for outcome in ["STRONG_BREAK", "WEAK_BREAK", "CHOP", "WEAK_BOUNCE", "STRONG_BOUNCE"]:
        subset = [r for r in results if r["query_outcome"] == outcome]
        if subset:
            outcome_rates_by_query[outcome] = {
                "n_queries": len(subset),
                "same_outcome_rate": np.mean([r["same_outcome_rate"] for r in subset]),
                "avg_similarity": np.mean([r["avg_similarity"] for r in subset]),
            }

    return {
        "n_episodes": n,
        "n_queries_tested": len(results),
        "k_neighbors": k,
        "avg_similarity": avg_similarity,
        "avg_same_outcome_rate": avg_same_outcome_rate,
        "outcome_rates_by_query": outcome_rates_by_query,
        "sample_results": results[:5],
    }


def main():
    lake_path = Path(__file__).parents[4] / "lake"

    esu5_path = lake_path / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_episodes"
    esz5_path = lake_path / "silver/product_type=future/symbol=ESZ5/table=market_by_price_10_pm_high_episodes"
    esh6_path = lake_path / "silver/product_type=future/symbol=ESH6/table=market_by_price_10_pm_high_episodes"

    all_dfs = []
    for symbol, path in [("ESU5", esu5_path), ("ESZ5", esz5_path), ("ESH6", esh6_path)]:
        if path.exists():
            dates = sorted([d.name.replace("dt=", "") for d in path.iterdir() if d.name.startswith("dt=")])
            df = load_episodes_from_lake(lake_path, symbol, "PM_HIGH", dates)
            if len(df) > 0:
                print(f"{symbol}: {df['episode_id'].nunique()} episodes")
                all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    total_episodes = df["episode_id"].nunique()
    print(f"\nTotal episodes: {total_episodes}")

    print("\nExtracting tensors...")
    tensors, metadata, feature_cols = extract_all_episode_tensors(df)
    print(f"Tensor shape: {tensors.shape}")

    print("\nFlattening and fitting StandardScaler + PCA...")
    tensors_flat = flatten_tensors(tensors)
    scaler, pca, pca_stats = fit_scaler_pca(tensors_flat, max_components=min(1024, len(tensors_flat) - 1))

    print("\n=== TESTING DIFFERENT EMBEDDING DIMENSIONS ===")

    for n_dims in [50, 100, 200, 500]:
        if n_dims > pca_stats["n_components_fit"]:
            continue

        print(f"\n--- {n_dims} dimensions ---")
        embeddings = create_embeddings_scaled(tensors_flat, scaler, pca, n_dims)

        variance = pca_stats["variance_explained_curve"][n_dims - 1] if n_dims <= len(pca_stats["variance_explained_curve"]) else 1.0
        print(f"Variance explained: {variance:.1%}")

        results = run_leave_one_out_test(embeddings, metadata, k=20, sample_size=200)

        print(f"Avg similarity: {results['avg_similarity']:.3f}")
        print(f"Same outcome rate: {results['avg_same_outcome_rate']:.1%}")

        print("By outcome:")
        for outcome, stats in sorted(results["outcome_rates_by_query"].items()):
            print(f"  {outcome}: {stats['same_outcome_rate']:.1%} (n={stats['n_queries']}, sim={stats['avg_similarity']:.3f})")

    print("\n=== SAMPLE RETRIEVALS ===")

    embeddings = create_embeddings_scaled(tensors_flat, scaler, pca, 100)
    index = build_faiss_index(embeddings)

    for outcome_type in ["STRONG_BREAK", "STRONG_BOUNCE"]:
        matching = [i for i, m in enumerate(metadata) if m["outcome"] == outcome_type]
        if not matching:
            continue

        query_idx = matching[0]
        sims, indices = query_similar(index, embeddings[query_idx], k=11)

        print(f"\nQuery: {metadata[query_idx]['episode_id']}")
        print(f"Outcome: {metadata[query_idx]['outcome']} (score={metadata[query_idx]['outcome_score']:.2f})")
        print("Top 10 similar episodes:")

        for j, (sim, idx) in enumerate(zip(sims[1:], indices[1:])):
            m = metadata[idx]
            print(f"  {j+1}. sim={sim:.3f} | {m['outcome']} | score={m['outcome_score']:.2f} | {m['episode_id']}")

    print("\n✓ FAISS retrieval test complete")


if __name__ == "__main__":
    np.random.seed(42)
    main()
