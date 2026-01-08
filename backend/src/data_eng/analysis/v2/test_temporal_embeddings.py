"""Test temporal-aware embeddings with derivatives for FAISS retrieval."""
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
    create_temporal_embeddings,
    fit_robust_pca,
    apply_robust_pca,
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


def run_leave_one_out_test(
    embeddings: np.ndarray,
    metadata: list,
    k: int = 20,
    sample_size: int = 200,
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

        query_outcome = metadata[query_idx]["outcome"]
        neighbor_outcomes = [metadata[i]["outcome"] for i in indices]
        outcome_counts = Counter(neighbor_outcomes)
        total = len(neighbor_outcomes)

        results.append({
            "query_outcome": query_outcome,
            "avg_similarity": np.mean(sims) if len(sims) > 0 else 0,
            "same_outcome_rate": outcome_counts.get(query_outcome, 0) / total if total > 0 else 0,
        })

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
        "avg_similarity": avg_similarity,
        "avg_same_outcome_rate": avg_same_outcome_rate,
        "outcome_rates_by_query": outcome_rates_by_query,
    }


def main():
    lake_path = Path(__file__).parents[4] / "lake"

    # Load all PM_HIGH episodes
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
    print(f"Raw tensor shape: {tensors.shape}")  # (n_episodes, 181, 207)

    # Compute outcome baseline (random would be this distribution)
    outcomes = [m["outcome"] for m in metadata]
    outcome_dist = Counter(outcomes)
    print(f"\nOutcome distribution (baseline for random):")
    for outcome, count in sorted(outcome_dist.items()):
        print(f"  {outcome}: {count/len(outcomes):.1%}")

    print("\n" + "="*60)
    print("APPROACH 1: Temporal embeddings WITH derivatives (6 segments)")
    print("="*60)

    temporal_emb, temporal_stats = create_temporal_embeddings(
        tensors, n_segments=6, add_derivatives=True
    )
    print(f"Temporal embedding shape: {temporal_emb.shape}")
    # 207 features * 3 (orig + vel + accel) * 6 segments * 4 stats = 14,904 dims

    # Test WITHOUT PCA first (just RobustScaler)
    print("\n--- No PCA (RobustScaler only) ---")
    from sklearn.preprocessing import RobustScaler
    scaler_only = RobustScaler()
    emb_scaled = scaler_only.fit_transform(
        np.nan_to_num(temporal_emb, nan=0.0, posinf=0.0, neginf=0.0)
    )
    print(f"Scaled embedding dim: {emb_scaled.shape[1]}")

    results = run_leave_one_out_test(emb_scaled, metadata, k=20, sample_size=200)
    print(f"Avg similarity: {results['avg_similarity']:.3f}")
    print(f"Same outcome rate: {results['avg_same_outcome_rate']:.1%}")
    print("By outcome:")
    for outcome, stats in sorted(results["outcome_rates_by_query"].items()):
        print(f"  {outcome}: {stats['same_outcome_rate']:.1%} (n={stats['n_queries']})")

    # Test with PCA at different dimensions
    scaler, pca, pca_stats = fit_robust_pca(temporal_emb)
    print(f"\nPCA stats: 95% variance at {pca_stats['n_components_for_95pct']} dims, "
          f"99% at {pca_stats['n_components_for_99pct']} dims")

    for n_dims in [500, 1000]:
        if n_dims > pca_stats["n_components_fit"]:
            continue
        print(f"\n--- {n_dims} PCA dimensions ---")
        emb_pca = apply_robust_pca(temporal_emb, scaler, pca, n_dims)
        var_idx = min(n_dims - 1, len(pca_stats["variance_curve"]) - 1)
        print(f"Variance explained: {pca_stats['variance_curve'][var_idx]:.1%}")

        results = run_leave_one_out_test(emb_pca, metadata, k=20, sample_size=200)
        print(f"Avg similarity: {results['avg_similarity']:.3f}")
        print(f"Same outcome rate: {results['avg_same_outcome_rate']:.1%}")
        print("By outcome:")
        for outcome, stats in sorted(results["outcome_rates_by_query"].items()):
            print(f"  {outcome}: {stats['same_outcome_rate']:.1%} (n={stats['n_queries']})")

    print("\n" + "="*60)
    print("APPROACH 2: Temporal embeddings WITHOUT derivatives (6 segments)")
    print("="*60)

    temporal_emb_no_deriv, stats_no_deriv = create_temporal_embeddings(
        tensors, n_segments=6, add_derivatives=False
    )
    print(f"Temporal embedding shape: {temporal_emb_no_deriv.shape}")
    # 207 features * 6 segments * 4 stats = 4,968 dims

    scaler2, pca2, pca_stats2 = fit_robust_pca(temporal_emb_no_deriv)
    print(f"PCA stats: 95% at {pca_stats2['n_components_for_95pct']} dims, "
          f"99% at {pca_stats2['n_components_for_99pct']} dims")

    for n_dims in [500, 1000]:
        if n_dims > pca_stats2["n_components_fit"]:
            continue
        print(f"\n--- {n_dims} PCA dimensions ---")
        emb_pca2 = apply_robust_pca(temporal_emb_no_deriv, scaler2, pca2, n_dims)
        var_idx = min(n_dims - 1, len(pca_stats2["variance_curve"]) - 1)
        print(f"Variance explained: {pca_stats2['variance_curve'][var_idx]:.1%}")

        results = run_leave_one_out_test(emb_pca2, metadata, k=20, sample_size=200)
        print(f"Avg similarity: {results['avg_similarity']:.3f}")
        print(f"Same outcome rate: {results['avg_same_outcome_rate']:.1%}")
        print("By outcome:")
        for outcome, stats in sorted(results["outcome_rates_by_query"].items()):
            print(f"  {outcome}: {stats['same_outcome_rate']:.1%} (n={stats['n_queries']})")

    print("\n" + "="*60)
    print("SAMPLE RETRIEVALS (best approach)")
    print("="*60)

    # Use no-PCA temporal embeddings with derivatives
    index = build_faiss_index(emb_scaled)

    for outcome_type in ["STRONG_BREAK", "STRONG_BOUNCE"]:
        matching = [i for i, m in enumerate(metadata) if m["outcome"] == outcome_type]
        if not matching:
            continue

        query_idx = matching[0]
        sims, indices = query_similar(index, emb_scaled[query_idx], k=11)

        print(f"\nQuery: {metadata[query_idx]['episode_id']}")
        print(f"Outcome: {metadata[query_idx]['outcome']} (score={metadata[query_idx]['outcome_score']:.2f})")
        print("Top 10 similar episodes:")

        for j, (sim, idx) in enumerate(zip(sims[1:], indices[1:])):
            m = metadata[idx]
            match = "✓" if m["outcome"] == outcome_type else " "
            print(f"  {j+1}. {match} sim={sim:.3f} | {m['outcome']} | score={m['outcome_score']:.2f}")

    print("\n✓ Temporal embeddings test complete")


if __name__ == "__main__":
    np.random.seed(42)
    main()
