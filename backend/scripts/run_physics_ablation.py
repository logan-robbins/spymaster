"""
Ablation Study: Physics vs Geometry (Phase 2)

This script tests the "Mechanism" hypothesis:
1. Physics Only: Uses Velocity/OFI/Force features (Sections B+D)
2. Geometry Only: Uses DCT Trajectory features (Section F)

It compares the predictive power and similarity scaling (Q1->Q4) of each feature set.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import retrieval components
from src.ml.index_builder import build_all_indices, load_episode_corpus
from src.ml.retrieval_engine import IndexManager, EpisodeQuery
from src.ml.outcome_aggregation import aggregate_query_results
from src.ml.constants import K_NEIGHBORS, M_CANDIDATES, VECTOR_SECTIONS, VECTOR_DIMENSION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature Definitions
# Dynamically loaded from constants to support  (or any future version)
def get_indices(section_name: str) -> List[int]:
    start, end = VECTOR_SECTIONS[section_name]
    return list(range(start, end))

PHYSICS_INDICES = get_indices('multiscale_dynamics') + get_indices('derived_physics')
GEOMETRY_INDICES = get_indices('trajectory_basis')

def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates between start and end (inclusive)."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates

def load_test_episodes(episodes_dir: Path, test_dates: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load episode vectors and metadata for test dates."""
    vectors_dir = episodes_dir / 'vectors'
    metadata_dir = episodes_dir / 'metadata'
    
    all_vectors = []
    all_metadata = []
    
    for date_str in test_dates:
        vector_file = vectors_dir / f'date={date_str}' / 'episodes.npy'
        if not vector_file.exists(): continue
        all_vectors.append(np.load(vector_file))
        
        metadata_file = metadata_dir / f'date={date_str}' / 'metadata.parquet'
        if not metadata_file.exists(): continue
        all_metadata.append(pd.read_parquet(metadata_file))
    
    if not all_vectors:
        raise ValueError(f"No test episodes found for dates: {test_dates}")
    
    test_vectors = np.vstack(all_vectors)
    test_metadata = pd.concat(all_metadata, ignore_index=True)
    return test_vectors, test_metadata

def run_ablation_pass(
    run_name: str,
    feature_indices: List[int],
    episodes_dir: Path,
    test_dates: List[str],
    output_base_dir: Path,
    test_vectors: np.ndarray,
    test_metadata: pd.DataFrame
):
    """Run a single ablation pass with specific feature mask."""
    logger.info(f"\n--- Starting Ablation Pass: {run_name} ---")
    logger.info(f"Using {len(feature_indices)} features")
    
    # 1. Build Training Indices with Mask
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_dir = output_base_dir / f'indices_{run_name}_{timestamp}'
    
    # Load corpus to determine training dates
    _, corpus_metadata = load_episode_corpus(episodes_dir)
    train_dates = [d for d in corpus_metadata['date'].astype(str).unique() if d not in test_dates]
    
    logger.info(f"Building indices for {run_name}...")
    build_all_indices(
        episodes_dir=episodes_dir,
        output_dir=index_dir,
        date_filter=train_dates,
        min_partition_size=1,
        overwrite_output_dir=True,
        feature_indices=feature_indices  # PASS MASK HERE
    )
    
    # 2. Retrieve (Bypass SimilarityQueryEngine to skip attribution which assumes )
    index_manager = IndexManager(index_dir)
    
    # Apply mask to test vectors too
    masked_test_vectors = test_vectors[:, feature_indices]
    
    logger.info(f"Running retrieval for {len(masked_test_vectors):,} episodes...")
    
    results = []
    # Masked vectors must be cast to float32 for FAISS
    masked_test_vectors = masked_test_vectors.astype(np.float32)

    for i, (vector, row) in enumerate(zip(masked_test_vectors, test_metadata.itertuples())):
        # Direct Query to IndexManager
        query_result = index_manager.query(
            level_kind=row.level_kind,
            direction=row.direction,
            time_bucket=row.time_bucket,
            query_vector=vector,
            k=50 # Top 50 directly
        )
        
        retrieved_metadata = query_result['metadata']
        
        if len(retrieved_metadata) == 0:
            continue
            
        # Aggregate Outcomes (No attribution)
        aggregated = aggregate_query_results(
            retrieved_metadata=retrieved_metadata,
            query_date=row.timestamp,
            compute_ci=False
        )
        
        probs = aggregated['outcome_probabilities'].get('probabilities', {})
        avg_sim = aggregated['outcome_probabilities'].get('avg_similarity', 0)
        
        results.append({
            'event_id': row.event_id,
            'actual_outcome': row.outcome_4min,
            'predicted_outcome': max(probs, key=probs.get) if probs else 'CHOP',
            'avg_similarity': avg_sim,
            'level_kind': row.level_kind,
            'n_neighbors': len(retrieved_metadata)
        })
        
    results_df = pd.DataFrame(results)
    
    # 3. Compute Simple Metrics
    valid_df = results_df[results_df['n_neighbors'] > 0]
    accuracy = (valid_df['predicted_outcome'] == valid_df['actual_outcome']).mean()
    
    # Breakdown by Level Kind
    level_breakdown = {}
    for kind in valid_df['level_kind'].unique():
        subset = valid_df[valid_df['level_kind'] == kind]
        acc = (subset['predicted_outcome'] == subset['actual_outcome']).mean()
        level_breakdown[kind] = {'accuracy': acc, 'count': len(subset)}

    # Q1 vs Q4
    sim_qs = valid_df['avg_similarity'].quantile([0.25, 0.75])
    q1_acc = (valid_df[valid_df['avg_similarity'] <= sim_qs[0.25]])['predicted_outcome'] == (valid_df[valid_df['avg_similarity'] <= sim_qs[0.25]])['actual_outcome']
    q4_acc = (valid_df[valid_df['avg_similarity'] > sim_qs[0.75]])['predicted_outcome'] == (valid_df[valid_df['avg_similarity'] > sim_qs[0.75]])['actual_outcome']
    
    logger.info(f"Pass {run_name} Results:")
    logger.info(f"  Overall Accuracy: {accuracy:.1%}")
    logger.info(f"  Q1 Accuracy: {q1_acc.mean():.1%}")
    logger.info(f"  Q4 Accuracy: {q4_acc.mean():.1%}")
    
    return {
        'run_name': run_name,
        'accuracy': accuracy,
        'q1_accuracy': q1_acc.mean(),
        'q4_accuracy': q4_acc.mean(),
        'n_features': len(feature_indices),
        'level_breakdown': level_breakdown
    }

def main():
    test_start = "2025-12-01"
    test_end = "2025-12-18"
    version = "3.1.0"
    
    test_dates = generate_date_range(test_start, test_end)
    backend_dir = Path(__file__).parent.parent
    episodes_dir = backend_dir / f'data/gold/episodes/es_level_episodes/version={version}'
    output_dir = backend_dir / 'data/ml/ablation_physics_v_geometry'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading Test Data...")
    test_vectors, test_metadata = load_test_episodes(episodes_dir, test_dates)
    
    results = []
    
    # Pass 1: Physics Only
    results.append(run_ablation_pass(
        "physics_only", PHYSICS_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Pass 2: Geometry Only
    results.append(run_ablation_pass(
        "geometry_only", GEOMETRY_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    # Pass 3: Combined (All Features)
    ALL_INDICES = list(range(VECTOR_DIMENSION))
    results.append(run_ablation_pass(
        "combined", ALL_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Save Report
    report_file = output_dir / 'comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n=== Final Comparison ===")
    for r in results:
        logger.info(f"{r['run_name']}: Acc={r['accuracy']:.1%}, Q4 vs Q1 Delta={(r['q4_accuracy'] - r['q1_accuracy']):.1%}")
        if 'level_breakdown' in r:
             logger.info("  Level Breakdown:")
             for kind, metrics in r['level_breakdown'].items():
                 logger.info(f"    {kind}: Acc={metrics['accuracy']:.1%}")

if __name__ == "__main__":
    main()
