"""
Ablation Study: Physics vs Geometry vs Market Tide

This script tests three hypotheses:
1. Physics Only: Uses Velocity/OFI/Force features
2. Geometry Only: Uses DCT Trajectory features
3. Market Tide Only: Uses Net Premium Flow features (Phase 4.5)

It compares the predictive power, similarity scaling (Q1->Q4), and Calibration Error (ECE) of each feature set.

Usage:
    uv run python scripts/run_physics_ablation.py \\
        --start-date 2025-10-20 \\
        --end-date 2025-09-30 \\
        --version 4.5.0
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
from src.ml.retrieval_engine import IndexManager
from src.ml.outcome_aggregation import aggregate_query_results
from src.ml.constants import VECTOR_SECTIONS, VECTOR_DIMENSION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature Definitions
def get_indices(section_name: str) -> List[int]:
    start, end = VECTOR_SECTIONS[section_name]
    return list(range(start, end))

PHYSICS_INDICES = get_indices('multiscale_dynamics') + get_indices('derived_physics')
GEOMETRY_INDICES = get_indices('trajectory_basis')
# Market Tide is indices 145, 146 (call_tide, put_tide) which are inside derived_physics (134-147)
# But we want to isolate them.
MARKET_TIDE_INDICES = [145, 146] 

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
    
    logger.info(f"Loading episodes from {episodes_dir} for {len(test_dates)} dates...")
    
    for date_str in test_dates:
        vector_file = vectors_dir / f'date={date_str}' / 'episodes.npy'
        if not vector_file.exists(): 
            logger.warning(f"  Missing vectors for {date_str}")
            continue
        all_vectors.append(np.load(vector_file))
        
        metadata_file = metadata_dir / f'date={date_str}' / 'metadata.parquet'
        if not metadata_file.exists(): 
            continue
        all_metadata.append(pd.read_parquet(metadata_file))
    
    if not all_vectors:
        raise ValueError(f"No test episodes found for dates: {test_dates}")
    
    test_vectors = np.vstack(all_vectors)
    test_metadata = pd.concat(all_metadata, ignore_index=True)
    logger.info(f"Loaded {len(test_vectors):,} episodes total.")
    return test_vectors, test_metadata

def compute_ece(predictions_df: pd.DataFrame, n_bins: int = 10) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) per outcome."""
    ece_results = {}
    outcomes = ['BREAK', 'REJECT', 'CHOP']
    
    for outcome in outcomes:
        prob_col = f'prob_{outcome.lower()}'
        if prob_col not in predictions_df.columns:
            # If naive retrieval has no probs, skip
            continue
            
        actual_binary = (predictions_df['actual_outcome'] == outcome).astype(int)
        predicted_probs = predictions_df[prob_col].values
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        ece = 0.0
        total_samples = len(predictions_df)
        
        if total_samples > 0:
            for b in range(n_bins):
                count = bin_counts[b]
                if count > 0:
                    mask = bin_indices == b
                    avg_prob = predicted_probs[mask].mean()
                    avg_acc = actual_binary[mask].mean()
                    ece += (count / total_samples) * abs(avg_acc - avg_prob)
        
        ece_results[outcome] = ece
        
    return ece_results

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
    logger.info(f"Using {len(feature_indices)} features. Indices: {feature_indices[:5]}...")
    
    # 1. Build Training Indices with Mask
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_dir = output_base_dir / f'indices_{run_name}_{timestamp}'
    
    # Load corpus to determine training dates
    _, corpus_metadata = load_episode_corpus(episodes_dir)
    train_dates = [d for d in corpus_metadata['date'].astype(str).unique() if d not in test_dates]
    
    if not train_dates:
        logger.warning("No training dates found (test set covers all available data?). Using LEAKY self-retrieval for debugging.")
        train_dates = test_dates # Fallback for pure debug if only 1 day exists
    
    logger.info(f"Building indices on {len(train_dates)} training dates...")
    build_all_indices(
        episodes_dir=episodes_dir,
        output_dir=index_dir,
        date_filter=train_dates,
        min_partition_size=1,
        overwrite_output_dir=True,
        feature_indices=feature_indices  # PASS MASK HERE
    )
    
    # 2. Retrieve 
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
        
        probs = {}
        avg_sim = 0.0
        
        if len(retrieved_metadata) > 0:
            # Aggregate Outcomes
            aggregated = aggregate_query_results(
                retrieved_metadata=retrieved_metadata,
                query_date=row.timestamp,
                compute_ci=False
            )
            probs = aggregated['outcome_probabilities'].get('probabilities', {})
            avg_sim = aggregated['outcome_probabilities'].get('avg_similarity', 0)
        else:
            # Default fallback
            probs = {'BREAK': 0.33, 'REJECT': 0.33, 'CHOP': 0.33} 
            avg_sim = 0.0
        
        results.append({
            'event_id': row.event_id,
            'actual_outcome': row.outcome_4min,
            'predicted_outcome': max(probs, key=probs.get) if probs else 'CHOP',
            'prob_break': probs.get('BREAK', 0),
            'prob_reject': probs.get('REJECT', 0),
            'prob_chop': probs.get('CHOP', 0),
            'avg_similarity': avg_sim,
            'level_kind': row.level_kind,
            'n_neighbors': len(retrieved_metadata)
        })
        
    results_df = pd.DataFrame(results)
    
    # 3. Compute Metrics
    valid_df = results_df[results_df['n_neighbors'] > 0]
    accuracy = 0.0
    if len(valid_df) > 0:
        accuracy = (valid_df['predicted_outcome'] == valid_df['actual_outcome']).mean()
    
    # Compute Calibration ECE
    ece_metrics = compute_ece(valid_df)
    
    # Q1 vs Q4
    q1_acc = 0.0
    q4_acc = 0.0
    
    if len(valid_df) > 10:
        sim_qs = valid_df['avg_similarity'].quantile([0.25, 0.75])
        q1_mask = valid_df['avg_similarity'] <= sim_qs[0.25]
        q4_mask = valid_df['avg_similarity'] > sim_qs[0.75]
        
        if q1_mask.sum() > 0:
            q1_acc = (valid_df[q1_mask]['predicted_outcome'] == valid_df[q1_mask]['actual_outcome']).mean()
        if q4_mask.sum() > 0:
            q4_acc = (valid_df[q4_mask]['predicted_outcome'] == valid_df[q4_mask]['actual_outcome']).mean()
    
    logger.info(f"Pass {run_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.1%}")
    logger.info(f"  ECE (Reject): {ece_metrics.get('REJECT', 0.0):.3f}")
    logger.info(f"  Q4 vs Q1 Delta: {(q4_acc - q1_acc):.1%}")
    
    return {
        'run_name': run_name,
        'accuracy': accuracy,
        'ece_reject': ece_metrics.get('REJECT', 0.0),
        'q1_accuracy': q1_acc,
        'q4_accuracy': q4_acc,
        'n_features': len(feature_indices),
        'vector_subset': run_name
    }

def main():
    parser = argparse.ArgumentParser(description='Run Physics vs Geometry vs Tide ablation')
    parser.add_argument('--start-date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--version', type=str, required=True, help='Canonical version')
    parser.add_argument('--episodes-dir', type=str, default=None, help='Override episodes dir')
    
    args = parser.parse_args()
    
    test_dates = generate_date_range(args.start_date, args.end_date)
    
    backend_dir = Path(__file__).parent.parent
    if args.episodes_dir:
        episodes_dir = Path(args.episodes_dir)
    else:
        episodes_dir = backend_dir / f'data/gold/episodes/es_level_episodes/version={args.version}'
        
    output_dir = backend_dir / 'data/ml/ablation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Ablation Study on {len(test_dates)} dates...")
    test_vectors, test_metadata = load_test_episodes(episodes_dir, test_dates)
    
    results = []
    
    # Pass 1: Geometry Only (Baseline)
    results.append(run_ablation_pass(
        "geometry_only", GEOMETRY_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Pass 2: Physics Only
    results.append(run_ablation_pass(
        "physics_only", PHYSICS_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Pass 3: Market Tide Only
    results.append(run_ablation_pass(
        "market_tide", MARKET_TIDE_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Pass 4: Combined
    ALL_INDICES = list(range(VECTOR_DIMENSION))
    results.append(run_ablation_pass(
        "combined", ALL_INDICES, episodes_dir, test_dates, output_dir, test_vectors, test_metadata
    ))
    
    # Save Report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'comparison_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n=== Final Comparison ===")
    print(f"{'Run':<15} | {'Acc':<8} | {'ECE(R)':<8} | {'Q4-Q1':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['run_name']:<15} | {r['accuracy']:.1%}   | {r['ece_reject']:.3f}    | {(r['q4_accuracy'] - r['q1_accuracy']):.1%}")

if __name__ == "__main__":
    main()
