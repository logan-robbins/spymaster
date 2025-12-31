"""
Ablation Study: Evaluate Retrieval System Predictive Power

This script evaluates whether the similarity retrieval system can accurately
predict outcomes (BREAK/REJECT/CHOP) by:

1. Splitting data into train/test sets by date
2. Building FAISS indices from training data only (excluding test dates)
3. For each test episode, retrieving similar historical episodes
4. Computing predicted outcome distribution from retrieved neighbors
5. Comparing predicted outcomes vs actual outcomes
6. Computing classification metrics (accuracy, precision, recall, F1, calibration)

Usage:
    # Test on specific dates (e.g., 12/15 and 12/18/2025)
    uv run python scripts/run_ablation_study.py \
        --test-dates 2025-12-15,2025-12-18 \
        --canonical-version 3.1.0 \
        --output-dir data/ml/ablation_results

    # Test on date range
    uv run python scripts/run_ablation_study.py \
        --test-start 2025-12-15 \
        --test-end 2025-12-18 \
        --canonical-version 3.1.0

Per IMPLEMENTATION_READY.md:
- Uses same retrieval engine as production
- Applies deduplication (max 2/day, 1/episode)
- Returns top K_NEIGHBORS (50) after M_CANDIDATES (500) over-fetch
- Computes weighted outcome probabilities (similarity^4 * recency decay)
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
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import retrieval components
from src.ml.index_builder import build_all_indices, load_episode_corpus
from src.ml.retrieval_engine import IndexManager, SimilarityQueryEngine, EpisodeQuery
from src.ml.outcome_aggregation import aggregate_query_results
from src.ml.episode_vector import get_feature_names, assign_time_bucket
from src.ml.constants import K_NEIGHBORS, M_CANDIDATES
from src.ml.tracking import tracking_run, log_metrics, log_artifacts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date_list(date_str: str) -> List[str]:
    """Parse comma-separated date list."""
    return [d.strip() for d in date_str.split(',') if d.strip()]


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


def load_test_episodes(
    episodes_dir: Path,
    test_dates: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load episode vectors and metadata for test dates.
    
    Args:
        episodes_dir: Episode corpus directory
        test_dates: List of test dates (YYYY-MM-DD)
    
    Returns:
        Tuple of (vectors [N × 144], metadata DataFrame)
    """
    logger.info(f"Loading test episodes for {len(test_dates)} dates...")
    
    vectors_dir = episodes_dir / 'vectors'
    metadata_dir = episodes_dir / 'metadata'
    
    all_vectors = []
    all_metadata = []
    
    for date_str in test_dates:
        # Load vectors
        vector_file = vectors_dir / f'date={date_str}' / 'episodes.npy'
        if not vector_file.exists():
            logger.warning(f"  Vectors not found for {date_str}, skipping")
            continue
        
        vectors = np.load(vector_file)
        all_vectors.append(vectors)
        
        # Load metadata
        metadata_file = metadata_dir / f'date={date_str}' / 'metadata.parquet'
        if not metadata_file.exists():
            logger.warning(f"  Metadata not found for {date_str}, skipping")
            continue
        
        metadata = pd.read_parquet(metadata_file)
        all_metadata.append(metadata)
    
    if not all_vectors:
        raise ValueError(f"No test episodes found for dates: {test_dates}")
    
    test_vectors = np.vstack(all_vectors)
    test_metadata = pd.concat(all_metadata, ignore_index=True)
    
    logger.info(f"  Loaded {len(test_vectors):,} test episodes")
    
    return test_vectors, test_metadata


def build_training_indices(
    episodes_dir: Path,
    index_dir: Path,
    test_dates: List[str]
) -> Dict[str, Any]:
    """
    Build FAISS indices from training data (excluding test dates).
    
    Args:
        episodes_dir: Episode corpus directory
        index_dir: Output directory for indices
        test_dates: List of test dates to exclude
    
    Returns:
        Dict with build statistics
    """
    logger.info("Building training indices (excluding test dates)...")
    logger.info(f"  Excluding test dates: {test_dates}")
    
    # Load full corpus
    corpus_vectors, corpus_metadata = load_episode_corpus(episodes_dir)
    
    # Filter out test dates
    train_mask = ~corpus_metadata['date'].astype(str).isin(test_dates)
    train_dates = corpus_metadata.loc[train_mask, 'date'].unique()
    
    logger.info(f"  Training corpus: {train_mask.sum():,} episodes from {len(train_dates)} dates")
    logger.info(f"  Test corpus: {(~train_mask).sum():,} episodes from {len(test_dates)} dates")
    
    # Build indices using only training dates
    result = build_all_indices(
        episodes_dir=episodes_dir,
        output_dir=index_dir,
        date_filter=[str(d) for d in train_dates],
        min_partition_size=1,  # Rare pattern detection - even 1-2 similar episodes are valuable
        overwrite_output_dir=True
    )
    
    return result


def run_retrieval_for_test_episodes(
    test_vectors: np.ndarray,
    test_metadata: pd.DataFrame,
    index_manager: IndexManager,
    k_neighbors: int = K_NEIGHBORS
) -> pd.DataFrame:
    """
    Run retrieval for all test episodes and collect predictions.
    
    Args:
        test_vectors: Test episode vectors [N × 144]
        test_metadata: Test episode metadata
        index_manager: IndexManager with training indices loaded
        k_neighbors: Number of neighbors to retrieve
    
    Returns:
        DataFrame with predictions for each test episode
    """
    logger.info(f"Running retrieval for {len(test_vectors):,} test episodes...")
    
    query_engine = SimilarityQueryEngine(
        index_manager=index_manager,
        k_neighbors=k_neighbors,
        m_candidates=M_CANDIDATES
    )
    
    results = []
    
    for i, (vector, row) in enumerate(zip(test_vectors, test_metadata.itertuples())):
        if i % 100 == 0:
            logger.info(f"  Processing episode {i+1}/{len(test_vectors)}...")
        
        # Build episode query
        episode_query = EpisodeQuery(
            level_kind=row.level_kind,
            level_price=row.level_price,
            direction=row.direction,
            time_bucket=row.time_bucket,
            vector=vector,
            emission_weight=getattr(row, 'emission_weight', 1.0),
            timestamp=row.timestamp,
            metadata={'event_id': row.event_id}
        )
        
        # Execute query
        query_result = query_engine.query(episode_query)
        
        # Extract predicted probabilities
        probs = query_result.outcome_probabilities.get('probabilities', {})
        n_retrieved = len(query_result.neighbors)
        avg_similarity = query_result.reliability.get('avg_similarity', 0)
        
        # Determine predicted outcome (argmax)
        if probs:
            predicted_outcome = max(probs, key=probs.get)
            predicted_prob = probs.get(predicted_outcome, 0)
        else:
            predicted_outcome = 'CHOP'  # Default if no neighbors
            predicted_prob = 0
        
        # Collect result
        results.append({
            'event_id': row.event_id,
            'date': str(row.date),
            'timestamp': row.timestamp,
            'level_kind': row.level_kind,
            'direction': row.direction,
            'time_bucket': row.time_bucket,
            'actual_outcome': row.outcome_4min,
            'predicted_outcome': predicted_outcome,
            'prob_break': probs.get('BREAK', 0),
            'prob_reject': probs.get('REJECT', 0),
            'prob_chop': probs.get('CHOP', 0),
            'predicted_prob': predicted_prob,
            'n_neighbors': n_retrieved,
            'avg_similarity': avg_similarity,
            'emission_weight': getattr(row, 'emission_weight', 1.0)
        })
    
    return pd.DataFrame(results)


def compute_calibration(
    predictions_df: pd.DataFrame,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics (reliability diagram data).
    
    Args:
        predictions_df: DataFrame with actual and predicted outcomes
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dict with calibration metrics per outcome class
    """
    calibration_results = {}
    
    for outcome in ['BREAK', 'REJECT', 'CHOP']:
        prob_col = f'prob_{outcome.lower()}'
        actual_binary = (predictions_df['actual_outcome'] == outcome).astype(int)
        predicted_probs = predictions_df[prob_col].values
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute calibration per bin
        bin_means = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_means.append(predicted_probs[mask].mean())
                bin_accuracies.append(actual_binary[mask].mean())
                bin_counts.append(mask.sum())
        
        # Expected Calibration Error (ECE)
        if bin_counts:
            total_samples = sum(bin_counts)
            ece = sum(
                (count / total_samples) * abs(acc - mean)
                for count, acc, mean in zip(bin_counts, bin_accuracies, bin_means)
            )
        else:
            ece = 0
        
        calibration_results[outcome] = {
            'ece': ece,
            'bin_means': bin_means,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts
        }
    
    return calibration_results


def compute_metrics(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions_df: DataFrame with actual and predicted outcomes
    
    Returns:
        Dict with all evaluation metrics
    """
    logger.info("Computing evaluation metrics...")
    
    # Filter to episodes with neighbors (can't evaluate if no retrieval)
    valid_mask = predictions_df['n_neighbors'] > 0
    valid_df = predictions_df[valid_mask]
    
    logger.info(f"  Valid episodes: {len(valid_df):,} / {len(predictions_df):,} ({100*len(valid_df)/len(predictions_df):.1f}%)")
    
    if len(valid_df) == 0:
        logger.error("No valid predictions to evaluate!")
        return {}
    
    y_true = valid_df['actual_outcome'].values
    y_pred = valid_df['predicted_outcome'].values
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['BREAK', 'REJECT', 'CHOP'], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['BREAK', 'REJECT', 'CHOP'])
    
    # Classification report
    report = classification_report(
        y_true, y_pred, labels=['BREAK', 'REJECT', 'CHOP'], output_dict=True
    )
    
    # Calibration
    calibration = compute_calibration(valid_df)
    
    # Stratified metrics
    stratified = {}
    
    # By level kind
    for level_kind in valid_df['level_kind'].unique():
        mask = valid_df['level_kind'] == level_kind
        if mask.sum() > 10:  # Require at least 10 samples
            stratified[f'level_{level_kind}'] = {
                'accuracy': accuracy_score(
                    valid_df.loc[mask, 'actual_outcome'],
                    valid_df.loc[mask, 'predicted_outcome']
                ),
                'n_samples': int(mask.sum())
            }
    
    # By direction
    for direction in valid_df['direction'].unique():
        mask = valid_df['direction'] == direction
        if mask.sum() > 10:
            stratified[f'direction_{direction}'] = {
                'accuracy': accuracy_score(
                    valid_df.loc[mask, 'actual_outcome'],
                    valid_df.loc[mask, 'predicted_outcome']
                ),
                'n_samples': int(mask.sum())
            }
    
    # By time bucket
    for bucket in valid_df['time_bucket'].unique():
        mask = valid_df['time_bucket'] == bucket
        if mask.sum() > 10:
            stratified[f'time_{bucket}'] = {
                'accuracy': accuracy_score(
                    valid_df.loc[mask, 'actual_outcome'],
                    valid_df.loc[mask, 'predicted_outcome']
                ),
                'n_samples': int(mask.sum())
            }
    
    # Reliability analysis
    # Group by similarity quality
    similarity_quantiles = valid_df['avg_similarity'].quantile([0.25, 0.5, 0.75])
    
    for q_name, q_val in [('Q1_low', 0), ('Q2', similarity_quantiles[0.25]), 
                          ('Q3', similarity_quantiles[0.5]), ('Q4_high', similarity_quantiles[0.75])]:
        if q_name == 'Q1_low':
            mask = valid_df['avg_similarity'] <= similarity_quantiles[0.25]
        elif q_name == 'Q2':
            mask = (valid_df['avg_similarity'] > similarity_quantiles[0.25]) & \
                   (valid_df['avg_similarity'] <= similarity_quantiles[0.5])
        elif q_name == 'Q3':
            mask = (valid_df['avg_similarity'] > similarity_quantiles[0.5]) & \
                   (valid_df['avg_similarity'] <= similarity_quantiles[0.75])
        else:  # Q4_high
            mask = valid_df['avg_similarity'] > similarity_quantiles[0.75]
        
        if mask.sum() > 10:
            stratified[f'similarity_{q_name}'] = {
                'accuracy': accuracy_score(
                    valid_df.loc[mask, 'actual_outcome'],
                    valid_df.loc[mask, 'predicted_outcome']
                ),
                'n_samples': int(mask.sum()),
                'avg_similarity': float(valid_df.loc[mask, 'avg_similarity'].mean())
            }
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'n_samples': int(len(valid_df)),
            'n_total': int(len(predictions_df)),
            'coverage': float(len(valid_df) / len(predictions_df))
        },
        'per_class': {
            'BREAK': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'REJECT': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            },
            'CHOP': {
                'precision': float(precision[2]),
                'recall': float(recall[2]),
                'f1': float(f1[2]),
                'support': int(support[2])
            }
        },
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': ['BREAK', 'REJECT', 'CHOP'],
        'classification_report': report,
        'calibration': calibration,
        'stratified_metrics': stratified
    }
    
    return metrics


def print_summary(metrics: Dict[str, Any], predictions_df: pd.DataFrame):
    """Print human-readable summary of results."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    overall = metrics['overall']
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {overall['accuracy']:.1%}")
    print(f"  Coverage: {overall['coverage']:.1%} ({overall['n_samples']:,} / {overall['n_total']:,} episodes)")
    
    print(f"\nPer-Class Performance:")
    for outcome in ['BREAK', 'REJECT', 'CHOP']:
        stats = metrics['per_class'][outcome]
        print(f"  {outcome:7s}: P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1']:.3f}, N={stats['support']:,}")
    
    print(f"\nCalibration (Expected Calibration Error):")
    for outcome in ['BREAK', 'REJECT', 'CHOP']:
        ece = metrics['calibration'][outcome]['ece']
        print(f"  {outcome:7s}: ECE={ece:.3f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    labels = ['BREAK', 'REJECT', 'CHOP']
    print(f"         Predicted →")
    print(f"Actual ↓  {'  '.join(f'{l:>7s}' for l in labels)}")
    for i, label in enumerate(labels):
        print(f"{label:7s}   {'  '.join(f'{cm[i,j]:7,d}' for j in range(3))}")
    
    if metrics['stratified_metrics']:
        print(f"\nStratified Performance (samples ≥ 10):")
        
        # Level kinds
        level_metrics = {k: v for k, v in metrics['stratified_metrics'].items() if k.startswith('level_')}
        if level_metrics:
            print("  By Level Kind:")
            for key, stats in sorted(level_metrics.items()):
                level_kind = key.replace('level_', '')
                print(f"    {level_kind:10s}: {stats['accuracy']:.1%} (N={stats['n_samples']:,})")
        
        # Directions
        dir_metrics = {k: v for k, v in metrics['stratified_metrics'].items() if k.startswith('direction_')}
        if dir_metrics:
            print("  By Direction:")
            for key, stats in sorted(dir_metrics.items()):
                direction = key.replace('direction_', '')
                print(f"    {direction:5s}: {stats['accuracy']:.1%} (N={stats['n_samples']:,})")
        
        # Time buckets
        time_metrics = {k: v for k, v in metrics['stratified_metrics'].items() if k.startswith('time_')}
        if time_metrics:
            print("  By Time Bucket:")
            for key, stats in sorted(time_metrics.items()):
                bucket = key.replace('time_', '')
                print(f"    {bucket:10s}: {stats['accuracy']:.1%} (N={stats['n_samples']:,})")
        
        # Similarity quartiles
        sim_metrics = {k: v for k, v in metrics['stratified_metrics'].items() if k.startswith('similarity_')}
        if sim_metrics:
            print("  By Similarity Quality:")
            for key, stats in sorted(sim_metrics.items()):
                quartile = key.replace('similarity_', '')
                print(f"    {quartile:10s}: {stats['accuracy']:.1%} (N={stats['n_samples']:,}, avg_sim={stats.get('avg_similarity', 0):.3f})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study to evaluate retrieval system predictive power'
    )
    
    # Test date specification (one of these required)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        '--test-dates',
        type=str,
        help='Comma-separated list of test dates (YYYY-MM-DD)'
    )
    date_group.add_argument(
        '--test-start',
        type=str,
        help='Start date for test range (inclusive, YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-end',
        type=str,
        help='End date for test range (inclusive, YYYY-MM-DD). Required if --test-start is used.'
    )
    
    # Data paths
    parser.add_argument(
        '--canonical-version',
        type=str,
        default='3.1.0',
        help='Canonical version for episode data'
    )
    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=None,
        help='Episode corpus directory (default: data/gold/episodes/es_level_episodes/version={version})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ml/ablation_results',
        help='Output directory for results'
    )
    
    # Retrieval parameters
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=K_NEIGHBORS,
        help=f'Number of neighbors to retrieve (default: {K_NEIGHBORS})'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_start and not args.test_end:
        parser.error("--test-end is required when --test-start is used")
    
    # Determine test dates
    if args.test_dates:
        test_dates = parse_date_list(args.test_dates)
    else:
        test_dates = generate_date_range(args.test_start, args.test_end)
    
    logger.info(f"Ablation Study Configuration:")
    logger.info(f"  Test dates: {test_dates}")
    logger.info(f"  Canonical version: {args.canonical_version}")
    logger.info(f"  K neighbors: {args.k_neighbors}")
    
    # Setup paths
    backend_dir = Path(__file__).parent.parent
    
    if args.episodes_dir:
        episodes_dir = Path(args.episodes_dir)
    else:
        episodes_dir = backend_dir / f'data/gold/episodes/es_level_episodes/version={args.canonical_version}'
    
    output_dir = backend_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary index directory for training indices
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_dir = output_dir / f'training_indices_{timestamp}'
    
    logger.info(f"  Episodes directory: {episodes_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Training indices: {index_dir}")
    
    # Setup tracking
    run_name = f"ablation_{timestamp}"
    params = {
        'test_dates': test_dates,
        'canonical_version': args.canonical_version,
        'k_neighbors': args.k_neighbors,
        'test_mode': 'ablation_study'
    }
    tags = {
        'experiment_type': 'ablation',
        'canonical_version': args.canonical_version
    }
    wandb_tags = ['ablation', f'v{args.canonical_version}']
    
    try:
        with tracking_run(
            run_name=run_name,
            experiment='retrieval_ablation',
            params=params,
            tags=tags,
            wandb_tags=wandb_tags,
            project='spymaster',
            repo_root=backend_dir.parent
        ) as tracking:
            # Step 1: Load test episodes
            test_vectors, test_metadata = load_test_episodes(episodes_dir, test_dates)
            
            # Step 2: Build training indices (excluding test dates)
            build_stats = build_training_indices(episodes_dir, index_dir, test_dates)
            
            logger.info(f"Built {build_stats['n_partitions_built']} training indices")
            
            # Step 3: Load indices
            index_manager = IndexManager(index_dir)
            
            # Step 4: Run retrieval for test episodes
            predictions_df = run_retrieval_for_test_episodes(
                test_vectors,
                test_metadata,
                index_manager,
                k_neighbors=args.k_neighbors
            )
            
            # Step 5: Compute metrics
            metrics = compute_metrics(predictions_df)
            
            # Step 6: Print summary
            print_summary(metrics, predictions_df)
            
            # Step 7: Save results
            # Save predictions
            predictions_file = output_dir / f'predictions_{timestamp}.parquet'
            predictions_df.to_parquet(predictions_file, index=False)
            logger.info(f"\nPredictions saved: {predictions_file}")
            
            # Save metrics
            metrics_file = output_dir / f'metrics_{timestamp}.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved: {metrics_file}")
            
            # Save config
            config = {
                'timestamp': timestamp,
                'test_dates': test_dates,
                'canonical_version': args.canonical_version,
                'k_neighbors': args.k_neighbors,
                'episodes_dir': str(episodes_dir),
                'n_test_episodes': len(test_vectors),
                'n_partitions_built': build_stats['n_partitions_built']
            }
            config_file = output_dir / f'config_{timestamp}.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Config saved: {config_file}")
            
            # Step 8: Log to MLFlow and WandB
            logger.info("Logging to MLFlow and WandB...")
            
            # Flatten metrics for logging
            flat_metrics = {
                'accuracy': metrics['overall']['accuracy'],
                'coverage': metrics['overall']['coverage'],
                'n_samples': metrics['overall']['n_samples'],
                'break_precision': metrics['per_class']['BREAK']['precision'],
                'break_recall': metrics['per_class']['BREAK']['recall'],
                'break_f1': metrics['per_class']['BREAK']['f1'],
                'reject_precision': metrics['per_class']['REJECT']['precision'],
                'reject_recall': metrics['per_class']['REJECT']['recall'],
                'reject_f1': metrics['per_class']['REJECT']['f1'],
                'chop_precision': metrics['per_class']['CHOP']['precision'],
                'chop_recall': metrics['per_class']['CHOP']['recall'],
                'chop_f1': metrics['per_class']['CHOP']['f1'],
                'ece_break': metrics['calibration']['BREAK']['ece'],
                'ece_reject': metrics['calibration']['REJECT']['ece'],
                'ece_chop': metrics['calibration']['CHOP']['ece'],
            }
            
            log_metrics(flat_metrics, tracking.wandb_run)
            
            # Log artifacts
            log_artifacts(
                [predictions_file, metrics_file, config_file],
                name='ablation_results',
                artifact_type='evaluation',
                wandb_run=tracking.wandb_run
            )
            
            logger.info(f"\n✅ Results logged to MLFlow and WandB")
            logger.info(f"   MLFlow: Run 'mlflow ui' and check experiment 'retrieval_ablation'")
            logger.info(f"   WandB: Check project 'spymaster' for run '{run_name}'")
            
            logger.info(f"\nAblation study complete!")
            logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.1%}")
            
            return 0
    
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # Cleanup temporary indices (optional - comment out to keep)
        # import shutil
        # if index_dir.exists():
        #     shutil.rmtree(index_dir)
        #     logger.info(f"Cleaned up temporary indices: {index_dir}")
        pass


if __name__ == "__main__":
    import sys
    sys.exit(main())

