"""
Calibration Baseline Runner.
Computes Expected Calibration Error (ECE) for the current Production Vector (144D).
Uses Leave-One-Out (LOO) retrieval on the full corpus.
"""
import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.getcwd())

from src.ml.index_builder import load_episode_corpus
from src.ml.retrieval_engine import IndexManager, SimilarityQueryEngine, EpisodeQuery
from src.ml.calibration_engine import CalibrationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Paths
    backend_root = Path(os.getcwd())
    data_root = backend_root / "data"
    episodes_dir = data_root / "gold/episodes/es_level_episodes/version=3.1.0"
    indices_dir = data_root / "gold/indices/es_level_indices"
    output_dir = data_root / "ml/calibration_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Calibration Run...")
    
    # 1. Load Validation Data (Full Corpus)
    logger.info(f"Loading episodes from {episodes_dir}...")
    try:
        vectors, metadata = load_episode_corpus(episodes_dir)
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        sys.exit(1)
        
    logger.info(f"Loaded {len(vectors)} episodes.")
    
    # 2. Initialize Retrieval Engine
    if not indices_dir.exists():
        logger.error(f"Indices not found at {indices_dir}. Run build_production_indices.py first.")
        sys.exit(1)
        
    index_manager = IndexManager(indices_dir)
    query_engine = SimilarityQueryEngine(index_manager)
    
    # 3. Run Retrieval (LOO)
    predictions = []
    
    logger.info("Running retrieval (Leave-One-Out)...")
    for i in tqdm(range(len(vectors))):
        vector = vectors[i]
        meta_row = metadata.iloc[i]
        
        # Build Query
        # Note: We must ensure we provide the same metadata fields expected by the engine
        # to filter correctly (though simple retrieval relies mostly on vector+partition keys)
        
        query = EpisodeQuery(
            level_kind=meta_row['level_kind'],
            level_price=meta_row['level_price'],
            direction=meta_row['direction'],
            time_bucket=meta_row['time_bucket'],
            vector=vector,
            emission_weight=1.0, 
            timestamp=meta_row['timestamp'], # Critical for self-exclusion
            metadata=meta_row.to_dict()
        )
        
        # Execute Query
        try:
            result = query_engine.query(query)
        except Exception as e:
            # logger.warning(f"Query failed for {i}: {e}")
            continue
            
        # Extract Probability of BREAK
        # 'probabilities' dict usually has 'BREAK', 'BOUNCE', 'CHOP'
        probs = result.outcome_probabilities.get('probabilities', {})
        p_break = probs.get('BREAK', 0.0)
        
        # Ground Truth
        # outcome_4min is the label. 
        # Note: In ablation, we used outcome_4min.
        true_outcome = meta_row.get('outcome_4min', 'CHOP')
        is_break = 1 if true_outcome == 'BREAK' else 0
        
        predictions.append({
            'uuid': meta_row.get('event_id'),
            'timestamp': meta_row['timestamp'],
            'p_break': p_break,
            'is_break': is_break,
            'date': meta_row['date'],
            'level_kind': meta_row['level_kind']
        })
        
    results_df = pd.DataFrame(predictions)
    logger.info(f"Computed predictions for {len(results_df)} episodes.")
    
    # 4. Compute Calibration
    if results_df.empty:
        logger.error("No results generated.")
        sys.exit(1)
        
    cal_engine = CalibrationEngine(n_bins=10)
    cal_result = cal_engine.compute_metrics(
        y_true=results_df['is_break'].values,
        y_prob=results_df['p_break'].values
    )
    
    # 5. Report & Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(results_df),
        "ece": cal_result.ece,
        "mce": cal_result.mce,
        "brier": cal_result.brier_score,
        "bins": {
            "pred": cal_result.prob_pred.tolist(),
            "true": cal_result.prob_true.tolist(),
            "counts": cal_result.bin_counts.tolist()
        }
    }
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"calibration_baseline_{run_id}.json"
    plot_path = output_dir / f"calibration_baseline_{run_id}.png"
    
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Processing plot...")
    cal_engine.plot_reliability_curve(
        cal_result, 
        title=f"Baseline Calibration (144D Vector)\nECE={cal_result.ece:.4f}",
        save_path=plot_path
    )
    
    print("\nCalibration Results (Full Corpus LOO):")
    print(f"  Count: {len(results_df)}")
    print(f"  ECE:   {cal_result.ece:.4f}")
    print(f"  MCE:   {cal_result.mce:.4f}")
    print(f"  Brier: {cal_result.brier_score:.4f}")
    print(f"\nSaved report to: {json_path}")
    print(f"Saved plot to:   {plot_path}")
    
    # 6. Stratified Analysis (Per Level Kind)
    print("\nStratified Calibration (ECE by Level Kind):")
    stratified_report = {}
    
    for level_kind, group in results_df.groupby('level_kind'):
        if len(group) < 20:
            print(f"  {level_kind:<10}: (N={len(group)}) - Too few samples")
            continue
            
        cal_res_strat = cal_engine.compute_metrics(
            y_true=group['is_break'].values,
            y_prob=group['p_break'].values
        )
        print(f"  {level_kind:<10}: ECE={cal_res_strat.ece:.4f} (N={len(group)})")
        stratified_report[level_kind] = {
            "ece": cal_res_strat.ece,
            "n": len(group)
        }
        
    # Update report with stratified data
    report['stratified'] = stratified_report
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
