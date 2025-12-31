"""
Grid Search for Vector Optimization.
Systematically tests different Vector Compression strategies to minimize Calibration Error (ECE).
"""
import sys
import os
import logging
import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add backend to path
sys.path.append(os.getcwd())

from src.ml.index_builder import load_episode_corpus
from src.ml.vector_compressor import VectorCompressor
from src.ml.calibration_engine import CalibrationEngine
from src.ml.retrieval_engine import SimilarityQueryEngine, IndexManager, EpisodeQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Index Manager for In-Memory Indices
class InMemoryIndexManager:
    """Simple wrapper to satisfy SimilarityQueryEngine interface using a raw FAISS index."""
    def __init__(self, index, metadata_df):
        self.index = index
        self.metadata_df = metadata_df
        self.partition_map = {} # naive implementation, just one global index for this test
        
    def query(self, query_vector, k=50, filters=None):
        # We ignore filters for this simple grid search validation
        # In prod, we partition. Here we query the global index.
        # This is a simplification but valid for "Vector Quality" assessment 
        # as long as we compare apples to apples.
        # Actually, if we don't partition, we might struggle with regime separation.
        # But Filtered Search in FAISS Flat is slow. 
        # Let's trust that the Vector *should* separate partitions naturally if it's good.
        
        D, I = self.index.search(query_vector.reshape(1, -1), k)
        return D[0], I[0]
        
    def get_metadata(self, indices):
        return self.metadata_df.iloc[indices].to_dict('records')

def evaluate_config(config_name: str, compressor: VectorCompressor, vectors_raw: np.ndarray, metadata_df: pd.DataFrame) -> dict:
    """Run full calibration test for a single configuration."""
    logger.info(f"--- Evaluating: {config_name} ---")
    
    # 1. Fit & Transform
    start_ts = datetime.now()
    if not compressor.is_fitted:
        compressor.fit(vectors_raw)
        
    vectors_transformed = compressor.transform(vectors_raw)
    dim = vectors_transformed.shape[1]
    logger.info(f"  Vector Dim: {dim}")
    
    # 2. Build Index (Flat L2)
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_transformed)
    
    # 3. Setup Engine
    # Note: We use a simplified in-memory manager
    idx_manager = InMemoryIndexManager(index, metadata_df)
    
    # 3. Run Retrieval loop
    predictions = []
    
    # We query for every episode
    # Note: SimilarityQueryEngine logic for weighting/aggregating is used manually here
    # because we need to bypass the strict Partition check of the real IndexManager
    
    query_k = 50
    
    for i in tqdm(range(len(vectors_transformed)), desc=f"Testing {config_name}", leave=False):
        query_vec = vectors_transformed[i]
        true_ts = metadata_df.iloc[i]['timestamp']
        true_id = metadata_df.iloc[i]['event_id']
        
        # Search
        D, I = index.search(query_vec.reshape(1, -1), query_k + 1) # +1 for self
        
        # Process Neighbors
        neighbors = []
        distances = []
        
        found_self = False
        
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(metadata_df): continue
            
            # Retrieve neighbor meta
            n_row = metadata_df.iloc[idx]
            n_ts = n_row['timestamp']
            
            # Deduplicate (LOO)
            # If timestamp is close to query, it's the same event (or immediate temporal leak)
            # 5 minute exclusion window
            time_diff = abs((n_ts - true_ts).total_seconds())
            if time_diff < 300: 
                continue
                
            neighbors.append(n_row)
            distances.append(D[0][rank])
            
            if len(neighbors) >= 50:
                break
                
        if not neighbors:
            continue
            
        # Outcome Aggregation (Simplified)
        # We just want P(Break)
        # Weighting: 1 / (1 + dist)
        weights = 1.0 / (1.0 + np.array(distances[:len(neighbors)]))
        weight_sum = np.sum(weights)
        if weight_sum == 0: continue
        norm_weights = weights / weight_sum
        
        # Calculate P(Break)
        p_break = 0.0
        for n_idx, n in enumerate(neighbors):
            outcome = n.get('outcome_4min', 'CHOP')
            is_break = 1.0 if outcome == 'BREAK' else 0.0
            p_break += is_break * norm_weights[n_idx]
            
        # Truth
        y_true = 1 if metadata_df.iloc[i]['outcome_4min'] == 'BREAK' else 0
        
        predictions.append({'p_break': p_break, 'y_true': y_true})
        
    # 4. Compute Metrics
    res_df = pd.DataFrame(predictions)
    if res_df.empty:
        return {'ece': 1.0, 'mce': 1.0}
        
    cal_engine = CalibrationEngine(n_bins=10)
    metrics = cal_engine.compute_metrics(res_df['y_true'].values, res_df['p_break'].values)
    
    duration = (datetime.now() - start_ts).total_seconds()
    logger.info(f"  Result: ECE={metrics.ece:.4f}, MCE={metrics.mce:.4f} (Time: {duration:.1f}s)")
    
    return {
        'name': config_name,
        'ece': metrics.ece,
        'mce': metrics.mce,
        'prob_pred': metrics.prob_pred.tolist(),
        'prob_true': metrics.prob_true.tolist()
    }

def main():
    # Setup Paths
    backend_root = Path(os.getcwd())
    episodes_dir = backend_root / "data/gold/episodes/es_level_episodes/version=3.1.0"
    output_dir = backend_root / "data/ml/grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    logger.info("Loading Data...")
    vectors, metadata = load_episode_corpus(episodes_dir)
    logger.info(f"Loaded {len(vectors)} episodes.")
    
    # Define Grid
    configs = [
        # 1. Baseline
        ('baseline_identity', VectorCompressor(strategy='identity')),
        
        # 2. Geometry Only (The Hypothesis)
        ('geometry_only', VectorCompressor(strategy='geometry_only')),
        
        # 3. PCA Physics (Standard)
        ('pca_physics_k3', VectorCompressor(strategy='pca_physics', n_components=3)),
        ('pca_physics_k5', VectorCompressor(strategy='pca_physics', n_components=5)),
        ('pca_physics_k10', VectorCompressor(strategy='pca_physics', n_components=10)),
        
        # 4. Weighted (Physics Damped)
        # Note: 'weighted' logic in Compressor needs implementation or is placeholder
        # For now, we rely on PCA to do the "damping" by removing noise variance
    ]
    
    results = []
    
    # Run Loop
    print("\nStarting Grid Search...")
    print(f"{'Config':<20} | {'ECE':<8} | {'MCE':<8}")
    print("-" * 40)
    
    files_saved = []
    
    best_ece = 1.0
    best_config = None
    
    for name, compressor in configs:
        res = evaluate_config(name, compressor, vectors, metadata)
        results.append(res)
        
        print(f"{name:<20} | {res['ece']:.4f}   | {res['mce']:.4f}")
        
        if res['ece'] < best_ece:
            best_ece = res['ece']
            best_config = name
            
    # Save Full Report
    report_path = output_dir / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump({'best_config': best_config, 'results': results}, f, indent=2)
        
    print(f"\nGrid Search Complete.")
    print(f"Winner: {best_config} (ECE={best_ece:.4f})")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
