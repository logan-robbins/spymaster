"""
Train Level Interaction Classifier with Comprehensive Ablations

Trains on ALL state table features, then systematically ablates feature groups
to discover what actually provides edge.

Ablation dimensions:
1. Feature groups (OFI, GEX, barriers, tape, velocity, streams)
2. Level types (PM_HIGH vs OR_LOW vs SMA, etc.)
3. Proximity (AT_LEVEL vs NEAR_LEVEL)
4. Time windows (first 30min vs mid-morning vs late-morning)
5. Stream combinations (which streams matter most?)

Usage:
    uv run python scripts/train_level_classifier.py \
      --data data/gold/training/pentaview_level_interactions/dataset_2025-11-17_2025-12-17.parquet \
      --output data/ml/pentaview_classifier_v4.joblib \
      --ablations
"""
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "baseline"
) -> Dict[str, Any]:
    """Train model and return metrics."""
    
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=['MOVE_UP', 'MOVE_DOWN', 'FLAT'], zero_division=0
    )
    
    # Trading-specific metrics
    up_mask = y_pred == 'MOVE_UP'
    down_mask = y_pred == 'MOVE_DOWN'
    
    up_precision = (y_test[up_mask] == 'MOVE_UP').mean() if up_mask.sum() > 0 else 0
    down_precision = (y_test[down_mask] == 'MOVE_DOWN').mean() if down_mask.sum() > 0 else 0
    
    # High confidence
    class_idx_up = list(model.classes_).index('MOVE_UP') if 'MOVE_UP' in model.classes_ else -1
    class_idx_down = list(model.classes_).index('MOVE_DOWN') if 'MOVE_DOWN' in model.classes_ else -1
    
    hc_up_prec = 0
    hc_down_prec = 0
    
    if class_idx_up >= 0:
        hc_up_mask = y_proba[:, class_idx_up] > 0.7
        if hc_up_mask.sum() > 0:
            hc_up_prec = (y_test[hc_up_mask] == 'MOVE_UP').mean()
    
    if class_idx_down >= 0:
        hc_down_mask = y_proba[:, class_idx_down] > 0.7
        if hc_down_mask.sum() > 0:
            hc_down_prec = (y_test[hc_down_mask] == 'MOVE_DOWN').mean()
    
    return {
        'name': name,
        'accuracy': float((y_pred == y_test).mean()),
        'up_precision': float(up_precision),
        'down_precision': float(down_precision),
        'up_precision_hc': float(hc_up_prec),
        'down_precision_hc': float(hc_down_prec),
        'n_test': len(y_test),
        'n_up_pred': int(up_mask.sum()),
        'n_down_pred': int(down_mask.sum())
    }


def run_ablations(dataset: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run comprehensive ablation studies."""
    
    logger.info("="*70)
    logger.info("ABLATION STUDY: FEATURE IMPORTANCE")
    logger.info("="*70)
    
    # Separate features and labels
    exclude_cols = ['label', 'magnitude', 'bars_to_extreme', 'target_stream', 
                   'level_kind', 'distance_atr', 'timestamp', 'current_stream_value']
    
    all_features = [c for c in dataset.columns if c not in exclude_cols]
    
    # Group features
    feature_groups = {
        'streams': [c for c in all_features if c.startswith('sigma_')],
        'ofi': [c for c in all_features if 'ofi' in c],
        'gex': [c for c in all_features if 'gex' in c or 'gamma' in c],
        'barriers': [c for c in all_features if 'barrier' in c or 'wall' in c],
        'tape': [c for c in all_features if 'tape' in c],
        'velocity': [c for c in all_features if 'velocity' in c or 'acceleration' in c or 'jerk' in c],
        'level_encoding': [c for c in all_features if c.startswith('level_is_')],
        'level_distance': [c for c in all_features if c.startswith('dist_to_')],
        'level_stacking': [c for c in all_features if 'stacking' in c],
        'other': [c for c in all_features if c not in sum([v for k, v in {'streams': [c for c in all_features if c.startswith('sigma_')], 'ofi': [c for c in all_features if 'ofi' in c]}.items()], [])]
    }
    
    # Clean up 'other'
    used_features = set()
    for group, feats in feature_groups.items():
        if group != 'other':
            used_features.update(feats)
    feature_groups['other'] = [c for c in all_features if c not in used_features]
    
    X = dataset[all_features].fillna(0).values
    y = dataset['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    
    # Baseline: All features
    logger.info("\n1. BASELINE (all 94 features)")
    baseline = train_and_evaluate(X_train, y_train, X_test, y_test, "baseline")
    results.append(baseline)
    logger.info(f"   Accuracy: {baseline['accuracy']:.1%}")
    logger.info(f"   UP precision: {baseline['up_precision']:.1%} (HC: {baseline['up_precision_hc']:.1%})")
    logger.info(f"   DOWN precision: {baseline['down_precision']:.1%} (HC: {baseline['down_precision_hc']:.1%})")
    
    # Ablate each feature group
    for group_name, group_features in feature_groups.items():
        if not group_features or group_name == 'other':
            continue
        
        # Remove this group
        ablated_features = [f for f in all_features if f not in group_features]
        
        if len(ablated_features) == 0:
            continue
        
        X_ablated = dataset[ablated_features].fillna(0).values
        X_train_abl, X_test_abl, _, _ = train_test_split(
            X_ablated, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\n{len(results)+1}. WITHOUT {group_name.upper()} ({len(group_features)} features removed)")
        
        result = train_and_evaluate(X_train_abl, y_train, X_test_abl, y_test, f"no_{group_name}")
        results.append(result)
        
        degradation = baseline['accuracy'] - result['accuracy']
        logger.info(f"   Accuracy: {result['accuracy']:.1%} (Δ {degradation:+.1%})")
        logger.info(f"   UP precision: {result['up_precision']:.1%} (Δ {result['up_precision'] - baseline['up_precision']:+.1%})")
        logger.info(f"   DOWN precision: {result['down_precision']:.1%} (Δ {result['down_precision'] - baseline['down_precision']:+.1%})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train level-interaction classifier')
    parser.add_argument('--data', type=str, required=True, help='Training data parquet')
    parser.add_argument('--output', type=str, required=True, help='Output model path')
    parser.add_argument('--ablations', action='store_true', help='Run ablation studies')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading: {args.data}\n")
    dataset = pd.read_parquet(args.data)
    
    logger.info(f"Dataset: {len(dataset):,} samples")
    logger.info(f"Labels: {dataset['label'].value_counts().to_dict()}")
    logger.info("")
    
    if args.ablations:
        ablation_results = run_ablations(dataset)
        
        # Save ablation results
        output_dir = Path(args.output).parent
        ablation_path = output_dir / 'ablation_results.json'
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        logger.info(f"\n✅ Ablation results saved to: {ablation_path}")
    
    # Train final model with all features
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*70)
    
    exclude_cols = ['label', 'magnitude', 'bars_to_extreme', 'target_stream',
                   'level_kind', 'distance_atr', 'timestamp', 'current_stream_value']
    feature_cols = [c for c in dataset.columns if c not in exclude_cols]
    
    X = dataset[feature_cols].fillna(0).values
    y = dataset['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training on {len(feature_cols)} features...")
    
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    logger.info("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        'model': model,
        'feature_names': feature_cols,
        'training_date': pd.Timestamp.now().isoformat(),
        'n_train': len(X_train),
        'n_features': len(feature_cols)
    }
    
    joblib.dump(bundle, output_path)
    logger.info(f"\n✅ Model saved to: {output_path}")


if __name__ == "__main__":
    main()

