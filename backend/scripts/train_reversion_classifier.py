"""
Train Mean-Reversion Classifier

Predicts:
- P(REVERT_DOWN | stream_state) - Fade overbought
- P(REVERT_UP | stream_state) - Buy oversold  
- P(NO_MOVE | stream_state) - Sit out

Usage:
    uv run python scripts/train_reversion_classifier.py \
      --data data/gold/training/reversion_samples/reversion_dataset_2025-11-17_2025-12-17.parquet \
      --output data/ml/reversion_models
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def train_reversion_classifier(
    dataset: pd.DataFrame,
    output_path: Path,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train and evaluate reversion classifier."""
    
    logger.info("="*70)
    logger.info("MEAN-REVERSION CLASSIFIER TRAINING")
    logger.info("="*70)
    logger.info(f"Total samples: {len(dataset):,}")
    
    # Separate features and labels
    label_col = 'label'
    exclude_cols = ['label', 'magnitude', 'target_stream', 'timestamp', 'current_value']
    feature_cols = [c for c in dataset.columns if c not in exclude_cols]
    
    X = dataset[feature_cols].fillna(0).values
    y = dataset[label_col].values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"  Sample features: {feature_cols[:5]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"\nTrain: {len(X_train):,}")
    logger.info(f"Test:  {len(X_test):,}")
    
    # Train classifier
    logger.info(f"\nTraining HistGradientBoostingClassifier...")
    
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info(f"\n{'='*70}")
    logger.info("TEST SET EVALUATION")
    logger.info(f"{'='*70}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Classification report
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['NO_MOVE', 'REVERT_UP', 'REVERT_DOWN'])
    logger.info("\nConfusion Matrix:")
    logger.info("                Pred:NO_MOVE  Pred:UP  Pred:DOWN")
    logger.info(f"Actual:NO_MOVE      {cm[0,0]:6d}     {cm[0,1]:5d}     {cm[0,2]:5d}")
    logger.info(f"Actual:REVERT_UP    {cm[1,0]:6d}     {cm[1,1]:5d}     {cm[1,2]:5d}")
    logger.info(f"Actual:REVERT_DOWN  {cm[2,0]:6d}     {cm[2,1]:5d}     {cm[2,2]:5d}")
    
    # Trading-specific metrics
    logger.info(f"\n{'='*70}")
    logger.info("TRADING EDGE ANALYSIS")
    logger.info(f"{'='*70}")
    
    # When model predicts REVERT_UP, what % actually went up?
    up_mask = y_pred == 'REVERT_UP'
    if up_mask.sum() > 0:
        precision_up = (y_test[up_mask] == 'REVERT_UP').mean()
        logger.info(f"When predict REVERT_UP:")
        logger.info(f"  Precision: {precision_up:.1%} (n={up_mask.sum()})")
        logger.info(f"  Edge: {(precision_up - 0.33)*100:+.1f}% vs random guess")
    
    # When model predicts REVERT_DOWN
    down_mask = y_pred == 'REVERT_DOWN'
    if down_mask.sum() > 0:
        precision_down = (y_test[down_mask] == 'REVERT_DOWN').mean()
        logger.info(f"\nWhen predict REVERT_DOWN:")
        logger.info(f"  Precision: {precision_down:.1%} (n={down_mask.sum()})")
        logger.info(f"  Edge: {(precision_down - 0.33)*100:+.1f}% vs random guess")
    
    # High-confidence predictions
    class_names = model.classes_
    for i, class_name in enumerate(class_names):
        if class_name == 'NO_MOVE':
            continue
        
        # Get probability for this class
        proba = y_proba[:, i]
        
        # High confidence = prob > 0.7
        high_conf_mask = proba > 0.7
        if high_conf_mask.sum() > 0:
            precision_high_conf = (y_test[high_conf_mask] == class_name).mean()
            logger.info(f"\nHigh Confidence {class_name} (prob > 70%):")
            logger.info(f"  Precision: {precision_high_conf:.1%} (n={high_conf_mask.sum()})")
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        'model': model,
        'feature_names': feature_cols,
        'class_names': model.classes_.tolist(),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    joblib.dump(bundle, output_path)
    logger.info(f"\n✅ Model saved to: {output_path}")
    
    return {
        'feature_names': feature_cols,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_accuracy': float((y_pred == y_test).mean())
    }


def main():
    parser = argparse.ArgumentParser(description='Train mean-reversion classifier')
    parser.add_argument('--data', type=str, required=True, help='Training data parquet')
    parser.add_argument('--output', type=str, required=True, help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading dataset: {args.data}")
    dataset = pd.read_parquet(args.data)
    
    # Train
    metrics = train_reversion_classifier(
        dataset=dataset,
        output_path=Path(args.output),
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()

