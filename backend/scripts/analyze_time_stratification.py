"""
Time-of-Day Stratification Analysis

Analyzes whether BREAK/REJECT dynamics differ by time bucket and if
separate models or dynamic thresholds are needed per time period.

Key Questions:
1. Does base BREAK rate vary significantly by time bucket?
2. Do feature importances differ by time?
3. Should we use time-specific models or just time-aware thresholds?

Usage:
    cd backend
    uv run python -m scripts.analyze_time_stratification \
        --version v4.0.0 \
        --start 2025-11-03 \
        --end 2025-12-19 \
        --horizon 4min \
        --output-json data/ml/time_stratification_analysis.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.constants import TIME_BUCKETS, HORIZONS


def load_episodes(version: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load episode metadata."""
    episodes_dir = Path(f'data/gold/episodes/es_level_episodes/version={version}/metadata')
    
    all_meta = []
    for date_dir in sorted(episodes_dir.glob('date=*')):
        date_str = date_dir.name.split('=')[1]
        if start_date <= date_str <= end_date:
            meta_path = date_dir / 'metadata.parquet'
            if meta_path.exists():
                all_meta.append(pd.read_parquet(meta_path))
    
    return pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()


def load_stream_bars(version: str) -> pd.DataFrame:
    """Load pentaview stream bars."""
    stream_dir = Path(f'data/gold/streams/pentaview/version={version}')
    
    all_streams = []
    for date_dir in sorted(stream_dir.glob('date=*')):
        stream_path = date_dir / 'stream_bars.parquet'
        if stream_path.exists():
            all_streams.append(pd.read_parquet(stream_path))
    
    return pd.concat(all_streams, ignore_index=True) if all_streams else pd.DataFrame()


def compute_time_bucket_stats(
    df: pd.DataFrame,
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics per time bucket.
    
    Returns:
        Dict[time_bucket] -> {
            'n': int,
            'break_rate': float,
            'reject_rate': float,
            'chop_rate': float
        }
    """
    stats_dict = {}
    
    for bucket in TIME_BUCKETS.keys():
        bucket_df = df[df['time_bucket'] == bucket]
        n = len(bucket_df)
        
        if n == 0:
            continue
        
        outcomes = bucket_df[outcome_col].value_counts()
        
        stats_dict[bucket] = {
            'n': n,
            'break_rate': float(outcomes.get('BREAK', 0) / n),
            'reject_rate': float(outcomes.get('REJECT', 0) / n),
            'chop_rate': float(outcomes.get('CHOP', 0) / n),
            'break_count': int(outcomes.get('BREAK', 0)),
            'reject_count': int(outcomes.get('REJECT', 0))
        }
    
    return stats_dict


def train_time_specific_models(
    df: pd.DataFrame,
    features: List[str],
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict]:
    """
    Train separate model for each time bucket.
    
    Returns:
        Dict[time_bucket] -> {
            'model': fitted model,
            'cv_auc': float,
            'feature_importances': Dict[feature_name, float],
            'n_train': int
        }
    """
    results = {}
    
    for bucket in TIME_BUCKETS.keys():
        bucket_df = df[df['time_bucket'] == bucket].copy()
        
        # Filter to BREAK/REJECT only
        mask = bucket_df[outcome_col].isin(['BREAK', 'REJECT'])
        bucket_df = bucket_df[mask]
        
        if len(bucket_df) < 50:
            print(f"  Skipping {bucket}: insufficient samples ({len(bucket_df)})")
            continue
        
        # Prepare data
        X = bucket_df[features].fillna(0).values
        y = (bucket_df[outcome_col] == 'BREAK').astype(int).values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        
        # CV score
        cv = StratifiedKFold(n_splits=min(5, len(bucket_df) // 20), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # Fit final
        model.fit(X_scaled, y)
        
        # Feature importances
        importances = dict(zip(features, model.feature_importances_))
        
        results[bucket] = {
            'model': model,
            'scaler': scaler,
            'cv_auc': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'feature_importances': {k: float(v) for k, v in importances.items()},
            'n_train': len(bucket_df),
            'break_rate': float(y.mean())
        }
        
        print(f"  {bucket}: n={len(bucket_df)}, AUC={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
    
    return results


def compare_global_vs_time_specific(
    df: pd.DataFrame,
    features: List[str],
    outcome_col: str = 'outcome_4min'
) -> Dict[str, float]:
    """
    Compare single global model vs time-specific models.
    
    Returns dict with AUC scores.
    """
    # Filter to BREAK/REJECT
    mask = df[outcome_col].isin(['BREAK', 'REJECT'])
    df_filtered = df[mask].copy()
    
    X = df_filtered[features].fillna(0).values
    y = (df_filtered[outcome_col] == 'BREAK').astype(int).values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Global model
    global_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_scores = cross_val_score(global_model, X_scaled, y, cv=cv, scoring='roc_auc')
    
    # Time-specific models (use Leave-One-Time-Bucket-Out)
    time_specific_scores = []
    
    for bucket in TIME_BUCKETS.keys():
        # Train on all except this bucket
        train_mask = df_filtered['time_bucket'] != bucket
        test_mask = df_filtered['time_bucket'] == bucket
        
        if test_mask.sum() < 10:
            continue
        
        X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        # Train bucket-specific model
        bucket_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        bucket_model.fit(X_train, y_train)
        
        # Score
        from sklearn.metrics import roc_auc_score
        y_pred = bucket_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        time_specific_scores.append(auc)
    
    return {
        'global_auc_mean': float(np.mean(global_scores)),
        'global_auc_std': float(np.std(global_scores)),
        'time_specific_auc_mean': float(np.mean(time_specific_scores)) if time_specific_scores else None,
        'time_specific_auc_std': float(np.std(time_specific_scores)) if time_specific_scores else None
    }


def analyze_feature_consistency(
    time_models: Dict[str, Dict],
    features: List[str]
) -> pd.DataFrame:
    """
    Analyze which features are consistently important across time buckets.
    
    Returns DataFrame with feature importance rankings per bucket.
    """
    # Build importance matrix
    importance_matrix = []
    buckets = []
    
    for bucket, model_dict in time_models.items():
        buckets.append(bucket)
        importances = [model_dict['feature_importances'].get(f, 0.0) for f in features]
        importance_matrix.append(importances)
    
    df = pd.DataFrame(importance_matrix, columns=features, index=buckets)
    
    # Add mean and std
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df.loc['cv'] = df.loc['std'] / (df.loc['mean'] + 1e-9)  # Coefficient of variation
    
    return df


def analyze_optimal_thresholds(
    df: pd.DataFrame,
    features: List[str],
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict[str, float]]:
    """
    Find optimal p_break threshold per time bucket to maximize precision.
    
    Returns Dict[time_bucket] -> {
        'optimal_threshold': float,
        'precision_at_threshold': float,
        'recall_at_threshold': float
    }
    """
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    
    # Filter
    mask = df[outcome_col].isin(['BREAK', 'REJECT'])
    df_filtered = df[mask].copy()
    
    results = {}
    
    for bucket in TIME_BUCKETS.keys():
        bucket_df = df_filtered[df_filtered['time_bucket'] == bucket]
        
        if len(bucket_df) < 30:
            continue
        
        X = bucket_df[features].fillna(0).values
        y = (bucket_df[outcome_col] == 'BREAK').astype(int).values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Quick model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        
        # Predict
        y_pred = model.predict_proba(X_scaled)[:, 1]
        
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_pred)
        
        # Find threshold with precision >= 0.65
        mask_good = precision >= 0.65
        if mask_good.sum() > 0:
            idx = np.where(mask_good)[0][0]
            optimal_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
            optimal_precision = float(precision[idx])
            optimal_recall = float(recall[idx])
        else:
            optimal_threshold = 0.6
            optimal_precision = float(precision[np.argmax(precision)])
            optimal_recall = float(recall[np.argmax(precision)])
        
        results[bucket] = {
            'optimal_threshold': optimal_threshold,
            'precision_at_threshold': optimal_precision,
            'recall_at_threshold': optimal_recall,
            'auc': float(roc_auc_score(y, y_pred)),
            'n': len(bucket_df)
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze time-of-day stratification')
    parser.add_argument('--version', default='v4.0.0', help='Data version')
    parser.add_argument('--start', default='2025-11-03', help='Start date')
    parser.add_argument('--end', default='2025-12-19', help='End date')
    parser.add_argument('--horizon', default='4min', help='Outcome horizon')
    parser.add_argument('--output-json', default='data/ml/time_stratification_analysis.json')
    
    args = parser.parse_args()
    
    outcome_col = f'outcome_{args.horizon}'
    
    print("=== Time-of-Day Stratification Analysis ===\n")
    
    # Load data
    print("Loading episodes...")
    episodes = load_episodes(args.version, args.start, args.end)
    print(f"Loaded {len(episodes)} episodes")
    
    print("\nLoading stream bars...")
    streams = load_stream_bars(args.version)
    print(f"Loaded {len(streams)} stream bars")
    
    # Merge
    print("\nMerging episodes with streams...")
    episodes['ts_str'] = episodes['timestamp'].astype(str)
    streams['ts_str'] = streams['timestamp'].astype(str)
    
    merged = episodes.merge(
        streams,
        on=['ts_str', 'level_kind'],
        how='left'
    )
    print(f"Merged: {len(merged)} rows")
    
    # Filter to predictable segment
    predictable_mask = (
        (merged['direction'] == 'UP') &
        (merged['level_kind'].isin(['PM_LOW', 'OR_LOW'])) &
        (~merged['time_bucket'].isin(['T0_15']))
    )
    
    predictable = merged[predictable_mask].copy()
    print(f"\nPredictable segment: {len(predictable)} episodes")
    
    # Stream features
    stream_features = ['sigma_s', 'sigma_r', 'sigma_b_slope', 'sigma_p_slope', 'sigma_d']
    
    # 1. Time bucket statistics
    print("\n=== 1. Time Bucket Statistics ===")
    time_stats = compute_time_bucket_stats(predictable, outcome_col)
    
    for bucket, stats in time_stats.items():
        print(f"\n{bucket}:")
        print(f"  N: {stats['n']}")
        print(f"  BREAK rate: {stats['break_rate']:.1%}")
        print(f"  REJECT rate: {stats['reject_rate']:.1%}")
        print(f"  CHOP rate: {stats['chop_rate']:.1%}")
    
    # Statistical test: Are BREAK rates different across buckets?
    print("\n=== 2. Statistical Tests ===")
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    
    contingency = []
    bucket_names = []
    for bucket, stats in time_stats.items():
        if stats['n'] >= 20:  # Minimum sample size
            contingency.append([stats['break_count'], stats['reject_count']])
            bucket_names.append(bucket)
    
    if len(contingency) >= 2:
        contingency = np.array(contingency).T
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        print(f"Chi-square test for time bucket independence:")
        print(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → SIGNIFICANT: BREAK rates differ by time bucket")
        else:
            print(f"  → Not significant: BREAK rates similar across time")
    
    # 3. Train time-specific models
    print("\n=== 3. Time-Specific Models ===")
    time_models = train_time_specific_models(predictable, stream_features, outcome_col)
    
    # 4. Compare global vs time-specific
    print("\n=== 4. Global vs Time-Specific Performance ===")
    comparison = compare_global_vs_time_specific(predictable, stream_features, outcome_col)
    
    print(f"Global model AUC: {comparison['global_auc_mean']:.3f}±{comparison['global_auc_std']:.3f}")
    if comparison['time_specific_auc_mean']:
        print(f"Time-specific models AUC: {comparison['time_specific_auc_mean']:.3f}±{comparison['time_specific_auc_std']:.3f}")
        
        improvement = comparison['time_specific_auc_mean'] - comparison['global_auc_mean']
        print(f"Improvement: {improvement:+.3f}")
        
        if improvement > 0.02:
            print("  → Moderate improvement with time-specific models")
        elif improvement > 0.05:
            print("  → Strong improvement with time-specific models")
        else:
            print("  → Minimal improvement, global model sufficient")
    
    # 5. Feature consistency
    print("\n=== 5. Feature Consistency Across Time ===")
    feature_consistency = analyze_feature_consistency(time_models, stream_features)
    
    print("\nTop features by mean importance:")
    mean_importances = feature_consistency.loc['mean'].sort_values(ascending=False)
    for feat, imp in mean_importances.items():
        cv = feature_consistency.loc['cv', feat]
        print(f"  {feat}: {imp:.3f} (CV={cv:.2f})")
    
    # 6. Optimal thresholds
    print("\n=== 6. Optimal Thresholds by Time ===")
    optimal_thresholds = analyze_optimal_thresholds(predictable, stream_features, outcome_col)
    
    for bucket, thresh_dict in optimal_thresholds.items():
        print(f"\n{bucket}:")
        print(f"  Optimal threshold: {thresh_dict['optimal_threshold']:.2f}")
        print(f"  Precision: {thresh_dict['precision_at_threshold']:.1%}")
        print(f"  Recall: {thresh_dict['recall_at_threshold']:.1%}")
        print(f"  AUC: {thresh_dict['auc']:.3f}")
    
    # Build output
    output = {
        'meta': {
            'version': args.version,
            'start_date': args.start,
            'end_date': args.end,
            'horizon': args.horizon,
            'n_episodes': len(predictable),
            'segment': 'UP + LOW_LEVELS + post_T0_15'
        },
        'time_bucket_stats': time_stats,
        'statistical_test': {
            'chi2': float(chi2),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        } if len(contingency) >= 2 else None,
        'model_comparison': comparison,
        'time_specific_models': {
            bucket: {
                'cv_auc': model_dict['cv_auc'],
                'cv_auc_std': model_dict['cv_auc_std'],
                'n_train': model_dict['n_train'],
                'break_rate': model_dict['break_rate'],
                'feature_importances': model_dict['feature_importances']
            }
            for bucket, model_dict in time_models.items()
        },
        'feature_consistency': feature_consistency.to_dict(),
        'optimal_thresholds': optimal_thresholds,
        'recommendation': None
    }
    
    # Generate recommendation
    if comparison.get('time_specific_auc_mean'):
        improvement = comparison['time_specific_auc_mean'] - comparison['global_auc_mean']
        if improvement > 0.05:
            recommendation = "Use time-specific models - significant performance gain"
        elif improvement > 0.02:
            recommendation = "Use time-adjusted thresholds - moderate gain without complexity"
        else:
            recommendation = "Use single global model - time-specific gains minimal"
    else:
        recommendation = "Insufficient data for time-specific models"
    
    output['recommendation'] = recommendation
    
    # Save
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Analysis saved to: {output_path}")
    print(f"Recommendation: {recommendation}")
    
    print(f"\n=== Key Insights ===")
    print(f"1. Time buckets have {'significantly' if output['statistical_test']['significant'] else 'similar'} different BREAK rates")
    print(f"2. {list(mean_importances.keys())[0]} is most important feature across all time buckets")
    print(f"3. Optimal thresholds range from {min(t['optimal_threshold'] for t in optimal_thresholds.values()):.2f} to {max(t['optimal_threshold'] for t in optimal_thresholds.values()):.2f}")


if __name__ == '__main__':
    main()

