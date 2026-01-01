"""
Level-Specific Analysis

Analyzes whether OR_LOW and PM_LOW exhibit different BREAK/REJECT dynamics
and whether they need separate models or level-specific adjustments.

Key Questions:
1. Do OR_LOW and PM_LOW have different base BREAK rates?
2. Do different features matter for each level type?
3. Is PM (first level of day) more predictable than OR (second level)?

Usage:
    cd backend
    uv run python -m scripts.analyze_level_specificity \
        --version v4.0.0 \
        --start 2025-11-03 \
        --end 2025-12-19 \
        --horizon 4min \
        --output-json data/ml/level_specificity_analysis.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(version: str, start_date: str, end_date: str):
    """Load episodes and streams."""
    # Episodes
    episodes_dir = Path(f'data/gold/episodes/es_level_episodes/version={version}/metadata')
    all_meta = []
    for date_dir in sorted(episodes_dir.glob('date=*')):
        date_str = date_dir.name.split('=')[1]
        if start_date <= date_str <= end_date:
            meta_path = date_dir / 'metadata.parquet'
            if meta_path.exists():
                all_meta.append(pd.read_parquet(meta_path))
    episodes = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    
    # Streams
    stream_dir = Path(f'data/gold/streams/pentaview/version={version}')
    all_streams = []
    for date_dir in sorted(stream_dir.glob('date=*')):
        stream_path = date_dir / 'stream_bars.parquet'
        if stream_path.exists():
            all_streams.append(pd.read_parquet(stream_path))
    streams = pd.concat(all_streams, ignore_index=True) if all_streams else pd.DataFrame()
    
    return episodes, streams


def compute_level_stats(
    df: pd.DataFrame,
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict]:
    """Compute statistics per level type."""
    stats = {}
    
    for level_kind in ['PM_LOW', 'OR_LOW']:
        level_df = df[df['level_kind'] == level_kind]
        
        if len(level_df) == 0:
            continue
        
        # Overall outcome distribution
        outcomes = level_df[outcome_col].value_counts()
        
        # By time bucket
        time_breakdown = {}
        for bucket in level_df['time_bucket'].unique():
            bucket_df = level_df[level_df['time_bucket'] == bucket]
            bucket_outcomes = bucket_df[outcome_col].value_counts()
            time_breakdown[bucket] = {
                'n': len(bucket_df),
                'break_rate': float(bucket_outcomes.get('BREAK', 0) / len(bucket_df)),
                'reject_rate': float(bucket_outcomes.get('REJECT', 0) / len(bucket_df)),
                'chop_rate': float(bucket_outcomes.get('CHOP', 0) / len(bucket_df))
            }
        
        stats[level_kind] = {
            'n_total': len(level_df),
            'break_count': int(outcomes.get('BREAK', 0)),
            'reject_count': int(outcomes.get('REJECT', 0)),
            'chop_count': int(outcomes.get('CHOP', 0)),
            'break_rate_overall': float(outcomes.get('BREAK', 0) / len(level_df)),
            'reject_rate_overall': float(outcomes.get('REJECT', 0) / len(level_df)),
            'time_breakdown': time_breakdown
        }
    
    return stats


def train_level_specific_models(
    df: pd.DataFrame,
    features: List[str],
    outcome_col: str = 'outcome_4min'
) -> Dict[str, Dict]:
    """Train separate models for PM_LOW and OR_LOW."""
    results = {}
    
    for level_kind in ['PM_LOW', 'OR_LOW']:
        level_df = df[df['level_kind'] == level_kind].copy()
        
        # Filter to BREAK/REJECT
        mask = level_df[outcome_col].isin(['BREAK', 'REJECT'])
        level_df = level_df[mask]
        
        if len(level_df) < 30:
            print(f"  {level_kind}: Insufficient samples ({len(level_df)})")
            continue
        
        X = level_df[features].fillna(0).values
        y = (level_df[outcome_col] == 'BREAK').astype(int).values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        
        # CV
        cv = StratifiedKFold(n_splits=min(5, len(level_df) // 20), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # Fit
        model.fit(X_scaled, y)
        
        # Feature importances
        importances = dict(zip(features, model.feature_importances_))
        
        # Analyze predictions
        y_pred = model.predict_proba(X_scaled)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_pred)
        
        results[level_kind] = {
            'model': model,
            'scaler': scaler,
            'cv_auc': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'feature_importances': {k: float(v) for k, v in importances.items()},
            'n_train': len(level_df),
            'break_rate': float(y.mean()),
            'precision_at_50': float(precision[np.argmin(np.abs(thresholds - 0.5))]),
            'recall_at_50': float(recall[np.argmin(np.abs(thresholds - 0.5))])
        }
        
        print(f"  {level_kind}: n={len(level_df)}, AUC={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}, break_rate={y.mean():.1%}")
    
    return results


def compare_feature_importances(
    pm_model: Dict,
    or_model: Dict,
    features: List[str]
) -> pd.DataFrame:
    """Compare feature importances between PM and OR models."""
    pm_imp = [pm_model['feature_importances'].get(f, 0.0) for f in features]
    or_imp = [or_model['feature_importances'].get(f, 0.0) for f in features]
    
    df = pd.DataFrame({
        'feature': features,
        'PM_LOW': pm_imp,
        'OR_LOW': or_imp,
        'diff': np.array(pm_imp) - np.array(or_imp),
        'abs_diff': np.abs(np.array(pm_imp) - np.array(or_imp))
    })
    
    df = df.sort_values('abs_diff', ascending=False)
    
    return df


def test_feature_distributions(
    df: pd.DataFrame,
    features: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Test if feature distributions differ between PM_LOW and OR_LOW.
    
    Uses Mann-Whitney U test for each feature.
    """
    results = {}
    
    pm_df = df[df['level_kind'] == 'PM_LOW']
    or_df = df[df['level_kind'] == 'OR_LOW']
    
    for feat in features:
        pm_vals = pm_df[feat].dropna()
        or_vals = or_df[feat].dropna()
        
        if len(pm_vals) < 10 or len(or_vals) < 10:
            continue
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(pm_vals, or_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pm_mean, pm_std = pm_vals.mean(), pm_vals.std()
        or_mean, or_std = or_vals.mean(), or_vals.std()
        pooled_std = np.sqrt((pm_std**2 + or_std**2) / 2)
        cohens_d = (pm_mean - or_mean) / (pooled_std + 1e-9)
        
        results[feat] = {
            'pm_mean': float(pm_mean),
            'pm_std': float(pm_std),
            'or_mean': float(or_mean),
            'or_std': float(or_std),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
        }
    
    return results


def analyze_level_sequence(
    df: pd.DataFrame,
    outcome_col: str = 'outcome_4min'
) -> Dict[str, float]:
    """
    Analyze if level order matters (is PM_LOW touched first vs OR_LOW first?).
    """
    # Group by date
    daily_groups = df.groupby('date')
    
    pm_first_breaks = []
    or_first_breaks = []
    
    for date, group in daily_groups:
        pm_rows = group[group['level_kind'] == 'PM_LOW'].sort_values('timestamp')
        or_rows = group[group['level_kind'] == 'OR_LOW'].sort_values('timestamp')
        
        if len(pm_rows) > 0 and len(or_rows) > 0:
            pm_first_time = pm_rows.iloc[0]['timestamp']
            or_first_time = or_rows.iloc[0]['timestamp']
            
            if pm_first_time < or_first_time:
                # PM touched first
                pm_outcome = pm_rows.iloc[0][outcome_col]
                pm_first_breaks.append(1 if pm_outcome == 'BREAK' else 0)
            else:
                # OR touched first
                or_outcome = or_rows.iloc[0][outcome_col]
                or_first_breaks.append(1 if or_outcome == 'BREAK' else 0)
    
    return {
        'pm_first_break_rate': float(np.mean(pm_first_breaks)) if pm_first_breaks else None,
        'or_first_break_rate': float(np.mean(or_first_breaks)) if or_first_breaks else None,
        'n_pm_first': len(pm_first_breaks),
        'n_or_first': len(or_first_breaks)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze level-specific dynamics')
    parser.add_argument('--version', default='v4.0.0')
    parser.add_argument('--start', default='2025-11-03')
    parser.add_argument('--end', default='2025-12-19')
    parser.add_argument('--horizon', default='4min')
    parser.add_argument('--output-json', default='data/ml/level_specificity_analysis.json')
    
    args = parser.parse_args()
    
    outcome_col = f'outcome_{args.horizon}'
    
    print("=== Level-Specific Analysis (PM_LOW vs OR_LOW) ===\n")
    
    # Load data
    print("Loading data...")
    episodes, streams = load_data(args.version, args.start, args.end)
    print(f"Loaded {len(episodes)} episodes, {len(streams)} stream bars")
    
    # Merge
    episodes['ts_str'] = episodes['timestamp'].astype(str)
    streams['ts_str'] = streams['timestamp'].astype(str)
    merged = episodes.merge(streams, on=['ts_str', 'level_kind'], how='left')
    
    # Filter to predictable segment
    predictable_mask = (
        (merged['direction'] == 'UP') &
        (merged['level_kind'].isin(['PM_LOW', 'OR_LOW'])) &
        (~merged['time_bucket'].isin(['T0_15']))
    )
    predictable = merged[predictable_mask].copy()
    
    print(f"Predictable segment: {len(predictable)} episodes")
    print(f"  PM_LOW: {(predictable['level_kind'] == 'PM_LOW').sum()}")
    print(f"  OR_LOW: {(predictable['level_kind'] == 'OR_LOW').sum()}")
    
    stream_features = ['sigma_s', 'sigma_r', 'sigma_b_slope', 'sigma_p_slope', 'sigma_d']
    
    # 1. Basic statistics
    print("\n=== 1. Level Statistics ===")
    level_stats = compute_level_stats(predictable, outcome_col)
    
    for level, stats in level_stats.items():
        print(f"\n{level}:")
        print(f"  Total: {stats['n_total']}")
        print(f"  BREAK rate: {stats['break_rate_overall']:.1%}")
        print(f"  REJECT rate: {stats['reject_rate_overall']:.1%}")
    
    # Statistical test: Are BREAK rates different?
    print("\n=== 2. Statistical Test ===")
    from scipy.stats import chi2_contingency
    
    pm_stats = level_stats.get('PM_LOW', {})
    or_stats = level_stats.get('OR_LOW', {})
    
    if pm_stats and or_stats:
        contingency = [
            [pm_stats['break_count'], pm_stats['reject_count']],
            [or_stats['break_count'], or_stats['reject_count']]
        ]
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        print(f"Chi-square test (PM_LOW vs OR_LOW):")
        print(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → SIGNIFICANT: BREAK rates differ between PM_LOW and OR_LOW")
        else:
            print(f"  → Not significant: Similar BREAK rates")
    
    # 3. Feature distributions
    print("\n=== 3. Feature Distribution Tests ===")
    feature_dists = test_feature_distributions(predictable, stream_features)
    
    print("\nFeatures with significant differences (p < 0.05):")
    for feat, test_result in feature_dists.items():
        if test_result['significant']:
            print(f"  {feat}:")
            print(f"    PM_LOW: {test_result['pm_mean']:.3f}±{test_result['pm_std']:.3f}")
            print(f"    OR_LOW: {test_result['or_mean']:.3f}±{test_result['or_std']:.3f}")
            print(f"    Cohen's d: {test_result['cohens_d']:.2f} ({test_result['effect_size']})")
    
    # 4. Level-specific models
    print("\n=== 4. Level-Specific Models ===")
    level_models = train_level_specific_models(predictable, stream_features, outcome_col)
    
    # 5. Feature importance comparison
    if 'PM_LOW' in level_models and 'OR_LOW' in level_models:
        print("\n=== 5. Feature Importance Comparison ===")
        feature_comp = compare_feature_importances(
            level_models['PM_LOW'],
            level_models['OR_LOW'],
            stream_features
        )
        
        print("\nTop features by importance difference:")
        for _, row in feature_comp.head(5).iterrows():
            print(f"  {row['feature']}:")
            print(f"    PM_LOW: {row['PM_LOW']:.3f}")
            print(f"    OR_LOW: {row['OR_LOW']:.3f}")
            print(f"    Diff: {row['diff']:+.3f}")
    
    # 6. Level sequence analysis
    print("\n=== 6. Level Sequence Analysis ===")
    sequence_analysis = analyze_level_sequence(predictable, outcome_col)
    
    if sequence_analysis['pm_first_break_rate']:
        print(f"When PM_LOW touched first:")
        print(f"  BREAK rate: {sequence_analysis['pm_first_break_rate']:.1%}")
        print(f"  N: {sequence_analysis['n_pm_first']}")
    
    if sequence_analysis['or_first_break_rate']:
        print(f"When OR_LOW touched first:")
        print(f"  BREAK rate: {sequence_analysis['or_first_break_rate']:.1%}")
        print(f"  N: {sequence_analysis['n_or_first']}")
    
    # Build output
    output = {
        'meta': {
            'version': args.version,
            'start_date': args.start,
            'end_date': args.end,
            'horizon': args.horizon,
            'n_episodes': len(predictable)
        },
        'level_stats': level_stats,
        'statistical_test': {
            'chi2': float(chi2),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        } if pm_stats and or_stats else None,
        'feature_distributions': feature_dists,
        'level_specific_models': {
            level: {
                'cv_auc': model_dict['cv_auc'],
                'cv_auc_std': model_dict['cv_auc_std'],
                'n_train': model_dict['n_train'],
                'break_rate': model_dict['break_rate'],
                'feature_importances': model_dict['feature_importances']
            }
            for level, model_dict in level_models.items()
        },
        'feature_importance_comparison': feature_comp.to_dict(orient='records') if 'PM_LOW' in level_models and 'OR_LOW' in level_models else None,
        'sequence_analysis': sequence_analysis,
        'recommendation': None
    }
    
    # Generate recommendation
    if output['statistical_test'] and output['statistical_test']['significant']:
        if 'PM_LOW' in level_models and 'OR_LOW' in level_models:
            auc_diff = abs(level_models['PM_LOW']['cv_auc'] - level_models['OR_LOW']['cv_auc'])
            if auc_diff > 0.05:
                recommendation = "Use level-specific models - significant performance difference"
            else:
                recommendation = "Use level-adjusted thresholds - different base rates but similar predictability"
        else:
            recommendation = "Acknowledge different base rates but insufficient data for separate models"
    else:
        recommendation = "Use unified model - PM_LOW and OR_LOW behave similarly"
    
    output['recommendation'] = recommendation
    
    # Save
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Analysis saved to: {output_path}")
    print(f"Recommendation: {recommendation}")
    
    if pm_stats and or_stats:
        print(f"\nKey Finding:")
        print(f"  PM_LOW BREAK rate: {pm_stats['break_rate_overall']:.1%}")
        print(f"  OR_LOW BREAK rate: {or_stats['break_rate_overall']:.1%}")
        print(f"  Difference: {abs(pm_stats['break_rate_overall'] - or_stats['break_rate_overall']):.1%}")


if __name__ == '__main__':
    main()

