"""
Objective function for zone width hyperparameter optimization.

Optimized for SPARSE, HIGH-PRECISION kNN retrieval:
- Goal: Find 5 similar past events with 80%+ same outcome
- Sparse events OK (quality > quantity)
- Optimize for precision@80%, not recall

Uses Optuna + MLflow to find optimal interaction zones and level types.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.common.config import CONFIG

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ZoneObjective:
    """Objective function for zone width optimization.
    
    Evaluates zone configurations by:
    1. Building feature set with given config
    2. Evaluating signal quality (event density, outcome balance)
    3. Training model and measuring performance
    4. Computing composite attribution score
    """
    
    def __init__(
        self,
        train_dates: List[str],
        val_dates: Optional[List[str]] = None,
        target_events_per_day: float = 50.0,
        dry_run: bool = False
    ):
        """
        Initialize objective function.
        
        Args:
            train_dates: Dates for training/validation
            val_dates: Optional separate validation dates
            target_events_per_day: Target event density
            dry_run: If True, use mock data instead of running pipeline
        """
        self.train_dates = train_dates
        self.val_dates = val_dates or train_dates[-5:]  # Last 5 days as val
        self.target_events_per_day = target_events_per_day
        self.dry_run = dry_run
        
        if not dry_run:
            from src.pipeline.pipelines.es_pipeline import build_es_pipeline
            self.pipeline = build_es_pipeline()
        else:
            self.pipeline = None
    
    def __call__(self, trial) -> float:
        """
        Evaluate a configuration (Optuna trial).
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Attribution score (higher is better)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required for hyperopt. Install: uv add optuna")
        
        # 1. Sample hyperparameters
        config = self._sample_config(trial)
        
        # 2. Build feature set with this config
        signals_train = self._build_features(self.train_dates, config)
        
        if signals_train is None or signals_train.empty:
            # No events generated - bad config
            return 0.0
        
        # 3. Evaluate signal quality
        quality_score, quality_metrics = self._evaluate_signal_quality(signals_train)
        
        if quality_score < 0.3:
            # Poor signal quality - prune this trial
            raise optuna.TrialPruned("Signal quality too low")
        
        # 4. Train and evaluate model
        model_score, model_metrics = self._evaluate_model(signals_train)
        
        # 5. Composite attribution score (Optimized for SPARSE kNN Retrieval)
        # 
        # Use case: "Find 5 similar past events, predict with 80%+ precision"
        # 
        # Key insight: Sparse high-quality events > Dense noisy events
        # kNN needs:
        #   - Physics distinctiveness (BREAK ≠ BOUNCE in feature space)
        #   - Retrieval coherence (similar physics → similar outcomes)
        #   - High precision when confident
        
        final_score = (
            0.50 * quality_score +    # Physics quality (kNN coherence + separation)
            0.30 * model_score +       # Precision@80% + AUC
            0.20 * quality_metrics.get('knn_purity_score', 0.5)  # Extra kNN weight
        )
        
        # 6. Log metrics to MLflow
        if MLFLOW_AVAILABLE:
            self._log_metrics(trial, config, quality_metrics, model_metrics, final_score)
        
        return final_score
    
    def _sample_config(self, trial) -> Dict[str, Any]:
        """
        Sample configuration from search space.
        
        Explores combinations of:
        - Zone widths (interaction/touch)
        - Outcome thresholds
        - Physics window sizes
        - Level type selection
        - Adaptive vs fixed zones
        """
        config = {
            # ===== Zone Widths =====
            # Per-level-type monitor bands (interaction zones)
            'monitor_band_pm': trial.suggest_float('monitor_band_pm', 2.0, 15.0),
            'monitor_band_or': trial.suggest_float('monitor_band_or', 2.0, 15.0),
            'monitor_band_sma': trial.suggest_float('monitor_band_sma', 2.0, 20.0),
            
            # Touch bands (precise contact detection)
            'touch_band': trial.suggest_float('touch_band', 1.0, 5.0),
            
            # ===== Outcome Parameters =====
            # Outcome threshold (in strikes, converted to ES points via ATM spacing)
            'outcome_strikes': trial.suggest_int('outcome_strikes', 2, 5),
            
            # Lookforward window (minutes to wait for outcome)
            'lookforward_minutes': trial.suggest_int('lookforward_minutes', 5, 15),
            
            # Lookback context (minutes to analyze approach)
            'lookback_minutes': trial.suggest_int('lookback_minutes', 5, 20),
            
            # ===== Zone Dynamics =====
            # ATR multiplier (0 = fixed zones, >0 = ATR-adaptive zones)
            'k_atr': trial.suggest_float('k_atr', 0.0, 0.5),
            
            # Minimum zone width (safety floor for ATR adaptive)
            'w_min': trial.suggest_float('w_min', 1.0, 5.0),
            
            # ===== Physics Windows (Base windows for engines) =====
            # Barrier lookback window (seconds for depth analysis)
            'W_b': trial.suggest_int('W_b', 120, 360, step=60),  # 2-6 minutes
            
            # Tape lookback window (seconds for flow analysis)
            'W_t': trial.suggest_int('W_t', 30, 120, step=30),   # 30s-2min
            
            # Fuel lookback window (seconds for GEX analysis)
            'W_g': trial.suggest_int('W_g', 30, 120, step=30),
            
            # ===== Multi-Window Feature Configuration =====
            # Which window combinations to use for multi-scale encoding?
            
            # Kinematic windows (velocity, accel, jerk) - UP TO 20MIN per user requirement
            'use_kin_1min': trial.suggest_categorical('use_kin_1min', [True, False]),
            'use_kin_3min': trial.suggest_categorical('use_kin_3min', [True, False]),
            'use_kin_5min': trial.suggest_categorical('use_kin_5min', [True, False]),
            'use_kin_10min': trial.suggest_categorical('use_kin_10min', [True, False]),
            'use_kin_20min': trial.suggest_categorical('use_kin_20min', [True, False]),  # Pre-approach context
            
            # OFI windows
            'use_ofi_30s': trial.suggest_categorical('use_ofi_30s', [True, False]),
            'use_ofi_60s': trial.suggest_categorical('use_ofi_60s', [True, False]),
            'use_ofi_120s': trial.suggest_categorical('use_ofi_120s', [True, False]),
            'use_ofi_300s': trial.suggest_categorical('use_ofi_300s', [True, False]),
            
            # Barrier evolution windows
            'use_barrier_1min': trial.suggest_categorical('use_barrier_1min', [True, False]),
            'use_barrier_3min': trial.suggest_categorical('use_barrier_3min', [True, False]),
            'use_barrier_5min': trial.suggest_categorical('use_barrier_5min', [True, False]),
            
            # ===== Level Type Selection =====
            'use_pm': trial.suggest_categorical('use_pm', [True, False]),
            'use_or': trial.suggest_categorical('use_or', [True, False]),
            'use_sma_200': trial.suggest_categorical('use_sma_200', [True, False]),
            'use_sma_400': trial.suggest_categorical('use_sma_400', [True, False]),
            
            # ===== Confirmation Strategy =====
            # Confirmation window before labeling (Stage B)
            'confirmation_seconds': trial.suggest_int('confirmation_seconds', 120, 360, step=60),
        }
        
        # Convert to CONFIG overrides
        config_overrides = {
            'MONITOR_BAND': config['monitor_band_pm'],  # Simplified - will need per-level support
            'TOUCH_BAND': config['touch_band'],
            'OUTCOME_THRESHOLD': config['outcome_strikes'] * float(CONFIG.ES_0DTE_STRIKE_SPACING),
            'LOOKFORWARD_MINUTES': config['lookforward_minutes'],
            'LOOKBACK_MINUTES': config['lookback_minutes'],
            'W_b': config['W_b'],
            'W_t': config['W_t'],
            'W_g': config['W_g'],
            'CONFIRMATION_WINDOW_SECONDS': config['confirmation_seconds'],
        }
        
        config['config_overrides'] = config_overrides
        return config
    
    def _build_features(
        self,
        dates: List[str],
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Build feature set with given config."""
        
        if self.dry_run:
            # Mock data for dry-run testing
            return self._generate_mock_signals(dates, config)
        
        # Real pipeline execution
        from src.common.utils.config_override import ConfigOverride
        
        all_signals = []
        
        with ConfigOverride(**config['config_overrides']):
            for date in dates:
                try:
                    signals = self.pipeline.run(date)
                    if signals is not None and not signals.empty:
                        all_signals.append(signals)
                except Exception as e:
                    print(f"Warning: Failed to process {date}: {e}")
                    continue
        
        if not all_signals:
            return None
        
        return pd.concat(all_signals, ignore_index=True)
    
    def _generate_mock_signals(
        self,
        dates: List[str],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate mock signals for dry-run testing."""
        
        # Simulate event generation based on config
        # Wider zones → more events
        base_events_per_day = 30
        zone_multiplier = config['monitor_band_pm'] / 5.0  # Relative to baseline 5.0
        events_per_day = int(base_events_per_day * zone_multiplier)
        
        total_events = len(dates) * events_per_day
        
        # Generate mock features
        np.random.seed(42)
        
        mock_data = {
            'event_id': [f'E{i:06d}' for i in range(total_events)],
            'date': np.random.choice(dates, total_events),
            'level_kind_name': np.random.choice(
                ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400'],
                total_events
            ),
            'level_price': np.random.uniform(5700, 5900, total_events),
            'direction': np.random.choice(['UP', 'DOWN'], total_events),
            
            # Mock physics features
            'velocity': np.random.randn(total_events) * 5,
            'acceleration': np.random.randn(total_events) * 2,
            'integrated_ofi': np.random.randn(total_events) * 1000,
            'barrier_depth': np.random.uniform(100, 10000, total_events),
            'gex_asymmetry': np.random.randn(total_events) * 50000,
            
            # Mock outcome (influenced by config)
            'outcome': np.random.choice(
                ['BREAK', 'BOUNCE', 'CHOP'],
                total_events,
                p=[0.35, 0.35, 0.30]
            )
        }
        
        return pd.DataFrame(mock_data)
    
    def _evaluate_signal_quality(
        self,
        signals_df: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate signal quality for neuro-hybrid system.
        
        A good configuration should generate events with:
        1. Appropriate density (not too sparse/dense)
        2. Balanced outcomes (BREAK/BOUNCE/CHOP all represented)
        3. High physics feature variance (diverse states)
        4. Physics-outcome correlation (features predict outcomes)
        5. Retrieval coherence (similar physics → similar outcomes)
        """
        
        event_count = len(signals_df)
        events_per_day = event_count / len(self.train_dates)
        
        # ========== 1. Event Density (RELAXED for sparse retrieval) ==========
        # Sparse events are OK! Quality > Quantity for kNN
        # Target 10-30 events/day (not 50)
        sparse_target = 20.0
        density_diff = abs(events_per_day - sparse_target)
        density_penalty = min(1.0, density_diff / sparse_target)
        density_score = 1 - density_penalty
        
        # But reject if TOO sparse (< 5/day) or TOO dense (> 100/day)
        if events_per_day < 5.0:
            return 0.0, {'error': 'too_sparse', 'events_per_day': events_per_day}
        if events_per_day > 100.0:
            return 0.0, {'error': 'too_dense', 'events_per_day': events_per_day}
        
        # ========== 2. Outcome Balance ==========
        outcomes = signals_df['outcome'].value_counts(normalize=True)
        break_rate = outcomes.get('BREAK', 0.0)
        bounce_rate = outcomes.get('BOUNCE', 0.0)
        chop_rate = outcomes.get('CHOP', 0.0)
        
        # Entropy (want balanced, not all CHOP)
        probs = [p for p in [break_rate, bounce_rate, chop_rate] if p > 0]
        if probs:
            entropy = -sum(p * np.log(p) for p in probs)
            max_entropy = np.log(3)
            entropy_score = entropy / max_entropy
        else:
            entropy_score = 0.0
        
        # Penalty if too much CHOP (want decisive outcomes)
        chop_penalty = max(0, chop_rate - 0.4)  # Penalize if >40% CHOP
        outcome_score = entropy_score * (1 - chop_penalty)
        
        # ========== 3. Physics Feature Variance ==========
        # For kNN retrieval to work, features must vary meaningfully
        # Check variance in key physics features
        
        physics_features = []
        feature_names = []
        
        # Add features that exist in the DataFrame
        if 'velocity' in signals_df.columns:
            physics_features.append(signals_df['velocity'])
            feature_names.append('velocity')
        if 'acceleration' in signals_df.columns:
            physics_features.append(signals_df['acceleration'])
            feature_names.append('acceleration')
        if 'integrated_ofi' in signals_df.columns:
            physics_features.append(signals_df['integrated_ofi'])
            feature_names.append('integrated_ofi')
        if 'barrier_depth' in signals_df.columns:
            physics_features.append(signals_df['barrier_depth'])
            feature_names.append('barrier_depth')
        if 'gex_asymmetry' in signals_df.columns:
            physics_features.append(signals_df['gex_asymmetry'])
            feature_names.append('gex_asymmetry')
        
        if physics_features:
            # Compute normalized variance for each feature
            variances = []
            for feat in physics_features:
                # Remove NaN/inf
                clean_feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_feat) > 0:
                    # Coefficient of variation (std/mean) or just std if mean near 0
                    mean = clean_feat.mean()
                    std = clean_feat.std()
                    if abs(mean) > 1e-6:
                        cv = abs(std / mean)
                    else:
                        cv = std
                    variances.append(min(cv, 10.0))  # Cap at 10 to avoid outliers
            
            if variances:
                # Want high variance (diverse physics states)
                avg_variance = np.mean(variances)
                # Score: 0 if variance=0, 1 if variance>=1
                variance_score = min(1.0, avg_variance)
            else:
                variance_score = 0.0
        else:
            variance_score = 0.5  # Neutral if no features
        
        # ========== 4. Physics-Outcome Correlation ==========
        # Check if physics features correlate with outcomes
        # Good: BREAK events have different physics than BOUNCE
        
        correlation_score = 0.5  # Default neutral
        
        if len(physics_features) > 0 and len(signals_df) > 20:
            # Encode outcomes as numeric
            outcome_map = {'BREAK': 1, 'BOUNCE': -1, 'CHOP': 0}
            y = signals_df['outcome'].map(outcome_map).fillna(0)
            
            correlations = []
            for feat in physics_features:
                clean_feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
                # Pearson correlation
                if clean_feat.std() > 0 and y.std() > 0:
                    corr = np.corrcoef(clean_feat, y)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Absolute correlation
            
            if correlations:
                # Want at least some features to correlate
                max_corr = max(correlations)
                avg_corr = np.mean(correlations)
                correlation_score = 0.7 * max_corr + 0.3 * avg_corr
        
        # ========== 5. kNN-5 Retrieval Coherence (CRITICAL for sparse retrieval!) ==========
        # Check: Do k=5 nearest neighbors have same outcome?
        knn_purity_score = 0.5  # Default neutral
        
        if SKLEARN_AVAILABLE and len(signals_df) > 20 and len(physics_features) > 0:
            try:
                # Build feature matrix (normalize for distance)
                from sklearn.preprocessing import StandardScaler
                
                feat_matrix = np.column_stack([
                    feat.replace([np.inf, -np.inf], np.nan).fillna(0)
                    for feat in physics_features
                ])
                
                scaler = StandardScaler()
                feat_norm = scaler.fit_transform(feat_matrix)
                
                # Find k=5 nearest neighbors
                nbrs = NearestNeighbors(n_neighbors=min(6, len(signals_df))).fit(feat_norm)
                distances, indices = nbrs.kneighbors(feat_norm)
                
                # Compute purity (fraction of neighbors with same outcome)
                outcome_map = {'BREAK': 1, 'BOUNCE': -1, 'CHOP': 0}
                y_numeric = signals_df['outcome'].map(outcome_map).fillna(0).values
                
                purities = []
                for i, neighbor_idx in enumerate(indices):
                    my_outcome = y_numeric[i]
                    neighbor_outcomes = y_numeric[neighbor_idx[1:]]  # Exclude self
                    if len(neighbor_outcomes) > 0:
                        purity = (neighbor_outcomes == my_outcome).sum() / len(neighbor_outcomes)
                        purities.append(purity)
                
                if purities:
                    knn_purity_score = np.mean(purities)
                    # Target: > 0.75 (3+ out of 5 neighbors same outcome)
            
            except Exception as e:
                print(f"Warning: kNN purity calculation failed: {e}")
        
        # ========== 6. Silhouette Score (Between-class separation) ==========
        silhouette_score_val = 0.0
        
        if SKLEARN_AVAILABLE and len(signals_df) > 20 and len(physics_features) > 0:
            try:
                # Encode outcomes for clustering
                outcome_map = {'BREAK': 0, 'BOUNCE': 1, 'CHOP': 2}
                y_labels = signals_df['outcome'].map(outcome_map).fillna(2).astype(int)
                
                # Need at least 2 classes
                if len(np.unique(y_labels)) > 1:
                    feat_matrix = np.column_stack([
                        feat.replace([np.inf, -np.inf], np.nan).fillna(0)
                        for feat in physics_features
                    ])
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    feat_norm = scaler.fit_transform(feat_matrix)
                    
                    silhouette_score_val = silhouette_score(feat_norm, y_labels)
                    # Target: > 0.4 (clear separation between BREAK/BOUNCE/CHOP)
            
            except Exception as e:
                print(f"Warning: Silhouette calculation failed: {e}")
        
        # ========== 7. Level Type Diversity ==========
        # Want events across multiple level types (not all from one level)
        
        if 'level_kind_name' in signals_df.columns:
            level_counts = signals_df['level_kind_name'].value_counts(normalize=True)
            level_probs = level_counts.values
            
            if len(level_probs) > 1:
                level_entropy = -sum(p * np.log(p) for p in level_probs if p > 0)
                max_level_entropy = np.log(len(level_probs))
                level_diversity_score = level_entropy / max_level_entropy
            else:
                level_diversity_score = 0.0  # All from one level type
        else:
            level_diversity_score = 0.5
        
        # ========== Composite Quality Score (Optimized for kNN) ==========
        # Note: Silhouette is logged for diagnostics but not used in the score.
        quality_score = (
            0.05 * density_score +            # Event density (relaxed - sparse OK)
            0.15 * outcome_score +            # Outcome balance (less critical)
            0.20 * variance_score +           # Physics variance
            0.20 * correlation_score +        # Physics-outcome link
            0.35 * knn_purity_score +         # kNN retrieval coherence (CRITICAL!)
            0.00 * silhouette_score_val +     # Logged only
            0.05 * level_diversity_score      # Level type diversity
        )
        
        metrics = {
            'event_count': float(event_count),
            'events_per_day': float(events_per_day),
            'break_rate': float(break_rate),
            'bounce_rate': float(bounce_rate),
            'chop_rate': float(chop_rate),
            'entropy_score': float(entropy_score),
            'density_score': float(density_score),
            'outcome_score': float(outcome_score),
            'variance_score': float(variance_score),
            'correlation_score': float(correlation_score),
            'knn_purity_score': float(knn_purity_score),
            'silhouette_score': float(silhouette_score_val),
            'level_diversity_score': float(level_diversity_score),
            'quality_score': float(quality_score)
        }
        
        return quality_score, metrics
    
    def _evaluate_model(
        self,
        signals_df: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train model and evaluate performance.
        
        CRITICAL: Optimize for PRECISION@80%, not accuracy or recall.
        Use case: "When model is 80%+ confident, it should be RIGHT"
        """
        
        if self.dry_run:
            # Mock model metrics for dry-run
            return self._generate_mock_model_metrics()
        
        # Real model training
        try:
            from src.ml.boosted_tree_train import train_and_evaluate
            
            model, metrics = train_and_evaluate(
                signals_df=signals_df,
                target='outcome',
                test_size=0.2,
                random_state=42
            )
            
            auc = metrics.get('auc', 0.0)
            brier = metrics.get('brier_score', 1.0)
            precision_80 = metrics.get('precision_at_80', 0.0)
            precision_90 = metrics.get('precision_at_90', 0.0)
            
            # Model score (HEAVILY weighted to precision at high confidence)
            model_score = (
                0.50 * precision_80 +      # When 80% confident, be right (CRITICAL)
                0.20 * precision_90 +      # When 90% confident, be very right
                0.20 * auc +               # Overall discrimination
                0.10 * (1 - brier)         # Calibration (less critical)
            )
            
            metrics['model_score'] = model_score
            return model_score, metrics
            
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            return 0.0, {}
    
    def _generate_mock_model_metrics(self) -> Tuple[float, Dict[str, float]]:
        """Generate mock model metrics for dry-run."""
        
        # Simulate reasonable model performance (optimized for precision)
        auc = np.random.uniform(0.65, 0.80)
        brier = np.random.uniform(0.15, 0.30)
        precision_80 = np.random.uniform(0.70, 0.90)  # High precision at 80% threshold
        precision_90 = np.random.uniform(0.80, 0.95)  # Very high at 90%
        
        model_score = (
            0.50 * precision_80 +
            0.20 * precision_90 +
            0.20 * auc +
            0.10 * (1 - brier)
        )
        
        metrics = {
            'auc': float(auc),
            'brier_score': float(brier),
            'precision_at_80': float(precision_80),
            'precision_at_90': float(precision_90),
            'model_score': float(model_score)
        }
        
        return model_score, metrics
    
    def _log_metrics(
        self,
        trial,
        config: Dict[str, Any],
        quality_metrics: Dict[str, float],
        model_metrics: Dict[str, float],
        final_score: float
    ):
        """Log metrics to MLflow."""
        
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Log hyperparameters
            for key, value in config.items():
                if key != 'config_overrides':
                    mlflow.log_param(f'trial_{trial.number}_{key}', value)
            
            # Log quality metrics
            for key, value in quality_metrics.items():
                mlflow.log_metric(f'trial_{trial.number}_quality_{key}', value)
            
            # Log model metrics
            for key, value in model_metrics.items():
                mlflow.log_metric(f'trial_{trial.number}_model_{key}', value)
            
            # Log final score
            mlflow.log_metric(f'trial_{trial.number}_final_score', final_score)
            
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
