"""LightGBM Multi-Feature (GBM_MF) walk-forward ML signal.

Gradient boosted trees on a rich feature vector from multiple derivative
channels.  Walk-forward with expanding window retrain and early stopping
on a held-out validation split.

Feature vector per bin (53 features total):
    1. Derivative asymmetries: 6 columns x 3 bands = 18
    2. Rolling OLS slopes of the 18 asymmetries (window=50) = 18
    3. PV net at spot (k=0) + rolling mean/std at [50, 300] = 5
    4. Mid-price return rolling mean/std/skew at [50, 200, 600] = 9
    5. Spread proxy + rolling mean/std at 200 = 3

Output: P(up) probabilities (probability mode).
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np

from src.experiment_harness.eval_engine import rolling_ols_slope, robust_zscore
from src.experiment_harness.signals.base import MLSignal, MLSignalResult
from src.experiment_harness.signals.features import (
    DEFAULT_BAND_DEFS,
    band_asymmetry,
    distance_weighted_sum,
    rolling_mean_std,
)
from src.experiment_harness.signals.labels import compute_labels

logger = logging.getLogger(__name__)

# Suppress LightGBM categorical warnings during training
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

_TICK_SIZE: float = 0.25

_DERIV_COLS: tuple[str, ...] = (
    "v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
)

_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": 1,
    "seed": 42,
}


class GBMMFSignal(MLSignal):
    """Walk-forward LightGBM on derivative asymmetries and return statistics.

    Trains a binary classifier (P(up) vs P(down)) on labeled bins,
    refitting every ``retrain_interval`` bins with early stopping on a
    held-out 20% validation split.

    Args:
        slope_window: Window size for rolling OLS slopes on asymmetries.
        rolling_windows: Windows for return rolling mean/std/skew.
        min_train_bins: Minimum bins before first prediction.
        retrain_interval: Bins between model refits.
        cooldown_bins: Minimum bins between signal firings (metadata only).
        num_boost_round: Maximum boosting rounds.
        early_stopping: Patience rounds for early stopping.
        lgb_params: Override LightGBM parameters.  Merged with defaults.
    """

    def __init__(
        self,
        slope_window: int = 50,
        rolling_windows: tuple[int, ...] = (50, 200, 600),
        min_train_bins: int = 2400,
        retrain_interval: int = 600,
        cooldown_bins: int = 30,
        num_boost_round: int = 200,
        early_stopping: int = 20,
        lgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.slope_window: int = slope_window
        self.rolling_windows: tuple[int, ...] = rolling_windows
        self.min_train_bins: int = min_train_bins
        self.retrain_interval: int = retrain_interval
        self.cooldown_bins: int = cooldown_bins
        self.num_boost_round: int = num_boost_round
        self.early_stopping: int = early_stopping
        self.lgb_params: dict[str, Any] = {**_DEFAULT_LGB_PARAMS}
        if lgb_params is not None:
            self.lgb_params.update(lgb_params)

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "gbm_mf"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return [
            "v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
            "pressure_variant", "vacuum_variant",
        ]

    def default_thresholds(self) -> list[float]:
        """Default probability thresholds for P(up)."""
        return [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

    @property
    def prediction_mode(self) -> str:
        """Predictions are P(up) probabilities."""
        return "probability"

    def _build_features(
        self,
        grids: dict[str, np.ndarray],
        mid_price: np.ndarray,
        n_bins: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Build feature matrix (n_bins, 53) with named columns.

        Args:
            grids: Dict mapping column name to (n_bins, 101) arrays.
            mid_price: (n_bins,) mid prices.
            n_bins: Number of bins.

        Returns:
            Tuple of (X, feature_names) where X is (n_bins, 53).
        """
        features: list[np.ndarray] = []
        names: list[str] = []

        # 1. Derivative asymmetries (18 features)
        asym_signals: list[np.ndarray] = []
        for col_name in _DERIV_COLS:
            grid: np.ndarray = grids[col_name]
            for band in DEFAULT_BAND_DEFS:
                band_name: str = band["name"]  # type: ignore[assignment]
                bid_cols: list[int] = band["bid_cols"]  # type: ignore[assignment]
                ask_cols: list[int] = band["ask_cols"]  # type: ignore[assignment]

                if "add" in col_name:
                    asym = grid[:, bid_cols].mean(axis=1) - grid[:, ask_cols].mean(axis=1)
                else:
                    asym = grid[:, ask_cols].mean(axis=1) - grid[:, bid_cols].mean(axis=1)

                features.append(asym.reshape(-1, 1))
                names.append(f"{col_name}_{band_name}_asym")
                asym_signals.append(asym)

        # 2. Rolling OLS slopes of asymmetries (18 features)
        for idx, asym in enumerate(asym_signals):
            slope: np.ndarray = rolling_ols_slope(asym, self.slope_window)
            slope = np.nan_to_num(slope, nan=0.0)
            features.append(slope.reshape(-1, 1))
            names.append(f"slope_{names[idx]}")

        # 3. PV net at spot + rolling stats (5 features)
        pv_spot: np.ndarray = (
            grids["pressure_variant"][:, 50] - grids["vacuum_variant"][:, 50]
        )
        features.append(pv_spot.reshape(-1, 1))
        names.append("pv_spot")
        for w in (50, 300):
            rm, rs = rolling_mean_std(pv_spot, w)
            features.append(rm.reshape(-1, 1))
            features.append(rs.reshape(-1, 1))
            names.extend([f"pv_spot_mean_{w}", f"pv_spot_std_{w}"])

        # 4. Mid-price return rolling stats (9 features: mean/std/skew x 3)
        returns: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        returns[1:] = np.diff(mid_price) / _TICK_SIZE
        for w in self.rolling_windows:
            rm, rs = rolling_mean_std(returns, w)
            features.append(rm.reshape(-1, 1))
            features.append(rs.reshape(-1, 1))
            names.extend([f"ret_mean_{w}", f"ret_std_{w}"])

            # Rolling skewness via cumsums
            cs: np.ndarray = np.cumsum(returns)
            cs2: np.ndarray = np.cumsum(returns ** 2)
            cs3: np.ndarray = np.cumsum(returns ** 3)
            skew: np.ndarray = np.zeros(n_bins, dtype=np.float64)
            for i in range(w - 1, n_bins):
                s = cs[i] - (cs[i - w] if i >= w else 0.0)
                s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
                s3 = cs3[i] - (cs3[i - w] if i >= w else 0.0)
                m = s / w
                var = s2 / w - m * m
                if var > 1e-12:
                    std = np.sqrt(var)
                    m3 = s3 / w - 3.0 * m * s2 / w + 2.0 * m ** 3
                    skew[i] = m3 / (std ** 3)
            features.append(skew.reshape(-1, 1))
            names.append(f"ret_skew_{w}")

        # 5. Spread proxy + rolling stats (3 features)
        spread_proxy: np.ndarray = (
            grids["pressure_variant"][:, 51] - grids["pressure_variant"][:, 49]
        )
        features.append(spread_proxy.reshape(-1, 1))
        names.append("spread_proxy")
        rm_sp, rs_sp = rolling_mean_std(spread_proxy, 200)
        features.append(rm_sp.reshape(-1, 1))
        features.append(rs_sp.reshape(-1, 1))
        names.extend(["spread_mean_200", "spread_std_200"])

        X: np.ndarray = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, names

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward LightGBM prediction across the dataset.

        Args:
            dataset: Dict with keys mid_price, ts_ns, n_bins, k_values,
                and all required_columns as (n_bins, 101) arrays.

        Returns:
            MLSignalResult with P(up) probabilities and validity mask.
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        grids: dict[str, np.ndarray] = {
            c: dataset[c] for c in self.required_columns
        }

        # Build features
        X, feat_names = self._build_features(grids, mid_price, n_bins)
        logger.info("GBM_MF: feature matrix shape=%s, %d features", X.shape, len(feat_names))

        # Compute labels and convert to binary
        labels: np.ndarray = compute_labels(mid_price, n_bins)
        binary_labels: np.ndarray = np.zeros(n_bins, dtype=np.int32)
        binary_labels[labels == 1] = 1
        binary_labels[labels == -1] = 0

        # Walk-forward prediction
        proba_up: np.ndarray = np.full(n_bins, 0.5, dtype=np.float64)
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)

        model: lgb.Booster | None = None
        last_train_bin: int = -1
        retrain_count: int = 0
        best_iterations: list[int] = []
        feature_importance: np.ndarray | None = None

        start_bin: int = self.min_train_bins

        for i in range(start_bin, n_bins):
            if model is None or (i - last_train_bin) >= self.retrain_interval:
                train_mask: np.ndarray = labels[:i] != 0
                if train_mask.sum() < 40:
                    continue

                X_train: np.ndarray = X[:i][train_mask]
                y_train: np.ndarray = binary_labels[:i][train_mask]

                # 80/20 split for early stopping
                split: int = int(len(X_train) * 0.8)
                if split < 20 or (len(X_train) - split) < 10:
                    continue

                dtrain: lgb.Dataset = lgb.Dataset(
                    X_train[:split], label=y_train[:split]
                )
                dval: lgb.Dataset = lgb.Dataset(
                    X_train[split:], label=y_train[split:], reference=dtrain
                )

                callbacks: list[Any] = [
                    lgb.early_stopping(self.early_stopping, verbose=False),
                ]
                model = lgb.train(
                    self.lgb_params,
                    dtrain,
                    num_boost_round=self.num_boost_round,
                    valid_sets=[dval],
                    callbacks=callbacks,
                )
                last_train_bin = i
                retrain_count += 1
                best_iterations.append(model.best_iteration)
                feature_importance = model.feature_importance(
                    importance_type="gain"
                )

            # Predict P(up) for current bin
            prob: float = float(model.predict(X[i : i + 1])[0])  # type: ignore[union-attr]
            proba_up[i] = prob
            has_prediction[i] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info(
            "GBM_MF: %d predictions, %d retrains",
            n_predicted,
            retrain_count,
        )

        # Build top-feature metadata
        top_features: dict[str, float] = {}
        if feature_importance is not None and len(feat_names) == len(feature_importance):
            top_idx: np.ndarray = np.argsort(feature_importance)[::-1][:10]
            top_features = {
                feat_names[idx]: float(feature_importance[idx])
                for idx in top_idx
            }

        return MLSignalResult(
            predictions=proba_up,
            has_prediction=has_prediction,
            metadata={
                "n_features": len(feat_names),
                "feature_names": feat_names,
                "retrain_count": retrain_count,
                "n_predicted": n_predicted,
                "best_iterations": best_iterations,
                "top_features_gain": top_features,
            },
        )
