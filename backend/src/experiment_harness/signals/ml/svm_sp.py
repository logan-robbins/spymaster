"""SVM Spatial Profile (SVM_SP) walk-forward ML signal.

LinearSVC on spatial pressure-vacuum profiles with rolling statistical
features.  Walk-forward expanding window with periodic retraining.

Feature vector per bin (45 features total):
    1. Spatial PV difference profile at 21 sampled ticks (every 5th tick)
    2. Band asymmetry (inner/mid/outer combined add+pull) rolling mean/std
       at 3 windows (50, 200, 600) = 3 bands x 3 windows x 2 stats = 18
    3. Mid-price return rolling mean/std at 3 windows = 6

Output: signed decision function values (confidence mode).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

# Tick size for converting mid_price returns to tick units
_TICK_SIZE: float = 0.25


class SVMSPSignal(MLSignal):
    """Walk-forward LinearSVC on spatial PV profiles with rolling features.

    Trains a linear SVM on the expanding window of labeled bins,
    refitting every ``retrain_interval`` bins.  Predictions are raw
    decision function values (signed distance from the separating
    hyperplane).  Positive = bullish, negative = bearish.

    Args:
        sampled_k_indices: Indices into the 101-tick grid to sample for
            the spatial PV difference profile.  Default: every 5th tick
            (21 samples).
        rolling_windows: Window sizes for rolling mean/std features on
            band asymmetry and mid-price returns.
        min_train_bins: Minimum number of bins before first prediction.
        retrain_interval: Number of bins between model refits.
        cooldown_bins: Minimum bins between signal firings (metadata only).
        svm_c: Regularization parameter for LinearSVC.
    """

    def __init__(
        self,
        sampled_k_indices: list[int] | None = None,
        rolling_windows: tuple[int, ...] = (50, 200, 600),
        min_train_bins: int = 1200,
        retrain_interval: int = 300,
        cooldown_bins: int = 30,
        svm_c: float = 0.1,
    ) -> None:
        if sampled_k_indices is None:
            sampled_k_indices = list(range(0, 101, 5))
        self.sampled_k_indices: list[int] = sampled_k_indices
        self.rolling_windows: tuple[int, ...] = rolling_windows
        self.min_train_bins: int = min_train_bins
        self.retrain_interval: int = retrain_interval
        self.cooldown_bins: int = cooldown_bins
        self.svm_c: float = svm_c

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "svm_sp"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return ["pressure_variant", "vacuum_variant", "v_add", "v_pull"]

    def default_thresholds(self) -> list[float]:
        """Default confidence thresholds for decision function values."""
        return [0.0, 0.2, 0.4, 0.6, 0.8]

    @property
    def prediction_mode(self) -> str:
        """Predictions are signed decision function values."""
        return "confidence"

    def _build_features(
        self,
        pv_diff: np.ndarray,
        v_add: np.ndarray,
        v_pull: np.ndarray,
        mid_price: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """Build feature matrix (n_bins, 45).

        Args:
            pv_diff: (n_bins, 101) pressure - vacuum difference field.
            v_add: (n_bins, 101) add velocity grid.
            v_pull: (n_bins, 101) pull velocity grid.
            mid_price: (n_bins,) mid prices.
            n_bins: Number of bins.

        Returns:
            (n_bins, 45) feature matrix with NaN/Inf cleaned to zero.
        """
        features_list: list[np.ndarray] = []

        # 1. Sampled spatial PV profile (21 features)
        spatial: np.ndarray = pv_diff[:, self.sampled_k_indices]
        features_list.append(spatial)

        # 2. Band asymmetry rolling stats (18 features)
        for band in DEFAULT_BAND_DEFS:
            bid_cols: list[int] = band["bid_cols"]  # type: ignore[assignment]
            ask_cols: list[int] = band["ask_cols"]  # type: ignore[assignment]

            add_asym: np.ndarray = (
                v_add[:, bid_cols].mean(axis=1) - v_add[:, ask_cols].mean(axis=1)
            )
            pull_asym: np.ndarray = (
                v_pull[:, ask_cols].mean(axis=1) - v_pull[:, bid_cols].mean(axis=1)
            )
            combined: np.ndarray = add_asym + pull_asym

            for w in self.rolling_windows:
                rmean, rstd = rolling_mean_std(combined, w)
                features_list.append(rmean.reshape(-1, 1))
                features_list.append(rstd.reshape(-1, 1))

        # 3. Mid-price return rolling stats (6 features)
        returns: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        returns[1:] = np.diff(mid_price) / _TICK_SIZE

        for w in self.rolling_windows:
            rmean, rstd = rolling_mean_std(returns, w)
            features_list.append(rmean.reshape(-1, 1))
            features_list.append(rstd.reshape(-1, 1))

        X: np.ndarray = np.hstack(features_list)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward SVM prediction across the dataset.

        Args:
            dataset: Dict with keys mid_price, ts_ns, n_bins, k_values,
                pressure_variant, vacuum_variant, v_add, v_pull (each
                (n_bins, 101) float64 arrays).

        Returns:
            MLSignalResult with decision function values and validity mask.
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        P: np.ndarray = dataset["pressure_variant"]
        V: np.ndarray = dataset["vacuum_variant"]
        v_add: np.ndarray = dataset["v_add"]
        v_pull: np.ndarray = dataset["v_pull"]

        pv_diff: np.ndarray = P - V

        # Build features
        X: np.ndarray = self._build_features(pv_diff, v_add, v_pull, mid_price, n_bins)
        logger.info("SVM_SP: feature matrix shape=%s", X.shape)

        # Compute labels
        labels: np.ndarray = compute_labels(mid_price, n_bins)

        # Walk-forward prediction
        predictions: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)

        model: LinearSVC | None = None
        scaler: StandardScaler | None = None
        last_train_bin: int = -1
        retrain_count: int = 0

        start_bin: int = self.min_train_bins

        for i in range(start_bin, n_bins):
            # Check if retrain is needed
            if model is None or (i - last_train_bin) >= self.retrain_interval:
                train_mask: np.ndarray = labels[:i] != 0
                if train_mask.sum() < 20:
                    continue

                X_train: np.ndarray = X[:i][train_mask]
                y_train: np.ndarray = labels[:i][train_mask]

                scaler = StandardScaler()
                X_train_s: np.ndarray = scaler.fit_transform(X_train)

                model = LinearSVC(
                    C=self.svm_c,
                    loss="hinge",
                    max_iter=2000,
                    class_weight="balanced",
                    dual=True,
                )
                model.fit(X_train_s, y_train)
                last_train_bin = i
                retrain_count += 1

            # Predict at current bin
            x_i: np.ndarray = scaler.transform(X[i : i + 1])  # type: ignore[union-attr]
            dec: float = float(model.decision_function(x_i)[0])
            predictions[i] = dec
            has_prediction[i] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info(
            "SVM_SP: %d predictions, %d retrains, feature_dim=%d",
            n_predicted,
            retrain_count,
            X.shape[1],
        )

        return MLSignalResult(
            predictions=predictions,
            has_prediction=has_prediction,
            metadata={
                "n_features": X.shape[1],
                "retrain_count": retrain_count,
                "n_predicted": n_predicted,
            },
        )
