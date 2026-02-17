"""Linear SVM Derivative (LSVM_DER) walk-forward ML signal.

Online linear SVM (SGDClassifier with hinge loss) on rolling derivative
slope features with periodic batch retrain.  Uses ONLY derivative-chain
features (v, a, j) with no pressure/vacuum composites.

Feature vector per bin (60 features total):
    1. Band asymmetries: 6 columns x 3 bands = 18
    2. Rolling OLS slopes at window=100: 18 asymmetries = 18
    3. Rolling OLS slopes at window=300: 18 asymmetries = 18
    4. Full-width distance-weighted divergences: 6 columns = 6

Output: signed decision function values (confidence mode).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

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

_DERIV_COLS: tuple[str, ...] = (
    "v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
)

# Full-width bid/ask columns for distance-weighted divergence
_BID_COLS_FULL: np.ndarray = np.arange(26, 50)   # k=-24..-1
_ASK_COLS_FULL: np.ndarray = np.arange(51, 75)   # k=+1..+24


class LSVMDERSignal(MLSignal):
    """Walk-forward SGDClassifier (online SVM) on derivative slopes.

    Combines batch refitting with incremental partial_fit updates
    between full refits.  Uses warm_start for continuity across
    partial updates.

    Args:
        slope_windows: Window sizes for rolling OLS slopes on
            band asymmetries.
        min_train: Minimum labeled bins before first prediction.
        retrain_interval: Bins between partial fit updates.
        full_refit_interval: Bins between full model refits from
            scratch (resets model weights entirely).
        cooldown_bins: Minimum bins between signal firings (metadata).
        alpha: SGD regularization strength (L2 penalty).
    """

    def __init__(
        self,
        slope_windows: tuple[int, ...] = (100, 300),
        min_train: int = 1500,
        retrain_interval: int = 600,
        full_refit_interval: int = 1200,
        cooldown_bins: int = 30,
        alpha: float = 1e-4,
    ) -> None:
        self.slope_windows: tuple[int, ...] = slope_windows
        self.min_train: int = min_train
        self.retrain_interval: int = retrain_interval
        self.full_refit_interval: int = full_refit_interval
        self.cooldown_bins: int = cooldown_bins
        self.alpha: float = alpha

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "lsvm_der"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return ["v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull"]

    def default_thresholds(self) -> list[float]:
        """Default confidence thresholds for decision function values."""
        return [0.0, 0.3, 0.5, 0.7, 1.0]

    @property
    def prediction_mode(self) -> str:
        """Predictions are signed decision function values."""
        return "confidence"

    def _build_features(
        self,
        grids: dict[str, np.ndarray],
        n_bins: int,
    ) -> np.ndarray:
        """Build feature matrix (n_bins, 60).

        Args:
            grids: Dict mapping column name to (n_bins, 101) arrays.
            n_bins: Number of bins.

        Returns:
            (n_bins, 60) feature matrix.
        """
        features: list[np.ndarray] = []

        # 1. Band asymmetries (18 features)
        band_asyms: list[np.ndarray] = []
        for col in _DERIV_COLS:
            g: np.ndarray = grids[col]
            for band in DEFAULT_BAND_DEFS:
                bid_cols: list[int] = band["bid_cols"]  # type: ignore[assignment]
                ask_cols: list[int] = band["ask_cols"]  # type: ignore[assignment]
                if "add" in col:
                    asym = g[:, bid_cols].mean(axis=1) - g[:, ask_cols].mean(axis=1)
                else:
                    asym = g[:, ask_cols].mean(axis=1) - g[:, bid_cols].mean(axis=1)
                features.append(asym.reshape(-1, 1))
                band_asyms.append(asym)

        # 2. Rolling OLS slopes of band asymmetries (18 x 2 = 36 features)
        for asym in band_asyms:
            for w in self.slope_windows:
                slope: np.ndarray = rolling_ols_slope(asym, w)
                slope = np.nan_to_num(slope, nan=0.0)
                features.append(slope.reshape(-1, 1))

        # 3. Full-width distance-weighted divergences (6 features)
        bid_weights: np.ndarray = 1.0 / (50 - _BID_COLS_FULL).astype(np.float64)
        bid_weights /= bid_weights.sum()
        ask_weights: np.ndarray = 1.0 / (_ASK_COLS_FULL - 50).astype(np.float64)
        ask_weights /= ask_weights.sum()

        for col in _DERIV_COLS:
            g = grids[col]
            bid_wm: np.ndarray = g[:, _BID_COLS_FULL] @ bid_weights
            ask_wm: np.ndarray = g[:, _ASK_COLS_FULL] @ ask_weights
            if "add" in col:
                div = bid_wm - ask_wm
            else:
                div = ask_wm - bid_wm
            features.append(div.reshape(-1, 1))

        X: np.ndarray = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward SGD prediction across the dataset.

        Uses warm_start=True for continuity.  Full refits create a
        fresh SGDClassifier; partial fits call model.partial_fit on
        recent data between full refits.

        Args:
            dataset: Dict with standard keys plus all required_columns.

        Returns:
            MLSignalResult with decision function values and validity mask.
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        grids: dict[str, np.ndarray] = {
            c: dataset[c] for c in self.required_columns
        }

        # Build features
        X: np.ndarray = self._build_features(grids, n_bins)
        logger.info("LSVM_DER: feature matrix shape=%s", X.shape)

        # Compute labels
        labels: np.ndarray = compute_labels(mid_price, n_bins)

        # Walk-forward prediction
        decisions: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)

        model: SGDClassifier | None = None
        scaler: StandardScaler | None = None
        last_train_bin: int = -1
        full_refit_count: int = 0
        partial_fit_count: int = 0

        start_bin: int = self.min_train

        for i in range(start_bin, n_bins):
            need_retrain: bool = (
                model is None or (i - last_train_bin) >= self.retrain_interval
            )

            if need_retrain:
                train_mask: np.ndarray = labels[:i] != 0
                if train_mask.sum() < 30:
                    continue

                X_train: np.ndarray = X[:i][train_mask]
                y_train: np.ndarray = labels[:i][train_mask]

                scaler = StandardScaler()
                X_train_s: np.ndarray = scaler.fit_transform(X_train)

                # Decide: full refit or partial update
                do_full_refit: bool = (
                    model is None
                    or (i - last_train_bin) >= self.full_refit_interval
                )

                if do_full_refit:
                    model = SGDClassifier(
                        loss="hinge",
                        alpha=self.alpha,
                        max_iter=1000,
                        class_weight="balanced",
                        warm_start=True,
                        random_state=42,
                    )
                    model.fit(X_train_s, y_train)
                    full_refit_count += 1
                else:
                    # Partial fit on recent data only
                    recent_mask: np.ndarray = np.zeros(i, dtype=bool)
                    recent_start: int = max(0, i - self.retrain_interval)
                    recent_mask[recent_start:i] = True
                    recent_mask &= labels[:i] != 0
                    if recent_mask.sum() > 5:
                        X_recent: np.ndarray = scaler.transform(
                            X[:i][recent_mask]
                        )
                        y_recent: np.ndarray = labels[:i][recent_mask]
                        model.partial_fit(X_recent, y_recent)
                        partial_fit_count += 1

                last_train_bin = i

            # Predict at current bin
            x_i: np.ndarray = scaler.transform(X[i : i + 1])  # type: ignore[union-attr]
            dec: float = float(model.decision_function(x_i)[0])  # type: ignore[union-attr]
            decisions[i] = dec
            has_prediction[i] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info(
            "LSVM_DER: %d predictions, %d full refits, %d partial fits",
            n_predicted,
            full_refit_count,
            partial_fit_count,
        )

        return MLSignalResult(
            predictions=decisions,
            has_prediction=has_prediction,
            metadata={
                "n_features": X.shape[1],
                "full_refit_count": full_refit_count,
                "partial_fit_count": partial_fit_count,
                "n_predicted": n_predicted,
            },
        )
