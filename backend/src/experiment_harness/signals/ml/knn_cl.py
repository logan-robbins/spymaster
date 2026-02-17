"""KNN Cluster Regime (KNN_CL) walk-forward ML signal.

K-Nearest Neighbors classifier on a compact microstructure state vector.
Non-parametric: no model fitting, pure similarity matching on standardized
features with distance-weighted Euclidean metric.

Feature vector per bin (35 features total):
    1. Derivative asymmetries: 6 columns x 3 bands = 18
    2. Rolling OLS slopes of combined asymmetry at 3 windows = 3
    3. Sampled spatial PV profile at 11 ticks (every 10th) = 11
    4. Mid-price momentum (rolling mean returns) at 3 windows = 3

Output: signed confidence = P(up) - P(down) (confidence mode).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

_TICK_SIZE: float = 0.25

_DERIV_COLS: tuple[str, ...] = (
    "v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
)


class KNNCLSignal(MLSignal):
    """Walk-forward KNN classifier on microstructure state vectors.

    Uses distance-weighted Euclidean KNN to classify the current
    microstructure state by voting from the K most similar historical
    states.  Re-standardizes features periodically.

    The original experiment swept k_neighbors=[5, 11, 21, 31] and
    found k=11 performed best.

    Args:
        k_neighbors: Number of nearest neighbors for voting.
        spatial_sample_cols: Indices into the 101-tick grid for the
            sampled PV profile.  Default: every 10th tick (11 samples).
        min_train: Minimum labeled bins before first prediction.
        restandardize_interval: Bins between StandardScaler refits.
        cooldown_bins: Minimum bins between signal firings (metadata).
    """

    def __init__(
        self,
        k_neighbors: int = 11,
        spatial_sample_cols: list[int] | None = None,
        min_train: int = 1800,
        restandardize_interval: int = 300,
        cooldown_bins: int = 30,
    ) -> None:
        if spatial_sample_cols is None:
            spatial_sample_cols = list(range(0, 101, 10))
        self.k_neighbors: int = k_neighbors
        self.spatial_sample_cols: list[int] = spatial_sample_cols
        self.min_train: int = min_train
        self.restandardize_interval: int = restandardize_interval
        self.cooldown_bins: int = cooldown_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "knn_cl"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return [
            "v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
            "pressure_variant", "vacuum_variant",
        ]

    def default_thresholds(self) -> list[float]:
        """Default confidence margin thresholds."""
        return [0.0, 0.1, 0.2, 0.3]

    @property
    def prediction_mode(self) -> str:
        """Predictions are signed confidence (P(up) - P(down))."""
        return "confidence"

    def _build_features(
        self,
        grids: dict[str, np.ndarray],
        mid_price: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """Build compact state vector (n_bins, 35).

        Args:
            grids: Dict mapping column name to (n_bins, 101) arrays.
            mid_price: (n_bins,) mid prices.
            n_bins: Number of bins.

        Returns:
            (n_bins, 35) feature matrix.
        """
        features: list[np.ndarray] = []

        # 1. Derivative asymmetries (18 features)
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

        # 2. Rolling slopes of combined asymmetry (3 features)
        combined: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        for col in ("v_add", "v_pull"):
            g = grids[col]
            for band in DEFAULT_BAND_DEFS:
                bid_cols = band["bid_cols"]  # type: ignore[assignment]
                ask_cols = band["ask_cols"]  # type: ignore[assignment]
                if "add" in col:
                    combined += g[:, bid_cols].mean(axis=1) - g[:, ask_cols].mean(axis=1)
                else:
                    combined += g[:, ask_cols].mean(axis=1) - g[:, bid_cols].mean(axis=1)

        for w in (10, 50, 200):
            slope: np.ndarray = rolling_ols_slope(combined, w)
            slope = np.nan_to_num(slope, nan=0.0)
            features.append(slope.reshape(-1, 1))

        # 3. Sampled spatial PV profile (11 features)
        pv: np.ndarray = grids["pressure_variant"] - grids["vacuum_variant"]
        features.append(pv[:, self.spatial_sample_cols])

        # 4. Mid-price momentum features (3 features)
        ret: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        ret[1:] = np.diff(mid_price) / _TICK_SIZE
        for w in (20, 100, 600):
            rm: np.ndarray = np.convolve(ret, np.ones(w) / w, mode="full")[:n_bins]
            features.append(rm.reshape(-1, 1))

        X: np.ndarray = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward KNN prediction across the dataset.

        Args:
            dataset: Dict with standard keys plus all required_columns.

        Returns:
            MLSignalResult with signed confidence values and validity mask.
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        grids: dict[str, np.ndarray] = {
            c: dataset[c] for c in self.required_columns
        }

        # Build features
        X: np.ndarray = self._build_features(grids, mid_price, n_bins)
        logger.info("KNN_CL: feature matrix shape=%s", X.shape)

        # Compute labels
        labels: np.ndarray = compute_labels(mid_price, n_bins)

        # Walk-forward prediction
        predictions: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)

        model: KNeighborsClassifier | None = None
        scaler: StandardScaler | None = None
        last_fit_bin: int = -1
        refit_count: int = 0

        start_bin: int = self.min_train

        for i in range(start_bin, n_bins):
            if model is None or (i - last_fit_bin) >= self.restandardize_interval:
                train_mask: np.ndarray = labels[:i] != 0
                n_labeled: int = int(train_mask.sum())
                if n_labeled < self.k_neighbors + 10:
                    continue

                X_train: np.ndarray = X[:i][train_mask]
                y_train: np.ndarray = labels[:i][train_mask]

                scaler = StandardScaler()
                X_train_s: np.ndarray = scaler.fit_transform(X_train)

                effective_k: int = min(self.k_neighbors, len(X_train))
                model = KNeighborsClassifier(
                    n_neighbors=effective_k,
                    weights="distance",
                    metric="euclidean",
                    n_jobs=1,
                )
                model.fit(X_train_s, y_train)
                last_fit_bin = i
                refit_count += 1

            # Predict at current bin
            x_i: np.ndarray = scaler.transform(X[i : i + 1])  # type: ignore[union-attr]
            proba: np.ndarray = model.predict_proba(x_i)[0]  # type: ignore[union-attr]
            classes: np.ndarray = model.classes_  # type: ignore[union-attr]

            # Extract P(+1) and P(-1)
            p_up: float = float(proba[classes == 1][0]) if 1 in classes else 0.0
            p_down: float = float(proba[classes == -1][0]) if -1 in classes else 0.0

            # Signed confidence: positive = bullish, negative = bearish
            predictions[i] = p_up - p_down
            has_prediction[i] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info(
            "KNN_CL: %d predictions, %d refits, k=%d",
            n_predicted,
            refit_count,
            self.k_neighbors,
        )

        return MLSignalResult(
            predictions=predictions,
            has_prediction=has_prediction,
            metadata={
                "n_features": X.shape[1],
                "k_neighbors": self.k_neighbors,
                "refit_count": refit_count,
                "n_predicted": n_predicted,
            },
        )
