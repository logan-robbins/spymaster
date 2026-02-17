"""Rolling PCA Anomaly Detection (PCA_AD) walk-forward ML signal.

Uses PCA on the spatial pressure-vacuum grid to detect anomalous
microstructure configurations.  Anomalies (high reconstruction error
or extreme PC scores) often precede regime transitions.

Pipeline:
    1. Rolling PCA on the trailing window of PV difference profiles
    2. Reconstruction error (L2 norm) and Mahalanobis distance on PC scores
    3. Robust z-score both anomaly metrics (window=300)
    4. Combined anomaly score = 0.5 * max(z_recon, 0) + 0.5 * max(z_mahal, 0)
    5. Directional bias from PC1 loadings + add-pull asymmetry field
    6. Final signal = direction * anomaly_score (signed confidence)

Output: signed anomaly-gated directional values (confidence mode).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiment_harness.eval_engine import rolling_ols_slope, robust_zscore
from experiment_harness.signals.base import MLSignal, MLSignalResult
from experiment_harness.signals.features import (
    DEFAULT_BAND_DEFS,
    band_asymmetry,
    distance_weighted_sum,
    rolling_mean_std,
)
from experiment_harness.signals.labels import compute_labels

logger = logging.getLogger(__name__)


class PCAADSignal(MLSignal):
    """Walk-forward rolling PCA anomaly detection with directional gating.

    Fits PCA on a trailing window of standardized PV difference profiles,
    then detects anomalies via reconstruction error and Mahalanobis
    distance.  The anomaly score is gated by a directional signal derived
    from PC1 loadings and the add-pull asymmetry field.

    This signal does not use labels for training -- PCA is unsupervised.
    The labels module is not called.  Direction is inferred from the
    spatial structure of the anomaly.

    Args:
        n_components: Number of PCA components to retain.
        pca_window: Trailing window size for PCA fitting.
        refit_interval: Bins between PCA refits.
        min_warmup: Minimum bins before first prediction (must exceed
            pca_window to have a full fitting window).
        cooldown_bins: Minimum bins between signal firings (metadata).
    """

    def __init__(
        self,
        n_components: int = 10,
        pca_window: int = 600,
        refit_interval: int = 100,
        min_warmup: int = 700,
        cooldown_bins: int = 30,
    ) -> None:
        self.n_components: int = n_components
        self.pca_window: int = pca_window
        self.refit_interval: int = refit_interval
        self.min_warmup: int = min_warmup
        self.cooldown_bins: int = cooldown_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "pca_ad"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return ["pressure_variant", "vacuum_variant", "v_add", "v_pull"]

    def default_thresholds(self) -> list[float]:
        """Default anomaly-gated signal thresholds."""
        return [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    @property
    def prediction_mode(self) -> str:
        """Predictions are signed anomaly-gated confidence values."""
        return "confidence"

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run rolling PCA anomaly detection across the dataset.

        Args:
            dataset: Dict with standard keys plus all required_columns.

        Returns:
            MLSignalResult with signed anomaly-gated direction values and
            validity mask (True for bins >= min_warmup).
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        P: np.ndarray = dataset["pressure_variant"]
        V: np.ndarray = dataset["vacuum_variant"]
        v_add: np.ndarray = dataset["v_add"]
        v_pull: np.ndarray = dataset["v_pull"]

        # Primary spatial field
        pv_diff: np.ndarray = P - V  # (n_bins, 101)

        # Add-pull asymmetry field for directional bias
        add_pull_diff: np.ndarray = v_add - v_pull  # (n_bins, 101)

        # Rolling PCA arrays
        recon_error: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        mahal_dist: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        pc1_score: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        pc1_direction: np.ndarray = np.zeros(n_bins, dtype=np.float64)

        pca_model: PCA | None = None
        pca_scaler: StandardScaler | None = None
        last_fit: int = -1
        pc1_dir_sign: float = 1.0
        refit_count: int = 0

        for i in range(self.min_warmup, n_bins):
            # Refit PCA on trailing window
            if pca_model is None or (i - last_fit) >= self.refit_interval:
                start: int = max(0, i - self.pca_window)
                window_data: np.ndarray = pv_diff[start:i]

                pca_scaler = StandardScaler()
                window_scaled: np.ndarray = pca_scaler.fit_transform(window_data)

                n_comp: int = min(
                    self.n_components,
                    window_data.shape[0],
                    window_data.shape[1],
                )
                pca_model = PCA(n_components=n_comp)
                pca_model.fit(window_scaled)

                # Direction from PC1 loadings: bid vs ask loading mass
                pc1_loadings: np.ndarray = pca_model.components_[0]
                bid_loading: float = float(pc1_loadings[:50].sum())
                ask_loading: float = float(pc1_loadings[51:].sum())
                pc1_dir_sign = 1.0 if bid_loading > ask_loading else -1.0

                last_fit = i
                refit_count += 1

            # Project current bin
            x_scaled: np.ndarray = pca_scaler.transform(pv_diff[i : i + 1])  # type: ignore[union-attr]
            scores: np.ndarray = pca_model.transform(x_scaled)[0]  # type: ignore[union-attr]
            x_recon: np.ndarray = pca_model.inverse_transform(  # type: ignore[union-attr]
                scores.reshape(1, -1)
            )

            # Reconstruction error (L2 norm)
            err: float = float(np.sqrt(np.sum((x_scaled[0] - x_recon[0]) ** 2)))
            recon_error[i] = err

            # Mahalanobis-like distance on PC scores
            variances: np.ndarray = pca_model.explained_variance_  # type: ignore[union-attr]
            valid_var: np.ndarray = variances > 1e-10
            if valid_var.sum() > 0:
                scaled_scores: np.ndarray = (
                    scores[valid_var] ** 2 / variances[valid_var]
                )
                mahal_dist[i] = float(np.sqrt(scaled_scores.sum()))

            # PC1 score (directional component)
            pc1_score[i] = float(scores[0]) * pc1_dir_sign

            # Direction from add-pull field
            ap: np.ndarray = add_pull_diff[i]
            bid_ap: float = float(ap[:50].mean())
            ask_ap: float = float(ap[51:].mean())
            pc1_direction[i] = bid_ap - ask_ap  # positive = bullish

        logger.info(
            "PCA_AD: %d PCA refits over %d active bins",
            refit_count,
            n_bins - self.min_warmup,
        )

        # Build anomaly-gated directional signal
        z_recon: np.ndarray = robust_zscore(recon_error, window=300)
        z_mahal: np.ndarray = robust_zscore(mahal_dist, window=300)

        # Combined anomaly score (only positive z-scores contribute)
        anomaly_score: np.ndarray = (
            0.5 * np.maximum(z_recon, 0.0) + 0.5 * np.maximum(z_mahal, 0.0)
        )

        # Directional bias: combine PC1 score with add-pull asymmetry
        z_pc1: np.ndarray = robust_zscore(pc1_score, window=300)
        z_dir: np.ndarray = robust_zscore(pc1_direction, window=300)
        direction_signal: np.ndarray = (
            0.6 * np.tanh(z_pc1 / 3.0) + 0.4 * np.tanh(z_dir / 3.0)
        )

        # Final signal: direction * anomaly gate
        signal: np.ndarray = direction_signal * anomaly_score

        # Build output arrays
        predictions: np.ndarray = signal.copy()
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)
        has_prediction[self.min_warmup :] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info("PCA_AD: %d predictions", n_predicted)

        return MLSignalResult(
            predictions=predictions,
            has_prediction=has_prediction,
            metadata={
                "n_components": self.n_components,
                "pca_window": self.pca_window,
                "refit_count": refit_count,
                "n_predicted": n_predicted,
            },
        )
