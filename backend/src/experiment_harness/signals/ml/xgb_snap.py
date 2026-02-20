"""XGBoost Snapshot (XGB_SNAP) walk-forward ML signal.

XGBoost on full spatial snapshots of pressure, vacuum, and spectrum
fields.  Lets the tree ensemble discover spatial patterns directly
from raw 101-tick profiles rather than hand-engineering features.

Feature vector per bin (163 features total):
    1. pressure_variant center window (k=-25..+25) = 51
    2. vacuum_variant center window = 51
    3. flow_score center window = 51
    4. Mid-price return rolling mean/std at [50, 200, 600] = 6
    5. Total pressure/vacuum by side (bid/ask) = 4

Output: P(up) probabilities (probability mode).
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import xgboost as xgb

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

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

_TICK_SIZE: float = 0.25

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "gamma": 0.1,
    "seed": 42,
    "nthread": 1,
    "verbosity": 0,
}


class XGBSnapSignal(MLSignal):
    """Walk-forward XGBoost on spatial field snapshots.

    Trains a binary classifier on center-window slices of pressure,
    vacuum, and spectrum grids augmented with return statistics and
    side totals.  Uses early stopping on a held-out 20% validation
    split.

    Args:
        center_window: Slice of the 101-tick axis to use as raw
            spatial features.  Default: slice(25, 76) = k=-25..+25.
        rolling_windows: Windows for return rolling mean/std.
        min_train: Minimum labeled bins before first prediction.
        retrain_interval: Bins between model refits.
        cooldown_bins: Minimum bins between signal firings (metadata).
        num_boost: Maximum boosting rounds.
        early_stop: Patience rounds for early stopping.
        xgb_params: Override XGBoost parameters.  Merged with defaults.
    """

    def __init__(
        self,
        center_window: slice = slice(25, 76),
        rolling_windows: tuple[int, ...] = (50, 200, 600),
        min_train: int = 3000,
        retrain_interval: int = 600,
        cooldown_bins: int = 30,
        num_boost: int = 150,
        early_stop: int = 15,
        xgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.center_window: slice = center_window
        self.rolling_windows: tuple[int, ...] = rolling_windows
        self.min_train: int = min_train
        self.retrain_interval: int = retrain_interval
        self.cooldown_bins: int = cooldown_bins
        self.num_boost: int = num_boost
        self.early_stop: int = early_stop
        self.xgb_params: dict[str, Any] = {**_DEFAULT_XGB_PARAMS}
        if xgb_params is not None:
            self.xgb_params.update(xgb_params)

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "xgb_snap"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs from the dataset."""
        return ["pressure_variant", "vacuum_variant", "flow_score"]

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
    ) -> np.ndarray:
        """Build feature matrix (n_bins, 163).

        Args:
            grids: Dict mapping column name to (n_bins, 101) arrays.
            mid_price: (n_bins,) mid prices.
            n_bins: Number of bins.

        Returns:
            (n_bins, 163) feature matrix.
        """
        features: list[np.ndarray] = []

        # 1. Center-window spatial profiles (51 each, 153 total)
        for col in ("pressure_variant", "vacuum_variant", "flow_score"):
            features.append(grids[col][:, self.center_window])

        # 2. Mid-price return rolling stats (6 features)
        ret: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        ret[1:] = np.diff(mid_price) / _TICK_SIZE
        for w in self.rolling_windows:
            rm, rs = rolling_mean_std(ret, w)
            features.append(rm.reshape(-1, 1))
            features.append(rs.reshape(-1, 1))

        # 3. Total pressure/vacuum by side (4 features)
        P: np.ndarray = grids["pressure_variant"]
        V: np.ndarray = grids["vacuum_variant"]
        features.append(P[:, 26:50].sum(axis=1).reshape(-1, 1))   # pressure bid
        features.append(P[:, 51:75].sum(axis=1).reshape(-1, 1))   # pressure ask
        features.append(V[:, 26:50].sum(axis=1).reshape(-1, 1))   # vacuum bid
        features.append(V[:, 51:75].sum(axis=1).reshape(-1, 1))   # vacuum ask

        X: np.ndarray = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward XGBoost prediction across the dataset.

        Args:
            dataset: Dict with standard keys plus all required_columns.

        Returns:
            MLSignalResult with P(up) probabilities and validity mask.
        """
        mid_price: np.ndarray = dataset["mid_price"]
        n_bins: int = dataset["n_bins"]
        grids: dict[str, np.ndarray] = {
            c: dataset[c] for c in self.required_columns
        }

        # Build features
        X: np.ndarray = self._build_features(grids, mid_price, n_bins)
        logger.info("XGB_SNAP: feature matrix shape=%s", X.shape)

        # Compute labels and convert to binary
        labels: np.ndarray = compute_labels(mid_price, n_bins)
        binary_labels: np.ndarray = np.zeros(n_bins, dtype=np.int32)
        binary_labels[labels == 1] = 1

        # Walk-forward prediction
        proba_up: np.ndarray = np.full(n_bins, 0.5, dtype=np.float64)
        has_prediction: np.ndarray = np.zeros(n_bins, dtype=bool)

        model: xgb.Booster | None = None
        last_train_bin: int = -1
        retrain_count: int = 0
        best_iterations: list[int] = []
        feature_importance: dict[str, float] = {}

        start_bin: int = self.min_train

        for i in range(start_bin, n_bins):
            if model is None or (i - last_train_bin) >= self.retrain_interval:
                train_mask: np.ndarray = labels[:i] != 0
                if train_mask.sum() < 50:
                    continue

                X_train: np.ndarray = X[:i][train_mask]
                y_train: np.ndarray = binary_labels[:i][train_mask]

                # 80/20 split for early stopping
                split: int = int(len(X_train) * 0.8)
                if split < 30 or (len(X_train) - split) < 10:
                    continue

                dtrain: xgb.DMatrix = xgb.DMatrix(
                    X_train[:split], label=y_train[:split]
                )
                dval: xgb.DMatrix = xgb.DMatrix(
                    X_train[split:], label=y_train[split:]
                )

                # Use callback-based early stopping (xgboost >= 2.0)
                callbacks: list[xgb.callback.EarlyStopping] = [
                    xgb.callback.EarlyStopping(
                        rounds=self.early_stop,
                        save_best=True,
                    ),
                ]

                model = xgb.train(
                    self.xgb_params,
                    dtrain,
                    num_boost_round=self.num_boost,
                    evals=[(dval, "val")],
                    callbacks=callbacks,
                    verbose_eval=False,
                )
                last_train_bin = i
                retrain_count += 1
                best_iterations.append(model.best_iteration)
                feature_importance = model.get_score(importance_type="gain")

            # Predict P(up) for current bin
            dtest: xgb.DMatrix = xgb.DMatrix(X[i : i + 1])
            prob: float = float(model.predict(dtest)[0])  # type: ignore[union-attr]
            proba_up[i] = prob
            has_prediction[i] = True

        n_predicted: int = int(has_prediction.sum())
        logger.info(
            "XGB_SNAP: %d predictions, %d retrains",
            n_predicted,
            retrain_count,
        )

        # Build top-feature metadata
        top_features: dict[str, float] = {}
        if feature_importance:
            sorted_imp = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            top_features = dict(sorted_imp)

        return MLSignalResult(
            predictions=proba_up,
            has_prediction=has_prediction,
            metadata={
                "n_features": X.shape[1],
                "retrain_count": retrain_count,
                "n_predicted": n_predicted,
                "best_iterations": best_iterations,
                "top_features_gain": top_features,
            },
        )
