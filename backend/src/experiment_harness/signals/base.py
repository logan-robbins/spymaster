"""Abstract base classes for experiment harness signal computation.

Two signal families exist:

StatisticalSignal
    Pure-arithmetic signals built from spatial grid data using rolling
    statistics, z-scores, asymmetries, and blending. No model fitting.
    Output is a continuous signal array thresholded for directional events.

MLSignal
    Walk-forward ML signals that train on historical labels and produce
    per-bin predictions (probabilities or decision-function values).
    Output includes a prediction mask indicating which bins have valid
    predictions.

Both families share a common dataset interface: a dict containing
mid_price, ts_ns, n_bins, k_values, and named (n_bins, 101) grid arrays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Output from a statistical signal computation.

    Attributes:
        signal: (n_bins,) continuous signal array. Positive values indicate
            bullish bias, negative values indicate bearish bias. Magnitude
            encodes confidence.
        metadata: Arbitrary key-value pairs for diagnostics (e.g.
            intermediate arrays, timing info, signal statistics).
    """

    signal: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MLSignalResult:
    """Output from an ML walk-forward signal computation.

    Attributes:
        predictions: (n_bins,) raw prediction values. For probability-mode
            signals this is P(up) in [0, 1]. For confidence-mode signals
            this is a signed decision function value.
        has_prediction: (n_bins,) boolean mask. True where the model
            produced a valid prediction (i.e. after sufficient training
            data accumulated).
        metadata: Arbitrary key-value pairs for diagnostics (e.g.
            feature importance, retrain count, training set sizes).
    """

    predictions: np.ndarray
    has_prediction: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class StatisticalSignal(ABC):
    """Interface for statistical (pure-arithmetic) signals.

    Implementations compute a continuous directional signal from the
    spatial grid without any model fitting. The signal is then
    thresholded externally to produce discrete trading events.

    Lifecycle:
        1. Instantiate with signal-specific parameters.
        2. Call compute() with a loaded dataset dict.
        3. Threshold the resulting signal array externally.
    """

    @abstractmethod
    def __init__(self, **params: Any) -> None:
        """Initialize with signal-specific configurable parameters.

        Args:
            **params: Keyword arguments specific to the signal
                implementation (e.g. slope windows, blend weights,
                z-score lookback).
        """

    @abstractmethod
    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute signal from a loaded dataset.

        Args:
            dataset: Dict with standard keys:
                - mid_price: (n_bins,) float64
                - ts_ns: (n_bins,) int64
                - n_bins: int
                - k_values: (101,) int32
                - <col_name>: (n_bins, 101) float64 for each grid column

        Returns:
            SignalResult with the computed signal array.
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs loaded from the dataset.

        Returns:
            List of column names (e.g. ["v_add", "v_pull"]).
        """

    @abstractmethod
    def default_thresholds(self) -> list[float]:
        """Default threshold grid for sweeping this signal.

        Returns:
            List of float thresholds to evaluate. Positive values;
            the sweep applies both +threshold and -threshold.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical signal name used for registry lookup and output files.

        Returns:
            Short lowercase identifier (e.g. "ads", "pfp", "erd").
        """


class MLSignal(ABC):
    """Interface for ML walk-forward signals.

    Implementations perform expanding-window training and one-step-ahead
    prediction across the dataset. Training uses TP/SL labels computed
    from mid_price via the shared labels module.

    Lifecycle:
        1. Instantiate with model hyperparameters.
        2. Call compute() with a loaded dataset dict.
        3. Threshold the predictions externally using the
           prediction_mode semantics.
    """

    @abstractmethod
    def __init__(self, **params: Any) -> None:
        """Initialize with model hyperparameters.

        Args:
            **params: Keyword arguments specific to the ML signal
                (e.g. min_train_bins, retrain_interval, model params).
        """

    @abstractmethod
    def compute(self, dataset: dict[str, Any]) -> MLSignalResult:
        """Run walk-forward train/predict across the dataset.

        Args:
            dataset: Dict with standard keys (see StatisticalSignal.compute).

        Returns:
            MLSignalResult with predictions and validity mask.
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Grid columns this signal needs loaded from the dataset.

        Returns:
            List of column names.
        """

    @abstractmethod
    def default_thresholds(self) -> list[float]:
        """Default threshold grid for this signal's predictions.

        For probability-mode signals: thresholds on P(up), e.g.
            [0.50, 0.55, 0.60]. Signal fires up when P(up) >= thr
            and down when P(up) <= 1-thr.

        For confidence-mode signals: thresholds on |decision_function|,
            e.g. [0.0, 0.2, 0.5]. Signal fires in the sign direction
            when |value| >= thr.

        Returns:
            List of float thresholds.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical signal name for registry lookup.

        Returns:
            Short lowercase identifier (e.g. "svm_sp", "gbm_mf").
        """

    @property
    def prediction_mode(self) -> str:
        """How to interpret predictions for thresholding.

        Returns:
            "probability" -- predictions are P(up) in [0, 1].
                Fire up when pred >= thr, down when pred <= 1-thr.
            "confidence" -- predictions are signed decision values.
                Fire up when pred > thr, down when pred < -thr.

        Default is "probability". Override in subclass if needed.
        """
        return "probability"
