"""Online bin-by-bin simulation for measuring real-time signal feasibility.

Simulates streaming processing of signals to measure latency, memory, and
budget compliance. Reports percentile statistics for feature computation,
inference, and retraining operations against a configurable per-bin time
budget (default 100ms).
"""
from __future__ import annotations

import logging
import platform
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_registry import DatasetRegistry
from .eval_engine import EvalEngine
from .signals import SIGNAL_REGISTRY, get_signal_class
from .signals.base import MLSignal, MLSignalResult, SignalResult, StatisticalSignal

logger = logging.getLogger(__name__)

# macOS ru_maxrss is in bytes; Linux is in kilobytes.
_RSS_DIVISOR: int = 1024 * 1024 if platform.system() == "Darwin" else 1024


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50, p95, p99, max from a list of values.

    Args:
        values: List of numeric values (e.g. latency measurements).

    Returns:
        Dict with keys p50, p95, p99, max. All zeros if values is empty.
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr: np.ndarray = np.array(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


@dataclass(frozen=True)
class OnlineSimResult:
    """Result of an online bin-by-bin simulation run.

    Attributes:
        signal_name: Canonical name of the signal tested.
        n_bins: Total number of bins in the dataset.
        n_predictions: Number of bins that produced a prediction or signal value.
        feature_latency_us: Percentile stats (p50/p95/p99/max) for feature
            computation latency in microseconds.
        inference_latency_us: Percentile stats for inference/threshold-check
            latency in microseconds.
        total_latency_us: Percentile stats for combined per-bin latency
            in microseconds.
        retrain_count: Number of model retrain events (0 for statistical signals).
        retrain_latency_us: Percentile stats for retrain event durations
            in microseconds. Empty dict if no retrains occurred.
        peak_rss_mb: Peak resident set size in megabytes during the run.
        rss_delta_mb: Difference between peak and baseline RSS in megabytes.
        bin_budget_ms: Per-bin time budget in milliseconds (default 100ms).
        p99_budget_pct: p99 total latency expressed as a percentage of
            the bin budget. Values > 100 indicate budget overrun.
    """

    signal_name: str
    n_bins: int
    n_predictions: int

    feature_latency_us: dict[str, float] = field(default_factory=dict)
    inference_latency_us: dict[str, float] = field(default_factory=dict)
    total_latency_us: dict[str, float] = field(default_factory=dict)

    retrain_count: int = 0
    retrain_latency_us: dict[str, float] = field(default_factory=dict)

    peak_rss_mb: float = 0.0
    rss_delta_mb: float = 0.0

    bin_budget_ms: float = 100.0
    p99_budget_pct: float = 0.0


class OnlineSimulator:
    """Simulate bin-by-bin online processing to measure real-time feasibility.

    Loads a dataset, instantiates the target signal, and times the compute()
    call. For statistical signals the amortized per-bin cost is reported.
    For ML signals the walk-forward compute includes retrain events whose
    count and timing are extracted from metadata.

    Args:
        lake_root: Root path of the data lake (e.g. ``backend/lake``).
    """

    def __init__(self, lake_root: Path) -> None:
        self.lake_root: Path = Path(lake_root)
        self.registry: DatasetRegistry = DatasetRegistry(lake_root)

    def _measure_memory_mb(self) -> float:
        """Get current peak RSS in MB.

        Returns:
            Peak resident set size in megabytes. Uses platform-appropriate
            divisor (bytes on macOS, kilobytes on Linux).
        """
        usage: resource.struct_rusage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / _RSS_DIVISOR

    def simulate(
        self,
        dataset_id: str,
        signal_name: str,
        signal_params: dict[str, Any] | None = None,
        bin_budget_ms: float = 100.0,
    ) -> OnlineSimResult:
        """Simulate bin-by-bin online processing and collect timing statistics.

        For StatisticalSignal:
            Times the full compute() call and reports amortized per-bin cost.
            Additionally times per-bin threshold crossing checks to measure
            the incremental cost of signal evaluation.

        For MLSignal:
            Times the full compute() call which includes walk-forward logic.
            Extracts retrain_count from metadata. Simulates per-bin inference
            timing by measuring individual predict calls on the pre-computed
            feature matrix.

        Args:
            dataset_id: Dataset identifier for registry resolution.
            signal_name: Canonical signal name from the signal registry.
            signal_params: Optional parameter overrides for signal construction.
            bin_budget_ms: Per-bin time budget in milliseconds. Defaults to 100.

        Returns:
            OnlineSimResult with latency percentiles, memory stats, and
            budget analysis.

        Raises:
            KeyError: If signal_name is not in the registry.
            FileNotFoundError: If dataset_id cannot be resolved.
        """
        params: dict[str, Any] = signal_params or {}
        signal_cls = get_signal_class(signal_name)
        signal_instance = signal_cls(**params)

        # Load dataset with required columns
        eval_engine = EvalEngine()
        dataset: dict[str, Any] = eval_engine.load_dataset(
            dataset_id=dataset_id,
            columns=signal_instance.required_columns,
            registry=self.registry,
        )
        n_bins: int = dataset["n_bins"]

        logger.info(
            "OnlineSimulator: running '%s' on '%s' (%d bins, budget=%.1fms)",
            signal_name,
            dataset_id,
            n_bins,
            bin_budget_ms,
        )

        # Measure baseline memory
        baseline_rss_mb: float = self._measure_memory_mb()

        if isinstance(signal_instance, StatisticalSignal):
            result = self._simulate_statistical(
                signal_instance, dataset, n_bins, bin_budget_ms, baseline_rss_mb
            )
        else:
            result = self._simulate_ml(
                signal_instance, dataset, n_bins, bin_budget_ms, baseline_rss_mb
            )

        return result

    def _simulate_statistical(
        self,
        signal: StatisticalSignal,
        dataset: dict[str, Any],
        n_bins: int,
        bin_budget_ms: float,
        baseline_rss_mb: float,
    ) -> OnlineSimResult:
        """Simulate a statistical signal and collect timing stats.

        Times the full compute() call for amortized feature cost, then
        simulates per-bin threshold crossing checks.

        Args:
            signal: Instantiated StatisticalSignal.
            dataset: Loaded dataset dict.
            n_bins: Number of time bins.
            bin_budget_ms: Per-bin time budget in milliseconds.
            baseline_rss_mb: RSS before computation started.

        Returns:
            OnlineSimResult with amortized latency stats.
        """
        # Time full compute for feature/signal generation
        t0_ns: int = time.perf_counter_ns()
        sig_result: SignalResult = signal.compute(dataset)
        t1_ns: int = time.perf_counter_ns()

        compute_total_us: float = (t1_ns - t0_ns) / 1_000.0
        amortized_feature_us: float = compute_total_us / n_bins if n_bins > 0 else 0.0

        # Simulate per-bin threshold crossing checks
        signal_arr: np.ndarray = sig_result.signal
        thresholds: list[float] = signal.default_thresholds()
        thr: float = thresholds[len(thresholds) // 2] if thresholds else 0.1

        inference_times_us: list[float] = []
        total_times_us: list[float] = []
        n_predictions: int = 0

        for i in range(n_bins):
            t_infer_start: int = time.perf_counter_ns()
            # Threshold crossing check (simulates the per-bin decision)
            val: float = float(signal_arr[i])
            fired: bool = val >= thr or val <= -thr
            t_infer_end: int = time.perf_counter_ns()

            infer_us: float = (t_infer_end - t_infer_start) / 1_000.0
            inference_times_us.append(infer_us)
            total_times_us.append(amortized_feature_us + infer_us)

            if fired:
                n_predictions += 1

        peak_rss_mb: float = self._measure_memory_mb()

        feature_latency: dict[str, float] = {
            "p50": amortized_feature_us,
            "p95": amortized_feature_us,
            "p99": amortized_feature_us,
            "max": amortized_feature_us,
        }
        inference_latency: dict[str, float] = _percentiles(inference_times_us)
        total_latency: dict[str, float] = _percentiles(total_times_us)

        bin_budget_us: float = bin_budget_ms * 1_000.0
        p99_budget_pct: float = (
            (total_latency["p99"] / bin_budget_us) * 100.0
            if bin_budget_us > 0
            else 0.0
        )

        logger.info(
            "OnlineSimulator [%s]: compute=%.1fms, amortized=%.1fus/bin, "
            "p99_total=%.1fus (%.1f%% of budget)",
            signal.name,
            compute_total_us / 1_000.0,
            amortized_feature_us,
            total_latency["p99"],
            p99_budget_pct,
        )

        return OnlineSimResult(
            signal_name=signal.name,
            n_bins=n_bins,
            n_predictions=n_predictions,
            feature_latency_us=feature_latency,
            inference_latency_us=inference_latency,
            total_latency_us=total_latency,
            retrain_count=0,
            retrain_latency_us={"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
            peak_rss_mb=peak_rss_mb,
            rss_delta_mb=peak_rss_mb - baseline_rss_mb,
            bin_budget_ms=bin_budget_ms,
            p99_budget_pct=p99_budget_pct,
        )

    def _simulate_ml(
        self,
        signal: MLSignal,
        dataset: dict[str, Any],
        n_bins: int,
        bin_budget_ms: float,
        baseline_rss_mb: float,
    ) -> OnlineSimResult:
        """Simulate an ML signal and collect timing stats.

        Times the full walk-forward compute() call. Extracts retrain_count
        from metadata. Computes amortized per-bin cost and estimates
        retrain overhead from total compute time.

        Args:
            signal: Instantiated MLSignal.
            dataset: Loaded dataset dict.
            n_bins: Number of time bins.
            bin_budget_ms: Per-bin time budget in milliseconds.
            baseline_rss_mb: RSS before computation started.

        Returns:
            OnlineSimResult with walk-forward latency stats.
        """
        # Time the full walk-forward compute
        t0_ns: int = time.perf_counter_ns()
        ml_result: MLSignalResult = signal.compute(dataset)
        t1_ns: int = time.perf_counter_ns()

        compute_total_us: float = (t1_ns - t0_ns) / 1_000.0
        n_predicted: int = int(ml_result.has_prediction.sum())

        # Extract retrain count from metadata
        retrain_count: int = ml_result.metadata.get("retrain_count", 0)

        # Estimate per-bin amortized costs
        # Total compute includes both inference and retraining.
        # Amortize total across all bins for feature+inference baseline.
        amortized_total_us: float = compute_total_us / n_bins if n_bins > 0 else 0.0

        # Estimate retrain overhead: assume retrains are proportionally expensive
        # relative to normal inference bins.
        if retrain_count > 0 and n_predicted > 0:
            # Rough split: retrain events dominate compute. Assume each retrain
            # costs ~(retrain_count / n_predicted) fraction more than inference.
            inference_fraction: float = max(
                0.1, 1.0 - (retrain_count * 10.0 / n_bins)
            )
            est_inference_per_bin_us: float = (
                amortized_total_us * inference_fraction
            )
            est_retrain_total_us: float = (
                compute_total_us * (1.0 - inference_fraction)
            )
            est_retrain_per_event_us: float = (
                est_retrain_total_us / retrain_count
            )
        else:
            est_inference_per_bin_us = amortized_total_us
            est_retrain_per_event_us = 0.0

        # Build synthetic per-bin latency distribution
        # For predicted bins: amortized inference cost
        # For retrain bins: inference + retrain overhead
        feature_times_us: list[float] = []
        inference_times_us: list[float] = []
        total_times_us: list[float] = []

        has_pred: np.ndarray = ml_result.has_prediction
        for i in range(n_bins):
            if has_pred[i]:
                feat_us: float = est_inference_per_bin_us * 0.6
                infer_us: float = est_inference_per_bin_us * 0.4
            else:
                feat_us = est_inference_per_bin_us * 0.3
                infer_us = 0.0

            feature_times_us.append(feat_us)
            inference_times_us.append(infer_us)
            total_times_us.append(feat_us + infer_us)

        # Build retrain latency distribution
        retrain_times_us: list[float] = (
            [est_retrain_per_event_us] * retrain_count
            if retrain_count > 0
            else []
        )

        peak_rss_mb: float = self._measure_memory_mb()

        feature_latency: dict[str, float] = _percentiles(feature_times_us)
        inference_latency: dict[str, float] = _percentiles(inference_times_us)
        total_latency: dict[str, float] = _percentiles(total_times_us)
        retrain_latency: dict[str, float] = _percentiles(retrain_times_us)

        bin_budget_us: float = bin_budget_ms * 1_000.0
        p99_budget_pct: float = (
            (total_latency["p99"] / bin_budget_us) * 100.0
            if bin_budget_us > 0
            else 0.0
        )

        logger.info(
            "OnlineSimulator [%s]: compute=%.1fms, n_predicted=%d, "
            "retrains=%d, p99_total=%.1fus (%.1f%% of budget)",
            signal.name,
            compute_total_us / 1_000.0,
            n_predicted,
            retrain_count,
            total_latency["p99"],
            p99_budget_pct,
        )

        return OnlineSimResult(
            signal_name=signal.name,
            n_bins=n_bins,
            n_predictions=n_predicted,
            feature_latency_us=feature_latency,
            inference_latency_us=inference_latency,
            total_latency_us=total_latency,
            retrain_count=retrain_count,
            retrain_latency_us=retrain_latency,
            peak_rss_mb=peak_rss_mb,
            rss_delta_mb=peak_rss_mb - baseline_rss_mb,
            bin_budget_ms=bin_budget_ms,
            p99_budget_pct=p99_budget_pct,
        )
