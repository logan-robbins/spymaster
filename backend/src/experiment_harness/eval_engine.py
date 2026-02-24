"""Evaluation engine for VP regime-detection experiments.

Extends the logic from ``eval_harness.py`` into a configurable, reusable class
with module-level utility functions. Supports:

- Dataset loading via ``DatasetRegistry``
- Directional signal detection with threshold crossing and cooldown
- TP/SL outcome evaluation with configurable parameters
- Threshold sweep across signal values
- ML prediction evaluation (probability and confidence modes)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.shared.zscore import robust_zscore_rolling_1d
from src.models.vacuum_pressure.scoring import score_dataset
from src.qmachina.serving_config import ScoringConfig

from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)

# Grid constants: k ranges from -50 to +50 inclusive = 101 ticks
K_MIN: int = -50
K_MAX: int = 50
N_TICKS: int = 101
_DERIVABLE_FLOW_COLUMNS: frozenset[str] = frozenset({"flow_score", "flow_state_code"})
_FLOW_SUPPORT_COLUMNS: tuple[str, ...] = ("composite_d1", "composite_d2", "composite_d3")


def _resolve_flow_scoring_config(dataset_root: Path) -> ScoringConfig:
    """Resolve flow-scoring params from dataset manifest, falling back to defaults."""
    manifest_path = dataset_root / "manifest.json"
    defaults = ScoringConfig()
    if not manifest_path.exists():
        return defaults

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to parse manifest for scoring config: {manifest_path}"
        ) from exc

    overrides: dict[str, Any] = {}
    flow_scoring = payload.get("flow_scoring")
    if isinstance(flow_scoring, dict):
        if "derivative_weights" in flow_scoring:
            overrides["derivative_weights"] = flow_scoring["derivative_weights"]
        if "tanh_scale" in flow_scoring:
            overrides["tanh_scale"] = flow_scoring["tanh_scale"]
        if "threshold_neutral" in flow_scoring:
            overrides["threshold_neutral"] = flow_scoring["threshold_neutral"]
        if "zscore_window_bins" in flow_scoring:
            overrides["zscore_window_bins"] = flow_scoring["zscore_window_bins"]
        if "zscore_min_periods" in flow_scoring:
            overrides["zscore_min_periods"] = flow_scoring["zscore_min_periods"]
    else:
        legacy_params = payload.get("grid_dependent_params")
        if isinstance(legacy_params, dict):
            if legacy_params.get("flow_derivative_weights") is not None:
                overrides["derivative_weights"] = legacy_params["flow_derivative_weights"]
            if legacy_params.get("flow_tanh_scale") is not None:
                overrides["tanh_scale"] = legacy_params["flow_tanh_scale"]

    if not overrides:
        return defaults

    merged = defaults.model_dump()
    merged.update(overrides)
    try:
        return ScoringConfig.model_validate(merged)
    except Exception as exc:
        raise ValueError(
            f"Invalid flow_scoring payload in manifest: {manifest_path}"
        ) from exc


def rolling_ols_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling OLS slope of a 1-D array.

    Uses the closed-form formula:
        slope = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
    where x = [0, 1, ..., w-1] and y = arr values in the window.

    Args:
        arr: Input 1-D array of float64 values.
        window: Rolling window size (must be >= 2).

    Returns:
        Array of same length with NaN for the first ``window - 1`` elements.
    """
    n: int = len(arr)
    result: np.ndarray = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 2:
        return result

    x: np.ndarray = np.arange(window, dtype=np.float64)
    sum_x: float = float(x.sum())
    sum_x2: float = float((x * x).sum())
    denom: float = window * sum_x2 - sum_x * sum_x

    if abs(denom) < 1e-30:
        return result

    # Initialize first full window.
    first_window = arr[:window]
    sum_y: float = float(first_window.sum())
    sum_xy: float = float(np.dot(x, first_window))
    result[window - 1] = (window * sum_xy - sum_x * sum_y) / denom

    # O(1) sliding updates:
    # T_new = T_old - S_old + y_old + (w-1) * y_new
    # S_new = S_old - y_old + y_new
    for i in range(window, n):
        y_old = float(arr[i - window])
        y_new = float(arr[i])
        sum_xy = sum_xy - sum_y + y_old + (window - 1) * y_new
        sum_y = sum_y - y_old + y_new
        result[i] = (window * sum_xy - sum_x * sum_y) / denom

    return result


def robust_zscore(
    arr: np.ndarray, window: int, min_periods: int = 30
) -> np.ndarray:
    """Robust z-score using rolling median and MAD (median absolute deviation).

    Computes z = (x - median) / (1.4826 * MAD) over a rolling window.
    The constant 1.4826 makes MAD a consistent estimator of standard deviation
    for normally distributed data.

    Args:
        arr: Input 1-D array of float64 values.
        window: Rolling window size.
        min_periods: Minimum number of observations required before computing
            a non-zero z-score. Defaults to 30.

    Returns:
        Array of same length. Returns 0.0 for bins with fewer than
        ``min_periods`` observations or near-zero MAD.
    """
    return robust_zscore_rolling_1d(
        arr,
        window=window,
        min_periods=min_periods,
        scale_eps=1e-12,
    )


class EvalEngine:
    """Configurable evaluation engine for TP/SL signal experiments.

    Encapsulates all evaluation parameters and provides methods for loading
    datasets, detecting signals, evaluating outcomes, and sweeping thresholds.

    Args:
        tp_ticks: Take-profit distance in ticks.
        sl_ticks: Stop-loss distance in ticks.
        max_hold_bins: Maximum holding period in time bins before timeout.
        warmup_bins: Number of initial bins to skip (allows rolling windows to fill).
        tick_size: Dollar value per tick (e.g. 0.25 for MNQ).
    """

    def __init__(
        self,
        tp_ticks: int = 8,
        sl_ticks: int = 4,
        max_hold_bins: int = 1200,
        warmup_bins: int = 300,
        tick_size: float = 0.25,
    ) -> None:
        self.tp_ticks: int = tp_ticks
        self.sl_ticks: int = sl_ticks
        self.max_hold_bins: int = max_hold_bins
        self.warmup_bins: int = warmup_bins
        self.tick_size: float = tick_size

    def load_dataset(
        self,
        dataset_id: str,
        columns: list[str],
        registry: DatasetRegistry,
    ) -> dict[str, Any]:
        """Load a dataset by ID using the registry and pivot grid columns.

        Reads ``bins.parquet`` for time-bin metadata and ``grid_clean.parquet``
        for grid columns, pivoting each requested column into an
        ``(n_bins, 101)`` array indexed by k = -50..+50.

        Args:
            dataset_id: Dataset identifier string.
            columns: Grid column names to load and pivot (e.g. ``["velocity"]``).
            registry: DatasetRegistry instance for path resolution.

        Returns:
            Dict containing:
                - ``bins``: DataFrame of time bins
                - ``mid_price``: float64 array of mid prices (n_bins,)
                - ``ts_ns``: int64 array of nanosecond timestamps (n_bins,)
                - ``n_bins``: number of time bins
                - ``k_values``: int32 array of k axis values (101,)
                - ``<col>``: float64 array (n_bins, 101) for each requested column

        Raises:
            FileNotFoundError: If the dataset cannot be resolved.
        """
        paths = registry.resolve(dataset_id)
        logger.info(
            "Loading dataset '%s': bins=%s, grid=%s",
            dataset_id,
            paths.bins_parquet,
            paths.grid_clean_parquet,
        )

        bins_df: pd.DataFrame = pd.read_parquet(paths.bins_parquet)
        bins_df = bins_df.sort_values("bin_seq").reset_index(drop=True)
        n_bins: int = len(bins_df)

        mid_price: np.ndarray = bins_df["mid_price"].values.astype(np.float64)
        ts_ns: np.ndarray = bins_df["ts_ns"].values.astype(np.int64)

        grid_schema_cols = set(pq.read_schema(paths.grid_clean_parquet).names)
        missing_cols = [c for c in columns if c not in grid_schema_cols]
        missing_flow_cols = [c for c in missing_cols if c in _DERIVABLE_FLOW_COLUMNS]
        missing_unsupported = [c for c in missing_cols if c not in _DERIVABLE_FLOW_COLUMNS]
        if missing_unsupported:
            raise KeyError(
                f"Dataset '{dataset_id}' missing required columns: {missing_unsupported}"
            )

        read_cols: list[str] = ["bin_seq", "k"] + [
            c for c in columns if c in grid_schema_cols
        ]
        if missing_flow_cols:
            missing_support = [c for c in _FLOW_SUPPORT_COLUMNS if c not in grid_schema_cols]
            if missing_support:
                raise KeyError(
                    f"Dataset '{dataset_id}' missing columns needed to derive flow fields: {missing_support}"
                )
            read_cols.extend(_FLOW_SUPPORT_COLUMNS)

        dedup_read_cols: list[str] = []
        seen: set[str] = set()
        for col in read_cols:
            if col in seen:
                continue
            seen.add(col)
            dedup_read_cols.append(col)

        grid_df: pd.DataFrame = pd.read_parquet(
            paths.grid_clean_parquet, columns=dedup_read_cols
        )

        if missing_flow_cols:
            scoring_cfg = _resolve_flow_scoring_config(paths.grid_clean_parquet.parent)
            score_input = grid_df[
                ["bin_seq", "k", "composite_d1", "composite_d2", "composite_d3"]
            ].copy()
            scored_df = score_dataset(score_input, scoring_cfg, N_TICKS)
            if "flow_score" in missing_flow_cols:
                grid_df["flow_score"] = scored_df["flow_score"].to_numpy(copy=False)
            if "flow_state_code" in missing_flow_cols:
                grid_df["flow_state_code"] = scored_df["flow_state_code"].to_numpy(copy=False)

        k_values: np.ndarray = np.arange(K_MIN, K_MAX + 1, dtype=np.int32)
        bin_seq_index = pd.Index(bins_df["bin_seq"].to_numpy(copy=False))

        result: dict[str, Any] = {
            "bins": bins_df,
            "mid_price": mid_price,
            "ts_ns": ts_ns,
            "n_bins": n_bins,
            "k_values": k_values,
        }

        # Pivot each column to (n_bins, 101) array
        bin_seqs: np.ndarray = grid_df["bin_seq"].to_numpy(copy=False)
        k_vals: np.ndarray = grid_df["k"].to_numpy(copy=False).astype(np.int32, copy=False)
        b_idxs = bin_seq_index.get_indexer(bin_seqs)
        k_idxs = k_vals - K_MIN
        valid_mask = (b_idxs >= 0) & (k_idxs >= 0) & (k_idxs < N_TICKS)

        for col in columns:
            arr: np.ndarray = np.zeros((n_bins, N_TICKS), dtype=np.float64)
            col_vals: np.ndarray = grid_df[col].to_numpy(dtype=np.float64, copy=False)
            arr[b_idxs[valid_mask], k_idxs[valid_mask]] = col_vals[valid_mask]

            result[col] = arr

        logger.info(
            "Loaded dataset '%s': %d bins, %d grid columns",
            dataset_id,
            n_bins,
            len(columns),
        )
        return result

    def detect_signals(
        self,
        signal: np.ndarray,
        threshold: float,
        cooldown_bins: int,
    ) -> list[dict[str, Any]]:
        """Detect directional signals from a continuous signal array.

        Fires when the signal crosses +/-threshold from a neutral or opposite
        state, respecting a cooldown period between consecutive signals.

        Args:
            signal: 1-D array of continuous signal values.
            threshold: Absolute threshold for signal firing (positive value).
            cooldown_bins: Minimum bins between consecutive signals.

        Returns:
            List of signal dicts, each with ``bin_idx`` (int) and
            ``direction`` (``"up"`` or ``"down"``).
        """
        signals: list[dict[str, Any]] = []
        prev_state: str = "flat"
        last_signal_bin: int = -cooldown_bins

        for i in range(len(signal)):
            if signal[i] >= threshold:
                cur_state: str = "up"
            elif signal[i] <= -threshold:
                cur_state = "down"
            else:
                cur_state = "flat"

            if cur_state != "flat" and cur_state != prev_state:
                if i - last_signal_bin >= cooldown_bins:
                    signals.append({"bin_idx": i, "direction": cur_state})
                    last_signal_bin = i

            prev_state = cur_state

        return signals

    def evaluate_tp_sl(
        self,
        *,
        signals: list[dict[str, Any]],
        mid_price: np.ndarray,
        ts_ns: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate TP/SL outcomes for a list of directional signals.

        For each signal, walks forward through mid_price checking whether
        take-profit or stop-loss is hit first, or the position times out.

        Args:
            signals: List of signal dicts with ``bin_idx`` and ``direction``.
            mid_price: Array of mid prices indexed by bin.
            ts_ns: Array of nanosecond timestamps indexed by bin.

        Returns:
            Dict with summary statistics:
                - ``n_signals``, ``n_up``, ``n_down``: signal counts
                - ``tp_rate``, ``sl_rate``, ``timeout_rate``: outcome fractions
                - ``events_per_hour``: signal frequency
                - ``mean_pnl_ticks``: average PnL in tick units
                - ``median_time_to_outcome_ms``: median time to TP/SL (excl. timeouts)
                - ``outcomes``: list of per-signal outcome dicts
        """
        tp_dollars: float = self.tp_ticks * self.tick_size
        sl_dollars: float = self.sl_ticks * self.tick_size

        outcomes: list[dict[str, Any]] = []

        for sig in signals:
            idx: int = sig["bin_idx"]
            direction: str = sig["direction"]
            entry_price: float = float(mid_price[idx])
            entry_ts: int = int(ts_ns[idx])

            if entry_price <= 0:
                continue

            end_idx: int = min(idx + self.max_hold_bins, len(mid_price))
            outcome: str = "timeout"
            outcome_idx: int = end_idx - 1
            outcome_price: float = (
                float(mid_price[outcome_idx])
                if outcome_idx < len(mid_price)
                else entry_price
            )

            for j in range(idx + 1, end_idx):
                price: float = float(mid_price[j])
                if price <= 0:
                    continue
                if direction == "up":
                    if price >= entry_price + tp_dollars:
                        outcome = "tp"
                        outcome_idx = j
                        outcome_price = price
                        break
                    if price <= entry_price - sl_dollars:
                        outcome = "sl"
                        outcome_idx = j
                        outcome_price = price
                        break
                else:
                    if price <= entry_price - tp_dollars:
                        outcome = "tp"
                        outcome_idx = j
                        outcome_price = price
                        break
                    if price >= entry_price + sl_dollars:
                        outcome = "sl"
                        outcome_idx = j
                        outcome_price = price
                        break

            time_to_outcome_ms: float = float(ts_ns[outcome_idx] - entry_ts) / 1e6

            pnl_direction: float = 1.0 if direction == "up" else -1.0
            outcomes.append(
                {
                    "bin_idx": idx,
                    "direction": direction,
                    "entry_price": entry_price,
                    "outcome": outcome,
                    "outcome_price": outcome_price,
                    "time_to_outcome_ms": time_to_outcome_ms,
                    "pnl_ticks": (outcome_price - entry_price)
                    / self.tick_size
                    * pnl_direction,
                }
            )

        n_total: int = len(outcomes)
        if n_total == 0:
            return {
                "n_signals": 0,
                "tp_rate": float("nan"),
                "sl_rate": float("nan"),
                "timeout_rate": float("nan"),
                "events_per_hour": 0.0,
                "mean_pnl_ticks": float("nan"),
                "median_time_to_outcome_ms": float("nan"),
                "outcomes": [],
            }

        n_tp: int = sum(1 for o in outcomes if o["outcome"] == "tp")
        n_sl: int = sum(1 for o in outcomes if o["outcome"] == "sl")
        n_timeout: int = sum(1 for o in outcomes if o["outcome"] == "timeout")

        eval_duration_hours: float = float(ts_ns[-1] - ts_ns[0]) / 3.6e12
        events_per_hour: float = (
            n_total / eval_duration_hours if eval_duration_hours > 0 else 0.0
        )

        times: list[float] = [
            o["time_to_outcome_ms"] for o in outcomes if o["outcome"] != "timeout"
        ]
        pnls: list[float] = [o["pnl_ticks"] for o in outcomes]

        return {
            "n_signals": n_total,
            "n_up": sum(1 for o in outcomes if o["direction"] == "up"),
            "n_down": sum(1 for o in outcomes if o["direction"] == "down"),
            "tp_rate": n_tp / n_total,
            "sl_rate": n_sl / n_total,
            "timeout_rate": n_timeout / n_total,
            "events_per_hour": events_per_hour,
            "median_time_to_outcome_ms": (
                float(np.median(times)) if times else float("nan")
            ),
            "mean_pnl_ticks": float(np.mean(pnls)),
            "outcomes": outcomes,
        }

    def sweep_thresholds(
        self,
        *,
        signal: np.ndarray,
        thresholds: list[float],
        cooldown_bins: int,
        mid_price: np.ndarray,
        ts_ns: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Sweep detection thresholds and evaluate TP/SL for each.

        For each threshold, detects signals (skipping the warmup window),
        then evaluates TP/SL outcomes. Returns summary stats per threshold
        without the full outcome lists.

        Args:
            signal: 1-D continuous signal array (full length, including warmup).
            thresholds: List of threshold values to sweep.
            cooldown_bins: Minimum bins between consecutive signals.
            mid_price: Array of mid prices indexed by bin.
            ts_ns: Array of nanosecond timestamps indexed by bin.

        Returns:
            List of result dicts (one per threshold), each augmented with
            ``threshold`` and ``cooldown_bins`` fields.
        """
        results: list[dict[str, Any]] = []

        for thr in thresholds:
            sigs: list[dict[str, Any]] = self.detect_signals(
                signal[self.warmup_bins :], thr, cooldown_bins
            )
            # Adjust bin_idx back to global index
            for s in sigs:
                s["bin_idx"] += self.warmup_bins

            eval_result: dict[str, Any] = self.evaluate_tp_sl(
                signals=sigs,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )
            # Remove full trade log from sweep results (too large)
            eval_result.pop("outcomes", None)
            eval_result["threshold"] = thr
            eval_result["cooldown_bins"] = cooldown_bins
            results.append(eval_result)

        return results

    def evaluate_ml_predictions(
        self,
        *,
        predictions: np.ndarray,
        has_prediction: np.ndarray,
        thresholds: list[float],
        cooldown_bins: int,
        mid_price: np.ndarray,
        ts_ns: np.ndarray,
        mode: str = "probability",
    ) -> list[dict[str, Any]]:
        """Evaluate ML model predictions with configurable threshold modes.

        Supports two modes:
        - ``probability``: fires up when P(up) >= thr, down when P(up) <= 1 - thr.
        - ``confidence``: fires when |decision_function| >= thr, direction from sign.

        Args:
            predictions: Array of prediction values (probabilities or decision scores).
            has_prediction: Boolean array indicating which bins have valid predictions.
            thresholds: List of threshold values to evaluate.
            cooldown_bins: Minimum bins between consecutive signals.
            mid_price: Array of mid prices indexed by bin.
            ts_ns: Array of nanosecond timestamps indexed by bin.
            mode: Either ``"probability"`` or ``"confidence"``.

        Returns:
            List of result dicts (one per threshold), each with summary stats
            plus ``threshold`` and ``cooldown_bins`` fields.

        Raises:
            ValueError: If mode is not ``"probability"`` or ``"confidence"``.
        """
        if mode not in ("probability", "confidence"):
            raise ValueError(f"mode must be 'probability' or 'confidence', got '{mode}'")

        results: list[dict[str, Any]] = []

        for thr in thresholds:
            sigs: list[dict[str, Any]] = []
            last_sig_bin: int = -cooldown_bins

            for i in range(self.warmup_bins, len(predictions)):
                if not has_prediction[i]:
                    continue
                if i - last_sig_bin < cooldown_bins:
                    continue

                if mode == "probability":
                    if predictions[i] >= thr:
                        sigs.append({"bin_idx": i, "direction": "up"})
                        last_sig_bin = i
                    elif predictions[i] <= (1.0 - thr):
                        sigs.append({"bin_idx": i, "direction": "down"})
                        last_sig_bin = i
                else:  # confidence
                    if abs(predictions[i]) >= thr:
                        direction: str = "up" if predictions[i] > 0 else "down"
                        sigs.append({"bin_idx": i, "direction": direction})
                        last_sig_bin = i

            eval_result: dict[str, Any] = self.evaluate_tp_sl(
                signals=sigs,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )
            eval_result.pop("outcomes", None)
            eval_result["threshold"] = thr
            eval_result["cooldown_bins"] = cooldown_bins
            results.append(eval_result)

        return results
