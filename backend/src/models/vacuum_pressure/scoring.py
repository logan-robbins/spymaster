"""Single-source spectrum scoring module enforcing the train/serve invariant.

This module is the SINGLE source of truth for all spectrum scoring computation.
It provides two APIs -- incremental (SpectrumScorer.update) and batch
(score_dataset) -- backed by identical internal logic.  The incremental API is
consumed by the server's stream_pipeline.py per bin.  The batch API is consumed
by the harness runner on full datasets.  Both produce identical results because
the batch path instantiates a SpectrumScorer and replays bins in order.

Scoring pipeline (per bin):
    1. Append d1/d2/d3 to independent ring buffers.
    2. Compute MAD-based robust z-score for each derivative.
    3. Weighted tanh blend -> score in [-1, 1].
    4. Discretize score -> state_code in {-1, 0, 1}.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...shared.zscore import (
    robust_z_current_vectorized,
    validate_non_negative_weight_vector,
    sanitize_unit_interval_array,
    validate_zscore_tanh_params,
    weighted_tanh_blend,
)

if TYPE_CHECKING:
    from ...qmachina.serving_config import ScoringConfig

logger: logging.Logger = logging.getLogger(__name__)

# Minimum scale threshold below which z-score is clamped to zero.
_SCALE_EPS: float = 1e-9


class SpectrumScorer:
    """Vectorized per-cell derivative scoring with ring-buffered z-scores.

    Each of the three derivative channels (d1, d2, d3) maintains an
    independent ring buffer of length ``zscore_window_bins``.  The robust
    z-score uses the median absolute deviation (MAD) estimator, converted
    to sigma-equivalent via the 1.4826 constant.

    All numpy operations are vectorized across cells.  No cross-cell
    coupling exists in the scoring phase.

    Args:
        config: ScoringConfig instance from serving_config.
        n_cells: Number of spatial cells (absolute ticks).

    Raises:
        ValueError: If config parameters are invalid or n_cells < 1.
    """

    __slots__ = (
        "_n_cells",
        "_weights",
        "_tanh_scale",
        "_threshold_neutral",
        "_window",
        "_min_periods",
        "_rings",
        "_write_idxs",
        "_counts",
        "_scratch",
        "_abs_scratch",
        "_z_out",
    )

    def __init__(self, config: ScoringConfig, n_cells: int) -> None:
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")

        window: int = config.zscore_window_bins
        min_periods: int = config.zscore_min_periods
        tanh_scale: float = config.tanh_scale
        threshold: float = config.threshold_neutral

        validate_zscore_tanh_params(
            zscore_window_bins=window,
            zscore_min_periods=min_periods,
            tanh_scale=tanh_scale,
            threshold_neutral=threshold,
        )
        weights = validate_non_negative_weight_vector(
            config.derivative_weights,
            expected_size=3,
            field_name="derivative_weights",
        )

        self._n_cells: int = n_cells
        self._weights: np.ndarray = weights
        self._tanh_scale: float = tanh_scale
        self._threshold_neutral: float = threshold
        self._window: int = window
        self._min_periods: int = min_periods

        # Ring buffers for d1, d2, d3: shape (window, n_cells).
        self._rings: list[np.ndarray] = [
            np.zeros((window, n_cells), dtype=np.float64) for _ in range(3)
        ]
        self._write_idxs: list[int] = [0, 0, 0]
        self._counts: list[int] = [0, 0, 0]

        # Pre-allocated scratch arrays for z-score computation.
        self._scratch: np.ndarray = np.empty((window, n_cells), dtype=np.float64)
        self._abs_scratch: np.ndarray = np.empty((window, n_cells), dtype=np.float64)
        self._z_out: list[np.ndarray] = [
            np.zeros(n_cells, dtype=np.float64) for _ in range(3)
        ]

    @property
    def n_cells(self) -> int:
        """Number of spatial cells this scorer operates on."""
        return self._n_cells

    @property
    def sample_counts(self) -> tuple[int, int, int]:
        """Current ring buffer fill counts for (d1, d2, d3)."""
        return (self._counts[0], self._counts[1], self._counts[2])

    def update(
        self,
        d1: np.ndarray,
        d2: np.ndarray,
        d3: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin incremental scoring update.

        Appends the derivative vectors to their ring buffers, computes
        robust z-scores, blends via weighted tanh, and discretizes.

        Args:
            d1: First derivative vector, shape (n_cells,).
            d2: Second derivative vector, shape (n_cells,).
            d3: Third derivative vector, shape (n_cells,).

        Returns:
            Tuple of (flow_score, flow_state_code).
            flow_score: float64 array in [-1, 1], shape (n_cells,).
            flow_state_code: int8 array in {-1, 0, 1}, shape (n_cells,).

        Raises:
            ValueError: If any input array has the wrong shape.
        """
        n: int = self._n_cells
        derivatives: tuple[np.ndarray, ...] = (d1, d2, d3)
        labels: tuple[str, ...] = ("d1", "d2", "d3")
        for arr, label in zip(derivatives, labels):
            if arr.shape != (n,):
                raise ValueError(
                    f"{label} shape must be ({n},), got {arr.shape}"
                )

        # Step 1: Append to ring buffers and compute robust z-scores.
        for ch in range(3):
            ring: np.ndarray = self._rings[ch]
            idx: int = self._write_idxs[ch]
            ring[idx] = derivatives[ch]
            self._write_idxs[ch] = (idx + 1) % self._window
            if self._counts[ch] < self._window:
                self._counts[ch] += 1

            self._robust_z(
                ring=ring,
                write_idx=self._write_idxs[ch],
                count=self._counts[ch],
                current=derivatives[ch],
                out=self._z_out[ch],
            )

        # Step 2: Weighted tanh blend.
        score = np.asarray(
            weighted_tanh_blend(
                self._z_out[0],
                self._z_out[1],
                self._z_out[2],
                d1_weight=float(self._weights[0]),
                d2_weight=float(self._weights[1]),
                d3_weight=float(self._weights[2]),
                tanh_scale=self._tanh_scale,
            ),
            dtype=np.float64,
        )
        sanitize_unit_interval_array(score)

        # Step 3: Discretize to state code.
        state_code: np.ndarray = np.zeros(n, dtype=np.int8)
        state_code[score >= self._threshold_neutral] = 1
        state_code[score <= -self._threshold_neutral] = -1

        return score, state_code

    def _robust_z(
        self,
        ring: np.ndarray,
        write_idx: int,
        count: int,
        current: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        """Compute MAD-based robust z-score for current values.

        Extracts the tail of the ring buffer (handling wrap-around),
        computes median and MAD, then converts to z-score using the
        1.4826 MAD-to-sigma constant.  Cells where the scale is below
        _SCALE_EPS get z=0 to avoid division by zero.

        When the ring buffer has fewer than ``min_periods`` samples,
        the output is filled with zeros (warmup period).

        Args:
            ring: Ring buffer array, shape (window, n_cells).
            write_idx: Next write position in the ring.
            count: Number of valid samples in the ring.
            current: Current derivative values, shape (n_cells,).
            out: Pre-allocated output array, shape (n_cells,).

        Returns:
            The ``out`` array filled with z-scores.
        """
        if count < self._min_periods:
            out.fill(0.0)
            return out

        # Extract the valid tail of the ring buffer into scratch.
        n_valid: int = min(self._window, count)
        start: int = (write_idx - n_valid) % self._window
        hist: np.ndarray = self._scratch[:n_valid]

        if start + n_valid <= self._window:
            hist[:] = ring[start : start + n_valid]
        else:
            first: int = self._window - start
            hist[:first] = ring[start:]
            hist[first:n_valid] = ring[: n_valid - first]

        work: np.ndarray = self._abs_scratch[:n_valid]
        return robust_z_current_vectorized(
            hist,
            current,
            scale_eps=_SCALE_EPS,
            out=out,
            work=work,
        )


def score_dataset(
    grid_df: pd.DataFrame,
    config: ScoringConfig,
    n_cells: int,
    *,
    bin_col: str = "bin_seq",
    k_col: str = "k",
) -> pd.DataFrame:
    """Add flow_score and flow_state_code columns to a grid DataFrame.

    Internally instantiates a SpectrumScorer and calls ``update()`` per bin
    in chronological order.  This guarantees identical ring-buffer state
    evolution as the incremental API, enforcing the train/serve invariant.

    The DataFrame must contain columns: ``bin_col``, ``k_col``,
    ``composite_d1``, ``composite_d2``, ``composite_d3``.  Each bin must
    have exactly ``n_cells`` rows (one per k value).

    Args:
        grid_df: DataFrame with per-bin, per-cell derivative data.
        config: ScoringConfig controlling z-score and blend parameters.
        n_cells: Expected number of spatial cells per bin.
        bin_col: Column name for the bin sequence number.
        k_col: Column name for the cell/tick index within each bin.

    Returns:
        The input DataFrame with ``flow_score`` and
        ``flow_state_code`` columns added in-place.

    Raises:
        KeyError: If required columns are missing from grid_df.
        ValueError: If any bin does not have exactly n_cells rows.
    """
    required_cols: list[str] = [
        bin_col, k_col, "composite_d1", "composite_d2", "composite_d3",
    ]
    missing: list[str] = [c for c in required_cols if c not in grid_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    scorer = SpectrumScorer(config, n_cells)

    n_rows = len(grid_df)
    if n_rows == 0:
        grid_df["flow_score"] = np.float64(0.0)
        grid_df["flow_state_code"] = np.int8(0)
        return grid_df

    bin_vals = grid_df[bin_col].to_numpy(copy=False)
    k_vals = grid_df[k_col].to_numpy(copy=False)
    d1_vals = grid_df["composite_d1"].to_numpy(dtype=np.float64, copy=False)
    d2_vals = grid_df["composite_d2"].to_numpy(dtype=np.float64, copy=False)
    d3_vals = grid_df["composite_d3"].to_numpy(dtype=np.float64, copy=False)

    # Single global ordering by (bin_seq, k) avoids per-group sort and .loc writes.
    order = np.lexsort((k_vals, bin_vals))
    sorted_bins = bin_vals[order]
    sorted_d1 = d1_vals[order]
    sorted_d2 = d2_vals[order]
    sorted_d3 = d3_vals[order]

    split_idx = np.flatnonzero(sorted_bins[1:] != sorted_bins[:-1]) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [n_rows]))

    sorted_scores = np.empty(n_rows, dtype=np.float64)
    sorted_states = np.empty(n_rows, dtype=np.int8)

    for start, end in zip(starts, ends, strict=False):
        size = end - start
        if size != n_cells:
            bin_id = sorted_bins[start]
            raise ValueError(f"Bin {bin_id} has {size} rows, expected {n_cells}")
        score, state_code = scorer.update(
            sorted_d1[start:end],
            sorted_d2[start:end],
            sorted_d3[start:end],
        )
        sorted_scores[start:end] = score
        sorted_states[start:end] = state_code

    flow_score = np.empty(n_rows, dtype=np.float64)
    flow_state = np.empty(n_rows, dtype=np.int8)
    flow_score[order] = sorted_scores
    flow_state[order] = sorted_states

    grid_df["flow_score"] = flow_score
    grid_df["flow_state_code"] = flow_state

    logger.debug(
        "score_dataset complete: %d bins, %d cells/bin, %d total rows",
        len(starts),
        n_cells,
        n_rows,
    )
    return grid_df
