"""Independent per-cell spectrum scoring and projection kernel.

All math is vectorized across cells. Each cell is scored independently from its
own time history; no cross-cell coupling is used in this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class SpectrumOutput:
    """Vectorized spectrum outputs for one emitted time bin."""

    score: np.ndarray
    state_code: np.ndarray
    projected_score_by_horizon: Dict[int, np.ndarray]


class IndependentCellSpectrum:
    """Vectorized per-cell derivative-only hybrid spectrum model."""

    def __init__(
        self,
        n_cells: int,
        windows: Sequence[int],
        rollup_weights: Sequence[float],
        derivative_weights: Sequence[float],
        tanh_scale: float,
        neutral_threshold: float,
        zscore_window_bins: int,
        zscore_min_periods: int,
        projection_horizons_ms: Sequence[int],
        default_dt_s: float,
    ) -> None:
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")
        if tanh_scale <= 0.0:
            raise ValueError(f"tanh_scale must be > 0, got {tanh_scale}")
        if not (0.0 < neutral_threshold < 1.0):
            raise ValueError(
                f"neutral_threshold must be in (0,1), got {neutral_threshold}"
            )
        if zscore_window_bins < 2:
            raise ValueError("zscore_window_bins must be >= 2")
        if zscore_min_periods < 2:
            raise ValueError("zscore_min_periods must be >= 2")
        if zscore_min_periods > zscore_window_bins:
            raise ValueError("zscore_min_periods cannot exceed zscore_window_bins")
        if default_dt_s <= 0.0:
            raise ValueError(f"default_dt_s must be > 0, got {default_dt_s}")

        win = np.asarray(list(windows), dtype=np.int32)
        if win.ndim != 1 or win.size == 0:
            raise ValueError("windows must be a non-empty 1D sequence")
        if np.any(win <= 0):
            raise ValueError("windows values must be > 0")

        roll_w = np.asarray(list(rollup_weights), dtype=np.float64)
        if roll_w.ndim != 1 or roll_w.size != win.size:
            raise ValueError("rollup_weights must have the same length as windows")
        if np.any(roll_w <= 0.0):
            raise ValueError("rollup_weights values must be > 0")
        roll_w = roll_w / float(roll_w.sum())

        deriv_w = np.asarray(list(derivative_weights), dtype=np.float64)
        if deriv_w.ndim != 1 or deriv_w.size != 3:
            raise ValueError("derivative_weights must contain exactly 3 values")
        if np.any(deriv_w <= 0.0):
            raise ValueError("derivative_weights values must be > 0")
        deriv_w = deriv_w / float(deriv_w.sum())

        horizons = np.asarray(list(projection_horizons_ms), dtype=np.int32)
        if horizons.ndim != 1 or horizons.size == 0:
            raise ValueError("projection_horizons_ms must be non-empty")
        if np.any(horizons <= 0):
            raise ValueError("projection_horizons_ms values must be > 0")

        self._n_cells = n_cells
        self._windows = win
        self._rollup_weights = roll_w
        self._deriv_weights = deriv_w
        self._tanh_scale = float(tanh_scale)
        self._neutral_threshold = float(neutral_threshold)
        self._zscore_window_bins = int(zscore_window_bins)
        self._zscore_min_periods = int(zscore_min_periods)
        self._projection_horizons_ms = tuple(int(x) for x in horizons.tolist())
        self._default_dt_s = float(default_dt_s)

        max_hist = int(max(self._windows.max(), self._zscore_window_bins, 3))
        self._hist_capacity = max_hist
        self._composite_ring = np.zeros((max_hist, n_cells), dtype=np.float64)
        self._d1_ring = np.zeros((max_hist, n_cells), dtype=np.float64)
        self._d2_ring = np.zeros((max_hist, n_cells), dtype=np.float64)
        self._d3_ring = np.zeros((max_hist, n_cells), dtype=np.float64)
        self._composite_write_idx = 0
        self._d1_write_idx = 0
        self._d2_write_idx = 0
        self._d3_write_idx = 0
        self._composite_count = 0
        self._d1_count = 0
        self._d2_count = 0
        self._d3_count = 0
        self._rolling_sum_by_window = np.zeros((self._windows.size, n_cells), dtype=np.float64)
        self._zscore_scratch = np.empty(
            (self._zscore_window_bins, n_cells), dtype=np.float64
        )
        self._zeros = np.zeros(n_cells, dtype=np.float64)

        self._prev_ts_ns: int | None = None
        self._prev_rolled: np.ndarray | None = None
        self._prev_d1: np.ndarray | None = None
        self._prev_d2: np.ndarray | None = None
        self._prev_score: np.ndarray | None = None
        self._prev_score_d1: np.ndarray | None = None

    @property
    def projection_horizons_ms(self) -> tuple[int, ...]:
        return self._projection_horizons_ms

    @property
    def latest_composite(self) -> np.ndarray:
        """Return the latest composite vector (or zeros before first update)."""
        if self._composite_count == 0:
            return self._zeros
        last_idx = (self._composite_write_idx - 1) % self._hist_capacity
        return self._composite_ring[last_idx]

    def _append_composite(self, composite: np.ndarray) -> None:
        """Append composite to ring and update rolling sums for configured windows."""
        write_idx = self._composite_write_idx
        prev_count = self._composite_count

        for idx, window in enumerate(self._windows.tolist()):
            if prev_count >= window:
                old_idx = (write_idx - int(window)) % self._hist_capacity
                self._rolling_sum_by_window[idx] -= self._composite_ring[old_idx]
            self._rolling_sum_by_window[idx] += composite

        self._composite_ring[write_idx] = composite
        self._composite_write_idx = (write_idx + 1) % self._hist_capacity
        if self._composite_count < self._hist_capacity:
            self._composite_count += 1

    def _append_ring(
        self,
        ring: np.ndarray,
        write_idx: int,
        count: int,
        x: np.ndarray,
    ) -> tuple[int, int]:
        """Append one vector to an arbitrary history ring."""
        ring[write_idx] = x
        write_idx = (write_idx + 1) % self._hist_capacity
        if count < self._hist_capacity:
            count += 1
        return write_idx, count

    def _copy_tail(
        self,
        ring: np.ndarray,
        write_idx: int,
        count: int,
        window: int,
    ) -> np.ndarray:
        """Copy newest `window` rows from ring into scratch as contiguous history."""
        n = min(window, count)
        if n <= 0:
            return self._zscore_scratch[:0]

        start = (write_idx - n) % self._hist_capacity
        if start + n <= self._hist_capacity:
            self._zscore_scratch[:n] = ring[start:start + n]
        else:
            first = self._hist_capacity - start
            self._zscore_scratch[:first] = ring[start:]
            self._zscore_scratch[first:n] = ring[: n - first]
        return self._zscore_scratch[:n]

    def _robust_z_last(
        self,
        ring: np.ndarray,
        write_idx: int,
        count: int,
        x: np.ndarray,
    ) -> np.ndarray:
        if count < self._zscore_min_periods:
            return self._zeros

        hist = self._copy_tail(
            ring=ring,
            write_idx=write_idx,
            count=count,
            window=self._zscore_window_bins,
        )

        med = np.median(hist, axis=0)
        mad = np.median(np.abs(hist - med), axis=0)
        scale = 1.4826 * mad
        valid = scale > 1e-9

        z = np.zeros_like(x)
        z[valid] = (x[valid] - med[valid]) / scale[valid]
        return z

    def update(
        self,
        ts_ns: int,
        pressure: np.ndarray,
        vacuum: np.ndarray,
    ) -> SpectrumOutput:
        if pressure.shape != (self._n_cells,):
            raise ValueError(
                f"pressure shape must be ({self._n_cells},), got {pressure.shape}"
            )
        if vacuum.shape != (self._n_cells,):
            raise ValueError(
                f"vacuum shape must be ({self._n_cells},), got {vacuum.shape}"
            )

        pressure_f = pressure.astype(np.float64, copy=False)
        vacuum_f = vacuum.astype(np.float64, copy=False)
        composite = (pressure_f - vacuum_f) / (np.abs(pressure_f) + np.abs(vacuum_f) + _EPS)
        self._append_composite(composite)

        rolled = np.zeros(self._n_cells, dtype=np.float64)
        for idx, window in enumerate(self._windows.tolist()):
            count = min(self._composite_count, int(window))
            if count > 0:
                rolled += self._rollup_weights[idx] * (
                    self._rolling_sum_by_window[idx] / float(count)
                )

        if self._prev_ts_ns is not None and ts_ns > self._prev_ts_ns:
            dt_s = (ts_ns - self._prev_ts_ns) / 1e9
        else:
            dt_s = self._default_dt_s
        if dt_s <= 0.0:
            dt_s = self._default_dt_s

        if self._prev_rolled is None:
            d1 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            d1 = (rolled - self._prev_rolled) / dt_s

        if self._prev_d1 is None:
            d2 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            d2 = (d1 - self._prev_d1) / dt_s

        if self._prev_d2 is None:
            d3 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            d3 = (d2 - self._prev_d2) / dt_s

        self._d1_write_idx, self._d1_count = self._append_ring(
            ring=self._d1_ring,
            write_idx=self._d1_write_idx,
            count=self._d1_count,
            x=d1,
        )
        self._d2_write_idx, self._d2_count = self._append_ring(
            ring=self._d2_ring,
            write_idx=self._d2_write_idx,
            count=self._d2_count,
            x=d2,
        )
        self._d3_write_idx, self._d3_count = self._append_ring(
            ring=self._d3_ring,
            write_idx=self._d3_write_idx,
            count=self._d3_count,
            x=d3,
        )

        z1 = self._robust_z_last(self._d1_ring, self._d1_write_idx, self._d1_count, d1)
        z2 = self._robust_z_last(self._d2_ring, self._d2_write_idx, self._d2_count, d2)
        z3 = self._robust_z_last(self._d3_ring, self._d3_write_idx, self._d3_count, d3)

        score = (
            self._deriv_weights[0] * np.tanh(z1 / self._tanh_scale)
            + self._deriv_weights[1] * np.tanh(z2 / self._tanh_scale)
            + self._deriv_weights[2] * np.tanh(z3 / self._tanh_scale)
        )
        score = np.clip(score, -1.0, 1.0)
        score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=-1.0)

        state_code = np.zeros(self._n_cells, dtype=np.int8)
        state_code[score >= self._neutral_threshold] = 1
        state_code[score <= -self._neutral_threshold] = -1

        if self._prev_score is None:
            score_d1 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            score_d1 = (score - self._prev_score) / dt_s

        if self._prev_score_d1 is None:
            score_d2 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            score_d2 = (score_d1 - self._prev_score_d1) / dt_s

        projected: Dict[int, np.ndarray] = {}
        for horizon_ms in self._projection_horizons_ms:
            h = float(horizon_ms) / 1000.0
            projected[horizon_ms] = np.nan_to_num(
                np.clip(
                    score + score_d1 * h + 0.5 * score_d2 * h * h,
                    -1.0,
                    1.0,
                ),
                nan=0.0, posinf=1.0, neginf=-1.0,
            )

        self._prev_ts_ns = ts_ns
        self._prev_rolled = rolled
        self._prev_d1 = d1
        self._prev_d2 = d2
        self._prev_score = score
        self._prev_score_d1 = score_d1

        return SpectrumOutput(
            score=score,
            state_code=state_code,
            projected_score_by_horizon=projected,
        )
