"""Independent per-cell spectrum scoring and projection kernel.

All math is vectorized across cells. Each cell is scored independently from its
own time history; no cross-cell coupling is used in this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .scoring import SpectrumScorer
from .serving_config import ScoringConfig

_EPS = 1e-12


@dataclass(frozen=True)
class SpectrumOutput:
    """Vectorized spectrum outputs for one emitted time bin."""

    score: np.ndarray
    state_code: np.ndarray
    projected_score_by_horizon: Dict[int, np.ndarray]
    composite: np.ndarray
    composite_d1: np.ndarray
    composite_d2: np.ndarray
    composite_d3: np.ndarray


@dataclass(frozen=True)
class ProjectionModelConfig:
    """Controls forward score projection behavior."""

    use_cubic: bool = False
    cubic_scale: float = 1.0 / 6.0
    damping_lambda: float = 0.0


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
        projection_model: ProjectionModelConfig | None = None,
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
        if projection_model is None:
            projection_model = ProjectionModelConfig()
        if projection_model.cubic_scale < 0.0:
            raise ValueError(
                f"projection cubic_scale must be >= 0, got {projection_model.cubic_scale}"
            )
        if projection_model.damping_lambda < 0.0:
            raise ValueError(
                f"projection damping_lambda must be >= 0, got {projection_model.damping_lambda}"
            )

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
        self._windows_py = tuple(int(x) for x in win.tolist())
        self._rollup_weights = roll_w
        self._deriv_weights = deriv_w
        self._tanh_scale = float(tanh_scale)
        self._inv_tanh_scale = 1.0 / self._tanh_scale
        self._neutral_threshold = float(neutral_threshold)
        self._zscore_window_bins = int(zscore_window_bins)
        self._zscore_min_periods = int(zscore_min_periods)
        self._projection_horizons_ms = tuple(int(x) for x in horizons.tolist())
        self._default_dt_s = float(default_dt_s)
        self._projection_model = projection_model

        max_hist = int(max(self._windows.max(), 3))
        self._hist_capacity = max_hist
        self._composite_ring = np.zeros((max_hist, n_cells), dtype=np.float64)
        self._composite_write_idx = 0
        self._composite_count = 0
        self._rolling_sum_by_window = np.zeros((self._windows.size, n_cells), dtype=np.float64)
        self._zeros = np.zeros(n_cells, dtype=np.float64)

        # Delegate scoring to the canonical SpectrumScorer (single source of truth).
        self._scorer = SpectrumScorer(
            ScoringConfig(
                zscore_window_bins=zscore_window_bins,
                zscore_min_periods=zscore_min_periods,
                derivative_weights=list(deriv_w),
                tanh_scale=tanh_scale,
                threshold_neutral=neutral_threshold,
            ),
            n_cells=n_cells,
        )

        self._prev_ts_ns: int | None = None
        self._prev_rolled: np.ndarray | None = None
        self._prev_d1: np.ndarray | None = None
        self._prev_d2: np.ndarray | None = None
        self._prev_score: np.ndarray | None = None
        self._prev_score_d1: np.ndarray | None = None
        self._prev_score_d2: np.ndarray | None = None

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

        for idx, window in enumerate(self._windows_py):
            if prev_count >= window:
                old_idx = (write_idx - window) % self._hist_capacity
                self._rolling_sum_by_window[idx] -= self._composite_ring[old_idx]
            self._rolling_sum_by_window[idx] += composite

        self._composite_ring[write_idx] = composite
        self._composite_write_idx = (write_idx + 1) % self._hist_capacity
        if self._composite_count < self._hist_capacity:
            self._composite_count += 1

    def _project_score_horizon(
        self,
        score: np.ndarray,
        score_d1: np.ndarray,
        score_d2: np.ndarray,
        score_d3: np.ndarray,
        horizon_s: float,
    ) -> np.ndarray:
        proj = score + score_d1 * horizon_s + 0.5 * score_d2 * horizon_s * horizon_s
        if self._projection_model.use_cubic:
            proj = proj + (
                self._projection_model.cubic_scale * score_d3 * horizon_s * horizon_s * horizon_s
            )
        if self._projection_model.damping_lambda > 0.0:
            proj = proj * np.exp(-self._projection_model.damping_lambda * horizon_s)
        np.clip(proj, -1.0, 1.0, out=proj)
        np.nan_to_num(proj, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return proj

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
        for idx, window in enumerate(self._windows_py):
            count = min(self._composite_count, window)
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

        # Delegate scoring to SpectrumScorer (single source of truth).
        score, state_code = self._scorer.update(d1, d2, d3)

        if self._prev_score is None:
            score_d1 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            score_d1 = (score - self._prev_score) / dt_s

        if self._prev_score_d1 is None:
            score_d2 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            score_d2 = (score_d1 - self._prev_score_d1) / dt_s
        if self._prev_score_d2 is None:
            score_d3 = np.zeros(self._n_cells, dtype=np.float64)
        else:
            score_d3 = (score_d2 - self._prev_score_d2) / dt_s

        projected: Dict[int, np.ndarray] = {}
        for horizon_ms in self._projection_horizons_ms:
            h = float(horizon_ms) / 1000.0
            projected[horizon_ms] = self._project_score_horizon(
                score=score,
                score_d1=score_d1,
                score_d2=score_d2,
                score_d3=score_d3,
                horizon_s=h,
            )

        self._prev_ts_ns = ts_ns
        self._prev_rolled = rolled
        self._prev_d1 = d1
        self._prev_d2 = d2
        self._prev_score = score
        self._prev_score_d1 = score_d1
        self._prev_score_d2 = score_d2

        return SpectrumOutput(
            score=score,
            state_code=state_code,
            projected_score_by_horizon=projected,
            composite=rolled,
            composite_d1=d1,
            composite_d2=d2,
            composite_d3=d3,
        )
