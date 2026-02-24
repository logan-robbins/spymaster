"""Train/serve invariant tests for scoring.py's SpectrumScorer and score_dataset.

Verifies that the incremental API (SpectrumScorer.update) and batch API
(score_dataset) produce identical results, that parameterization flows
through correctly, and that edge cases are handled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.models.vacuum_pressure.scoring import SpectrumScorer, score_dataset
from src.qmachina.serving_config import ScoringConfig
from src.models.vacuum_pressure.spectrum import IndependentCellSpectrum


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


def _generate_derivatives(
    rng: np.random.Generator,
    n_cells: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = rng.standard_normal((n_bins, n_cells)) * 0.1
    d2 = rng.standard_normal((n_bins, n_cells)) * 0.01
    d3 = rng.standard_normal((n_bins, n_cells)) * 0.001
    return d1, d2, d3


def _build_grid_df(
    d1: np.ndarray,
    d2: np.ndarray,
    d3: np.ndarray,
    n_cells: int,
) -> pd.DataFrame:
    n_bins = d1.shape[0]
    rows = []
    for b in range(n_bins):
        for k in range(n_cells):
            rows.append({
                "bin_seq": b,
                "k": k,
                "composite_d1": d1[b, k],
                "composite_d2": d2[b, k],
                "composite_d3": d3[b, k],
            })
    return pd.DataFrame(rows)


# ===================================================================
# Test 1: Golden vectors -- known inputs produce known outputs
# ===================================================================

# Locked golden values from seed=42, default ScoringConfig, 5 cells, bin 100.
_GOLDEN_BIN100_SCORES = np.array([
    2.475041700366693020e-01,
    1.150841444583974305e-01,
    5.433544837187773840e-02,
    -1.932690478377614152e-01,
    -3.780128121873298497e-01,
], dtype=np.float64)

_GOLDEN_BIN100_STATES = np.array([1, 0, 0, -1, -1], dtype=np.int8)


class TestGoldenVectors:
    N_CELLS = 5
    N_BINS = 200
    WARMUP_END = 74  # first non-warmup bin (0-indexed), min_periods=75

    def _run_all_bins(
        self, config: ScoringConfig,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng = np.random.default_rng(seed=42)
        d1, d2, d3 = _generate_derivatives(rng, self.N_CELLS, self.N_BINS)
        scorer = SpectrumScorer(config, self.N_CELLS)
        scores, states = [], []
        for i in range(self.N_BINS):
            s, st = scorer.update(d1[i], d2[i], d3[i])
            scores.append(s.copy())
            states.append(st.copy())
        return scores, states

    def test_warmup_zeros(self, default_config: ScoringConfig) -> None:
        scores, states = self._run_all_bins(default_config)
        for i in range(self.WARMUP_END):
            np.testing.assert_array_equal(scores[i], 0.0)
            np.testing.assert_array_equal(states[i], 0)

    def test_post_warmup_ranges(self, default_config: ScoringConfig) -> None:
        scores, states = self._run_all_bins(default_config)
        for i in range(self.WARMUP_END, self.N_BINS):
            assert np.all(scores[i] >= -1.0)
            assert np.all(scores[i] <= 1.0)
            assert np.all(np.isin(states[i], [-1, 0, 1]))

    def test_golden_bin100(self, default_config: ScoringConfig) -> None:
        scores, states = self._run_all_bins(default_config)
        np.testing.assert_allclose(
            scores[100], _GOLDEN_BIN100_SCORES, atol=1e-15,
        )
        np.testing.assert_array_equal(states[100], _GOLDEN_BIN100_STATES)


# ===================================================================
# Test 2: Incremental vs batch API parity
# ===================================================================

class TestIncrementalBatchParity:
    N_CELLS = 10
    N_BINS = 150

    def test_bitwise_parity(self, default_config: ScoringConfig, rng: np.random.Generator) -> None:
        d1, d2, d3 = _generate_derivatives(rng, self.N_CELLS, self.N_BINS)

        # Incremental path.
        scorer = SpectrumScorer(default_config, self.N_CELLS)
        inc_scores = np.empty((self.N_BINS, self.N_CELLS), dtype=np.float64)
        inc_states = np.empty((self.N_BINS, self.N_CELLS), dtype=np.int8)
        for i in range(self.N_BINS):
            s, st = scorer.update(d1[i], d2[i], d3[i])
            inc_scores[i] = s
            inc_states[i] = st

        # Batch path.
        df = _build_grid_df(d1, d2, d3, self.N_CELLS)
        result = score_dataset(df, default_config, self.N_CELLS)

        # Extract batch results in bin_seq/k order.
        batch_df = result.sort_values(["bin_seq", "k"]).reset_index(drop=True)
        batch_scores = batch_df["flow_score"].to_numpy(dtype=np.float64).reshape(
            self.N_BINS, self.N_CELLS,
        )
        batch_states = batch_df["flow_state_code"].to_numpy(dtype=np.int8).reshape(
            self.N_BINS, self.N_CELLS,
        )

        max_score_diff = np.max(np.abs(inc_scores - batch_scores))
        assert max_score_diff < 1e-14, f"max score diff = {max_score_diff}"
        np.testing.assert_array_equal(inc_states, batch_states)


# ===================================================================
# Test 3: Parameter sensitivity
# ===================================================================

class TestParameterSensitivity:
    N_CELLS = 5
    N_BINS = 150

    def test_tanh_scale_affects_scores(self, rng: np.random.Generator) -> None:
        d1, d2, d3 = _generate_derivatives(rng, self.N_CELLS, self.N_BINS)

        config_a = ScoringConfig(tanh_scale=3.0)
        config_b = ScoringConfig(tanh_scale=10.0)

        scorer_a = SpectrumScorer(config_a, self.N_CELLS)
        scorer_b = SpectrumScorer(config_b, self.N_CELLS)

        diverged = False
        for i in range(self.N_BINS):
            sa, _ = scorer_a.update(d1[i], d2[i], d3[i])
            sb, _ = scorer_b.update(d1[i], d2[i], d3[i])
            if not np.allclose(sa, sb, atol=1e-15):
                diverged = True
                break

        assert diverged, "Different tanh_scale values must produce different scores"

    def test_threshold_affects_states(self, rng: np.random.Generator) -> None:
        d1, d2, d3 = _generate_derivatives(rng, self.N_CELLS, self.N_BINS)

        config_a = ScoringConfig(threshold_neutral=0.05)
        config_b = ScoringConfig(threshold_neutral=0.45)

        scorer_a = SpectrumScorer(config_a, self.N_CELLS)
        scorer_b = SpectrumScorer(config_b, self.N_CELLS)

        diverged = False
        for i in range(self.N_BINS):
            _, sta = scorer_a.update(d1[i], d2[i], d3[i])
            _, stb = scorer_b.update(d1[i], d2[i], d3[i])
            if not np.array_equal(sta, stb):
                diverged = True
                break

        assert diverged, "Different threshold_neutral values must produce different state_codes"


# ===================================================================
# Test 4: Spectrum delegation parity
# ===================================================================

class TestSpectrumDelegation:
    N_CELLS = 5
    N_BINS = 120

    def test_spectrum_scorer_parity(self, rng: np.random.Generator) -> None:
        windows = [3, 6]
        rollup_weights = [1.0, 1.0]
        derivative_weights = [0.55, 0.30, 0.15]
        tanh_scale = 3.0
        neutral_threshold = 0.15
        zscore_window_bins = 50
        zscore_min_periods = 20
        dt_s = 0.5

        spectrum = IndependentCellSpectrum(
            n_cells=self.N_CELLS,
            windows=windows,
            rollup_weights=rollup_weights,
            derivative_weights=derivative_weights,
            tanh_scale=tanh_scale,
            neutral_threshold=neutral_threshold,
            zscore_window_bins=zscore_window_bins,
            zscore_min_periods=zscore_min_periods,
            projection_horizons_ms=[100],
            default_dt_s=dt_s,
        )

        # Normalize derivative weights identically to how spectrum.py does it.
        dw = np.asarray(derivative_weights, dtype=np.float64)
        dw = dw / dw.sum()

        standalone_scorer = SpectrumScorer(
            ScoringConfig(
                zscore_window_bins=zscore_window_bins,
                zscore_min_periods=zscore_min_periods,
                derivative_weights=list(dw),
                tanh_scale=tanh_scale,
                threshold_neutral=neutral_threshold,
            ),
            n_cells=self.N_CELLS,
        )

        pressure_data = rng.uniform(0.0, 10.0, (self.N_BINS, self.N_CELLS))
        vacuum_data = rng.uniform(0.0, 10.0, (self.N_BINS, self.N_CELLS))

        for i in range(self.N_BINS):
            ts_ns = int((i + 1) * dt_s * 1e9)
            out = spectrum.update(
                ts_ns=ts_ns,
                pressure=pressure_data[i],
                vacuum=vacuum_data[i],
            )
            # Feed the same d1/d2/d3 that spectrum computed to standalone scorer.
            standalone_score, standalone_state = standalone_scorer.update(
                out.composite_d1, out.composite_d2, out.composite_d3,
            )
            np.testing.assert_allclose(
                out.score, standalone_score, atol=1e-15,
                err_msg=f"Score mismatch at bin {i}",
            )
            np.testing.assert_array_equal(
                out.state_code, standalone_state,
                err_msg=f"State code mismatch at bin {i}",
            )


# ===================================================================
# Test 5: Edge cases
# ===================================================================

class TestEdgeCases:

    def test_single_cell(self, default_config: ScoringConfig) -> None:
        scorer = SpectrumScorer(default_config, n_cells=1)
        rng = np.random.default_rng(seed=99)
        for _ in range(100):
            d1 = rng.standard_normal(1) * 0.1
            d2 = rng.standard_normal(1) * 0.01
            d3 = rng.standard_normal(1) * 0.001
            score, state = scorer.update(d1, d2, d3)
            assert score.shape == (1,)
            assert state.shape == (1,)
            assert -1.0 <= score[0] <= 1.0
            assert state[0] in (-1, 0, 1)

    def test_all_zero_derivatives(self, default_config: ScoringConfig) -> None:
        scorer = SpectrumScorer(default_config, n_cells=3)
        zeros = np.zeros(3, dtype=np.float64)
        for _ in range(200):
            score, state = scorer.update(zeros, zeros, zeros)
        np.testing.assert_array_equal(score, 0.0)
        np.testing.assert_array_equal(state, 0)

    def test_constant_input_produces_zero(self) -> None:
        config = ScoringConfig(zscore_window_bins=50, zscore_min_periods=10)
        scorer = SpectrumScorer(config, n_cells=4)
        constant = np.full(4, 0.5, dtype=np.float64)
        for _ in range(100):
            score, state = scorer.update(constant, constant, constant)
        # Constant input -> MAD=0 -> scale<eps -> z=0 -> score=0.
        np.testing.assert_array_equal(score, 0.0)
        np.testing.assert_array_equal(state, 0)

    @pytest.mark.parametrize("n_cells", [1, 2, 50])
    def test_score_dataset_single_bin(
        self, default_config: ScoringConfig, n_cells: int,
    ) -> None:
        rng = np.random.default_rng(seed=7)
        d1, d2, d3 = _generate_derivatives(rng, n_cells, 1)
        df = _build_grid_df(d1, d2, d3, n_cells)
        result = score_dataset(df, default_config, n_cells)
        # Single bin during warmup -> all zeros.
        np.testing.assert_array_equal(
            result["flow_score"].to_numpy(), 0.0,
        )
        np.testing.assert_array_equal(
            result["flow_state_code"].to_numpy(), 0,
        )
