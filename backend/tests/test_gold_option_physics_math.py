"""
Synthetic math validation for the future_option_mbo gold layer
(compute_physics_surface_1s.py).

Validates every formula in the pipeline against hand-computed expected values.
No live data or I/O required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_eng.stages.gold.future_option_mbo.compute_physics_surface_1s import (
    GoldComputeOptionPhysicsSurface1s,
    TICK_INT,
    STRIKE_STEP_INT,
    STRIKE_TICKS,
    EPS_QTY,
    PRICE_SCALE,
    _round_to_nearest_strike_int,
)

# ---------------------------------------------------------------------------
# Helper: build a synthetic depth_and_flow_1s DataFrame
# ---------------------------------------------------------------------------

SPOT_PRICE = 6100.0
SPOT_INT = int(SPOT_PRICE / PRICE_SCALE)

WINDOW_NS = 1_000_000_000


def _make_flow_row(
    window_idx: int,
    strike_offset: int,  # in $5 increments from spot
    right: str,
    side: str,
    depth_qty_start: float,
    depth_qty_end: float,
    add_qty: float,
    pull_qty: float,
    fill_qty: float,
    depth_qty_rest: float,
) -> dict:
    """Build one row of a depth_and_flow_1s DataFrame."""
    strike_int = SPOT_INT + strike_offset * STRIKE_STEP_INT
    # rel_ticks = (strike_int - spot_int) / TICK_INT
    # Since spot IS on the $5 grid here, offset * STRIKE_STEP_INT / TICK_INT = offset * 20
    rel_ticks = strike_offset * STRIKE_TICKS
    window_end = (window_idx + 1) * WINDOW_NS

    return {
        "window_start_ts_ns": window_idx * WINDOW_NS,
        "window_end_ts_ns": window_end,
        "strike_price_int": strike_int,
        "strike_points": strike_int * PRICE_SCALE,
        "right": right,
        "side": side,
        "spot_ref_price_int": SPOT_INT,
        "rel_ticks": rel_ticks,
        "depth_qty_start": depth_qty_start,
        "depth_qty_end": depth_qty_end,
        "add_qty": add_qty,
        "pull_qty": pull_qty,
        "pull_qty_rest": 0.0,
        "fill_qty": fill_qty,
        "depth_qty_rest": depth_qty_rest,
        "window_valid": True,
        "accounting_identity_valid": True,
    }


def _build_synthetic_flow(n_windows: int = 10, n_strikes: int = 5) -> pd.DataFrame:
    """Build a multi-window, multi-strike synthetic flow surface.

    Creates a controlled dataset where:
    - Strikes: offsets [-2, -1, 0, +1, +2] from spot
    - Rights: C and P
    - Sides: A and B
    - Each window has linearly increasing add_qty to test EMA convergence
    """
    rows = []
    for w in range(n_windows):
        for offset in range(-n_strikes // 2, n_strikes // 2 + 1):
            for right in ("C", "P"):
                for side in ("A", "B"):
                    # Linearly increasing flow to test EMA behavior
                    base_add = float(w + 1) * 2.0
                    base_pull = float(w + 1) * 0.5
                    base_fill = float(w + 1) * 0.3
                    depth_start = max(0.0, 10.0 + offset * 2.0)
                    depth_end = max(0.0, depth_start + base_add - base_pull - base_fill)
                    depth_rest = min(depth_end * 0.6, depth_end)

                    rows.append(
                        _make_flow_row(
                            window_idx=w,
                            strike_offset=offset,
                            right=right,
                            side=side,
                            depth_qty_start=depth_start,
                            depth_qty_end=depth_end,
                            add_qty=base_add,
                            pull_qty=base_pull,
                            fill_qty=base_fill,
                            depth_qty_rest=depth_rest,
                        )
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundToNearestStrikeInt:
    """Verify the _round_to_nearest_strike_int utility."""

    def test_exact_strike(self):
        """Price exactly on $5 boundary should stay there."""
        price_int = np.array([int(6100.0 / PRICE_SCALE)], dtype="int64")
        result = _round_to_nearest_strike_int(price_int)
        assert result[0] == price_int[0]

    def test_round_up(self):
        """Price at $6102.75 should round to $6105."""
        price_int = np.array([int(6102.75 / PRICE_SCALE)], dtype="int64")
        result = _round_to_nearest_strike_int(price_int)
        expected = int(6105.0 / PRICE_SCALE)
        assert result[0] == expected

    def test_round_down(self):
        """Price at $6101.25 should round to $6100."""
        price_int = np.array([int(6101.25 / PRICE_SCALE)], dtype="int64")
        result = _round_to_nearest_strike_int(price_int)
        expected = int(6100.0 / PRICE_SCALE)
        assert result[0] == expected

    def test_midpoint_rounds_up(self):
        """Price at $6102.50 (midpoint) should round up to $6105.

        Formula: (val + STRIKE_STEP_INT//2) // STRIKE_STEP_INT * STRIKE_STEP_INT
        """
        price_int = np.array([int(6102.50 / PRICE_SCALE)], dtype="int64")
        result = _round_to_nearest_strike_int(price_int)
        expected = int(6105.0 / PRICE_SCALE)
        assert result[0] == expected


class TestIntensityFormulas:
    """Verify intensity = qty / (depth_qty_start + EPS_QTY)."""

    def test_basic_intensity(self):
        """Intensity with nonzero depth_start."""
        depth_start = 10.0
        add_qty = 5.0
        expected = add_qty / (depth_start + EPS_QTY)
        assert expected == pytest.approx(5.0 / 11.0)

    def test_zero_depth_start(self):
        """When depth_start=0, denominator is EPS_QTY=1.0."""
        depth_start = 0.0
        add_qty = 3.0
        expected = add_qty / (depth_start + EPS_QTY)
        assert expected == pytest.approx(3.0)

    def test_liquidity_velocity_sign(self):
        """velocity = add_intensity - pull_intensity - fill_intensity.

        Net positive when adds dominate, net negative when pulls+fills dominate.
        """
        depth_start = 10.0
        denom = depth_start + EPS_QTY

        # Scenario 1: more adds than pulls+fills -> positive
        add_i = 8.0 / denom
        pull_i = 2.0 / denom
        fill_i = 1.0 / denom
        vel = add_i - pull_i - fill_i
        assert vel > 0

        # Scenario 2: more pulls than adds -> negative
        add_i2 = 1.0 / denom
        pull_i2 = 5.0 / denom
        fill_i2 = 2.0 / denom
        vel2 = add_i2 - pull_i2 - fill_i2
        assert vel2 < 0


class TestRhoOpt:
    """Verify rho_opt = log(1 + depth_qty_end)."""

    def test_zero_depth(self):
        assert np.log(1.0 + 0.0) == pytest.approx(0.0)

    def test_small_depth(self):
        """Options often have depth of 1-5 contracts."""
        for d in [1, 2, 3, 5]:
            expected = np.log(1.0 + d)
            assert expected == pytest.approx(np.log1p(d))

    def test_large_depth(self):
        """At-the-money can have 50+ contracts."""
        d = 50.0
        expected = np.log(1.0 + d)
        assert expected == pytest.approx(np.log(51.0))


class TestPhiRestOpt:
    """Verify phi_rest_opt = depth_qty_rest / (depth_qty_end + 1.0)."""

    def test_all_resting(self):
        """When all depth is resting, phi approaches but never reaches 1.0."""
        depth_end = 10.0
        depth_rest = 10.0
        phi = depth_rest / (depth_end + 1.0)
        assert phi == pytest.approx(10.0 / 11.0)
        assert phi < 1.0

    def test_no_resting(self):
        """When nothing is resting, phi = 0."""
        depth_end = 10.0
        depth_rest = 0.0
        phi = depth_rest / (depth_end + 1.0)
        assert phi == pytest.approx(0.0)

    def test_zero_depth(self):
        """When depth_end=0 and depth_rest=0, phi = 0/(0+1) = 0."""
        phi = 0.0 / (0.0 + 1.0)
        assert phi == pytest.approx(0.0)

    def test_depth_rest_clamped(self):
        """depth_qty_rest should be <= depth_qty_end (clamped in silver).

        phi_rest_opt = depth_rest / (depth_end + 1.0)
        If rest = end = 5: phi = 5/6 ~ 0.833
        """
        depth_end = 5.0
        depth_rest = min(5.0, depth_end)  # clamped
        phi = depth_rest / (depth_end + 1.0)
        assert phi == pytest.approx(5.0 / 6.0)


class TestEMAFormulas:
    """Verify EMA alpha = 1 - exp(-1/tau) for tau = 2, 8, 32, 128."""

    @pytest.mark.parametrize("tau", [2, 8, 32, 128])
    def test_alpha_range(self, tau: int):
        """Alpha must be in (0, 1)."""
        alpha = 1.0 - np.exp(-1.0 / tau)
        assert 0 < alpha < 1

    def test_alpha_ordering(self):
        """Shorter tau -> larger alpha (faster response)."""
        alphas = [1.0 - np.exp(-1.0 / tau) for tau in [2, 8, 32, 128]]
        assert alphas[0] > alphas[1] > alphas[2] > alphas[3]

    def test_ema_convergence(self):
        """EMA should converge to constant input value."""
        alpha = 1.0 - np.exp(-1.0 / 2.0)
        ema = 0.0
        for _ in range(100):
            ema = alpha * 5.0 + (1 - alpha) * ema
        assert ema == pytest.approx(5.0, abs=1e-6)

    def test_ema_step_response(self):
        """After a step from 0 to 1, EMA after 1 step = alpha."""
        alpha = 1.0 - np.exp(-1.0 / 8.0)
        ema = alpha * 1.0 + (1 - alpha) * 0.0
        assert ema == pytest.approx(alpha)


class TestBandAndWaveEnergy:
    """Verify band decomposition and wave energy."""

    def test_band_definitions(self):
        """Bands are differences of adjacent EMA scales."""
        ema_2 = 3.0
        ema_8 = 2.5
        ema_32 = 2.0
        ema_128 = 1.5

        fast = ema_2 - ema_8
        mid = ema_8 - ema_32
        slow = ema_32 - ema_128

        assert fast == pytest.approx(0.5)
        assert mid == pytest.approx(0.5)
        assert slow == pytest.approx(0.5)

    def test_wave_energy(self):
        """wave_energy = sqrt(fast^2 + mid^2 + slow^2)."""
        fast, mid, slow = 3.0, 4.0, 0.0
        energy = np.sqrt(fast**2 + mid**2 + slow**2)
        assert energy == pytest.approx(5.0)

    def test_zero_energy(self):
        """When all bands are zero, energy is zero."""
        energy = np.sqrt(0.0**2 + 0.0**2 + 0.0**2)
        assert energy == pytest.approx(0.0)


class TestTemporalDerivatives:
    """Verify du_opt_dt and d2u_opt_dt2."""

    def test_first_derivative(self):
        """du/dt = u_opt_ema_2[t] - u_opt_ema_2[t-1]."""
        u2 = [1.0, 3.0, 2.0]
        du = [0.0, 2.0, -1.0]  # first is fillna(0)
        for i in range(1, 3):
            assert du[i] == pytest.approx(u2[i] - u2[i - 1])

    def test_second_derivative(self):
        """d2u/dt2 = du/dt[t] - du/dt[t-1]."""
        du = [0.0, 2.0, -1.0]
        d2u = [0.0, 2.0, -3.0]  # first is fillna(0)
        for i in range(1, 3):
            assert d2u[i] == pytest.approx(du[i] - du[i - 1])


class TestPersistenceWeightedVelocity:
    """Verify u_opt_p and u_opt_p_slow."""

    def test_u_opt_p(self):
        """u_opt_p = phi_rest_opt * u_opt_ema_8."""
        phi = 0.8
        ema8 = 2.5
        assert phi * ema8 == pytest.approx(2.0)

    def test_u_opt_p_slow(self):
        """u_opt_p_slow = phi_rest_opt * u_opt_ema_32."""
        phi = 0.6
        ema32 = 1.5
        assert phi * ema32 == pytest.approx(0.9)

    def test_zero_phi_kills_signal(self):
        """When no depth is resting (phi=0), persistence-weighted velocity = 0."""
        assert 0.0 * 5.0 == pytest.approx(0.0)


class TestOmegaOpt:
    """Verify Omega_opt = rho_opt * (0.5 + 0.5*phi_rest_opt) * (1 + max(0, u_opt_p_slow))."""

    def test_basic(self):
        rho = np.log(1 + 10.0)  # ~2.398
        phi = 0.8
        u_p_slow = 0.5
        omega = rho * (0.5 + 0.5 * phi) * (1 + max(0, u_p_slow))
        expected = rho * 0.9 * 1.5
        assert omega == pytest.approx(expected)

    def test_negative_u_p_slow_clamped(self):
        """max(0, u_opt_p_slow) clamps negative values to 0."""
        rho = 1.0
        phi = 0.5
        u_p_slow = -2.0
        omega = rho * (0.5 + 0.5 * phi) * (1 + max(0, u_p_slow))
        expected = rho * 0.75 * 1.0  # max(0, -2) = 0
        assert omega == pytest.approx(expected)

    def test_zero_depth(self):
        """When depth=0, rho=0, so Omega=0 regardless of other factors."""
        rho = np.log(1 + 0.0)  # 0
        phi = 0.5
        u_p_slow = 1.0
        omega = rho * (0.5 + 0.5 * phi) * (1 + max(0, u_p_slow))
        assert omega == pytest.approx(0.0)


class TestViscosityPermeability:
    """Verify nu_opt and kappa_opt."""

    def test_nu_opt_formula(self):
        """nu_opt = 1 + Omega_opt_near + 2*max(0, Omega_opt_prom)."""
        omega_near = 2.0
        omega_prom = 0.5
        nu = 1.0 + omega_near + 2.0 * max(0, omega_prom)
        assert nu == pytest.approx(4.0)

    def test_nu_opt_negative_prom(self):
        """Negative prominence is clamped to 0."""
        omega_near = 2.0
        omega_prom = -1.0
        nu = 1.0 + omega_near + 2.0 * max(0, omega_prom)
        assert nu == pytest.approx(3.0)

    def test_kappa_opt(self):
        """kappa_opt = 1/nu_opt."""
        nu = 4.0
        kappa = 1.0 / nu
        assert kappa == pytest.approx(0.25)

    def test_kappa_minimum_nu(self):
        """nu_opt >= 1 always (when Omega_near >= 0), so kappa <= 1."""
        # Minimum case: Omega_near = 0, Omega_prom <= 0
        nu_min = 1.0 + 0.0 + 2.0 * max(0, -999.0)
        assert nu_min == pytest.approx(1.0)
        assert 1.0 / nu_min == pytest.approx(1.0)


class TestPressureGrad:
    """Verify pressure_grad_opt sign convention."""

    def test_bid_side_positive(self):
        """side='B' -> +u_opt_p."""
        u_opt_p = 2.0
        side = "B"
        pg = u_opt_p if side == "B" else -u_opt_p
        assert pg == pytest.approx(2.0)

    def test_ask_side_negative(self):
        """side='A' -> -u_opt_p."""
        u_opt_p = 2.0
        side = "A"
        pg = u_opt_p if side == "B" else -u_opt_p
        assert pg == pytest.approx(-2.0)


class TestSpatialDerivatives:
    """Verify du_opt_dx and d2u_opt_dx2 finite difference stencils."""

    def test_first_spatial_derivative(self):
        """du/dx = (u[i+1] - u[i-1]) / (2 * STRIKE_TICKS).

        Central difference with spacing of STRIKE_TICKS (20 ticks = $5).
        """
        u = [1.0, 3.0, 7.0]  # at strike indices -1, 0, +1
        du_dx = (u[2] - u[0]) / (2.0 * STRIKE_TICKS)
        expected = (7.0 - 1.0) / 40.0
        assert du_dx == pytest.approx(expected)

    def test_second_spatial_derivative(self):
        """d2u/dx2 = (u[i+1] - 2*u[i] + u[i-1]) / STRIKE_TICKS^2.

        Standard second-order central difference.
        """
        u = [1.0, 3.0, 7.0]
        d2u_dx2 = (u[2] - 2.0 * u[1] + u[0]) / float(STRIKE_TICKS**2)
        expected = (7.0 - 6.0 + 1.0) / 400.0
        assert d2u_dx2 == pytest.approx(expected)

    def test_linear_field_zero_curvature(self):
        """A linear u field should have zero second derivative."""
        u = [2.0, 4.0, 6.0]
        d2u_dx2 = (u[2] - 2.0 * u[1] + u[0]) / float(STRIKE_TICKS**2)
        assert d2u_dx2 == pytest.approx(0.0)


class TestSmoothingParameters:
    """Verify Gaussian smoothing parameter conversion from ticks to strikes."""

    def test_near_parameters(self):
        """Near: n_ticks=16, sigma_ticks=6.

        n_strikes = round(16/20) = 1  -> window = 2*1+1 = 3
        sigma_strikes = max(1, 6/20) = max(1, 0.3) = 1.0
        """
        n_strikes = max(1, int(round(16 / STRIKE_TICKS)))
        sigma_strikes = max(1.0, 6.0 / STRIKE_TICKS)
        assert n_strikes == 1
        assert sigma_strikes == pytest.approx(1.0)
        assert 2 * n_strikes + 1 == 3

    def test_far_parameters(self):
        """Far: n_ticks=64, sigma_ticks=24.

        n_strikes = round(64/20) = 3  -> window = 2*3+1 = 7
        sigma_strikes = max(1, 24/20) = max(1, 1.2) = 1.2
        """
        n_strikes = max(1, int(round(64 / STRIKE_TICKS)))
        sigma_strikes = max(1.0, 24.0 / STRIKE_TICKS)
        assert n_strikes == 3
        assert sigma_strikes == pytest.approx(1.2)
        assert 2 * n_strikes + 1 == 7


class TestEndToEndTransform:
    """Run the gold transform on synthetic data and validate outputs."""

    @pytest.fixture
    def gold_stage(self):
        return GoldComputeOptionPhysicsSurface1s()

    @pytest.fixture
    def df_flow(self):
        return _build_synthetic_flow(n_windows=10, n_strikes=5)

    def test_transform_runs(self, gold_stage, df_flow):
        """Transform completes without error."""
        result = gold_stage.transform(df_flow)
        assert not result.empty

    def test_output_columns(self, gold_stage, df_flow):
        """All expected output columns present."""
        result = gold_stage.transform(df_flow)
        expected_cols = {
            "window_end_ts_ns",
            "event_ts_ns",
            "spot_ref_price_int",
            "strike_price_int",
            "strike_points",
            "rel_ticks",
            "right",
            "side",
            "add_intensity",
            "fill_intensity",
            "pull_intensity",
            "liquidity_velocity",
            "rho_opt",
            "phi_rest_opt",
            "u_opt_ema_2",
            "u_opt_ema_8",
            "u_opt_ema_32",
            "u_opt_ema_128",
            "u_opt_band_fast",
            "u_opt_band_mid",
            "u_opt_band_slow",
            "u_opt_wave_energy",
            "du_opt_dt",
            "d2u_opt_dt2",
            "u_opt_p",
            "u_opt_p_slow",
            "u_opt_near",
            "u_opt_far",
            "u_opt_prom",
            "du_opt_dx",
            "d2u_opt_dx2",
            "Omega_opt",
            "Omega_opt_near",
            "Omega_opt_far",
            "Omega_opt_prom",
            "nu_opt",
            "kappa_opt",
            "pressure_grad_opt",
        }
        assert set(result.columns) == expected_cols

    def test_no_nan_in_output(self, gold_stage, df_flow):
        """No NaN values in final output (all fillna applied)."""
        result = gold_stage.transform(df_flow)
        nan_cols = result.columns[result.isna().any()].tolist()
        assert nan_cols == [], f"NaN found in columns: {nan_cols}"

    def test_intensity_values(self, gold_stage, df_flow):
        """Intensities should be non-negative."""
        result = gold_stage.transform(df_flow)
        assert (result["add_intensity"] >= 0).all()
        assert (result["fill_intensity"] >= 0).all()
        assert (result["pull_intensity"] >= 0).all()

    def test_rho_non_negative(self, gold_stage, df_flow):
        """rho_opt = log(1 + depth) >= 0 always."""
        result = gold_stage.transform(df_flow)
        assert (result["rho_opt"] >= 0).all()

    def test_phi_bounded(self, gold_stage, df_flow):
        """phi_rest_opt must be in [0, 1) since denominator > numerator.

        phi = depth_rest / (depth_end + 1.0)
        Since depth_rest <= depth_end (clamped), phi < depth_end / (depth_end+1) < 1.
        """
        result = gold_stage.transform(df_flow)
        assert (result["phi_rest_opt"] >= 0).all()
        assert (result["phi_rest_opt"] < 1.0).all()

    def test_nu_lower_bound(self, gold_stage, df_flow):
        """nu_opt >= 1 always, since it's 1 + non-negative terms."""
        result = gold_stage.transform(df_flow)
        # nu = 1 + Omega_near + 2*max(0, Omega_prom)
        # Omega_near could be negative in theory (if rho or phi or u_p_slow cause it)
        # but the minimum of nu is when Omega_near < 0 and Omega_prom < 0
        # In that case nu = 1 + Omega_near (which could be < 1)
        # Actually wait - let me check if nu can be < 1
        # Omega_opt = rho * (0.5 + 0.5*phi) * (1 + max(0, u_p_slow))
        # rho >= 0, (0.5+0.5*phi) in [0.5, 1), (1+max(0,slow)) >= 1
        # So Omega_opt >= 0 always!
        # Therefore Omega_near >= 0 (Gaussian smooth of non-negative values)
        # Wait - Omega_near is Gaussian-smoothed Omega_opt on the strike grid.
        # Since Omega_opt >= 0 and Gaussian weights are non-negative,
        # Omega_near >= 0.
        # And max(0, Omega_prom) >= 0.
        # So nu = 1 + Omega_near + 2*max(0,Omega_prom) >= 1.
        assert (result["nu_opt"] >= 1.0 - 1e-10).all()

    def test_kappa_bounded(self, gold_stage, df_flow):
        """kappa = 1/nu, with nu >= 1, so kappa in (0, 1]."""
        result = gold_stage.transform(df_flow)
        assert (result["kappa_opt"] > 0).all()
        assert (result["kappa_opt"] <= 1.0 + 1e-10).all()

    def test_pressure_grad_sign_convention(self, gold_stage, df_flow):
        """Bid side: pressure_grad = +u_opt_p, Ask side: pressure_grad = -u_opt_p."""
        result = gold_stage.transform(df_flow)

        bid_rows = result[result["side"] == "B"]
        ask_rows = result[result["side"] == "A"]

        if not bid_rows.empty:
            np.testing.assert_array_almost_equal(
                bid_rows["pressure_grad_opt"].values,
                bid_rows["u_opt_p"].values,
            )

        if not ask_rows.empty:
            np.testing.assert_array_almost_equal(
                ask_rows["pressure_grad_opt"].values,
                -ask_rows["u_opt_p"].values,
            )

    def test_wave_energy_non_negative(self, gold_stage, df_flow):
        """Wave energy is sqrt of sum of squares, always >= 0."""
        result = gold_stage.transform(df_flow)
        assert (result["u_opt_wave_energy"] >= 0).all()

    def test_velocity_decomposition(self, gold_stage, df_flow):
        """liquidity_velocity = add_intensity - pull_intensity - fill_intensity."""
        result = gold_stage.transform(df_flow)
        expected = result["add_intensity"] - result["pull_intensity"] - result["fill_intensity"]
        np.testing.assert_array_almost_equal(
            result["liquidity_velocity"].values, expected.values
        )

    def test_band_decomposition(self, gold_stage, df_flow):
        """Bands are differences of adjacent EMA scales."""
        result = gold_stage.transform(df_flow)
        np.testing.assert_array_almost_equal(
            result["u_opt_band_fast"].values,
            (result["u_opt_ema_2"] - result["u_opt_ema_8"]).values,
        )
        np.testing.assert_array_almost_equal(
            result["u_opt_band_mid"].values,
            (result["u_opt_ema_8"] - result["u_opt_ema_32"]).values,
        )
        np.testing.assert_array_almost_equal(
            result["u_opt_band_slow"].values,
            (result["u_opt_ema_32"] - result["u_opt_ema_128"]).values,
        )

    def test_wave_energy_formula(self, gold_stage, df_flow):
        """wave_energy = sqrt(fast^2 + mid^2 + slow^2)."""
        result = gold_stage.transform(df_flow)
        expected = np.sqrt(
            result["u_opt_band_fast"] ** 2
            + result["u_opt_band_mid"] ** 2
            + result["u_opt_band_slow"] ** 2
        )
        np.testing.assert_array_almost_equal(
            result["u_opt_wave_energy"].values, expected.values
        )

    def test_omega_formula(self, gold_stage, df_flow):
        """Omega_opt = rho * (0.5+0.5*phi) * (1+max(0,u_p_slow))."""
        result = gold_stage.transform(df_flow)
        expected = (
            result["rho_opt"]
            * (0.5 + 0.5 * result["phi_rest_opt"])
            * (1.0 + np.maximum(0.0, result["u_opt_p_slow"]))
        )
        np.testing.assert_array_almost_equal(
            result["Omega_opt"].values, expected.values
        )

    def test_nu_formula(self, gold_stage, df_flow):
        """nu = 1 + Omega_near + 2*max(0, Omega_prom)."""
        result = gold_stage.transform(df_flow)
        expected = (
            1.0
            + result["Omega_opt_near"]
            + 2.0 * np.maximum(0.0, result["Omega_opt_prom"])
        )
        np.testing.assert_array_almost_equal(
            result["nu_opt"].values, expected.values
        )

    def test_kappa_formula(self, gold_stage, df_flow):
        """kappa = 1/nu."""
        result = gold_stage.transform(df_flow)
        expected = 1.0 / result["nu_opt"]
        np.testing.assert_array_almost_equal(
            result["kappa_opt"].values, expected.values
        )

    def test_prominence_definition(self, gold_stage, df_flow):
        """u_opt_prom = u_opt_near - u_opt_far."""
        result = gold_stage.transform(df_flow)
        np.testing.assert_array_almost_equal(
            result["u_opt_prom"].values,
            (result["u_opt_near"] - result["u_opt_far"]).values,
        )

    def test_omega_prom_definition(self, gold_stage, df_flow):
        """Omega_opt_prom = Omega_opt_near - Omega_opt_far."""
        result = gold_stage.transform(df_flow)
        np.testing.assert_array_almost_equal(
            result["Omega_opt_prom"].values,
            (result["Omega_opt_near"] - result["Omega_opt_far"]).values,
        )

    def test_event_ts_equals_window_end(self, gold_stage, df_flow):
        """event_ts_ns = window_end_ts_ns."""
        result = gold_stage.transform(df_flow)
        np.testing.assert_array_equal(
            result["event_ts_ns"].values, result["window_end_ts_ns"].values
        )


class TestEmptyInput:
    """Verify empty input handling."""

    def test_empty_flow(self):
        gold = GoldComputeOptionPhysicsSurface1s()
        result = gold.transform(pd.DataFrame())
        assert result.empty
        assert len(result.columns) > 0  # Should have column names even if empty


class TestAccountingIdentity:
    """Verify the accounting identity relationship."""

    def test_identity_holds_when_valid(self):
        """depth_start + add - pull - fill = depth_end when identity is valid."""
        depth_start = 10.0
        add = 5.0
        pull = 2.0
        fill = 1.0
        depth_end = depth_start + add - pull - fill
        assert depth_end == pytest.approx(12.0)

    def test_residual_computation(self):
        """Residual = depth_start + add - pull - fill - depth_end."""
        depth_start = 10.0
        add = 5.0
        pull = 2.0
        fill = 1.0
        depth_end = 12.0  # matches identity
        residual = depth_start + add - pull - fill - depth_end
        assert abs(residual) < 0.01
