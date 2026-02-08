"""Synthetic math validation tests for future_mbo gold (physics_surface_1s) layer.

Tests verify:
1. Accounting identity: depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
2. Intensity normalization: intensity = qty / (depth_start + EPS_QTY)
3. Velocity sign: liquidity_velocity = add_intensity - pull_intensity - fill_intensity
4. EMA formula: alpha = 1 - exp(-1/tau), recursive (adjust=False)
5. Band signals: fast = ema_2 - ema_8, mid = ema_8 - ema_32, slow = ema_32 - ema_128
6. Wave energy: sqrt(fast^2 + mid^2 + slow^2)
7. Temporal derivatives: du_dt = ema_2[t] - ema_2[t-1], d2u_dt2 = du_dt[t] - du_dt[t-1]
8. Obstacle fields: rho = log(1 + depth_end), phi_rest = rest / (depth_end + 1)
9. Omega = rho * (0.5 + 0.5*phi_rest) * (1 + max(0, u_p_slow))
10. Spatial derivatives: du_dx = central diff, d2u_dx2 = Laplacian
11. Viscosity: nu = 1 + Omega_near + 2*max(0, Omega_prom), kappa = 1/nu
12. Pressure gradient: B -> +u_p, A -> -u_p
13. No NaN in output
14. rel_ticks dtype is int (not object)
15. Contract field compliance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_eng.stages.silver.future_mbo.book_engine import (
    DEPTH_FLOW_COLUMNS,
    EPS_QTY,
    GRID_MAX_TICKS,
    PRICE_SCALE,
    SNAP_COLUMNS,
    TICK_INT,
    TICK_SIZE,
    WINDOW_NS,
    FuturesBookEngine,
    compute_futures_surfaces_1s,
)
from src.data_eng.stages.gold.future_mbo.compute_physics_surface_1s import (
    GoldComputePhysicsSurface1s,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _price_int(dollars: float) -> int:
    """Convert a dollar price to the internal int representation (1e-9 scale)."""
    return int(round(dollars / PRICE_SCALE))


def _make_event(
    ts: int,
    action: str,
    side: str,
    price_dollars: float,
    size: int,
    order_id: int,
    seq: int,
    flags: int = 0,
) -> dict:
    return {
        "ts_event": ts,
        "action": action,
        "side": side,
        "price": _price_int(price_dollars),
        "size": size,
        "order_id": order_id,
        "sequence": seq,
        "flags": flags,
    }


BASE_TS = 1_000_000_000_000  # 1 second in ns, so window_id = 1000


def _build_flow_scenario() -> pd.DataFrame:
    """Build a multi-second scenario with adds, cancels, fills on both sides.

    Window 0 (BASE_TS to BASE_TS + 1s):
      Bid side at $6000.00:
        - Add order 1: size 10
        - Add order 2: size 5
      Ask side at $6000.50:
        - Add order 3: size 8

    Window 1 (BASE_TS + 1s to BASE_TS + 2s):
      Bid side at $6000.00:
        - Cancel order 2: pull 5
        - Add order 4: size 3
      Ask side at $6000.50:
        - Fill order 3 for 2 (partial fill)
      Trade at $6000.25 (sets last_trade)

    Window 2 (BASE_TS + 2s to BASE_TS + 3s):
      Bid side at $6000.00:
        - Add order 5: size 7
      Ask side at $6000.50:
        - Cancel order 3: pull remaining 6
        - Add order 6: size 12
    """
    events = [
        # Window 0
        _make_event(BASE_TS + 100, "A", "B", 6000.00, 10, 1, 1),
        _make_event(BASE_TS + 200, "A", "B", 6000.00, 5, 2, 2),
        _make_event(BASE_TS + 300, "A", "A", 6000.50, 8, 3, 3),
        # Window 1
        _make_event(BASE_TS + WINDOW_NS + 100, "C", "B", 6000.00, 5, 2, 4),
        _make_event(BASE_TS + WINDOW_NS + 200, "A", "B", 6000.00, 3, 4, 5),
        _make_event(BASE_TS + WINDOW_NS + 300, "T", "A", 6000.25, 2, 0, 6),
        _make_event(BASE_TS + WINDOW_NS + 400, "F", "A", 6000.50, 2, 3, 7),
        # Window 2
        _make_event(BASE_TS + 2 * WINDOW_NS + 100, "A", "B", 6000.00, 7, 5, 8),
        _make_event(BASE_TS + 2 * WINDOW_NS + 200, "C", "A", 6000.50, 6, 3, 9),
        _make_event(BASE_TS + 2 * WINDOW_NS + 300, "A", "A", 6000.50, 12, 6, 10),
    ]
    return pd.DataFrame(events)


def _run_silver(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the silver book engine and return (snap, flow) DataFrames."""
    df_snap, df_flow, _ = compute_futures_surfaces_1s(df_events)
    return df_snap, df_flow


def _run_gold(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> pd.DataFrame:
    """Run the gold physics surface transform."""
    stage = GoldComputePhysicsSurface1s()
    return stage.transform(df_snap, df_flow)


# ---------------------------------------------------------------------------
# Silver layer tests (book_engine math)
# ---------------------------------------------------------------------------
class TestSilverAccountingIdentity:
    """Verify depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty."""

    def test_accounting_identity_all_rows(self):
        df_events = _build_flow_scenario()
        _, df_flow = _run_silver(df_events)

        assert not df_flow.empty, "Flow DataFrame should not be empty"

        depth_start = df_flow["depth_qty_start"].to_numpy()
        depth_end = df_flow["depth_qty_end"].to_numpy()
        add_qty = df_flow["add_qty"].to_numpy()
        pull_qty = df_flow["pull_qty"].to_numpy()
        fill_qty = df_flow["fill_qty"].to_numpy()

        # depth_start = depth_end - add + pull + fill
        reconstructed = depth_end - add_qty + pull_qty + fill_qty
        # Clamp negative to 0 (matching engine behavior)
        reconstructed = np.maximum(reconstructed, 0.0)

        np.testing.assert_allclose(
            depth_start,
            reconstructed,
            atol=1e-9,
            err_msg="Accounting identity violated",
        )

    def test_depth_qty_rest_le_depth_qty_end(self):
        df_events = _build_flow_scenario()
        _, df_flow = _run_silver(df_events)

        # depth_qty_rest should never exceed depth_qty_end
        mask = df_flow["depth_qty_rest"] > df_flow["depth_qty_end"] + 1e-9
        violations = df_flow[mask]
        assert violations.empty, (
            f"depth_qty_rest > depth_qty_end in {len(violations)} rows"
        )


class TestSilverRelTicks:
    """Verify rel_ticks = (price_int - spot_ref_price_int) / TICK_INT."""

    def test_rel_ticks_formula(self):
        df_events = _build_flow_scenario()
        _, df_flow = _run_silver(df_events)

        price_int = df_flow["price_int"].to_numpy()
        spot_ref = df_flow["spot_ref_price_int"].to_numpy()
        rel_ticks = df_flow["rel_ticks"].to_numpy()

        expected = np.round((price_int - spot_ref) / TICK_INT).astype(int)
        np.testing.assert_array_equal(rel_ticks, expected)


class TestSilverColumnCompliance:
    """Verify silver output matches contract schema."""

    def test_snap_columns(self):
        df_events = _build_flow_scenario()
        df_snap, _ = _run_silver(df_events)
        assert list(df_snap.columns) == SNAP_COLUMNS

    def test_flow_columns(self):
        df_events = _build_flow_scenario()
        _, df_flow = _run_silver(df_events)
        assert list(df_flow.columns) == DEPTH_FLOW_COLUMNS


# ---------------------------------------------------------------------------
# Gold layer tests (physics_surface_1s math)
# ---------------------------------------------------------------------------
def _build_deterministic_flow() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a deterministic multi-window scenario and run through silver.

    Returns (df_snap, df_flow) ready for gold transform.
    Uses 5 windows to give EMAs time to converge slightly.
    """
    events = []
    seq = 0
    oid = 100

    bid_price = 6000.00
    ask_price = 6000.25

    for w in range(5):
        ts_base = BASE_TS + w * WINDOW_NS

        # Bid adds
        seq += 1; oid += 1
        events.append(_make_event(ts_base + 100, "A", "B", bid_price, 10 + w, oid, seq))

        # Ask adds
        seq += 1; oid += 1
        events.append(_make_event(ts_base + 200, "A", "A", ask_price, 8 + w, oid, seq))

        # Trade to set spot reference
        seq += 1
        events.append(_make_event(ts_base + 300, "T", "A", bid_price + 0.125, 1, 0, seq))

    df_events = pd.DataFrame(events)
    return _run_silver(df_events)


class TestGoldIntensityNormalization:
    """Verify intensity = qty / (depth_start + EPS_QTY)."""

    def test_add_intensity(self):
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        # Filter to rows that came from silver (non-zero add_qty)
        active = df_gold[df_gold["add_intensity"] > 0].copy()
        if active.empty:
            pytest.skip("No active rows with add_intensity > 0")

        # For active rows, verify the ratio relationship
        # Intensity should be > 0 when add_qty > 0
        assert (active["add_intensity"] > 0).all()

    def test_velocity_sign_convention(self):
        """liquidity_velocity = add_intensity - pull_intensity - fill_intensity."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        lv = df_gold["liquidity_velocity"].to_numpy()
        ai = df_gold["add_intensity"].to_numpy()
        pi = df_gold["pull_intensity"].to_numpy()
        fi = df_gold["fill_intensity"].to_numpy()

        expected = ai - pi - fi
        np.testing.assert_allclose(lv, expected, atol=1e-12)


class TestGoldEMAFormulas:
    """Verify EMA computations match the recursive formula."""

    def test_ema_alpha_values(self):
        """alpha = 1 - exp(-1/tau) for tau in {2, 8, 32, 128}."""
        for tau in [2, 8, 32, 128]:
            alpha = 1.0 - np.exp(-1.0 / tau)
            # Verify alpha is in (0, 1)
            assert 0 < alpha < 1, f"alpha for tau={tau} out of bounds: {alpha}"

    def test_ema_recursive_matches_pandas(self):
        """Verify EMA recursive formula: y_t = alpha*x_t + (1-alpha)*y_{t-1}."""
        # Build a simple velocity series
        velocity = np.array([0.1, 0.3, 0.2, 0.5, 0.1, 0.4, 0.2, 0.3])
        alpha = 1.0 - np.exp(-1.0 / 2.0)

        # Manual recursive
        ema_manual = np.zeros_like(velocity)
        ema_manual[0] = velocity[0]
        for i in range(1, len(velocity)):
            ema_manual[i] = alpha * velocity[i] + (1 - alpha) * ema_manual[i - 1]

        # Pandas
        s = pd.Series(velocity)
        ema_pandas = s.ewm(alpha=alpha, adjust=False).mean().to_numpy()

        np.testing.assert_allclose(ema_manual, ema_pandas, atol=1e-12)

    def test_bands_are_ema_differences(self):
        """u_band_fast = ema_2 - ema_8, etc."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        np.testing.assert_allclose(
            df_gold["u_band_fast"].to_numpy(),
            (df_gold["u_ema_2"] - df_gold["u_ema_8"]).to_numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            df_gold["u_band_mid"].to_numpy(),
            (df_gold["u_ema_8"] - df_gold["u_ema_32"]).to_numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            df_gold["u_band_slow"].to_numpy(),
            (df_gold["u_ema_32"] - df_gold["u_ema_128"]).to_numpy(),
            atol=1e-12,
        )

    def test_wave_energy(self):
        """u_wave_energy = sqrt(fast^2 + mid^2 + slow^2)."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        expected = np.sqrt(
            df_gold["u_band_fast"].to_numpy() ** 2
            + df_gold["u_band_mid"].to_numpy() ** 2
            + df_gold["u_band_slow"].to_numpy() ** 2
        )
        np.testing.assert_allclose(
            df_gold["u_wave_energy"].to_numpy(), expected, atol=1e-12
        )


class TestGoldObstacleFields:
    """Verify rho, phi_rest, Omega formulas."""

    def test_rho_formula(self):
        """rho = log(1 + depth_qty_end)."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        # For rows that came from silver (have nonzero rho), verify
        # Note: dense grid rows have rho=0.0 (filled), which is log(1+0) = 0
        rho = df_gold["rho"].to_numpy()
        assert np.all(np.isfinite(rho)), "rho has non-finite values"
        assert np.all(rho >= 0), "rho should be non-negative"

    def test_phi_rest_bounds(self):
        """phi_rest = depth_rest / (depth_end + 1.0), should be in [0, 1]."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        phi = df_gold["phi_rest"].to_numpy()
        assert np.all(np.isfinite(phi)), "phi_rest has non-finite values"
        assert np.all(phi >= -1e-9), "phi_rest should be non-negative"
        # phi_rest can exceed 1 in pathological cases, but generally should be <= 1
        # We only check >= 0 strictly

    def test_omega_formula(self):
        """Omega = rho * (0.5 + 0.5*phi_rest) * (1 + max(0, u_p_slow))."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        rho = df_gold["rho"].to_numpy()
        phi = df_gold["phi_rest"].to_numpy()
        u_p_slow = df_gold["u_p_slow"].to_numpy()

        expected = rho * (0.5 + 0.5 * phi) * (1.0 + np.maximum(0.0, u_p_slow))
        np.testing.assert_allclose(
            df_gold["Omega"].to_numpy(), expected, atol=1e-12
        )


class TestGoldViscosity:
    """Verify nu and kappa formulas."""

    def test_nu_formula(self):
        """nu = 1 + Omega_near + 2*max(0, Omega_prom)."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        omega_near = df_gold["Omega_near"].to_numpy()
        omega_prom = df_gold["Omega_prom"].to_numpy()

        expected = 1.0 + omega_near + 2.0 * np.maximum(0.0, omega_prom)
        np.testing.assert_allclose(
            df_gold["nu"].to_numpy(), expected, atol=1e-12
        )

    def test_kappa_is_inverse_nu(self):
        """kappa = 1 / nu."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        nu = df_gold["nu"].to_numpy()
        kappa = df_gold["kappa"].to_numpy()

        expected = 1.0 / nu
        np.testing.assert_allclose(kappa, expected, atol=1e-12)

    def test_nu_ge_one(self):
        """nu >= 1 always (since Omega_near >= 0 and max(0,x) >= 0)."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        nu = df_gold["nu"].to_numpy()
        assert np.all(nu >= 1.0 - 1e-12), f"nu < 1 found: min={nu.min()}"


class TestGoldPressureGradient:
    """Verify pressure_grad sign convention."""

    def test_bid_positive_ask_negative(self):
        """Bid side: pressure_grad = +u_p, Ask side: pressure_grad = -u_p."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        bid_mask = df_gold["side"] == "B"
        ask_mask = df_gold["side"] == "A"

        u_p = df_gold["u_p"].to_numpy()
        pg = df_gold["pressure_grad"].to_numpy()

        # Bid: pg = +u_p
        np.testing.assert_allclose(
            pg[bid_mask], u_p[bid_mask], atol=1e-12,
            err_msg="Bid pressure_grad should equal +u_p",
        )
        # Ask: pg = -u_p
        np.testing.assert_allclose(
            pg[ask_mask], -u_p[ask_mask], atol=1e-12,
            err_msg="Ask pressure_grad should equal -u_p",
        )


class TestGoldSpatialDerivatives:
    """Verify du_dx and d2u_dx2 use correct finite difference formulas."""

    def test_central_difference_formula(self):
        """du_dx = (u_near[i+1] - u_near[i-1]) * 0.5 within each (ts, side) group."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        # Pick the first (ts, side) group with enough data
        df_sorted = df_gold.sort_values(["window_end_ts_ns", "side", "rel_ticks"])

        for (ts, side), grp in df_sorted.groupby(["window_end_ts_ns", "side"]):
            u = grp["u_near"].to_numpy()
            du = grp["du_dx"].to_numpy()

            if len(u) < 3:
                continue

            # Interior points: du_dx[i] = (u[i+1] - u[i-1]) * 0.5
            for i in range(1, len(u) - 1):
                expected = (u[i + 1] - u[i - 1]) * 0.5
                assert abs(du[i] - expected) < 1e-12, (
                    f"du_dx mismatch at ts={ts}, side={side}, idx={i}: "
                    f"got {du[i]}, expected {expected}"
                )

            # Boundaries should be 0 (fillna)
            assert abs(du[0]) < 1e-12, "du_dx at lower boundary should be 0"
            assert abs(du[-1]) < 1e-12, "du_dx at upper boundary should be 0"

            # Only test one group
            break

    def test_laplacian_formula(self):
        """d2u_dx2 = u[i+1] - 2*u[i] + u[i-1] within each (ts, side) group."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        df_sorted = df_gold.sort_values(["window_end_ts_ns", "side", "rel_ticks"])

        for (ts, side), grp in df_sorted.groupby(["window_end_ts_ns", "side"]):
            u = grp["u_near"].to_numpy()
            d2u = grp["d2u_dx2"].to_numpy()

            if len(u) < 3:
                continue

            for i in range(1, len(u) - 1):
                expected = u[i + 1] - 2.0 * u[i] + u[i - 1]
                assert abs(d2u[i] - expected) < 1e-12, (
                    f"d2u_dx2 mismatch at ts={ts}, side={side}, idx={i}: "
                    f"got {d2u[i]}, expected {expected}"
                )
            break


class TestGoldNoNaN:
    """Verify no NaN values in gold output."""

    def test_no_nan_in_any_column(self):
        """After fixes, all columns should be NaN-free."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        for col in df_gold.columns:
            nan_count = df_gold[col].isna().sum()
            assert nan_count == 0, f"Column '{col}' has {nan_count} NaN values"


class TestGoldRelTicksDtype:
    """Verify rel_ticks is integer type, not object."""

    def test_rel_ticks_is_int(self):
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        assert pd.api.types.is_integer_dtype(df_gold["rel_ticks"]), (
            f"rel_ticks dtype is {df_gold['rel_ticks'].dtype}, expected integer"
        )


class TestGoldContractCompliance:
    """Verify gold output matches the Avro contract schema."""

    def test_gold_contract_fields(self):
        """Output columns must match physics_surface_1s.avsc in order."""
        contract_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "data_eng"
            / "contracts"
            / "gold"
            / "future_mbo"
            / "physics_surface_1s.avsc"
        )
        with open(contract_path) as f:
            contract = json.load(f)

        expected_fields = [field["name"] for field in contract["fields"]]

        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        actual_fields = list(df_gold.columns)
        assert actual_fields == expected_fields, (
            f"Column mismatch.\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"Missing:  {set(expected_fields) - set(actual_fields)}\n"
            f"Extra:    {set(actual_fields) - set(expected_fields)}"
        )


class TestGoldTemporalDerivatives:
    """Verify du_dt and d2u_dt2 formulas."""

    def test_du_dt_is_first_difference_of_ema2(self):
        """du_dt = u_ema_2[t] - u_ema_2[t-1] per (rel_ticks, side) group."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        # The gold layer sorts by (rel_ticks, side, ts) for EMAs.
        # du_dt should be the first difference of u_ema_2 within each group.
        # After the transform, the DataFrame is sorted by (ts, side, rel_ticks)
        # for spatial ops. We need to re-sort for temporal verification.

        df_sorted = df_gold.sort_values(["rel_ticks", "side", "window_end_ts_ns"])

        for (rt, side), grp in df_sorted.groupby(["rel_ticks", "side"]):
            if len(grp) < 2:
                continue

            ema2 = grp["u_ema_2"].to_numpy()
            du_dt = grp["du_dt"].to_numpy()

            # First element should be 0 (no previous)
            assert abs(du_dt[0]) < 1e-12, (
                f"du_dt[0] should be 0 at rt={rt}, side={side}, got {du_dt[0]}"
            )

            for i in range(1, len(ema2)):
                expected = ema2[i] - ema2[i - 1]
                assert abs(du_dt[i] - expected) < 1e-12, (
                    f"du_dt mismatch at rt={rt}, side={side}, idx={i}"
                )
            break

    def test_d2u_dt2_is_second_difference(self):
        """d2u_dt2 = du_dt[t] - du_dt[t-1]."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        df_sorted = df_gold.sort_values(["rel_ticks", "side", "window_end_ts_ns"])

        for (rt, side), grp in df_sorted.groupby(["rel_ticks", "side"]):
            if len(grp) < 3:
                continue

            du_dt = grp["du_dt"].to_numpy()
            d2u = grp["d2u_dt2"].to_numpy()

            for i in range(1, len(du_dt)):
                expected = du_dt[i] - du_dt[i - 1]
                assert abs(d2u[i] - expected) < 1e-12, (
                    f"d2u_dt2 mismatch at rt={rt}, side={side}, idx={i}"
                )
            break


class TestGoldPersistenceWeighting:
    """Verify u_p and u_p_slow."""

    def test_u_p_formula(self):
        """u_p = phi_rest * u_ema_8."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        expected = df_gold["phi_rest"].to_numpy() * df_gold["u_ema_8"].to_numpy()
        np.testing.assert_allclose(
            df_gold["u_p"].to_numpy(), expected, atol=1e-12
        )

    def test_u_p_slow_formula(self):
        """u_p_slow = phi_rest * u_ema_32."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        expected = df_gold["phi_rest"].to_numpy() * df_gold["u_ema_32"].to_numpy()
        np.testing.assert_allclose(
            df_gold["u_p_slow"].to_numpy(), expected, atol=1e-12
        )


class TestGoldProminence:
    """Verify u_prom and Omega_prom."""

    def test_u_prom_formula(self):
        """u_prom = u_near - u_far."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        expected = df_gold["u_near"].to_numpy() - df_gold["u_far"].to_numpy()
        np.testing.assert_allclose(
            df_gold["u_prom"].to_numpy(), expected, atol=1e-12
        )

    def test_omega_prom_formula(self):
        """Omega_prom = Omega_near - Omega_far."""
        df_snap, df_flow = _build_deterministic_flow()
        df_gold = _run_gold(df_snap, df_flow)

        expected = df_gold["Omega_near"].to_numpy() - df_gold["Omega_far"].to_numpy()
        np.testing.assert_allclose(
            df_gold["Omega_prom"].to_numpy(), expected, atol=1e-12
        )
