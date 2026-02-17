"""Comprehensive math validation for the vacuum pressure engine.

Tests cover: derivative chain, composite formula, force model,
book stress, spectrum NaN guard, decay, fills, and modifies.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.event_engine import (
    AbsoluteTickEngine,
    TAU_ACCELERATION,
    TAU_JERK,
    TAU_REST_DECAY,
    TAU_VELOCITY,
    _ema_alpha,
    _update_derivative_chain,
    _update_derivative_chain_from_delta,
)
from src.vacuum_pressure.spectrum import IndependentCellSpectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_with_anchor(n_ticks: int = 100) -> AbsoluteTickEngine:
    """Create an engine with anchor established via bid+ask at prices 50/51."""
    engine = AbsoluteTickEngine(n_ticks=n_ticks, tick_int=1)
    engine.update(ts_ns=1_000_000_000, action="A", side="B",
                  price_int=50, size=10, order_id=1, flags=0)
    engine.update(ts_ns=1_000_000_000, action="A", side="A",
                  price_int=51, size=10, order_id=2, flags=0)
    assert engine.anchor_tick_idx >= 0
    return engine


def _small_spectrum(**overrides) -> IndependentCellSpectrum:
    """Create a small spectrum kernel for testing."""
    defaults = dict(
        n_cells=3,
        windows=[2, 4],
        rollup_weights=[1.0, 1.0],
        derivative_weights=[0.55, 0.30, 0.15],
        tanh_scale=3.0,
        neutral_threshold=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_ms=[100, 200],
        default_dt_s=0.1,
    )
    defaults.update(overrides)
    return IndependentCellSpectrum(**defaults)


# ===================================================================
# Group 1: Derivative chain analytical verification
# ===================================================================


def test_ema_alpha_known_values() -> None:
    """Verify EMA alpha against hand-computed exp() values."""
    # dt=tau: alpha = 1 - exp(-1) = 0.63212...
    assert _ema_alpha(2.0, 2.0) == pytest.approx(1 - math.exp(-1), abs=1e-10)

    # dt=0: guard returns 0
    assert _ema_alpha(0.0, 2.0) == 0.0

    # dt/tau > 50: clamp to 1.0
    assert _ema_alpha(100.0, 1.0) == 1.0

    # dt=1, tau=2: alpha = 1 - exp(-0.5) = 0.39347...
    assert _ema_alpha(1.0, 2.0) == pytest.approx(1 - math.exp(-0.5), abs=1e-10)

    # Negative dt: guard returns 0
    assert _ema_alpha(-1.0, 2.0) == 0.0


def test_derivative_chain_from_delta_single_step() -> None:
    """Verify derivative chain with delta=10, dt=1s from zeros."""
    v, a, j = _update_derivative_chain_from_delta(
        delta=10.0, dt_s=1.0,
        v_prev=0.0, a_prev=0.0, j_prev=0.0,
    )

    # rate = 10/1 = 10
    # alpha_v = 1 - exp(-1/2) = 0.39347
    alpha_v = 1 - math.exp(-1.0 / TAU_VELOCITY)
    expected_v = alpha_v * 10.0
    assert v == pytest.approx(expected_v, abs=1e-6)

    # dv_rate = (v_new - 0) / 1 = v_new
    alpha_a = 1 - math.exp(-1.0 / TAU_ACCELERATION)
    expected_a = alpha_a * expected_v
    assert a == pytest.approx(expected_a, abs=1e-6)

    # da_rate = (a_new - 0) / 1 = a_new
    alpha_j = 1 - math.exp(-1.0 / TAU_JERK)
    expected_j = alpha_j * expected_a
    assert j == pytest.approx(expected_j, abs=1e-6)

    # Sanity: values positive and decreasing magnitude
    assert v > a > j > 0


def test_derivative_chain_from_value_single_step() -> None:
    """Value-based chain (0→10, dt=1s) matches delta-based chain (delta=10, dt=1s)."""
    v_delta, a_delta, j_delta = _update_derivative_chain_from_delta(
        delta=10.0, dt_s=1.0,
        v_prev=0.0, a_prev=0.0, j_prev=0.0,
    )
    v_value, a_value, j_value = _update_derivative_chain(
        prev_value=0.0, new_value=10.0, dt_s=1.0,
        v_prev=0.0, a_prev=0.0, j_prev=0.0,
    )

    assert v_value == pytest.approx(v_delta, abs=1e-12)
    assert a_value == pytest.approx(a_delta, abs=1e-12)
    assert j_value == pytest.approx(j_delta, abs=1e-12)


def test_derivative_chain_two_steps_accumulation() -> None:
    """Second step with delta=0 decays velocity toward zero."""
    v1, a1, j1 = _update_derivative_chain_from_delta(
        delta=10.0, dt_s=1.0,
        v_prev=0.0, a_prev=0.0, j_prev=0.0,
    )
    v2, a2, j2 = _update_derivative_chain_from_delta(
        delta=0.0, dt_s=1.0,
        v_prev=v1, a_prev=a1, j_prev=j1,
    )

    # Velocity should decay (rate=0 blended with prior)
    alpha_v = 1 - math.exp(-1.0 / TAU_VELOCITY)
    expected_v2 = alpha_v * 0.0 + (1 - alpha_v) * v1
    assert v2 == pytest.approx(expected_v2, abs=1e-6)
    assert v2 < v1

    # All remain finite
    assert math.isfinite(v2) and math.isfinite(a2) and math.isfinite(j2)


# ===================================================================
# Group 2: Composite formula (after fix)
# ===================================================================


def test_composite_bounded_opposite_sign() -> None:
    """P=5, V=-5 produces composite ≈ 1.0 (not ±∞)."""
    kernel = _small_spectrum()

    p = np.array([5.0, -5.0, 0.0], dtype=np.float64)
    v = np.array([-5.0, 5.0, 0.0], dtype=np.float64)
    out = kernel.update(100_000_000, p, v)

    composite = kernel.latest_composite
    # (5-(-5)) / (|5|+|-5|+eps) = 10/10 ≈ 1.0
    assert composite[0] == pytest.approx(1.0, abs=1e-9)
    # (-5-5) / (|-5|+|5|+eps) = -10/10 ≈ -1.0
    assert composite[1] == pytest.approx(-1.0, abs=1e-9)
    # (0-0) / (0+0+eps) ≈ 0
    assert composite[2] == pytest.approx(0.0, abs=1e-9)

    assert np.all(np.abs(composite) <= 1.0 + 1e-9)
    assert np.isfinite(out.score).all()


def test_composite_near_cancellation() -> None:
    """P ≈ -V: denominator uses |P|+|V|, stays bounded."""
    kernel = _small_spectrum(n_cells=1, windows=[2], rollup_weights=[1.0])

    p = np.array([5.0], dtype=np.float64)
    v = np.array([-5.0001], dtype=np.float64)
    out = kernel.update(100_000_000, p, v)

    composite = kernel.latest_composite
    # (5 - (-5.0001)) / (5 + 5.0001 + eps) = 10.0001 / 10.0001 ≈ 1.0
    assert np.abs(composite[0]) <= 1.0 + 1e-9
    assert np.isfinite(composite[0])
    assert np.isfinite(out.score).all()


def test_composite_all_zero() -> None:
    """P=V=0 → composite = 0."""
    kernel = _small_spectrum()

    p = np.zeros(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    out = kernel.update(100_000_000, p, v)

    composite = kernel.latest_composite
    np.testing.assert_allclose(composite, 0.0, atol=1e-9)
    assert np.isfinite(out.score).all()


# ===================================================================
# Group 3: Force model edge cases
# ===================================================================


def test_force_model_all_zero_derivatives() -> None:
    """No time-separated events → pressure=vacuum=0."""
    engine = AbsoluteTickEngine(n_ticks=20, tick_int=1)

    # Two events at same timestamp: dt=0, derivatives skip
    engine.update(ts_ns=1_000_000_000, action="A", side="B",
                  price_int=10, size=5, order_id=1, flags=0)
    engine.update(ts_ns=1_000_000_000, action="A", side="A",
                  price_int=11, size=5, order_id=2, flags=0)

    arrays = engine.grid_snapshot_arrays()
    assert float(arrays["pressure_variant"].sum()) == 0.0
    assert float(arrays["vacuum_variant"].sum()) == 0.0


def test_force_model_pure_add_velocity() -> None:
    """Two adds with dt=1s → pressure > 0, vacuum ≈ 0 at that tick."""
    engine = _engine_with_anchor()

    # Third add at price=50, 1s later
    engine.update(ts_ns=2_000_000_000, action="A", side="B",
                  price_int=50, size=5, order_id=3, flags=0)
    # Fourth add at same price, 1s later
    engine.update(ts_ns=3_000_000_000, action="A", side="B",
                  price_int=50, size=5, order_id=4, flags=0)

    idx = engine._price_to_idx(50)
    assert idx is not None
    arrays = engine.grid_snapshot_arrays()

    assert arrays["pressure_variant"][idx] > 0.0
    assert arrays["vacuum_variant"][idx] == pytest.approx(0.0, abs=1e-12)


def test_force_model_pure_cancel_velocity() -> None:
    """Add then cancel 1s later → pressure ≈ 0, vacuum > 0 at that tick."""
    engine = _engine_with_anchor()

    # Add at price=50 already done (order_id=1). Touch tick to set timestamp.
    engine.update(ts_ns=2_000_000_000, action="A", side="B",
                  price_int=50, size=5, order_id=3, flags=0)
    # Cancel it 1s later
    engine.update(ts_ns=3_000_000_000, action="C", side="B",
                  price_int=50, size=0, order_id=3, flags=0)

    idx = engine._price_to_idx(50)
    assert idx is not None
    arrays = engine.grid_snapshot_arrays()

    # Vacuum should be positive (pull velocity > 0)
    assert arrays["vacuum_variant"][idx] > 0.0


def test_force_model_tiny_dt() -> None:
    """1ns gap: derivatives near zero, no blow-up."""
    engine = _engine_with_anchor()

    engine.update(ts_ns=2_000_000_000, action="A", side="B",
                  price_int=50, size=5, order_id=3, flags=0)
    # 1ns later
    engine.update(ts_ns=2_000_000_001, action="A", side="B",
                  price_int=50, size=5, order_id=4, flags=0)

    idx = engine._price_to_idx(50)
    arrays = engine.grid_snapshot_arrays()

    # alpha ≈ 5e-10 (essentially zero), so derivatives stay near zero
    assert np.isfinite(arrays["v_add"][idx])
    assert np.isfinite(arrays["pressure_variant"][idx])
    assert np.isfinite(arrays["vacuum_variant"][idx])


# ===================================================================
# Group 4: Book stress test
# ===================================================================


def test_book_order_count_bounded_after_cycles() -> None:
    """1000 add/cancel cycles → order_count returns to baseline."""
    engine = _engine_with_anchor()
    baseline = engine.order_count  # 2 orders (bid+ask for anchor)

    for i in range(1000):
        oid = 100 + i
        engine.update(ts_ns=2_000_000_000 + i * 1000,
                      action="A", side="B", price_int=49,
                      size=1, order_id=oid, flags=0)
        engine.update(ts_ns=2_000_000_000 + i * 1000 + 500,
                      action="C", side="B", price_int=49,
                      size=0, order_id=oid, flags=0)

    assert engine.order_count == baseline

    metrics = engine.book_metrics()
    assert metrics["order_count"] == baseline


def test_book_order_count_net_adds() -> None:
    """500 adds, 250 cancels → order_count = baseline + 250."""
    engine = _engine_with_anchor()
    baseline = engine.order_count

    oids = list(range(100, 600))
    for oid in oids:
        engine.update(ts_ns=2_000_000_000, action="A", side="B",
                      price_int=49, size=1, order_id=oid, flags=0)

    # Cancel first 250
    for oid in oids[:250]:
        engine.update(ts_ns=3_000_000_000, action="C", side="B",
                      price_int=49, size=0, order_id=oid, flags=0)

    assert engine.order_count == baseline + 250

    metrics = engine.book_metrics()
    assert metrics["total_bid_qty"] > 0


# ===================================================================
# Group 5: Spectrum NaN guard
# ===================================================================


def test_spectrum_score_never_nan_extreme_inputs() -> None:
    """±1e15 pressure/vacuum → finite score in [-1,1]."""
    kernel = _small_spectrum()

    extremes = [
        (np.array([1e15, -1e15, 0.0]), np.array([-1e15, 1e15, 0.0])),
        (np.array([1e15, 1e15, 1e15]), np.array([1e15, 1e15, 1e15])),
        (np.array([-1e15, -1e15, -1e15]), np.array([-1e15, -1e15, -1e15])),
    ]

    for i, (p, v) in enumerate(extremes):
        out = kernel.update((i + 1) * 100_000_000, p, v)
        assert np.isfinite(out.score).all(), f"NaN in score at iteration {i}"
        assert np.all(np.abs(out.score) <= 1.0 + 1e-9)


def test_spectrum_projection_never_nan() -> None:
    """20 bins alternating extremes → all projections finite."""
    kernel = _small_spectrum()

    for i in range(20):
        sign = 1.0 if i % 2 == 0 else -1.0
        p = np.array([sign * 1e10, 0.0, -sign * 1e10], dtype=np.float64)
        v = np.array([-sign * 1e10, 0.0, sign * 1e10], dtype=np.float64)
        out = kernel.update((i + 1) * 100_000_000, p, v)

    for horizon_ms, proj in out.projected_score_by_horizon.items():
        assert np.isfinite(proj).all(), f"NaN in proj_score_h{horizon_ms}"
        assert np.all(np.abs(proj) <= 1.0 + 1e-9)


def test_spectrum_constant_then_spike() -> None:
    """10 constant + 1 spike → finite score in [-1,1]."""
    kernel = _small_spectrum()

    base_p = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    base_v = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    for i in range(10):
        kernel.update((i + 1) * 100_000_000, base_p, base_v)

    # Spike
    spike_p = np.array([1e12, 1.0, 1.0], dtype=np.float64)
    out = kernel.update(11 * 100_000_000, spike_p, base_v)

    assert np.isfinite(out.score).all()
    assert np.all(np.abs(out.score) <= 1.0 + 1e-9)


# ===================================================================
# Group 6: Decay verification
# ===================================================================


def test_mass_decay_at_tau_boundary() -> None:
    """Mass decays to exp(-1) ≈ 0.3679 of original after TAU_REST_DECAY seconds."""
    engine = _engine_with_anchor()

    # Add at price 50, size 100 (adds to existing 10)
    engine.update(ts_ns=2_000_000_000, action="A", side="B",
                  price_int=50, size=100, order_id=10, flags=0)

    idx = engine._price_to_idx(50)
    assert idx is not None
    mass_after_add = float(engine._add_mass[idx])
    assert mass_after_add > 0

    # Send a zero-delta event TAU_REST_DECAY seconds later
    tau_ns = int(TAU_REST_DECAY * 1e9)
    engine.update(ts_ns=2_000_000_000 + tau_ns, action="A", side="B",
                  price_int=50, size=0, order_id=11, flags=0)

    mass_after_decay = float(engine._add_mass[idx])
    expected = mass_after_add * math.exp(-1.0)
    assert mass_after_decay == pytest.approx(expected, abs=1e-6)


def test_mass_decay_multiple_intervals() -> None:
    """Verify mass at t=10, 20, 30s matches exp(-t/30)."""
    engine = _engine_with_anchor()

    engine.update(ts_ns=2_000_000_000, action="A", side="B",
                  price_int=50, size=100, order_id=10, flags=0)

    idx = engine._price_to_idx(50)
    assert idx is not None
    initial_mass = float(engine._add_mass[idx])

    checkpoints = [10.0, 20.0, 30.0]
    prev_ts = 2_000_000_000

    for dt_s in checkpoints:
        ts = prev_ts + int(dt_s * 1e9)
        oid = 20 + int(dt_s)
        engine.update(ts_ns=ts, action="A", side="B",
                      price_int=50, size=0, order_id=oid, flags=0)

        mass = float(engine._add_mass[idx])
        total_elapsed = (ts - 2_000_000_000) / 1e9
        expected = initial_mass * math.exp(-total_elapsed / TAU_REST_DECAY)
        assert mass == pytest.approx(expected, abs=1e-4), \
            f"at t={total_elapsed}s: got {mass}, expected {expected}"
        prev_ts = ts


def test_passive_time_advance_decays_untouched_tick_state() -> None:
    """advance_time() decays active tick state even without new events."""
    engine = _engine_with_anchor(n_ticks=200)

    # Build positive state at price 50.
    engine.update(
        ts_ns=2_000_000_000,
        action="A",
        side="B",
        price_int=50,
        size=5,
        order_id=10,
        flags=0,
    )
    engine.update(
        ts_ns=3_000_000_000,
        action="A",
        side="B",
        price_int=50,
        size=5,
        order_id=11,
        flags=0,
    )

    idx = engine._price_to_idx(50)
    assert idx is not None

    before = engine.grid_snapshot_arrays()
    v_add_before = float(before["v_add"][idx])
    add_mass_before = float(before["add_mass"][idx])
    pressure_before = float(before["pressure_variant"][idx])
    last_event_before = int(before["last_event_id"][idx])

    assert v_add_before > 0.0
    assert add_mass_before > 0.0
    assert pressure_before > 0.0

    engine.advance_time(63_000_000_000)

    after = engine.grid_snapshot_arrays()
    v_add_after = float(after["v_add"][idx])
    add_mass_after = float(after["add_mass"][idx])
    pressure_after = float(after["pressure_variant"][idx])

    assert 0.0 <= v_add_after < v_add_before
    assert 0.0 <= add_mass_after < add_mass_before
    assert 0.0 <= pressure_after < pressure_before
    assert int(after["last_event_id"][idx]) == last_event_before
    assert int(engine._last_ts_ns[idx]) == 63_000_000_000


# ===================================================================
# Group 7: Fill operations
# ===================================================================


def test_fill_partial_reduces_depth() -> None:
    """Fill 3 of 10 → rest_depth drops from 10 to 7."""
    engine = _engine_with_anchor()

    idx = engine._price_to_idx(50)
    assert idx is not None
    assert engine._rest_depth[idx] == 10.0

    # Partial fill of order_id=1 (bid at 50, size=10)
    engine.update(ts_ns=2_000_000_000, action="F", side="B",
                  price_int=50, size=3, order_id=1, flags=0)

    assert engine._rest_depth[idx] == 7.0
    assert engine.order_count == 2  # order still exists with reduced qty
    assert engine._orders[1].qty == 7


def test_fill_complete_removes_order() -> None:
    """Fill 10 of 10 → order removed, depth = 0 at that price."""
    engine = _engine_with_anchor()

    idx = engine._price_to_idx(50)
    assert idx is not None

    # Complete fill
    engine.update(ts_ns=2_000_000_000, action="F", side="B",
                  price_int=50, size=10, order_id=1, flags=0)

    assert engine._rest_depth[idx] == 0.0
    assert 1 not in engine._orders
    assert engine.order_count == 1  # only ask order remains


# ===================================================================
# Group 8: Modify operations
# ===================================================================


def test_modify_same_price_size_increase() -> None:
    """Modify size 10→15 at same price → add_delta=5, rest_depth=15."""
    engine = _engine_with_anchor()

    idx = engine._price_to_idx(50)
    assert idx is not None
    assert engine._rest_depth[idx] == 10.0

    engine.update(ts_ns=2_000_000_000, action="M", side="B",
                  price_int=50, size=15, order_id=1, flags=0)

    assert engine._rest_depth[idx] == 15.0
    assert engine._orders[1].qty == 15


def test_modify_price_change_touches_both_ticks() -> None:
    """Modify price 50→55 → pull at 50, add at 55."""
    engine = _engine_with_anchor()

    idx_50 = engine._price_to_idx(50)
    idx_55 = engine._price_to_idx(55)
    assert idx_50 is not None and idx_55 is not None

    # Before modify: depth at 50 = 10, depth at 55 = 0
    assert engine._rest_depth[idx_50] == 10.0
    assert engine._rest_depth[idx_55] == 0.0

    # Modify order 1: move from price=50 to price=55, size=10
    engine.update(ts_ns=2_000_000_000, action="M", side="B",
                  price_int=55, size=10, order_id=1, flags=0)

    assert engine._rest_depth[idx_50] == 0.0
    assert engine._rest_depth[idx_55] == 10.0
    assert engine._orders[1].price_int == 55
