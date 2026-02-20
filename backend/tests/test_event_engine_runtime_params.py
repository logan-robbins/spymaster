from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.event_engine import AbsoluteTickEngine


def _primed_engine(**kwargs: float) -> AbsoluteTickEngine:
    engine = AbsoluteTickEngine(n_ticks=128, tick_int=1, **kwargs)
    # Establish anchor around [50, 51].
    engine.update(
        ts_ns=1_000_000_000,
        action="A",
        side="B",
        price_int=50,
        size=10,
        order_id=1,
        flags=0,
    )
    engine.update(
        ts_ns=1_000_000_000,
        action="A",
        side="A",
        price_int=51,
        size=10,
        order_id=2,
        flags=0,
    )
    return engine


def _drive_add_sequence(engine: AbsoluteTickEngine) -> float:
    engine.update(
        ts_ns=2_000_000_000,
        action="A",
        side="B",
        price_int=50,
        size=5,
        order_id=3,
        flags=0,
    )
    engine.update(
        ts_ns=3_000_000_000,
        action="A",
        side="B",
        price_int=50,
        size=5,
        order_id=4,
        flags=0,
    )
    idx = engine._price_to_idx(50)
    assert idx is not None
    arrays = engine.grid_snapshot_arrays()
    return float(arrays["pressure_variant"][idx])


def test_runtime_force_coefficients_change_pressure_magnitude() -> None:
    e1 = _primed_engine(c2_v_rest_pos=0.0, c3_a_add=0.0, c1_v_add=1.0)
    e2 = _primed_engine(c2_v_rest_pos=0.0, c3_a_add=0.0, c1_v_add=2.0)

    p1 = _drive_add_sequence(e1)
    p2 = _drive_add_sequence(e2)

    assert p1 > 0.0
    assert p2 > p1
    assert p2 / p1 == pytest.approx(2.0, rel=1e-6)


def test_runtime_tau_velocity_changes_response_speed() -> None:
    fast = _primed_engine(c2_v_rest_pos=0.0, c3_a_add=0.0, tau_velocity=1.0)
    slow = _primed_engine(c2_v_rest_pos=0.0, c3_a_add=0.0, tau_velocity=8.0)

    p_fast = _drive_add_sequence(fast)
    p_slow = _drive_add_sequence(slow)

    assert p_fast > p_slow
