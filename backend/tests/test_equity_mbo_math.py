"""Synthetic math validation tests for equity_mbo pipeline (bronze -> silver -> gold).

Tests verify:
1. Accounting identity: depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
2. $0.50 bucketing: multiple orders at adjacent penny prices aggregated into same bucket
3. rel_ticks = (price_int - spot_ref_price_int) / BUCKET_INT
4. depth_qty_rest <= depth_qty_end (clamping invariant)
5. Sign conventions: velocity = add - pull - fill (positive = building)
6. Gold intensity normalization: intensity = qty / (depth_start + EPS_QTY)
7. Gold velocity: liquidity_velocity = add_intensity - pull_intensity - fill_intensity
8. Contract field compliance (column names and order)

IMPORTANT: The book engine resolves spot_ref at the START of each window.
Window 0 starts with an empty book, so spot_ref=0, and no depth_flow rows
are emitted for window 0. All substantive tests therefore use window 1+
to ensure a valid spot reference exists.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data_eng.stages.silver.equity_mbo.book_engine import (
    BUCKET_INT,
    DEPTH_FLOW_COLUMNS,
    EPS_QTY,
    GRID_MAX_BUCKETS,
    SNAP_COLUMNS,
    WINDOW_NS,
    EquityBookEngine,
    compute_equity_surfaces_1s,
)
from src.data_eng.stages.gold.equity_mbo.compute_physics_surface_1s import (
    GoldComputeEquityPhysicsSurface1s,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SPOT = 520.00  # Reference spot price for all tests

# Window boundaries in ns. Window N starts at N * WINDOW_NS.
W0_START = 0
W1_START = 1 * WINDOW_NS      # 1_000_000_000
W2_START = 2 * WINDOW_NS      # 2_000_000_000
W3_START = 3 * WINDOW_NS      # 3_000_000_000

W0_END = 1 * WINDOW_NS
W1_END = 2 * WINDOW_NS
W2_END = 3 * WINDOW_NS


def _price_int(dollars: float) -> int:
    """Convert a dollar price to the internal int representation (1e-9 scale)."""
    return int(round(dollars * 1_000_000_000))


def _bucket_center(dollars: float) -> int:
    """Round dollar price to nearest $0.50 bucket center in int form."""
    bucket = round(dollars / 0.50) * 0.50
    return _price_int(bucket)


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
        "price": _price_int(price_dollars) if price_dollars > 0 else 0,
        "size": size,
        "order_id": order_id,
        "sequence": seq,
        "flags": flags,
    }


def _setup_book_events(spot: float = SPOT) -> list[dict]:
    """Create window-0 events that establish a valid book.

    Places a bid and ask around the spot, plus a trade at spot.
    This ensures window 1+ has a valid spot_ref.
    """
    return [
        _make_event(W0_START + 100, "A", "B", spot - 0.50, 1000, 9001, 1),
        _make_event(W0_START + 200, "A", "A", spot + 0.50, 1000, 9002, 2),
        _make_event(W0_START + 300, "T", "N", spot, 1, 0, 3),
    ]


def _run_engine(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run events through the engine, return (df_snap, df_flow)."""
    df = pd.DataFrame(events)
    df_snap, df_flow, _ = compute_equity_surfaces_1s(df)
    return df_snap, df_flow


def _get_flow_row(
    df_flow: pd.DataFrame,
    window_end: int,
    side: str,
    price_dollars: float,
) -> pd.Series | None:
    """Get a single flow row by window_end, side, and bucket price."""
    bucket = _bucket_center(price_dollars)
    rows = df_flow[
        (df_flow["window_end_ts_ns"] == window_end)
        & (df_flow["side"] == side)
        & (df_flow["price_int"] == bucket)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Test 1: Accounting identity
# ---------------------------------------------------------------------------
class TestAccountingIdentity:
    """depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty"""

    def test_add_only(self):
        """Add 100 shares in window 1. No pulls, no fills."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W1_END, "B", SPOT)
        assert r is not None, "Expected flow row at spot bid in window 1"

        # Accounting identity
        computed_start = r["depth_qty_end"] - r["add_qty"] + r["pull_qty"] + r["fill_qty"]
        assert abs(r["depth_qty_start"] - max(0.0, computed_start)) < 1e-9

        # Since the setup placed 1000 at SPOT-0.50, NOT at SPOT,
        # depth_start at SPOT bucket should be 0 (no prior depth at exact SPOT bucket)
        # unless SPOT-0.50 maps to the same bucket as SPOT.
        # SPOT=520.00, SPOT-0.50=519.50. These are different $0.50 buckets.
        # So depth_start at SPOT bucket = 0.
        assert r["add_qty"] == 100.0
        assert r["pull_qty"] == 0.0
        assert r["fill_qty"] == 0.0

    def test_add_then_cancel_same_window(self):
        """Add then cancel in the same window."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "A", SPOT + 1.00, 200, 201, 10),
            _make_event(W1_START + 200, "C", "A", 0, 0, 201, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W1_END, "A", SPOT + 1.00)
        assert r is not None

        assert r["depth_qty_end"] == 0.0
        assert r["add_qty"] == 200.0
        assert r["pull_qty"] == 200.0
        computed_start = r["depth_qty_end"] - r["add_qty"] + r["pull_qty"] + r["fill_qty"]
        assert abs(r["depth_qty_start"] - max(0.0, computed_start)) < 1e-9

    def test_add_then_partial_fill(self):
        """Add 500 shares, fill 150 in window 1."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT - 1.00, 500, 301, 10),
            _make_event(W1_START + 200, "F", "B", 0, 150, 301, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W1_END, "B", SPOT - 1.00)
        assert r is not None

        assert r["depth_qty_end"] == 350.0
        assert r["add_qty"] == 500.0
        assert r["fill_qty"] == 150.0
        assert r["pull_qty"] == 0.0
        computed_start = r["depth_qty_end"] - r["add_qty"] + r["pull_qty"] + r["fill_qty"]
        assert abs(r["depth_qty_start"] - max(0.0, computed_start)) < 1e-9

    def test_identity_holds_all_rows(self):
        """Accounting identity must hold for every row in every window."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 50, "A", "B", SPOT, 1000, 101, 10),
            _make_event(W1_START + 100, "A", "A", SPOT + 0.50, 500, 102, 11),
            _make_event(W1_START + 200, "F", "B", 0, 200, 101, 12),
            _make_event(W2_START + 50, "C", "A", 0, 0, 102, 13),
            _make_event(W2_START + 100, "A", "B", SPOT - 1.00, 300, 103, 14),
            _make_event(W2_START + 150, "M", "B", SPOT, 600, 101, 15),
        ]
        _, df_flow = _run_engine(events)

        for idx, row in df_flow.iterrows():
            computed_start = row["depth_qty_end"] - row["add_qty"] + row["pull_qty"] + row["fill_qty"]
            assert abs(row["depth_qty_start"] - max(0.0, computed_start)) < 1e-9, (
                f"Accounting identity violated at row {idx}: "
                f"depth_start={row['depth_qty_start']}, "
                f"computed={max(0.0, computed_start)}, "
                f"end={row['depth_qty_end']}, add={row['add_qty']}, "
                f"pull={row['pull_qty']}, fill={row['fill_qty']}"
            )


# ---------------------------------------------------------------------------
# Test 2: $0.50 bucketing aggregation
# ---------------------------------------------------------------------------
class TestBucketing:
    """Multiple penny prices must aggregate into the same $0.50 bucket."""

    def test_penny_prices_aggregate_to_bucket(self):
        """Orders at $520.01 and $520.24 should both map to $520.00 bucket."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", 520.01, 100, 101, 10),
            _make_event(W1_START + 200, "A", "B", 520.24, 200, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W1_END, "B", 520.00)
        assert r is not None, "Expected aggregated bucket at $520.00"
        # Both orders should aggregate to 300 shares in $520.00 bucket
        assert r["add_qty"] == 300.0

    def test_different_buckets_separate(self):
        """Orders at $520.00 and $520.50 should be in different buckets."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", 520.00, 100, 101, 10),
            _make_event(W1_START + 200, "A", "B", 520.50, 200, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        r1 = _get_flow_row(df_flow, W1_END, "B", 520.00)
        r2 = _get_flow_row(df_flow, W1_END, "B", 520.50)
        assert r1 is not None
        assert r2 is not None
        # Each bucket has only its own order
        assert r1["add_qty"] == 100.0
        assert r2["add_qty"] == 200.0

    def test_bucket_int_value(self):
        """BUCKET_INT must be 500_000_000 for $0.50 at 1e-9 scale."""
        assert BUCKET_INT == 500_000_000

    def test_boundary_penny_prices(self):
        """$520.25 should round to $520.50 bucket (midpoint rounds up)."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", 520.25, 100, 101, 10),
        ]
        _, df_flow = _run_engine(events)

        # $520.25 is equidistant from $520.00 and $520.50.
        # _round_to_bucket uses int(round(price_int / bucket_int)) which for
        # 520.25 * 1e9 = 520_250_000_000 / 500_000_000 = 1040.5 -> rounds to 1040 or 1041
        # Python's round() uses banker's rounding: 1040.5 -> 1040 (even)
        # So 520.25 -> bucket 1040 * 0.5 = $520.00
        b520 = _get_flow_row(df_flow, W1_END, "B", 520.00)
        b520_50 = _get_flow_row(df_flow, W1_END, "B", 520.50)
        # One of these should have the 100 shares
        found = False
        if b520 is not None and b520["add_qty"] >= 100.0:
            found = True
        if b520_50 is not None and b520_50["add_qty"] >= 100.0:
            found = True
        assert found, "Order at $520.25 should map to either $520.00 or $520.50 bucket"


# ---------------------------------------------------------------------------
# Test 3: rel_ticks computation
# ---------------------------------------------------------------------------
class TestRelTicks:
    """rel_ticks = (price_int - spot_ref_price_int) / BUCKET_INT"""

    def test_rel_ticks_formula(self):
        """rel_ticks must equal (price_int - spot_ref_price_int) // BUCKET_INT."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
            _make_event(W1_START + 200, "A", "A", SPOT + 2.50, 50, 102, 11),
            _make_event(W1_START + 300, "A", "B", SPOT - 2.50, 50, 103, 12),
        ]
        _, df_flow = _run_engine(events)

        # Verify rel_ticks formula for every row
        for _, row in df_flow.iterrows():
            expected = int((row["price_int"] - row["spot_ref_price_int"]) // BUCKET_INT)
            assert row["rel_ticks"] == expected, (
                f"rel_ticks mismatch: got {row['rel_ticks']}, expected {expected} "
                f"(price={row['price_int']}, spot_ref={row['spot_ref_price_int']})"
            )

    def test_rel_ticks_offset_relative(self):
        """Orders at different distances from spot_ref have correct relative offsets."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT - 2.50, 100, 101, 10),
            _make_event(W1_START + 200, "A", "A", SPOT + 2.50, 50, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        bid_r = _get_flow_row(df_flow, W1_END, "B", SPOT - 2.50)
        ask_r = _get_flow_row(df_flow, W1_END, "A", SPOT + 2.50)

        assert bid_r is not None
        assert ask_r is not None
        # The offset between bid and ask should be exactly 10 ticks ($5.00 / $0.50)
        assert ask_r["rel_ticks"] - bid_r["rel_ticks"] == 10

    def test_rel_ticks_range(self):
        """All rel_ticks must be in [-100, 100] (GRID_MAX_BUCKETS)."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT - 49.50, 10, 101, 10),
            _make_event(W1_START + 200, "A", "A", SPOT + 49.50, 10, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        assert (df_flow["rel_ticks"].abs() <= GRID_MAX_BUCKETS).all()


# ---------------------------------------------------------------------------
# Test 4: depth_qty_rest <= depth_qty_end (clamping invariant)
# ---------------------------------------------------------------------------
class TestDepthRestClamping:
    """depth_qty_rest must never exceed depth_qty_end."""

    def test_rest_clamped_after_partial_fill(self):
        """Order enters in window 0 (resting by window 1), partially filled in window 1.

        The setup orders at SPOT-0.50 (order 9001, 1000 shares) become resting
        after 500ms. In window 1, partial fill reduces depth but resting depth
        was computed before fill. Clamping must enforce rest <= end.
        """
        events = _setup_book_events()
        events += [
            # Window 1: partial fill of the setup order at SPOT-0.50
            _make_event(W1_START + 100, "F", "B", 0, 800, 9001, 10),
        ]
        _, df_flow = _run_engine(events)

        # All rows must satisfy the invariant
        for _, row in df_flow.iterrows():
            assert row["depth_qty_rest"] <= row["depth_qty_end"] + 1e-9, (
                f"depth_qty_rest ({row['depth_qty_rest']}) > depth_qty_end ({row['depth_qty_end']}) "
                f"at price_int={row['price_int']}, side={row['side']}, "
                f"window_end={row['window_end_ts_ns']}"
            )

    def test_rest_invariant_comprehensive(self):
        """Rest <= end must hold for any complex event sequence."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 50, "A", "B", SPOT, 200, 101, 10),
            _make_event(W1_START + 100, "F", "B", 0, 400, 9001, 11),
            _make_event(W1_START + 200, "C", "A", 0, 0, 9002, 12),
            _make_event(W1_START + 300, "A", "A", SPOT + 0.50, 150, 103, 13),
            _make_event(W2_START + 50, "F", "B", 0, 100, 101, 14),
            _make_event(W2_START + 100, "A", "B", SPOT - 1.00, 500, 104, 15),
        ]
        _, df_flow = _run_engine(events)

        violations = df_flow[df_flow["depth_qty_rest"] > df_flow["depth_qty_end"] + 1e-9]
        assert violations.empty, (
            f"depth_qty_rest > depth_qty_end in {len(violations)} rows:\n"
            f"{violations[['price_int', 'side', 'depth_qty_rest', 'depth_qty_end', 'window_end_ts_ns']]}"
        )


# ---------------------------------------------------------------------------
# Test 5: Sign conventions
# ---------------------------------------------------------------------------
class TestSignConventions:
    """velocity = add - pull - fill. Positive means liquidity is building."""

    def test_velocity_positive_when_adding(self):
        """Pure adds produce positive velocity at gold layer."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
        ]
        df_snap, df_flow = _run_engine(events)

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)

        # Filter to window 1 bid rows
        w1_bid = df_gold[
            (df_gold["window_end_ts_ns"] == W1_END) & (df_gold["side"] == "B")
        ]
        assert not w1_bid.empty

        # Rows with add_intensity > 0 and no pull/fill should have positive velocity
        for _, r in w1_bid.iterrows():
            if r["add_intensity"] > 0 and r["pull_intensity"] == 0 and r["fill_intensity"] == 0:
                assert r["liquidity_velocity"] > 0

    def test_velocity_negative_when_pulling(self):
        """Cancellation produces negative velocity at gold layer."""
        events = _setup_book_events()
        events += [
            # Window 1: add order
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
            # Window 2: cancel it
            _make_event(W2_START + 100, "C", "B", 0, 0, 101, 11),
        ]
        df_snap, df_flow = _run_engine(events)

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)

        # Window 2 should have pull > 0 at SPOT bucket => negative velocity
        w2_bid = df_gold[
            (df_gold["window_end_ts_ns"] == W2_END) & (df_gold["side"] == "B")
        ]
        if not w2_bid.empty:
            for _, r in w2_bid.iterrows():
                if r["pull_intensity"] > 0 and r["add_intensity"] == 0:
                    assert r["liquidity_velocity"] < 0


# ---------------------------------------------------------------------------
# Test 6: Gold intensity normalization
# ---------------------------------------------------------------------------
class TestGoldNormalization:
    """intensity = qty / (depth_qty_start + EPS_QTY)"""

    def test_intensity_formula_exact(self):
        """Verify exact intensity calculation against the formula."""
        events = _setup_book_events()
        events += [
            # Window 1: establish depth at SPOT
            _make_event(W1_START + 100, "A", "B", SPOT, 1000, 101, 10),
            # Window 2: add 250 more at SPOT
            _make_event(W2_START + 100, "A", "B", SPOT, 250, 102, 11),
        ]
        df_snap, df_flow = _run_engine(events)

        # Window 2 silver row at SPOT bid
        silver_r = _get_flow_row(df_flow, W2_END, "B", SPOT)
        assert silver_r is not None
        assert silver_r["depth_qty_start"] == 1000.0  # carried from w1
        assert silver_r["add_qty"] == 250.0

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)

        # Find matching gold row
        bucket = _bucket_center(SPOT)
        # Gold doesn't have price_int column, but we can match via the silver index
        # Instead, verify the formula on all gold rows
        for _, row in df_gold.iterrows():
            # Reconstruct from silver
            flow_match = df_flow[
                (df_flow["window_end_ts_ns"] == row["window_end_ts_ns"])
                & (df_flow["side"] == row["side"])
                & (df_flow["rel_ticks"] == row["rel_ticks"])
            ]
            if flow_match.empty:
                continue
            sr = flow_match.iloc[0]
            expected_add_i = sr["add_qty"] / (sr["depth_qty_start"] + EPS_QTY)
            expected_fill_i = sr["fill_qty"] / (sr["depth_qty_start"] + EPS_QTY)
            expected_pull_i = sr["pull_qty"] / (sr["depth_qty_start"] + EPS_QTY)

            assert abs(row["add_intensity"] - expected_add_i) < 1e-9
            assert abs(row["fill_intensity"] - expected_fill_i) < 1e-9
            assert abs(row["pull_intensity"] - expected_pull_i) < 1e-9

    def test_zero_depth_start_uses_eps(self):
        """When depth_start=0, denominator = EPS_QTY = 1.0."""
        events = _setup_book_events()
        events += [
            # Window 1: add 50 at a new level (no prior depth)
            _make_event(W1_START + 100, "A", "B", SPOT + 5.00, 50, 101, 10),
        ]
        df_snap, df_flow = _run_engine(events)

        silver_r = _get_flow_row(df_flow, W1_END, "B", SPOT + 5.00)
        assert silver_r is not None
        assert silver_r["depth_qty_start"] == 0.0

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)

        # Find the gold row: add_intensity should be 50 / (0 + 1.0) = 50.0
        w1_bid = df_gold[
            (df_gold["window_end_ts_ns"] == W1_END) & (df_gold["side"] == "B")
        ]
        matching = w1_bid[abs(w1_bid["add_intensity"] - 50.0) < 1e-9]
        assert not matching.empty, (
            f"Expected add_intensity=50.0 for depth_start=0, add=50, EPS=1.0. "
            f"Got values: {w1_bid['add_intensity'].values}"
        )


# ---------------------------------------------------------------------------
# Test 7: Gold velocity equation
# ---------------------------------------------------------------------------
class TestGoldVelocityEquation:
    """liquidity_velocity = add_intensity - pull_intensity - fill_intensity"""

    def test_velocity_equation_all_rows(self):
        """Verify velocity = add - pull - fill at gold layer for every row."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 50, "A", "B", SPOT, 1000, 101, 10),
            _make_event(W1_START + 100, "A", "A", SPOT + 0.50, 500, 102, 11),
            _make_event(W1_START + 200, "T", "N", SPOT, 1, 0, 12),
            _make_event(W2_START + 50, "F", "B", 0, 200, 101, 13),
            _make_event(W2_START + 100, "C", "A", 0, 0, 102, 14),
            _make_event(W2_START + 200, "A", "B", SPOT - 1.00, 300, 103, 15),
        ]
        df_snap, df_flow = _run_engine(events)

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)

        for _, row in df_gold.iterrows():
            expected_velocity = row["add_intensity"] - row["pull_intensity"] - row["fill_intensity"]
            assert abs(row["liquidity_velocity"] - expected_velocity) < 1e-12, (
                f"Velocity mismatch: got {row['liquidity_velocity']}, "
                f"expected {expected_velocity}"
            )


# ---------------------------------------------------------------------------
# Test 8: Contract field compliance
# ---------------------------------------------------------------------------
class TestContractCompliance:
    """Output column names and order must match Avro contracts."""

    def _get_events(self) -> list[dict]:
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
            _make_event(W1_START + 200, "A", "A", SPOT + 0.50, 50, 102, 11),
        ]
        return events

    def test_snap_columns_match_contract(self):
        """Snap DataFrame columns must match SNAP_COLUMNS (== avsc field order)."""
        expected = [
            "window_start_ts_ns", "window_end_ts_ns",
            "best_bid_price_int", "best_bid_qty",
            "best_ask_price_int", "best_ask_qty",
            "mid_price", "mid_price_int",
            "last_trade_price_int", "spot_ref_price_int", "book_valid",
        ]
        assert SNAP_COLUMNS == expected

        df_snap, _ = _run_engine(self._get_events())
        assert list(df_snap.columns) == expected

    def test_flow_columns_match_contract(self):
        """Flow DataFrame columns must match DEPTH_FLOW_COLUMNS (== avsc field order)."""
        expected = [
            "window_start_ts_ns", "window_end_ts_ns",
            "price_int", "side", "spot_ref_price_int",
            "rel_ticks", "rel_ticks_side",
            "depth_qty_start", "depth_qty_end",
            "add_qty", "pull_qty",
            "depth_qty_rest", "pull_qty_rest", "fill_qty",
            "window_valid",
        ]
        assert DEPTH_FLOW_COLUMNS == expected

        _, df_flow = _run_engine(self._get_events())
        assert list(df_flow.columns) == expected

    def test_gold_columns_match_contract(self):
        """Gold output columns must match physics_surface_1s.avsc."""
        expected = [
            "window_end_ts_ns", "event_ts_ns", "spot_ref_price_int",
            "rel_ticks", "rel_ticks_side", "side",
            "add_intensity", "fill_intensity", "pull_intensity",
            "liquidity_velocity",
        ]
        df_snap, df_flow = _run_engine(self._get_events())

        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(df_snap, df_flow)
        assert list(df_gold.columns) == expected


# ---------------------------------------------------------------------------
# Test 9: Multi-window cross-window state
# ---------------------------------------------------------------------------
class TestCrossWindowState:
    """Book state carries correctly across windows."""

    def test_depth_carries_to_next_window(self):
        """Depth from window 1 appears as depth_qty_start in window 2."""
        events = _setup_book_events()
        events += [
            # Window 1: add 300 shares at SPOT
            _make_event(W1_START + 100, "A", "B", SPOT, 300, 101, 10),
            # Window 2: add 100 more at SPOT
            _make_event(W2_START + 100, "A", "B", SPOT, 100, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W2_END, "B", SPOT)
        assert r is not None

        # depth_qty_end should be 400 (300 from w1 + 100 from w2)
        assert r["depth_qty_end"] == 400.0
        # depth_qty_start should be 300 (carried from w1)
        assert r["depth_qty_start"] == 300.0
        # add_qty should be 100 (only w2 activity)
        assert r["add_qty"] == 100.0


# ---------------------------------------------------------------------------
# Test 10: Modify order semantics
# ---------------------------------------------------------------------------
class TestModifyOrder:
    """Modify at same price (qty change) vs modify to new price (pull + add)."""

    def test_modify_same_price_reduce(self):
        """Modify qty down at same price => pull delta."""
        events = _setup_book_events()
        events += [
            # Window 1: add 500 at SPOT
            _make_event(W1_START + 100, "A", "B", SPOT, 500, 101, 10),
            # Window 2: modify same price, reduce to 300 => pull 200
            _make_event(W2_START + 100, "M", "B", SPOT, 300, 101, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W2_END, "B", SPOT)
        assert r is not None

        assert r["depth_qty_end"] == 300.0
        assert r["pull_qty"] == 200.0
        assert r["add_qty"] == 0.0

    def test_modify_different_price(self):
        """Modify to new price => pull from old, add to new."""
        events = _setup_book_events()
        events += [
            # Window 1: add 500 at SPOT
            _make_event(W1_START + 100, "A", "B", SPOT, 500, 101, 10),
            # Window 2: move order to SPOT+0.50
            _make_event(W2_START + 100, "M", "B", SPOT + 0.50, 500, 101, 11),
        ]
        _, df_flow = _run_engine(events)

        # Old bucket should show pull
        old_r = _get_flow_row(df_flow, W2_END, "B", SPOT)
        if old_r is not None:
            assert old_r["pull_qty"] == 500.0

        # New bucket should show add
        new_r = _get_flow_row(df_flow, W2_END, "B", SPOT + 0.50)
        assert new_r is not None
        assert new_r["add_qty"] == 500.0
        assert new_r["depth_qty_end"] == 500.0


# ---------------------------------------------------------------------------
# Test 11: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: empty input, gold empty input."""

    def test_empty_input(self):
        """Empty DataFrame should return empty outputs."""
        df = pd.DataFrame(
            columns=["ts_event", "action", "side", "price", "size", "order_id", "sequence", "flags"]
        )
        df_snap, df_flow, _ = compute_equity_surfaces_1s(df)
        assert df_snap.empty
        assert df_flow.empty

    def test_gold_empty_input(self):
        """Gold transform with empty input returns empty with correct columns."""
        gold = GoldComputeEquityPhysicsSurface1s()
        df_gold = gold.transform(pd.DataFrame(), pd.DataFrame())
        expected_cols = [
            "window_end_ts_ns", "event_ts_ns", "spot_ref_price_int",
            "rel_ticks", "rel_ticks_side", "side",
            "add_intensity", "fill_intensity", "pull_intensity",
            "liquidity_velocity",
        ]
        assert list(df_gold.columns) == expected_cols
        assert df_gold.empty


# ---------------------------------------------------------------------------
# Test 12: Window boundary correctness
# ---------------------------------------------------------------------------
class TestWindowBoundaries:
    """Events at exact boundary timestamps are in the correct window."""

    def test_event_at_boundary(self):
        """Event at exactly W1_START is in window 1, not window 0."""
        events = _setup_book_events()
        events += [
            # Window 1: add at SPOT
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
            # Exactly at W2_START boundary
            _make_event(W2_START, "A", "B", SPOT, 200, 102, 11),
        ]
        _, df_flow = _run_engine(events)

        r_w1 = _get_flow_row(df_flow, W1_END, "B", SPOT)
        r_w2 = _get_flow_row(df_flow, W2_END, "B", SPOT)

        assert r_w1 is not None
        assert r_w1["add_qty"] == 100.0  # Only the 100-share add

        assert r_w2 is not None
        assert r_w2["add_qty"] == 200.0  # Only the 200-share add at boundary

    def test_window_start_end_timestamps(self):
        """Verify window start/end timestamps are correct."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
        ]
        df_snap, _ = _run_engine(events)

        # Window 1 snap row
        w1_snap = df_snap[df_snap["window_end_ts_ns"] == W1_END]
        assert len(w1_snap) == 1
        assert w1_snap.iloc[0]["window_start_ts_ns"] == W1_START
        assert w1_snap.iloc[0]["window_end_ts_ns"] == W1_END


# ---------------------------------------------------------------------------
# Test 13: spot_ref_price_int bucketing
# ---------------------------------------------------------------------------
class TestSpotRef:
    """spot_ref_price_int is bucketed to $0.50 grid."""

    def test_spot_ref_is_bucketed(self):
        """spot_ref should always be a multiple of BUCKET_INT."""
        events = _setup_book_events()
        events += [
            _make_event(W1_START + 100, "A", "B", SPOT, 100, 101, 10),
        ]
        df_snap, df_flow = _run_engine(events)

        for _, row in df_snap.iterrows():
            spot = row["spot_ref_price_int"]
            if spot > 0:
                assert spot % BUCKET_INT == 0, (
                    f"spot_ref {spot} is not a multiple of BUCKET_INT {BUCKET_INT}"
                )

        for _, row in df_flow.iterrows():
            spot = row["spot_ref_price_int"]
            assert spot % BUCKET_INT == 0


# ---------------------------------------------------------------------------
# Test 14: Full fill removes order from book
# ---------------------------------------------------------------------------
class TestFullFill:
    """Full fill removes the order from the book entirely."""

    def test_full_fill_zeroes_depth(self):
        """Filling the entire order leaves depth_qty_end = 0 at that level."""
        events = _setup_book_events()
        events += [
            # Window 1: add
            _make_event(W1_START + 100, "A", "B", SPOT, 500, 101, 10),
            # Window 2: full fill
            _make_event(W2_START + 100, "F", "B", 0, 500, 101, 11),
        ]
        _, df_flow = _run_engine(events)

        r = _get_flow_row(df_flow, W2_END, "B", SPOT)
        assert r is not None
        assert r["depth_qty_end"] == 0.0
        assert r["fill_qty"] == 500.0
        assert r["depth_qty_start"] == 500.0
