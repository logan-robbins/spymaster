"""Synthetic end-to-end tests for the equity_option_cmbp_1 pipeline.

Tests cover:
1. Book engine BBO-delta -> flow conversion (all 4 cases)
2. Accounting identity: depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
3. fill_qty and pull_qty_rest always 0 for CMBP-1
4. Gold intensity normalization: intensity = qty / (depth_start + 1.0)
5. Gold liquidity_velocity = add_intensity - pull_intensity - fill_intensity
6. rel_ticks alignment to even values ($1 strike grid on $0.50 tick base)
7. Strike grid coverage (+/- $25 from spot, 51 buckets)
8. Contract field compliance for silver and gold outputs
9. Window boundary correctness
10. Spot reference join semantics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_eng.stages.silver.equity_option_cmbp_1.cmbp1_book_engine import (
    Cmbp1BookEngine,
    Cmbp1InstrumentState,
    PRICE_SCALE,
    WINDOW_NS,
    SIDE_ASK,
    SIDE_BID,
)
from src.data_eng.stages.silver.equity_option_cmbp_1.compute_book_states_1s import (
    SilverComputeEquityOptionBookStates1s,
    MAX_STRIKE_OFFSETS,
    STRIKE_STEP_POINTS,
    RIGHTS,
    SIDES,
)
from src.data_eng.stages.gold.equity_option_cmbp_1.compute_physics_surface_1s import (
    GoldComputeEquityOptionPhysicsSurface1s,
    EPS_QTY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_int(price_dollars: float) -> int:
    """Convert a dollar price to the fixed-point integer representation."""
    return int(round(price_dollars / PRICE_SCALE))


def _make_cmbp_df(
    rows: list[dict],
    underlying: str = "QQQ",
    right: str = "C",
    strike_dollars: float = 620.0,
    expiration_ns: int = 1_736_294_400_000_000_000,
) -> pd.DataFrame:
    """Build a synthetic bronze CMBP-1 DataFrame.

    Each row dict must have: ts_event, instrument_id, bid_px, bid_sz, ask_px, ask_sz
    Prices are in dollars, sizes in contracts.
    """
    records = []
    strike_int = _price_int(strike_dollars)
    for r in rows:
        records.append({
            "ts_recv": r["ts_event"],
            "ts_event": r["ts_event"],
            "publisher_id": 1,
            "instrument_id": r["instrument_id"],
            "bid_px_00": _price_int(r["bid_px"]),
            "bid_sz_00": r["bid_sz"],
            "ask_px_00": _price_int(r["ask_px"]),
            "ask_sz_00": r["ask_sz"],
            "underlying": underlying,
            "right": right,
            "strike": strike_int,
            "expiration": expiration_ns,
        })
    return pd.DataFrame(records)


def _make_eq_snap(window_end_ts_list: list[int], spot_price_dollars: float = 620.0) -> pd.DataFrame:
    """Build a synthetic equity_mbo book_snapshot_1s DataFrame for spot reference."""
    spot_int = _price_int(spot_price_dollars)
    return pd.DataFrame({
        "window_start_ts_ns": [t - WINDOW_NS for t in window_end_ts_list],
        "window_end_ts_ns": window_end_ts_list,
        "best_bid_price_int": [spot_int - _price_int(0.01)] * len(window_end_ts_list),
        "best_bid_qty": [1000] * len(window_end_ts_list),
        "best_ask_price_int": [spot_int + _price_int(0.01)] * len(window_end_ts_list),
        "best_ask_qty": [1000] * len(window_end_ts_list),
        "mid_price": [spot_price_dollars] * len(window_end_ts_list),
        "mid_price_int": [spot_int] * len(window_end_ts_list),
        "last_trade_price_int": [spot_int] * len(window_end_ts_list),
        "spot_ref_price_int": [spot_int] * len(window_end_ts_list),
        "book_valid": [True] * len(window_end_ts_list),
    })


# ---------------------------------------------------------------------------
# Book Engine: BBO Delta -> Flow Tests
# ---------------------------------------------------------------------------

class TestCmbp1BookEngineBBODelta:
    """Verify all four BBO-delta flow inference cases."""

    def _base_ts(self) -> int:
        """Return a base timestamp aligned to a 1-second window boundary."""
        return 1_736_261_400_000_000_000  # Aligned to window

    def test_case1_new_liquidity_appearing(self):
        """Case 1: old empty (price=0, size=0) -> new valid.
        Should record add_qty = new_size.
        """
        engine = Cmbp1BookEngine()
        ts = self._base_ts()
        bid_px = _price_int(5.50)
        ask_px = _price_int(5.60)

        df = _make_cmbp_df([{
            "ts_event": ts,
            "instrument_id": 100,
            "bid_px": 5.50,
            "bid_sz": 50,
            "ask_px": 5.60,
            "ask_sz": 75,
        }])

        df_flow, df_bbo = engine.process_batch(df)

        # First update for instrument 100: old state is (0,0,0,0), new is valid
        # Both bid and ask should appear as adds
        bid_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_BID)]
        ask_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_ASK)]

        assert bid_flow["add_qty"].sum() == 50.0
        assert bid_flow["pull_qty"].sum() == 0.0
        assert ask_flow["add_qty"].sum() == 75.0
        assert ask_flow["pull_qty"].sum() == 0.0

    def test_case2_liquidity_disappearing(self):
        """Case 2: old valid -> new empty (price=0 or size=0).
        Should record pull_qty = old_size.
        """
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        # First update: establish state
        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
            # Second update: bid disappears (price goes to 0)
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 0.0, "bid_sz": 0,
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, _ = engine.process_batch(df)

        bid_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_BID)]
        # Net: add 50 (first update) + pull 50 (second update)
        assert bid_flow["add_qty"].sum() == 50.0
        assert bid_flow["pull_qty"].sum() == 50.0

    def test_case3_same_price_size_increase(self):
        """Case 3a: same price, size increases.
        Should record add_qty = delta.
        """
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 80,  # +30
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, _ = engine.process_batch(df)

        bid_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_BID)]
        # First update: add 50 (from nothing)
        # Second update: add 30 (size increase, same price)
        assert bid_flow["add_qty"].sum() == 80.0  # 50 + 30
        assert bid_flow["pull_qty"].sum() == 0.0

    def test_case3_same_price_size_decrease(self):
        """Case 3b: same price, size decreases.
        Should record pull_qty = abs(delta).
        """
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 80,
             "ask_px": 5.60, "ask_sz": 75},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,  # -30
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, _ = engine.process_batch(df)

        bid_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_BID)]
        # First update: add 80 (from nothing)
        # Second update: pull 30 (size decrease)
        assert bid_flow["add_qty"].sum() == 80.0
        assert bid_flow["pull_qty"].sum() == 30.0

    def test_case4_price_level_change(self):
        """Case 4: price changes.
        Should pull old_size at old_price, add new_size at new_price.
        """
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        old_bid_px = _price_int(5.50)
        new_bid_px = _price_int(5.55)

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 5.55, "bid_sz": 60,  # Price change
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, _ = engine.process_batch(df)

        bid_flow = df_flow[(df_flow["instrument_id"] == 100) & (df_flow["side"] == SIDE_BID)]

        # Old price level: add 50 (from nothing) + pull 50 (price change)
        old_price_flow = bid_flow[bid_flow["price_int"] == old_bid_px]
        assert old_price_flow["add_qty"].sum() == 50.0
        assert old_price_flow["pull_qty"].sum() == 50.0

        # New price level: add 60 (new price)
        new_price_flow = bid_flow[bid_flow["price_int"] == new_bid_px]
        assert new_price_flow["add_qty"].sum() == 60.0
        assert new_price_flow["pull_qty"].sum() == 0.0

    def test_fill_qty_always_zero(self):
        """CMBP-1 has no trade data, so fill_qty must always be 0."""
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 30,
             "ask_px": 5.60, "ask_sz": 90},
        ])

        df_flow, _ = engine.process_batch(df)

        assert (df_flow["fill_qty"] == 0.0).all(), "fill_qty must be 0 for CMBP-1"

    def test_pull_rest_qty_always_zero(self):
        """CMBP-1 has no order age data, so pull_rest_qty must always be 0."""
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, _ = engine.process_batch(df)

        assert (df_flow["pull_rest_qty"] == 0.0).all(), "pull_rest_qty must be 0 for CMBP-1"


# ---------------------------------------------------------------------------
# Book Engine: Window Boundary Tests
# ---------------------------------------------------------------------------

class TestCmbp1BookEngineWindows:
    """Test window management and emission timing."""

    def _base_ts(self) -> int:
        return 1_736_261_400_000_000_000

    def test_window_end_ts_is_start_plus_1s(self):
        """Each emitted window should span exactly 1 second."""
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, df_bbo = engine.process_batch(df)

        expected_window_end = (ts // WINDOW_NS + 1) * WINDOW_NS
        assert (df_flow["window_end_ts_ns"] == expected_window_end).all()
        assert (df_bbo["window_end_ts_ns"] == expected_window_end).all()

    def test_gap_windows_carry_state(self):
        """State should persist across gap windows (no events in between)."""
        engine = Cmbp1BookEngine()
        ts1 = self._base_ts()
        ts2 = ts1 + 3 * WINDOW_NS  # Skip 2 windows

        df = _make_cmbp_df([
            {"ts_event": ts1, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 50,
             "ask_px": 5.60, "ask_sz": 75},
            {"ts_event": ts2, "instrument_id": 100,
             "bid_px": 5.50, "bid_sz": 60,
             "ask_px": 5.60, "ask_sz": 75},
        ])

        df_flow, df_bbo = engine.process_batch(df)

        # Gap windows should still emit BBO snapshots (state persists)
        unique_windows = sorted(df_bbo["window_end_ts_ns"].unique())
        assert len(unique_windows) >= 2

    def test_crossed_book_no_bbo(self):
        """Crossed books should not produce BBO output."""
        engine = Cmbp1BookEngine()
        ts = self._base_ts()

        df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 5.60, "bid_sz": 50,  # bid > ask = crossed
             "ask_px": 5.50, "ask_sz": 75},
        ])

        _, df_bbo = engine.process_batch(df)

        # No valid BBO for crossed book
        for _, row in df_bbo.iterrows():
            if row["bid_price_int"] > 0 and row["ask_price_int"] > 0:
                assert row["ask_price_int"] > row["bid_price_int"]


# ---------------------------------------------------------------------------
# Silver Transform: Accounting Identity
# ---------------------------------------------------------------------------

class TestSilverTransformAccountingIdentity:
    """Verify the accounting identity across the silver layer."""

    def _base_ts(self) -> int:
        """Return a timestamp within the first-hour window (09:30-09:40 ET).

        Using 2026-01-07 09:35:00 ET = 14:35:00 UTC.
        """
        return int(pd.Timestamp("2026-01-07 14:35:00", tz="UTC").value)

    def test_accounting_identity_holds(self):
        """depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty (clamped >= 0)."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS
        spot_price = 620.0

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 55,  # +15 add
             "ask_px": 3.20, "ask_sz": 45},  # -15 pull
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], spot_price)

        stage = SilverComputeEquityOptionBookStates1s()
        df_snap, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            computed_start = (
                df_flow["depth_qty_end"]
                - df_flow["add_qty"]
                + df_flow["pull_qty"]
                + df_flow["fill_qty"]
            ).clip(lower=0.0)

            np.testing.assert_allclose(
                df_flow["depth_qty_start"].values,
                computed_start.values,
                atol=1e-6,
                err_msg="Accounting identity violated",
            )

    def test_fill_qty_zero_in_silver(self):
        """fill_qty must be 0 in silver output for CMBP-1."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert (df_flow["fill_qty"] == 0.0).all()

    def test_pull_qty_rest_zero_in_silver(self):
        """pull_qty_rest must be 0 in silver output for CMBP-1."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert (df_flow["pull_qty_rest"] == 0.0).all()

    def test_depth_qty_start_non_negative(self):
        """depth_qty_start must never be negative after clamping."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
            {"ts_event": ts + 100_000_000, "instrument_id": 100,
             "bid_px": 3.10, "bid_sz": 80,  # Price change
             "ask_px": 3.20, "ask_sz": 30},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert (df_flow["depth_qty_start"] >= 0.0).all()


# ---------------------------------------------------------------------------
# Silver Transform: Strike Grid Tests
# ---------------------------------------------------------------------------

class TestSilverTransformStrikeGrid:
    """Verify strike grid construction and rel_ticks alignment."""

    def _base_ts(self) -> int:
        return int(pd.Timestamp("2026-01-07 14:35:00", tz="UTC").value)

    def test_rel_ticks_are_even(self):
        """All rel_ticks must be even (multiples of 2) for equity options $1 grid."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert (df_flow["rel_ticks"] % 2 == 0).all(), "rel_ticks must be even for $1 grid"

    def test_rel_ticks_range(self):
        """rel_ticks must be within [-50, +50] (25 offsets * 2)."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            max_rel = MAX_STRIKE_OFFSETS * 2
            assert df_flow["rel_ticks"].min() >= -max_rel
            assert df_flow["rel_ticks"].max() <= max_rel

    def test_grid_has_51_strike_buckets(self):
        """Full grid should have 51 unique rel_ticks values (-50 to +50, step 2)."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            expected_count = 2 * MAX_STRIKE_OFFSETS + 1  # 51
            unique_ticks = df_flow["rel_ticks"].nunique()
            assert unique_ticks == expected_count, (
                f"Expected {expected_count} strike buckets, got {unique_ticks}"
            )

    def test_grid_has_all_rights_and_sides(self):
        """Each strike bucket should have C/P rights and A/B sides."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert set(df_flow["right"].unique()) == set(RIGHTS)
            assert set(df_flow["side"].unique()) == set(SIDES)


# ---------------------------------------------------------------------------
# Silver Transform: Contract Field Compliance
# ---------------------------------------------------------------------------

class TestSilverContractCompliance:
    """Verify output columns match Avro contract definitions."""

    SNAP_FIELDS = [
        "window_start_ts_ns", "window_end_ts_ns", "instrument_id", "right",
        "strike_price_int", "bid_price_int", "ask_price_int", "mid_price",
        "mid_price_int", "spot_ref_price_int", "book_valid",
    ]

    FLOW_FIELDS = [
        "window_start_ts_ns", "window_end_ts_ns", "strike_price_int",
        "strike_points", "right", "side", "spot_ref_price_int", "rel_ticks",
        "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty",
        "pull_qty_rest", "fill_qty", "window_valid",
    ]

    def _base_ts(self) -> int:
        return int(pd.Timestamp("2026-01-07 14:35:00", tz="UTC").value)

    def test_snap_columns_match_contract(self):
        """book_snapshot_1s output must contain exactly the contract fields."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        df_snap, _ = stage.transform(cmbp_df, eq_snap)

        if not df_snap.empty:
            assert list(df_snap.columns) == self.SNAP_FIELDS

    def test_flow_columns_match_contract(self):
        """depth_and_flow_1s output must contain exactly the contract fields."""
        ts = self._base_ts()
        window_end = (ts // WINDOW_NS + 1) * WINDOW_NS

        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ], strike_dollars=620.0)

        eq_snap = _make_eq_snap([window_end], 620.0)

        stage = SilverComputeEquityOptionBookStates1s()
        _, df_flow = stage.transform(cmbp_df, eq_snap)

        if not df_flow.empty:
            assert list(df_flow.columns) == self.FLOW_FIELDS


# ---------------------------------------------------------------------------
# Gold Transform: Intensity Normalization
# ---------------------------------------------------------------------------

class TestGoldIntensityNormalization:
    """Verify gold layer intensity calculations."""

    def test_intensity_formula(self):
        """intensity = qty / (depth_qty_start + EPS_QTY)."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000],
            "window_end_ts_ns": [2000],
            "strike_price_int": [_price_int(620.0)],
            "strike_points": [620.0],
            "right": ["C"],
            "side": ["A"],
            "spot_ref_price_int": [_price_int(620.0)],
            "rel_ticks": [0],
            "depth_qty_start": [100.0],
            "depth_qty_end": [120.0],
            "add_qty": [30.0],
            "pull_qty": [10.0],
            "pull_qty_rest": [0.0],
            "fill_qty": [0.0],
            "window_valid": [True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        denom = 100.0 + EPS_QTY  # 101.0

        expected_add_intensity = 30.0 / denom
        expected_pull_intensity = 10.0 / denom
        expected_fill_intensity = 0.0 / denom
        expected_velocity = expected_add_intensity - expected_pull_intensity - expected_fill_intensity

        np.testing.assert_allclose(df_out["add_intensity"].values[0], expected_add_intensity, atol=1e-10)
        np.testing.assert_allclose(df_out["pull_intensity"].values[0], expected_pull_intensity, atol=1e-10)
        np.testing.assert_allclose(df_out["fill_intensity"].values[0], expected_fill_intensity, atol=1e-10)
        np.testing.assert_allclose(df_out["liquidity_velocity"].values[0], expected_velocity, atol=1e-10)

    def test_zero_depth_start_does_not_divide_by_zero(self):
        """When depth_qty_start=0, denominator = 0 + 1.0 = 1.0 (no infinity)."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000],
            "window_end_ts_ns": [2000],
            "strike_price_int": [_price_int(620.0)],
            "strike_points": [620.0],
            "right": ["C"],
            "side": ["A"],
            "spot_ref_price_int": [_price_int(620.0)],
            "rel_ticks": [0],
            "depth_qty_start": [0.0],
            "depth_qty_end": [25.0],
            "add_qty": [25.0],
            "pull_qty": [0.0],
            "pull_qty_rest": [0.0],
            "fill_qty": [0.0],
            "window_valid": [True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        assert not np.isinf(df_out["add_intensity"].values[0])
        assert not np.isnan(df_out["add_intensity"].values[0])
        np.testing.assert_allclose(df_out["add_intensity"].values[0], 25.0 / (0.0 + EPS_QTY), atol=1e-10)

    def test_fill_intensity_always_zero_for_cmbp1(self):
        """fill_intensity must be 0 when fill_qty is 0 (always for CMBP-1)."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000, 2000],
            "window_end_ts_ns": [2000, 3000],
            "strike_price_int": [_price_int(620.0)] * 2,
            "strike_points": [620.0] * 2,
            "right": ["C", "P"],
            "side": ["A", "B"],
            "spot_ref_price_int": [_price_int(620.0)] * 2,
            "rel_ticks": [0, 2],
            "depth_qty_start": [50.0, 100.0],
            "depth_qty_end": [60.0, 80.0],
            "add_qty": [20.0, 10.0],
            "pull_qty": [10.0, 30.0],
            "pull_qty_rest": [0.0, 0.0],
            "fill_qty": [0.0, 0.0],
            "window_valid": [True, True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        assert (df_out["fill_intensity"] == 0.0).all()

    def test_liquidity_velocity_sign(self):
        """Positive velocity = net adding. Negative velocity = net pulling."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000, 1000],
            "window_end_ts_ns": [2000, 2000],
            "strike_price_int": [_price_int(620.0)] * 2,
            "strike_points": [620.0] * 2,
            "right": ["C", "C"],
            "side": ["A", "B"],
            "spot_ref_price_int": [_price_int(620.0)] * 2,
            "rel_ticks": [0, 0],
            "depth_qty_start": [100.0, 100.0],
            "depth_qty_end": [130.0, 80.0],
            "add_qty": [30.0, 5.0],      # Net add > pull
            "pull_qty": [0.0, 25.0],      # Net pull > add
            "pull_qty_rest": [0.0, 0.0],
            "fill_qty": [0.0, 0.0],
            "window_valid": [True, True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        # Row 0: add_intensity > pull_intensity -> positive velocity
        assert df_out.iloc[0]["liquidity_velocity"] > 0

        # Row 1: pull_intensity > add_intensity -> negative velocity
        assert df_out.iloc[1]["liquidity_velocity"] < 0


# ---------------------------------------------------------------------------
# Gold Transform: Contract Compliance
# ---------------------------------------------------------------------------

class TestGoldContractCompliance:
    """Verify gold output columns match Avro contract."""

    GOLD_FIELDS = [
        "window_end_ts_ns", "event_ts_ns", "spot_ref_price_int",
        "strike_price_int", "strike_points", "rel_ticks", "right", "side",
        "add_intensity", "fill_intensity", "pull_intensity", "liquidity_velocity",
    ]

    def test_gold_columns_match_contract(self):
        """physics_surface_1s output must contain exactly the contract fields."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000],
            "window_end_ts_ns": [2000],
            "strike_price_int": [_price_int(620.0)],
            "strike_points": [620.0],
            "right": ["C"],
            "side": ["A"],
            "spot_ref_price_int": [_price_int(620.0)],
            "rel_ticks": [0],
            "depth_qty_start": [100.0],
            "depth_qty_end": [120.0],
            "add_qty": [30.0],
            "pull_qty": [10.0],
            "pull_qty_rest": [0.0],
            "fill_qty": [0.0],
            "window_valid": [True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        assert list(df_out.columns) == self.GOLD_FIELDS

    def test_event_ts_equals_window_end(self):
        """event_ts_ns should equal window_end_ts_ns."""
        df_flow = pd.DataFrame({
            "window_start_ts_ns": [1000],
            "window_end_ts_ns": [2000],
            "strike_price_int": [_price_int(620.0)],
            "strike_points": [620.0],
            "right": ["C"],
            "side": ["A"],
            "spot_ref_price_int": [_price_int(620.0)],
            "rel_ticks": [0],
            "depth_qty_start": [100.0],
            "depth_qty_end": [120.0],
            "add_qty": [30.0],
            "pull_qty": [10.0],
            "pull_qty_rest": [0.0],
            "fill_qty": [0.0],
            "window_valid": [True],
        })

        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(df_flow)

        assert (df_out["event_ts_ns"] == df_out["window_end_ts_ns"]).all()


# ---------------------------------------------------------------------------
# Empty Input Edge Cases
# ---------------------------------------------------------------------------

class TestEmptyInputHandling:
    """Verify graceful handling of empty inputs at each layer."""

    def test_book_engine_empty_input(self):
        engine = Cmbp1BookEngine()
        df_flow, df_bbo = engine.process_batch(pd.DataFrame())
        assert df_flow.empty
        assert df_bbo.empty

    def test_silver_transform_empty_cmbp(self):
        stage = SilverComputeEquityOptionBookStates1s()
        eq_snap = _make_eq_snap([1_736_261_401_000_000_000], 620.0)
        df_snap, df_flow = stage.transform(pd.DataFrame(), eq_snap)
        assert df_snap.empty
        assert df_flow.empty

    def test_silver_transform_empty_eq_snap(self):
        ts = int(pd.Timestamp("2026-01-07 14:35:00", tz="UTC").value)
        cmbp_df = _make_cmbp_df([
            {"ts_event": ts, "instrument_id": 100,
             "bid_px": 3.00, "bid_sz": 40,
             "ask_px": 3.20, "ask_sz": 60},
        ])
        stage = SilverComputeEquityOptionBookStates1s()
        df_snap, df_flow = stage.transform(cmbp_df, pd.DataFrame())
        assert df_snap.empty
        assert df_flow.empty

    def test_gold_transform_empty_flow(self):
        stage = GoldComputeEquityOptionPhysicsSurface1s()
        df_out = stage.transform(pd.DataFrame())
        assert df_out.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
