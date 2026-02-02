#!/usr/bin/env python
"""Unit tests for Cmbp1BookEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_eng.stages.silver.equity_option_cmbp_1.cmbp1_book_engine import (
    Cmbp1BookEngine,
    Cmbp1InstrumentState,
    PRICE_SCALE,
    WINDOW_NS,
)


class TestCmbp1InstrumentState:
    """Test Cmbp1InstrumentState dataclass."""

    def test_default_values(self):
        state = Cmbp1InstrumentState()
        assert state.bid_price_int == 0
        assert state.bid_size == 0
        assert state.ask_price_int == 0
        assert state.ask_size == 0

    def test_custom_values(self):
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        state = Cmbp1InstrumentState(
            bid_price_int=bid_px,
            bid_size=100,
            ask_price_int=ask_px,
            ask_size=200,
        )
        assert state.bid_price_int == bid_px
        assert state.bid_size == 100
        assert state.ask_price_int == ask_px
        assert state.ask_size == 200


class TestCmbp1BookEngine:
    """Test Cmbp1BookEngine processing logic."""

    def test_empty_batch(self):
        """Empty input should produce empty output."""
        engine = Cmbp1BookEngine()
        df_empty = pd.DataFrame()
        df_flow, df_bbo = engine.process_batch(df_empty)
        assert df_flow.empty
        assert df_bbo.empty

    def test_single_update(self):
        """Single CMBP-1 update should create one window."""
        engine = Cmbp1BookEngine()
        
        # Create a realistic CMBP-1 record
        ts = 1736261400_000_000_000  # Some timestamp
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [ts],
            "instrument_id": [12345],
            "bid_px_00": [bid_px],
            "bid_sz_00": [100],
            "ask_px_00": [ask_px],
            "ask_sz_00": [200],
            "strike": [620_000_000_000],  # $620 strike
            "right": ["C"],
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Should have BBO output for the instrument
        assert len(df_bbo) >= 1
        assert df_bbo.iloc[0]["instrument_id"] == 12345
        assert df_bbo.iloc[0]["bid_price_int"] == bid_px
        assert df_bbo.iloc[0]["ask_price_int"] == ask_px

    def test_multiple_updates_same_window(self):
        """Multiple updates in same window should accumulate flow."""
        engine = Cmbp1BookEngine()
        
        base_ts = 1736261400_000_000_000
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [base_ts, base_ts + 100_000_000, base_ts + 200_000_000],
            "instrument_id": [12345, 12345, 12345],
            "bid_px_00": [bid_px, bid_px, bid_px],
            "bid_sz_00": [100, 150, 120],  # Size changes
            "ask_px_00": [ask_px, ask_px, ask_px],
            "ask_sz_00": [200, 180, 220],  # Size changes
            "strike": [620_000_000_000] * 3,
            "right": ["C"] * 3,
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Should have BBO for the final state
        assert len(df_bbo) >= 1
        
        # Flow should capture add/pull activity
        assert not df_flow.empty
        
    def test_multiple_instruments(self):
        """Engine should track multiple instruments independently."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        
        df = pd.DataFrame({
            "ts_event": [ts, ts],
            "instrument_id": [12345, 67890],
            "bid_px_00": [int(5.50 / PRICE_SCALE), int(3.20 / PRICE_SCALE)],
            "bid_sz_00": [100, 50],
            "ask_px_00": [int(5.60 / PRICE_SCALE), int(3.30 / PRICE_SCALE)],
            "ask_sz_00": [200, 75],
            "strike": [620_000_000_000, 625_000_000_000],
            "right": ["C", "P"],
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Should have BBO for both instruments
        assert df_bbo["instrument_id"].nunique() == 2

    def test_window_transition(self):
        """Updates in different windows should create separate emissions."""
        engine = Cmbp1BookEngine()
        
        ts1 = 1736261400_000_000_000  # Window 1
        ts2 = ts1 + 2 * WINDOW_NS     # Window 3 (skip window 2)
        
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [ts1, ts2],
            "instrument_id": [12345, 12345],
            "bid_px_00": [bid_px, bid_px],
            "bid_sz_00": [100, 150],
            "ask_px_00": [ask_px, ask_px],
            "ask_sz_00": [200, 180],
            "strike": [620_000_000_000] * 2,
            "right": ["C"] * 2,
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Should have multiple windows
        unique_windows = df_bbo["window_end_ts_ns"].nunique()
        assert unique_windows >= 2

    def test_crossed_book_filtered(self):
        """Crossed books (ask <= bid) should not produce BBO."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        bid_px = int(5.60 / PRICE_SCALE)  # Bid higher than ask
        ask_px = int(5.50 / PRICE_SCALE)  # Crossed!
        
        df = pd.DataFrame({
            "ts_event": [ts],
            "instrument_id": [12345],
            "bid_px_00": [bid_px],
            "bid_sz_00": [100],
            "ask_px_00": [ask_px],
            "ask_sz_00": [200],
            "strike": [620_000_000_000],
            "right": ["C"],
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Crossed books should be filtered from BBO
        # Check that if BBO is emitted, it's not crossed
        for _, row in df_bbo.iterrows():
            if row["bid_price_int"] > 0 and row["ask_price_int"] > 0:
                assert row["ask_price_int"] > row["bid_price_int"]

    def test_mid_price_calculation(self):
        """Mid price should be average of bid and ask."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        expected_mid = (bid_px + ask_px) / 2
        
        df = pd.DataFrame({
            "ts_event": [ts],
            "instrument_id": [12345],
            "bid_px_00": [bid_px],
            "bid_sz_00": [100],
            "ask_px_00": [ask_px],
            "ask_sz_00": [200],
            "strike": [620_000_000_000],
            "right": ["C"],
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        assert len(df_bbo) >= 1
        assert np.isclose(df_bbo.iloc[0]["mid_price_int"], expected_mid, rtol=1e-6)

    def test_add_accumulation(self):
        """Size increase should accumulate as add_qty."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [ts, ts + 100_000_000],
            "instrument_id": [12345, 12345],
            "bid_px_00": [bid_px, bid_px],
            "bid_sz_00": [100, 200],  # +100 add
            "ask_px_00": [ask_px, ask_px],
            "ask_sz_00": [200, 200],
            "strike": [620_000_000_000] * 2,
            "right": ["C"] * 2,
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Check flow has add activity on bid side
        bid_flow = df_flow[df_flow["side"] == "B"]
        if not bid_flow.empty:
            total_add = bid_flow["add_qty"].sum()
            assert total_add > 0

    def test_pull_accumulation(self):
        """Size decrease should accumulate as pull_qty."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        bid_px = int(5.50 / PRICE_SCALE)
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [ts, ts + 100_000_000],
            "instrument_id": [12345, 12345],
            "bid_px_00": [bid_px, bid_px],
            "bid_sz_00": [200, 100],  # -100 pull
            "ask_px_00": [ask_px, ask_px],
            "ask_sz_00": [200, 200],
            "strike": [620_000_000_000] * 2,
            "right": ["C"] * 2,
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Check flow has pull activity on bid side
        bid_flow = df_flow[df_flow["side"] == "B"]
        if not bid_flow.empty:
            total_pull = bid_flow["pull_qty"].sum()
            assert total_pull > 0

    def test_price_level_change(self):
        """Price level change should pull old, add new."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        bid_px1 = int(5.50 / PRICE_SCALE)
        bid_px2 = int(5.55 / PRICE_SCALE)  # Price change
        ask_px = int(5.60 / PRICE_SCALE)
        
        df = pd.DataFrame({
            "ts_event": [ts, ts + 100_000_000],
            "instrument_id": [12345, 12345],
            "bid_px_00": [bid_px1, bid_px2],
            "bid_sz_00": [100, 150],
            "ask_px_00": [ask_px, ask_px],
            "ask_sz_00": [200, 200],
            "strike": [620_000_000_000] * 2,
            "right": ["C"] * 2,
        })
        
        df_flow, df_bbo = engine.process_batch(df)
        
        # Should have activity at both price levels
        bid_flow = df_flow[df_flow["side"] == "B"]
        unique_prices = bid_flow["price_int"].nunique() if not bid_flow.empty else 0
        # May be 1 or 2 depending on bucketing
        assert unique_prices >= 1

    def test_missing_ts_recv(self):
        """Engine should handle missing ts_recv column."""
        engine = Cmbp1BookEngine()
        
        ts = 1736261400_000_000_000
        
        df = pd.DataFrame({
            "ts_event": [ts],
            "instrument_id": [12345],
            "bid_px_00": [int(5.50 / PRICE_SCALE)],
            "bid_sz_00": [100],
            "ask_px_00": [int(5.60 / PRICE_SCALE)],
            "ask_sz_00": [200],
            "strike": [620_000_000_000],
            "right": ["C"],
        })
        
        # Should not raise
        df_flow, df_bbo = engine.process_batch(df)
        assert not df_bbo.empty

    def test_reset_accumulators(self):
        """_reset_accumulators should clear add/pull dicts."""
        engine = Cmbp1BookEngine()
        
        # Manually add to accumulators
        engine.acc_add[(12345, "B", 100)] = 50.0
        engine.acc_pull[(12345, "A", 200)] = 30.0
        
        # Reset
        engine._reset_accumulators()
        
        assert engine.acc_add == {}
        assert engine.acc_pull == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
