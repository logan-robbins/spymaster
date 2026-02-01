"""
Unit tests for batch_download_equities.py

Tests the 0DTE filtering logic fix for equity options.
"""
import sys
from datetime import datetime
from pathlib import Path

import databento as db
import pandas as pd
import pytest

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "scripts"))

from batch_download_equities import load_0dte_assets


class TestLoad0DTEAssets:
    """Test 0DTE asset loading with timezone fix"""
    
    @pytest.mark.parametrize("session_date,expected_count,expected_exp_date", [
        ("2026-01-06", 194, "2026-01-06"),  # Tuesday - should get Tuesday options
        ("2026-01-07", 192, "2026-01-07"),  # Wednesday
        ("2026-01-08", 192, "2026-01-08"),  # Thursday
        ("2026-01-09", 246, "2026-01-09"),  # Friday - was broken
        ("2026-01-13", 200, "2026-01-13"),  # Monday
        ("2026-01-16", 432, "2026-01-16"),  # Friday - weekly expiry
    ])
    def test_load_0dte_correct_expiration(self, session_date, expected_count, expected_exp_date):
        """Test that load_0dte_assets returns options expiring on session date, not T+1"""
        raw_symbols = load_0dte_assets("QQQ", session_date)
        
        # Verify count
        assert len(raw_symbols) == expected_count, \
            f"{session_date}: Expected {expected_count} symbols, got {len(raw_symbols)}"
        
        # Verify all symbols are for the correct expiration date
        exp_compact = expected_exp_date.replace("2026-01-", "2601")
        for sym in raw_symbols:
            assert exp_compact in sym, \
                f"Symbol {sym} does not contain expected date {exp_compact}"
    
    def test_load_0dte_friday_not_empty(self):
        """Regression test: Friday 0DTE should not return empty list"""
        raw_symbols = load_0dte_assets("QQQ", "2026-01-09")
        assert len(raw_symbols) > 0, "Friday should have 0DTE options"

    def test_load_0dte_works_when_definition_only_in_job_dir(self):
        """Regression: some dates only have OPRA definition under OPRA-*/ job dirs."""
        raw_symbols = load_0dte_assets("QQQ", "2026-01-29")
        assert len(raw_symbols) > 0, "Expected 0DTE symbols for 2026-01-29"
        assert any("260129" in s for s in raw_symbols), "Expected raw_symbols to encode 2026-01-29"
    
    def test_load_0dte_filters_calls_and_puts_only(self):
        """Test that only calls and puts are included"""
        raw_symbols = load_0dte_assets("QQQ", "2026-01-06")
        
        # All symbols should be QQQ options (format: "QQQ   YYMMDDCNNNNN...")
        for sym in raw_symbols:
            assert "QQQ" in sym, f"Symbol {sym} should contain QQQ"
            # Check for C or P in the right position (after YYMMDD)
            # Format: "QQQ   260106C00500000" or "QQQ   260106P00500000"
            parts = sym.split()
            if len(parts) >= 2:
                date_cp_strike = parts[1]  # "260106C00500000"
                option_type = date_cp_strike[6]  # 7th char should be C or P
                assert option_type in ["C", "P"], \
                    f"Symbol {sym} should have C or P at position 6, got {option_type}"
    
    def test_load_0dte_missing_file_raises(self):
        """Test that missing definition file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_0dte_assets("QQQ", "2099-12-31")
    
    def test_load_0dte_no_options_raises(self):
        """Test that date with no 0DTE options raises ValueError"""
        # Use a date that exists but has no same-day options (requires actual data)
        # This is a placeholder - would need a real test case
        pass


class TestTimezoneHandling:
    """Test timezone handling in expiration date conversion"""
    
    def test_expiration_field_is_midnight_utc(self):
        """Verify OPRA expiration field represents midnight UTC on expiration date"""
        # Load a definition file
        def_path = backend_dir / "lake/raw/source=databento/dataset=definition/venue=opra/opra-pillar-20260106.definition.dbn"
        
        if not def_path.exists():
            pytest.skip("Test data not available")
        
        store = db.DBNStore.from_file(str(def_path))
        df = store.to_df(price_type=db.common.enums.PriceType.FIXED, pretty_ts=False, map_symbols=True)
        
        # Get QQQ options expiring Jan 7 (next day from session date Jan 6)
        qqq = df[df['underlying'].astype(str).str.upper() == 'QQQ'].copy()
        qqq = qqq[qqq['instrument_class'].isin(['C', 'P'])].copy()
        jan7_symbols = qqq[qqq['raw_symbol'].astype(str).str.contains('260107')].copy()
        
        if len(jan7_symbols) > 0:
            # Check expiration timestamp
            exp_ts = jan7_symbols['expiration'].iloc[0]
            exp_utc = pd.to_datetime(exp_ts, utc=True)
            
            # Should be midnight UTC on Jan 7
            assert exp_utc.hour == 0, "Expiration should be at midnight UTC"
            assert exp_utc.minute == 0
            assert exp_utc.second == 0
            assert exp_utc.date() == datetime(2026, 1, 7).date(), \
                f"Expiration should be 2026-01-07 in UTC, got {exp_utc.date()}"
            
            # When converted to ET, it becomes 7 PM on Jan 6
            exp_et = exp_utc.tz_convert('America/New_York')
            assert exp_et.hour == 19, "Expiration in ET should be 7 PM previous day"
            assert exp_et.date() == datetime(2026, 1, 6).date(), \
                "Expiration in ET should be Jan 6"
    
    def test_utc_date_extraction_is_correct(self):
        """Verify extracting date in UTC gives correct expiration date"""
        def_path = backend_dir / "lake/raw/source=databento/dataset=definition/venue=opra/opra-pillar-20260109.definition.dbn"
        
        if not def_path.exists():
            pytest.skip("Test data not available")
        
        store = db.DBNStore.from_file(str(def_path))
        df = store.to_df(price_type=db.common.enums.PriceType.FIXED, pretty_ts=False, map_symbols=True)
        
        qqq = df[df['underlying'].astype(str).str.upper() == 'QQQ'].copy()
        qqq = qqq[qqq['instrument_class'].isin(['C', 'P'])].copy()
        
        # Get Friday Jan 9 options
        jan9_symbols = qqq[qqq['raw_symbol'].astype(str).str.contains('260109')].copy()
        
        if len(jan9_symbols) > 0:
            # BUGGY way (current code)
            exp_dates_buggy = (
                pd.to_datetime(jan9_symbols['expiration'].astype('int64'), utc=True)
                .dt.tz_convert('America/New_York')
                .dt.date.astype(str)
            )
            buggy_matches = (exp_dates_buggy == '2026-01-09').sum()
            
            # FIXED way
            exp_dates_fixed = (
                pd.to_datetime(jan9_symbols['expiration'].astype('int64'), utc=True)
                .dt.date.astype(str)
            )
            fixed_matches = (exp_dates_fixed == '2026-01-09').sum()
            
            assert buggy_matches == 0, "Buggy code should match 0 Friday options"
            assert fixed_matches == 246, f"Fixed code should match 246 Friday options, got {fixed_matches}"


class TestDateRange:
    """Test date range generation"""
    
    def test_date_range_excludes_weekends(self):
        """Test that date_range only returns weekdays"""
        from batch_download_equities import date_range
        
        dates = date_range("2026-01-05", "2026-01-12")
        
        # Should exclude Sat Jan 10, Sun Jan 11
        expected = ["2026-01-05", "2026-01-06", "2026-01-07", 
                   "2026-01-08", "2026-01-09", "2026-01-12"]
        assert dates == expected
    
    def test_date_range_single_day(self):
        """Test date range with same start and end"""
        from batch_download_equities import date_range
        
        dates = date_range("2026-01-06", "2026-01-06")
        assert dates == ["2026-01-06"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
