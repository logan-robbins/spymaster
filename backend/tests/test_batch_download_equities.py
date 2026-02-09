"""
Unit tests for batch_download_equities.py

Tests the 0DTE filtering logic and 3-phase flow control for equity options.
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import databento as db
import pandas as pd
import pytest

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "scripts"))

import batch_download_equities as equities_dl
from batch_download_equities import load_0dte_assets


def _resolve_definition_path(symbol: str, session_date: str) -> Path:
    """Resolve definition path, checking both new cache location and legacy job dirs."""
    date_compact = session_date.replace("-", "")
    # New path format: dataset=definition/venue=opra/symbol={SYM}/
    new_path = equities_dl.target_path_options_definition(symbol, date_compact)
    if new_path.exists():
        return new_path
    # Legacy: search OPRA-* job dirs for definition files
    opra_dir = backend_dir / "lake" / "raw" / "source=databento" / "dataset=definition" / "venue=opra"
    if opra_dir.exists():
        for job_dir in sorted(opra_dir.iterdir()):
            if not job_dir.is_dir() or not job_dir.name.startswith("OPRA-"):
                continue
            for f in job_dir.iterdir():
                if f.name.endswith(".definition.dbn") or f.name.endswith(".definition.dbn.zst"):
                    if date_compact in f.name:
                        return f
    pytest.skip(f"No definition file for {symbol} {session_date}")


class TestLoad0DTEAssets:
    """Test 0DTE asset loading with timezone fix"""

    @pytest.mark.parametrize("session_date,expected_count,expected_exp_date", [
        ("2026-01-06", 194, "2026-01-06"),
        ("2026-01-07", 192, "2026-01-07"),
        ("2026-01-08", 192, "2026-01-08"),
        ("2026-01-09", 246, "2026-01-09"),
        ("2026-01-13", 200, "2026-01-13"),
        ("2026-01-16", 432, "2026-01-16"),
    ])
    def test_load_0dte_correct_expiration(self, session_date, expected_count, expected_exp_date):
        """Test that load_0dte_assets returns options expiring on session date, not T+1"""
        def_path = _resolve_definition_path("QQQ", session_date)
        raw_symbols = load_0dte_assets(def_path, "QQQ", session_date)

        assert len(raw_symbols) == expected_count, \
            f"{session_date}: Expected {expected_count} symbols, got {len(raw_symbols)}"

        exp_compact = expected_exp_date.replace("2026-01-", "2601")
        for sym in raw_symbols:
            assert exp_compact in sym, \
                f"Symbol {sym} does not contain expected date {exp_compact}"

    def test_load_0dte_friday_not_empty(self):
        """Regression test: Friday 0DTE should not return empty list"""
        def_path = _resolve_definition_path("QQQ", "2026-01-09")
        raw_symbols = load_0dte_assets(def_path, "QQQ", "2026-01-09")
        assert len(raw_symbols) > 0, "Friday should have 0DTE options"

    def test_load_0dte_works_when_definition_only_in_job_dir(self):
        """Regression: some dates only have OPRA definition under OPRA-*/ job dirs."""
        def_path = _resolve_definition_path("QQQ", "2026-01-29")
        raw_symbols = load_0dte_assets(def_path, "QQQ", "2026-01-29")
        assert len(raw_symbols) > 0, "Expected 0DTE symbols for 2026-01-29"
        assert any("260129" in s for s in raw_symbols), "Expected raw_symbols to encode 2026-01-29"

    def test_load_0dte_filters_calls_and_puts_only(self):
        """Test that only calls and puts are included"""
        def_path = _resolve_definition_path("QQQ", "2026-01-06")
        raw_symbols = load_0dte_assets(def_path, "QQQ", "2026-01-06")

        for sym in raw_symbols:
            assert "QQQ" in sym, f"Symbol {sym} should contain QQQ"
            parts = sym.split()
            if len(parts) >= 2:
                date_cp_strike = parts[1]
                option_type = date_cp_strike[6]
                assert option_type in ["C", "P"], \
                    f"Symbol {sym} should have C or P at position 6, got {option_type}"

    def test_load_0dte_missing_file_raises(self):
        """Test that missing definition file raises FileNotFoundError"""
        fake_path = Path("/nonexistent/fake.dbn.zst")
        with pytest.raises(FileNotFoundError):
            load_0dte_assets(fake_path, "QQQ", "2099-12-31")


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


# ---------------------------------------------------------------------------
# 3-phase flow control tests (mock-based, no real data needed)
# ---------------------------------------------------------------------------

class TestPhase2SubmitsEquityAndOptions:
    """Verify phase2 filters 0DTE and submits equity + options batch jobs."""

    def test_phase2_submits_equity_mbo_and_options_cmbp1(self, monkeypatch, tmp_path) -> None:
        """Phase2 should submit equity MBO + options cmbp-1 + statistics jobs."""
        calls: list[dict] = []

        def fake_submit_job(**kwargs):
            calls.append(kwargs)
            return "job-id"

        monkeypatch.setattr(equities_dl, "submit_job", fake_submit_job)
        monkeypatch.setattr(
            equities_dl, "target_path_options_definition",
            lambda symbol, dc: tmp_path / f"def-{symbol}-{dc}.dbn.zst",
        )
        def_path = tmp_path / "def-QQQ-20260206.dbn.zst"
        def_path.touch()

        monkeypatch.setattr(
            equities_dl, "load_0dte_assets",
            lambda definition_path, symbol, date_str: ["QQQ   260206C00500000", "QQQ   260206P00500000"],
        )

        tracker: dict = {"jobs": {}, "pending_downloads": []}

        equities_dl.process_session_day_phase2(
            client=object(),
            tracker=tracker,
            symbol="QQQ",
            date_str="2026-02-06",
            equity_schemas=["mbo"],
            options_schemas=["cmbp-1", "statistics"],
            log_path=tmp_path / "equities.log",
            pause_seconds=0,
        )

        assert len(calls) == 3

        # First call: equity MBO
        eq_call = calls[0]
        assert eq_call["product_type"] == equities_dl.PRODUCT_EQUITY
        assert eq_call["dataset"] == "XNAS.ITCH"
        assert eq_call["schema"] == "mbo"
        assert eq_call["symbols"] == ["QQQ"]
        assert eq_call["stype_in"] == "raw_symbol"

        # Remaining calls: options cmbp-1 + statistics
        opt_calls = calls[1:]
        assert [c["schema"] for c in opt_calls] == ["cmbp-1", "statistics"]
        assert all(c["product_type"] == equities_dl.PRODUCT_EQUITY_OPTIONS for c in opt_calls)
        assert all(c["dataset"] == "OPRA.PILLAR" for c in opt_calls)
        assert all(c["stype_in"] == "raw_symbol" for c in opt_calls)
        assert all(
            c["symbols"] == ["QQQ   260206C00500000", "QQQ   260206P00500000"]
            for c in opt_calls
        )

    def test_phase2_gracefully_handles_no_0dte(self, monkeypatch, tmp_path) -> None:
        """Phase2 should still submit equity MBO even when no 0DTE options exist."""
        calls: list[dict] = []

        def fake_submit_job(**kwargs):
            calls.append(kwargs)
            return "job-id"

        monkeypatch.setattr(equities_dl, "submit_job", fake_submit_job)
        monkeypatch.setattr(
            equities_dl, "target_path_options_definition",
            lambda symbol, dc: tmp_path / f"def-{symbol}-{dc}.dbn.zst",
        )
        def_path = tmp_path / "def-QQQ-20260207.dbn.zst"
        def_path.touch()

        # Simulate no 0DTE options found
        monkeypatch.setattr(
            equities_dl, "load_0dte_assets",
            lambda definition_path, symbol, date_str: (_ for _ in ()).throw(
                ValueError(f"No QQQ 0DTE definitions for {date_str}")
            ),
        )

        tracker: dict = {"jobs": {}, "pending_downloads": []}

        # Should NOT raise — graceful skip
        equities_dl.process_session_day_phase2(
            client=object(),
            tracker=tracker,
            symbol="QQQ",
            date_str="2026-02-07",
            equity_schemas=["mbo"],
            options_schemas=["cmbp-1", "statistics"],
            log_path=tmp_path / "equities.log",
            pause_seconds=0,
        )

        # Only equity MBO job should be submitted, no options jobs
        assert len(calls) == 1
        assert calls[0]["product_type"] == equities_dl.PRODUCT_EQUITY
        assert calls[0]["schema"] == "mbo"

    def test_phase2_skips_definition_schema_in_options(self, monkeypatch, tmp_path) -> None:
        """Phase2 should skip 'definition' schema when submitting options data jobs."""
        calls: list[dict] = []

        def fake_submit_job(**kwargs):
            calls.append(kwargs)
            return "job-id"

        monkeypatch.setattr(equities_dl, "submit_job", fake_submit_job)
        monkeypatch.setattr(
            equities_dl, "target_path_options_definition",
            lambda symbol, dc: tmp_path / f"def-{symbol}-{dc}.dbn.zst",
        )
        (tmp_path / "def-QQQ-20260206.dbn.zst").touch()
        monkeypatch.setattr(
            equities_dl, "load_0dte_assets",
            lambda definition_path, symbol, date_str: ["QQQ   260206C00500000"],
        )

        tracker: dict = {"jobs": {}, "pending_downloads": []}

        equities_dl.process_session_day_phase2(
            client=object(),
            tracker=tracker,
            symbol="QQQ",
            date_str="2026-02-06",
            equity_schemas=["mbo"],
            options_schemas=["definition", "cmbp-1"],
            log_path=tmp_path / "equities.log",
            pause_seconds=0,
        )

        # Should have: equity mbo + options cmbp-1 (definition skipped)
        assert len(calls) == 2
        schemas = [c["schema"] for c in calls]
        assert "definition" not in schemas

    def test_phase2_skips_when_definition_not_available(self, monkeypatch, tmp_path) -> None:
        """Phase2 should skip if definition file does not exist yet."""
        calls: list[dict] = []

        def fake_submit_job(**kwargs):
            calls.append(kwargs)
            return "job-id"

        monkeypatch.setattr(equities_dl, "submit_job", fake_submit_job)
        monkeypatch.setattr(
            equities_dl, "target_path_options_definition",
            lambda symbol, dc: tmp_path / "nonexistent.dbn.zst",
        )

        tracker: dict = {"jobs": {}, "pending_downloads": []}

        equities_dl.process_session_day_phase2(
            client=object(),
            tracker=tracker,
            symbol="QQQ",
            date_str="2026-02-06",
            equity_schemas=["mbo"],
            options_schemas=["cmbp-1"],
            log_path=tmp_path / "equities.log",
            pause_seconds=0,
        )

        # No jobs should be submitted
        assert len(calls) == 0


class TestSubmitJobPayloadDetection:
    """Verify submit_job detects payload changes via symbols_digest."""

    def test_resubmits_when_symbols_change(self, monkeypatch, tmp_path) -> None:
        """submit_job should resubmit when symbol list changes."""
        import hashlib

        mock_client = MagicMock()
        mock_client.batch.submit_job.return_value = {"id": "new-job-id"}

        tracker: dict = {"jobs": {}, "pending_downloads": []}
        log_path = tmp_path / "equities.log"

        monkeypatch.setattr(equities_dl, "save_job_tracker", lambda t: None)

        # First submission
        job_id_1 = equities_dl.submit_job(
            client=mock_client,
            tracker=tracker,
            dataset="OPRA.PILLAR",
            schema="cmbp-1",
            symbol="QQQ",
            date_str="2026-02-06",
            symbols=["SYM_A", "SYM_B"],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=equities_dl.PRODUCT_EQUITY_OPTIONS,
        )
        assert job_id_1 == "new-job-id"
        assert mock_client.batch.submit_job.call_count == 1

        # Mark as done
        key = "OPRA.PILLAR|cmbp-1|QQQ|2026-02-06|equity_options"
        tracker["jobs"][key]["state"] = "done"

        # Second submission with different symbols — should resubmit
        mock_client.batch.submit_job.return_value = {"id": "newer-job-id"}
        job_id_2 = equities_dl.submit_job(
            client=mock_client,
            tracker=tracker,
            dataset="OPRA.PILLAR",
            schema="cmbp-1",
            symbol="QQQ",
            date_str="2026-02-06",
            symbols=["SYM_A", "SYM_B", "SYM_C"],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=equities_dl.PRODUCT_EQUITY_OPTIONS,
        )
        assert job_id_2 == "newer-job-id"
        assert mock_client.batch.submit_job.call_count == 2

    def test_skips_when_symbols_unchanged(self, monkeypatch, tmp_path) -> None:
        """submit_job should skip when symbol list is unchanged and state is done."""
        mock_client = MagicMock()
        mock_client.batch.submit_job.return_value = {"id": "job-id"}

        tracker: dict = {"jobs": {}, "pending_downloads": []}
        log_path = tmp_path / "equities.log"

        monkeypatch.setattr(equities_dl, "save_job_tracker", lambda t: None)

        # First submission
        equities_dl.submit_job(
            client=mock_client,
            tracker=tracker,
            dataset="OPRA.PILLAR",
            schema="cmbp-1",
            symbol="QQQ",
            date_str="2026-02-06",
            symbols=["SYM_A", "SYM_B"],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=equities_dl.PRODUCT_EQUITY_OPTIONS,
        )

        # Mark as done
        key = "OPRA.PILLAR|cmbp-1|QQQ|2026-02-06|equity_options"
        tracker["jobs"][key]["state"] = "done"

        # Same symbols — should skip
        result = equities_dl.submit_job(
            client=mock_client,
            tracker=tracker,
            dataset="OPRA.PILLAR",
            schema="cmbp-1",
            symbol="QQQ",
            date_str="2026-02-06",
            symbols=["SYM_A", "SYM_B"],
            stype_in="raw_symbol",
            log_path=log_path,
            product_type=equities_dl.PRODUCT_EQUITY_OPTIONS,
        )
        assert result is None
        assert mock_client.batch.submit_job.call_count == 1


class TestProductTypeConstants:
    """Verify product type constants are defined correctly."""

    def test_product_types_are_distinct(self) -> None:
        types = {
            equities_dl.PRODUCT_EQUITY,
            equities_dl.PRODUCT_EQUITY_OPTIONS,
            equities_dl.PRODUCT_EQUITY_OPTIONS_DEF,
        }
        assert len(types) == 3

    def test_has_pending_jobs_filters_by_product_type(self) -> None:
        tracker: dict = {
            "jobs": {
                "key1": {"state": "submitted", "product_type": equities_dl.PRODUCT_EQUITY_OPTIONS_DEF},
                "key2": {"state": "downloaded", "product_type": equities_dl.PRODUCT_EQUITY},
            },
            "pending_downloads": [],
        }
        assert equities_dl.has_pending_jobs(tracker, equities_dl.PRODUCT_EQUITY_OPTIONS_DEF) is True
        assert equities_dl.has_pending_jobs(tracker, equities_dl.PRODUCT_EQUITY) is False
        assert equities_dl.has_pending_jobs(tracker, equities_dl.PRODUCT_EQUITY_OPTIONS) is False


class TestTargetPathForJob:
    """Verify target_path_for_job routes correctly for all product types."""

    def test_equity_mbo_path(self) -> None:
        path = equities_dl.target_path_for_job("XNAS.ITCH", "mbo", "QQQ", "20260206", equities_dl.PRODUCT_EQUITY)
        assert "product_type=equity_mbo" in str(path)
        assert "symbol=QQQ" in str(path)

    def test_equity_options_def_path(self) -> None:
        path = equities_dl.target_path_for_job("OPRA.PILLAR", "definition", "QQQ", "20260206", equities_dl.PRODUCT_EQUITY_OPTIONS_DEF)
        assert "dataset=definition" in str(path)
        assert "venue=opra" in str(path)

    def test_equity_options_cmbp1_path(self) -> None:
        path = equities_dl.target_path_for_job("OPRA.PILLAR", "cmbp-1", "QQQ", "20260206", equities_dl.PRODUCT_EQUITY_OPTIONS)
        assert "product_type=equity_option_cmbp_1" in str(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
