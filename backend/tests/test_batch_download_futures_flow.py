"""Unit tests for futures batch download flow control."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "scripts"))

import batch_download_futures as futures_dl


# ---------------------------------------------------------------------------
# Front-month resolution tests
# ---------------------------------------------------------------------------

def test_resolve_front_month_extracts_raw_symbol(tmp_path) -> None:
    """Verify resolve_front_month extracts raw_symbol from streamed definition."""
    import pandas as pd

    mock_store = MagicMock()
    mock_store.to_df.return_value = pd.DataFrame({
        "raw_symbol": ["SIH6"],
        "instrument_class": ["F"],
    })

    mock_client = MagicMock()
    mock_client.timeseries.get_range.return_value = mock_store

    result = futures_dl.resolve_front_month(
        client=mock_client,
        symbol="SI",
        session_date="2026-02-06",
        log_path=tmp_path / "futures.log",
    )

    assert result == "SIH6"
    mock_client.timeseries.get_range.assert_called_once_with(
        dataset="GLBX.MDP3",
        symbols=["SI.v.0"],
        stype_in="continuous",
        schema="definition",
        start="2026-02-06",
        end="2026-02-07",
    )


def test_resolve_front_month_raises_on_empty(tmp_path) -> None:
    """Verify resolve_front_month fails clearly when no definition returned."""
    import pandas as pd

    mock_store = MagicMock()
    mock_store.to_df.return_value = pd.DataFrame()

    mock_client = MagicMock()
    mock_client.timeseries.get_range.return_value = mock_store

    with pytest.raises(ValueError, match="No definition"):
        futures_dl.resolve_front_month(
            client=mock_client,
            symbol="FAKE",
            session_date="2026-02-06",
            log_path=tmp_path / "futures.log",
        )


# ---------------------------------------------------------------------------
# MNQ options parent tests (equity index: A/B/C/D suffix for Mon-Thu)
# ---------------------------------------------------------------------------

def test_mnq_options_parents_friday_week1() -> None:
    """Feb 6 2026 = Friday week 1 -> MQ1.OPT only."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-06")
    assert parents == ["MQ1.OPT"]


def test_mnq_options_parents_monday_week2() -> None:
    """Feb 9 2026 = Monday week 2 -> D2A.OPT only."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-09")
    assert parents == ["D2A.OPT"]


def test_mnq_options_parents_tuesday_week1() -> None:
    """Feb 3 2026 = Tuesday week 1 -> D1B.OPT only."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-03")
    assert parents == ["D1B.OPT"]


def test_mnq_options_parents_wednesday_week1() -> None:
    """Feb 4 2026 = Wednesday week 1 -> D1C.OPT only."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-04")
    assert parents == ["D1C.OPT"]


def test_mnq_options_parents_thursday_week1() -> None:
    """Feb 5 2026 = Thursday week 1 -> D1D.OPT only."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-05")
    assert parents == ["D1D.OPT"]


def test_mnq_options_parents_third_friday_quarterly() -> None:
    """Mar 20 2026 = 3rd Friday of quarterly month -> MQ3.OPT + MNQ.OPT."""
    parents = futures_dl.options_parents_for("MNQ", "2026-03-20")
    assert "MQ3.OPT" in parents
    assert "MNQ.OPT" in parents


def test_mnq_options_parents_third_friday_non_quarterly() -> None:
    """Feb 20 2026 = 3rd Friday of non-quarterly month -> MQ3.OPT only (no MNQ.OPT)."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-20")
    assert parents == ["MQ3.OPT"]


def test_mnq_options_parents_last_business_day() -> None:
    """Feb 27 2026 = last business day (Fri) -> MQ4.OPT + MQE.OPT."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-27")
    assert "MQ4.OPT" in parents
    assert "MQE.OPT" in parents


def test_mnq_options_parents_saturday_empty() -> None:
    """Feb 7 2026 = Saturday -> empty list."""
    parents = futures_dl.options_parents_for("MNQ", "2026-02-07")
    assert parents == []


# ---------------------------------------------------------------------------
# ES options parent tests (equity index: E{n}{letter} codes)
# ---------------------------------------------------------------------------

def test_es_friday_week1() -> None:
    """Feb 6 2026 = Friday week 1 -> EW1.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-06")
    assert parents == ["EW1.OPT"]


def test_es_monday_week2() -> None:
    """Feb 9 2026 = Monday week 2 -> E2A.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-09")
    assert parents == ["E2A.OPT"]


def test_es_tuesday_week1() -> None:
    """Feb 3 2026 = Tuesday week 1 -> E1B.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-03")
    assert parents == ["E1B.OPT"]


def test_es_wednesday_week1() -> None:
    """Feb 4 2026 = Wednesday week 1 -> E1C.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-04")
    assert parents == ["E1C.OPT"]


def test_es_thursday_week1() -> None:
    """Feb 5 2026 = Thursday week 1 -> E1D.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-05")
    assert parents == ["E1D.OPT"]


def test_es_third_friday_quarterly() -> None:
    """Mar 20 2026 = 3rd Friday of quarterly month -> EW3.OPT + ES.OPT."""
    parents = futures_dl.options_parents_for("ES", "2026-03-20")
    assert "EW3.OPT" in parents
    assert "ES.OPT" in parents
    assert len(parents) == 2


def test_es_third_friday_non_quarterly() -> None:
    """Feb 20 2026 = 3rd Friday of non-quarterly month -> EW3.OPT only."""
    parents = futures_dl.options_parents_for("ES", "2026-02-20")
    assert parents == ["EW3.OPT"]


def test_es_last_business_day() -> None:
    """Feb 27 2026 = last business day (Fri) -> EW4.OPT + EW.OPT."""
    parents = futures_dl.options_parents_for("ES", "2026-02-27")
    assert "EW4.OPT" in parents
    assert "EW.OPT" in parents


def test_es_saturday_empty() -> None:
    """Feb 7 2026 = Saturday -> empty list."""
    parents = futures_dl.options_parents_for("ES", "2026-02-07")
    assert parents == []


# ---------------------------------------------------------------------------
# GC options parent tests (metals: M/T/W/R suffix for Mon-Thu)
# GC has quarterly=None, so monthly (OG) is the standard option.
# ---------------------------------------------------------------------------

def test_gc_friday_week1() -> None:
    """Feb 6 2026 = Friday week 1 -> OG1.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-06")
    assert parents == ["OG1.OPT"]


def test_gc_monday_week2() -> None:
    """Feb 9 2026 = Monday week 2 -> G2M.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-09")
    assert parents == ["G2M.OPT"]


def test_gc_tuesday_week1() -> None:
    """Feb 3 2026 = Tuesday week 1 -> G1T.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-03")
    assert parents == ["G1T.OPT"]


def test_gc_wednesday_week1() -> None:
    """Feb 4 2026 = Wednesday week 1 -> G1W.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-04")
    assert parents == ["G1W.OPT"]


def test_gc_thursday_week1() -> None:
    """Feb 5 2026 = Thursday week 1 -> G1R.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-05")
    assert parents == ["G1R.OPT"]


def test_gc_third_friday_quarterly() -> None:
    """Mar 20 2026 = 3rd Friday of quarterly month.

    GC has quarterly=None, so the monthly OG parent is included instead
    of a separate quarterly parent. Result: OG3.OPT + OG.OPT.
    """
    parents = futures_dl.options_parents_for("GC", "2026-03-20")
    assert "OG3.OPT" in parents
    assert "OG.OPT" in parents
    assert len(parents) == 2


def test_gc_third_friday_non_quarterly() -> None:
    """Feb 20 2026 = 3rd Friday of non-quarterly month -> OG3.OPT only."""
    parents = futures_dl.options_parents_for("GC", "2026-02-20")
    assert parents == ["OG3.OPT"]


def test_gc_last_business_day() -> None:
    """Feb 27 2026 = last business day (Fri) -> OG4.OPT + OG.OPT."""
    parents = futures_dl.options_parents_for("GC", "2026-02-27")
    assert "OG4.OPT" in parents
    assert "OG.OPT" in parents


def test_gc_saturday_empty() -> None:
    """Feb 7 2026 = Saturday -> empty list."""
    parents = futures_dl.options_parents_for("GC", "2026-02-07")
    assert parents == []


# ---------------------------------------------------------------------------
# SI options parent tests (metals: different prefix pattern)
# ---------------------------------------------------------------------------

def test_si_monday_week1() -> None:
    """Feb 2 2026 = Monday week 1 -> M1S.OPT only."""
    parents = futures_dl.options_parents_for("SI", "2026-02-02")
    assert parents == ["M1S.OPT"]


def test_si_friday_week2() -> None:
    """Feb 13 2026 = Friday week 2 -> SO2.OPT only."""
    parents = futures_dl.options_parents_for("SI", "2026-02-13")
    assert parents == ["SO2.OPT"]


# ---------------------------------------------------------------------------
# CL options parent tests (energy: ML/NL/WL/XL prefix)
# ---------------------------------------------------------------------------

def test_cl_monday_week1() -> None:
    """Feb 2 2026 = Monday week 1 -> ML1.OPT only."""
    parents = futures_dl.options_parents_for("CL", "2026-02-02")
    assert parents == ["ML1.OPT"]


def test_cl_thursday_week1() -> None:
    """Feb 5 2026 = Thursday week 1 -> XL1.OPT only."""
    parents = futures_dl.options_parents_for("CL", "2026-02-05")
    assert parents == ["XL1.OPT"]


def test_cl_friday_week3() -> None:
    """Feb 20 2026 = Friday week 3 -> LO3.OPT only."""
    parents = futures_dl.options_parents_for("CL", "2026-02-20")
    assert parents == ["LO3.OPT"]


# ---------------------------------------------------------------------------
# 6E options parent tests (FX: MO/TU/WE/SU prefix)
# ---------------------------------------------------------------------------

def test_6e_monday_week1() -> None:
    """Feb 2 2026 = Monday week 1 -> MO1.OPT only."""
    parents = futures_dl.options_parents_for("6E", "2026-02-02")
    assert parents == ["MO1.OPT"]


def test_6e_tuesday_week1() -> None:
    """Feb 3 2026 = Tuesday week 1 -> TU1.OPT only."""
    parents = futures_dl.options_parents_for("6E", "2026-02-03")
    assert parents == ["TU1.OPT"]


def test_6e_friday_week1() -> None:
    """Feb 6 2026 = Friday week 1 -> 1EU.OPT only."""
    parents = futures_dl.options_parents_for("6E", "2026-02-06")
    assert parents == ["1EU.OPT"]


# ---------------------------------------------------------------------------
# Unknown symbol fallback
# ---------------------------------------------------------------------------

def test_unknown_symbol_falls_back_to_opt() -> None:
    """Unknown symbols not in OPTIONS_CONFIG fall back to {symbol}.OPT."""
    parents = futures_dl.options_parents_for("ZB", "2026-02-06")
    assert parents == ["ZB.OPT"]


def test_unknown_symbol_fallback_is_date_independent() -> None:
    """Fallback returns same parent regardless of date."""
    assert futures_dl.options_parents_for("ZN", "2026-02-06") == ["ZN.OPT"]
    assert futures_dl.options_parents_for("ZN", "2026-03-20") == ["ZN.OPT"]
    assert futures_dl.options_parents_for("ZN", "2026-02-27") == ["ZN.OPT"]


# ---------------------------------------------------------------------------
# Process session day tests (3-phase architecture)
# ---------------------------------------------------------------------------

def test_process_session_day_phase2_submits_front_month_and_0dte(monkeypatch, tmp_path) -> None:
    """Verify phase2 filters 0DTE and submits futures + options batch jobs."""
    calls: list[dict] = []

    def fake_submit_job(**kwargs):
        calls.append(kwargs)
        return "job-id"

    monkeypatch.setattr(futures_dl, "submit_job", fake_submit_job)

    # Redirect definition path to tmp_path so no real lake artifacts
    monkeypatch.setattr(
        futures_dl, "target_path_options_definition",
        lambda symbol, dc: tmp_path / f"def-{symbol}-{dc}.dbn.zst",
    )
    def_path = tmp_path / "def-SI-20260206.dbn.zst"
    def_path.touch()

    monkeypatch.setattr(
        futures_dl,
        "load_0dte_option_raw_symbols",
        lambda definition_path, active_contract, session_date: ["SIOPT1", "SIOPT2"],
    )

    tracker = {"jobs": {}, "pending_downloads": []}

    futures_dl.process_session_day_phase2(
        client=object(),
        tracker=tracker,
        symbol="SI",
        date_str="2026-02-06",
        active_contract="SIH6",
        include_futures=True,
        options_schemas=["mbo", "statistics"],
        log_path=tmp_path / "futures.log",
        pause_seconds=0,
    )

    assert len(calls) == 3

    # First call: futures MBO for front-month only
    futures_call = calls[0]
    assert futures_call["product_type"] == futures_dl.PRODUCT_FUTURES
    assert futures_call["schema"] == "mbo"
    assert futures_call["stype_in"] == "raw_symbol"
    assert futures_call["symbols"] == ["SIH6"]

    # Remaining calls: options MBO + statistics for 0DTE contracts
    options_calls = calls[1:]
    assert [c["schema"] for c in options_calls] == ["mbo", "statistics"]
    assert all(c["product_type"] == futures_dl.PRODUCT_FUTURES_OPTIONS for c in options_calls)
    assert all(c["stype_in"] == "raw_symbol" for c in options_calls)
    assert all(c["symbols"] == ["SIOPT1", "SIOPT2"] for c in options_calls)


def test_process_session_day_phase2_skips_futures_when_not_included(monkeypatch, tmp_path) -> None:
    """Verify include_futures=False skips the futures MBO job."""
    calls: list[dict] = []

    def fake_submit_job(**kwargs):
        calls.append(kwargs)
        return "job-id"

    monkeypatch.setattr(futures_dl, "submit_job", fake_submit_job)

    # Redirect definition path to tmp_path
    monkeypatch.setattr(
        futures_dl, "target_path_options_definition",
        lambda symbol, dc: tmp_path / f"def-{symbol}-{dc}.dbn.zst",
    )
    def_path = tmp_path / "def-SI-20260206.dbn.zst"
    def_path.touch()

    monkeypatch.setattr(
        futures_dl,
        "load_0dte_option_raw_symbols",
        lambda definition_path, active_contract, session_date: ["SIOPT1"],
    )

    tracker = {"jobs": {}, "pending_downloads": []}

    futures_dl.process_session_day_phase2(
        client=object(),
        tracker=tracker,
        symbol="SI",
        date_str="2026-02-06",
        active_contract="SIH6",
        include_futures=False,
        options_schemas=["mbo", "statistics"],
        log_path=tmp_path / "futures.log",
        pause_seconds=0,
    )

    # Only options jobs, no futures MBO
    assert len(calls) == 2
    assert all(c["product_type"] == futures_dl.PRODUCT_FUTURES_OPTIONS for c in calls)


# ---------------------------------------------------------------------------
# OPTIONS_CONFIG completeness validation
# ---------------------------------------------------------------------------

def test_all_configured_products_have_required_fields() -> None:
    """Every product in OPTIONS_CONFIG has all required fields."""
    required_fields = {"quarterly", "eom", "friday", "daily", "max_weeks"}
    for symbol, cfg in futures_dl.OPTIONS_CONFIG.items():
        missing = required_fields - set(cfg.keys())
        assert not missing, f"{symbol} missing fields: {missing}"


def test_all_daily_codes_have_four_weekdays() -> None:
    """Every product's daily dict covers Mon(0) through Thu(3)."""
    for symbol, cfg in futures_dl.OPTIONS_CONFIG.items():
        daily = cfg["daily"]
        assert set(daily.keys()) == {0, 1, 2, 3}, (
            f"{symbol} daily keys should be {{0,1,2,3}}, got {set(daily.keys())}"
        )


def test_all_daily_codes_match_max_weeks() -> None:
    """Every daily code list has exactly max_weeks entries."""
    for symbol, cfg in futures_dl.OPTIONS_CONFIG.items():
        max_weeks = cfg["max_weeks"]
        for weekday, codes in cfg["daily"].items():
            assert len(codes) == max_weeks, (
                f"{symbol} daily[{weekday}] has {len(codes)} codes, expected {max_weeks}"
            )
