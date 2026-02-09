"""Unit tests for futures batch download flow control."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "scripts"))

import batch_download_futures as futures_dl


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

    import pytest
    with pytest.raises(ValueError, match="No definition"):
        futures_dl.resolve_front_month(
            client=mock_client,
            symbol="FAKE",
            session_date="2026-02-06",
            log_path=tmp_path / "futures.log",
        )


def test_process_session_day_submits_front_month_and_0dte(monkeypatch, tmp_path) -> None:
    """Verify process_session_day resolves front month, streams defs, and submits batch jobs."""
    calls: list[dict] = []

    def fake_submit_job(**kwargs):
        calls.append(kwargs)
        return "job-id"

    monkeypatch.setattr(futures_dl, "submit_job", fake_submit_job)
    monkeypatch.setattr(
        futures_dl,
        "resolve_front_month",
        lambda client, symbol, session_date, log_path: "SIH6",
    )
    monkeypatch.setattr(
        futures_dl,
        "stream_options_definition",
        lambda client, symbol, session_date, log_path: tmp_path / "fake_def.dbn.zst",
    )
    monkeypatch.setattr(
        futures_dl,
        "load_0dte_option_raw_symbols",
        lambda definition_path, active_contract, session_date: ["SIOPT1", "SIOPT2"],
    )

    tracker = {"jobs": {}, "pending_downloads": []}

    futures_dl.process_session_day(
        client=object(),
        tracker=tracker,
        symbol="SI",
        date_str="2026-02-06",
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


def test_process_session_day_skips_futures_when_not_included(monkeypatch, tmp_path) -> None:
    """Verify --include-futures=False skips the futures MBO job."""
    calls: list[dict] = []

    def fake_submit_job(**kwargs):
        calls.append(kwargs)
        return "job-id"

    monkeypatch.setattr(futures_dl, "submit_job", fake_submit_job)
    monkeypatch.setattr(
        futures_dl,
        "resolve_front_month",
        lambda client, symbol, session_date, log_path: "SIH6",
    )
    monkeypatch.setattr(
        futures_dl,
        "stream_options_definition",
        lambda client, symbol, session_date, log_path: tmp_path / "fake_def.dbn.zst",
    )
    monkeypatch.setattr(
        futures_dl,
        "load_0dte_option_raw_symbols",
        lambda definition_path, active_contract, session_date: ["SIOPT1"],
    )

    tracker = {"jobs": {}, "pending_downloads": []}

    futures_dl.process_session_day(
        client=object(),
        tracker=tracker,
        symbol="SI",
        date_str="2026-02-06",
        include_futures=False,
        options_schemas=["mbo", "statistics"],
        log_path=tmp_path / "futures.log",
        pause_seconds=0,
    )

    # Only options jobs, no futures MBO
    assert len(calls) == 2
    assert all(c["product_type"] == futures_dl.PRODUCT_FUTURES_OPTIONS for c in calls)
