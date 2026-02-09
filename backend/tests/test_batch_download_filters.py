"""Unit tests for batch download contract/0DTE filtering helpers."""

import sys
from pathlib import Path

import pandas as pd
import pytest

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "scripts"))

from batch_download_equities import (
    filter_0dte_equity_option_raw_symbols_from_definitions,
    parse_symbols as parse_equity_symbols,
)
from batch_download_futures import (
    filter_0dte_futures_option_raw_symbols_from_definitions,
    parse_symbols as parse_futures_symbols,
)


def _ts_ns(iso_dt: str) -> int:
    return int(pd.Timestamp(iso_dt, tz="UTC").value)


class TestFuturesOption0DTEFilter:
    def test_filters_to_active_underlying_and_session_date(self) -> None:
        defs = pd.DataFrame(
            {
                "raw_symbol": [
                    "E1A   260109C05000000",
                    "E1A   260109P05000000",
                    "E1A   260110C05000000",
                    "EW1   260109C05000000",
                ],
                "underlying": ["ESH6", "ESH6", "ESH6", "ESM6"],
                "instrument_class": ["C", "P", "C", "C"],
                "expiration": [
                    _ts_ns("2026-01-09T21:00:00"),
                    _ts_ns("2026-01-09T21:00:00"),
                    _ts_ns("2026-01-10T21:00:00"),
                    _ts_ns("2026-01-09T21:00:00"),
                ],
            }
        )

        got = filter_0dte_futures_option_raw_symbols_from_definitions(defs, "ESH6", "2026-01-09")

        assert got == ["E1A   260109C05000000", "E1A   260109P05000000"]

    def test_raises_when_no_matches(self) -> None:
        defs = pd.DataFrame(
            {
                "raw_symbol": ["E1A   260110C05000000"],
                "underlying": ["ESH6"],
                "instrument_class": ["C"],
                "expiration": [_ts_ns("2026-01-10T21:00:00")],
            }
        )

        with pytest.raises(ValueError):
            filter_0dte_futures_option_raw_symbols_from_definitions(defs, "ESH6", "2026-01-09")


class TestEquityOption0DTEFilter:
    def test_uses_utc_expiration_date_for_opra(self) -> None:
        defs = pd.DataFrame(
            {
                "raw_symbol": ["QQQ   260109C00500000", "QQQ   260109P00500000"],
                "underlying": ["QQQ", "QQQ"],
                "instrument_class": ["C", "P"],
                "expiration": [
                    _ts_ns("2026-01-09T00:00:00"),
                    _ts_ns("2026-01-09T00:00:00"),
                ],
            }
        )

        got = filter_0dte_equity_option_raw_symbols_from_definitions(defs, "QQQ", "2026-01-09")

        assert got == ["QQQ   260109C00500000", "QQQ   260109P00500000"]

    def test_rejects_other_underlyings(self) -> None:
        defs = pd.DataFrame(
            {
                "raw_symbol": ["SPY   260109C00500000"],
                "underlying": ["SPY"],
                "instrument_class": ["C"],
                "expiration": [_ts_ns("2026-01-09T00:00:00")],
            }
        )

        with pytest.raises(ValueError):
            filter_0dte_equity_option_raw_symbols_from_definitions(defs, "QQQ", "2026-01-09")


class TestSymbolParsing:
    def test_parse_futures_symbols_accepts_generic_roots(self) -> None:
        got = parse_futures_symbols("ES,SI,CL,6E")
        assert got == ["ES", "SI", "CL", "6E"]

    def test_parse_futures_symbols_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError):
            parse_futures_symbols("SI.FUT")

    def test_parse_equity_symbols_accepts_generic_tickers(self) -> None:
        got = parse_equity_symbols("QQQ,AAPL,BRK.B")
        assert got == ["QQQ", "AAPL", "BRK.B"]

    def test_parse_equity_symbols_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError):
            parse_equity_symbols("QQQ$WEIRD")
