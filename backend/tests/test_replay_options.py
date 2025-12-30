import pytest

from src.io.bronze import BronzeReader


def test_bronze_option_trades_available():
    reader = BronzeReader()
    dates = reader.get_available_dates("options/trades", "underlying=SPY")
    if not dates:
        pytest.skip("No Bronze option trades available")

    df = reader.read_option_trades(date=dates[-1], underlying="SPY")
    assert not df.empty, "Expected option trades in Bronze"

    required_cols = {
        "ts_event_ns",
        "ts_recv_ns",
        "underlying",
        "option_symbol",
        "exp_date",
        "strike",
        "right",
        "price",
        "size"
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns in Bronze option trades: {missing}"
