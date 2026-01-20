"""
Unit tests for Arrow/Pydantic schema definitions.
"""

import pytest
from datetime import date
import pyarrow as pa

from src.common.schemas import (
    SchemaRegistry,
    SchemaVersion,
    EventSourceEnum,
    AggressorEnum,
    pydantic_to_arrow_table,
    OptionTrade,
    MBP10,
    BidAskLevelModel,
    LevelSignal,
    LevelKind,
    Direction,
    Signal,
    Confidence,
    FuelEffect,
    BarrierState,
    OutcomeLabel,
)


@pytest.fixture
def valid_ts_event_ns():
    return 1765886400000000000


@pytest.fixture
def valid_ts_recv_ns(valid_ts_event_ns):
    return valid_ts_event_ns + 1_000_000


class TestSchemaRegistry:

    def test_list_schemas_returns_registered(self):
        schemas = SchemaRegistry.list_schemas()
        assert any('options.trades' in s for s in schemas)
        assert any('futures.mbp10' in s for s in schemas)

    def test_get_nonexistent_returns_none(self):
        assert SchemaRegistry.get('nonexistent') is None


class TestSchemaVersion:

    def test_full_name_includes_version(self):
        sv = SchemaVersion('options.trades', 1, 'bronze')
        assert 'options.trades' in sv.full_name

    def test_repr(self):
        sv = SchemaVersion('options.trades', 1, 'bronze')
        assert 'options.trades' in repr(sv)
        assert 'bronze' in repr(sv)


class TestMBP10:

    def test_valid_mbp10(self, valid_ts_event_ns, valid_ts_recv_ns):
        mbp = MBP10(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='ES',
            bid_px_1=6875.00,
            bid_sz_1=100,
            ask_px_1=6875.25,
            ask_sz_1=150,
        )
        assert mbp.bid_px_1 == 6875.00
        assert mbp.ask_sz_1 == 150

    def test_from_levels_factory(self, valid_ts_event_ns, valid_ts_recv_ns):
        levels = [
            BidAskLevelModel(bid_px=6875.00, bid_sz=100, ask_px=6875.25, ask_sz=150),
            BidAskLevelModel(bid_px=6874.75, bid_sz=200, ask_px=6875.50, ask_sz=100),
        ]
        mbp = MBP10.from_levels(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='ES',
            levels=levels,
            is_snapshot=True,
        )
        assert mbp.bid_px_1 == 6875.00
        assert mbp.bid_px_2 == 6874.75
        assert mbp.is_snapshot is True

    def test_get_bid_levels(self, valid_ts_event_ns, valid_ts_recv_ns):
        mbp = MBP10(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='ES',
            bid_px_1=100.0, bid_sz_1=10,
            bid_px_2=99.0, bid_sz_2=20,
            ask_px_1=101.0, ask_sz_1=15,
        )
        bid_levels = mbp.get_bid_levels()
        assert len(bid_levels) == 2
        assert bid_levels[0] == (100.0, 10)
        assert bid_levels[1] == (99.0, 20)


class TestOptionTrade:

    def test_valid_option_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        trade = OptionTrade(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            underlying='ES',
            option_symbol='ESZ5 C6900',
            exp_date=date(2025, 12, 19),
            strike=6900.0,
            right='C',
            price=45.50,
            size=10,
        )
        assert trade.underlying == 'ES'
        assert trade.strike == 6900.0
        assert trade.right == 'C'

    def test_invalid_right_raises(self, valid_ts_event_ns, valid_ts_recv_ns):
        with pytest.raises(ValueError):
            OptionTrade(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.REPLAY,
                underlying='ES',
                option_symbol='ESZ5 X6900',
                exp_date=date(2025, 12, 19),
                strike=6900.0,
                right='X',
                price=45.50,
                size=10,
            )


class TestLevelSignal:

    def test_valid_level_signal(self, valid_ts_event_ns):
        signal = LevelSignal(
            event_id='test_001',
            ts_event_ns=valid_ts_event_ns,
            level_price=6800.0,
            level_kind=LevelKind.PM_HIGH,
            direction=Direction.UP,
            signal=Signal.BREAK,
            confidence=Confidence.HIGH,
            fuel_effect=FuelEffect.AMPLIFY,
            barrier_state=BarrierState.VACUUM,
        )
        assert signal.level_price == 6800.0
        assert signal.signal == Signal.BREAK

    def test_json_roundtrip(self, valid_ts_event_ns):
        signal = LevelSignal(
            event_id='test_002',
            ts_event_ns=valid_ts_event_ns,
            level_price=6850.0,
            level_kind=LevelKind.PM_LOW,
            direction=Direction.DOWN,
            signal=Signal.REJECT,
            confidence=Confidence.MEDIUM,
            fuel_effect=FuelEffect.NEUTRAL,
            barrier_state=BarrierState.WALL,
        )
        signal_dict = signal.model_dump()
        signal_restored = LevelSignal(**signal_dict)
        assert signal_restored.event_id == signal.event_id
        assert signal_restored.level_price == signal.level_price


class TestPydanticToArrow:

    def test_convert_option_trades(self, valid_ts_event_ns, valid_ts_recv_ns):
        trades = [
            OptionTrade(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.REPLAY,
                underlying='ES',
                option_symbol='ESZ5 C6900',
                exp_date=date(2025, 12, 19),
                strike=6900.0,
                right='C',
                price=45.50,
                size=10,
            ),
            OptionTrade(
                ts_event_ns=valid_ts_event_ns + 1000,
                ts_recv_ns=valid_ts_recv_ns + 1000,
                source=EventSourceEnum.REPLAY,
                underlying='ES',
                option_symbol='ESZ5 P6800',
                exp_date=date(2025, 12, 19),
                strike=6800.0,
                right='P',
                price=32.25,
                size=5,
            ),
        ]
        table = pydantic_to_arrow_table(trades, OptionTrade._arrow_schema)
        assert len(table) == 2
        assert 'underlying' in table.column_names

    def test_empty_list_returns_empty_table(self):
        table = pydantic_to_arrow_table([], OptionTrade._arrow_schema)
        assert len(table) == 0
        assert table.schema == OptionTrade._arrow_schema
