"""
Unit tests for Arrow/Pydantic schema definitions.

Tests:
- Schema validation (valid and invalid inputs)
- Arrow schema generation
- Pydantic-to-Arrow conversion
- Schema registry
"""

import pytest
from datetime import date
import pyarrow as pa

from src.schemas import (
    # Base
    SchemaRegistry,
    SchemaVersion,
    EventSourceEnum,
    AggressorEnum,
    pydantic_to_arrow_table,
    # Bronze
    StockTradeV1,
    StockQuoteV1,
    OptionTradeV1,
    GreeksSnapshotV1,
    FuturesTradeV1,
    MBP10V1,
    BidAskLevelModel,
    # Silver
    OptionTradeEnrichedV1,
    # Gold
    LevelSignalV1,
    BarrierStateEnum,
    DirectionEnum,
    FuelEffectEnum,
    SignalEnum,
    ConfidenceEnum,
    LevelKindEnum,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_ts_event_ns():
    """Valid timestamp: 2025-12-16 12:00:00 UTC"""
    return 1765886400000000000


@pytest.fixture
def valid_ts_recv_ns(valid_ts_event_ns):
    """Valid receive timestamp (1ms after event)"""
    return valid_ts_event_ns + 1_000_000


# =============================================================================
# Test Schema Registry
# =============================================================================

class TestSchemaRegistry:
    """Test schema registry functionality."""

    def test_list_schemas_returns_all_registered(self):
        """All expected schemas should be registered."""
        schemas = SchemaRegistry.list_schemas()
        expected = [
            'stocks.trades.v1',
            'stocks.quotes.v1',
            'options.trades.v1',
            'options.greeks_snapshots.v1',
            'futures.trades.v1',
            'futures.mbp10.v1',
            'options.trades_enriched.v1',
            'levels.signals.v1',
        ]
        for name in expected:
            assert name in schemas, f"Missing schema: {name}"

    def test_get_schema_by_name(self):
        """Can retrieve schema class by full name."""
        cls = SchemaRegistry.get('stocks.trades.v1')
        assert cls is StockTradeV1

    def test_get_arrow_schema_by_name(self):
        """Can retrieve Arrow schema by full name."""
        arrow_schema = SchemaRegistry.get_arrow_schema('stocks.trades.v1')
        assert isinstance(arrow_schema, pa.Schema)
        assert 'ts_event_ns' in arrow_schema.names
        assert 'price' in arrow_schema.names

    def test_get_nonexistent_schema_returns_none(self):
        """Non-existent schema returns None."""
        assert SchemaRegistry.get('nonexistent.v1') is None


class TestSchemaVersion:
    """Test SchemaVersion class."""

    def test_full_name_format(self):
        """Full name should be 'name.vN' format."""
        sv = SchemaVersion('stocks.trades', 1, 'bronze')
        assert sv.full_name == 'stocks.trades.v1'

    def test_repr(self):
        """Repr should show full name and tier."""
        sv = SchemaVersion('stocks.trades', 1, 'bronze')
        assert 'stocks.trades.v1' in repr(sv)
        assert 'bronze' in repr(sv)


# =============================================================================
# Test Bronze Schemas
# =============================================================================

class TestStockTradeV1:
    """Test stocks.trades.v1 schema."""

    def test_valid_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid trade should pass validation."""
        trade = StockTradeV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            symbol='SPY',
            price=687.50,
            size=100,
        )
        assert trade.symbol == 'SPY'
        assert trade.price == 687.50
        assert trade.size == 100

    def test_valid_trade_with_optionals(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Trade with all optional fields."""
        trade = StockTradeV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='SPY',
            price=687.50,
            size=100,
            exchange=4,
            conditions=[0, 12],
            seq=12345,
        )
        assert trade.exchange == 4
        assert trade.conditions == [0, 12]
        assert trade.seq == 12345

    def test_invalid_price_zero(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Price must be > 0."""
        with pytest.raises(ValueError):
            StockTradeV1(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                price=0,  # Invalid
                size=100,
            )

    def test_invalid_size_zero(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Size must be >= 1."""
        with pytest.raises(ValueError):
            StockTradeV1(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                price=687.50,
                size=0,  # Invalid
            )

    def test_invalid_timestamp_too_old(self, valid_ts_recv_ns):
        """Timestamp before year 2000 should fail."""
        with pytest.raises(ValueError, match="outside valid range"):
            StockTradeV1(
                ts_event_ns=100_000_000,  # Very old
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                price=687.50,
                size=100,
            )

    def test_arrow_schema_fields(self):
        """Arrow schema should have correct fields and types."""
        schema = StockTradeV1._arrow_schema
        assert schema.field('ts_event_ns').type == pa.int64()
        assert schema.field('price').type == pa.float64()
        assert schema.field('size').type == pa.int32()
        assert schema.field('exchange').nullable is True
        assert schema.field('symbol').nullable is False


class TestStockQuoteV1:
    """Test stocks.quotes.v1 schema."""

    def test_valid_quote(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid quote should pass validation."""
        quote = StockQuoteV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            symbol='SPY',
            bid_px=687.48,
            ask_px=687.52,
            bid_sz=500,
            ask_sz=300,
        )
        assert quote.bid_px == 687.48
        assert quote.ask_px == 687.52

    def test_crossed_quote_raises_error(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Crossed quote (ask < bid) should fail."""
        with pytest.raises(ValueError, match="Crossed quote"):
            StockQuoteV1(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                bid_px=687.52,
                ask_px=687.48,  # ask < bid
                bid_sz=500,
                ask_sz=300,
            )

    def test_zero_bid_ask_allowed(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Zero bid/ask is allowed (halted or no quotes)."""
        quote = StockQuoteV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            symbol='SPY',
            bid_px=0.0,
            ask_px=0.0,
            bid_sz=0,
            ask_sz=0,
        )
        assert quote.bid_px == 0.0


class TestOptionTradeV1:
    """Test options.trades.v1 schema."""

    def test_valid_option_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid option trade should pass."""
        trade = OptionTradeV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            underlying='SPY',
            option_symbol='O:SPY251216C00687000',
            exp_date=date(2025, 12, 16),
            strike=687.0,
            right='C',
            price=1.50,
            size=10,
        )
        assert trade.underlying == 'SPY'
        assert trade.strike == 687.0
        assert trade.right == 'C'

    def test_valid_put_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Put option should pass."""
        trade = OptionTradeV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            underlying='SPY',
            option_symbol='O:SPY251216P00680000',
            exp_date=date(2025, 12, 16),
            strike=680.0,
            right='P',
            price=2.00,
            size=5,
            aggressor=AggressorEnum.SELL,
        )
        assert trade.right == 'P'
        assert trade.aggressor == AggressorEnum.SELL

    def test_invalid_right(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Right must be 'C' or 'P'."""
        with pytest.raises(ValueError):
            OptionTradeV1(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                underlying='SPY',
                option_symbol='O:SPY251216X00687000',
                exp_date=date(2025, 12, 16),
                strike=687.0,
                right='X',  # Invalid
                price=1.50,
                size=10,
            )


class TestGreeksSnapshotV1:
    """Test options.greeks_snapshots.v1 schema."""

    def test_valid_greeks(self, valid_ts_event_ns):
        """Valid Greeks snapshot should pass."""
        greeks = GreeksSnapshotV1(
            ts_event_ns=valid_ts_event_ns,
            source=EventSourceEnum.MASSIVE_REST,
            underlying='SPY',
            option_symbol='O:SPY251216C00687000',
            delta=0.55,
            gamma=0.08,
            theta=-0.15,
            vega=0.12,
        )
        assert greeks.delta == 0.55
        assert greeks.gamma == 0.08

    def test_delta_range_validation(self, valid_ts_event_ns):
        """Delta must be between -1 and +1."""
        with pytest.raises(ValueError):
            GreeksSnapshotV1(
                ts_event_ns=valid_ts_event_ns,
                source=EventSourceEnum.MASSIVE_REST,
                underlying='SPY',
                option_symbol='O:SPY251216C00687000',
                delta=1.5,  # Invalid
                gamma=0.08,
                theta=-0.15,
                vega=0.12,
            )


class TestFuturesTradeV1:
    """Test futures.trades.v1 schema."""

    def test_valid_es_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid ES trade should pass."""
        trade = FuturesTradeV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='ES',
            price=6875.25,
            size=5,
            aggressor=AggressorEnum.BUY,
        )
        assert trade.symbol == 'ES'
        assert trade.price == 6875.25
        assert trade.aggressor == AggressorEnum.BUY


class TestMBP10V1:
    """Test futures.mbp10.v1 schema."""

    def test_valid_mbp10(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid MBP-10 should pass."""
        mbp = MBP10V1(
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
        """Test from_levels factory method."""
        levels = [
            BidAskLevelModel(bid_px=6875.00, bid_sz=100, ask_px=6875.25, ask_sz=150),
            BidAskLevelModel(bid_px=6874.75, bid_sz=200, ask_px=6875.50, ask_sz=100),
        ]
        mbp = MBP10V1.from_levels(
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
        """Test get_bid_levels helper."""
        mbp = MBP10V1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.REPLAY,
            symbol='ES',
            bid_px_1=6875.00,
            bid_sz_1=100,
            bid_px_2=6874.75,
            bid_sz_2=200,
            ask_px_1=6875.25,
            ask_sz_1=150,
        )
        bid_levels = mbp.get_bid_levels()
        assert len(bid_levels) == 2
        assert bid_levels[0] == (6875.00, 100)
        assert bid_levels[1] == (6874.75, 200)


# =============================================================================
# Test Silver Schema
# =============================================================================

class TestOptionTradeEnrichedV1:
    """Test options.trades_enriched.v1 schema."""

    def test_valid_enriched_trade(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Valid enriched trade should pass."""
        trade = OptionTradeEnrichedV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            underlying='SPY',
            option_symbol='O:SPY251216C00687000',
            exp_date=date(2025, 12, 16),
            strike=687.0,
            right='C',
            price=1.50,
            size=10,
            greeks_snapshot_id='abc123',
            delta=0.55,
            gamma=0.08,
        )
        assert trade.delta == 0.55
        assert trade.gamma == 0.08

    def test_compute_notionals(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Test compute_notionals helper."""
        trade = OptionTradeEnrichedV1(
            ts_event_ns=valid_ts_event_ns,
            ts_recv_ns=valid_ts_recv_ns,
            source=EventSourceEnum.MASSIVE_WS,
            underlying='SPY',
            option_symbol='O:SPY251216C00687000',
            exp_date=date(2025, 12, 16),
            strike=687.0,
            right='C',
            price=1.50,
            size=10,
            delta=0.55,
            gamma=0.08,
        )
        trade.compute_notionals()
        assert trade.delta_notional == 0.55 * 10 * 100
        assert trade.gamma_notional == 0.08 * 10 * 100


# =============================================================================
# Test Gold Schema
# =============================================================================

class TestLevelSignalV1:
    """Test levels.signals.v1 schema."""

    def test_valid_level_signal(self, valid_ts_event_ns):
        """Valid level signal should pass."""
        signal = LevelSignalV1(
            ts_event_ns=valid_ts_event_ns,
            underlying='SPY',
            spot=687.50,
            bid=687.48,
            ask=687.52,
            level_id='ROUND_687',
            level_kind=LevelKindEnum.ROUND,
            level_price=687.0,
            direction=DirectionEnum.SUPPORT,
            distance=0.50,
            break_score_raw=75.5,
            break_score_smooth=72.0,
            signal=SignalEnum.CONTESTED,
            confidence=ConfidenceEnum.MEDIUM,
            barrier_state=BarrierStateEnum.CONSUMED,
        )
        assert signal.level_id == 'ROUND_687'
        assert signal.break_score_raw == 75.5
        assert signal.signal == SignalEnum.CONTESTED

    def test_full_level_signal(self, valid_ts_event_ns):
        """Level signal with all fields populated."""
        signal = LevelSignalV1(
            ts_event_ns=valid_ts_event_ns,
            underlying='SPY',
            spot=687.50,
            bid=687.48,
            ask=687.52,
            level_id='STRIKE_685',
            level_kind=LevelKindEnum.STRIKE,
            level_price=685.0,
            direction=DirectionEnum.SUPPORT,
            distance=2.50,
            break_score_raw=88.0,
            break_score_smooth=85.0,
            signal=SignalEnum.BREAK,
            confidence=ConfidenceEnum.HIGH,
            barrier_state=BarrierStateEnum.VACUUM,
            barrier_delta_liq=-5000.0,
            barrier_replenishment_ratio=0.15,
            barrier_added=1000,
            barrier_canceled=5000,
            barrier_filled=1000,
            tape_imbalance=-0.45,
            tape_buy_vol=50000,
            tape_sell_vol=150000,
            tape_velocity=-0.08,
            tape_sweep_detected=True,
            tape_sweep_direction='DOWN',
            tape_sweep_notional=1250000.0,
            fuel_effect=FuelEffectEnum.AMPLIFY,
            fuel_net_dealer_gamma=-185000.0,
            fuel_call_wall=690.0,
            fuel_put_wall=680.0,
            fuel_hvl=685.0,
            runway_direction='DOWN',
            runway_next_level_id='PUT_WALL',
            runway_next_level_price=680.0,
            runway_distance=5.0,
            runway_quality='CLEAR',
            note='Vacuum + dealers chase; sweep confirms',
        )
        assert signal.barrier_state == BarrierStateEnum.VACUUM
        assert signal.tape_sweep_detected is True
        assert signal.fuel_effect == FuelEffectEnum.AMPLIFY

    def test_break_score_range(self, valid_ts_event_ns):
        """Break score must be 0-100."""
        with pytest.raises(ValueError):
            LevelSignalV1(
                ts_event_ns=valid_ts_event_ns,
                underlying='SPY',
                spot=687.50,
                bid=687.48,
                ask=687.52,
                level_id='ROUND_687',
                level_kind=LevelKindEnum.ROUND,
                level_price=687.0,
                direction=DirectionEnum.SUPPORT,
                distance=0.50,
                break_score_raw=150.0,  # Invalid: > 100
                break_score_smooth=72.0,
                signal=SignalEnum.NEUTRAL,
                confidence=ConfidenceEnum.LOW,
                barrier_state=BarrierStateEnum.NEUTRAL,
            )


# =============================================================================
# Test Pydantic to Arrow Conversion
# =============================================================================

class TestPydanticToArrow:
    """Test conversion from Pydantic models to Arrow tables."""

    def test_stock_trades_to_arrow(self, valid_ts_event_ns, valid_ts_recv_ns):
        """Convert stock trades to Arrow table."""
        trades = [
            StockTradeV1(
                ts_event_ns=valid_ts_event_ns,
                ts_recv_ns=valid_ts_recv_ns,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                price=687.50,
                size=100,
            ),
            StockTradeV1(
                ts_event_ns=valid_ts_event_ns + 1_000_000,
                ts_recv_ns=valid_ts_recv_ns + 1_000_000,
                source=EventSourceEnum.MASSIVE_WS,
                symbol='SPY',
                price=687.52,
                size=200,
            ),
        ]
        table = pydantic_to_arrow_table(trades, StockTradeV1._arrow_schema)
        assert table.num_rows == 2
        assert table.column('price').to_pylist() == [687.50, 687.52]
        assert table.column('size').to_pylist() == [100, 200]

    def test_empty_list_to_arrow(self):
        """Empty list should produce empty table with schema."""
        table = pydantic_to_arrow_table([], StockTradeV1._arrow_schema)
        assert table.num_rows == 0
        assert table.schema == StockTradeV1._arrow_schema
