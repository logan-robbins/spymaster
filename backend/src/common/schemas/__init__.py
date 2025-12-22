"""
Arrow/Pydantic schema definitions for Bronze/Silver/Gold data tiers.

This module provides:
- Pydantic v2 models for runtime validation
- PyArrow schemas for Parquet storage
- Schema registry for version tracking

Schemas per PLAN.md §2.4:

Bronze (raw, normalized):
- stocks.trades.v1 - Stock trades (SPY)
- stocks.quotes.v1 - Stock NBBO quotes (SPY)
- options.trades.v1 - Option trades
- options.greeks_snapshots.v1 - Greeks from REST API
- futures.trades.v1 - ES futures trades
- futures.mbp10.v1 - ES MBP-10 depth

Silver (cleaned, enriched):
- options.trades_enriched.v1 - Option trades with Greeks joined

Gold (derived analytics):
- levels.signals.v1 - Level break/reject signals

Usage:
    from src.common.schemas import (
        StockTradeV1,
        StockQuoteV1,
        OptionTradeV1,
        SchemaRegistry,
    )

    # Create a validated record
    trade = StockTradeV1(
        ts_event_ns=1734567890000000000,
        ts_recv_ns=1734567890001000000,
        source='massive_ws',
        symbol='SPY',
        price=687.50,
        size=100,
    )

    # Get Arrow schema for Parquet writing
    arrow_schema = StockTradeV1._arrow_schema

    # List all registered schemas
    SchemaRegistry.list_schemas()
"""

# Base utilities
from .base import (
    SchemaVersion,
    SchemaRegistry,
    BaseEventModel,
    EventSourceEnum,
    AggressorEnum,
    ARROW_TYPE_MAP,
    build_arrow_schema,
    pydantic_to_arrow_table,
)

# Bronze tier schemas
from .stocks_trades import StockTradeV1
from .stocks_quotes import StockQuoteV1
from .options_trades import OptionTradeV1
from .options_greeks import GreeksSnapshotV1
from .futures_trades import FuturesTradeV1
from .futures_mbp10 import MBP10V1, BidAskLevelModel

# Silver tier schemas
from .options_trades_enriched import OptionTradeEnrichedV1

# Gold tier schemas
from .levels_signals import (
    LevelSignalV1,
    BarrierStateEnum,
    DirectionEnum,
    FuelEffectEnum,
    SignalEnum,
    ConfidenceEnum,
    RunwayQualityEnum,
    LevelKindEnum,
)

__all__ = [
    # Base
    'SchemaVersion',
    'SchemaRegistry',
    'BaseEventModel',
    'EventSourceEnum',
    'AggressorEnum',
    'ARROW_TYPE_MAP',
    'build_arrow_schema',
    'pydantic_to_arrow_table',

    # Bronze
    'StockTradeV1',
    'StockQuoteV1',
    'OptionTradeV1',
    'GreeksSnapshotV1',
    'FuturesTradeV1',
    'MBP10V1',
    'BidAskLevelModel',

    # Silver
    'OptionTradeEnrichedV1',

    # Gold
    'LevelSignalV1',
    'BarrierStateEnum',
    'DirectionEnum',
    'FuelEffectEnum',
    'SignalEnum',
    'ConfidenceEnum',
    'RunwayQualityEnum',
    'LevelKindEnum',
]

# Version information
__version__ = '1.0.0'
__schema_versions__ = {
    'stocks.trades': 1,
    'stocks.quotes': 1,
    'options.trades': 1,
    'options.greeks_snapshots': 1,
    'options.trades_enriched': 1,
    'futures.trades': 1,
    'futures.mbp10': 1,
    'levels.signals': 1,
}
