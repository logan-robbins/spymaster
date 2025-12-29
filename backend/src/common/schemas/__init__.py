"""
Arrow/Pydantic schema definitions for Bronze/Silver/Gold data tiers.

This module provides:
- Pydantic v2 models for runtime validation
- PyArrow schemas for Parquet storage
- Schema registry for version tracking

Schemas per PLAN.md §2.4:

Bronze (raw, normalized):
- options.trades.v1 - ES option trades
- futures.trades.v1 - ES futures trades
- futures.mbp10.v1 - ES MBP-10 depth

Silver (cleaned, enriched):
- options.trades_enriched.v1 - Option trades with Greeks joined

Gold (derived analytics):
- levels.signals.v1 - Level break/reject signals

Usage:
    from src.common.schemas import (
        OptionTradeV1,
        FuturesTradeV1,
        SchemaRegistry,
    )

    # Create a validated record
    trade = FuturesTradeV1(
        ts_event_ns=1734567890000000000,
        ts_recv_ns=1734567890001000000,
        source='direct_feed',
        symbol='ES',
        price=6870.50,
        size=5,
    )

    # Get Arrow schema for Parquet writing
    arrow_schema = FuturesTradeV1._arrow_schema

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
from .options_trades import OptionTradeV1
from .futures_trades import FuturesTradeV1
from .futures_mbp10 import MBP10V1, BidAskLevelModel

# Silver tier schemas
from .options_trades_enriched import OptionTradeEnrichedV1
from .silver_features import SilverFeaturesESPipelineV1, validate_silver_features

# Gold tier schemas
from .levels_signals import (
    LevelSignalV1,
    LevelKind,
    OutcomeLabel,
    BarrierState,
    Direction,
    Signal,
    Confidence,
    FuelEffect,
    RunwayQuality,
)
from .gold_training import GoldTrainingESPipelineV1, validate_gold_training

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
    'OptionTradeV1',
    'FuturesTradeV1',
    'MBP10V1',
    'BidAskLevelModel',
    
    # Silver
    'OptionTradeEnrichedV1',
    'SilverFeaturesESPipelineV1',
    'validate_silver_features',
    
    # Gold
    'GoldTrainingESPipelineV1',
    'validate_gold_training',
    'LevelSignalV1',
    'LevelKind',
    'OutcomeLabel',
    'BarrierState',
    'Direction',
    'Signal',
    'Confidence',
    'FuelEffect',
    'RunwayQuality',
]

# Version information
__version__ = '1.0.0'
__schema_versions__ = {
    'options.trades': 1,
    'options.trades_enriched': 1,
    'futures.trades': 1,
    'futures.mbp10': 1,
    'levels.signals': 1,
}
