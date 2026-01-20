"""
Arrow/Pydantic schema definitions for Bronze/Silver/Gold data tiers.

This module provides:
- Pydantic models for runtime validation
- PyArrow schemas for Parquet storage
- Schema registry for version tracking

Schemas:

Bronze (raw, normalized):
- options.trades - ES option trades
- futures.mbp10 - ES MBP-10 depth

Silver (cleaned, enriched):
- options.trades_enriched - Option trades with Greeks joined
- features.es_pipeline - Silver tier ML features

Gold (derived analytics):
- levels.signals - Level break/reject signals
- training.es_pipeline - Gold tier ML training datasets

Usage:
    from src.common.schemas import (
        OptionTrade,
        MBP10,
        SchemaRegistry,
    )

    # Get Arrow schema for Parquet writing
    arrow_schema = OptionTrade._arrow_schema

    # List all registered schemas
    SchemaRegistry.list_schemas()
"""

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

from .options_trades import OptionTrade
from .futures_mbp10 import MBP10, BidAskLevelModel

from .options_trades_enriched import OptionTradeEnriched
from .silver_features import SilverFeaturesESPipeline, validate_silver_features

from .levels_signals import (
    LevelSignal,
    LevelKind,
    OutcomeLabel,
    BarrierState,
    Direction,
    Signal,
    Confidence,
    FuelEffect,
    RunwayQuality,
)
from .gold_training import GoldTrainingESPipeline, validate_gold_training

__all__ = [
    'SchemaVersion',
    'SchemaRegistry',
    'BaseEventModel',
    'EventSourceEnum',
    'AggressorEnum',
    'ARROW_TYPE_MAP',
    'build_arrow_schema',
    'pydantic_to_arrow_table',
    'OptionTrade',
    'MBP10',
    'BidAskLevelModel',
    'OptionTradeEnriched',
    'SilverFeaturesESPipeline',
    'validate_silver_features',
    'GoldTrainingESPipeline',
    'validate_gold_training',
    'LevelSignal',
    'LevelKind',
    'OutcomeLabel',
    'BarrierState',
    'Direction',
    'Signal',
    'Confidence',
    'FuelEffect',
    'RunwayQuality',
]
