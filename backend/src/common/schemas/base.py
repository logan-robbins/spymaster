"""
Base schema utilities for Arrow/Pydantic schema definitions.

Provides:
- SchemaRegistry for version tracking and validation
- Base Pydantic model configuration
- Arrow schema utilities and type mappings
- Conversion helpers between Pydantic models and Arrow tables
"""

from typing import Dict, Type, Any, Optional, List, ClassVar
from datetime import date
from enum import Enum

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, field_validator


class SchemaVersion:
    """Track schema version metadata."""

    def __init__(self, name: str, version: int, tier: str):
        """
        Args:
            name: Schema name (e.g., 'futures.trades')
            version: Version number (e.g., 1)
            tier: Data tier ('bronze', 'silver', 'gold')
        """
        self.name = name
        self.version = version
        self.tier = tier

    @property
    def full_name(self) -> str:
        """Returns 'futures.trades.v1' format."""
        return f"{self.name}.v{self.version}"

    def __repr__(self) -> str:
        return f"SchemaVersion({self.full_name!r}, tier={self.tier!r})"


class BaseEventModel(BaseModel):
    """
    Base Pydantic model for all event schemas.

    Provides common configuration and validation.
    """
    model_config = ConfigDict(
        # Validate on assignment for runtime safety
        validate_assignment=True,
        # Enforce schema strictness to keep a single canonical contract
        extra='forbid',
        # Use enum values in serialization
        use_enum_values=True,
        # Strict type coercion
        strict=False,
        # Allow population by field name
        populate_by_name=True,
    )

    # Override in subclasses
    _schema_version: ClassVar[SchemaVersion]
    _arrow_schema: ClassVar[pa.Schema]


class EventSourceEnum(str, Enum):
    """Event source enumeration (string-based for Pydantic serialization)."""
    REPLAY = "replay"
    SIM = "sim"
    DIRECT_FEED = "direct_feed"


class AggressorEnum(int, Enum):
    """Trade aggressor side (int-based for storage efficiency)."""
    BUY = 1
    SELL = -1
    MID = 0


# Arrow type mappings for common fields
ARROW_TYPE_MAP: Dict[str, pa.DataType] = {
    # Timestamps (nanoseconds)
    'ts_event_ns': pa.int64(),
    'ts_recv_ns': pa.int64(),
    'confirm_ts_ns': pa.int64(),

    # Identifiers
    'symbol': pa.utf8(),
    'underlying': pa.utf8(),
    'option_symbol': pa.utf8(),
    'source': pa.utf8(),  # enum stored as string

    # Prices
    'price': pa.float64(),
    'bid_px': pa.float64(),
    'ask_px': pa.float64(),
    'opt_bid': pa.float64(),
    'opt_ask': pa.float64(),
    'strike': pa.float64(),
    'spot': pa.float64(),
    'bid': pa.float64(),
    'ask': pa.float64(),
    'anchor_spot': pa.float64(),

    # Sizes
    'size': pa.int32(),
    'bid_sz': pa.int32(),
    'ask_sz': pa.int32(),

    # Exchange codes
    'exchange': pa.int16(),
    'bid_exch': pa.int16(),
    'ask_exch': pa.int16(),

    # Trade conditions
    'conditions': pa.list_(pa.int16()),

    # Sequence numbers
    'seq': pa.int64(),

    # Aggressor
    'aggressor': pa.int8(),

    # Greeks
    'delta': pa.float64(),
    'gamma': pa.float64(),
    'theta': pa.float64(),
    'vega': pa.float64(),
    'implied_volatility': pa.float64(),
    'open_interest': pa.int64(),

    # Dates
    'exp_date': pa.date32(),

    # Option right
    'right': pa.utf8(),

    # Boolean flags
    'is_snapshot': pa.bool_(),

    # String fields
    'snapshot_id': pa.utf8(),
    'level_id': pa.utf8(),
    'level_kind': pa.utf8(),
    'direction': pa.utf8(),
    'signal': pa.utf8(),
    'confidence': pa.utf8(),
    'note': pa.utf8(),

    # Scores
    'break_score_raw': pa.float64(),
    'break_score_smooth': pa.float64(),
    'distance': pa.float64(),
    'distance_signed': pa.float64(),
    'atr': pa.float64(),
    'distance_atr': pa.float64(),
    'distance_pct': pa.float64(),
    'distance_signed_atr': pa.float64(),
    'distance_signed_pct': pa.float64(),
    'level_price_pct': pa.float64(),
    'level_price': pa.float64(),

    # Barrier metrics
    'barrier_state': pa.utf8(),
    'barrier_delta_liq': pa.float64(),
    'barrier_replenishment_ratio': pa.float64(),
    'barrier_replenishment_trend': pa.float64(),
    'barrier_delta_liq_trend': pa.float64(),
    'barrier_added': pa.int64(),
    'barrier_canceled': pa.int64(),
    'barrier_filled': pa.int64(),
    'barrier_delta_liq_nonzero': pa.int8(),
    'barrier_delta_liq_log': pa.float64(),

    # Tape metrics
    'tape_imbalance': pa.float64(),
    'tape_buy_vol': pa.int64(),
    'tape_sell_vol': pa.int64(),
    'tape_velocity': pa.float64(),
    'tape_sweep_detected': pa.bool_(),
    'tape_sweep_direction': pa.utf8(),
    'tape_sweep_notional': pa.float64(),
    'tape_velocity_trend': pa.float64(),
    'tape_imbalance_trend': pa.float64(),

    # Fuel metrics
    'fuel_effect': pa.utf8(),
    'fuel_net_dealer_gamma': pa.float64(),
    'fuel_call_wall': pa.float64(),
    'fuel_put_wall': pa.float64(),
    'fuel_hvl': pa.float64(),
    'gamma_bucket': pa.utf8(),

    # Runway metrics
    'runway_direction': pa.utf8(),
    'runway_next_level_id': pa.utf8(),
    'runway_next_level_price': pa.float64(),
    'runway_distance': pa.float64(),
    'runway_quality': pa.utf8(),

    # Labels and derived outcomes
    'tradeable_1': pa.int8(),
    'tradeable_2': pa.int8(),
    'direction_sign': pa.int8(),
    'attempt_index': pa.int32(),
    'attempt_cluster_id': pa.int32(),
    'confluence_alignment': pa.int8(),
    'wall_ratio_nonzero': pa.int8(),
    'wall_ratio_log': pa.float64(),

    # MBP-10 level prices and sizes (flattened)
    **{f'bid_px_{i}': pa.float64() for i in range(1, 11)},
    **{f'ask_px_{i}': pa.float64() for i in range(1, 11)},
    **{f'bid_sz_{i}': pa.int32() for i in range(1, 11)},
    **{f'ask_sz_{i}': pa.int32() for i in range(1, 11)},
}


def build_arrow_schema(
    fields: List[tuple],
    metadata: Optional[Dict[str, str]] = None
) -> pa.Schema:
    """
    Build an Arrow schema from field definitions.

    Args:
        fields: List of (name, type, nullable) tuples
        metadata: Optional schema-level metadata

    Returns:
        PyArrow Schema
    """
    arrow_fields = []
    for field_def in fields:
        if len(field_def) == 2:
            name, dtype = field_def
            nullable = True
        else:
            name, dtype, nullable = field_def

        # Resolve string type names from map
        if isinstance(dtype, str):
            dtype = ARROW_TYPE_MAP.get(dtype, pa.utf8())

        arrow_fields.append(pa.field(name, dtype, nullable=nullable))

    return pa.schema(arrow_fields, metadata=metadata)


def pydantic_to_arrow_table(
    records: List[BaseEventModel],
    schema: pa.Schema
) -> pa.Table:
    """
    Convert a list of Pydantic models to an Arrow Table.

    Args:
        records: List of Pydantic model instances
        schema: Arrow schema for the table

    Returns:
        PyArrow Table
    """
    if not records:
        return pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)

    # Convert models to dicts
    data = {}
    for field in schema:
        field_name = field.name
        values = []
        for record in records:
            val = getattr(record, field_name, None)
            # Handle enum conversion
            if isinstance(val, Enum):
                val = val.value
            # Handle date conversion
            if isinstance(val, date):
                val = val.isoformat() if pa.types.is_string(field.type) else val
            values.append(val)
        data[field_name] = values

    return pa.Table.from_pydict(data, schema=schema)


class SchemaRegistry:
    """
    Registry for all schema versions.

    Provides lookup by name/version and validation utilities.
    """

    _schemas: Dict[str, Type[BaseEventModel]] = {}
    _arrow_schemas: Dict[str, pa.Schema] = {}

    @classmethod
    def register(cls, model_class: Type[BaseEventModel]) -> Type[BaseEventModel]:
        """Register a schema model class."""
        schema_version = model_class._schema_version
        cls._schemas[schema_version.full_name] = model_class
        if hasattr(model_class, '_arrow_schema'):
            cls._arrow_schemas[schema_version.full_name] = model_class._arrow_schema
        return model_class

    @classmethod
    def get(cls, schema_name: str) -> Optional[Type[BaseEventModel]]:
        """Get schema model by full name (e.g., 'futures.trades.v1')."""
        return cls._schemas.get(schema_name)

    @classmethod
    def get_arrow_schema(cls, schema_name: str) -> Optional[pa.Schema]:
        """Get Arrow schema by full name."""
        # First check cached schemas
        if schema_name in cls._arrow_schemas:
            return cls._arrow_schemas[schema_name]
        # Fall back to looking up from model class (handles late binding)
        model_class = cls._schemas.get(schema_name)
        if model_class and hasattr(model_class, '_arrow_schema'):
            return model_class._arrow_schema
        return None

    @classmethod
    def list_schemas(cls) -> List[str]:
        """List all registered schema names."""
        return list(cls._schemas.keys())
