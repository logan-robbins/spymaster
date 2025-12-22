"""
stocks.trades.v1 schema - Bronze tier stock trades.

Fields per PLAN.md §2.4:
- ts_event_ns: int64 (UTC nanoseconds)
- ts_recv_ns: int64 (UTC nanoseconds)
- source: utf8 (EventSource enum value)
- symbol: utf8 (e.g., 'SPY')
- price: float64
- size: int32
- exchange: int16 (optional)
- conditions: list<int16> (optional)
- seq: int64 (optional)
"""

from typing import Optional, List, ClassVar

import pyarrow as pa
from pydantic import Field, field_validator

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    EventSourceEnum,
    build_arrow_schema,
)


@SchemaRegistry.register
class StockTradeV1(BaseEventModel):
    """
    Bronze tier stock trade event.

    Represents a single stock trade with vendor metadata.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='stocks.trades',
        version=1,
        tier='bronze'
    )

    # Required fields
    ts_event_ns: int = Field(
        ...,
        description="Event timestamp in Unix nanoseconds UTC"
    )
    ts_recv_ns: int = Field(
        ...,
        description="Receive timestamp in Unix nanoseconds UTC"
    )
    source: EventSourceEnum = Field(
        ...,
        description="Event source (massive_ws, polygon_ws, replay, etc.)"
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock symbol (e.g., 'SPY')"
    )
    price: float = Field(
        ...,
        gt=0,
        description="Trade price"
    )
    size: int = Field(
        ...,
        ge=1,
        description="Trade size in shares"
    )

    # Optional fields
    exchange: Optional[int] = Field(
        default=None,
        description="Exchange code if available"
    )
    conditions: Optional[List[int]] = Field(
        default=None,
        description="Vendor-specific trade condition codes"
    )
    seq: Optional[int] = Field(
        default=None,
        ge=0,
        description="Monotonic sequence number for ordering diagnostics"
    )

    @field_validator('ts_event_ns', 'ts_recv_ns')
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is reasonable (after year 2000, before 2100)."""
        min_ts = 946_684_800_000_000_000  # 2000-01-01
        max_ts = 4_102_444_800_000_000_000  # 2100-01-01
        if not min_ts <= v <= max_ts:
            raise ValueError(f"Timestamp {v} outside valid range")
        return v


# Arrow schema definition
StockTradeV1._arrow_schema = build_arrow_schema(
    fields=[
        ('ts_event_ns', pa.int64(), False),
        ('ts_recv_ns', pa.int64(), False),
        ('source', pa.utf8(), False),
        ('symbol', pa.utf8(), False),
        ('price', pa.float64(), False),
        ('size', pa.int32(), False),
        ('exchange', pa.int16(), True),
        ('conditions', pa.list_(pa.int16()), True),
        ('seq', pa.int64(), True),
    ],
    metadata={
        'schema_name': 'stocks.trades.v1',
        'tier': 'bronze',
        'description': 'Stock trade events',
    }
)
