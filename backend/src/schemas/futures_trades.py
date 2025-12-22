"""
futures.trades.v1 schema - Bronze tier futures trades (ES).

Fields per PLAN.md §2.4:
- ts_event_ns: int64 (UTC nanoseconds)
- ts_recv_ns: int64 (UTC nanoseconds)
- source: utf8 (EventSource enum value)
- symbol: utf8 (e.g., 'ES' or full contract 'ESH6')
- price: float64
- size: int32
- aggressor: int8 (BUY=+1, SELL=-1, MID=0)
- exchange: utf8 (optional, venue/exchange)
- conditions: list<int16> (optional)
- seq: int64 (optional)

Used for ES L2 barrier physics when enabled.
"""

from typing import Optional, List, ClassVar

import pyarrow as pa
from pydantic import Field, field_validator

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    EventSourceEnum,
    AggressorEnum,
    build_arrow_schema,
)


@SchemaRegistry.register
class FuturesTradeV1(BaseEventModel):
    """
    Bronze tier futures trade event (e.g., ES).

    Represents a single futures trade with aggressor side.
    Used for ES barrier/tape physics.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='futures.trades',
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
        description="Event source"
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Futures symbol (e.g., 'ES', 'ESH6', 'ES.c.0')"
    )
    price: float = Field(
        ...,
        gt=0,
        description="Trade price"
    )
    size: int = Field(
        ...,
        ge=1,
        description="Number of contracts"
    )

    # Optional fields with defaults
    aggressor: AggressorEnum = Field(
        default=AggressorEnum.MID,
        description="Aggressor side: BUY=+1 (lifted ask), SELL=-1 (hit bid), MID=0"
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Exchange/venue code"
    )
    conditions: Optional[List[int]] = Field(
        default=None,
        description="Vendor-specific trade condition codes"
    )
    seq: Optional[int] = Field(
        default=None,
        ge=0,
        description="Monotonic sequence number"
    )

    @field_validator('ts_event_ns', 'ts_recv_ns')
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is reasonable."""
        min_ts = 946_684_800_000_000_000  # 2000-01-01
        max_ts = 4_102_444_800_000_000_000  # 2100-01-01
        if not min_ts <= v <= max_ts:
            raise ValueError(f"Timestamp {v} outside valid range")
        return v


# Arrow schema definition
FuturesTradeV1._arrow_schema = build_arrow_schema(
    fields=[
        ('ts_event_ns', pa.int64(), False),
        ('ts_recv_ns', pa.int64(), False),
        ('source', pa.utf8(), False),
        ('symbol', pa.utf8(), False),
        ('price', pa.float64(), False),
        ('size', pa.int32(), False),
        ('aggressor', pa.int8(), False),
        ('exchange', pa.utf8(), True),
        ('conditions', pa.list_(pa.int16()), True),
        ('seq', pa.int64(), True),
    ],
    metadata={
        'schema_name': 'futures.trades.v1',
        'tier': 'bronze',
        'description': 'Futures trade events (ES) with aggressor',
    }
)
