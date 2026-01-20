"""
options.trades schema - Bronze tier option trades.

- ts_event_ns: int64 (UTC nanoseconds)
- ts_recv_ns: int64 (UTC nanoseconds)
- source: utf8 (EventSource enum value)
- underlying: utf8 (e.g., 'ES')
- option_symbol: utf8 (vendor symbol, e.g., ES option symbol)
- exp_date: date32 (expiration date)
- strike: float64
- right: utf8 ('C' or 'P')
- price: float64
- size: int32
- opt_bid: float64 (optional, option BBO if available)
- opt_ask: float64 (optional)
- aggressor: int8 (BUY=+1, SELL=-1, MID=0)
- conditions: list<int16> (optional)
- seq: int64 (optional)
"""

from typing import Optional, List, ClassVar
from datetime import date

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
class OptionTrade(BaseEventModel):
    """
    Bronze tier option trade event.

    Represents a single option trade with inferred aggressor
    and optional BBO context.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='options.trades',
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
    underlying: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Underlying symbol (e.g., 'ES')"
    )
    option_symbol: str = Field(
        ...,
        min_length=1,
        description="Vendor option symbol (e.g., ES option symbol)"
    )
    exp_date: date = Field(
        ...,
        description="Option expiration date"
    )
    strike: float = Field(
        ...,
        gt=0,
        description="Strike price"
    )
    right: str = Field(
        ...,
        pattern=r'^[CP]$',
        description="Option right: 'C' (call) or 'P' (put)"
    )
    price: float = Field(
        ...,
        ge=0,
        description="Trade price"
    )
    size: int = Field(
        ...,
        ge=1,
        description="Number of contracts"
    )

    # Optional fields
    opt_bid: Optional[float] = Field(
        default=None,
        ge=0,
        description="Option best bid at time of trade"
    )
    opt_ask: Optional[float] = Field(
        default=None,
        ge=0,
        description="Option best ask at time of trade"
    )
    aggressor: AggressorEnum = Field(
        default=AggressorEnum.MID,
        description="Inferred aggressor (BUY=+1 lifted ask, SELL=-1 hit bid, MID=0 unknown)"
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

    @field_validator('exp_date')
    @classmethod
    def validate_exp_date(cls, v: date) -> date:
        """Validate expiration is not in distant past."""
        min_date = date(2000, 1, 1)
        if v < min_date:
            raise ValueError(f"Expiration date {v} before year 2000")
        return v


# Arrow schema definition
OptionTrade._arrow_schema = build_arrow_schema(
    fields=[
        ('ts_event_ns', pa.int64(), False),
        ('ts_recv_ns', pa.int64(), False),
        ('source', pa.utf8(), False),
        ('underlying', pa.utf8(), False),
        ('option_symbol', pa.utf8(), False),
        ('exp_date', pa.date32(), False),
        ('strike', pa.float64(), False),
        ('right', pa.utf8(), False),
        ('price', pa.float64(), False),
        ('size', pa.int32(), False),
        ('opt_bid', pa.float64(), True),
        ('opt_ask', pa.float64(), True),
        ('aggressor', pa.int8(), False),
        ('conditions', pa.list_(pa.int16()), True),
        ('seq', pa.int64(), True),
    ],
    metadata={
        'schema_name': 'options.trades',
        'tier': 'bronze',
        'description': 'Option trade events with aggressor inference',
    }
)
