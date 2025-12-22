"""
stocks.quotes.v1 schema - Bronze tier stock quotes (NBBO).

Fields per PLAN.md §2.4:
- ts_event_ns: int64 (UTC nanoseconds)
- ts_recv_ns: int64 (UTC nanoseconds)
- source: utf8 (EventSource enum value)
- symbol: utf8 (e.g., 'SPY')
- bid_px: float64
- ask_px: float64
- bid_sz: int32 (shares, not round lots per SEC MDI 2025-11-03)
- ask_sz: int32 (shares)
- bid_exch: int16 (optional)
- ask_exch: int16 (optional)
- seq: int64 (optional)
"""

from typing import Optional, ClassVar

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    EventSourceEnum,
    build_arrow_schema,
)


@SchemaRegistry.register
class StockQuoteV1(BaseEventModel):
    """
    Bronze tier stock quote (NBBO) event.

    Represents best bid/ask from consolidated feed.

    Note: As of 2025-11-03, Massive reports bid_sz/ask_sz in SHARES
    (not round lots) per SEC MDI rules.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='stocks.quotes',
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
        max_length=10,
        description="Stock symbol"
    )
    bid_px: float = Field(
        ...,
        ge=0,
        description="Best bid price"
    )
    ask_px: float = Field(
        ...,
        ge=0,
        description="Best ask price"
    )
    bid_sz: int = Field(
        ...,
        ge=0,
        description="Bid size in shares (not round lots)"
    )
    ask_sz: int = Field(
        ...,
        ge=0,
        description="Ask size in shares (not round lots)"
    )

    # Optional fields
    bid_exch: Optional[int] = Field(
        default=None,
        description="Bid exchange code"
    )
    ask_exch: Optional[int] = Field(
        default=None,
        description="Ask exchange code"
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

    @model_validator(mode='after')
    def validate_spread(self):
        """Validate ask >= bid (crossed quotes are invalid)."""
        if self.ask_px > 0 and self.bid_px > 0:
            if self.ask_px < self.bid_px:
                raise ValueError(
                    f"Crossed quote: ask ({self.ask_px}) < bid ({self.bid_px})"
                )
        return self


# Arrow schema definition
StockQuoteV1._arrow_schema = build_arrow_schema(
    fields=[
        ('ts_event_ns', pa.int64(), False),
        ('ts_recv_ns', pa.int64(), False),
        ('source', pa.utf8(), False),
        ('symbol', pa.utf8(), False),
        ('bid_px', pa.float64(), False),
        ('ask_px', pa.float64(), False),
        ('bid_sz', pa.int32(), False),
        ('ask_sz', pa.int32(), False),
        ('bid_exch', pa.int16(), True),
        ('ask_exch', pa.int16(), True),
        ('seq', pa.int64(), True),
    ],
    metadata={
        'schema_name': 'stocks.quotes.v1',
        'tier': 'bronze',
        'description': 'Stock NBBO quote events (sizes in shares)',
    }
)
