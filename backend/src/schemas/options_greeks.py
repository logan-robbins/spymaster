"""
options.greeks_snapshots.v1 schema - Bronze tier Greeks snapshots from REST API.

Fields per PLAN.md §2.4:
- ts_event_ns: int64 (snapshot time, UTC nanoseconds)
- source: utf8 (EventSource enum value)
- underlying: utf8 (e.g., 'SPY')
- option_symbol: utf8 (vendor symbol)
- delta: float64
- gamma: float64
- theta: float64
- vega: float64
- implied_volatility: float64 (optional)
- open_interest: int64 (optional)
- snapshot_id: utf8 (optional, hash of content/time for dedup)
"""

from typing import Optional, ClassVar

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
class GreeksSnapshotV1(BaseEventModel):
    """
    Bronze tier Greeks snapshot from REST API or cache.

    Represents point-in-time Greeks values for an option contract.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='options.greeks_snapshots',
        version=1,
        tier='bronze'
    )

    # Required fields
    ts_event_ns: int = Field(
        ...,
        description="Snapshot timestamp in Unix nanoseconds UTC"
    )
    source: EventSourceEnum = Field(
        ...,
        description="Event source (massive_rest, polygon_rest, etc.)"
    )
    underlying: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Underlying symbol"
    )
    option_symbol: str = Field(
        ...,
        min_length=1,
        description="Vendor option symbol"
    )
    delta: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Option delta (-1 to +1)"
    )
    gamma: float = Field(
        ...,
        ge=0,
        description="Option gamma (non-negative)"
    )
    theta: float = Field(
        ...,
        description="Option theta (typically negative)"
    )
    vega: float = Field(
        ...,
        ge=0,
        description="Option vega (non-negative)"
    )

    # Optional fields
    implied_volatility: Optional[float] = Field(
        default=None,
        ge=0,
        description="Implied volatility (decimal, e.g., 0.25 = 25%)"
    )
    open_interest: Optional[int] = Field(
        default=None,
        ge=0,
        description="Open interest (number of contracts)"
    )
    snapshot_id: Optional[str] = Field(
        default=None,
        description="Hash of content/time for deduplication"
    )

    @field_validator('ts_event_ns')
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is reasonable."""
        min_ts = 946_684_800_000_000_000  # 2000-01-01
        max_ts = 4_102_444_800_000_000_000  # 2100-01-01
        if not min_ts <= v <= max_ts:
            raise ValueError(f"Timestamp {v} outside valid range")
        return v


# Arrow schema definition
GreeksSnapshotV1._arrow_schema = build_arrow_schema(
    fields=[
        ('ts_event_ns', pa.int64(), False),
        ('source', pa.utf8(), False),
        ('underlying', pa.utf8(), False),
        ('option_symbol', pa.utf8(), False),
        ('delta', pa.float64(), False),
        ('gamma', pa.float64(), False),
        ('theta', pa.float64(), False),
        ('vega', pa.float64(), False),
        ('implied_volatility', pa.float64(), True),
        ('open_interest', pa.int64(), True),
        ('snapshot_id', pa.utf8(), True),
    ],
    metadata={
        'schema_name': 'options.greeks_snapshots.v1',
        'tier': 'bronze',
        'description': 'Greeks snapshots from REST API',
    }
)
