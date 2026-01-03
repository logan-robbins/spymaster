"""
futures.mbp10.v1 schema - Bronze tier ES MBP-10 (Market-by-Price top 10 levels).

Fields per PLAN.md §2.4:
- ts_event_ns: int64 (UTC nanoseconds)
- ts_recv_ns: int64 (UTC nanoseconds)
- source: utf8 (EventSource enum value)
- symbol: utf8 (e.g., 'ES')
- 10 bid levels: bid_px_1..10 (float64), bid_sz_1..10 (int32)
- 10 ask levels: ask_px_1..10 (float64), ask_sz_1..10 (int32)
- is_snapshot: bool (optional, differentiates snapshot vs incremental)
- seq: int64 (optional)

Used for ES L2 barrier physics when enabled.
"""

from typing import Optional, List, ClassVar

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    EventSourceEnum,
    build_arrow_schema,
)


class BidAskLevelModel(BaseEventModel):
    """Single bid/ask level in MBP."""
    bid_px: float = Field(..., ge=0, description="Bid price")
    bid_sz: int = Field(..., ge=0, description="Bid size")
    ask_px: float = Field(..., ge=0, description="Ask price")
    ask_sz: int = Field(..., ge=0, description="Ask size")


@SchemaRegistry.register
class MBP10V1(BaseEventModel):
    """
    Bronze tier MBP-10 (Market-by-Price) snapshot.

    Represents top 10 price levels per side for ES futures.
    Used for barrier physics liquidity analysis.

    Note: This model stores levels as a nested list for convenience,
    but the Arrow schema flattens to bid_px_1..10, ask_px_1..10, etc.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='futures.mbp10',
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
        description="Futures symbol"
    )

    # Flattened level fields (10 levels)
    bid_px_1: float = Field(default=0.0, ge=0)
    bid_px_2: float = Field(default=0.0, ge=0)
    bid_px_3: float = Field(default=0.0, ge=0)
    bid_px_4: float = Field(default=0.0, ge=0)
    bid_px_5: float = Field(default=0.0, ge=0)
    bid_px_6: float = Field(default=0.0, ge=0)
    bid_px_7: float = Field(default=0.0, ge=0)
    bid_px_8: float = Field(default=0.0, ge=0)
    bid_px_9: float = Field(default=0.0, ge=0)
    bid_px_10: float = Field(default=0.0, ge=0)

    bid_sz_1: int = Field(default=0, ge=0)
    bid_sz_2: int = Field(default=0, ge=0)
    bid_sz_3: int = Field(default=0, ge=0)
    bid_sz_4: int = Field(default=0, ge=0)
    bid_sz_5: int = Field(default=0, ge=0)
    bid_sz_6: int = Field(default=0, ge=0)
    bid_sz_7: int = Field(default=0, ge=0)
    bid_sz_8: int = Field(default=0, ge=0)
    bid_sz_9: int = Field(default=0, ge=0)
    bid_sz_10: int = Field(default=0, ge=0)

    ask_px_1: float = Field(default=0.0, ge=0)
    ask_px_2: float = Field(default=0.0, ge=0)
    ask_px_3: float = Field(default=0.0, ge=0)
    ask_px_4: float = Field(default=0.0, ge=0)
    ask_px_5: float = Field(default=0.0, ge=0)
    ask_px_6: float = Field(default=0.0, ge=0)
    ask_px_7: float = Field(default=0.0, ge=0)
    ask_px_8: float = Field(default=0.0, ge=0)
    ask_px_9: float = Field(default=0.0, ge=0)
    ask_px_10: float = Field(default=0.0, ge=0)

    ask_sz_1: int = Field(default=0, ge=0)
    ask_sz_2: int = Field(default=0, ge=0)
    ask_sz_3: int = Field(default=0, ge=0)
    ask_sz_4: int = Field(default=0, ge=0)
    ask_sz_5: int = Field(default=0, ge=0)
    ask_sz_6: int = Field(default=0, ge=0)
    ask_sz_7: int = Field(default=0, ge=0)
    ask_sz_8: int = Field(default=0, ge=0)
    ask_sz_9: int = Field(default=0, ge=0)
    ask_sz_10: int = Field(default=0, ge=0)

    # Optional fields
    is_snapshot: bool = Field(
        default=False,
        description="True if this is a full snapshot vs incremental update"
    )
    seq: Optional[int] = Field(
        default=None,
        ge=0,
        description="Monotonic sequence number"
    )
    
    # OFI fields for true event-based OFI (Cont et al. 2014)
    action: Optional[str] = Field(
        default=None,
        description="Event action: A=Add, C=Cancel, M=Modify, T=Trade"
    )
    side: Optional[str] = Field(
        default=None,
        description="Side affected: A=Ask, B=Bid, N=None"
    )
    action_price: Optional[float] = Field(
        default=None,
        ge=0,
        description="Price level affected by this action"
    )
    action_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Size of the action"
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

    @classmethod
    def from_levels(
        cls,
        ts_event_ns: int,
        ts_recv_ns: int,
        source: EventSourceEnum,
        symbol: str,
        levels: List[BidAskLevelModel],
        is_snapshot: bool = False,
        seq: Optional[int] = None
    ) -> "MBP10V1":
        """
        Create from a list of BidAskLevel objects.

        Args:
            levels: List of up to 10 BidAskLevel objects
        """
        data = {
            'ts_event_ns': ts_event_ns,
            'ts_recv_ns': ts_recv_ns,
            'source': source,
            'symbol': symbol,
            'is_snapshot': is_snapshot,
            'seq': seq,
        }

        # Flatten levels into bid_px_N, bid_sz_N, ask_px_N, ask_sz_N
        for i, level in enumerate(levels[:10], start=1):
            data[f'bid_px_{i}'] = level.bid_px
            data[f'bid_sz_{i}'] = level.bid_sz
            data[f'ask_px_{i}'] = level.ask_px
            data[f'ask_sz_{i}'] = level.ask_sz

        return cls(**data)

    def get_bid_levels(self) -> List[tuple]:
        """Return list of (price, size) tuples for bids."""
        levels = []
        for i in range(1, 11):
            px = getattr(self, f'bid_px_{i}')
            sz = getattr(self, f'bid_sz_{i}')
            if px > 0 or sz > 0:
                levels.append((px, sz))
        return levels

    def get_ask_levels(self) -> List[tuple]:
        """Return list of (price, size) tuples for asks."""
        levels = []
        for i in range(1, 11):
            px = getattr(self, f'ask_px_{i}')
            sz = getattr(self, f'ask_sz_{i}')
            if px > 0 or sz > 0:
                levels.append((px, sz))
        return levels


# Arrow schema definition (flattened levels)
_mbp10_fields = [
    ('ts_event_ns', pa.int64(), False),
    ('ts_recv_ns', pa.int64(), False),
    ('source', pa.utf8(), False),
    ('symbol', pa.utf8(), False),
]

# Add 10 bid levels
for i in range(1, 11):
    _mbp10_fields.append((f'bid_px_{i}', pa.float64(), False))
    _mbp10_fields.append((f'bid_sz_{i}', pa.int32(), False))

# Add 10 ask levels
for i in range(1, 11):
    _mbp10_fields.append((f'ask_px_{i}', pa.float64(), False))
    _mbp10_fields.append((f'ask_sz_{i}', pa.int32(), False))

# Add optional fields
_mbp10_fields.extend([
    ('is_snapshot', pa.bool_(), False),
    ('seq', pa.int64(), True),
    # OFI fields for true event-based OFI (Cont et al. 2014)
    ('action', pa.utf8(), True),       # A=Add, C=Cancel, M=Modify, T=Trade
    ('side', pa.utf8(), True),         # A=Ask, B=Bid, N=None
    ('action_price', pa.float64(), True),
    ('action_size', pa.int32(), True),
])

MBP10V1._arrow_schema = build_arrow_schema(
    fields=_mbp10_fields,
    metadata={
        'schema_name': 'futures.mbp10.v2',  # Version bump for OFI fields
        'tier': 'bronze',
        'description': 'ES MBP-10 with action/side for true OFI computation',
    }
)
