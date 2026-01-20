"""
options.trades_enriched.v1 schema - Silver tier enriched option trades.

Fields per PLAN.md §2.4:
Extends options.trades with:
- greeks_snapshot_id: utf8 (reference to greeks snapshot used)
- delta: float64 (as-of joined)
- gamma: float64 (as-of joined)
- delta_notional: float64 (delta * size * contract multiplier)
- gamma_notional: float64 (gamma * size * contract multiplier)
- join_tolerance_ms: int64 (tolerance used for as-of join)

Silver tier: cleaned, normalized, deduped, with Greeks joined.
"""

from typing import Optional, List, ClassVar
from datetime import date

import pyarrow as pa
from pydantic import Field, field_validator, computed_field

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    EventSourceEnum,
    AggressorEnum,
    build_arrow_schema,
)


@SchemaRegistry.register
class OptionTradeEnriched(BaseEventModel):
    """
    Silver tier enriched option trade.

    Extends bronze option trades with Greeks from as-of join.
    Includes computed notional fields for delta and gamma.
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='options.trades_enriched',
        version=1,
        tier='silver'
    )

    # --- Bronze fields (from options.trades) ---

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
        description="Underlying symbol"
    )
    option_symbol: str = Field(
        ...,
        min_length=1,
        description="Vendor option symbol"
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
        description="Option right: 'C' or 'P'"
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
        description="Inferred aggressor"
    )
    conditions: Optional[List[int]] = Field(
        default=None,
        description="Vendor-specific trade conditions"
    )
    seq: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sequence number"
    )

    # --- Silver enrichment fields ---

    greeks_snapshot_id: Optional[str] = Field(
        default=None,
        description="ID of greeks snapshot used for enrichment"
    )
    delta: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Option delta from as-of joined snapshot"
    )
    gamma: Optional[float] = Field(
        default=None,
        ge=0,
        description="Option gamma from as-of joined snapshot"
    )
    delta_notional: Optional[float] = Field(
        default=None,
        description="delta * size * contract multiplier"
    )
    gamma_notional: Optional[float] = Field(
        default=None,
        description="gamma * size * contract multiplier"
    )
    join_tolerance_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Tolerance in ms used for as-of join"
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

    def compute_notionals(self) -> "OptionTradeEnriched":
        """
        Compute delta_notional and gamma_notional from delta, gamma, size.

        Returns self for chaining.
        """
        from src.common.config import CONFIG
        contract_multiplier = CONFIG.OPTION_CONTRACT_MULTIPLIER
        if self.delta is not None:
            self.delta_notional = self.delta * self.size * contract_multiplier
        if self.gamma is not None:
            self.gamma_notional = self.gamma * self.size * contract_multiplier
        return self


# Arrow schema definition
OptionTradeEnriched._arrow_schema = build_arrow_schema(
    fields=[
        # Bronze fields
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

        # Silver enrichment fields
        ('greeks_snapshot_id', pa.utf8(), True),
        ('delta', pa.float64(), True),
        ('gamma', pa.float64(), True),
        ('delta_notional', pa.float64(), True),
        ('gamma_notional', pa.float64(), True),
        ('join_tolerance_ms', pa.int64(), True),
    ],
    metadata={
        'schema_name': 'options.trades_enriched.v1',
        'tier': 'silver',
        'description': 'Enriched option trades with Greeks from as-of join',
    }
)
