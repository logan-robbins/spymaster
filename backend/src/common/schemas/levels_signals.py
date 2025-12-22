"""
levels.signals.v1 schema - Gold tier level signals (derived analytics).

Fields per PLAN.md §2.4 and §6.4:
- ts_event_ns: int64 (snap tick time, UTC nanoseconds)
- underlying: utf8 ('SPY')
- spot: float64 (current SPY price)
- bid: float64 (current SPY bid)
- ask: float64 (current SPY ask)
- level_id: utf8 (e.g., 'STRIKE_545', 'ROUND_680')
- level_kind: utf8 (VWAP, STRIKE, ROUND, etc.)
- level_price: float64
- direction: utf8 (SUPPORT or RESISTANCE)
- distance: float64 (spot - level_price)
- break_score_raw: float64 (0-100)
- break_score_smooth: float64 (0-100, smoothed)
- signal: utf8 (BREAK, REJECT, CONTESTED, NEUTRAL)
- confidence: utf8 (HIGH, MEDIUM, LOW)
- barrier metrics (flattened)
- tape metrics (flattened)
- fuel metrics (flattened)
- runway metrics (flattened)
- note: utf8 (optional human-readable summary)
"""

from typing import Optional, ClassVar
from enum import Enum

import pyarrow as pa
from pydantic import Field, field_validator

from .base import (
    BaseEventModel,
    SchemaVersion,
    SchemaRegistry,
    build_arrow_schema,
)


class BarrierStateEnum(str, Enum):
    """Barrier physics state."""
    VACUUM = "VACUUM"
    WALL = "WALL"
    ABSORPTION = "ABSORPTION"
    CONSUMED = "CONSUMED"
    WEAK = "WEAK"
    NEUTRAL = "NEUTRAL"


class DirectionEnum(str, Enum):
    """Price direction or level test direction."""
    UP = "UP"
    DOWN = "DOWN"
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class FuelEffectEnum(str, Enum):
    """Gamma/hedging effect on price movement."""
    AMPLIFY = "AMPLIFY"
    DAMPEN = "DAMPEN"
    NEUTRAL = "NEUTRAL"


class SignalEnum(str, Enum):
    """Break/reject signal classification."""
    BREAK = "BREAK"
    REJECT = "REJECT"
    CONTESTED = "CONTESTED"
    NEUTRAL = "NEUTRAL"


class ConfidenceEnum(str, Enum):
    """Signal confidence level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RunwayQualityEnum(str, Enum):
    """Quality of runway (obstacles between levels)."""
    CLEAR = "CLEAR"
    OBSTRUCTED = "OBSTRUCTED"


class LevelKindEnum(str, Enum):
    """Type of price level."""
    VWAP = "VWAP"
    STRIKE = "STRIKE"
    ROUND = "ROUND"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    CALL_WALL = "CALL_WALL"
    PUT_WALL = "PUT_WALL"
    GAMMA_FLIP = "GAMMA_FLIP"
    USER_HOTZONE = "USER_HOTZONE"


@SchemaRegistry.register
class LevelSignalV1(BaseEventModel):
    """
    Gold tier level signal - complete break/reject analytics.

    Combines barrier, tape, fuel, and runway analysis for a single level.
    Published on snap tick cadence (100-250ms).
    """

    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='levels.signals',
        version=1,
        tier='gold'
    )

    # Market context
    ts_event_ns: int = Field(
        ...,
        description="Snap tick timestamp in Unix nanoseconds UTC"
    )
    underlying: str = Field(
        ...,
        description="Underlying symbol (SPY)"
    )
    spot: float = Field(
        ...,
        gt=0,
        description="Current SPY spot price"
    )
    bid: float = Field(
        ...,
        ge=0,
        description="Current SPY best bid"
    )
    ask: float = Field(
        ...,
        ge=0,
        description="Current SPY best ask"
    )

    # Level identity
    level_id: str = Field(
        ...,
        description="Level identifier (e.g., 'STRIKE_545')"
    )
    level_kind: LevelKindEnum = Field(
        ...,
        description="Level type"
    )
    level_price: float = Field(
        ...,
        gt=0,
        description="Level price"
    )
    direction: DirectionEnum = Field(
        ...,
        description="SUPPORT (price above level) or RESISTANCE (price below)"
    )
    distance: float = Field(
        ...,
        description="Distance from spot to level (spot - level_price)"
    )

    # Scores and signals
    break_score_raw: float = Field(
        ...,
        ge=0,
        le=100,
        description="Raw composite break score (0-100)"
    )
    break_score_smooth: float = Field(
        ...,
        ge=0,
        le=100,
        description="Smoothed break score (EWMA)"
    )
    signal: SignalEnum = Field(
        ...,
        description="Signal classification"
    )
    confidence: ConfidenceEnum = Field(
        ...,
        description="Signal confidence"
    )

    # Barrier metrics (flattened)
    barrier_state: BarrierStateEnum = Field(
        ...,
        description="Barrier physics state"
    )
    barrier_delta_liq: float = Field(
        default=0.0,
        description="Net liquidity change (added - canceled - filled)"
    )
    barrier_replenishment_ratio: float = Field(
        default=0.0,
        description="added / (canceled + filled + epsilon)"
    )
    barrier_added: int = Field(
        default=0,
        ge=0,
        description="Size added to defending quote"
    )
    barrier_canceled: int = Field(
        default=0,
        ge=0,
        description="Size canceled from defending quote"
    )
    barrier_filled: int = Field(
        default=0,
        ge=0,
        description="Size filled at defending quote"
    )

    # Tape metrics (flattened)
    tape_imbalance: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Buy/sell imbalance near level (-1 to +1)"
    )
    tape_buy_vol: int = Field(
        default=0,
        ge=0,
        description="Buy volume near level"
    )
    tape_sell_vol: int = Field(
        default=0,
        ge=0,
        description="Sell volume near level"
    )
    tape_velocity: float = Field(
        default=0.0,
        description="Price velocity ($/sec)"
    )
    tape_sweep_detected: bool = Field(
        default=False,
        description="Whether sweep was detected"
    )
    tape_sweep_direction: Optional[str] = Field(
        default=None,
        description="Sweep direction if detected (UP/DOWN)"
    )
    tape_sweep_notional: float = Field(
        default=0.0,
        ge=0,
        description="Sweep notional value if detected"
    )

    # Fuel metrics (flattened)
    fuel_effect: FuelEffectEnum = Field(
        default=FuelEffectEnum.NEUTRAL,
        description="Gamma effect on price movement"
    )
    fuel_net_dealer_gamma: float = Field(
        default=0.0,
        description="Net dealer gamma near level"
    )
    fuel_call_wall: Optional[float] = Field(
        default=None,
        description="Call wall strike if identified"
    )
    fuel_put_wall: Optional[float] = Field(
        default=None,
        description="Put wall strike if identified"
    )
    fuel_hvl: Optional[float] = Field(
        default=None,
        description="Gamma flip level (HVL) if computed"
    )

    # Runway metrics (flattened)
    runway_direction: Optional[str] = Field(
        default=None,
        description="Expected move direction after break/reject"
    )
    runway_next_level_id: Optional[str] = Field(
        default=None,
        description="ID of next obstacle level"
    )
    runway_next_level_price: Optional[float] = Field(
        default=None,
        description="Price of next obstacle level"
    )
    runway_distance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Distance to next obstacle"
    )
    runway_quality: Optional[str] = Field(
        default=None,
        description="CLEAR or OBSTRUCTED"
    )

    # Optional note
    note: Optional[str] = Field(
        default=None,
        description="Human-readable signal summary"
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
LevelSignalV1._arrow_schema = build_arrow_schema(
    fields=[
        # Market context
        ('ts_event_ns', pa.int64(), False),
        ('underlying', pa.utf8(), False),
        ('spot', pa.float64(), False),
        ('bid', pa.float64(), False),
        ('ask', pa.float64(), False),

        # Level identity
        ('level_id', pa.utf8(), False),
        ('level_kind', pa.utf8(), False),
        ('level_price', pa.float64(), False),
        ('direction', pa.utf8(), False),
        ('distance', pa.float64(), False),

        # Scores
        ('break_score_raw', pa.float64(), False),
        ('break_score_smooth', pa.float64(), False),
        ('signal', pa.utf8(), False),
        ('confidence', pa.utf8(), False),

        # Barrier metrics
        ('barrier_state', pa.utf8(), False),
        ('barrier_delta_liq', pa.float64(), False),
        ('barrier_replenishment_ratio', pa.float64(), False),
        ('barrier_added', pa.int64(), False),
        ('barrier_canceled', pa.int64(), False),
        ('barrier_filled', pa.int64(), False),

        # Tape metrics
        ('tape_imbalance', pa.float64(), False),
        ('tape_buy_vol', pa.int64(), False),
        ('tape_sell_vol', pa.int64(), False),
        ('tape_velocity', pa.float64(), False),
        ('tape_sweep_detected', pa.bool_(), False),
        ('tape_sweep_direction', pa.utf8(), True),
        ('tape_sweep_notional', pa.float64(), False),

        # Fuel metrics
        ('fuel_effect', pa.utf8(), False),
        ('fuel_net_dealer_gamma', pa.float64(), False),
        ('fuel_call_wall', pa.float64(), True),
        ('fuel_put_wall', pa.float64(), True),
        ('fuel_hvl', pa.float64(), True),

        # Runway metrics
        ('runway_direction', pa.utf8(), True),
        ('runway_next_level_id', pa.utf8(), True),
        ('runway_next_level_price', pa.float64(), True),
        ('runway_distance', pa.float64(), True),
        ('runway_quality', pa.utf8(), True),

        # Note
        ('note', pa.utf8(), True),
    ],
    metadata={
        'schema_name': 'levels.signals.v1',
        'tier': 'gold',
        'description': 'Level break/reject signals with full metrics',
    }
)
