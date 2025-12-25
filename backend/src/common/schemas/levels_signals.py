# src/common/schemas/levels_signals.py
"""
Comprehensive research schema for PLAN.md experimentation factory (Agent A-D).

This schema includes all production metrics to enable full feature experimentation:
- Agent A: Physics Engine (basic + advanced barrier/tape metrics)
- Agent B: Context Engine (level identification + market context)
- Agent C: Research/Labeling (outcome classification)
- Agent D: Pipeline integration

All advanced metrics are optional with defaults, allowing incremental implementation.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List

# --- Enumerations ---

class LevelKind(str, Enum):
    """Type of price level being tested."""
    PM_HIGH = "PM_HIGH"           # Pre-Market High
    PM_LOW = "PM_LOW"             # Pre-Market Low
    OR_HIGH = "OR_HIGH"           # Opening Range High (first 15m)
    OR_LOW = "OR_LOW"             # Opening Range Low
    SMA_200 = "SMA_200"           # 200 Simple Moving Average
    SMA_400 = "SMA_400"           # 400 Simple Moving Average
    STRIKE = "STRIKE"             # Standard Round Number (e.g., 500.00)
    VWAP = "VWAP"                 # Volume-Weighted Average Price
    ROUND = "ROUND"               # Round number level
    SESSION_HIGH = "SESSION_HIGH" # Session high
    SESSION_LOW = "SESSION_LOW"   # Session low
    CALL_WALL = "CALL_WALL"       # Call wall from gamma
    PUT_WALL = "PUT_WALL"         # Put wall from gamma
    GAMMA_FLIP = "GAMMA_FLIP"     # Gamma flip level (HVL)
    USER_HOTZONE = "USER_HOTZONE" # User-defined level

class OutcomeLabel(str, Enum):
    """Outcome classification for level test."""
    BOUNCE = "BOUNCE"     # Rejection (level holds)
    BREAK = "BREAK"       # Continuation (level fails)
    CHOP = "CHOP"         # No resolution
    UNDEFINED = "UNDEFINED"

class BarrierState(str, Enum):
    """Barrier physics state (liquidity behavior)."""
    VACUUM = "VACUUM"         # Liquidity pulled without fills (easy break)
    WALL = "WALL"             # Strong replenishment (reject likely)
    ABSORPTION = "ABSORPTION" # Liquidity consumed but replenished
    CONSUMED = "CONSUMED"     # Liquidity eaten faster than replenished
    WEAK = "WEAK"             # Defending size below baseline
    NEUTRAL = "NEUTRAL"       # Normal state

class Direction(str, Enum):
    """Price direction or level test direction."""
    UP = "UP"               # Moving up / resistance test
    DOWN = "DOWN"           # Moving down / support test
    SUPPORT = "SUPPORT"     # Level acting as support (price above)
    RESISTANCE = "RESISTANCE" # Level acting as resistance (price below)

class Signal(str, Enum):
    """Break/reject signal classification."""
    BREAK = "BREAK"           # Break imminent
    REJECT = "REJECT"         # Rejection likely
    CONTESTED = "CONTESTED"   # Unclear outcome
    NEUTRAL = "NEUTRAL"       # No clear signal

class Confidence(str, Enum):
    """Signal confidence level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class FuelEffect(str, Enum):
    """Gamma/hedging effect on price movement."""
    AMPLIFY = "AMPLIFY"   # Dealers chase (short gamma)
    DAMPEN = "DAMPEN"     # Dealers fade (long gamma)
    NEUTRAL = "NEUTRAL"   # No strong effect

class RunwayQuality(str, Enum):
    """Quality of runway (obstacles between levels)."""
    CLEAR = "CLEAR"       # No obstacles
    OBSTRUCTED = "OBSTRUCTED" # Obstacles present

# --- Main Schema ---

class LevelSignalV1(BaseModel):
    """
    Comprehensive Level Signal Schema.
    
    Represents one instance of price touching/approaching a level,
    with full physics, context, and outcome features for ML/research.
    """
    
    # ========== Identity & Timestamps ==========
    event_id: str = Field(..., description="Unique event identifier")
    ts_event_ns: int = Field(..., description="Event timestamp in nanoseconds UTC")
    confirm_ts_ns: Optional[int] = Field(default=None, description="Confirmation timestamp t1 (ns)")
    symbol: str = Field(default="SPY", description="Underlying symbol")
    
    # ========== Market Context ==========
    spot: Optional[float] = Field(default=None, description="Current spot price")
    bid: Optional[float] = Field(default=None, description="Current best bid")
    ask: Optional[float] = Field(default=None, description="Current best ask")
    
    # ========== Level Identity ==========
    level_price: float = Field(..., description="Price level being tested")
    level_kind: LevelKind = Field(..., description="Type of level")
    level_id: Optional[str] = Field(default=None, description="Level identifier (e.g., 'STRIKE_687')")
    direction: Optional[Direction] = Field(default=None, description="SUPPORT or RESISTANCE")
    distance: Optional[float] = Field(default=None, description="Distance from spot to level")
    direction_sign: Optional[int] = Field(default=None, description="Direction sign (UP=1, DOWN=-1)")
    distance_signed: Optional[float] = Field(default=None, description="Signed distance (level_price - spot)")
    atr: Optional[float] = Field(default=None, description="ATR at event time for normalization")
    
    # ========== Context Features (Agent B) ==========
    is_first_15m: bool = Field(default=False, description="True if 09:30-09:45 ET")
    dist_to_sma_200: Optional[float] = Field(default=None, description="Distance to 200 SMA")
    dist_to_sma_400: Optional[float] = Field(default=None, description="Distance to 400 SMA")
    dist_to_pm_high: Optional[float] = Field(default=None, description="Distance to pre-market high")
    dist_to_pm_low: Optional[float] = Field(default=None, description="Distance to pre-market low")
    sma_200: Optional[float] = Field(default=None, description="SMA-200 value at event time")
    sma_400: Optional[float] = Field(default=None, description="SMA-400 value at event time")
    sma_200_slope: Optional[float] = Field(default=None, description="SMA-200 slope ($/min)")
    sma_400_slope: Optional[float] = Field(default=None, description="SMA-400 slope ($/min)")
    sma_200_slope_5bar: Optional[float] = Field(default=None, description="SMA-200 slope over last 5x2min bars ($/min)")
    sma_400_slope_5bar: Optional[float] = Field(default=None, description="SMA-400 slope over last 5x2min bars ($/min)")
    sma_spread: Optional[float] = Field(default=None, description="SMA-200 minus SMA-400 ($)")
    mean_reversion_pressure_200: Optional[float] = Field(default=None, description="Distance to SMA-200 normalized by volatility")
    mean_reversion_pressure_400: Optional[float] = Field(default=None, description="Distance to SMA-400 normalized by volatility")
    mean_reversion_velocity_200: Optional[float] = Field(default=None, description="Change in distance to SMA-200 per minute")
    mean_reversion_velocity_400: Optional[float] = Field(default=None, description="Change in distance to SMA-400 per minute")
    confluence_count: Optional[int] = Field(default=None, description="Count of nearby key levels")
    confluence_weighted_score: Optional[float] = Field(default=None, description="Weighted confluence score")
    confluence_min_distance: Optional[float] = Field(default=None, description="Closest distance to a secondary key level")
    confluence_pressure: Optional[float] = Field(default=None, description="Normalized confluence strength")
    confluence_alignment: Optional[int] = Field(default=None, description="Alignment of stacked levels with approach direction (-1,0,1)")
    gamma_flow_velocity: Optional[float] = Field(default=None, description="Net dealer gamma flow per minute near level")
    gamma_flow_impulse: Optional[float] = Field(default=None, description="Short-term gamma flow deviation vs baseline")
    gamma_flow_accel_1m: Optional[float] = Field(default=None, description="Change in 1-minute gamma flow velocity (acceleration)")
    gamma_flow_accel_3m: Optional[float] = Field(default=None, description="Change in 3-minute gamma flow velocity (acceleration)")
    dealer_pressure: Optional[float] = Field(default=None, description="Normalized dealer mechanics pressure")
    dealer_pressure_accel: Optional[float] = Field(default=None, description="Normalized dealer pressure acceleration (1-minute window)")
    liquidity_pressure: Optional[float] = Field(default=None, description="Normalized liquidity pressure indicator")
    tape_pressure: Optional[float] = Field(default=None, description="Normalized tape momentum indicator")
    gamma_pressure: Optional[float] = Field(default=None, description="Normalized gamma pressure indicator")
    gamma_pressure_accel: Optional[float] = Field(default=None, description="Normalized gamma pressure acceleration indicator")
    reversion_pressure: Optional[float] = Field(default=None, description="Normalized mean reversion pressure indicator")
    net_break_pressure: Optional[float] = Field(default=None, description="Composite break/bounce pressure indicator")
    
    # ========== Basic Physics Features (Agent A - Core) ==========
    wall_ratio: float = Field(default=0.0, description="Size at Level / Avg Volume")
    replenishment_speed_ms: Optional[float] = Field(default=None, description="Time to reload after sweep (ms)")
    gamma_exposure: float = Field(default=0.0, description="Net GEX at this strike")
    tape_velocity: float = Field(default=0.0, description="Trades per second in last 5s")
    
    # ========== Scores & Signals ==========
    break_score_raw: Optional[float] = Field(default=None, ge=0, le=100, description="Raw composite break score (0-100)")
    break_score_smooth: Optional[float] = Field(default=None, ge=0, le=100, description="Smoothed break score (EWMA)")
    signal: Optional[Signal] = Field(default=None, description="Signal classification")
    confidence: Optional[Confidence] = Field(default=None, description="Signal confidence")
    
    # ========== Barrier Metrics (Advanced Physics) ==========
    barrier_state: Optional[BarrierState] = Field(default=None, description="Barrier physics state")
    barrier_delta_liq: Optional[float] = Field(default=None, description="Net liquidity change (added - canceled - filled)")
    barrier_replenishment_ratio: Optional[float] = Field(default=None, description="added / (canceled + filled + epsilon)")
    barrier_added: Optional[int] = Field(default=None, ge=0, description="Size added to defending quote")
    barrier_canceled: Optional[int] = Field(default=None, ge=0, description="Size canceled from defending quote")
    barrier_filled: Optional[int] = Field(default=None, ge=0, description="Size filled at defending quote")
    
    # ========== Tape Metrics (Advanced Momentum) ==========
    tape_imbalance: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Buy/sell imbalance near level (-1 to +1)")
    tape_buy_vol: Optional[int] = Field(default=None, ge=0, description="Buy volume near level")
    tape_sell_vol: Optional[int] = Field(default=None, ge=0, description="Sell volume near level")
    tape_sweep_detected: Optional[bool] = Field(default=None, description="Whether sweep was detected")
    tape_sweep_direction: Optional[str] = Field(default=None, description="Sweep direction if detected (UP/DOWN)")
    tape_sweep_notional: Optional[float] = Field(default=None, ge=0, description="Sweep notional value if detected")
    
    # ========== Fuel Metrics (Gamma/Hedging) ==========
    fuel_effect: Optional[FuelEffect] = Field(default=None, description="Gamma effect on price movement")
    fuel_net_dealer_gamma: Optional[float] = Field(default=None, description="Net dealer gamma near level")
    fuel_call_wall: Optional[float] = Field(default=None, description="Call wall strike if identified")
    fuel_put_wall: Optional[float] = Field(default=None, description="Put wall strike if identified")
    fuel_hvl: Optional[float] = Field(default=None, description="Gamma flip level (HVL) if computed")
    
    # ========== Runway Metrics (Room to Run) ==========
    runway_direction: Optional[str] = Field(default=None, description="Expected move direction after break/reject")
    runway_next_level_id: Optional[str] = Field(default=None, description="ID of next obstacle level")
    runway_next_level_price: Optional[float] = Field(default=None, description="Price of next obstacle level")
    runway_distance: Optional[float] = Field(default=None, ge=0, description="Distance to next obstacle")
    runway_quality: Optional[RunwayQuality] = Field(default=None, description="CLEAR or OBSTRUCTED")
    
    # ========== Outcome (Agent C) ==========
    outcome: OutcomeLabel = Field(default=OutcomeLabel.UNDEFINED, description="Outcome classification")
    future_price_5min: Optional[float] = Field(default=None, description="Price 5 minutes after touch")
    anchor_spot: Optional[float] = Field(default=None, description="Spot price at confirmation time t1")
    tradeable_1: Optional[int] = Field(default=None, description="Reached $1.00 in either direction within 5 minutes")
    tradeable_2: Optional[int] = Field(default=None, description="Reached $2.00 in either direction within 5 minutes")
    strength_signed: Optional[float] = Field(default=None, description="Signed strength: break-direction excursion minus opposite")
    strength_abs: Optional[float] = Field(default=None, description="Absolute max excursion in either direction")
    time_to_threshold_1: Optional[float] = Field(default=None, description="Seconds to first $1 move in either direction")
    time_to_threshold_2: Optional[float] = Field(default=None, description="Seconds to first $2 move in either direction")
    
    # ========== Optional Note ==========
    note: Optional[str] = Field(default=None, description="Human-readable signal summary")
