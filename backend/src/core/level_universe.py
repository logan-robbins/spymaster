"""
Level Universe: generate critical levels for monitoring.

Agent F deliverable per §12 of PLAN.md.

This module provides a consistent set of levels that the system monitors:
- VWAP (session VWAP)
- Round numbers (every $1 near spot)
- Option strikes (from MarketState flow data)
- Flow-derived walls (call/put walls from gamma flow)
- User-defined hotzones (manual override levels)

All levels are returned as Level objects with stable IDs.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum

from .market_state import MarketState
from src.common.config import CONFIG


class LevelKind(Enum):
    """Type of level (for classification and display)."""
    PM_HIGH = "PM_HIGH"
    PM_LOW = "PM_LOW"
    OR_HIGH = "OR_HIGH"
    OR_LOW = "OR_LOW"
    SMA_200 = "SMA_200"
    SMA_400 = "SMA_400"
    VWAP = "VWAP"
    STRIKE = "STRIKE"
    ROUND = "ROUND"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    CALL_WALL = "CALL_WALL"
    PUT_WALL = "PUT_WALL"
    USER_HOTZONE = "USER_HOTZONE"
    GAMMA_FLIP = "GAMMA_FLIP"


@dataclass
class Level:
    """
    A critical price level to monitor.
    
    Attributes:
        id: Stable identifier (e.g., "VWAP", "STRIKE_545", "ROUND_550")
        price: Level price in dollars
        kind: Type of level
        metadata: Optional dict for additional info (e.g., strike right, gamma values)
    """
    id: str
    price: float
    kind: LevelKind
    metadata: Optional[dict] = None
    valid_from_ns: Optional[int] = None
    valid_to_ns: Optional[int] = None
    dynamic: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Level):
            return False
        return self.id == other.id


class LevelUniverse:
    """
    Generates and maintains the active set of critical levels.
    
    Usage:
        universe = LevelUniverse()
        levels = universe.get_levels(market_state)
    """
    
    def __init__(self, user_hotzones: Optional[List[float]] = None):
        """
        Initialize level universe.
        
        Args:
            user_hotzones: Optional list of user-defined price levels to always monitor
        """
        self.user_hotzones = user_hotzones or []
        
        # Cached wall levels (updated when flow changes significantly)
        self._cached_call_wall: Optional[float] = None
        self._cached_put_wall: Optional[float] = None
        self._cached_gamma_flip: Optional[float] = None
    
    def get_levels(self, market_state: MarketState, ts_ns: Optional[int] = None) -> List[Level]:
        """
        Generate all active levels for the current market state.
        
        Args:
            market_state: Current market state with spot, flow data, etc.
            
        Returns:
            List of Level objects
        """
        levels: List[Level] = []

        spot = market_state.get_spot()
        if spot is None:
            # No spot price yet, return empty
            return levels
        snapshot_ts_ns = ts_ns if ts_ns is not None else market_state.get_current_ts_ns()
        dynamic_kinds = {
            LevelKind.VWAP,
            LevelKind.SMA_200,
            LevelKind.SMA_400,
            LevelKind.SESSION_HIGH,
            LevelKind.SESSION_LOW,
            LevelKind.CALL_WALL,
            LevelKind.PUT_WALL,
            LevelKind.GAMMA_FLIP
        }
        
        # ========== VWAP ==========
        if CONFIG.VWAP_ENABLED:
            vwap = market_state.get_vwap()
            if vwap is not None:
                levels.append(Level(
                    id="VWAP",
                    price=vwap,
                    kind=LevelKind.VWAP,
                    valid_from_ns=snapshot_ts_ns,
                    dynamic=True
                ))

        # ========== Structural levels (Context) ==========
        pm_high = market_state.get_premarket_high()
        if pm_high is not None:
            levels.append(Level(
                id="PM_HIGH",
                price=pm_high,
                kind=LevelKind.PM_HIGH,
                valid_from_ns=snapshot_ts_ns
            ))

        pm_low = market_state.get_premarket_low()
        if pm_low is not None:
            levels.append(Level(
                id="PM_LOW",
                price=pm_low,
                kind=LevelKind.PM_LOW,
                valid_from_ns=snapshot_ts_ns
            ))

        or_high = market_state.get_opening_range_high()
        if or_high is not None:
            levels.append(Level(
                id="OR_HIGH",
                price=or_high,
                kind=LevelKind.OR_HIGH,
                valid_from_ns=snapshot_ts_ns
            ))

        or_low = market_state.get_opening_range_low()
        if or_low is not None:
            levels.append(Level(
                id="OR_LOW",
                price=or_low,
                kind=LevelKind.OR_LOW,
                valid_from_ns=snapshot_ts_ns
            ))

        sma_200 = market_state.get_sma_200()
        if sma_200 is not None:
            levels.append(Level(
                id="SMA_200",
                price=sma_200,
                kind=LevelKind.SMA_200,
                valid_from_ns=snapshot_ts_ns,
                dynamic=True
            ))

        sma_400 = market_state.get_sma_400()
        if sma_400 is not None:
            levels.append(Level(
                id="SMA_400",
                price=sma_400,
                kind=LevelKind.SMA_400,
                valid_from_ns=snapshot_ts_ns,
                dynamic=True
            ))
        
        # ========== Session High/Low ==========
        session_high = market_state.get_session_high()
        if session_high is not None:
            levels.append(Level(
                id="SESSION_HIGH",
                price=session_high,
                kind=LevelKind.SESSION_HIGH,
                valid_from_ns=snapshot_ts_ns,
                dynamic=True
            ))
        
        session_low = market_state.get_session_low()
        if session_low is not None:
            levels.append(Level(
                id="SESSION_LOW",
                price=session_low,
                kind=LevelKind.SESSION_LOW,
                valid_from_ns=snapshot_ts_ns,
                dynamic=True
            ))
        
        # ========== Round numbers ==========
        round_levels = self._generate_round_levels(spot, snapshot_ts_ns)
        levels.extend(round_levels)
        
        # ========== Option strikes ==========
        strike_levels = self._generate_strike_levels(market_state, spot, snapshot_ts_ns)
        levels.extend(strike_levels)
        
        # ========== Flow-derived walls ==========
        wall_levels = self._generate_wall_levels(market_state, spot, snapshot_ts_ns)
        levels.extend(wall_levels)
        
        # ========== User hotzones ==========
        for hotzone_price in self.user_hotzones:
            levels.append(Level(
                id=f"HOTZONE_{int(hotzone_price)}",
                price=hotzone_price,
                kind=LevelKind.USER_HOTZONE,
                valid_from_ns=snapshot_ts_ns
            ))
        
        # Deduplicate by ID (in case of overlaps)
        unique_levels = list({level.id: level for level in levels}.values())
        for level in unique_levels:
            level.dynamic = level.dynamic or level.kind in dynamic_kinds
            if level.valid_from_ns is None:
                level.valid_from_ns = snapshot_ts_ns
        
        return unique_levels
    
    def _generate_round_levels(self, spot: float, ts_ns: Optional[int]) -> List[Level]:
        """
        Generate round number levels near spot.
        
        Args:
            spot: Current spot price
            
        Returns:
            List of round levels
        """
        levels = []
        spacing = CONFIG.ROUND_LEVELS_SPACING
        range_dollars = CONFIG.STRIKE_RANGE  # use same range as strikes
        
        # Generate rounds from spot - range to spot + range
        min_level = int((spot - range_dollars) / spacing) * spacing
        max_level = int((spot + range_dollars) / spacing) * spacing
        
        current = min_level
        while current <= max_level:
            if current > 0:  # skip negative prices
                levels.append(Level(
                    id=f"ROUND_{int(current)}",
                    price=float(current),
                    kind=LevelKind.ROUND,
                    valid_from_ns=ts_ns
                ))
            current += spacing
        
        return levels
    
    def _generate_strike_levels(
        self,
        market_state: MarketState,
        spot: float,
        ts_ns: Optional[int]
    ) -> List[Level]:
        """
        Generate strike levels from active option flows.
        
        Args:
            market_state: Market state with option flows
            spot: Current spot price
            
        Returns:
            List of strike levels
        """
        levels = []
        strikes_seen = set()
        
        # Get option flows near spot
        range_dollars = CONFIG.STRIKE_RANGE
        option_flows = market_state.get_option_flows_near_level(
            level_price=spot,
            strike_range=range_dollars
        )
        
        for flow in option_flows:
            strike = flow.strike
            if strike not in strikes_seen:
                strikes_seen.add(strike)
                levels.append(Level(
                    id=f"STRIKE_{int(strike)}",
                    price=strike,
                    kind=LevelKind.STRIKE,
                    metadata={
                        "has_calls": any(f.strike == strike and f.right == 'C' for f in option_flows),
                        "has_puts": any(f.strike == strike and f.right == 'P' for f in option_flows)
                    },
                    valid_from_ns=ts_ns
                ))
        
        return levels
    
    def _generate_wall_levels(
        self,
        market_state: MarketState,
        spot: float,
        ts_ns: Optional[int]
    ) -> List[Level]:
        """
        Generate flow-derived wall levels (call wall, put wall, gamma flip).
        
        This method computes walls from net dealer gamma flow near strikes.
        Per §5.3 of PLAN.md, walls are identified as strikes with highest
        magnitude gamma flow for calls and puts.
        
        Args:
            market_state: Market state with option flows
            spot: Current spot price
            
        Returns:
            List of wall levels
        """
        levels = []
        
        # Get option flows across all active strikes
        # We'll look at a wider range for wall identification
        wall_range = CONFIG.STRIKE_RANGE * 2  # look ±10 instead of ±5
        option_flows = market_state.get_option_flows_near_level(
            level_price=spot,
            strike_range=wall_range
        )
        
        if not option_flows:
            return levels
        
        # Aggregate gamma flow by strike and right
        call_gamma_by_strike = {}
        put_gamma_by_strike = {}
        
        for flow in option_flows:
            strike = flow.strike
            if flow.right == 'C':
                if strike not in call_gamma_by_strike:
                    call_gamma_by_strike[strike] = 0.0
                call_gamma_by_strike[strike] += flow.net_gamma_flow
            elif flow.right == 'P':
                if strike not in put_gamma_by_strike:
                    put_gamma_by_strike[strike] = 0.0
                put_gamma_by_strike[strike] += flow.net_gamma_flow
        
        # Find call wall: strike with highest (most negative) dealer gamma from calls
        # Per §5.3: dealers are SHORT gamma when customers buy, so net_gamma_flow is negative
        # Call wall = strike where dealers have most negative gamma (highest customer demand)
        if call_gamma_by_strike:
            call_wall_strike = min(call_gamma_by_strike.items(), key=lambda x: x[1])[0]
            call_wall_gamma = call_gamma_by_strike[call_wall_strike]
            
            # Only create wall if gamma is meaningfully negative (threshold TBD)
            if call_wall_gamma < -1000:  # arbitrary threshold for now
                self._cached_call_wall = call_wall_strike
                levels.append(Level(
                    id=f"CALL_WALL",
                    price=call_wall_strike,
                    kind=LevelKind.CALL_WALL,
                    metadata={
                        "net_dealer_gamma": call_wall_gamma,
                        "strike": call_wall_strike
                    },
                    valid_from_ns=ts_ns,
                    dynamic=True
                ))
        
        # Find put wall: strike with highest (most negative) dealer gamma from puts
        if put_gamma_by_strike:
            put_wall_strike = min(put_gamma_by_strike.items(), key=lambda x: x[1])[0]
            put_wall_gamma = put_gamma_by_strike[put_wall_strike]
            
            if put_wall_gamma < -1000:
                self._cached_put_wall = put_wall_strike
                levels.append(Level(
                    id=f"PUT_WALL",
                    price=put_wall_strike,
                    kind=LevelKind.PUT_WALL,
                    metadata={
                        "net_dealer_gamma": put_wall_gamma,
                        "strike": put_wall_strike
                    },
                    valid_from_ns=ts_ns,
                    dynamic=True
                ))
        
        # Optional: Gamma flip level (HVL approximation)
        # This is where cumulative dealer gamma changes sign
        # For v1, we'll skip this and let Agent E implement if needed
        
        return levels
    
    def set_user_hotzones(self, hotzones: List[float]):
        """
        Update user-defined hotzone levels.
        
        Args:
            hotzones: List of price levels to monitor
        """
        self.user_hotzones = hotzones
    
    def get_cached_walls(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get cached wall levels (call wall, put wall, gamma flip).
        
        Returns:
            (call_wall, put_wall, gamma_flip) tuple
        """
        return (self._cached_call_wall, self._cached_put_wall, self._cached_gamma_flip)
