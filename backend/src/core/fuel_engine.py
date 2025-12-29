"""
Fuel Engine: Dealer gamma impulse from ES 0DTE options

Estimates whether dealers will AMPLIFY or DAMPEN a move near a level
by computing net dealer gamma transfer from option trades.

Based on PLAN.md §5.3

Interfaces consumed:
- event_types (Agent A): OptionTrade dataclass
- market_state (Agent C): MarketState with option flow aggregates
- config (Agent A): window sizes, strike ranges
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
from collections import defaultdict

from .market_state import MarketState, OptionFlowAggregate
from src.common.config import CONFIG


class FuelEffect(str, Enum):
    """Gamma effect classification"""
    AMPLIFY = "AMPLIFY"    # Dealers short gamma → trend accelerant
    DAMPEN = "DAMPEN"      # Dealers long gamma → mean reversion
    NEUTRAL = "NEUTRAL"    # Minimal gamma exposure


@dataclass
class GammaWall:
    """Identified gamma wall (call or put)"""
    strike: float
    net_gamma: float
    wall_type: str  # 'CALL' or 'PUT'
    strength: float  # Magnitude indicator


@dataclass
class FuelMetrics:
    """Output of fuel engine computation"""
    effect: FuelEffect
    net_dealer_gamma: float  # Negative = dealers short gamma (AMPLIFY)
    call_wall: Optional[GammaWall]
    put_wall: Optional[GammaWall]
    hvl: Optional[float]  # High Volatility Line (gamma flip level)
    confidence: float  # 0-1, based on total flow activity
    gamma_by_strike: dict  # {strike: net_gamma} for debugging/visualization


class FuelEngine:
    """
    Computes dealer gamma exposure and flow-based walls.
    
    For a given level L:
    - Sum net dealer gamma from option trades at strikes near level
    - Identify call/put walls (strikes with max gamma concentration)
    - Classify effect: AMPLIFY (dealers short gamma) or DAMPEN (dealers long gamma)
    
    Key insight from MarketState:
    - MarketState.option_flows stores net_gamma_flow which is NET DEALER GAMMA
    - Customer buys option → dealer sells gamma → net_gamma_flow is NEGATIVE
    - Negative net_dealer_gamma → dealers SHORT gamma → will hedge by chasing → AMPLIFY
    
    Interfaces consumed:
    - MarketState queries (Agent C) for option flows near level
    - Config thresholds (Agent A) for strike range, windows
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: Config object (defaults to global CONFIG)
        """
        self.config = config or CONFIG
        
        # Extract config values
        self.window_seconds = self.config.W_g  # Rolling window for option flows
        self.wall_window_seconds = self.config.W_wall  # Lookback for wall identification
        self.strike_range = self.config.FUEL_STRIKE_RANGE  # ±N ES points around level
        
        # Thresholds for classification
        self.gamma_threshold = 10000.0  # Minimum |gamma| for non-neutral classification
        self.wall_strength_threshold = 50000.0  # Minimum gamma for strong wall
    
    def compute_fuel_state(
        self,
        level_price: float,
        market_state: MarketState,
        exp_date_filter: Optional[str] = None
    ) -> FuelMetrics:
        """
        Compute fuel metrics for a level.
        
        Args:
            level_price: The critical level being tested
            market_state: MarketState instance with option flow aggregates
            exp_date_filter: Optional expiration date filter (e.g., "2025-12-16" for 0DTE)
            
        Returns:
            FuelMetrics with gamma effect and wall identification
        """
        # Get option flows near the level
        flows = market_state.get_option_flows_near_level(
            level_price=level_price,
            strike_range=self.strike_range,
            exp_date_filter=exp_date_filter
        )
        
        if not flows:
            # No option flow data available
            return self._neutral_metrics()
        
        # Compute net dealer gamma near level
        net_dealer_gamma = sum(flow.net_gamma_flow for flow in flows)
        
        # Build gamma by strike for analysis
        gamma_by_strike = self._aggregate_gamma_by_strike(flows)
        
        # Identify call and put walls
        call_wall = self._identify_call_wall(flows, level_price)
        put_wall = self._identify_put_wall(flows, level_price)
        
        # Estimate HVL (gamma flip level) - optional
        hvl = self._estimate_hvl(gamma_by_strike, level_price)
        
        # Classify effect
        effect = self._classify_effect(net_dealer_gamma)
        
        # Compute confidence based on total flow activity
        total_volume = sum(flow.cumulative_volume for flow in flows)
        confidence = min(1.0, total_volume / 1000.0)  # Scale to 1.0 at 1000 contracts
        
        return FuelMetrics(
            effect=effect,
            net_dealer_gamma=net_dealer_gamma,
            call_wall=call_wall,
            put_wall=put_wall,
            hvl=hvl,
            confidence=confidence,
            gamma_by_strike=gamma_by_strike
        )
    
    def _neutral_metrics(self) -> FuelMetrics:
        """Return neutral metrics when no data available."""
        return FuelMetrics(
            effect=FuelEffect.NEUTRAL,
            net_dealer_gamma=0.0,
            call_wall=None,
            put_wall=None,
            hvl=None,
            confidence=0.0,
            gamma_by_strike={}
        )
    
    def _aggregate_gamma_by_strike(
        self,
        flows: List[OptionFlowAggregate]
    ) -> dict:
        """
        Aggregate net gamma by strike (combine calls and puts at same strike).
        
        Args:
            flows: List of option flow aggregates
            
        Returns:
            Dict[strike: float, net_gamma: float]
        """
        gamma_by_strike = defaultdict(float)
        
        for flow in flows:
            gamma_by_strike[flow.strike] += flow.net_gamma_flow
        
        return dict(gamma_by_strike)
    
    def _identify_call_wall(
        self,
        flows: List[OptionFlowAggregate],
        level_price: float
    ) -> Optional[GammaWall]:
        """
        Identify call wall: strike with maximum POSITIVE gamma flow (dealers LONG call gamma).
        
        Call wall acts as resistance (dealers hedging by selling as price rises).
        
        Args:
            flows: List of option flow aggregates
            level_price: Current level price for context
            
        Returns:
            GammaWall or None if no significant wall
        """
        # Filter to calls only, above current level
        call_flows = [
            flow for flow in flows 
            if flow.right == 'C' and flow.strike >= level_price
        ]
        
        if not call_flows:
            return None
        
        # Find strike with maximum positive gamma flow
        # Positive net_gamma_flow means customers SOLD options → dealers BOUGHT → long gamma
        max_gamma = max(flow.net_gamma_flow for flow in call_flows)
        
        if max_gamma < self.wall_strength_threshold:
            return None
        
        # Find the strike with max gamma
        wall_strike = max(call_flows, key=lambda f: f.net_gamma_flow).strike
        
        return GammaWall(
            strike=wall_strike,
            net_gamma=max_gamma,
            wall_type='CALL',
            strength=max_gamma / self.wall_strength_threshold  # Normalized strength
        )
    
    def _identify_put_wall(
        self,
        flows: List[OptionFlowAggregate],
        level_price: float
    ) -> Optional[GammaWall]:
        """
        Identify put wall: strike with maximum POSITIVE gamma flow (dealers LONG put gamma).
        
        Put wall acts as support (dealers hedging by buying as price falls).
        
        Args:
            flows: List of option flow aggregates
            level_price: Current level price for context
            
        Returns:
            GammaWall or None if no significant wall
        """
        # Filter to puts only, below current level
        put_flows = [
            flow for flow in flows 
            if flow.right == 'P' and flow.strike <= level_price
        ]
        
        if not put_flows:
            return None
        
        # Find strike with maximum positive gamma flow
        max_gamma = max(flow.net_gamma_flow for flow in put_flows)
        
        if max_gamma < self.wall_strength_threshold:
            return None
        
        # Find the strike with max gamma
        wall_strike = max(put_flows, key=lambda f: f.net_gamma_flow).strike
        
        return GammaWall(
            strike=wall_strike,
            net_gamma=max_gamma,
            wall_type='PUT',
            strength=max_gamma / self.wall_strength_threshold
        )
    
    def _estimate_hvl(
        self,
        gamma_by_strike: dict,
        level_price: float
    ) -> Optional[float]:
        """
        Estimate HVL (High Volatility Line / gamma flip level).
        
        This is the strike where net dealer gamma changes sign.
        - Below HVL: dealers typically short gamma (need to chase)
        - Above HVL: dealers typically long gamma (dampening moves)
        
        Simple heuristic: find strike nearest to level where gamma crosses zero.
        
        Args:
            gamma_by_strike: Dict mapping strike to net gamma
            level_price: Current level price
            
        Returns:
            HVL strike or None if cannot determine
        """
        if len(gamma_by_strike) < 3:
            return None
        
        # Sort strikes
        sorted_strikes = sorted(gamma_by_strike.keys())
        
        # Look for sign changes
        for i in range(len(sorted_strikes) - 1):
            strike_low = sorted_strikes[i]
            strike_high = sorted_strikes[i + 1]
            gamma_low = gamma_by_strike[strike_low]
            gamma_high = gamma_by_strike[strike_high]
            
            # Check for sign change
            if gamma_low * gamma_high < 0:
                # Sign change detected, interpolate
                # Linear interpolation to find zero crossing
                if gamma_high != gamma_low:
                    hvl = strike_low + (strike_high - strike_low) * (
                        -gamma_low / (gamma_high - gamma_low)
                    )
                    return hvl
        
        # No clear flip detected
        return None
    
    def _classify_effect(self, net_dealer_gamma: float) -> FuelEffect:
        """
        Classify gamma effect based on net dealer gamma.
        
        Logic:
        - net_dealer_gamma < -threshold: dealers SHORT gamma → AMPLIFY (chase moves)
        - net_dealer_gamma > +threshold: dealers LONG gamma → DAMPEN (fade moves)
        - else: NEUTRAL
        
        Args:
            net_dealer_gamma: Net dealer gamma near level
            
        Returns:
            FuelEffect classification
        """
        if net_dealer_gamma < -self.gamma_threshold:
            return FuelEffect.AMPLIFY
        elif net_dealer_gamma > self.gamma_threshold:
            return FuelEffect.DAMPEN
        else:
            return FuelEffect.NEUTRAL
    
    def get_all_walls(
        self,
        market_state: MarketState,
        exp_date_filter: Optional[str] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> Tuple[Optional[GammaWall], Optional[GammaWall]]:
        """
        Identify global call and put walls across entire strike range.
        
        Useful for level universe generation and market regime analysis.
        
        Args:
            market_state: MarketState instance
            exp_date_filter: Optional expiration date filter
            min_strike: Optional minimum strike (default: spot - 10)
            max_strike: Optional maximum strike (default: spot + 10)
            
        Returns:
            (call_wall, put_wall) tuple
        """
        spot = market_state.get_spot()
        if spot is None:
            return None, None
        
        # Default strike range if not specified
        if min_strike is None:
            min_strike = spot - 10.0
        if max_strike is None:
            max_strike = spot + 10.0
        
        # Get all option flows in range
        all_flows = []
        for (strike, right, exp_date), flow in market_state.option_flows.items():
            if exp_date_filter and exp_date != exp_date_filter:
                continue
            if min_strike <= strike <= max_strike:
                all_flows.append(flow)
        
        if not all_flows:
            return None, None
        
        # Use same wall identification logic
        call_wall = self._identify_call_wall(all_flows, spot)
        put_wall = self._identify_put_wall(all_flows, spot)
        
        return call_wall, put_wall
