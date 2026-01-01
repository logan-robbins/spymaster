"""
Confluence Computer: real-time confluence feature computation for live streaming.

Simplified version of pipeline's compute_confluence logic for real-time use.
Computes:
- confluence_count: Number of nearby key levels
- confluence_pressure: Weighted pressure from stacked levels
- confluence_alignment: Whether level aligns with market structure
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .level_universe import Level, LevelKind
from .market_state import MarketState
from src.common.config import CONFIG


class ConfluenceAlignment(Enum):
    """Alignment of level with market structure."""
    OPPOSED = -1    # Level opposes trend/structure
    NEUTRAL = 0     # No clear alignment
    ALIGNED = 1     # Level aligns with trend/structure


@dataclass
class ConfluenceMetrics:
    """Confluence metrics for a single level."""
    confluence_count: int                    # Number of nearby key levels
    confluence_pressure: float               # Normalized pressure (0-1)
    confluence_alignment: ConfluenceAlignment  # Trend alignment
    confluence_min_distance: Optional[float] # Distance to nearest confluence level
    
    # Individual level breakdown (for debugging)
    nearby_levels: List[Tuple[str, float, float]]  # (level_id, distance, weight)


class ConfluenceComputer:
    """
    Compute confluence features in real-time for streaming use.
    
    Usage:
        computer = ConfluenceComputer()
        metrics = computer.compute(target_level, all_levels, market_state)
    """
    
    # Weights for different level kinds
    LEVEL_WEIGHTS = {
        LevelKind.PM_HIGH: 1.0,
        LevelKind.PM_LOW: 1.0,
        LevelKind.OR_HIGH: 0.9,
        LevelKind.OR_LOW: 0.9,
        LevelKind.SMA_90: 0.8,
        LevelKind.EMA_20: 0.8,
        LevelKind.VWAP: 0.7,
        LevelKind.SESSION_HIGH: 0.6,
        LevelKind.SESSION_LOW: 0.6,
        LevelKind.CALL_WALL: 1.0,
        LevelKind.PUT_WALL: 1.0,
        LevelKind.GAMMA_FLIP: 0.85
    }
    
    def __init__(self, confluence_band: Optional[float] = None):
        """
        Initialize confluence computer.
        
        Args:
            confluence_band: Distance band for confluence detection (default: CONFIG.CONFLUENCE_BAND)
        """
        self.confluence_band = confluence_band or CONFIG.CONFLUENCE_BAND
        self.total_weight = sum(self.LEVEL_WEIGHTS.values())
    
    def compute(
        self,
        target_level: Level,
        all_levels: List[Level],
        market_state: MarketState,
        direction: str  # "SUPPORT" or "RESISTANCE"
    ) -> ConfluenceMetrics:
        """
        Compute confluence metrics for target level.
        
        Args:
            target_level: Level to analyze
            all_levels: All active levels in universe
            market_state: Current market state
            direction: Level direction ("SUPPORT" or "RESISTANCE")
            
        Returns:
            ConfluenceMetrics with confluence features
        """
        target_price = target_level.price
        target_kind = target_level.kind
        
        # Find nearby levels (exclude self)
        nearby_levels: List[Tuple[str, float, float]] = []  # (id, distance, weight)
        distances: List[float] = []
        weights: List[float] = []
        
        for level in all_levels:
            # Skip self
            if level.id == target_level.id:
                continue
            
            # Skip if not a weighted level kind
            if level.kind not in self.LEVEL_WEIGHTS:
                continue
            
            # Compute distance
            distance = abs(level.price - target_price)
            
            # Skip if outside band
            if distance > self.confluence_band:
                continue
            
            # Get weight for this level kind
            weight = self.LEVEL_WEIGHTS[level.kind]
            
            distances.append(distance)
            weights.append(weight)
            nearby_levels.append((level.id, distance, weight))
        
        # Compute confluence metrics
        if not distances:
            return ConfluenceMetrics(
                confluence_count=0,
                confluence_pressure=0.0,
                confluence_alignment=ConfluenceAlignment.NEUTRAL,
                confluence_min_distance=None,
                nearby_levels=[]
            )
        
        # Count of nearby levels
        confluence_count = len(distances)
        
        # Minimum distance to any confluent level
        min_distance = min(distances)
        
        # Weighted pressure with distance decay
        weighted_score = 0.0
        for dist, weight in zip(distances, weights):
            # Distance decay: linearly from 1.0 at distance=0 to 0.0 at distance=band
            decay = max(0.0, 1.0 - (dist / self.confluence_band))
            weighted_score += weight * decay
        
        # Normalize by total possible weight
        confluence_pressure = weighted_score / self.total_weight if self.total_weight > 0 else 0.0
        
        # Compute confluence alignment
        alignment = self._compute_alignment(
            target_level=target_level,
            all_levels=all_levels,
            market_state=market_state,
            direction=direction
        )
        
        return ConfluenceMetrics(
            confluence_count=confluence_count,
            confluence_pressure=confluence_pressure,
            confluence_alignment=alignment,
            confluence_min_distance=min_distance,
            nearby_levels=nearby_levels
        )
    
    def _compute_alignment(
        self,
        target_level: Level,
        all_levels: List[Level],
        market_state: MarketState,
        direction: str
    ) -> ConfluenceAlignment:
        """
        Compute whether level aligns with market structure (SMAs, momentum).
        
        Logic:
        - For SUPPORT (spot > level): ALIGNED if price above SMAs and SMAs rising
        - For RESISTANCE (spot < level): ALIGNED if price below SMAs and SMAs falling
        - OPPOSED if structure contradicts
        - NEUTRAL if no clear structure
        """
        spot = market_state.get_spot()
        if spot is None:
            return ConfluenceAlignment.NEUTRAL
        
        # Find MA levels
        sma_90 = None
        ema_20 = None
        for level in all_levels:
            if level.kind == LevelKind.SMA_90:
                sma_90 = level.price
            elif level.kind == LevelKind.EMA_20:
                ema_20 = level.price

        # If no SMAs available, can't compute alignment
        if sma_90 is None and ema_20 is None:
            return ConfluenceAlignment.NEUTRAL

        # Determine trend from SMA relationship
        # Bullish if EMA_20 > SMA_90 (shorter MA above longer MA)
        # Bearish if EMA_20 < SMA_90
        trend_bullish = None
        if ema_20 is not None and sma_90 is not None:
            if ema_20 > sma_90 + 0.05:  # Small threshold to avoid noise
                trend_bullish = True
            elif ema_20 < sma_90 - 0.05:
                trend_bullish = False
        
        if trend_bullish is None:
            return ConfluenceAlignment.NEUTRAL
        
        # Check alignment based on direction and trend
        if direction == "SUPPORT":
            # Support level: aligned if trend is bullish (expecting bounce)
            if trend_bullish:
                return ConfluenceAlignment.ALIGNED
            else:
                return ConfluenceAlignment.OPPOSED
        else:  # RESISTANCE
            # Resistance level: aligned if trend is bearish (expecting rejection)
            if not trend_bullish:
                return ConfluenceAlignment.ALIGNED
            else:
                return ConfluenceAlignment.OPPOSED
    
    def compute_hierarchical_level(
        self,
        metrics: ConfluenceMetrics,
        level_kind: LevelKind
    ) -> int:
        """
        Compute hierarchical confluence level (0-10) based on confluence metrics.
        
        Scale:
        0 = Undefined (no confluence)
        1-2 = ULTRA_PREMIUM (high confluence, strong alignment, premium levels)
        3-4 = PREMIUM (good confluence, premium levels)
        5-6 = STRONG (decent confluence)
        7-8 = MODERATE (some confluence)
        9-10 = CONSOLIDATION (high confluence but weaker levels)
        
        Args:
            metrics: ConfluenceMetrics from compute()
            level_kind: Type of level being evaluated
            
        Returns:
            Confluence level 0-10
        """
        if metrics.confluence_count == 0:
            return 0
        
        # Premium level kinds (structural importance)
        premium_levels = {
            LevelKind.PM_HIGH, LevelKind.PM_LOW,
            LevelKind.OR_HIGH, LevelKind.OR_LOW,
            LevelKind.CALL_WALL, LevelKind.PUT_WALL
        }
        
        is_premium = level_kind in premium_levels
        
        # Base score from confluence metrics
        if metrics.confluence_count >= 4 and metrics.confluence_pressure > 0.7:
            # Very high confluence
            base_score = 9 if not is_premium else 1  # Premium gets lower (better) score
        elif metrics.confluence_count >= 3 and metrics.confluence_pressure > 0.5:
            # High confluence
            base_score = 7 if not is_premium else 3
        elif metrics.confluence_count >= 2 and metrics.confluence_pressure > 0.3:
            # Moderate confluence
            base_score = 5 if not is_premium else 3
        else:
            # Low confluence
            base_score = 7 if not is_premium else 5
        
        # Adjust based on alignment
        if metrics.confluence_alignment == ConfluenceAlignment.ALIGNED:
            base_score = max(1, base_score - 1)  # Better score
        elif metrics.confluence_alignment == ConfluenceAlignment.OPPOSED:
            base_score = min(10, base_score + 1)  # Worse score
        
        return base_score


def get_confluence_level_name(level: int) -> str:
    """
    Map hierarchical confluence level to human-readable name.
    
    Args:
        level: Confluence level 0-10
        
    Returns:
        String name (ULTRA_PREMIUM, PREMIUM, STRONG, etc.)
    """
    if level == 0:
        return "UNDEFINED"
    elif level <= 2:
        return "ULTRA_PREMIUM"
    elif level <= 4:
        return "PREMIUM"
    elif level <= 6:
        return "STRONG"
    elif level <= 8:
        return "MODERATE"
    else:
        return "CONSOLIDATION"
