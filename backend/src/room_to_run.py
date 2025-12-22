"""
Room to Run: compute runway to next obstacle after break/reject.

Agent F deliverable per §12 of PLAN.md.

This module analyzes the path from a given level in a specific direction
and computes:
- Distance to next obstacle (nearest level)
- Runway quality (CLEAR vs OBSTRUCTED based on intermediate levels)

Per §5.5 of PLAN.md, this helps quantify "how far can price run" after
a break or reject signal.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .level_universe import Level, LevelKind


class Direction(Enum):
    """Direction of expected move."""
    UP = "UP"
    DOWN = "DOWN"


class RunwayQuality(Enum):
    """Quality of runway to next obstacle."""
    CLEAR = "CLEAR"  # No strong walls in between
    OBSTRUCTED = "OBSTRUCTED"  # Strong walls blocking path


@dataclass
class Runway:
    """
    Runway computation result.
    
    Attributes:
        direction: Direction of expected move (UP/DOWN)
        distance: Distance in dollars to next obstacle
        next_obstacle: The Level object that is the next obstacle
        quality: CLEAR or OBSTRUCTED
        intermediate_levels: List of levels between current and next obstacle
    """
    direction: Direction
    distance: float
    next_obstacle: Optional[Level]
    quality: RunwayQuality
    intermediate_levels: List[Level]


class RoomToRun:
    """
    Computes runway (room to run) from a given level in a direction.
    
    Usage:
        rtr = RoomToRun()
        runway = rtr.compute_runway(
            current_level=level_545,
            direction=Direction.DOWN,
            all_levels=levels,
            spot=545.42
        )
    """
    
    # Level kinds that are considered "strong obstacles"
    STRONG_OBSTACLE_KINDS = {
        LevelKind.CALL_WALL,
        LevelKind.PUT_WALL,
        LevelKind.SESSION_HIGH,
        LevelKind.SESSION_LOW,
        LevelKind.VWAP,
        LevelKind.GAMMA_FLIP
    }
    
    def compute_runway(
        self,
        current_level: Level,
        direction: Direction,
        all_levels: List[Level],
        spot: float
    ) -> Runway:
        """
        Compute runway from current level in given direction.
        
        Args:
            current_level: The level we're analyzing (e.g., level being broken/rejected)
            direction: Direction of expected move (UP/DOWN)
            all_levels: All active levels in the universe
            spot: Current spot price
            
        Returns:
            Runway object with distance, next obstacle, and quality
        """
        current_price = current_level.price
        
        # Filter levels in the target direction
        if direction == Direction.UP:
            candidate_levels = [
                level for level in all_levels
                if level.price > current_price
            ]
            # Sort ascending (nearest first)
            candidate_levels.sort(key=lambda x: x.price)
        else:  # Direction.DOWN
            candidate_levels = [
                level for level in all_levels
                if level.price < current_price
            ]
            # Sort descending (nearest first)
            candidate_levels.sort(key=lambda x: x.price, reverse=True)
        
        if not candidate_levels:
            # No obstacles in this direction
            return Runway(
                direction=direction,
                distance=float('inf'),
                next_obstacle=None,
                quality=RunwayQuality.CLEAR,
                intermediate_levels=[]
            )
        
        # Next obstacle is the nearest level
        next_obstacle = candidate_levels[0]
        distance = abs(next_obstacle.price - current_price)
        
        # Determine runway quality based on intermediate levels
        # "Obstructed" if there are strong walls between current and next
        intermediate_levels = self._get_intermediate_levels(
            current_price=current_price,
            target_price=next_obstacle.price,
            all_levels=candidate_levels  # pass all candidate levels
        )
        
        quality = self._assess_runway_quality(intermediate_levels)
        
        return Runway(
            direction=direction,
            distance=distance,
            next_obstacle=next_obstacle,
            quality=quality,
            intermediate_levels=intermediate_levels
        )
    
    def _get_intermediate_levels(
        self,
        current_price: float,
        target_price: float,
        all_levels: List[Level]
    ) -> List[Level]:
        """
        Get levels between current price and target price.
        
        Args:
            current_price: Starting price
            target_price: Ending price
            all_levels: All candidate levels
            
        Returns:
            List of levels between current and target
        """
        min_price = min(current_price, target_price)
        max_price = max(current_price, target_price)
        
        intermediate = [
            level for level in all_levels
            if min_price < level.price < max_price
        ]
        
        return intermediate
    
    def _assess_runway_quality(self, intermediate_levels: List[Level]) -> RunwayQuality:
        """
        Assess whether runway is CLEAR or OBSTRUCTED.
        
        Logic:
        - CLEAR: No strong obstacles in between
        - OBSTRUCTED: At least one strong obstacle (wall, session high/low, etc.)
        
        Args:
            intermediate_levels: Levels between current and target
            
        Returns:
            RunwayQuality enum
        """
        for level in intermediate_levels:
            if level.kind in self.STRONG_OBSTACLE_KINDS:
                return RunwayQuality.OBSTRUCTED
        
        return RunwayQuality.CLEAR
    
    def compute_bidirectional_runway(
        self,
        current_level: Level,
        all_levels: List[Level],
        spot: float
    ) -> tuple[Runway, Runway]:
        """
        Compute runway in both directions (UP and DOWN).
        
        Args:
            current_level: The level we're analyzing
            all_levels: All active levels
            spot: Current spot price
            
        Returns:
            (runway_up, runway_down) tuple
        """
        runway_up = self.compute_runway(current_level, Direction.UP, all_levels, spot)
        runway_down = self.compute_runway(current_level, Direction.DOWN, all_levels, spot)
        
        return (runway_up, runway_down)


def get_break_direction(level_price: float, spot: float) -> Direction:
    """
    Determine break direction based on level position relative to spot.
    
    Per §3.2 of PLAN.md:
    - If spot > level: level is SUPPORT, break direction is DOWN
    - If spot < level: level is RESISTANCE, break direction is UP
    
    Args:
        level_price: Level price
        spot: Current spot price
        
    Returns:
        Direction enum (UP or DOWN)
    """
    if spot > level_price:
        # Level is support, break means going down through it
        return Direction.DOWN
    else:
        # Level is resistance, break means going up through it
        return Direction.UP


def get_reject_direction(level_price: float, spot: float) -> Direction:
    """
    Determine reject/bounce direction based on level position relative to spot.
    
    Reject direction is opposite of break direction:
    - If spot > level: reject means bouncing UP
    - If spot < level: reject means bouncing DOWN
    
    Args:
        level_price: Level price
        spot: Current spot price
        
    Returns:
        Direction enum (UP or DOWN)
    """
    break_dir = get_break_direction(level_price, spot)
    return Direction.DOWN if break_dir == Direction.UP else Direction.UP
