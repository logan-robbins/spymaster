"""
Test the comprehensive LevelSignal schema with all production fields.

Demonstrates backward compatibility and optional field usage.
"""

import pytest
import time
from src.common.schemas import (
    LevelSignal,
    LevelKind,
    OutcomeLabel,
    BarrierState,
    Direction,
    Signal,
    Confidence,
    FuelEffect,
    RunwayQuality,
)


class TestComprehensiveSchema:
    """Test comprehensive LevelSignal schema."""
    
    def test_minimal_schema_still_works(self):
        """Test that minimal fields (Agent A original) still work."""
        signal = LevelSignal(
            event_id="test_001",
            ts_event_ns=time.time_ns(),
            symbol="ES",
            level_price=6870.0,
            level_kind=LevelKind.STRIKE,
            is_first_15m=True,
            wall_ratio=2.5,
            gamma_exposure=5000000.0,
            tape_velocity=12.5,
        )
        
        assert signal.event_id == "test_001"
        assert signal.level_price == 6870.0
        assert signal.wall_ratio == 2.5
        # Optional fields should have defaults
        assert signal.outcome == OutcomeLabel.UNDEFINED
        assert signal.barrier_state is None
        assert signal.tape_imbalance is None
    
    def test_full_schema_with_all_metrics(self):
        """Test schema with all production metrics populated."""
        signal = LevelSignal(
            # Identity
            event_id="test_002",
            ts_event_ns=time.time_ns(),
            symbol="ES",
            
            # Market context
            spot=6874.2,
            bid=6874.1,
            ask=6874.3,
            
            # Level identity
            level_price=6870.0,
            level_kind=LevelKind.STRIKE,
            level_id="STRIKE_6870",
            direction=Direction.SUPPORT,
            distance=4.2,
            
            # Context
            is_first_15m=True,
            dist_to_sma_90=2.5,
            
            # Basic physics
            wall_ratio=3.5,
            replenishment_speed_ms=35.0,
            gamma_exposure=5000000.0,
            tape_velocity=15.0,
            
            # Scores
            break_score_raw=88.5,
            break_score_smooth=81.2,
            signal=Signal.BREAK,
            confidence=Confidence.HIGH,
            
            # Barrier metrics
            barrier_state=BarrierState.VACUUM,
            barrier_delta_liq=-8200.0,
            barrier_replenishment_ratio=0.15,
            barrier_added=3100,
            barrier_canceled=9800,
            barrier_filled=1500,
            
            # Tape metrics
            tape_imbalance=-0.45,
            tape_buy_vol=120000,
            tape_sell_vol=320000,
            tape_sweep_detected=True,
            tape_sweep_direction="DOWN",
            tape_sweep_notional=1250000.0,
            
            # Fuel metrics
            fuel_effect=FuelEffect.AMPLIFY,
            fuel_net_dealer_gamma=-185000.0,
            fuel_call_wall=6900.0,
            fuel_put_wall=6840.0,
            fuel_hvl=6870.0,
            
            # Runway metrics
            runway_direction="DOWN",
            runway_next_level_id="PUT_WALL",
            runway_next_level_price=6840.0,
            runway_distance=30.0,
            runway_quality=RunwayQuality.CLEAR,
            
            # Outcome
            outcome=OutcomeLabel.BREAK,
            future_price=6835.0,
            
            # Note
            note="Vacuum + dealers chase; sweep confirms"
        )
        
        # Verify all fields
        assert signal.spot == 6874.2
        assert signal.barrier_state == BarrierState.VACUUM
        assert signal.tape_imbalance == -0.45
        assert signal.fuel_effect == FuelEffect.AMPLIFY
        assert signal.runway_quality == RunwayQuality.CLEAR
        assert signal.signal == Signal.BREAK
        assert signal.confidence == Confidence.HIGH
    
    def test_incremental_feature_addition(self):
        """Test adding features incrementally (Agent A → B → C workflow)."""
        # Agent A: Basic physics
        signal = LevelSignalV1(
            event_id="test_003",
            ts_event_ns=time.time_ns(),
            level_price=687.0,
            level_kind=LevelKind.STRIKE,
            wall_ratio=2.0,
            gamma_exposure=3000000.0,
            tape_velocity=10.0,
        )
        
        # Agent B: Add context
        signal.is_first_15m = True
        signal.dist_to_sma_90 = 0.5
        signal.spot = 6873.50
        
        # Advanced physics (could be Agent A extended)
        signal.barrier_state = BarrierState.WALL
        signal.barrier_replenishment_ratio = 1.8
        signal.tape_imbalance = 0.3
        
        # Agent C: Add outcome
        signal.outcome = OutcomeLabel.BOUNCE
        signal.future_price = 6878.0
        
        # Verify incremental additions
        assert signal.wall_ratio == 2.0
        assert signal.is_first_15m is True
        assert signal.barrier_state == BarrierState.WALL
        assert signal.outcome == OutcomeLabel.BOUNCE
    
    def test_all_enums_accessible(self):
        """Test that all enum types are properly defined."""
        # LevelKind
        assert LevelKind.STRIKE == "STRIKE"
        assert LevelKind.VWAP == "VWAP"
        assert LevelKind.CALL_WALL == "CALL_WALL"
        
        # OutcomeLabel
        assert OutcomeLabel.BOUNCE == "BOUNCE"
        assert OutcomeLabel.BREAK == "BREAK"
        
        # BarrierState
        assert BarrierState.VACUUM == "VACUUM"
        assert BarrierState.WALL == "WALL"
        
        # Direction
        assert Direction.SUPPORT == "SUPPORT"
        assert Direction.RESISTANCE == "RESISTANCE"
        
        # Signal
        assert Signal.BREAK == "BREAK"
        assert Signal.REJECT == "REJECT"
        
        # Confidence
        assert Confidence.HIGH == "HIGH"
        assert Confidence.MEDIUM == "MEDIUM"
        
        # FuelEffect
        assert FuelEffect.AMPLIFY == "AMPLIFY"
        assert FuelEffect.DAMPEN == "DAMPEN"
        
        # RunwayQuality
        assert RunwayQuality.CLEAR == "CLEAR"
        assert RunwayQuality.OBSTRUCTED == "OBSTRUCTED"
    
    def test_schema_validation_constraints(self):
        """Test that schema validation works for constrained fields."""
        # Valid scores (0-100)
        signal = LevelSignalV1(
            event_id="test_004",
            ts_event_ns=time.time_ns(),
            level_price=687.0,
            level_kind=LevelKind.STRIKE,
            wall_ratio=2.0,
            gamma_exposure=0.0,
            tape_velocity=0.0,
            break_score_raw=85.5,
            break_score_smooth=80.0,
        )
        assert signal.break_score_raw == 85.5
        
        # Valid imbalance (-1 to +1)
        signal.tape_imbalance = -0.75
        assert signal.tape_imbalance == -0.75
        
        # Valid barrier sizes (>= 0)
        signal.barrier_added = 5000
        signal.barrier_canceled = 2000
        signal.barrier_filled = 1000
        assert signal.barrier_added == 5000
    
    def test_feature_vector_extraction(self):
        """Test extracting features for ML experimentation."""
        signal = LevelSignalV1(
            event_id="test_005",
            ts_event_ns=time.time_ns(),
            level_price=687.0,
            level_kind=LevelKind.STRIKE,
            wall_ratio=2.5,
            gamma_exposure=5000000.0,
            tape_velocity=12.0,
            barrier_replenishment_ratio=0.8,
            tape_imbalance=0.3,
            fuel_net_dealer_gamma=-100000.0,
            break_score_raw=75.0,
        )
        
        # Extract feature vector
        features = {
            'wall_ratio': signal.wall_ratio,
            'gamma_exposure': signal.gamma_exposure,
            'tape_velocity': signal.tape_velocity,
            'barrier_replenishment_ratio': signal.barrier_replenishment_ratio or 0.0,
            'tape_imbalance': signal.tape_imbalance or 0.0,
            'fuel_net_dealer_gamma': signal.fuel_net_dealer_gamma or 0.0,
            'break_score_raw': signal.break_score_raw or 0.0,
        }
        
        # Verify all features extracted
        assert len(features) == 7
        assert all(isinstance(v, (int, float)) for v in features.values())
    
    def test_json_serialization(self):
        """Test that schema can be serialized to/from JSON."""
        signal = LevelSignalV1(
            event_id="test_006",
            ts_event_ns=time.time_ns(),
            level_price=687.0,
            level_kind=LevelKind.STRIKE,
            wall_ratio=2.0,
            gamma_exposure=0.0,
            tape_velocity=10.0,
            barrier_state=BarrierState.VACUUM,
            signal=Signal.BREAK,
        )
        
        # Serialize to dict
        signal_dict = signal.model_dump()
        assert signal_dict['level_kind'] == 'STRIKE'
        assert signal_dict['barrier_state'] == 'VACUUM'
        
        # Deserialize from dict
        signal_restored = LevelSignalV1(**signal_dict)
        assert signal_restored.level_kind == LevelKind.STRIKE
        assert signal_restored.barrier_state == BarrierState.VACUUM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
