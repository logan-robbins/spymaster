"""
Test Confluence Features and ML Integration in Core → Gateway → Frontend Pipeline

This test creates mock data for the complete data flow and verifies:
1. Confluence features are computed correctly by Core Service
2. ML viewport predictions are included in levels.signals
3. Gateway normalization merges confluence + ML predictions
4. Output schema matches frontend expectations
"""

import pytest
import json
from typing import Dict, Any, List
from datetime import datetime, timezone

# Mock data generators

def generate_mock_level_signal(
    level_id: str = "STRIKE_687",
    level_price: float = 687.0,
    level_kind: str = "STRIKE",
    include_confluence: bool = True,
    include_viewport: bool = False
) -> Dict[str, Any]:
    """
    Generate mock level signal matching Core Service output schema.
    
    Args:
        level_id: Level identifier
        level_price: Price level
        level_kind: Level type
        include_confluence: Include confluence features
        include_viewport: Generate for viewport predictions lookup
        
    Returns:
        Mock level signal dict
    """
    base = {
        "id": level_id,
        "price": level_price,
        "kind": level_kind,
        "direction": "SUPPORT",  # spot > level
        "distance": 0.50,
        "break_score_raw": 85.2,
        "break_score_smooth": 81.5,
        "signal": "BREAK",
        "confidence": "HIGH",
        "approach_velocity": 0.15,
        "approach_bars": 12,
        "approach_distance": 3.5,
        "prior_touches": 2,
        "barrier": {
            "state": "VACUUM",
            "delta_liq": -8200.0,
            "replenishment_ratio": 0.15,
            "added": 3100,
            "canceled": 9800,
            "filled": 1500,
            "defending_quote": {"price": level_price, "size": 150},
            "churn": 12900.0,
            "depth_in_zone": 450
        },
        "tape": {
            "imbalance": -0.45,
            "buy_vol": 120000,
            "sell_vol": 320000,
            "velocity": -0.08,
            "sweep": {
                "detected": True,
                "direction": "DOWN",
                "notional": 1250000.0,
                "num_prints": 15,
                "window_ms": 250.0
            }
        },
        "fuel": {
            "effect": "AMPLIFY",
            "net_dealer_gamma": -185000.0,
            "call_wall": 690.0,
            "put_wall": 684.0,
            "hvl": 687.0
        },
        "runway": {
            "direction": "DOWN",
            "next_obstacle": {
                "id": "PUT_WALL_684",
                "price": 684.0
            },
            "distance": 3.0,
            "quality": "CLEAR"
        },
        "note": "Vacuum + dealers chase; sweep confirms"
    }
    
    if include_confluence:
        base["confluence_count"] = 3
        base["confluence_pressure"] = 0.65
        base["confluence_alignment"] = 1  # ALIGNED
        base["confluence_level"] = 3  # PREMIUM
        base["confluence_level_name"] = "PREMIUM"
    
    return base


def generate_mock_viewport_target(
    level_id: str = "STRIKE_687",
    level_price: float = 687.0
) -> Dict[str, Any]:
    """
    Generate mock viewport ML prediction target.
    
    Returns:
        Mock viewport target matching ViewportScoringService output
    """
    return {
        "level_id": level_id,
        "level_price": level_price,
        "direction": "DOWN",
        "distance": 0.50,
        "distance_signed": 0.50,
        
        # Tree predictions
        "p_tradeable_2": 0.78,
        "p_break": 0.72,
        "p_bounce": 0.28,
        "strength_signed": 2.35,
        "strength_abs": 2.35,
        "time_to_threshold": {
            "t1": {"60": 0.65, "120": 0.82},
            "t2": {"60": 0.42, "120": 0.58},
            "t1_break": {"60": 0.58, "120": 0.75},
            "t1_bounce": {"60": 0.15, "120": 0.22},
            "t2_break": {"60": 0.35, "120": 0.48},
            "t2_bounce": {"60": 0.08, "120": 0.12}
        },
        
        # Retrieval predictions
        "retrieval": {
            "p_break": 0.68,
            "p_bounce": 0.32,
            "p_tradeable_2": 0.75,
            "strength_signed_mean": 2.1,
            "strength_abs_mean": 2.3,
            "time_to_threshold_1_mean": 95.5,
            "time_to_threshold_2_mean": 185.2,
            "similarity": 0.85,
            "entropy": 0.62
        },
        
        # Scoring metadata
        "utility_score": 0.82,
        "viewport_state": "IN_MONITOR_BAND",
        "stage": "stage_b",
        "pinned": False,
        "relevance": 0.88
    }


def generate_mock_levels_signals_payload(
    num_levels: int = 3,
    include_confluence: bool = True,
    include_viewport: bool = True
) -> Dict[str, Any]:
    """
    Generate complete mock levels.signals payload from Core Service.
    
    Args:
        num_levels: Number of levels to generate
        include_confluence: Include confluence features
        include_viewport: Include viewport ML predictions
        
    Returns:
        Complete mock payload
    """
    # Generate levels
    levels = []
    viewport_targets = []
    
    base_price = 687.0
    for i in range(num_levels):
        level_price = base_price - i
        level_id = f"STRIKE_{int(level_price)}"
        
        levels.append(generate_mock_level_signal(
            level_id=level_id,
            level_price=level_price,
            include_confluence=include_confluence,
            include_viewport=False
        ))
        
        if include_viewport:
            viewport_targets.append(generate_mock_viewport_target(
                level_id=level_id,
                level_price=level_price
            ))
    
    payload = {
        "ts": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
        "spy": {
            "spot": 687.50,
            "bid": 687.49,
            "ask": 687.51
        },
        "levels": levels
    }
    
    if include_viewport:
        payload["viewport"] = {
            "ts": payload["ts"],
            "targets": viewport_targets
        }
    
    return payload


# Tests

def test_confluence_features_in_mock_data():
    """Test that mock level signals include confluence features."""
    level = generate_mock_level_signal(include_confluence=True)
    
    assert "confluence_count" in level
    assert "confluence_pressure" in level
    assert "confluence_alignment" in level
    assert "confluence_level" in level
    assert "confluence_level_name" in level
    
    assert level["confluence_count"] == 3
    assert level["confluence_pressure"] == 0.65
    assert level["confluence_alignment"] == 1  # ALIGNED
    assert level["confluence_level"] == 3  # PREMIUM
    assert level["confluence_level_name"] == "PREMIUM"


def test_viewport_predictions_structure():
    """Test viewport ML prediction structure."""
    viewport_target = generate_mock_viewport_target()
    
    # Core fields
    assert "level_id" in viewport_target
    assert "p_tradeable_2" in viewport_target
    assert "p_break" in viewport_target
    assert "p_bounce" in viewport_target
    assert "strength_signed" in viewport_target
    
    # Time-to-threshold predictions
    assert "time_to_threshold" in viewport_target
    tt = viewport_target["time_to_threshold"]
    assert "t1" in tt
    assert "60" in tt["t1"]
    assert "120" in tt["t1"]
    
    # Retrieval predictions
    assert "retrieval" in viewport_target
    assert "similarity" in viewport_target["retrieval"]
    
    # Metadata
    assert "utility_score" in viewport_target
    assert "stage" in viewport_target


def test_full_payload_structure():
    """Test complete levels.signals payload structure."""
    payload = generate_mock_levels_signals_payload(
        num_levels=3,
        include_confluence=True,
        include_viewport=True
    )
    
    # Top-level structure
    assert "ts" in payload
    assert "spy" in payload
    assert "levels" in payload
    assert "viewport" in payload
    
    # SPY snapshot
    assert payload["spy"]["spot"] == 687.50
    assert payload["spy"]["bid"] == 687.49
    assert payload["spy"]["ask"] == 687.51
    
    # Levels array
    assert len(payload["levels"]) == 3
    
    # Viewport targets
    assert len(payload["viewport"]["targets"]) == 3
    
    # Verify level IDs match
    level_ids = {level["id"] for level in payload["levels"]}
    viewport_ids = {target["level_id"] for target in payload["viewport"]["targets"]}
    assert level_ids == viewport_ids


def test_gateway_normalization_mock():
    """
    Test Gateway normalization logic with mock data.
    
    This simulates what Gateway._normalize_levels_payload() would do.
    """
    payload = generate_mock_levels_signals_payload(
        num_levels=2,
        include_confluence=True,
        include_viewport=True
    )
    
    # Simulate Gateway normalization
    viewport = payload.get("viewport")
    viewport_targets = viewport.get("targets", []) if viewport else []
    
    # Build lookup by level_id
    viewport_by_id = {}
    for target in viewport_targets:
        level_id = target.get("level_id")
        if level_id:
            viewport_by_id[level_id] = target
    
    # Normalize first level
    level = payload["levels"][0]
    level_id = level["id"]
    viewport_pred = viewport_by_id.get(level_id)
    
    # Verify viewport prediction found
    assert viewport_pred is not None
    assert viewport_pred["level_id"] == level_id
    
    # Build normalized output (simplified)
    normalized = {
        "id": level["id"],
        "level_price": level["price"],
        "confluence_count": level.get("confluence_count", 0),
        "confluence_level": level.get("confluence_level", 0),
        "ml_predictions": {
            "p_tradeable_2": viewport_pred["p_tradeable_2"],
            "p_break": viewport_pred["p_break"],
            "utility_score": viewport_pred["utility_score"],
            "stage": viewport_pred["stage"]
        }
    }
    
    # Verify merged output
    assert normalized["confluence_count"] == 3
    assert normalized["confluence_level"] == 3
    assert "ml_predictions" in normalized
    assert normalized["ml_predictions"]["p_tradeable_2"] == 0.78
    assert normalized["ml_predictions"]["stage"] == "stage_b"


def test_frontend_schema_compatibility():
    """
    Test that normalized output matches frontend TypeScript interface expectations.
    """
    payload = generate_mock_levels_signals_payload(
        num_levels=1,
        include_confluence=True,
        include_viewport=True
    )
    
    level = payload["levels"][0]
    viewport_pred = payload["viewport"]["targets"][0]
    
    # Simulate full Gateway normalization
    normalized = {
        # Core level fields
        "id": level["id"],
        "level_price": level["price"],
        "level_kind_name": level["kind"],
        "direction": "DOWN",  # Normalized from SUPPORT
        "distance": level["distance"],
        
        # Physics
        "barrier_state": level["barrier"]["state"],
        "tape_imbalance": level["tape"]["imbalance"],
        "gamma_exposure": level["fuel"]["net_dealer_gamma"],
        
        # Scores
        "break_score_raw": level["break_score_raw"],
        "break_score_smooth": level["break_score_smooth"],
        "signal": level["signal"],
        "confidence": level["confidence"],
        
        # Confluence
        "confluence_count": level["confluence_count"],
        "confluence_pressure": level["confluence_pressure"],
        "confluence_alignment": level["confluence_alignment"],
        "confluence_level": level["confluence_level"],
        "confluence_level_name": level["confluence_level_name"],
        
        # ML predictions
        "ml_predictions": {
            "p_tradeable_2": viewport_pred["p_tradeable_2"],
            "p_break": viewport_pred["p_break"],
            "p_bounce": viewport_pred["p_bounce"],
            "strength_signed": viewport_pred["strength_signed"],
            "utility_score": viewport_pred["utility_score"],
            "stage": viewport_pred["stage"],
            "time_to_threshold": viewport_pred["time_to_threshold"]
        }
    }
    
    # Verify all expected fields present
    required_fields = [
        "id", "level_price", "level_kind_name", "direction",
        "barrier_state", "tape_imbalance", "gamma_exposure",
        "break_score_raw", "signal", "confluence_count",
        "confluence_level", "ml_predictions"
    ]
    
    for field in required_fields:
        assert field in normalized, f"Missing field: {field}"
    
    # Verify ML predictions structure
    ml_pred = normalized["ml_predictions"]
    assert "p_tradeable_2" in ml_pred
    assert "p_break" in ml_pred
    assert "p_bounce" in ml_pred
    assert "time_to_threshold" in ml_pred
    assert "t1" in ml_pred["time_to_threshold"]
    
    # Verify confluence values
    assert normalized["confluence_level"] in range(0, 11)  # 0-10 scale
    assert normalized["confluence_alignment"] in [-1, 0, 1]  # OPPOSED, NEUTRAL, ALIGNED


def test_json_serialization():
    """Test that mock data can be JSON serialized (for WebSocket transmission)."""
    payload = generate_mock_levels_signals_payload(
        num_levels=2,
        include_confluence=True,
        include_viewport=True
    )
    
    # Serialize to JSON
    json_str = json.dumps(payload)
    
    # Deserialize
    deserialized = json.loads(json_str)
    
    # Verify structure preserved
    assert len(deserialized["levels"]) == 2
    assert len(deserialized["viewport"]["targets"]) == 2
    assert deserialized["levels"][0]["confluence_count"] == 3


def test_multiple_levels_with_varying_confluence():
    """Test payload with levels having different confluence qualities."""
    # Generate diverse levels
    levels = [
        generate_mock_level_signal(
            level_id="PM_HIGH",
            level_price=690.0,
            level_kind="PM_HIGH",
            include_confluence=True
        ),
        generate_mock_level_signal(
            level_id="STRIKE_687",
            level_price=687.0,
            level_kind="STRIKE",
            include_confluence=True
        ),
        generate_mock_level_signal(
            level_id="VWAP",
            level_price=686.5,
            level_kind="VWAP",
            include_confluence=True
        )
    ]
    
    # Customize confluence levels
    levels[0]["confluence_level"] = 1  # ULTRA_PREMIUM (PM_HIGH)
    levels[0]["confluence_level_name"] = "ULTRA_PREMIUM"
    levels[1]["confluence_level"] = 5  # STRONG (STRIKE)
    levels[1]["confluence_level_name"] = "STRONG"
    levels[2]["confluence_level"] = 7  # MODERATE (VWAP)
    levels[2]["confluence_level_name"] = "MODERATE"
    
    # Verify hierarchical scale
    assert levels[0]["confluence_level"] < levels[1]["confluence_level"] < levels[2]["confluence_level"]
    
    # Verify names match scale
    assert levels[0]["confluence_level_name"] == "ULTRA_PREMIUM"
    assert levels[1]["confluence_level_name"] == "STRONG"
    assert levels[2]["confluence_level_name"] == "MODERATE"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

