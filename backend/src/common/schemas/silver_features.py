"""
silver.features.es_pipeline.v1 schema - Silver tier ML features.

Generated from ES futures + ES options multi-window physics pipeline.
This is the authoritative schema for pipeline output validation.

Architecture: ES futures (spot + liquidity) + ES 0DTE options (gamma)
Inference: Event-driven (zone entry + adaptive cadence)
Features: 182 columns (10 identity + 108 engineered features + 64 labels)
Levels: 6 kinds (PM/OR high/low + SMA_200/400)
Outcome: Triple-barrier with volatility-scaled barrier, multi-timeframe (2min/4min/8min)
RTH: 09:30-13:30 ET (first 4 hours)
"""

from typing import ClassVar, List
import pyarrow as pa

from .base import SchemaVersion, SchemaRegistry, build_arrow_schema


IDENTITY_COLUMNS: List[str] = [
    "event_id",
    "ts_ns",
    "timestamp",
    "level_price",
    "level_kind",
    "level_kind_name",
    "direction",
    "entry_price",
    "zone_width",
    "date",
]

LABEL_BASE_COLUMNS: List[str] = [
    "outcome",
    "excursion_max",
    "excursion_min",
    "strength_signed",
    "strength_abs",
    "time_to_threshold_1",
    "time_to_threshold_2",
    "time_to_break_1",
    "time_to_break_2",
    "time_to_bounce_1",
    "time_to_bounce_2",
    "tradeable_1",
    "tradeable_2",
    "confirm_ts_ns",
    "anchor_spot",
    "future_price",
]

LABEL_SUFFIXES: List[str] = ["2min", "4min", "8min"]
LABEL_COLUMNS: List[str] = [
    f"{base}_{suffix}" for suffix in LABEL_SUFFIXES for base in LABEL_BASE_COLUMNS
] + LABEL_BASE_COLUMNS


class SilverFeaturesESPipelineV1:
    """
    Silver tier features from ES pipeline.
    
    This schema represents the output of the 16-stage ES pipeline:
    - Identity fields (event_id, timestamps, level info) = 10 columns
    - Physics features (barrier/tape/fuel) = 11 columns
    - Multi-window kinematics = 19 columns
    - Multi-window OFI = 9 columns
    - Barrier evolution = 7 columns
    - GEX features = 15 columns
    - Level distances + stacking = 16 columns
    - Force/mass validation = 3 columns
    - Approach context = 5 columns
    - Session timing = 2 columns
    - Sparse transforms = 4 columns
    - Normalized features = 11 columns
    - Attempt clustering = 6 columns
    - Multi-timeframe labels = 3 × 16 = 48 columns (2/4/8min)
    - Primary labels = 16 columns
    """
    
    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='features.es_pipeline',
        version=1,
        tier='silver'
    )


# Arrow schema definition (182 columns)
SilverFeaturesESPipelineV1._arrow_schema = pa.schema([
    # ===== IDENTITY (10 columns) =====
    ('event_id', pa.utf8(), False),
    ('ts_ns', pa.int64(), False),
    ('timestamp', pa.timestamp('ns'), False),
    ('level_price', pa.float64(), False),
    ('level_kind', pa.int8(), False),
    ('level_kind_name', pa.utf8(), False),
    ('direction', pa.utf8(), False),
    ('entry_price', pa.float64(), False),
    ('zone_width', pa.float64(), False),
    ('date', pa.utf8(), False),
    
    # ===== BARRIER PHYSICS (4 columns) =====
    ('barrier_state', pa.utf8(), False),
    ('barrier_delta_liq', pa.float64(), False),
    ('barrier_replenishment_ratio', pa.float64(), False),
    ('wall_ratio', pa.float64(), False),
    
    # ===== TAPE PHYSICS (5 columns) =====
    ('tape_imbalance', pa.float64(), False),
    ('tape_buy_vol', pa.int64(), False),
    ('tape_sell_vol', pa.int64(), False),
    ('tape_velocity', pa.float64(), False),
    ('sweep_detected', pa.bool_(), False),
    
    # ===== FUEL PHYSICS (2 columns) =====
    ('fuel_effect', pa.utf8(), False),
    ('gamma_exposure', pa.float64(), False),
    
    # ===== MULTI-WINDOW KINEMATICS (19 columns: 5 windows × 3 metrics + momentum_trend for 3/5/10/20min) =====
    ('velocity_1min', pa.float64(), False),
    ('acceleration_1min', pa.float64(), False),
    ('jerk_1min', pa.float64(), False),
    
    ('velocity_3min', pa.float64(), False),
    ('acceleration_3min', pa.float64(), False),
    ('jerk_3min', pa.float64(), False),
    ('momentum_trend_3min', pa.float64(), False),
    
    ('velocity_5min', pa.float64(), False),
    ('acceleration_5min', pa.float64(), False),
    ('jerk_5min', pa.float64(), False),
    ('momentum_trend_5min', pa.float64(), False),
    
    ('velocity_10min', pa.float64(), False),
    ('acceleration_10min', pa.float64(), False),
    ('jerk_10min', pa.float64(), False),
    ('momentum_trend_10min', pa.float64(), False),
    
    ('velocity_20min', pa.float64(), False),
    ('acceleration_20min', pa.float64(), False),
    ('jerk_20min', pa.float64(), False),
    ('momentum_trend_20min', pa.float64(), False),
    
    # ===== MULTI-WINDOW OFI (9 columns: 4 windows + derivatives) =====
    ('ofi_30s', pa.float64(), False),
    ('ofi_near_level_30s', pa.float64(), False),
    ('ofi_60s', pa.float64(), False),
    ('ofi_near_level_60s', pa.float64(), False),
    ('ofi_120s', pa.float64(), False),
    ('ofi_near_level_120s', pa.float64(), False),
    ('ofi_300s', pa.float64(), False),
    ('ofi_near_level_300s', pa.float64(), False),
    ('ofi_acceleration', pa.float64(), False),
    
    # ===== BARRIER EVOLUTION (7 columns: 3 windows + current) =====
    ('barrier_delta_1min', pa.float64(), False),
    ('barrier_pct_change_1min', pa.float64(), False),
    ('barrier_delta_3min', pa.float64(), False),
    ('barrier_pct_change_3min', pa.float64(), False),
    ('barrier_delta_5min', pa.float64(), False),
    ('barrier_pct_change_5min', pa.float64(), False),
    ('barrier_depth_current', pa.float64(), False),
    
    # ===== LEVEL DISTANCES (16 columns: 6 levels × 2 + tested level + stacking) =====
    ('dist_to_pm_high', pa.float64(), False),
    ('dist_to_pm_high_atr', pa.float64(), False),
    ('dist_to_pm_low', pa.float64(), False),
    ('dist_to_pm_low_atr', pa.float64(), False),
    ('dist_to_or_high', pa.float64(), False),
    ('dist_to_or_high_atr', pa.float64(), False),
    ('dist_to_or_low', pa.float64(), False),
    ('dist_to_or_low_atr', pa.float64(), False),
    ('dist_to_sma_200', pa.float64(), False),
    ('dist_to_sma_200_atr', pa.float64(), False),
    ('dist_to_sma_400', pa.float64(), False),
    ('dist_to_sma_400_atr', pa.float64(), False),
    ('dist_to_tested_level', pa.float64(), False),
    ('level_stacking_2pt', pa.int8(), False),
    ('level_stacking_5pt', pa.int8(), False),
    ('level_stacking_10pt', pa.int8(), False),
    
    # ===== GEX FEATURES (15 columns: strike bands) =====
    ('gex_above_1strike', pa.float64(), False),
    ('gex_below_1strike', pa.float64(), False),
    ('call_gex_above_1strike', pa.float64(), False),
    ('put_gex_below_1strike', pa.float64(), False),
    ('gex_above_2strike', pa.float64(), False),
    ('gex_below_2strike', pa.float64(), False),
    ('call_gex_above_2strike', pa.float64(), False),
    ('put_gex_below_2strike', pa.float64(), False),
    ('gex_above_3strike', pa.float64(), False),
    ('gex_below_3strike', pa.float64(), False),
    ('call_gex_above_3strike', pa.float64(), False),
    ('put_gex_below_3strike', pa.float64(), False),
    ('gex_asymmetry', pa.float64(), False),
    ('gex_ratio', pa.float64(), False),
    ('net_gex_2strike', pa.float64(), False),
    
    # ===== FORCE/MASS VALIDATION (3 columns) =====
    ('predicted_accel', pa.float64(), False),
    ('accel_residual', pa.float64(), False),
    ('force_mass_ratio', pa.float64(), False),
    
    # ===== APPROACH CONTEXT (5 columns) =====
    ('atr', pa.float64(), False),
    ('approach_velocity', pa.float64(), False),
    ('approach_bars', pa.int32(), False),
    ('approach_distance', pa.float64(), False),
    ('prior_touches', pa.int32(), False),
    
    # ===== SESSION TIMING (2 columns) =====
    ('minutes_since_open', pa.float64(), False),
    ('bars_since_open', pa.int32(), False),
    
    # ===== SPARSE TRANSFORMS (4 columns) =====
    ('wall_ratio_nonzero', pa.int8(), False),
    ('wall_ratio_log', pa.float64(), False),
    ('barrier_delta_liq_nonzero', pa.int8(), False),
    ('barrier_delta_liq_log', pa.float64(), False),
    
    # ===== NORMALIZED FEATURES (11 columns) =====
    ('spot', pa.float64(), False),
    ('distance_signed', pa.float64(), False),
    ('distance_signed_atr', pa.float64(), False),
    ('distance_signed_pct', pa.float64(), False),
    ('dist_to_pm_high_pct', pa.float64(), False),
    ('dist_to_pm_low_pct', pa.float64(), False),
    ('dist_to_sma_200_pct', pa.float64(), False),
    ('dist_to_sma_400_pct', pa.float64(), False),
    ('approach_distance_atr', pa.float64(), False),
    ('approach_distance_pct', pa.float64(), False),
    ('level_price_pct', pa.float64(), False),
    
    # ===== ATTEMPT CLUSTERING (6 columns) =====
    ('attempt_index', pa.int64(), False),
    ('attempt_cluster_id', pa.int64(), False),
    ('barrier_replenishment_trend', pa.float64(), False),
    ('barrier_delta_liq_trend', pa.float64(), False),
    ('tape_velocity_trend', pa.float64(), False),
    ('tape_imbalance_trend', pa.float64(), False),
    
    # ===== LABELS: 2-MINUTE CONFIRMATION (16 columns) =====
    ('outcome_2min', pa.utf8(), False),
    ('excursion_max_2min', pa.float64(), False),
    ('excursion_min_2min', pa.float64(), False),
    ('strength_signed_2min', pa.float64(), False),
    ('strength_abs_2min', pa.float64(), False),
    ('time_to_threshold_1_2min', pa.float64(), True),  # Nullable (may not reach)
    ('time_to_threshold_2_2min', pa.float64(), True),
    ('time_to_break_1_2min', pa.float64(), True),
    ('time_to_break_2_2min', pa.float64(), True),
    ('time_to_bounce_1_2min', pa.float64(), True),
    ('time_to_bounce_2_2min', pa.float64(), True),
    ('tradeable_1_2min', pa.int8(), False),
    ('tradeable_2_2min', pa.int8(), False),
    ('confirm_ts_ns_2min', pa.int64(), True),
    ('anchor_spot_2min', pa.float64(), False),
    ('future_price_2min', pa.float64(), False),
    
    # ===== LABELS: 4-MINUTE CONFIRMATION (16 columns) =====
    ('outcome_4min', pa.utf8(), False),
    ('excursion_max_4min', pa.float64(), False),
    ('excursion_min_4min', pa.float64(), False),
    ('strength_signed_4min', pa.float64(), False),
    ('strength_abs_4min', pa.float64(), False),
    ('time_to_threshold_1_4min', pa.float64(), True),
    ('time_to_threshold_2_4min', pa.float64(), True),
    ('time_to_break_1_4min', pa.float64(), True),
    ('time_to_break_2_4min', pa.float64(), True),
    ('time_to_bounce_1_4min', pa.float64(), True),
    ('time_to_bounce_2_4min', pa.float64(), True),
    ('tradeable_1_4min', pa.int8(), False),
    ('tradeable_2_4min', pa.int8(), False),
    ('confirm_ts_ns_4min', pa.int64(), True),
    ('anchor_spot_4min', pa.float64(), False),
    ('future_price_4min', pa.float64(), False),
    
    # ===== LABELS: 8-MINUTE CONFIRMATION (16 columns) =====
    ('outcome_8min', pa.utf8(), False),
    ('excursion_max_8min', pa.float64(), False),
    ('excursion_min_8min', pa.float64(), False),
    ('strength_signed_8min', pa.float64(), False),
    ('strength_abs_8min', pa.float64(), False),
    ('time_to_threshold_1_8min', pa.float64(), True),
    ('time_to_threshold_2_8min', pa.float64(), True),
    ('time_to_break_1_8min', pa.float64(), True),
    ('time_to_break_2_8min', pa.float64(), True),
    ('time_to_bounce_1_8min', pa.float64(), True),
    ('time_to_bounce_2_8min', pa.float64(), True),
    ('tradeable_1_8min', pa.int8(), False),
    ('tradeable_2_8min', pa.int8(), False),
    ('confirm_ts_ns_8min', pa.int64(), True),
    ('anchor_spot_8min', pa.float64(), False),
    ('future_price_8min', pa.float64(), False),
    
    # ===== LABELS: PRIMARY (NO SUFFIX) (16 columns) =====
    # These are aliases/copies of the primary timeframe labels
    ('outcome', pa.utf8(), False),
    ('excursion_max', pa.float64(), False),
    ('excursion_min', pa.float64(), False),
    ('strength_signed', pa.float64(), False),
    ('strength_abs', pa.float64(), False),
    ('time_to_threshold_1', pa.float64(), True),
    ('time_to_threshold_2', pa.float64(), True),
    ('time_to_break_1', pa.float64(), True),
    ('time_to_break_2', pa.float64(), True),
    ('time_to_bounce_1', pa.float64(), True),
    ('time_to_bounce_2', pa.float64(), True),
    ('tradeable_1', pa.int8(), False),
    ('tradeable_2', pa.int8(), False),
    ('confirm_ts_ns', pa.int64(), True),
    ('anchor_spot', pa.float64(), False),
    ('future_price', pa.float64(), False),
], metadata={
    'schema_name': 'features.es_pipeline.v1',
    'tier': 'silver',
    'pipeline': 'es_pipeline',
    'pipeline_version': '2.0.0',
    'description': 'ES futures + ES options multi-window physics features',
    'total_columns': '182',
    'generated_from': '16-stage ES pipeline',
    'inference_model': 'event-driven (zone entry + adaptive cadence)',
})

FEATURE_COLUMNS: List[str] = [
    name
    for name in SilverFeaturesESPipelineV1._arrow_schema.names
    if name not in IDENTITY_COLUMNS and name not in LABEL_COLUMNS
]


# Convenience function to validate DataFrame against schema
def validate_silver_features(df) -> bool:
    """
    Validate that a DataFrame matches the Silver features schema.
    
    Args:
        df: DataFrame from ES pipeline
        
    Returns:
        True if schema matches, raises ValueError otherwise
    """
    import pandas as pd
    import pyarrow as pa
    
    schema = SilverFeaturesESPipelineV1._arrow_schema
    expected_cols = set(schema.names)
    actual_cols = set(df.columns)
    
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    
    if missing or extra:
        errors = []
        if missing:
            errors.append(f"Missing columns ({len(missing)}): {sorted(list(missing))[:10]}")
        if extra:
            errors.append(f"Extra columns ({len(extra)}): {sorted(list(extra))[:10]}")
        raise ValueError(f"Schema mismatch: {'; '.join(errors)}")
    
    return True


# Note: SchemaRegistry.register decorator not applicable for non-Pydantic schemas
# This schema is registered via __init__.py import
