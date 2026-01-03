"""
silver.features.es_pipeline.v1 schema - Silver tier ML features (Single-Level).

Generated from ES futures + ES options multi-window physics pipeline.
This schema represents OUTPUT from a SINGLE-LEVEL pipeline run.
Cross-level features (dist_to_pm_high, etc.) are computed post-merge.

Architecture: ES futures (spot + liquidity) + ES 0DTE options (gamma)
Inference: Event-driven (zone entry)
Levels: Pipeline runs for ONE level at a time (PM_HIGH, PM_LOW, OR_HIGH, etc.)
Outcome: First-crossing with fixed 12.5pt threshold, multi-timeframe (2min/4min/8min)
RTH: 09:30-12:30 ET (first 3 hours)
"""

from typing import ClassVar, List, Set
import pyarrow as pa

from .base import SchemaVersion


# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

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

# Labels that should NEVER be in feature vectors
LABEL_COLUMNS: Set[str] = {
    # Primary labels (8min)
    "outcome", "excursion_favorable", "excursion_adverse",
    "excursion_max", "excursion_min", "strength_signed", "strength_abs",
    "time_to_break_1", "time_to_bounce_1",
    # 2min labels
    "outcome_2min", "time_to_break_1_2min", "time_to_bounce_1_2min",
    # 4min labels
    "outcome_4min", "time_to_break_1_4min", "time_to_bounce_1_4min",
    # 8min labels
    "outcome_8min", "time_to_break_1_8min", "time_to_bounce_1_8min",
}


class SilverFeaturesESPipelineV1:
    """
    Silver tier features from single-level ES pipeline.
    
    This schema represents the output of a level-specific pipeline run.
    When multiple levels are merged, additional cross-level features are added.
    
    Feature groups:
    - Identity fields (10 columns)
    - Barrier physics (4 columns)
    - Tape physics (5 columns)
    - Fuel physics (2 columns)
    - Multi-window kinematics (19 columns)
    - Multi-window OFI (13 columns)
    - Microstructure (4 columns)
    - Barrier evolution (7 columns)
    - GEX features (21 columns)
    - Level distances (5 columns) - single level only
    - Force/mass validation (6 columns)
    - Approach context (3 columns)
    - Session timing (3 columns)
    - Sparse transforms (4 columns)
    - Normalized features (7 columns)
    - Attempt clustering (6 columns)
    - Labels (18 columns)
    
    Total: ~127 columns (varies slightly based on pipeline version)
    """
    
    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='features.es_pipeline',
        version=1,
        tier='silver'
    )


# Arrow schema definition for single-level pipeline output
SilverFeaturesESPipelineV1._arrow_schema = pa.schema([
    # ===== IDENTITY (10 columns) =====
    ('event_id', pa.utf8(), False),
    ('ts_ns', pa.int64(), False),
    ('timestamp', pa.timestamp('ns', tz='UTC'), False),
    ('level_price', pa.float64(), False),
    ('level_kind', pa.int8(), False),
    ('level_kind_name', pa.utf8(), False),
    ('direction', pa.utf8(), False),
    ('entry_price', pa.float64(), False),
    ('zone_width', pa.float64(), False),
    ('date', pa.utf8(), False),
    
    # ===== BARRIER PHYSICS (4 columns) =====
    ('barrier_state', pa.int8(), False),  # Encoded: 0=NONE, 1=WEAK, 2=MODERATE, 3=STRONG
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
    ('fuel_effect', pa.int8(), False),  # Encoded: -1=HEADWIND, 0=NEUTRAL, 1=TAILWIND
    ('gamma_exposure', pa.float64(), False),
    
    # ===== MULTI-WINDOW KINEMATICS (19 columns) =====
    # Direction-signed: positive = approaching level
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
    
    # ===== MULTI-WINDOW OFI (13 columns) =====
    # True event-based OFI using action/side from MBP-10
    ('ofi_30s', pa.float64(), False),
    ('ofi_near_level_30s', pa.float64(), False),
    ('ofi_above_5pt_30s', pa.float64(), False),
    ('ofi_below_5pt_30s', pa.float64(), False),
    ('ofi_60s', pa.float64(), False),
    ('ofi_near_level_60s', pa.float64(), False),
    ('ofi_120s', pa.float64(), False),
    ('ofi_near_level_120s', pa.float64(), False),
    ('ofi_300s', pa.float64(), False),
    ('ofi_near_level_300s', pa.float64(), False),
    ('ofi_above_5pt_300s', pa.float64(), False),
    ('ofi_below_5pt_300s', pa.float64(), False),
    ('ofi_acceleration', pa.float64(), False),
    
    # ===== MICROSTRUCTURE (4 columns) =====
    # Level-relative: computed within ±2.5pt of level_price
    ('vacuum_duration_bid', pa.float64(), True),  # Nullable - may not have vacuum
    ('vacuum_duration_ask', pa.float64(), True),
    ('replenishment_latency_bid', pa.float64(), True),
    ('replenishment_latency_ask', pa.float64(), True),
    
    # ===== BARRIER EVOLUTION (7 columns) =====
    # How depth changes at level over time
    ('barrier_delta_1min', pa.float64(), False),
    ('barrier_pct_change_1min', pa.float64(), False),
    ('barrier_delta_3min', pa.float64(), False),
    ('barrier_pct_change_3min', pa.float64(), False),
    ('barrier_delta_5min', pa.float64(), False),
    ('barrier_pct_change_5min', pa.float64(), False),
    ('barrier_depth_current', pa.float64(), False),
    
    # ===== GEX FEATURES (21 columns) =====
    # Level-relative gamma exposure
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
    ('gex_above_level', pa.float64(), False),
    ('gex_below_level', pa.float64(), False),
    ('call_gex_above_level', pa.float64(), False),
    ('call_gex_below_level', pa.float64(), False),
    ('put_gex_above_level', pa.float64(), False),
    ('put_gex_below_level', pa.float64(), False),
    ('gex_asymmetry', pa.float64(), False),
    ('gex_ratio', pa.float64(), False),
    ('net_gex_2strike', pa.float64(), False),
    
    # ===== LEVEL DISTANCES - SINGLE LEVEL (5 columns) =====
    # Only distance to THE level being tested
    # Cross-level distances (dist_to_pm_high, etc.) added post-merge
    ('dist_to_level', pa.float64(), False),
    ('dist_to_level_atr', pa.float64(), False),
    ('level_stacking_2pt', pa.int8(), False),  # Set to 0 for single-level
    ('level_stacking_5pt', pa.int8(), False),  # Set to 0 for single-level
    ('level_stacking_10pt', pa.int8(), False), # Set to 0 for single-level
    
    # ===== FORCE/MASS VALIDATION (6 columns) =====
    # F=ma consistency checks
    ('force_proxy', pa.float64(), False),
    ('mass_proxy', pa.float64(), False),
    ('predicted_accel', pa.float64(), False),
    ('accel_residual', pa.float64(), False),
    ('force_mass_ratio', pa.float64(), False),
    ('flow_alignment', pa.float64(), False),
    
    # ===== APPROACH CONTEXT (3 columns) =====
    ('approach_velocity', pa.float64(), False),  # pt/min toward level
    ('approach_bars', pa.int32(), False),        # consecutive bars toward level
    ('approach_distance', pa.float64(), False),  # total distance in lookback
    
    # ===== SESSION TIMING (3 columns) =====
    ('minutes_since_open', pa.float64(), False),  # relative to 09:30 ET
    ('bars_since_open', pa.int32(), False),
    ('or_active', pa.int32(), False),  # 1 if >= 15 min since open
    
    # ===== SPARSE TRANSFORMS (4 columns) =====
    ('wall_ratio_nonzero', pa.int8(), False),
    ('wall_ratio_log', pa.float64(), False),
    ('barrier_delta_liq_nonzero', pa.int8(), False),
    ('barrier_delta_liq_log', pa.float64(), False),
    
    # ===== NORMALIZED FEATURES (7 columns) =====
    ('atr', pa.float64(), False),
    ('spot', pa.float64(), False),
    ('distance_signed', pa.float64(), False),
    ('distance_signed_atr', pa.float64(), False),
    ('distance_signed_pct', pa.float64(), False),
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
    
    # ===== LABELS: MULTI-TIMEFRAME (18 columns) =====
    # CRITICAL: These are LABELS, not features. Never use in feature vectors.
    
    # 2min horizon (scalp)
    ('outcome_2min', pa.utf8(), False),
    ('time_to_break_1_2min', pa.float64(), True),
    ('time_to_bounce_1_2min', pa.float64(), True),
    
    # 4min horizon (day trade)
    ('outcome_4min', pa.utf8(), False),
    ('time_to_break_1_4min', pa.float64(), True),
    ('time_to_bounce_1_4min', pa.float64(), True),
    
    # 8min horizon (swing) - PRIMARY
    ('outcome_8min', pa.utf8(), False),
    ('time_to_break_1_8min', pa.float64(), True),
    ('time_to_bounce_1_8min', pa.float64(), True),
    
    # Primary labels (aliases for 8min)
    ('outcome', pa.utf8(), False),
    ('time_to_break_1', pa.float64(), True),
    ('time_to_bounce_1', pa.float64(), True),
    
    # Continuous outcomes (from 8min window)
    ('excursion_favorable', pa.float64(), False),
    ('excursion_adverse', pa.float64(), False),
    ('excursion_max', pa.float64(), False),
    ('excursion_min', pa.float64(), False),
    ('strength_signed', pa.float64(), False),
    ('strength_abs', pa.float64(), False),
    
], metadata={
    'schema_name': 'features.es_pipeline.v1',
    'tier': 'silver',
    'pipeline': 'es_pipeline',
    'pipeline_version': '4.7.0',
    'description': 'Single-level ES pipeline output. Cross-level features added post-merge.',
    'inference_model': 'event-driven (zone entry)',
})


# Feature columns (excludes identity and labels)
FEATURE_COLUMNS: List[str] = [
    name
    for name in SilverFeaturesESPipelineV1._arrow_schema.names
    if name not in IDENTITY_COLUMNS and name not in LABEL_COLUMNS
]


def validate_silver_features(df, strict: bool = False) -> bool:
    """
    Validate that a DataFrame matches the Silver features schema.
    
    Args:
        df: DataFrame from ES pipeline
        strict: If True, raise on mismatch. If False, log warning only.
        
    Returns:
        True if schema matches (or non-strict mode)
        
    Raises:
        ValueError: If strict=True and schema doesn't match
    """
    import logging
    logger = logging.getLogger(__name__)
    
    schema = SilverFeaturesESPipelineV1._arrow_schema
    expected_cols = set(schema.names)
    actual_cols = set(df.columns)
    
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    
    if missing or extra:
        errors = []
        if missing:
            errors.append(f"Missing ({len(missing)}): {sorted(list(missing))[:10]}...")
        if extra:
            errors.append(f"Extra ({len(extra)}): {sorted(list(extra))[:10]}...")
        
        msg = f"Schema mismatch: {'; '.join(errors)}"
        
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(f"  ⚠️  {msg}")
            return False
    
    return True


def get_feature_columns() -> List[str]:
    """Return list of feature column names (excludes identity and labels)."""
    return FEATURE_COLUMNS.copy()


def get_label_columns() -> Set[str]:
    """Return set of label column names (must not be used as features)."""
    return LABEL_COLUMNS.copy()
