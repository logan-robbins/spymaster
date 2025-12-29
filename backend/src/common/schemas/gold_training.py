"""
gold.training.es_pipeline.v1 schema - Gold tier ML training datasets.

Promoted from Silver features after curation and quality validation.
This is the authoritative schema for Gold training data used by ML models.

Purpose: Production ML training datasets for XGBoost + kNN models
Source: Curated from silver/features/es_pipeline/
Usage: Model training, hyperparameter optimization, evaluation
"""

from typing import ClassVar
import pyarrow as pa

from .base import SchemaVersion
from .silver_features import SilverFeaturesESPipelineV1


class GoldTrainingESPipelineV1:
    """
    Gold tier training dataset from ES pipeline.
    
    This is identical to Silver features schema but represents curated,
    validated, production-quality data promoted to Gold tier.
    
    Curation criteria:
    - Non-null coverage thresholds met
    - Front-month purity validated
    - Causality checks passed
    - RTH-only filtered (09:30-13:30 ET)
    - Deterministic event IDs
    """
    
    _schema_version: ClassVar[SchemaVersion] = SchemaVersion(
        name='training.es_pipeline',
        version=1,
        tier='gold'
    )


# Gold training schema is identical to Silver features
# (Gold tier represents quality/curation, not schema changes)
GoldTrainingESPipelineV1._arrow_schema = pa.schema(
    SilverFeaturesESPipelineV1._arrow_schema,
    metadata={
        'schema_name': 'training.es_pipeline.v1',
        'tier': 'gold',
        'pipeline': 'es_pipeline',
        'pipeline_version': '2.0.0',
        'description': 'Curated ML training dataset from ES futures + options',
        'total_columns': '182',
        'source': 'silver/features/es_pipeline/',
        'quality_gates': 'front_month_purity,causality,rth_only,non_null_coverage',
    }
)


def validate_gold_training(df) -> bool:
    """
    Validate that a DataFrame matches the Gold training schema.
    
    Args:
        df: DataFrame from Gold curator
        
    Returns:
        True if schema matches, raises ValueError otherwise
    """
    from .silver_features import validate_silver_features
    
    # Gold training uses same schema as Silver features
    return validate_silver_features(df)

