"""
Feature Manifest Schema for Silver Layer Versioning.

Defines the structure for tracking feature engineering experiments,
ensuring reproducibility and version management.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import yaml
import json


@dataclass
class FeatureGroup:
    """A logical group of related features."""
    name: str
    description: str
    columns: List[str]
    dependencies: List[str] = field(default_factory=list)  # Other groups this depends on


@dataclass
class SourceConfig:
    """Source data configuration."""
    layer: str  # 'bronze' or 'raw'
    schemas: List[str]  # e.g., ['futures/trades', 'options/trades']
    date_range: Optional[Dict[str, str]] = None  # {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}


@dataclass
class Parameters:
    """Feature engineering parameters."""
    # Physics windows
    W_b: int = 240  # Barrier window (seconds)
    W_t: int = 60   # Tape window (seconds)
    W_g: int = 60   # Fuel window (seconds)
    
    # Spatial bands
    MONITOR_BAND: float = 0.25  # Monitor band ($)
    TOUCH_BAND: float = 0.10    # Touch band ($)
    TAPE_BAND: float = 0.50     # Tape band ($)
    
    # Thresholds
    R_vac: float = 0.3   # VACUUM threshold
    R_wall: float = 1.5  # WALL threshold
    
    # Confirmation
    CONFIRMATION_WINDOW_SECONDS: int = 240  # Stage B confirmation
    
    # SMA warmup
    SMA_WARMUP_DAYS: int = 3
    
    # Labeling
    LOOKFORWARD_MINUTES: int = 8
    OUTCOME_THRESHOLD: float = 2.0  # $2 move
    
    # Custom parameters (for experiment-specific values)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        # Keep custom as-is
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ValidationMetrics:
    """Validation metrics for the feature set."""
    date_range: Dict[str, str]  # {'start': ..., 'end': ...}
    signal_count: int
    feature_count: int
    null_rates: Dict[str, float]  # column -> null rate
    schema_hash: Optional[str] = None  # MD5 of feature schema for validation


@dataclass
class FeatureManifest:
    """
    Complete manifest for a Silver feature set version.
    
    This is the single source of truth for reproducing a feature engineering experiment.
    """
    version: str  # Semantic version: vMAJOR.MINOR_name
    name: str
    description: str
    created_at: str  # ISO 8601 timestamp
    created_by: str = "pipeline"
    
    source: SourceConfig = field(default_factory=lambda: SourceConfig(layer="bronze", schemas=[]))
    feature_groups: List[FeatureGroup] = field(default_factory=list)
    parameters: Parameters = field(default_factory=Parameters)
    validation: Optional[ValidationMetrics] = None
    
    parent_version: Optional[str] = None  # For incremental changes
    tags: List[str] = field(default_factory=list)  # e.g., ['mechanics_only', 'baseline']
    notes: str = ""
    
    def to_yaml(self) -> str:
        """Serialize to YAML."""
        data = asdict(self)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'FeatureManifest':
        """Deserialize from YAML."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureManifest':
        """Create from dictionary."""
        # Convert nested dicts to dataclasses
        if 'source' in data and isinstance(data['source'], dict):
            data['source'] = SourceConfig(**data['source'])
        
        if 'feature_groups' in data:
            data['feature_groups'] = [
                FeatureGroup(**fg) if isinstance(fg, dict) else fg
                for fg in data['feature_groups']
            ]
        
        if 'parameters' in data and isinstance(data['parameters'], dict):
            data['parameters'] = Parameters(**data['parameters'])
        
        if 'validation' in data and isinstance(data['validation'], dict):
            data['validation'] = ValidationMetrics(**data['validation'])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> 'FeatureManifest':
        """Load from YAML file."""
        with open(path, 'r') as f:
            return cls.from_yaml(f.read())
    
    def to_file(self, path: Path):
        """Save to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_yaml())
    
    def get_output_path(self, base_dir: Path) -> Path:
        """Get output directory for this feature set."""
        return base_dir / "silver" / "features" / self.version


@dataclass
class ExperimentRecord:
    """Record of a feature engineering experiment."""
    id: str  # exp001, exp002, etc.
    version: str  # Reference to FeatureManifest version
    created_at: str
    status: str  # 'running', 'completed', 'failed', 'archived'
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None  # Parent experiment ID
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # ML model results (if trained)
    model_path: Optional[str] = None
    model_metrics: Optional[Dict[str, float]] = None  # AUC, precision, etc.


@dataclass
class ExperimentRegistry:
    """Registry of all feature experiments."""
    experiments: List[ExperimentRecord] = field(default_factory=list)
    
    def add(self, experiment: ExperimentRecord):
        """Add an experiment record."""
        self.experiments.append(experiment)
    
    def get_by_id(self, exp_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        for exp in self.experiments:
            if exp.id == exp_id:
                return exp
        return None
    
    def get_by_version(self, version: str) -> List[ExperimentRecord]:
        """Get all experiments for a feature version."""
        return [exp for exp in self.experiments if exp.version == version]
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {'experiments': [asdict(exp) for exp in self.experiments]},
            indent=2,
            default=str
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentRegistry':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        experiments = [ExperimentRecord(**exp) for exp in data.get('experiments', [])]
        return cls(experiments=experiments)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ExperimentRegistry':
        """Load from JSON file."""
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def to_file(self, path: Path):
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())


# Predefined feature groups for common patterns
FEATURE_GROUPS_MECHANICS_ONLY = [
    FeatureGroup(
        name="barrier_physics",
        description="Order book liquidity defense metrics",
        columns=[
            "barrier_state",
            "barrier_delta_liq",
            "barrier_delta_liq_log",
            "barrier_replenishment_ratio",
            "wall_ratio",
            "wall_ratio_log",
            "wall_ratio_nonzero",
        ]
    ),
    FeatureGroup(
        name="tape_physics",
        description="Trade flow momentum metrics",
        columns=[
            "tape_imbalance",
            "tape_velocity",
            "sweep_detected",
        ]
    ),
    FeatureGroup(
        name="fuel_physics",
        description="Dealer gamma exposure and hedging pressure",
        columns=[
            "gamma_exposure",
            "fuel_effect",
            "dealer_pressure",
        ]
    ),
    FeatureGroup(
        name="dealer_velocity",
        description="Dealer flow dynamics",
        columns=[
            "gamma_flow_velocity",
            "gamma_flow_impulse",
            "gamma_flow_accel_1m",
            "gamma_flow_accel_5m",
            "dealer_pressure_accel",
        ]
    ),
    FeatureGroup(
        name="pressure_indicators",
        description="Composite pressure metrics",
        columns=[
            "liquidity_pressure",
            "tape_pressure",
            "gamma_pressure",
            "gamma_pressure_accel",
            "net_break_pressure",
        ]
    ),
]

FEATURE_GROUPS_TA = [
    FeatureGroup(
        name="approach_context",
        description="Price action and approach dynamics",
        columns=[
            "approach_velocity",
            "approach_distance",
            "approach_distance_atr",
            "distance_signed",
            "distance_signed_atr",
            "distance_pct",
        ]
    ),
    FeatureGroup(
        name="sma_context",
        description="Moving average positioning",
        columns=[
            "sma_200",
            "sma_400",
            "sma_200_slope",
            "sma_400_slope",
            "dist_to_sma_200",
            "dist_to_sma_400",
        ]
    ),
    FeatureGroup(
        name="confluence",
        description="Level stacking and confluence metrics",
        columns=[
            "confluence_count",
            "confluence_pressure",
            "confluence_alignment",
            "confluence_level",
        ]
    ),
]


def create_mechanics_only_manifest(version: str = "v1.0_mechanics_only") -> FeatureManifest:
    """Create a mechanics-only baseline manifest."""
    return FeatureManifest(
        version=version,
        name="mechanics_only",
        description="Pure physics-based features (barrier, tape, fuel) without TA",
        created_at=datetime.utcnow().isoformat() + "Z",
        source=SourceConfig(
            layer="bronze",
            schemas=["futures/trades", "futures/mbp10", "options/trades"]
        ),
        feature_groups=FEATURE_GROUPS_MECHANICS_ONLY,
        parameters=Parameters(),
        tags=["mechanics", "baseline", "physics_only"],
        notes="Baseline model using only market mechanics (no TA features)"
    )


def create_full_ensemble_manifest(version: str = "v2.0_full_ensemble") -> FeatureManifest:
    """Create a full ensemble manifest with all features."""
    return FeatureManifest(
        version=version,
        name="full_ensemble",
        description="Complete feature set including mechanics and TA",
        created_at=datetime.utcnow().isoformat() + "Z",
        source=SourceConfig(
            layer="bronze",
            schemas=["futures/trades", "futures/mbp10", "options/trades"]
        ),
        feature_groups=FEATURE_GROUPS_MECHANICS_ONLY + FEATURE_GROUPS_TA,
        parameters=Parameters(),
        parent_version="v1.0_mechanics_only",
        tags=["full", "ensemble", "mechanics", "ta"],
        notes="Full feature set combining mechanics and technical analysis"
    )
