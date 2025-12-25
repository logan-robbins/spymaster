"""
Configuration: single source of truth for all parameters.

Per §9 of PLAN.md, this centralizes:
- Window sizes for barrier/tape/fuel/velocity calculations
- Monitoring bands (how close to a level before we compute signals)
- Thresholds for state classification (vacuum, wall, sweep, etc.)
- Composite score weights
- Smoothing parameters (EWMA half-lives)

All values are tunable mechanical constants (no trained calibration in v1).
"""

import os
from dataclasses import dataclass, field


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_DEFAULT_DATA_ROOT = os.path.join(_BASE_DIR, "data", "lake")


@dataclass
class Config:
    """
    Global configuration for the Break/Reject Physics Engine.
    
    Usage:
        config = Config()
        barrier_window = config.W_b
    """
    
    # ========== Window sizes (seconds) ==========
    # NOTE: Touch timestamps are at the START of minute bars, so these windows
    # look FORWARD in time to analyze trades/quotes that happened during the bar.
    W_b: float = 240.0  # Barrier engine: confirmation window (aligned with Stage B t1)
    W_t: float = 60.0  # Tape engine: fast imbalance window near level
    W_g: float = 60.0  # Fuel engine: option flow window for net dealer gamma
    W_v: float = 3.0   # Velocity: window for slope calculation
    W_wall: float = 300.0  # Call/Put wall lookback window (5 minutes)
    DEALER_FLOW_WINDOW_MINUTES: int = 5
    DEALER_FLOW_BASELINE_MINUTES: int = 20
    DEALER_FLOW_ACCEL_SHORT_MINUTES: int = 1
    DEALER_FLOW_ACCEL_LONG_MINUTES: int = 3
    
    # ========== Monitoring bands (SPY dollars) ==========
    # Critical zone where bounce/break decision happens
    MONITOR_BAND: float = 0.25  # compute full signals if |spot - L| <= $0.25
    TOUCH_BAND: float = 0.10    # tight band for "touching level"
    CONFLUENCE_BAND: float = 0.20  # band for nearby key level confluence
    
    # Barrier engine: zone around strike-aligned level
    # SPY strikes at $1 intervals → ES at $10 intervals (40 ticks between strikes)
    # ±8 ticks = ±$2 ES = ±$0.20 SPY - captures order book as price approaches strike
    BARRIER_ZONE_ES_TICKS: int = 8  # ±8 ES ticks (±$2.00 ES = ±$0.20 SPY)
    
    # ========== Barrier thresholds (ES contracts) ==========
    R_vac: float = 0.3   # Replenishment ratio threshold for VACUUM
    R_wall: float = 1.5  # Replenishment ratio threshold for WALL/ABSORPTION
    F_thresh: int = 100  # Delta liquidity threshold (ES contracts, not shares)
    
    # Optional: percentile for WEAK state
    WEAK_PERCENTILE: float = 0.20  # bottom 20th percentile of defending size
    WEAK_LOOKBACK: float = 1800.0  # 30 minutes
    
    # ========== Tape thresholds (SPY scale - converted to ES internally) ==========
    TAPE_BAND: float = 0.50  # price band around level for tape imbalance (SPY dollars)
    SWEEP_MIN_NOTIONAL: float = 500_000.0  # minimum notional for sweep detection (ES = $50/pt)
    SWEEP_MAX_GAP_MS: int = 100  # max gap between prints in a sweep cluster
    SWEEP_MIN_VENUES: int = 1    # ES only trades on CME so set to 1
    
    # ========== Fuel thresholds ==========
    FUEL_STRIKE_RANGE: float = 2.0  # consider strikes within ±N dollars of level
    DEALER_FLOW_STRIKE_RANGE: float = 2.0  # strike range for dealer flow velocity

    # ========== Mean reversion (SMA) settings ==========
    SMA_SLOPE_WINDOW_MINUTES: int = 20
    SMA_SLOPE_SHORT_BARS: int = 5
    MEAN_REVERSION_VOL_WINDOW_MINUTES: int = 20
    MEAN_REVERSION_VELOCITY_WINDOW_MINUTES: int = 10
    SMA_WARMUP_DAYS: int = 3
    
    # ========== Score weights ==========
    w_L: float = 0.45  # Liquidity (Barrier) weight
    w_H: float = 0.35  # Hedge (Fuel) weight
    w_T: float = 0.20  # Tape weight
    
    # ========== Trigger thresholds ==========
    BREAK_SCORE_THRESHOLD: float = 80.0
    REJECT_SCORE_THRESHOLD: float = 20.0
    TRIGGER_HOLD_TIME: float = 3.0  # seconds score must be sustained

    # ========== Outcome labeling ==========
    # Threshold for BREAK/BOUNCE classification - must move 2 strikes for meaningful options trade
    OUTCOME_THRESHOLD: float = 2.0  # $2.00 SPY = 2 strikes minimum for BREAK/BOUNCE
    STRENGTH_THRESHOLD_1: float = 1.0  # $1.00 move
    STRENGTH_THRESHOLD_2: float = 2.0  # $2.00 move
    LOOKFORWARD_MINUTES: int = 8    # Forward window for outcome determination (8 min to cover all confirmations)
    LOOKBACK_MINUTES: int = 10      # Backward window for approach context
    
    # Multi-timeframe confirmation windows
    # Generates outcomes at 2min, 4min, 8min to train models on different horizons
    CONFIRMATION_WINDOW_SECONDS: float = 240.0  # Primary confirmation (4 minutes)
    CONFIRMATION_WINDOWS_MULTI: list = field(
        default_factory=lambda: [120.0, 240.0, 480.0]
    )  # 2min, 4min, 8min
    
    # ========== Smoothing parameters (EWMA half-lives in seconds) ==========
    tau_score: float = 2.0        # break score smoothing
    tau_velocity: float = 1.5     # tape velocity smoothing
    tau_delta_liq: float = 3.0    # barrier delta_liq smoothing
    tau_replenish: float = 3.0    # replenishment ratio smoothing
    tau_dealer_gamma: float = 5.0 # net dealer gamma smoothing

    # ========== Normalization scales (pressure indicators) ==========
    BARRIER_DELTA_LIQ_NORM: float = 200.0
    WALL_RATIO_NORM: float = 2.0
    TAPE_VELOCITY_NORM: float = 0.5  # $/sec, aligns with score engine
    GAMMA_EXPOSURE_NORM: float = 100000.0
    GAMMA_FLOW_NORM: float = 50000.0
    GAMMA_FLOW_ACCEL_NORM: float = 50000.0
    
    # ========== Snap tick cadence ==========
    SNAP_INTERVAL_MS: int = 250  # publish level signals every 250ms
    
    # ========== Level universe settings (SPY ~600 price) ==========
    ROUND_LEVELS_SPACING: float = 1.0  # generate round levels every $1
    STRIKE_RANGE: float = 5.0          # monitor strikes within ±$5 of spot
    VWAP_ENABLED: bool = True

    # ========== Viewport / Focus settings ==========
    VIEWPORT_SCAN_RADIUS: float = 1.0
    VIEWPORT_MAX_TARGETS: int = 8
    VIEWPORT_W_DISTANCE: float = 0.45
    VIEWPORT_W_VELOCITY: float = 0.20
    VIEWPORT_W_CONFLUENCE: float = 0.20
    VIEWPORT_W_GAMMA: float = 0.15
    APPROACH_VELOCITY_NORM: float = 0.50

    # ========== Touch clustering ==========
    TOUCH_CLUSTER_TIME_MINUTES: int = 15
    TOUCH_CLUSTER_PRICE_BAND: float = 0.10

    # ========== Normalization ==========
    ATR_WINDOW_MINUTES: int = 14

    # ========== Confluence feature settings ==========
    VOLUME_LOOKBACK_DAYS: int = 7          # Days for relative volume baseline
    SMA_PROXIMITY_THRESHOLD: float = 0.005  # 0.5% of spot for "close to SMA"
    WALL_PROXIMITY_DOLLARS: float = 1.0     # $1 (1 strike) for GEX wall proximity
    REL_VOL_HIGH_THRESHOLD: float = 1.3     # 30% above average = HIGH volume
    REL_VOL_LOW_THRESHOLD: float = 0.7      # 30% below average = LOW volume

    # ========== Feasibility gate thresholds ==========
    FEASIBILITY_TAPE_IMBALANCE: float = 0.20
    FEASIBILITY_GAMMA_EXPOSURE: float = 50000.0
    FEASIBILITY_LOGIT_STEP: float = 1.0
    FEASIBILITY_LOGIT_CAP: float = 2.5
    
    # ========== Storage/replay settings ==========
    DATA_ROOT: str = os.getenv("DATA_ROOT", _DEFAULT_DATA_ROOT)
    MICRO_BATCH_INTERVAL_S: float = 5.0  # flush bronze parquet every 5s
    
    # ========== Out-of-order tolerance ==========
    LATENESS_BUFFER_MS: int = 500  # tolerate events up to 500ms late
    
    # ========== NATS settings (Phase 2) ==========
    NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
    
    # ========== S3/MinIO settings (Phase 2) ==========
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "spymaster-lake")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    
    # ========== Replay settings ==========
    REPLAY_SPEED: float = float(os.getenv("REPLAY_SPEED", "1.0"))  # 1.0 = realtime, 0 = fast as possible


# Singleton instance
CONFIG = Config()
