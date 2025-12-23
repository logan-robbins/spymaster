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
from dataclasses import dataclass


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
    W_b: float = 60.0  # Barrier engine: forward window for quote/trade accounting (full bar)
    W_t: float = 60.0  # Tape engine: forward imbalance window near level (full bar)
    W_g: float = 60.0  # Fuel engine: option flow window for net dealer gamma
    W_v: float = 3.0   # Velocity: window for slope calculation
    W_wall: float = 300.0  # Call/Put wall lookback window (5 minutes)
    
    # ========== Monitoring bands (SPY dollars) ==========
    # Critical zone where bounce/break decision happens
    MONITOR_BAND: float = 0.25  # compute full signals if |spot - L| <= $0.25
    TOUCH_BAND: float = 0.10    # tight band for "touching level"
    
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
    LOOKFORWARD_MINUTES: int = 5    # Forward window for outcome determination
    LOOKBACK_MINUTES: int = 10      # Backward window for approach context
    
    # ========== Smoothing parameters (EWMA half-lives in seconds) ==========
    tau_score: float = 2.0        # break score smoothing
    tau_velocity: float = 1.5     # tape velocity smoothing
    tau_delta_liq: float = 3.0    # barrier delta_liq smoothing
    tau_replenish: float = 3.0    # replenishment ratio smoothing
    tau_dealer_gamma: float = 5.0 # net dealer gamma smoothing
    
    # ========== Snap tick cadence ==========
    SNAP_INTERVAL_MS: int = 250  # publish level signals every 250ms
    
    # ========== Level universe settings (SPY ~600 price) ==========
    ROUND_LEVELS_SPACING: float = 1.0  # generate round levels every $1
    STRIKE_RANGE: float = 5.0          # monitor strikes within ±$5 of spot
    VWAP_ENABLED: bool = True
    
    # ========== Storage/replay settings ==========
    DATA_ROOT: str = "backend/data/lake/"
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

