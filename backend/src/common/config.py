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
_DEFAULT_DATA_ROOT = os.path.join(_BASE_DIR, "data")


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
    
    # ========== Monitoring bands (ES index points) ==========
    # ES 0DTE strike spacing: 5 points ATM on expiry (wider farther OTM).
    # ES tick: 0.25 points (20 ticks per 5-point strike interval).
    # 
    # KEY: Interaction zone is DIFFERENT from outcome threshold!
    # - Interaction zone: WHERE we detect events (tight around level)
    # - Outcome barrier: HOW FAR price moves for BREAK/BOUNCE (volatility-scaled)
    # 
    # Per user specification: ±5 points for interaction zone
    MONITOR_BAND: float = 5.0   # interaction zone: ±5 ES points (±20 ticks, ~0.2 strike)
    TOUCH_BAND: float = 2.0     # touch zone: ±2 ES points (±8 ticks, very precise)
    CONFLUENCE_BAND: float = 5.0  # band for nearby key level confluence (1 strike @ 5pt spacing)
    
    # Barrier engine: zone around strike-aligned level
    # ES options strikes: 0.25 point tick size, typically $5-$25 intervals (20-100 ticks)
    # ±8 ticks = ±$2.00 ES - captures order book as price approaches strike
    BARRIER_ZONE_ES_TICKS: int = 8  # ±8 ES ticks (±$2.00 ES)
    
    # ========== Barrier thresholds (ES contracts) ==========
    R_vac: float = 0.3   # Replenishment ratio threshold for VACUUM
    R_wall: float = 1.5  # Replenishment ratio threshold for WALL/ABSORPTION
    F_thresh: int = 100  # Delta liquidity threshold (ES contracts, not shares)
    
    # Optional: percentile for WEAK state
    WEAK_PERCENTILE: float = 0.20  # bottom 20th percentile of defending size
    WEAK_LOOKBACK: float = 1800.0  # 30 minutes
    
    # ========== Tape thresholds (ES points) ==========
    TAPE_BAND: float = 0.50  # price band around level for tape imbalance (ES points)
    SWEEP_MIN_NOTIONAL: float = 500_000.0  # minimum notional for sweep detection (ES = $50/pt)
    SWEEP_MAX_GAP_MS: int = 100  # max gap between prints in a sweep cluster
    SWEEP_MIN_VENUES: int = 1    # ES only trades on CME so set to 1
    
    # ========== Fuel thresholds ==========
    # ES 0DTE options: 5-point strike spacing ATM on expiry (wider farther OTM).
    # Use point-based ranges and/or nearest listed strikes (no fixed strike grid).
    # 
    # GAMMA IMPACT ASSESSMENT (Cboe, SpotGamma, Menthor Q studies):
    # - Net gamma exposure is 0.04-0.17% of ES daily volume
    # - Hedging flows are balanced (not directional drivers)
    # - Effect on ES: pinning/chop near strikes, NOT sustained breaks
    # - Liquidity (order book) + Tape (directional flow) are primary drivers
    FUEL_STRIKE_RANGE: float = 15.0  # consider strikes within ±15 points
    DEALER_FLOW_STRIKE_RANGE: float = 15.0  # strike range for dealer flow velocity
    USE_GAMMA_BUCKET_FILTER: bool = False  # Disable gamma regime filtering in kNN (gamma effects too small)
    GAMMA_FEATURE_WEIGHT: float = 0.3  # Downweight gamma features in ML training (vs 1.0 for liquidity/tape)
    OPTION_CONTRACT_MULTIPLIER: float = 50.0  # ES options contract multiplier ($50/point)

    # ========== Mean reversion (SMA) settings ==========
    SMA_SLOPE_WINDOW_MINUTES: int = 20
    SMA_SLOPE_SHORT_BARS: int = 5
    MEAN_REVERSION_VOL_WINDOW_MINUTES: int = 20
    MEAN_REVERSION_VELOCITY_WINDOW_MINUTES: int = 10
    SMA_WARMUP_DAYS: int = 3
    
    # ========== Score weights ==========
    # Note: Gamma effects are small relative to ES futures liquidity (0.04-0.17% of volume)
    # Liquidity (order book) and Tape (directional flow) are primary drivers
    w_L: float = 0.55  # Liquidity (Barrier) weight - INCREASED (primary driver)
    w_H: float = 0.10  # Hedge (Fuel) weight - REDUCED (gamma overstated, pinning only)
    w_T: float = 0.35  # Tape weight - INCREASED (directional flow matters)
    
    # ========== Trigger thresholds ==========
    BREAK_SCORE_THRESHOLD: float = 80.0
    REJECT_SCORE_THRESHOLD: float = 20.0
    TRIGGER_HOLD_TIME: float = 3.0  # seconds score must be sustained

    # ========== v1 Scope: ES futures + ES options, first 4 hours ==========
    # v1: focus on first 4 hours (09:30-13:30 ET)
    # 
    # FINAL ARCHITECTURE: ES Options + ES Futures (PERFECT ALIGNMENT)
    # - ES options: Cash-settled, European-style, on E-mini S&P 500
    # - ES futures: SAME underlying instrument!
    # - Same venue (CME), same participants, same tick size
    # - Zero conversion needed - ES = ES!
    
    # RTH (Regular Trading Hours) - Equity market hours
    RTH_START_HOUR: int = 9
    RTH_START_MINUTE: int = 30
    RTH_END_HOUR: int = 13  # v1: 13:30 (first 4 hours only)
    RTH_END_MINUTE: int = 30
    
    # PREMARKET (ES Futures) - CRITICAL: Read ES_PREMARKET_DEFINITION.md
    # ES trades 24/7, but we define "premarket" as 4:00 AM - 9:30 AM ET
    # - Aligns with equity premarket (04:00 AM - 09:30 AM ET)
    # - Captures morning session sentiment before equity open
    # - Excludes overnight ES action (6:00 PM prev day - 4:00 AM)
    # - PM_HIGH/PM_LOW from THIS window become structural levels for RTH
    # - SMA warmup includes premarket bars, but "since_open" starts at 9:30!
    PREMARKET_START_HOUR: int = 4  # 4:00 AM ET (can experiment with other values: 0, 2, 6, 7, 18)
    PREMARKET_START_MINUTE: int = 0
    
    # ========== Outcome labeling ==========
    # Labels are volatility-scaled (dynamic barrier) with a fixed horizon.
    # If volatility is missing, fall back to static thresholds for determinism.
    ES_0DTE_STRIKE_SPACING: float = 5.0     # ES 0DTE ATM spacing (CME standard on expiry)
    ES_0DTE_STRIKE_SPACING_WIDE: float = 25.0  # Farther OTM or longer-dated contracts
    OUTCOME_THRESHOLD: float = 15.0         # Fallback only if vol-based barrier unavailable
    STRENGTH_THRESHOLD_1: float = 5.0       # Fallback only if vol-based barrier unavailable
    STRENGTH_THRESHOLD_2: float = 15.0      # Fallback only if vol-based barrier unavailable
    LOOKFORWARD_MINUTES: int = 8            # Default horizon (minutes) when not using multi-timeframe
    LOOKBACK_MINUTES: int = 10              # Backward window for approach context
    
    # Multi-timeframe horizons for outcome labeling (seconds)
    CONFIRMATION_WINDOW_SECONDS: float = 240.0  # Primary confirmation (4 minutes)
    CONFIRMATION_WINDOWS_MULTI: list = field(
        default_factory=lambda: [120.0, 240.0, 480.0]
    )  # 2min, 4min, 8min

    # Volatility-scaled barrier for labels (points)
    LABEL_VOL_WINDOW_SECONDS: int = 120  # realized vol window for dynamic barrier (seconds)
    LABEL_BARRIER_SCALE: float = 1.0     # scale factor on sigma*sqrt(horizon)
    LABEL_BARRIER_MIN_POINTS: float = 5.0
    LABEL_BARRIER_MAX_POINTS: float = 50.0
    LABEL_T1_FRACTION: float = 0.33      # threshold_1 = fraction of barrier (tradeable_1)
    
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

    # ========== Adaptive inference cadence ==========
    INFERENCE_VOL_WINDOW_SECONDS: int = 120
    INFERENCE_MIN_SIGMA_POINTS: float = 0.5
    INFERENCE_Z_ENGAGED: float = 1.5
    INFERENCE_Z_APPROACH: float = 2.0  # Updated: 2.0 ATR (compromise analyst opinion)
    INFERENCE_INTERVAL_ENGAGED_S: float = 0.25
    INFERENCE_INTERVAL_APPROACH_S: float = 2.0
    INFERENCE_INTERVAL_FAR_S: float = 10.0
    INFERENCE_TAPE_IMBALANCE_JUMP: float = 0.25
    INFERENCE_GAMMA_FLIP_THRESHOLD: float = 0.0
    
    # ========== Level universe settings (ES ~5700-5800 index) ==========
    # ES levels are in index points (same as ES futures/options)
    # v1: We don't generate ROUND levels (removed from level universe)
    ROUND_LEVELS_SPACING: float = 10.0  # not used in v1 (for reference: ES rounds at 10 pt intervals)
    STRIKE_RANGE: float = 50.0          # monitor strikes within ±50 points of spot
    VWAP_ENABLED: bool = False  # v1: VWAP removed from level universe

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
    VOLUME_LOOKBACK_DAYS: int = 3          # Days for relative volume baseline
    SMA_PROXIMITY_THRESHOLD: float = 0.005  # 0.5% of spot for "close to SMA"
    WALL_PROXIMITY_POINTS: float = 15.0    # 15 ES points for GEX wall proximity
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
