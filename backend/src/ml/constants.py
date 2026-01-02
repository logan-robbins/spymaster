"""Constants for ML/retrieval system - Analyst Opinion Specification."""

# ─── Zone Thresholds (ATR units) ───
# Analyst recommendation: tighter zones for higher quality anchors
Z_APPROACH_ATR = 2.0  # Compromise: 1.25 (analyst) vs 3.0 (original)
Z_CONTACT_ATR = 0.20  # "At level" threshold
Z_EXIT_ATR = 2.5      # Episode exit threshold
MAX_EXIT_GAP_BARS = 2  # Allow brief excursions

# ─── Anchor Emission Gates ───
MIN_APPROACH_BARS = 2  # Minimum bars approaching level
MIN_APPROACH_V_ATR_PER_MIN = 0.10  # Minimum velocity (ATR/min)

# ─── Retrieval Parameters ───
M_CANDIDATES = 500  # Over-fetch from FAISS
K_NEIGHBORS = 50    # Final neighbors after dedup
MAX_PER_DAY = 2     # Max neighbors from same date
MAX_PER_EPISODE = 1 # Max neighbors from same episode_id

# Neighbor weighting
SIM_POWER = 4.0  # Power transform on similarity
RECENCY_HALFLIFE_DAYS = 60  # Exponential decay halflife

# Quality thresholds
MIN_SIMILARITY_THRESHOLD = 0.70
MIN_SAMPLES_THRESHOLD = 30
N_EFF_MIN = 15  # Minimum effective sample size

# ─── Time Buckets (5 buckets per analyst) ───
# Split first 30 min to separate OR formation (0-15) from post-OR (15-30)
TIME_BUCKETS = {
    'T0_15': (0, 15),      # OR formation period
    'T15_30': (15, 30),    # Post-OR early
    'T30_60': (30, 60),
    'T60_120': (60, 120),
    'T120_180': (120, 180)
}

# ─── Episode Vector Dimensions ───
# ─── Episode Vector Dimensions ───
VECTOR_DIMENSION = 195  # Phase 4.6 (Split Flow): 183 + 8 (OFI Splits) + 4 (Tide Splits)

# Vector section boundaries
VECTOR_SECTIONS = {
    'context_regime': (0, 59),        # 59 dims (Unchanged)
    'multiscale_dynamics': (59, 107), # 48 dims (+8 OFI Splits)
    'micro_history': (107, 142),      # 35 dims (Shift +8)
    'derived_physics': (142, 159),    # 17 dims (+4 Tide Splits, Shift +8)
    'online_trends': (159, 163),      # 4 dims
    'trajectory_basis': (163, 195),   # 32 dims
}

# ─── Normalization Parameters ───
LOOKBACK_DAYS = 60
CLIP_SIGMA = 4.0

# ─── Validation Thresholds ───
MIN_PARTITION_SIZE = 100
DRIFT_WARNING_WASSERSTEIN = 0.5
DRIFT_ALERT_MEAN_SHIFT = 2.0
CALIBRATION_WARNING_ECE = 0.10

# ─── Level Kinds ───
LEVEL_KINDS = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']

# ─── Horizons ───
HORIZONS = {
    '4min': 240,  # Fast signal
    '8min': 480   # Confirmation (primary)
}
PRIMARY_HORIZON = '8min'

# ─── State Table ───
STATE_CADENCE_SEC = 15  # 15s "Goldilocks" cadence per Microstructure Research
SESSION_START_HOUR = 9
SESSION_START_MIN = 30
SESSION_END_HOUR = 12
SESSION_END_MIN = 30
LOOKBACK_WINDOW_MIN = 20  # For trajectory basis DCT

# ─── Outcome Thresholds ───
# Using existing threshold logic (1.0 ATR for break/bounce)
THETA_BREAK = 1.0  # ATR units
THETA_REJECT = 1.0  # ATR units
