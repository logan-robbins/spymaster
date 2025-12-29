State Table Source:
Current: May forward-fill features from event table
Analyst: Insists on computing directly from streaming accumulators (no forward-fill)
Critical for arbitrary bar-close queries (e.g., 16-minute mark)
Time Buckets: You have 4 buckets, analyst has 5 buckets
Analyst splits first 30 min: T0_15 (OR formation) + T15_30 (post-OR)
Results in 60 partitions vs your 48
Medium Impact Differences (Recommended Enhancements)
Log Transforms: Analyst uses barrier_delta_liq_log and wall_ratio_log in micro-history (you use raw values)
Retrieval Deduplication: Analyst retrieves 500 candidates then applies:
MAX_PER_DAY = 2 (max 2 neighbors from same date)
MAX_PER_EPISODE = 1 (max 1 from same episode)
You retrieve 100 without explicit dedup
Normalization: You use global stats, analyst recommends partition-aware (per level_kind × direction × time_bucket with fallbacks)
Neighbor Weighting: Analyst adds recency decay (exp(-age_days/60)) and power transform (sim^4)
Index Type: You auto-select Flat/IVF/IVFPQ, analyst recommends HNSW universally (better for incremental updates)
Low Impact Differences (Statistical Refinements)
Outcome Aggregation: You use weighted mean, analyst uses Dirichlet posterior with priors
Context Redundancy: You encode level_kind/direction in vector (redundant since they're partition keys)