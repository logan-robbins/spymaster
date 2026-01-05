# Pipeline Optimization Analysis

**Target**: Apple M4 Silicon 128GB RAM

## Measured Data Volume (per day)

| Dataset | Rows | Memory |
|---------|------|--------|
| `market_by_price_10_clean` (tick input) | **1.3-1.9M** | 1.3-1.8 GB |
| `market_by_price_10_bar5s` (bar output) | 15-17k | 33 MB |

## Executive Summary

| Stage | Current Bottleneck | Optimization | Speedup Est. |
|-------|-------------------|--------------|--------------|
| `SilverComputeBar5sFeatures` | Sequential Python loop over **1.9M ticks/day** | Numba JIT + parallel bars | 10-50x |
| `SilverComputeApproachFeatures` | Python loops for derivatives | Vectorized numpy | 5-10x |
| `SilverExtractLevelEpisodes` | Python loop for trigger detection | Vectorized masks | 2-3x |
| `GoldExtractSetupVectors` | Episode loop | Parallel extraction | 2-4x |
| Cross-date | Already parallel | Increase workers | Linear scaling |

---

## Priority 1: `compute_bar5s_features` (CRITICAL)

**File**: `stages/silver/future/mbp10_bar5s/compute.py`

**Current**: Sequential Python loop over 300,000-600,000 tick events per day

```python
# Lines 116-129 - THE MAIN BOTTLENECK
for idx in range(n):  # n = 500,000+ ticks
    ts_event = int(ticks.ts_event[idx])
    bar_start = int(bar_starts[idx])
    
    if current_bar is not None and bar_start > current_bar.bar_start_ns:
        finalize_bar(current_bar, pre_state, symbol, bars)
        current_bar = BarAccumulator(bar_start)
    
    post_state.load_from_arrays(ticks, idx)
    
    if current_bar is not None:
        process_event(ticks, idx, ts_event, pre_state, post_state, current_bar)
    
    pre_state.copy_from(post_state)
```

**Problem**: 
- ~500k iterations in pure Python
- `BarAccumulator` uses dict-based accumulators (slow)
- `BookState` operations are already numpy but called per-tick

**Solution**: Numba JIT compilation

1. Convert `BarAccumulator` dicts to numpy arrays
2. JIT compile the tick processing loop
3. Pre-compute bar boundaries to enable parallel processing

```python
# bar_accumulator_jit.py - NEW FILE
import numpy as np
from numba import jit, prange
from numba.typed import Dict as NumbaDict

BANDS = ["p0_1", "p1_2", "p2_3", "p3_5", "p5_10"]
N_BANDS = 5

@jit(nopython=True, cache=True)
def process_tick_batch(
    ts_event: np.ndarray,
    action: np.ndarray,
    side: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    bid_px: np.ndarray,  # (n, 10)
    ask_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_sz: np.ndarray,
    bid_ct: np.ndarray,
    ask_ct: np.ndarray,
    bar_starts: np.ndarray,
    bar_duration_ns: int,
) -> tuple:
    """Process all ticks within a single bar - JIT compiled."""
    # ... implementation
```

**Estimated Speedup**: 10-50x for this stage alone

---

## Priority 2: Derivative Features

**File**: `stages/silver/future/compute_approach_features.py`

**Current**: Nested Python loops (lines 239-264)

```python
for touch_id in df["touch_id"].unique():
    for short_name, full_col in DERIV_BASE_FEATURES.items():
        for window in DERIV_WINDOWS:
            # Python loop for d1
            for i in range(window, len(vals)):
                d1[i] = (vals[i] - vals[i - window]) / window
            # Python loop for d2  
            for i in range(2 * window, len(vals)):
                d2[i] = (d1[i] - d1[i - window]) / window
```

**Solution**: Vectorized numpy operations

```python
def _compute_derivative_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
    for short_name, full_col in DERIV_BASE_FEATURES.items():
        if full_col not in df.columns:
            continue
        
        for window in DERIV_WINDOWS:
            d1_col = f"bar5s_deriv_{short_name}_d1_w{window}"
            d2_col = f"bar5s_deriv_{short_name}_d2_w{window}"
            
            # Vectorized first derivative per group
            df[d1_col] = df.groupby("touch_id")[full_col].transform(
                lambda x: (x - x.shift(window)) / window
            )
            
            # Vectorized second derivative per group
            df[d2_col] = df.groupby("touch_id")[d1_col].transform(
                lambda x: (x - x.shift(window)) / window
            )
    
    return df
```

**Estimated Speedup**: 5-10x

---

## Priority 3: Cumulative Features

**File**: `stages/silver/future/compute_approach_features.py`

**Current**: Mask-based loop (lines 178-236)

```python
for touch_id in df["touch_id"].unique():
    mask = df["touch_id"] == touch_id
    df.loc[mask, "bar5s_cumul_trade_vol"] = df.loc[mask, "bar5s_trade_vol_sum"].cumsum()
    # ... more cumsum operations
```

**Solution**: Single groupby().cumsum() call

```python
def _compute_cumulative_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
    cumsum_mappings = {
        "bar5s_cumul_trade_vol": "bar5s_trade_vol_sum",
        "bar5s_cumul_signed_trade_vol": "bar5s_trade_signed_vol_sum",
        "bar5s_cumul_aggbuy_vol": "bar5s_trade_aggbuy_vol_sum",
        "bar5s_cumul_aggsell_vol": "bar5s_trade_aggsell_vol_sum",
        "bar5s_cumul_msg_cnt": "bar5s_meta_msg_cnt_sum",
        "bar5s_cumul_trade_cnt": "bar5s_trade_cnt_sum",
        "bar5s_cumul_add_cnt": "bar5s_meta_add_cnt_sum",
        "bar5s_cumul_cancel_cnt": "bar5s_meta_cancel_cnt_sum",
    }
    
    for out_col, in_col in cumsum_mappings.items():
        if in_col in df.columns:
            df[out_col] = df.groupby("touch_id")[in_col].cumsum()
    
    # Flow cumsum
    for band in FLOW_BANDS:
        for side in ["bid", "ask"]:
            in_col = f"bar5s_flow_net_vol_{side}_{band}_sum"
            out_col = f"bar5s_cumul_flow_net_{side}_{band}"
            if in_col in df.columns:
                df[out_col] = df.groupby("touch_id")[in_col].cumsum()
    
    # Rate calculation (vectorized)
    df["bars_elapsed"] = df.groupby("touch_id").cumcount() + 1
    df["bar5s_cumul_signed_trade_vol_rate"] = df["bar5s_cumul_signed_trade_vol"] / df["bars_elapsed"]
    
    return df
```

**Estimated Speedup**: 2-5x

---

## Priority 4: Setup Signature Features

**File**: `stages/silver/future/compute_approach_features.py` (lines 332-410)

**Current**: Another mask-based loop over touch_ids

**Solution**: Use groupby().apply() or vectorized aggregations

```python
def _compute_setup_signature_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
    dist_col = "bar5s_approach_dist_to_level_pts_eob"
    
    # Groupby aggregations
    aggs = df.groupby("touch_id").agg(
        setup_start_dist=(dist_col, "first"),
        setup_min_dist=(dist_col, lambda x: np.abs(x).min()),
        setup_max_dist=(dist_col, lambda x: np.abs(x).max()),
        setup_total_trade_vol=("bar5s_trade_vol_sum", "sum"),
        setup_total_signed_vol=("bar5s_trade_signed_vol_sum", "sum"),
        setup_bid_wall_max_z=("bar5s_wall_bid_maxz_eob", "max"),
        setup_ask_wall_max_z=("bar5s_wall_ask_maxz_eob", "max"),
        n_bars=(dist_col, "count"),
    )
    
    # Merge back
    df = df.merge(aggs.add_prefix("bar5s_"), on="touch_id", how="left")
    
    return df
```

---

## Priority 5: Cross-Date Parallelization

**Current**: `--workers N` flag in runner.py

**Recommendation**: With 128GB RAM, increase default workers

```python
# runner.py modification
DEFAULT_WORKERS = min(os.cpu_count(), 12)  # M4 has 10-14 cores
```

**Memory per date**: ~2-4GB for full pipeline
**Safe parallelism**: 16-24 dates concurrently

---

## Implementation Plan

### Phase 1: Vectorize approach features (1 day) ✅ COMPLETED
- [x] Replace derivative loops with groupby.transform
- [x] Replace cumulative loops with groupby.cumsum
- [x] Replace setup signature loops with vectorized groupby operations
- Benchmark: expect 5-10x on approach stage

### Phase 2: Numba JIT for bar5s (2-3 days) ✅ COMPLETED
- [x] Create `numba_core.py` with JIT-compiled tick processing functions
- [x] Convert dict-based accumulators to numpy arrays
- [x] JIT compile tick processing core with `process_all_ticks()`
- [x] Pre-compute bar boundaries with `unique_bar_starts`

**Benchmark Results (2025-06-05, 1.89M ticks → 17,281 bars):**
- JIT compilation overhead: ~0.5s (first run only)
- Steady-state throughput: **159,000 ticks/sec**
- Processing time: ~12s for 1.9M ticks

### Phase 3: Parallel enhancements (0.5 day) ✅ COMPLETED
- [x] Increase default workers to 8 (from 1)
- [ ] Add memory-aware worker selection (future enhancement)

### Phase 4: I/O optimization (0.5 day) ✅ VERIFIED
- [x] Parquet compression already optimal: `zstd` level 3 in `io.py:138`
- [x] Pre-sort by ts_event already implemented: `process_dbn.py:149`

---

## Quick Wins ✅ ALL COMPLETED

### 1. Vectorized derivatives ✅
Replaced with `groupby.transform(lambda x: (x - x.shift(window)) / window)`

### 2. Groupby cumsum ✅
Replaced mask-based cumsum with `groupby[col].cumsum()`

### 3. Increase workers ✅
Default increased from 1 to 8 in `runner.py`

