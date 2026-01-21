# Technical Implementation Plan: future_option_mbo Pipeline

**Target:** Add GEX (Gamma Exposure) features from ES future options MBO data to the existing `future_mbo` pipeline feature set.

**Goal:** The final gold trigger vectors should contain both the existing vacuum/physics features AND new GEX-derived features computed from option order flow.

---

## 1. Data Discovery Summary

### 1.1 Raw Data Location
```
backend/lake/raw/source=databento/product_type=future_option/symbol=ES/table=market_by_order_dbn/
├── condition.json
├── manifest.json
├── metadata.json
├── symbology.json
└── glbx-mdp3-YYYYMMDD.mbo.dbn.zst  (daily files)
```

- **Schema:** MBO (Market By Order) - same rtype=160 as futures
- **Query:** `ES.OPT` parent symbol (all ES options)
- **Date range:** 2026-01-06 through 2026-01-19

### 1.2 Option Symbol Format (from symbology.json)
- Standard options: `ESH6 C2100` (underlying=ESH6, right=C, strike=2100)
- Standard options: `ESH6 P8500` (underlying=ESH6, right=P, strike=8500)
- Exotic spreads: `UD:1V:*` patterns (filter these out)

### 1.3 Existing Contracts for future_option (trades/nbbo/statistics)
These are SEPARATE schemas for different MBP data. The new MBO data has the same schema as futures MBO.

---

## 2. Architecture Decision

**Approach:** Create a `future_option_mbo` product type that processes option MBO data in parallel with `future_mbo`, then JOIN the features at the gold layer.

**Rationale:**
1. Options data is ~10x larger than futures (thousands of strikes per day)
2. GEX aggregation must be strike-bucket relative to P_ref level
3. Final feature vector dimension must remain fixed for FAISS index compatibility
4. Silver vacuum features (5s windows) align perfectly with GEX aggregation windows

---

## 3. Pipeline Stages

### 3.1 Bronze Stage: `BronzeIngestFutureOptionMboPreview`

**File:** `backend/src/data_eng/stages/bronze/future_option_mbo/ingest_preview.py`

**Input:** Raw DBN files at `backend/lake/raw/source=databento/product_type=future_option/symbol=ES/table=market_by_order_dbn/`

**Output:** `bronze.future_option_mbo.mbo` partitions per underlying contract per day

**Logic:**
1. Read DBN files for date (same pattern as `future_mbo`)
2. Filter to rtype=160 (MBO records only)
3. Filter out spread symbols (contains `-`) and exotic symbols (starts with `UD:`)
4. Parse option symbol to extract:
   - `underlying` (ESH6, ESM6, etc.)
   - `right` (C or P)
   - `strike` (numeric)
5. Enforce session window: 05:00-16:00 NY
6. Cast types (same as futures MBO)
7. Write partitions keyed by `underlying` (NOT the full option symbol)

**Contract schema (new):** `backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc`
```json
{
  "namespace": "com.marketdata.bronze.future_option_mbo",
  "type": "record",
  "name": "OptionMboEvent",
  "fields": [
    {"name": "ts_recv", "type": "long"},
    {"name": "ts_event", "type": "long"},
    {"name": "order_id", "type": "long"},
    {"name": "price", "type": "long"},
    {"name": "size", "type": "long"},
    {"name": "action", "type": "string"},
    {"name": "side", "type": "string"},
    {"name": "sequence", "type": "long"},
    {"name": "underlying", "type": "string"},
    {"name": "option_symbol", "type": "string"},
    {"name": "exp_date", "type": "string"},
    {"name": "strike", "type": "long"},
    {"name": "right", "type": "string"},
    {"name": "channel_id", "type": "long"},
    {"name": "rtype", "type": "long"},
    {"name": "publisher_id", "type": "long"},
    {"name": "flags", "type": "long"},
    {"name": "instrument_id", "type": "long"},
    {"name": "ts_in_delta", "type": "long"}
  ]
}
```

**Symbol Parsing Function:**
```python
import re
OPTION_SYMBOL_RE = re.compile(r'^(ES[HMUZ]\d{1,2})\s+([CP])(\d+)$')

def parse_option_symbol(symbol: str) -> tuple[str, str, int] | None:
    """Parse 'ESH6 C2100' -> ('ESH6', 'C', 2100) or None if invalid."""
    match = OPTION_SYMBOL_RE.match(symbol.strip())
    if not match:
        return None
    return (match.group(1), match.group(2), int(match.group(3)))
```

---

### 3.2 Silver Stage: `SilverComputeOptionGex5s`

**File:** `backend/src/data_eng/stages/silver/future_option_mbo/compute_option_gex_5s.py`

**Input:** `bronze.future_option_mbo.mbo`

**Output:** `silver.future_option_mbo.option_gex_5s`

**Logic:**

#### 3.2.1 Greeks Calculation
Since the raw MBO data does not include Greeks, implement Black-Scholes approximation:

```python
from scipy.stats import norm
import numpy as np

def black_scholes_greeks(
    S: float,      # spot price
    K: float,      # strike
    T: float,      # time to expiry (years)
    r: float,      # risk-free rate (use 0.05)
    sigma: float,  # implied vol (use 0.20 default or estimate from spread)
    right: str,    # 'C' or 'P'
) -> tuple[float, float]:
    """Return (delta, gamma) for option."""
    if T <= 0:
        return (0.0, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if right == 'C':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    
    return (delta, gamma)
```

**Note:** For better accuracy, consider estimating implied volatility from option MBO bid/ask spread or use a volatility surface if available.

#### 3.2.2 GEX Aggregation by Strike Bucket

Define strike buckets relative to P_ref (same as vacuum buckets):
- `at`: strike within 0-5 pts of P_ref
- `near`: strike within 6-15 pts of P_ref
- `mid`: strike within 16-30 pts of P_ref
- `far`: strike within 31-50 pts of P_ref
- `beyond`: strike > 50 pts from P_ref

For each 5-second window, compute:
```python
GEX_FEATURES = [
    # Call GEX above P_ref
    "gex_call_above_at",
    "gex_call_above_near",
    "gex_call_above_mid",
    "gex_call_above_far",
    # Put GEX above P_ref
    "gex_put_above_at",
    "gex_put_above_near",
    "gex_put_above_mid",
    "gex_put_above_far",
    # Call GEX below P_ref
    "gex_call_below_at",
    "gex_call_below_near",
    "gex_call_below_mid",
    "gex_call_below_far",
    # Put GEX below P_ref
    "gex_put_below_at",
    "gex_put_below_near",
    "gex_put_below_mid",
    "gex_put_below_far",
    # Net GEX (calls - puts)
    "gex_net_above_at",
    "gex_net_above_near",
    "gex_net_below_at",
    "gex_net_below_near",
    # Aggregates
    "gex_total_above",
    "gex_total_below",
    "gex_asymmetry",  # (above - below) / (above + below + eps)
    "gex_call_put_ratio",  # call_gex / (put_gex + eps)
]
```

#### 3.2.3 Order Flow Tracking

Track option order book state similar to futures vacuum:
- Maintain order book per strike/right
- Track: add_qty, cancel_qty, fill_qty per bucket
- Compute flow metrics: intensity, pull/add ratio

```python
GEX_FLOW_FEATURES = [
    "opt_add_intensity_call_above",
    "opt_add_intensity_put_below",
    "opt_pull_intensity_call_above",
    "opt_pull_intensity_put_below",
    "opt_flow_imbalance_above",  # call_add - put_add above P_ref
    "opt_flow_imbalance_below",  # put_add - call_add below P_ref
]
```

#### 3.2.4 Window Alignment

The 5-second window boundaries MUST align exactly with `future_mbo` vacuum windows:
- Window ID = `ts_event // 5_000_000_000`
- Use same WINDOW_NS = 5_000_000_000

**Contract schema:** `backend/src/data_eng/contracts/silver/future_option_mbo/option_gex_5s.avsc`
- Include all GEX features
- Include all flow features
- Include `window_start_ts_ns`, `window_end_ts_ns`, `P_ref`, `approach_dir`

---

### 3.3 Gold Stage: Merge GEX with Trigger Vectors

**Option A (Recommended): Extend `GoldBuildMboTriggerVectors`**

Modify `backend/src/data_eng/stages/gold/future_mbo/build_trigger_vectors.py` to:
1. Read BOTH `silver.future_mbo.mbo_level_vacuum_5s` AND `silver.future_option_mbo.option_gex_5s`
2. Join on `window_end_ts_ns` (exact match)
3. Concatenate GEX features to the feature vector

**Option B: Create Separate Stage**

Create `GoldBuildMboTriggerVectorsWithGex` that:
1. Reads trigger vectors from existing gold stage
2. Reads GEX silver features
3. Joins and extends vectors

**Recommended:** Option A - modify existing stage to keep pipeline simple.

#### 3.3.1 Vector Schema Extension

Update `backend/src/data_eng/vector_schema.py`:

```python
# Add to F_DOWN (approach from above - price going down)
GEX_DOWN = [
    "gex_call_above_at",
    "gex_call_above_near",
    "gex_put_below_at",
    "gex_put_below_near",
    "gex_net_above_at",
    "gex_net_below_at",
    "gex_asymmetry",
    "opt_flow_imbalance_above",
    "opt_flow_imbalance_below",
]

# Add to F_UP (approach from below - price going up)
GEX_UP = [
    "gex_put_below_at",
    "gex_put_below_near",
    "gex_call_above_at",
    "gex_call_above_near",
    "gex_net_below_at",
    "gex_net_above_at",
    "gex_asymmetry_inv",  # inverted for up approach
    "opt_flow_imbalance_below",
    "opt_flow_imbalance_above",
]
```

This adds ~18 base GEX features × 4 derivatives (d1/d2/d3) × 7 window blocks = 504 dimensions.

**IMPORTANT:** New vector dimension requires:
1. Re-running silver for all dates
2. Re-building trigger vectors for all dates
3. Re-computing norm_stats_seed.json
4. Re-building FAISS index

---

### 3.4 Contract-Day Selection

**Reuse existing selector:** The option MBO data should use the SAME contract selection as futures:
- Read `backend/lake/selection/mbo_contract_day_selection.parquet`
- Use `selected_symbol` to filter options by underlying

**Rationale:** Options on ESH6 should be processed when ESH6 is the dominant futures contract.

---

## 4. Dataset Configuration

Add to `backend/src/data_eng/config/datasets.yaml`:

```yaml
# Bronze - Future Option MBO
bronze.future_option_mbo.mbo:
  path: bronze/source=databento/product_type=future_option_mbo/symbol={symbol}/table=mbo
  format: parquet
  partition_keys: [symbol, dt]
  contract: src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc

# Silver - Future Option MBO
silver.future_option_mbo.option_gex_5s:
  path: silver/product_type=future_option_mbo/symbol={symbol}/table=option_gex_5s
  format: parquet
  partition_keys: [symbol, dt]
  contract: src/data_eng/contracts/silver/future_option_mbo/option_gex_5s.avsc
```

---

## 5. Pipeline Registration

Update `backend/src/data_eng/pipeline.py`:

```python
elif product_type == "future_option_mbo":
    from .stages.bronze.future_option_mbo.ingest_preview import BronzeIngestFutureOptionMboPreview
    from .stages.silver.future_option_mbo.compute_option_gex_5s import SilverComputeOptionGex5s

    if layer == "bronze":
        return [BronzeIngestFutureOptionMboPreview()]
    elif layer == "silver":
        return [SilverComputeOptionGex5s()]
    elif layer == "gold":
        return []  # Gold features merged into future_mbo pipeline
    elif layer == "all":
        return [
            BronzeIngestFutureOptionMboPreview(),
            SilverComputeOptionGex5s(),
        ]
```

---

## 6. Implementation Order

### Phase 1: Bronze Ingestion
1. Create `backend/src/data_eng/stages/bronze/future_option_mbo/__init__.py`
2. Create `backend/src/data_eng/stages/bronze/future_option_mbo/ingest_preview.py`
3. Create `backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc`
4. Add dataset to `datasets.yaml`
5. Test: Run bronze for single date, verify partition structure

### Phase 2: Silver GEX Computation
1. Create `backend/src/data_eng/stages/silver/future_option_mbo/__init__.py`
2. Create `backend/src/data_eng/stages/silver/future_option_mbo/compute_option_gex_5s.py`
3. Create `backend/src/data_eng/contracts/silver/future_option_mbo/option_gex_5s.avsc`
4. Implement Black-Scholes Greeks calculator
5. Implement GEX aggregation by strike bucket
6. Add dataset to `datasets.yaml`
7. Test: Verify 5s windows align with vacuum windows

### Phase 3: Gold Integration
1. Update `vector_schema.py` with GEX feature definitions
2. Update `build_trigger_vectors.py` to read and join GEX features
3. Update `mbo_trigger_vectors.avsc` contract (add vector_dim update)
4. Update `mbo_level_vacuum_5s.avsc` if needed
5. Test: Verify joined vectors have correct dimension

### Phase 4: Index Rebuild
1. Run full silver rebuild (both pipelines)
2. Run trigger vector rebuild
3. Regenerate `norm_stats_seed.json` for new dimension
4. Rebuild FAISS index
5. Run signal and pressure stream stages
6. Validate retrieval still works

---

## 7. Testing Strategy

### 7.1 Unit Tests
- `test_parse_option_symbol()` - verify symbol parsing
- `test_black_scholes_greeks()` - verify delta/gamma calculations
- `test_gex_bucket_assignment()` - verify strike bucket logic

### 7.2 Integration Tests
- Verify bronze option partitions match futures contract selection
- Verify silver GEX windows align with vacuum windows
- Verify gold vector dimension is consistent

### 7.3 Validation
- Check GEX sign conventions (calls positive gamma, puts positive gamma)
- Verify GEX asymmetry is bounded [-1, 1]
- Confirm no NaN/Inf in features

---

## 8. Key Files to Create/Modify

### New Files
```
backend/src/data_eng/stages/bronze/future_option_mbo/__init__.py
backend/src/data_eng/stages/bronze/future_option_mbo/ingest_preview.py
backend/src/data_eng/stages/silver/future_option_mbo/__init__.py
backend/src/data_eng/stages/silver/future_option_mbo/compute_option_gex_5s.py
backend/src/data_eng/contracts/bronze/future_option_mbo/__init__.py
backend/src/data_eng/contracts/bronze/future_option_mbo/mbo.avsc
backend/src/data_eng/contracts/silver/future_option_mbo/__init__.py
backend/src/data_eng/contracts/silver/future_option_mbo/option_gex_5s.avsc
```

### Modified Files
```
backend/src/data_eng/config/datasets.yaml
backend/src/data_eng/pipeline.py
backend/src/data_eng/vector_schema.py
backend/src/data_eng/stages/gold/future_mbo/build_trigger_vectors.py
backend/src/data_eng/contracts/gold/future_mbo/mbo_trigger_vectors.avsc
backend/src/data_eng/contracts/silver/future_mbo/mbo_level_vacuum_5s.avsc (if GEX added there)
```

---

## 9. Reference Patterns

### Bronze Ingestion Pattern
Reference: `backend/src/data_eng/stages/bronze/future_mbo/ingest_preview.py`
- Lines 21-122: Full implementation showing DBN reading, filtering, partitioning

### Silver Vacuum Pattern
Reference: `backend/src/data_eng/stages/silver/future_mbo/compute_level_vacuum_5s.py`
- Lines 133-186: Stage setup with StageIO
- Lines 197-445: Core computation with order state tracking
- Lines 448-473: Bucket assignment logic
- Lines 476-563: Snapshot computation
- Lines 566-643: Feature calculation

### Gold Vector Building Pattern
Reference: `backend/src/data_eng/stages/gold/future_mbo/build_trigger_vectors.py`
- Lines 94-178: Multi-input stage with join logic
- Lines 255-347: Vector construction

### Contract Selection Pattern
Reference: `backend/src/data_eng/retrieval/mbo_contract_day_selector.py`
- Lines 46-66: Symbol discovery
- Lines 290-297: Load selection map

---

## 10. Environment Variables

No new environment variables required. Existing variables used:
- `MBO_SELECTION_PATH`: Selection map path (defaults to `backend/lake/selection/mbo_contract_day_selection.parquet`)
- `LEVEL_ID`: Level identifier (e.g., `pm_high`)
- `MBO_INDEX_DIR`: Index directory path

---

## 11. Execution Commands

### Bronze Processing
```bash
uv run python -m src.data_eng.runner \
  --product-type future_option_mbo \
  --layer bronze \
  --symbol ES \
  --dates 2026-01-06:2026-01-19 \
  --workers 4
```

### Silver Processing
```bash
uv run python -m src.data_eng.runner \
  --product-type future_option_mbo \
  --layer silver \
  --symbol ES \
  --dates 2026-01-06:2026-01-19 \
  --workers 4
```

### Full Rebuild (after integration)
Use existing rebuild script which should be updated to include option processing:
```bash
nohup bash backend/scripts/rebuild_future_mbo_all_pmhigh.sh > backend/logs/rebuild_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

---

## 12. Critical Constraints

1. **Window Alignment:** GEX 5s windows MUST have identical `window_start_ts_ns` and `window_end_ts_ns` as vacuum windows for proper join.

2. **P_ref Consistency:** GEX bucket assignment uses SAME P_ref as vacuum computation (premarket high from 05:00-09:30).

3. **Contract Selection:** Options processed for underlying ESH6 only when ESH6 is selected contract for that date.

4. **Vector Dimension:** Any change to vector dimension requires full pipeline rebuild including FAISS index.

5. **Feature Order:** GEX features must be appended in consistent order to maintain vector schema compatibility.

6. **Nullable Handling:** Days with no option data should produce zero-valued GEX features, NOT skip the date.
