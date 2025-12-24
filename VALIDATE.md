# VALIDATE - Bulletproof Data Pipeline Validation

**Purpose**: Comprehensive validation protocol to ensure Bronze → Silver → Gold pipeline produces ML-ready data with statistical integrity, no data corruption, and accurate physics calculations.

**Audience**: AI Coding Agent executing validation after Bronze backfill completion.

**Context**: December 2025 dataset (15 trading days) backfilled with optimized batch sizes (7.5M trades, 15M MBP-10). Known issues discovered during backfill require post-processing cleanup.

---

## 0) Current State & Known Issues

### Bronze Data Location
```
backend/data/lake/bronze/
├── futures/
│   ├── trades/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
│   └── mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/*.parquet
└── options/
    └── trades/underlying=SPY/date=YYYY-MM-DD/*.parquet
```

### Completed Backfill Details
- **ES Futures**: 16 dates (12-01 through 12-19, excluding 12-06, 12-07, 12-13)
- **SPY Options**: 15 dates (same as ES, excluding 12-14 Saturday)
- **Batch Sizes**: 7.5M trades, 15M MBP-10 (optimized for 128GB RAM M4 Silicon)
- **Clean Flag**: Used `--clean` to remove old duplicate data from previous runs

### Known Data Quality Issues (MUST FIX)

**1. Multiple ES Contracts in Bronze**
- **Problem**: DBN files contain multiple ES contracts (weekly, monthly, front-month)
- **Symptom**: Price range shows $58-$6,900 instead of expected $6,800-$6,900
- **Examples**:
  - `ES_294973`: Front month contract (~96% volume, $6,800-$6,900) ✅ KEEP
  - `ES_42007065`: Weekly contract (~1% volume, $58-$59) ❌ REMOVE
  - `ES_42140878`: Next month contract (~2% volume, $6,900-$7,000) ❌ REMOVE
- **Impact**: Price analysis distorted, OHLCV incorrect, level detection wrong
- **Solution**: Filter Bronze data to dominant symbol (ES_294973) per date before Silver tier

**2. Minor MBP-10 Level Inversions**
- **Issue**: 0.1% of rows have bid_px_9 < bid_px_10 (sparse book at deep levels)
- **Impact**: Minimal - deep levels rarely used
- **Action**: Document but acceptable for physics calculations

**3. Null Fields (Expected/Normal)**
- `conditions`: 100% null (optional trade flags)
- `opt_bid/opt_ask`: 100% null (not in trade data)
- **Impact**: None - these are optional fields

---

## 1) Bronze Data Validation

### 1.1 File Integrity Check

**Objective**: Verify all expected dates have complete data with no corrupted files.

```python
# Check expected vs actual dates
expected_dates = [
    '2025-12-01', '2025-12-02', '2025-12-03', '2025-12-04', '2025-12-05',
    '2025-12-08', '2025-12-09', '2025-12-10', '2025-12-11', '2025-12-12',
    '2025-12-14', '2025-12-15', '2025-12-16', '2025-12-17', '2025-12-18', '2025-12-19'
]

for date in expected_dates:
    # Check ES trades
    trades_path = f"bronze/futures/trades/symbol=ES/date={date}"
    assert trades_path.exists(), f"Missing ES trades for {date}"
    
    # Check ES MBP-10
    mbp10_path = f"bronze/futures/mbp10/symbol=ES/date={date}"
    assert mbp10_path.exists(), f"Missing ES MBP-10 for {date}"
    
    # Check SPY options (except 12-14 Saturday)
    if date != '2025-12-14':
        opts_path = f"bronze/options/trades/underlying=SPY/date={date}"
        assert opts_path.exists(), f"Missing SPY options for {date}"
```

### 1.2 Row Count Consistency

**Critical**: Verify no duplicate data from previous backfill runs.

```python
# For each date, check row counts are reasonable
import pandas as pd
from pathlib import Path

def validate_row_counts(date):
    bronze = Path("backend/data/lake/bronze")
    
    # ES Trades: Expect 200k-800k per weekday
    trades_files = list((bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"))
    trade_count = sum(len(pd.read_parquet(f)) for f in trades_files)
    
    # ES MBP-10: Expect 5M-25M per weekday
    mbp10_files = list((bronze / f"futures/mbp10/symbol=ES/date={date}").rglob("*.parquet"))
    mbp10_count = sum(len(pd.read_parquet(f)) for f in mbp10_files)
    
    # SPY Options: Expect 600k-900k per weekday
    opts_files = list((bronze / f"options/trades/underlying=SPY/date={date}").rglob("*.parquet"))
    opts_count = sum(len(pd.read_parquet(f)) for f in opts_files) if opts_files else 0
    
    print(f"{date}:")
    print(f"  Trades: {trade_count:,}")
    print(f"  MBP-10: {mbp10_count:,}")
    print(f"  Options: {opts_count:,}")
    
    # Saturday 12-14 should have minimal data
    if date == '2025-12-14':
        assert trade_count < 20000, "12-14 Saturday should have minimal trades"
    else:
        assert 100000 < trade_count < 1000000, f"Unusual trade count: {trade_count:,}"
        assert 2000000 < mbp10_count < 30000000, f"Unusual MBP-10 count: {mbp10_count:,}"
    
    return trade_count, mbp10_count, opts_count
```

### 1.3 Exact Duplicate Detection

**Critical**: Verify `--clean` flag eliminated duplicates from previous runs.

```python
def check_duplicates(date):
    """Check for exact duplicate rows (same timestamp, price, size)."""
    bronze = Path("backend/data/lake/bronze")
    
    # Sample trades from one busy hour (hour=15)
    trades_files = list((bronze / f"futures/trades/symbol=ES/date={date}/hour=15").glob("*.parquet"))
    
    if trades_files:
        df = pd.read_parquet(trades_files[0])
        
        # Check for exact duplicates
        exact_dups = df.duplicated(keep=False).sum()
        
        # Check for timestamp+price+size duplicates (trade-level)
        trade_dups = df.duplicated(subset=['ts_event_ns', 'price', 'size'], keep=False).sum()
        
        print(f"{date} duplicates:")
        print(f"  Exact duplicates: {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)")
        print(f"  Trade duplicates: {trade_dups:,} ({trade_dups/len(df)*100:.2f}%)")
        
        # MUST be < 1% after --clean flag
        assert exact_dups / len(df) < 0.01, "Too many duplicates! Old data not cleaned?"
```

---

## 2) Bronze ES Contract Filtering (REQUIRED)

**Critical**: Filter to front-month main contract before creating Silver tier.

### 2.1 Identify Dominant Symbol Per Date

```python
def get_main_contract_symbol(date):
    """Find the dominant ES contract symbol for a date (front month)."""
    bronze = Path("backend/data/lake/bronze")
    trades_files = list((bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"))
    
    # Sample files to find dominant symbol
    symbol_counts = {}
    for f in trades_files[:5]:  # Sample first 5 files
        df = pd.read_parquet(f)
        for symbol, count in df['symbol'].value_counts().items():
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + count
    
    # Get dominant symbol (should be ES_294973 or similar)
    main_symbol = max(symbol_counts, key=symbol_counts.get)
    volume_pct = symbol_counts[main_symbol] / sum(symbol_counts.values())
    
    print(f"{date}: Main contract = {main_symbol} ({volume_pct*100:.1f}% volume)")
    
    # Main contract should be 95%+ of volume
    assert volume_pct > 0.90, f"Main contract only {volume_pct*100:.1f}% - data issue?"
    
    return main_symbol
```

### 2.2 Create Filtered Bronze Files

**Options:**

**A) Filter In-Place** (Fast, overwrites Bronze):
```python
def filter_bronze_to_main_contract(date, main_symbol):
    """Filter Bronze Parquet files to main contract only."""
    bronze = Path("backend/data/lake/bronze")
    trades_path = bronze / f"futures/trades/symbol=ES/date={date}"
    
    for parquet_file in trades_path.rglob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        
        # Filter to main contract
        df_clean = df[df['symbol'] == main_symbol].copy()
        
        # Check impact
        removed = len(df) - len(df_clean)
        print(f"  {parquet_file.name}: removed {removed} non-main rows ({removed/len(df)*100:.1f}%)")
        
        # Overwrite with clean data
        df_clean.to_parquet(parquet_file, index=False, compression='zstd', compression_level=3)
```

**B) Create Bronze_Clean Tier** (Safer, keeps original):
```python
def create_bronze_clean(date, main_symbol):
    """Create bronze_clean tier with filtered data."""
    bronze = Path("backend/data/lake/bronze")
    bronze_clean = Path("backend/data/lake/bronze_clean")
    
    # Process trades
    for src_file in (bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"):
        df = pd.read_parquet(src_file)
        df_clean = df[df['symbol'] == main_symbol].copy()
        
        # Write to bronze_clean with same structure
        dst_file = bronze_clean / src_file.relative_to(bronze)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(dst_file, index=False, compression='zstd', compression_level=3)
```

**Recommendation**: Use Option A (in-place) since `--clean` already ran and we want corrected Bronze.

### 2.3 Validate Price Ranges Post-Filter

```python
def validate_price_range(date):
    """Verify price range after filtering (ES front month December 2025)."""
    bronze = Path("backend/data/lake/bronze")
    
    # Sample trades
    sample_file = list((bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"))[0]
    df = pd.read_parquet(sample_file)
    
    price_min = df['price'].min()
    price_max = df['price'].max()
    
    print(f"{date} price range: ${price_min:.2f} - ${price_max:.2f}")
    
    # December 2025 ES front month range: ~$6,800 - $6,950
    assert 6700 < price_min < 6900, f"Min price {price_min:.2f} out of expected range"
    assert 6800 < price_max < 7000, f"Max price {price_max:.2f} out of expected range"
    
    # SPY equivalent check
    spy_min = price_min / 10
    spy_max = price_max / 10
    print(f"  SPY equivalent: ${spy_min:.2f} - ${spy_max:.2f}")
    
    # SPY was ~$680-$695 in December 2025
    assert 670 < spy_min < 700, "SPY equivalent out of expected range"
    assert 680 < spy_max < 710, "SPY equivalent out of expected range"
```

---

## 3) Statistical Validation

### 3.1 ES Futures Trade Distributions

```python
def validate_trade_distributions(dates):
    """Statistical validation of trade characteristics."""
    
    for date in dates:
        bronze = Path("backend/data/lake/bronze")
        
        # Load all trades for date
        trades_files = list((bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in trades_files])
        
        print(f"\n{date} Trade Statistics:")
        print(f"  Total trades: {len(df):,}")
        print(f"  Price: μ={df['price'].mean():.2f}, σ={df['price'].std():.2f}")
        print(f"  Size: median={df['size'].median()}, p95={df['size'].quantile(0.95)}")
        print(f"  Time span: {(df['ts_event_ns'].max() - df['ts_event_ns'].min()) / 1e9 / 3600:.1f} hours")
        
        # Aggressor balance (should be ~50/50 buy/sell)
        aggressor_dist = df['aggressor'].value_counts(normalize=True)
        print(f"  Aggressor balance: {aggressor_dist.to_dict()}")
        
        # Check for suspicious patterns
        assert df['price'].std() > 1, "Price variance too low"
        assert 0.3 < aggressor_dist.get(1, 0) < 0.7, "Aggressor balance skewed"
        
        # Check time ordering
        assert df['ts_event_ns'].is_monotonic_increasing, "Time not sorted!"
```

### 3.2 MBP-10 Depth Quality

```python
def validate_mbp10_quality(date):
    """Validate MBP-10 depth characteristics."""
    
    bronze = Path("backend/data/lake/bronze")
    
    # Sample large file from busy hour
    mbp_files = list((bronze / f"futures/mbp10/symbol=ES/date={date}/hour=15").glob("*.parquet"))
    df = pd.read_parquet(mbp_files[0]) if mbp_files else None
    
    if df is not None:
        print(f"\n{date} MBP-10 Quality:")
        print(f"  Total snapshots: {len(df):,}")
        
        # Spread analysis
        spread = df['ask_px_1'] - df['bid_px_1']
        print(f"  Spread: median={spread.median():.2f}, p95={spread.quantile(0.95):.2f}")
        
        # Bid/ask validity
        valid_spread = (df['ask_px_1'] >= df['bid_px_1']).all()
        print(f"  Valid spread: {valid_spread}")
        
        # Level ordering (allow small violations at deep levels)
        bid_violations = sum((df[f'bid_px_{i}'] < df[f'bid_px_{i+1}']).sum() for i in range(1, 10))
        violation_pct = bid_violations / (len(df) * 9) * 100
        print(f"  Bid ordering violations: {violation_pct:.2f}%")
        
        assert valid_spread, "Invalid bid/ask spread detected"
        assert violation_pct < 1.0, "Too many level ordering violations"
```

### 3.3 Options Flow Characteristics

```python
def validate_options_flow(date):
    """Validate SPY options characteristics for gamma calculations."""
    
    bronze = Path("backend/data/lake/bronze")
    opts_file = list((bronze / f"options/trades/underlying=SPY/date={date}").glob("*.parquet"))[0]
    df = pd.read_parquet(opts_file)
    
    print(f"\n{date} Options Statistics:")
    print(f"  Total trades: {len(df):,}")
    print(f"  Unique strikes: {df['strike'].nunique()}")
    print(f"  Expirations: {df['exp_date'].unique()}")
    
    # Check for 0DTE dominance (same-day expiration)
    dte = (pd.to_datetime(df['exp_date']) - pd.to_datetime(date)).dt.days
    dte_dist = dte.value_counts().head()
    print(f"  DTE distribution:\n{dte_dist}")
    
    # Price sanity
    print(f"  Option price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Strike range should bracket spot
    strike_min = df['strike'].min()
    strike_max = df['strike'].max()
    print(f"  Strike range: ${strike_min:.2f} - ${strike_max:.2f}")
    
    # Size distribution
    print(f"  Size: median={df['size'].median()}, p95={df['size'].quantile(0.95)}")
    
    # Assertions
    assert dte_dist.iloc[0] / len(df) > 0.5, "0DTE should dominate volume"
    assert 20 < df['strike'].nunique() < 200, "Unusual strike count"
```

---

## 4) Cross-Data Consistency

### 4.1 Time Range Alignment

**Critical**: ES and Options must overlap for physics calculations.

```python
def validate_time_alignment(date):
    """Ensure ES futures and options have overlapping time ranges."""
    
    bronze = Path("backend/data/lake/bronze")
    
    # ES trades time range
    trades_files = list((bronze / f"futures/trades/symbol=ES/date={date}").rglob("*.parquet"))
    df_trades = pd.concat([pd.read_parquet(f) for f in trades_files])
    
    es_start = pd.to_datetime(df_trades['ts_event_ns'].min(), unit='ns')
    es_end = pd.to_datetime(df_trades['ts_event_ns'].max(), unit='ns')
    
    # Options time range
    opts_file = list((bronze / f"options/trades/underlying=SPY/date={date}").glob("*.parquet"))[0]
    df_opts = pd.read_parquet(opts_file)
    
    opts_start = pd.to_datetime(df_opts['ts_event_ns'].min(), unit='ns')
    opts_end = pd.to_datetime(df_opts['ts_event_ns'].max(), unit='ns')
    
    print(f"\n{date} Time Alignment:")
    print(f"  ES:      {es_start} to {es_end}")
    print(f"  Options: {opts_start} to {opts_end}")
    
    # Calculate overlap
    overlap_start = max(es_start, opts_start)
    overlap_end = min(es_end, opts_end)
    overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
    
    print(f"  Overlap: {overlap_hours:.1f} hours")
    
    # Must have at least 6 hours overlap (market hours)
    assert overlap_hours > 6, f"Insufficient overlap: {overlap_hours:.1f}h"
```

### 4.2 Price Correlation (ES vs SPY)

```python
def validate_es_spy_correlation(date):
    """Verify ES/SPY ~10:1 price relationship."""
    
    bronze = Path("backend/data/lake/bronze")
    
    # Sample ES midpoint from MBP-10
    mbp_file = list((bronze / f"futures/mbp10/symbol=ES/date={date}/hour=15").glob("*.parquet"))[0]
    df_mbp = pd.read_parquet(mbp_file)
    
    es_mid = (df_mbp['bid_px_1'] + df_mbp['ask_px_1']) / 2
    spy_implied = es_mid / 10
    
    # Sample SPY option strikes (should cluster around SPY price)
    opts_file = list((bronze / f"options/trades/underlying=SPY/date={date}").glob("*.parquet"))[0]
    df_opts = pd.read_parquet(opts_file)
    
    # ATM strike should be close to SPY price
    strike_median = df_opts['strike'].median()
    
    print(f"\n{date} ES/SPY Relationship:")
    print(f"  ES mid: ${es_mid.mean():.2f}")
    print(f"  SPY implied: ${spy_implied.mean():.2f}")
    print(f"  Options ATM strike: ${strike_median:.2f}")
    print(f"  Difference: ${abs(spy_implied.mean() - strike_median):.2f}")
    
    # Should be within $5
    assert abs(spy_implied.mean() - strike_median) < 5, "ES/SPY price mismatch"
```

---

## 5) Vectorized Pipeline Execution

**Once Bronze is validated and filtered**, run the vectorized pipeline to create Gold signals.

### 5.1 Run Pipeline for All Dates

```bash
cd backend/

# Run vectorized pipeline for all validated dates
uv run python -m src.pipeline.vectorized_pipeline \
  --dates 2025-12-01,2025-12-02,2025-12-03,2025-12-04,2025-12-05,2025-12-08,2025-12-09,2025-12-10,2025-12-11,2025-12-12,2025-12-15,2025-12-16,2025-12-17,2025-12-18,2025-12-19 \
  --output data/lake/gold/research/signals_vectorized.parquet \
  --verbose
```

**Expected Output:**
- Path: `data/lake/gold/research/signals_vectorized.parquet`
- Schema: Must match `backend/features.json`
- Contains: Level touch events with forward-looking labels (BREAK/BOUNCE)

### 5.2 Gold Dataset Validation

**Per VALIDATE.md original §3 & §4:**

```python
def validate_gold_dataset():
    """Comprehensive Gold dataset validation."""
    
    df = pd.read_parquet("data/lake/gold/research/signals_vectorized.parquet")
    
    print("="*70)
    print("GOLD DATASET VALIDATION")
    print("="*70)
    
    # Basic stats
    print(f"\nDataset size: {len(df):,} events")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {len(df.columns)}")
    
    # Outcome distribution
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())
    
    # Barrier state distribution
    print(f"\nBarrier state distribution:")
    print(df['barrier_state'].value_counts())
    
    # Fuel effect distribution
    print(f"\nFuel effect distribution:")
    print(df['fuel_effect'].value_counts())
    
    # Distance range (monitor band filter)
    print(f"\nDistance range: [{df['distance'].min():.3f}, {df['distance'].max():.3f}]")
    
    # Non-zero features
    print(f"\nFeature coverage:")
    print(f"  tape_velocity non-zero: {(df['tape_velocity'] != 0).mean()*100:.1f}%")
    print(f"  gamma_exposure non-zero: {(df['gamma_exposure'] != 0).mean()*100:.1f}%")
    
    # Mathematical invariants (VALIDATE.md §4)
    print(f"\nMathematical Invariants:")
    
    # Confirm time = ts + 240s (4 minutes)
    confirm_delta = (df['confirm_ts_ns'] - df['ts_ns']) / 1e9
    print(f"  Confirm delta: {confirm_delta.mean():.1f}s ± {confirm_delta.std():.1f}s")
    assert abs(confirm_delta.mean() - 240.0) < 1.0, f"Confirmation time != 240s: {confirm_delta.mean():.1f}s"
    
    # Distance = |level_price - spot|
    distance_calc = (df['level_price'] - df['spot']).abs()
    distance_match = np.allclose(df['distance'], distance_calc, atol=0.01)
    print(f"  Distance geometry: {distance_match}")
    
    # Direction consistency
    up_valid = df[df['direction'] == 'UP']['distance_signed'].ge(-0.01).all()
    down_valid = df[df['direction'] == 'DOWN']['distance_signed'].le(0.01).all()
    print(f"  Direction consistency: UP={up_valid}, DOWN={down_valid}")
    
    # Tradeable consistency
    t1_match = ((df['tradeable_1'] == 1) == df['time_to_threshold_1'].notna()).mean()
    t2_match = ((df['tradeable_2'] == 1) == df['time_to_threshold_2'].notna()).mean()
    print(f"  Tradeable consistency: t1={t1_match:.1%}, t2={t2_match:.1%}")
    
    # Assertions
    assert len(df) > 10000, "Too few events - data missing?"
    assert df['outcome'].value_counts().get('BREAK', 0) > 0, "No BREAK outcomes"
    assert df['outcome'].value_counts().get('BOUNCE', 0) > 0, "No BOUNCE outcomes"
    assert (df['gamma_exposure'] != 0).mean() > 0.9, "Missing gamma data"
    assert (df['tape_velocity'] != 0).mean() > 0.9, "Missing tape data"
    assert distance_match, "Distance calculation error"
    assert up_valid and down_valid, "Direction logic error"
    
    print("\n✅ GOLD DATASET VALIDATED")
```

---

## 6) Final Checklist

Before declaring the pipeline ready for ML training:

### Bronze Tier
- [ ] All 15 trading days have trades, MBP-10, and options
- [ ] ES contracts filtered to front-month main contract only
- [ ] No exact duplicates (< 0.1% duplicate rate)
- [ ] Price ranges correct: ES $6,800-$6,950, SPY $680-$695
- [ ] Time-sorted within each Parquet file
- [ ] Row counts reasonable (no 4x inflation from old runs)

### Statistical Validation
- [ ] Trade aggressor balance ~50/50
- [ ] MBP-10 spread < 1.0 at p95
- [ ] Bid/ask spread always valid
- [ ] Options 0DTE > 50% volume
- [ ] ES/SPY price correlation within $5

### Cross-Data Validation
- [ ] ES and Options time ranges overlap > 6 hours per day
- [ ] All dates have both ES and Options (except 12-14)
- [ ] No missing hours during market session (9:30-16:00 ET)

### Gold Dataset
- [ ] Output file exists: `data/lake/gold/research/signals_vectorized.parquet`
- [ ] Schema matches `backend/features.json`
- [ ] > 10,000 labeled events
- [ ] BREAK and BOUNCE outcomes both present (not all CHOP)
- [ ] Barrier states: NEUTRAL majority, VACUUM/WALL present
- [ ] Fuel effects: DAMPEN majority, AMPLIFY present
- [ ] Distance all in [0, 0.50] range (monitor band)
- [ ] tape_velocity > 90% non-zero
- [ ] gamma_exposure > 90% non-zero
- [ ] Mathematical invariants pass (§4)

### Final Sign-Off

Once all checks pass:

```python
print("="*70)
print("🎯 PIPELINE VALIDATION COMPLETE")
print("="*70)
print("\nDataset Ready for ML Training:")
print(f"  Bronze: 15 days, {total_trades:,} trades, {total_mbp10:,} MBP-10 snapshots")
print(f"  Options: 15 days, {total_options:,} SPY 0DTE trades")
print(f"  Gold: {len(gold_df):,} labeled level touch events")
print(f"  Features: {len(gold_df.columns)} columns")
print(f"  Outcome balance: BREAK={break_pct:.1%}, BOUNCE={bounce_pct:.1%}")
print("\n✅ READY FOR ML TRAINING")
```

---

## 7) Troubleshooting Guide

### Issue: Price range still shows $58-$6,900

**Cause**: ES contract filtering (Step 2) not completed  
**Fix**: Run `filter_bronze_to_main_contract()` for all dates

### Issue: gamma_exposure all zeros

**Cause**: Options data not ingested or missing for date  
**Fix**: Verify `bronze/options/trades/underlying=SPY/date=X` exists, re-run flat file download if missing

### Issue: > 100k events but all UNDEFINED outcome

**Cause**: Forward window incomplete - not enough future data for labeling  
**Fix**: Check OHLCV building includes full 5-minute forward window

### Issue: Distance > 0.50

**Cause**: Monitor band filter not applied  
**Fix**: Verify vectorized pipeline uses `CONFIG.MONITOR_BAND=0.50` and filters `distance <= 0.50`

### Issue: Duplicates still present

**Cause**: Old Bronze data not cleaned before new backfill  
**Fix**: Re-run backfill with `--clean` flag to delete existing data first

---

## 8) Web Search Verification (If Needed)

If statistical distributions seem unusual, search for market context:

**December 2025 Market Context:**
- SPY price range: $680-$695
- VIX levels: Check if high volatility period
- Market holidays: 12-06, 12-07 (weekends), 12-13 (missing data), 12-25 (Christmas)
- 0DTE options: Explosive growth in 2024-2025, should dominate SPY volume

**Known Market Events:**
- Search: "SPY December 2025 price"
- Search: "VIX December 2025"
- Search: "Market holidays December 2025"

---

## Summary

This validation protocol ensures:
1. ✅ Bronze data is clean (no duplicates, correct contracts)
2. ✅ Statistical distributions are normal (no corruption)
3. ✅ ES/SPY/Options data aligns temporally
4. ✅ Gold dataset has complete physics features
5. ✅ Mathematical invariants hold (no labeling errors)
6. ✅ Dataset is balanced and ready for ML

**AI Agent**: Execute steps 1-6 sequentially. Stop at first assertion failure and report issue.
