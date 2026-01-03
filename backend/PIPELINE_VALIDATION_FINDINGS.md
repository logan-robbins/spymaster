# Pipeline Validation Findings

**Date Range**: June 11-13, 2025  
**Version**: 4.7.0 (MBP-10 w/ action/side for true OFI)
**Methodology**: Stage-by-stage validation with research-grade quantitative analysis


## COMMAND EXAMPLE ALWAYS 

```bash
cd /Users/loganrobbins/research/qmachina/spymaster/backend && nohup uv run python -m scripts.run_pipeline \
  --pipeline bronze_to_silver \
  --start 2025-06-11 \
  --end 2025-06-13 \
  --workers 3 \
  --checkpoint-dir data/checkpoints \
  --canonical-version 4.0.0 \
  --resume-from-stage 7 \
  --stop-at-stage 7 \
  --write-outputs \
  > logs/stage7_velocity_fix.log 2>&1 &
  ```
---

## Stage 0-2 Consolidated Analysis: 2025-06-11 (pm_high)

**Status**: ✅ PASSED (with warnings)  
**Version**: 4.7.0 - Unified MBP-10 source w/ action/side for true OFI

---

### 1. TRADES (from MBP-10 action='T')

| Metric | Value | Status |
|--------|-------|--------|
| **Total Trades** | 478,211 | ✅ |
| Price Range | 6006.25 - 6221.50 | ✅ |
| Mean Price | 6042.31 | ✅ |
| **Nulls** | 0 (0.00%) | ✅ |
| **Zeros** | 0 (0.00%) | ✅ |
| Aggressor | Buy: 49.8%, Sell: 49.2%, Mid: 0.0% | ✅ |
| **Size Outliers** | 24,613 (5.15%) | ⚠️ Large blocks |
| Price Outliers (3×IQR) | 24 (0.01%) | ✅ |

**✅ VALIDATION**: Trades successfully extracted from MBP-10 `action='T'` events. Separate trades schema no longer needed.

---

### 2. MBP-10 SNAPSHOTS (with action/side for OFI)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Snapshots** | 145,878 (downsampled) | ✅ |
| Coverage | 12.1 hours (04:00-16:00 ET) | ✅ |

**Event Action Distribution:**
| Action | Count | % | Meaning |
|--------|-------|---|---------|
| A (Add) | 76,633 | 52.5% | New order placed |
| C (Cancel) | 51,456 | 35.3% | Order removed |
| M (Modify) | 17,398 | 11.9% | Order changed |
| T (Trade) | 391 | 0.3% | Trade executed (downsampled) |

**Side Distribution:**
| Side | Count | % | Meaning |
|------|-------|---|---------|
| A (Ask) | 73,034 | 50.1% | Sell-side event |
| B (Bid) | 72,827 | 49.9% | Buy-side event |
| N (None) | 17 | 0.0% | No side (e.g., clear) |

**Book State Quality:**
- Best bid range: 6006.25 - 6220.25 ✅
- Best ask range: 6006.50 - 6271.00 ✅
- Nulls: 0 ✅

**⚠️ WARNING: action_price contamination**
- Range: 4947.25 - 6567.50 (includes MES/spread prices <6000)
- Impact: Downsampled snapshots still contain some contamination
- Mitigation: Trades are filtered (price 3000-10000) in extraction query

---

### 3. OHLCV DATA (Hierarchical Resampling: trades → 10s → 1min → 2min)

#### 10-Second OHLCV
| Metric | Value | Status |
|--------|-------|--------|
| Bars | 4,249 | ✅ |
| Coverage | 07:56 - 20:03 UTC (12.1 hours) | ✅ |
| Price range | Open: 6007-6139, High: 6018-6222, Low: 6006-6090, Close: 6007-6180 | ✅ |
| Volume range | 1 - 25,902 per bar | ✅ |
| **Nulls** | 0 | ✅ |
| **Zeros** | 0 | ✅ |

#### 1-Minute OHLCV
| Metric | Value | Status |
|--------|-------|--------|
| Bars | 728 | ✅ |
| Coverage | 07:56 - 20:03 UTC | ✅ |
| Price range | 6006-6222 | ✅ |
| Volume | 12 - 47,765 per bar (mean: 1,937) | ✅ |
| **ATR** | 34.75 - 82.27 (mean: 56.94) | ✅ |
| **Volatility** | 0.00 - 41.10 (mean: 14.50) | ✅ |

#### 2-Minute OHLCV (with warmup)
| Metric | Value | Status |
|--------|-------|--------|
| Bars | 1,054 (includes warmup) | ✅ |
| Coverage | **2025-06-10 00:00** - 2025-06-11 20:02 UTC | ✅ |
| Price range | 5991-6222 | ✅ |
| Volume | 100 - 128,184 per bar (mean: 4,270) | ✅ |

**✅ VALIDATION**: OHLCV hierarchical resampling working correctly. Warmup data included for 2min bars.

---

### 4. OPTIONS DATA

| Metric | Value | Status |
|--------|-------|--------|
| **Total Trades** | 4,400 | ✅ |
| Coverage | 11:06 - 06:05 UTC (19+ hours, includes overnight) | ✅ |
| Strike Range | 4175 - 6800 (625 points) | ✅ |
| Mean Strike | 5890.45 | ✅ |
| Right | Put: 56.5%, Call: 43.5% | ✅ |
| **Delta** | -1.00 to 1.00 (mean: 0.05) | ✅ |
| Delta = 0 | 779 (17.7%) | ⚠️ Deep OTM |
| **Gamma** | 0.00 to 0.025 (mean: 0.0027) | ✅ |
| Gamma = 0 | 262 (5.95%) | ⚠️ Deep OTM |

**⚠️ WARNING**: 17.7% of options have delta=0 (deep out-of-the-money). This is expected for 0DTE options far from spot.

---

### 5. MARKET STATE

| Metric | Value | Status |
|--------|-------|--------|
| **Spot Price** | 6026.25 | ✅ |

---

## ISSUES FOUND

### ⚠️ WARNING: MBP-10 action_price Contamination
- **Issue**: Downsampled MBP-10 snapshots include prices 4947-6567 (outside valid ES range)
- **Root Cause**: Downsampling preserves contaminated records from spreads/MES
- **Impact**: **NONE** on current features - trades are filtered during extraction
- **Future Risk**: Will affect true OFI computation if not addressed
- **Mitigation**: Add price filter to `read_futures_mbp10_downsampled` query

### ⚠️ INFO: Options Deep OTM
- **Issue**: 17.7% delta=0, 5.95% gamma=0
- **Root Cause**: 0DTE options far from spot (strikes 4175-6800 vs spot 6026)
- **Impact**: Expected behavior, no action needed
- **Note**: GEX features should handle this gracefully

---

## COVERAGE SUMMARY

| Data Type | Records | Coverage | Quality |
|-----------|---------|----------|---------|
| Trades (from MBP-10) | 478,211 | 12.1 hours | ✅ 100% |
| MBP-10 (action/side) | 145,878 | 12.1 hours | ✅ 100% |
| OHLCV 10s | 4,249 | 12.1 hours | ✅ 100% |
| OHLCV 1min | 728 | 12.1 hours | ✅ 100% |
| OHLCV 2min | 1,054 | 44.0 hours (w/ warmup) | ✅ 100% |
| Options | 4,400 | 19+ hours | ✅ 100% |

**✅ ALL DATA QUALITY CHECKS PASSED**

---

