# Pipeline Stage Validation Analysis

> **Audience**: Senior Data Scientists evaluating pipeline integrity  
> **Goal**: Understand what each validation catches, assess thresholds against market reality, identify genuine bugs vs. false positives

---

## Validation Philosophy

Each stage validation should answer:
1. **What is being computed?** (Feature engineering goal)
2. **What invariants must hold?** (Hard failures = propagate errors)
3. **What is market-dependent?** (Soft warnings = regime-specific behavior)
4. **What threshold is realistic?** (Tune to actual ES market microstructure)

---

## Stage 0: LoadBronze

**Purpose**: Load raw Bronze data (ES futures trades, MBP-10 depth, ES 0DTE options)

### Critical Checks (Errors - Must Pass)
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **Required outputs present** | Missing data = entire pipeline fails | 🔴 FATAL |
| **Non-empty DataFrames** | Zero rows = no features possible | 🔴 FATAL |
| **Schema compliance** | Missing columns = downstream crashes | 🔴 FATAL |
| **Timestamp monotonicity** | Non-sorted = OHLCV bars wrong | 🔴 HIGH |
| **Negative sizes** | Data corruption indicator | 🔴 HIGH |

### Soft Checks (Warnings - Review)
| Check | Current Threshold | Market Reality | Assessment |
|-------|-------------------|----------------|------------|
| **Price range** | 3000 < price < 10000 | ES ~5000-6000 in 2025 | ⚠️ **TUNE**: 50.90 min is DATA BUG |
| **Front-month purity** | >95% dominant contract | Roll dates have 2 contracts | ✅ Reasonable (95% good) |
| **MBP-10 duration** | "Short" if <6h | Databento gives RTH only sometimes | ⚠️ **REVIEW**: May be normal |
| **0DTE expiry** | Flags non-0DTE | We want 0DTE only | ✅ Correct intent |

### Findings (2025-06-11)
```json
warnings: [
  "Unusual price range: 50.90 - 6221.50",  // 🔴 BUG: 50.90 is impossible for ES
  "Short MBP-10 duration: 4.13 hours",      // ⚠️  NORMAL: Databento RTH windowing
  "Non-0DTE options found: ['2025-06-11']"  // ❓ REVIEW: Same-day expiry = 0DTE
]
```

**Action Items**:
- [ ] Investigate 50.90 price - likely bad tick in Bronze or Databento artifact
- [ ] Relax MBP-10 duration check (RTH = 3.5-6.5h is normal)
- [ ] Fix 0DTE logic (expiry == session_date IS 0DTE)

---

## Stage 1: BuildOHLCV (1min)

**Purpose**: Aggregate tick data into 1min OHLCV bars for ATR computation

### Critical Checks (Errors)
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **Bar count >0** | No bars = no ATR = Stage 14 fails | 🔴 HIGH |
| **Timestamp gaps <10min** | Gaps = missing market hours | 🟡 MEDIUM |
| **OHLC consistency** | L<=O,C<=H = data integrity | 🔴 FATAL |
| **All bars have volume** | Vol=0 = bad aggregation | 🔴 HIGH |
| **No NaN in OHLCV** | NaN propagates to all features | 🔴 FATAL |

### Soft Checks (Warnings)
| Check | Current Threshold | Market Reality | Assessment |
|-------|-------------------|----------------|------------|
| **Starts at 18:00 ET** | Premarket begins 18:00 | Databento may start later | ⚠️ Soften to 20:00 |
| **ATR range** | "reasonable" (no bounds) | ES ATR ~30-80 points | ✅ Add explicit bounds |

### Findings (2025-06-11)
```
Bars: 1380
Duration: 18:00 ET -> 17:00 ET (23h coverage) ✅
ATR: mean=53.46, std=8.79, range=[12.09, 82.27] ✅
Warning: "Missing early premarket: starts at 20:00 ET"
```

**Assessment**: ✅ **PASS** - ATR looks healthy, premarket warning is acceptable

---

## Stage 4: InitMarketState

**Purpose**: Initialize market state with Greeks, active contract detection, option aggregation

### Critical Checks (Errors)
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **Active contract detected** | No contract = no Greeks | 🔴 FATAL |
| **Option strikes parsed** | Bad strikes = wrong gamma | 🔴 HIGH |
| **Expiry dates valid** | Wrong expiry = GEX mismatch | 🔴 HIGH |

### Soft Checks (Warnings)
| Check | Current Threshold | Market Reality | Assessment |
|-------|-------------------|----------------|------------|
| **Option count** | "Low" if <1000 | Varies by day/volume | ⚠️ Make threshold dynamic |
| **Strike spacing** | Expects 5-pt | ES uses 5pt for 0DTE | ✅ Correct |

**Action Items**:
- [ ] Validate that "option_flows" aggregation works correctly
- [ ] Check if call/put ratio is within reasonable bounds (30-70%)

---

## Stage 7: ComputePhysics

**Purpose**: Compute barrier states, tape metrics, fuel effects, **Market Tide**

### Critical Checks (Errors)
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **Barrier states assigned** | Used in feature vector | 🟡 MEDIUM |
| **No NaN in barrier_delta_liq** | Core barrier metric | 🔴 HIGH |
| **Tide features computed** | Call/put tide for v4.0.0 | 🟡 MEDIUM |

### Market-Specific Considerations
| Feature | Expected Sparsity | Why | Assessment |
|---------|-------------------|-----|------------|
| **call_tide** | ~5-15% nonzero | Options only traded near-the-money | ✅ NORMAL |
| **put_tide** | ~5-15% nonzero | Same as call_tide | ✅ NORMAL |
| **barrier_state=WALL** | ~5-20% of touches | Walls are rare events | ✅ NORMAL |
| **fuel_effect≠NEUTRAL** | Very rare | Explosions uncommon | ⚠️ **REVIEW** if always NEUTRAL |

**Findings (2025-06-11)**:
```
Barrier states: {'NEUTRAL': 438, 'WEAK': 40, 'WALL': 6, 'VACUUM': 2}
Fuel effects: {'NEUTRAL': 486}  // 🔴 ALL NEUTRAL - is this a bug?
```

**Action Items**:
- [ ] Investigate why fuel_effect is 100% NEUTRAL (v4.0.0 Silver shows same)
- [ ] Check if fuel_effect logic is disabled or thresholds too strict

---

## Stage 14: ComputeForceMass

**Purpose**: F=ma validation features (force/mass ratio, predicted accel)

### Critical Checks
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **force_proxy computed** | Core microstructure signal | 🟡 MEDIUM |
| **mass_proxy nonzero** | Denominator in ratio | 🔴 HIGH (div by zero) |
| **Residuals finite** | Regression quality check | 🟡 LOW |

**Action Items**:
- [ ] Validate that force_mass_ratio doesn't have infs (from zero mass)
- [ ] Check if residuals correlate with actual outcomes

---

## Stage 16: LabelOutcomes

**Purpose**: First-crossing labels (BREAK/REJECT/CHOP) with ATR-normalized thresholds

### Critical Checks (Errors)
| Check | Why It Matters | Propagation Risk |
|-------|----------------|------------------|
| **Outcome distribution** | All REJECT = threshold too high | 🔴 FATAL (useless labels) |
| **No NaN in time_to_break** | Used for filtering | 🟡 MEDIUM |
| **Excursion values finite** | Core outcome metric | 🔴 HIGH |

### Distribution Sanity
| Metric | Healthy Range | Red Flag | Assessment |
|--------|---------------|----------|------------|
| **REJECT %** | 60-90% | >95% = threshold broken | ✅ 78% is good |
| **BREAK %** | 10-40% | <5% = no signal | ✅ 22% is good |
| **CHOP %** | 0-10% | >20% = indecision | ✅ <5% is ideal |

**Findings (2025-06-11)**:
```
4min: {'REJECT': 381 (78%), 'BREAK': 105 (22%)}  ✅ HEALTHY
8min: {'REJECT': 381 (78%), 'BREAK': 105 (22%)}  ✅ HEALTHY
```

**Assessment**: ✅ **EXCELLENT** - Label distribution is ideal for training

---

## Summary: What to Mark COMPLETE

A stage is **COMPLETE** when:
1. ✅ Validation script runs without errors
2. ✅ Critical checks pass (no propagation risk)
3. ✅ Warnings are understood and documented as "normal market behavior"
4. ✅ Thresholds are tuned to ES microstructure reality

### Current Status (2025-06-11)
- **Stage 0**: ⚠️ NEEDS REVIEW (50.90 price bug)
- **Stage 1**: ✅ READY TO MARK COMPLETE
- **Stage 7**: ⚠️ NEEDS REVIEW (fuel_effect always NEUTRAL)
- **Stage 14**: ❓ NEEDS VALIDATION RUN
- **Stage 16**: ✅ READY TO MARK COMPLETE

---

## Next Steps

1. Run all 18 validation scripts for 2025-06-11
2. Document findings in this file
3. Fix genuine bugs (e.g., 50.90 price)
4. Relax false-positive warnings (e.g., MBP-10 duration)
5. Mark stages COMPLETE only after senior review

