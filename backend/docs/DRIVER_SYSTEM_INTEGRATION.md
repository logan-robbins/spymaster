# Driver Attribution System - Integration Guide

**Created**: 2025-12-31  
**Status**: Implementation Ready  
**Goal**: Transform 144D ML predictions into actionable trader intelligence

---

## System Overview

This system adds **explainability and context** to the Pentaview level break/reject predictions. Instead of showing traders a raw probability, it explains:

1. **WHY** a level will break or bounce (driver decomposition)
2. **WHEN** predictions are most reliable (time stratification)
3. **WHICH LEVELS** behave differently (level-specific analysis)
4. **WHAT COMBOS** create edge (feature interactions)
5. **HOW TO DISPLAY** this to traders (dashboard spec)

---

## Components Built

### 1. Driver Attribution Module
**File**: `backend/src/ml/driver_attribution.py`

**Purpose**: Decomposes 144D predictions into trader-understandable drivers

**Key Classes**:
- `DriverAttributor`: Main attribution engine
  - Uses feature importance + ablation to quantify contributions
  - Maps technical features to trader language
  - Computes percentiles vs historical distribution

- `DriverAttribution`: Attribution result
  - `break_drivers`: Top factors pushing toward BREAK
  - `reject_drivers`: Top factors pushing toward REJECT
  - `section_contributions`: Physics domain importance
  - `confidence`, `edge`: Risk management info

**Usage**:
```python
from src.ml.driver_attribution import DriverAttributor

attributor = DriverAttributor(
    model=trained_model,
    scaler=scaler,
    base_rate=0.455,
    historical_stats=stats_dict
)

attribution = attributor.explain(episode_vector_144d)

# Output
print(f"P(BREAK): {attribution.p_break:.1%}")
print(f"Edge: {attribution.edge:+.1%}")
print(f"Top BREAK driver: {attribution.break_drivers[0].name}")
```

**Demo Script**: `scripts/demo_driver_attribution.py`

---

### 2. Time Stratification Analysis
**File**: `backend/scripts/analyze_time_stratification.py`

**Purpose**: Determine if different time buckets need different models/thresholds

**Key Analyses**:
1. **Base Rate by Time**: Does BREAK rate vary by time bucket?
2. **Model Performance**: Do time-specific models outperform global model?
3. **Feature Consistency**: Which features matter across all times?
4. **Optimal Thresholds**: Time-adjusted decision thresholds

**Output**: `data/ml/time_stratification_analysis.json`

**Key Findings** (from research):
- T30_60 has highest BREAK rate (55% vs 45% overall)
- T0_15 (market open) is choppiest, lower predictability
- Optimal thresholds range from 0.50 to 0.65 depending on time

**Recommendation Logic**:
- If time-specific AUC improvement > 5%: Use separate models
- If improvement 2-5%: Use time-adjusted thresholds
- If improvement < 2%: Use global model

**Usage**:
```bash
cd backend
uv run python -m scripts.analyze_time_stratification \
  --version v4.0.0 \
  --start 2025-11-03 \
  --end 2025-12-19 \
  --horizon 4min \
  --output-json data/ml/time_stratification_analysis.json
```

---

### 3. Level Specificity Analysis
**File**: `backend/scripts/analyze_level_specificity.py`

**Purpose**: Determine if OR_LOW and PM_LOW need separate treatment

**Key Analyses**:
1. **Base Rate Comparison**: Different BREAK rates?
2. **Feature Distributions**: Do features differ between levels?
3. **Model Performance**: Level-specific vs unified model
4. **Sequence Effects**: Does "first touch of day" matter?

**Output**: `data/ml/level_specificity_analysis.json`

**Key Questions Answered**:
- Is PM (first level) more/less predictable than OR (second level)?
- Do different features matter for each?
- Should threshold be adjusted by level type?

**Usage**:
```bash
cd backend
uv run python -m scripts.analyze_level_specificity \
  --version v4.0.0 \
  --start 2025-11-03 \
  --end 2025-12-19 \
  --horizon 4min \
  --output-json data/ml/level_specificity_analysis.json
```

---

### 4. Feature Interaction Discovery
**File**: `backend/scripts/discover_feature_interactions.py`

**Purpose**: Quantify combo effects (e.g., sigma_s × proximity)

**Key Analyses**:
1. **Hypothesized Interactions**: Test specific combos from research
   - sigma_s × proximity: "Close + Clean Setup"
   - sigma_d × gamma_exposure: "Dealer positioning alignment"
   - ofi_acceleration × barrier_delta: "Flow meets weak barrier"

2. **Discovery Mode**: Test all pairwise combinations (optional, slow)

3. **Interaction Patterns**: 2×2 grid analysis showing how BREAK rate varies

**Output**: `data/ml/feature_interactions.json`

**Interaction Format**:
```json
{
  "sigma_s_x_proximity": {
    "description": "Clean Setup × Close Approach",
    "auc_main_effects": 0.722,
    "auc_with_interaction": 0.765,
    "improvement": 0.043,
    "interaction_coefficient": 0.237,
    "p_value": 0.003,
    "significant": true
  }
}
```

**Usage**:
```bash
cd backend

# Test hypothesized interactions (fast)
uv run python -m scripts.discover_feature_interactions \
  --version v4.0.0 \
  --output-json data/ml/feature_interactions.json

# Discover all interactions (slow)
uv run python -m scripts.discover_feature_interactions \
  --version v4.0.0 \
  --discover-all \
  --output-json data/ml/feature_interactions.json
```

---

### 5. Driver Dashboard UI Specification

**Purpose**: Real-time trader interface showing WHY a level will break or bounce

#### Layout Design

```
┌────────────────────────────────────────────────────────────┐
│ LEVEL: OR_LOW @ 5985.25                                   │
│ DIRECTION: ↑ UP (Approaching from below)                  │
│ DISTANCE: 0.43 ATR (Close)                                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  MODEL PREDICTION                                          │
│  ┌────────────────────────────────────┐                   │
│  │  BREAK: 67%  ████████████░░░░░     │                   │
│  │  REJECT: 33%  ████░░░░░░░░░░░░░    │                   │
│  └────────────────────────────────────┘                   │
│                                                            │
│  Edge: +22% above base rate (45%)                         │
│  Confidence: HIGH                                          │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  BREAK DRIVERS (Pushing Up)                               │
│  ┌────────────────────────────────────┐                   │
│  │ ✓ Clean Setup            ████████░░ 85%  +22%         │
│  │ ✓ Flow Acceleration      ███████░░░ 72%  +15%         │
│  │ ✓ Weak Barrier Above     ██░░░░░░░░ 22%  +12%         │
│  │ ✓ Dealer Short Gamma     ████████░░ 78%  +8%          │
│  └────────────────────────────────────┘                   │
│                                                            │
│  REJECT DRIVERS (Pushing Down)                            │
│  ┌────────────────────────────────────┐                   │
│  │ ✗ 4th Touch Attempt      ████░░░░░░ 35%  -5%          │
│  │ ✗ Gamma Wall Below       ███░░░░░░░ 28%  -3%          │
│  └────────────────────────────────────┘                   │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  KEY COMBOS                                                │
│  • Close + Clean Setup: Strong synergy (+18% edge)        │
│  • Dealer positioning aligned with flow                   │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  CONTEXT                                                   │
│  Time: T30_60 (Peak breakout window)                      │
│  Level Type: OR_LOW (2nd level of day)                    │
│  Today's PM_LOW: Rejected at 09:47                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Component Specifications

**1. Level Header**
- Level Type: `PM_HIGH | PM_LOW | OR_HIGH | OR_LOW`
- Price: Actual level price
- Direction: `↑ UP` or `↓ DOWN` with arrow
- Distance: Current distance in ATR with label (Very Close / Close / Approaching)

**2. Prediction Display**
- Horizontal bar chart: BREAK % (blue) vs REJECT % (red)
- Edge calculation: `(p_break - base_rate) * 100`
- Confidence badge: `VERY HIGH | HIGH | MODERATE | LOW`

**3. Driver Lists**
Each driver shows:
- Icon: ✓ (BREAK) or ✗ (REJECT)
- Name: Trader-friendly (not technical)
- Strength Bar: Visual 0-100% meter
- Percentile: Where current value sits vs historical
- Edge Contribution: Marginal probability added/subtracted

**Driver Name Mapping**:
```python
{
    'sigma_s': 'Clean Setup',
    'ofi_acceleration': 'Flow Acceleration',
    'barrier_delta_liq': 'Barrier Weakness',
    'gamma_exposure': 'Dealer Short Gamma',
    'proximity': 'Very Close Approach',
    'level_stacking_5pt': 'Level Confluence'
}
```

**4. Key Combos**
- 1-3 bullet points showing active interaction effects
- "Close + Clean Setup: Strong synergy (+18% edge)"

**5. Context Panel**
- Time bucket with label (e.g., "T30_60: Peak Breakout Window")
- Level sequence (1st/2nd support of day)
- Session history (prior level outcomes today)

#### WebSocket Message Format

```json
{
  "type": "level_signal",
  "timestamp": "2025-12-31T10:15:30Z",
  "level": {
    "kind": "OR_LOW",
    "price": 5985.25,
    "direction": "UP",
    "distance_atr": 0.43
  },
  "prediction": {
    "p_break": 0.67,
    "p_reject": 0.33,
    "base_rate": 0.455,
    "edge": 0.22,
    "confidence": "HIGH",
    "method": "model"
  },
  "drivers": {
    "break": [
      {
        "name": "Clean Setup",
        "strength": 85,
        "edge": 0.22,
        "description": "Strong setup quality (top 15%)"
      }
    ],
    "reject": [
      {
        "name": "4th Touch Attempt",
        "strength": 35,
        "edge": -0.05,
        "description": "Multiple prior touches"
      }
    ]
  },
  "combos": [
    {
      "description": "Close + Clean Setup",
      "edge": 0.18,
      "active": true
    }
  ],
  "context": {
    "time_bucket": "T30_60",
    "time_bucket_label": "Peak Breakout Window",
    "level_sequence": "2nd support of day"
  }
}
```

---

## Integration Workflow

### Phase 1: Analysis & Model Training

```bash
cd backend

# 1. Run time stratification analysis
uv run python -m scripts.analyze_time_stratification \
  --version v4.0.0 --horizon 4min \
  --output-json data/ml/time_stratification.json

# 2. Run level specificity analysis
uv run python -m scripts.analyze_level_specificity \
  --version v4.0.0 --horizon 4min \
  --output-json data/ml/level_specificity.json

# 3. Discover feature interactions
uv run python -m scripts.discover_feature_interactions \
  --version v4.0.0 --horizon 4min \
  --output-json data/ml/feature_interactions.json

# 4. Train 144D model (if not already done)
# See PENTAVIEW_RESEARCH.md "Train New Model" section
```

**Outputs**:
- `data/ml/time_stratification.json`
- `data/ml/level_specificity.json`
- `data/ml/feature_interactions.json`
- `data/ml/break_predictor_144d.joblib` (trained model)

---

### Phase 2: Historical Stats Pre-Computation

**Purpose**: Driver percentiles require historical feature distributions

**Script** (to be created):
```python
# scripts/compute_historical_stats.py

from src.ml.driver_attribution import compute_historical_stats
from src.ml.episode_vector import construct_episodes_from_events
import joblib

# Load all historical episodes
episodes_df = load_all_episodes('v4.0.0')

# Construct vectors
vectors = construct_episode_vectors(episodes_df)

# Compute stats
stats = compute_historical_stats(vectors)

# Save
joblib.dump(stats, 'data/ml/historical_stats_v4.0.0.joblib')
```

**Output**: `data/ml/historical_stats_v4.0.0.joblib`

**Update Schedule**: Daily or weekly

---

### Phase 3: Real-Time Attribution Integration

**Location**: `backend/src/gateway/socket_broadcaster.py`

**Pseudocode**:
```python
from src.ml.driver_attribution import DriverAttributor
import joblib

# Load at startup
model_artifact = joblib.load('data/ml/break_predictor_144d.joblib')
historical_stats = joblib.load('data/ml/historical_stats_v4.0.0.joblib')

attributor = DriverAttributor(
    model=model_artifact['model'],
    scaler=model_artifact['scaler'],
    base_rate=0.455,
    historical_stats=historical_stats
)

def on_state_update(level_signal):
    """Called every 30s for each active level."""
    
    # Check if in approach zone
    if level_signal.distance_atr > 2.0:
        return  # Too far, don't generate signal
    
    # Construct episode vector
    episode_vector = construct_episode_vector(
        current_bar=level_signal.current_state,
        trajectory_window=level_signal.history_20min
    )
    
    # Generate attribution
    attribution = attributor.explain(episode_vector)
    
    # Format for frontend
    message = {
        'type': 'level_signal',
        'level': {
            'kind': level_signal.level_kind,
            'price': level_signal.level_price,
            'direction': level_signal.direction,
            'distance_atr': level_signal.distance_atr
        },
        'prediction': {
            'p_break': attribution.p_break,
            'edge': attribution.edge,
            'confidence': attribution.confidence
        },
        'drivers': attribution.to_dict()['break_drivers' | 'reject_drivers'],
        'section_contributions': attribution.section_contributions
    }
    
    # Broadcast via WebSocket
    broadcast_to_clients(message)
```

---

### Phase 4: Frontend Implementation

**Component**: `frontend/src/app/components/driver-dashboard/`

**Files**:
- `driver-dashboard.component.ts`
- `driver-dashboard.component.html`
- `driver-dashboard.component.css`

**WebSocket Subscription**:
```typescript
export class DriverDashboardComponent {
  levelSignal$: Observable<LevelSignal>;
  
  ngOnInit() {
    this.levelSignal$ = this.websocketService
      .messages$
      .pipe(filter(msg => msg.type === 'level_signal'));
  }
}
```

**Template**:
```html
<div class="driver-dashboard" *ngIf="levelSignal$ | async as signal">
  <!-- Prediction Bar -->
  <div class="prediction">
    <div class="break-bar" [style.width.%]="signal.prediction.p_break * 100">
      {{ signal.prediction.p_break | percent }}
    </div>
  </div>
  
  <!-- BREAK Drivers -->
  <div class="drivers break">
    <h3>BREAK Drivers</h3>
    <div *ngFor="let driver of signal.drivers.break" class="driver-item">
      <span class="icon">✓</span>
      <span class="name">{{ driver.name }}</span>
      <div class="strength-bar" [style.width.%]="driver.strength"></div>
      <span class="edge">{{ driver.edge | percent:'+' }}</span>
    </div>
  </div>
  
  <!-- REJECT Drivers -->
  <div class="drivers reject">
    <h3>REJECT Drivers</h3>
    <div *ngFor="let driver of signal.drivers.reject" class="driver-item">
      <span class="icon">✗</span>
      <span class="name">{{ driver.name }}</span>
      <div class="strength-bar" [style.width.%]="driver.strength"></div>
      <span class="edge">{{ driver.edge | percent }}</span>
    </div>
  </div>
</div>
```

---

## Decision Logic from Analyses

### Time Bucket Adjustment

Based on time stratification results:

```python
def get_time_adjusted_threshold(time_bucket: str, base_threshold: float = 0.5) -> float:
    """Adjust BREAK threshold based on time bucket."""
    
    # Load time stratification results
    time_stats = load_json('data/ml/time_stratification.json')
    
    # Get optimal threshold for this bucket
    if time_bucket in time_stats['optimal_thresholds']:
        return time_stats['optimal_thresholds'][time_bucket]['optimal_threshold']
    
    return base_threshold
```

### Level-Specific Adjustment

Based on level specificity results:

```python
def get_level_adjusted_base_rate(level_kind: str) -> float:
    """Get level-specific base BREAK rate."""
    
    level_stats = load_json('data/ml/level_specificity.json')
    
    if level_kind in level_stats['level_stats']:
        return level_stats['level_stats'][level_kind]['break_rate_overall']
    
    return 0.455  # Default
```

### Combo Detection

Based on feature interactions:

```python
def detect_active_combos(episode_vector: np.ndarray) -> List[Dict]:
    """Detect active interaction effects."""
    
    interactions = load_json('data/ml/feature_interactions.json')
    active_combos = []
    
    for interaction_key, interaction_data in interactions['hypothesized_interactions'].items():
        if not interaction_data['significant']:
            continue
        
        # Get feature values
        feat1_idx = feature_names.index(interaction_data['feat1'])
        feat2_idx = feature_names.index(interaction_data['feat2'])
        
        feat1_val = episode_vector[feat1_idx]
        feat2_val = episode_vector[feat2_idx]
        
        # Check if both are "high" (> 75th percentile)
        if feat1_val > percentile_75[feat1_idx] and feat2_val > percentile_75[feat2_idx]:
            active_combos.append({
                'description': interaction_data['description'],
                'edge': interaction_data['improvement'],
                'active': True
            })
    
    return active_combos
```

---

## Testing Strategy

### Unit Tests

```bash
# Test driver attribution
uv run pytest tests/test_driver_attribution.py -v

# Test time/level adjustment logic
uv run pytest tests/test_stratification_logic.py -v

# Test combo detection
uv run pytest tests/test_interaction_detection.py -v
```

### Integration Tests

```bash
# End-to-end: state update → attribution → WebSocket
uv run pytest tests/test_driver_integration.py -v
```

### Demo/Validation

```bash
# Generate example attributions
uv run python -m scripts.demo_driver_attribution \
  --version v4.0.0 \
  --output-json data/ml/driver_attribution_demo.json
```

---

## Monitoring & Maintenance

### Daily Tasks
- [ ] Update historical stats with new day's data
- [ ] Check model calibration (are predictions accurate?)
- [ ] Review trader feedback on driver usefulness

### Weekly Tasks
- [ ] Re-run stratification analyses with expanded data
- [ ] Test new interaction hypotheses
- [ ] Validate driver mappings still accurate

### Monthly Tasks
- [ ] Retrain 144D model with full historical data
- [ ] Update feature importance rankings
- [ ] Revise dashboard based on trader usage patterns

---

## Success Metrics

### Model Performance
- AUC ≥ 0.85 on predictable segment (144D model)
- Calibration error < 10% (predicted vs actual BREAK rate)
- Effective N ≥ 15 for real-time signals

### Trader Adoption
- % of signals where trader action aligns with HIGH confidence prediction
- Trader survey: "Do drivers help you understand WHY?"
- Reduction in "false positive" complaints (misunderstood signals)

### System Reliability
- Latency: Attribution computed in < 100ms
- Uptime: Dashboard updates 99%+ of 30s intervals
- Error rate: < 1% of attributions fail gracefully

---

## FAQ

**Q: Why 144D vector instead of just 5 streams?**  
A: 144D achieves 0.868 AUC vs 0.722 for streams. The extra dimensions capture GEX, barrier evolution, flow acceleration that streams don't fully represent.

**Q: Do I need to run ALL the analysis scripts?**  
A: For production, yes. For demo/testing, just the driver attribution is enough. Stratification and interactions inform UI display but aren't required for basic signal.

**Q: How often should I retrain the model?**  
A: Start with weekly, move to daily if you see drift. Monitor calibration error as key metric.

**Q: What if a driver has no historical stats?**  
A: Defaults to 50th percentile. Non-critical failure. Log warning and continue.

**Q: Can this work with other models (not RandomForest)?**  
A: Yes! `DriverAttributor` works with any sklearn-compatible model that has `predict_proba()`. For models without feature_importances_, it uses perturbation-based attribution (slower but works).

---

## Next Steps

1. **Run all analyses** on v4.0.0 data to generate JSON outputs
2. **Implement Phase 2** (historical stats computation)
3. **Integrate Phase 3** (real-time attribution in gateway)
4. **Build Phase 4** (frontend dashboard component)
5. **Test end-to-end** with demo data
6. **Deploy to production** with monitoring

**Estimated Timeline**: 2-3 weeks for full implementation

---

## Summary

This driver attribution system transforms the Pentaview ML model from a "black box probability" into an explainable, trader-friendly tool. By answering:

- **WHY** will this level break/bounce? → Driver decomposition
- **WHEN** is prediction most reliable? → Time stratification
- **WHICH LEVELS** behave differently? → Level-specific analysis  
- **WHAT COMBOS** create edge? → Feature interactions
- **HOW TO SHOW** this to traders? → Dashboard spec

...we enable traders to:
1. **Trust** the predictions (transparency)
2. **Learn** market dynamics (education)
3. **Act** with confidence (better decisions)

The key insight: **Traders don't just want predictions, they want REASONS.**

