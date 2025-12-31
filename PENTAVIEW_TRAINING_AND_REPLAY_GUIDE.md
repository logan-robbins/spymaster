# Pentaview Training & Replay Integration Guide

## Executive Summary

This guide explains how to train Pentaview projection models on historical data (2025-11-15 through 2025-12-15) and integrate them with the replay engine to display real-time projections on 2-minute charts as bronze futures data plays back.

**Key Principles:**
- Projection models are trained on **30-second bar cadence** (matching Stage 16 state table output)
- Inference runs every **30 seconds** when a new state sample arrives
- Requires **16 prior bars (8 minutes)** minimum history per level
- Start ingestion **16 bars before RTH (09:22 AM)** to enable inference at RTH start (09:30 AM)
- By 16 bars after RTH start (09:38 AM), will have 32 bars of history for improved predictions

---

## Part 1: Understanding the Data Flow

### Complete Pipeline Architecture

```
Bronze Futures (Parquet)
    ↓
Silver Pipeline (Stages 0-16)
    ├─ Stage 0: Load Bronze
    ├─ Stage 1-2: OHLCV Aggregation (1min, 2min)
    ├─ Stage 3: Initialize Market State
    ├─ Stage 4-15: Physics, OFI, Barriers, GEX Features
    └─ Stage 16: Materialize State Table (30s cadence)
         ↓
Pentaview Pipeline
    ├─ Use 30s state samples directly (no aggregation)
    ├─ Compute 5 canonical streams (M,F,B,D,S)
    ├─ Compute 2 merged streams (P,R)
    └─ Compute derivatives (slope, curvature, jerk)
         ↓
Gold Layer: stream_bars.parquet (32 columns @ 30s cadence)
         ↓
Projection Models (trained on 30s streams)
    └─ Forecast 5-min ahead (10 bars @ 30s) with uncertainty bands
         ↓
Replay Engine + UI
    └─ Display 2min candles + stream overlays + 30s projection updates
```

### Key Insight: Two Distinct Modes

**Training Mode** (Historical):
- Processes entire days of historical bronze data
- Generates stream bars from completed state tables
- Builds training datasets from stream history
- Trains projection models on past patterns

**Replay Mode** (Live/Simulated):
- Reads bronze futures trades progressively
- Publishes to NATS message bus
- Pipeline processes in real-time
- Generates streams every 30s (matching state table cadence)
- Applies trained projection models every 30s
- Updates UI every 30s with new projections

---

## Part 2: Training Pentaview on Historical Data

### Prerequisites

**✓ ASSUMED:** Silver layer state tables exist for 2025-11-07 through 2025-12-15

**Training Window:** 2025-11-15 through 2025-12-15 (30 days)

**Constants (keep explicit throughout):**
- `canonical_version`: `3.1.0`
- `train_start`: `2025-11-15`
- `train_end`: `2025-12-15`
- `dataset_version`: `v20251115_20251215`

**Verify Silver Data Exists:**
```bash
cd backend

# Check state tables exist for training range
ls -d data/silver/state/es_level_state/version=3.1.0/date=2025-11-{15..30}
ls -d data/silver/state/es_level_state/version=3.1.0/date=2025-12-{01..15}

# Should see directories for each date, e.g.:
# data/silver/state/es_level_state/version=3.1.0/date=2025-11-15/
# data/silver/state/es_level_state/version=3.1.0/date=2025-11-16/
# ... etc.
```

**Expected State Table Location:**
```
data/silver/state/es_level_state/version=3.1.0/date=YYYY-MM-DD/state.parquet
```

**If state tables missing:** Run pipeline Stage 0-16 first:
```bash
cd backend
uv run python -m scripts.run_pipeline \
  --start 2025-11-15 \
  --end 2025-12-15 \
  --canonical-version 3.1.0
```

---

### Step 1: Compute Normalization Statistics

**Purpose:** Calculate robust median/MAD statistics for feature normalization across the 30-day training window.

**Command:**
```bash
cd backend

# Compute stats using 30-day lookback (2025-11-15 to 2025-12-15)
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 30 \
  --end-date 2025-12-15 \
  --canonical-version 3.1.0 \
  --output-name current
```

**Why 30 days?** `end_date - 30 days = 2025-11-15`, matching the training window start.

**Output Location:** `data/gold/streams/normalization/current.json`

**What it contains:**
```json
{
  "version": "1.0",
  "created_at": "2025-12-30T...",
  "n_samples": ~23400,
  "stratify_by": ["time_bucket"],
  "global_stats": {
    "velocity_1min": {"method": "zscore", "mean": 0.12, "std": 0.45},
    "ofi_60s": {"method": "robust", "median": 125.0, "mad": 450.0}
  },
  "stratified_stats": {
    "T0_15": {...},
    "T15_30": {...},
    "T30_60": {...},
    "T60_120": {...},
    "T120_180": {...}
  }
}
```

**Validation:**
```bash
# Verify file exists
ls -lh data/gold/streams/normalization/current.json

# Check sample count
python -c "import json; d=json.load(open('data/gold/streams/normalization/current.json')); print(f'Samples: {d[\"n_samples\"]:,}')"
```

---

### Step 2: Run Pentaview Pipeline

**Purpose:** Transform 30-second state tables into normalized 30-second stream bars.

**Command:**
```bash
cd backend

# Batch processing (Nov 15 - Dec 15)
# IMPORTANT: --date is REQUIRED even when using --start/--end
# NOTE: Pentaview should use bar_freq='30s' to match state table cadence
uv run python -m scripts.run_pentaview_pipeline \
  --date 2025-11-15 \
  --start 2025-11-15 \
  --end 2025-12-15 \
  --canonical-version 3.1.0 \
  --bar-freq 30s
```

**⚠️ Critical Quirk:** The script requires `--date` even when batch processing with `--start/--end`. Set `--date` equal to `--start`.

**Output Location:** `data/gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD/stream_bars.parquet`

**Validation:**
```bash
cd backend

# Validate output for a single date
uv run python -m scripts.validate_pentaview \
  --date 2025-11-15 \
  --canonical-version 3.1.0

# Quick inspection
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/gold/streams/pentaview/version=3.1.0/date=2025-11-15/stream_bars.parquet')
print(f'Stream bars: {len(df):,}')
print(f'Columns: {len(df.columns)} columns')
print(f'Streams present: {[c for c in df.columns if c.startswith(\"sigma_\")]}')
print(f'Value range (sigma_p): [{df.sigma_p.min():.3f}, {df.sigma_p.max():.3f}]')
print(f'All values in [-1,+1]: {(df.sigma_p.between(-1, 1).all())}')
"
```

**Expected:**
- ~780 rows per date (30s bars × 6.5 hours RTH ≈ 780 bars)
- 32 columns: 5 canonical + 2 merged + 3 composites + derivatives
- All stream values bounded in [-1, +1]

---

### Step 3: Build Projection Training Dataset

**Purpose:** Extract training samples (16-bar history + 10-bar future targets) for projection model training.

**Command:**
```bash
cd backend

# Build dataset for all primary streams (16-bar lookback, 10-bar horizon @ 30s)
uv run python -m scripts.build_projection_dataset \
  --start 2025-11-15 \
  --end 2025-12-15 \
  --canonical-version 3.1.0 \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r \
  --lookback 16 \
  --horizon 10 \
  --output-dir data/gold/training/projection_samples \
  --version v30s_20251115_20251215
```

**Output Files:**
```
data/gold/training/projection_samples/
├── projection_samples_sigma_p_v30s_20251115_20251215.npz
├── projection_samples_sigma_m_v30s_20251115_20251215.npz
├── projection_samples_sigma_f_v30s_20251115_20251215.npz
├── projection_samples_sigma_b_v30s_20251115_20251215.npz
└── projection_samples_sigma_r_v30s_20251115_20251215.npz
```

**What's inside** (each .npz file):
```python
{
    'stream_hist': np.ndarray,      # [N, 16] - lookback history (8 min @ 30s)
    'slope_hist': np.ndarray,       # [N, 16] - slope history
    'current_value': np.ndarray,    # [N] - current stream value
    'future_target': np.ndarray,    # [N, 10] - future 10 bars (5 min @ 30s)
    'setup_weight': np.ndarray,     # [N] - quality weights
    'cross_streams': np.ndarray,    # [N, n_streams, 5] - context
    'static_features': np.ndarray   # [N, n_static] - level/time
}
```

---

### Step 4: Train Projection Models

**Purpose:** Train quantile polynomial regression models to forecast 5-minute ahead (10 bars @ 30s cadence).

**Command:**
```bash
cd backend

# Train all streams (16-bar lookback, 10-bar horizon)
uv run python -m scripts.train_projection_models \
  --stream all \
  --data-path data/gold/training/projection_samples \
  --output-dir data/ml/projection_models \
  --version v30s_20251115_20251215 \
  --lookback-bars 16 \
  --horizon-bars 10 \
  --epochs 200 \
  --learning-rate 0.05 \
  --max-depth 6 \
  --val-ratio 0.2
```

**Output Models:**
```
data/ml/projection_models/
├── projection_sigma_p_v30s_20251115_20251215.joblib
├── projection_sigma_m_v30s_20251115_20251215.joblib
├── projection_sigma_f_v30s_20251115_20251215.joblib
├── projection_sigma_b_v30s_20251115_20251215.joblib
└── projection_sigma_r_v30s_20251115_20251215.joblib
```

**Training Metrics** (logged to MLFlow + W&B):
```yaml
Experiment: stream_projection
Run: projection_pressure_v2025_12

Key Metrics:
  val_path_r2_q50: 0.42          # R² for path prediction (target: > 0.30)
  val_endpoint_mae_q50: 0.12     # 20-min horizon error (target: < 0.20)
  val_coverage_80pct: 0.79       # Calibration (target: ~0.80)
  val_band_width: 0.35           # Uncertainty band width
  q50_r2_mean: 0.65              # Coefficient fit quality
```

**View Results**:
```bash
cd backend
mlflow ui
# Open: http://localhost:5000
# Navigate to: Experiments → stream_projection
# Compare runs: Select all 5 streams → Click "Compare"
```

---

## Part 3: Replay Engine Integration

### Architecture Overview

**Replay Engine** (`backend/src/ingestion/databento/replay.py`):
- Reads bronze futures trades from parquet files
- Publishes to NATS subjects (`market.futures.trades`, `market.futures.mbp10`)
- Supports configurable speed (0 = max, 1.0 = realtime, 2.0 = 2x speed)
- Automatically filters to RTH start (9:30 AM ET) when `REPLAY_DATE` is set

**Real-Time Pipeline** (Docker services):
- **NATS**: Message bus for trade streams
- **Lake Service**: Writes trades to bronze layer
- **Core Service**: Runs pipeline stages 0-16 + Pentaview in streaming mode
- **Gateway Service**: WebSocket API for frontend

**Frontend** (`frontend/`):
- Angular app with TradingView charts
- Subscribes to WebSocket for real-time stream updates
- Displays 2-min candles + stream overlays + projections

### Inference Cadence

**30-Second Stream Generation:**
- State table rows generated every 30 seconds (Stage 16 output)
- Pentaview computes stream values every 30 seconds
- Projection inference runs every 30 seconds
- Stream values and projections emitted to Gateway every 30 seconds

**Display on 2-Minute Chart:**
- Frontend displays 2-minute OHLCV candles (price data)
- Stream overlays update every 30 seconds (more responsive)
- Projection bands update every 30 seconds
- Chart shows both 2-min price candles and 30s stream/projection overlays

---

### Step 5: Prepare Replay Date

**Choose date:** `2025-12-18`

**Verify prerequisites:**
```bash
cd backend

# 1. Check bronze futures data exists
ls -lh data/bronze/futures/symbol=ES/date=2025-12-18/
# Should see: trades_ES_2025-12-18.parquet

# 2. Verify/generate state table for replay date
ls -lh data/silver/state/es_level_state/version=3.1.0/date=2025-12-18/
# If missing, generate it:
uv run python -m scripts.run_pipeline \
  --date 2025-12-18 \
  --canonical-version 3.1.0

# 3. Generate Pentaview streams for replay date
uv run python -m scripts.run_pentaview_pipeline \
  --date 2025-12-18 \
  --canonical-version 3.1.0
```

**⚠️ Critical Warmup Configuration:**

**Start ingestion at 09:22 AM** (16 bars × 30s = 8 minutes before RTH):
- ✓ Accumulate 16 bars of history by 09:30 AM RTH start
- ✓ First projection available at 09:30 AM (exactly at RTH open)
- ✓ By 09:38 AM (16 bars after RTH), have 32 bars total history
- ✓ PM_HIGH/PM_LOW levels established during warmup
- ✓ SMA values warmed up

**Replay Configuration:**
```bash
# Start at 09:22 AM ET (16 bars before RTH)
REPLAY_START_TIME="09:22:00"
REPLAY_DATE="2025-12-18"
```

---

### Step 6: Configure Replay Parameters

**Environment Variables:**
```bash
export REPLAY_DATE=2025-12-18
export REPLAY_SPEED=1.0              # 1x realtime (or 0 for max speed)
export REPLAY_USE_BRONZE_FUTURES=true
export REPLAY_FUTURES_SYMBOL=ES
export REPLAY_INCLUDE_OPTIONS=false  # Skip options for now
```

**Replay Start Time:**
- Replay engine automatically filters to RTH start (9:30 AM ET) based on `REPLAY_DATE`
- For level warmup, consider starting ingestion earlier but rendering chart from 9:30

---

### Step 7: Start Docker Services

**Purpose:** Start NATS, Lake, Core, and Gateway services for real-time processing.

**Command:**
```bash
# Navigate to project root
cd /Users/loganrobbins/research/qmachina/spymaster

# Start services with REPLAY_DATE
export REPLAY_DATE=2025-12-18
docker-compose up -d nats lake gateway

# Restart Core to pick up REPLAY_DATE
docker-compose up -d --force-recreate core

# Wait for services to be ready
sleep 5

# Check Gateway health
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

---

### Step 8: Run Replay Engine

**Purpose:** Publish bronze trades to NATS for real-time processing.

**Command:**
```bash
cd backend

# Run replay at 1x realtime speed
REPLAY_SPEED=1.0 REPLAY_DATE=2025-12-18 \
  uv run python -m src.ingestion.databento.replay

# For faster testing (2x speed)
REPLAY_SPEED=2.0 REPLAY_DATE=2025-12-18 \
  uv run python -m src.ingestion.databento.replay

# For max speed validation (no delays)
REPLAY_SPEED=0 REPLAY_DATE=2025-12-18 \
  uv run python -m src.ingestion.databento.replay
```

**What happens:**
1. Replay engine reads bronze trades from `data/bronze/futures/symbol=ES/date=2025-12-18/`
2. Filters to trades >= RTH start (9:30 AM ET) automatically
3. Publishes trades to NATS at configured speed
4. Core service processes trades through pipeline stages 0-16
5. Pentaview generates stream values every 30s, aggregated to 2-min bars
6. Gateway emits stream updates via WebSocket to frontend

**Monitoring:**
```bash
# Watch replay stats
tail -f backend/logs/replay.log

# Check NATS message flow
docker exec -it nats nats sub "market.futures.trades"

# Monitor Core service processing
docker logs -f core | grep -E "(stream|projection)"
```

---

## Part 4: Projection Inference at 30-Second Cadence

### Real-Time Projection Flow

**Pipeline:**
```
Every 30s:
    ↓
Core generates new state table row (Stage 16)
    ↓
Pentaview computes stream values (sigma_p, sigma_m, etc.)
    ↓
Check history length >= 16 bars
    ↓
Load last 16 bars of stream history
    ↓
Apply projection model → predict next 10 bars (5 min @ 30s)
    ↓
Generate q10, q50, q90 curves
    ↓
Send projection update to Gateway
    ↓
Gateway emits WebSocket event: "projection_update"
    ↓
Frontend updates chart overlay (2-min candles + 30s stream/projection overlays)
```

**Timing:**
- **09:22 AM**: Start ingestion (16 bars before RTH)
- **09:30 AM**: RTH start, first projection available (16 bars accumulated)
- **09:38 AM**: 32 bars accumulated (16 before + 16 after RTH start)
- **Every 30s thereafter**: New projection with growing history

### Implementation Options

#### Option A: Core Service Auto-Projection (Recommended)

**Modify Core Service** to run projections automatically after each Pentaview update.

**Location**: `backend/src/core/streaming_pipeline.py` (or equivalent)

**Pseudocode**:
```python
from src.ml.projection.stream_projector import StreamProjector

class StreamingPipeline:
    def __init__(self):
        self.projectors = {
            'sigma_p': StreamProjector.load('data/ml/projection_models/projection_sigma_p_v30s_20251115_20251215.joblib'),
            'sigma_m': StreamProjector.load('data/ml/projection_models/projection_sigma_m_v30s_20251115_20251215.joblib'),
            # ... other streams
        }
    
    async def on_stream_update(self, stream_bar: StreamBar):
        """Called every 30 seconds when Pentaview emits a new stream sample."""
        
        # Get last 16 bars of history (from in-memory buffer or DB)
        history = self.get_stream_history(stream_bar.level_kind, lookback=16)
        
        if len(history) < 16:
            return  # Need minimum 16 bars
        
        # Run projections for all streams
        projections = {}
        for stream_name, projector in self.projectors.items():
            proj = projector.predict(
                stream_hist=history[stream_name],
                current_bar=stream_bar,
                horizon=10  # 10 bars = 5 minutes @ 30s
            )
            projections[stream_name] = proj
        
        # Emit projection update via Gateway
        await self.gateway.emit('projection_update', {
            'timestamp': stream_bar.timestamp,
            'level_kind': stream_bar.level_kind,
            'projections': projections
        })
```

#### Option B: Gateway Service Projections

**Modify Gateway** to compute projections on-demand when frontend requests.

**API Endpoint**:
```python
@app.post('/api/projections/compute')
async def compute_projection(request: ProjectionRequest):
    """
    Compute projection for a specific stream at current bar.
    
    Request:
        {
            "stream": "sigma_p",
            "level_kind": "OR_HIGH",
            "lookback_bars": 20,
            "horizon_bars": 10
        }
    
    Response:
        {
            "timestamp": "2025-12-18T10:32:00",
            "stream": "sigma_p",
            "current_value": 0.67,
            "q10_curve": [0.67, 0.65, 0.62, ..., 0.45],
            "q50_curve": [0.67, 0.69, 0.71, ..., 0.58],
            "q90_curve": [0.67, 0.73, 0.79, ..., 0.72]
        }
    """
    projector = StreamProjector.load(f'models/projection_{request.stream}_v2025_12.joblib')
    history = await get_stream_history(request.stream, request.level_kind, request.lookback_bars)
    
    projection = projector.predict(stream_hist=history, horizon=request.horizon_bars)
    return projection
```

#### Option C: Frontend-Triggered Projections

**Frontend calls Gateway API every 30s** (or when new bar arrives).

**Angular Service**:
```typescript
@Injectable()
export class ProjectionService {
  private projectionTimer$ = interval(30000);  // Every 30 seconds
  
  startAutoProjections() {
    this.projectionTimer$.pipe(
      switchMap(() => this.computeProjection('sigma_p', 20, 10))
    ).subscribe(projection => {
      this.updateChartOverlay(projection);
    });
  }
  
  computeProjection(stream: string, lookback: number, horizon: number): Observable<Projection> {
    return this.http.post<Projection>('/api/projections/compute', {
      stream,
      level_kind: 'OR_HIGH',
      lookback_bars: lookback,
      horizon_bars: horizon
    });
  }
}
```

---

### Recommended Approach: Option A (Core Service)

**Why**:
- **Efficiency**: Compute once, broadcast to all connected clients
- **Consistency**: Same projections for all users
- **Low latency**: No round-trip delay for frontend requests
- **Scalability**: Projections pre-computed, not on-demand

**Implementation**:
1. Load projection models in Core service at startup
2. Maintain in-memory buffer of last 20 stream bars per level
3. After each Pentaview update (2-min bar), run projections
4. Emit projection update via Gateway WebSocket
5. Frontend subscribes to `projection_update` event

---

## Part 5: Frontend Chart Updates

### TradingView Lightweight Charts Integration

**Chart Structure**:
```typescript
interface ChartData {
  // Candlestick series (left Y-axis)
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Stream overlays (right Y-axis, -1 to +1)
  sigma_p: number;
  sigma_m: number;
  sigma_f: number;
  sigma_b: number;
  sigma_s: number;
  
  // Projection bands (right Y-axis, -1 to +1)
  sigma_p_q10: number[];  // Lower bound (next 10 bars)
  sigma_p_q50: number[];  // Median forecast (next 10 bars)
  sigma_p_q90: number[];  // Upper bound (next 10 bars)
}
```

### WebSocket Subscription

**Angular Service**:
```typescript
import { WebSocketSubject } from 'rxjs/webSocket';

@Injectable()
export class RealtimeStreamService {
  private ws$: WebSocketSubject<any>;
  
  connect() {
    this.ws$ = webSocket('ws://localhost:8000/ws');
    
    // Subscribe to stream updates
    this.ws$.pipe(
      filter(msg => msg.type === 'stream_update')
    ).subscribe(update => {
      this.handleStreamUpdate(update);
    });
    
    // Subscribe to projection updates
    this.ws$.pipe(
      filter(msg => msg.type === 'projection_update')
    ).subscribe(update => {
      this.handleProjectionUpdate(update);
    });
  }
  
  handleStreamUpdate(update: StreamUpdate) {
    const bar = update.bar;
    
    // Add new candlestick data point
    this.candlestickSeries.update({
      time: bar.timestamp,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close
    });
    
    // Update stream overlay
    this.streamSeries.update({
      time: bar.timestamp,
      value: bar.sigma_p
    });
  }
  
  handleProjectionUpdate(update: ProjectionUpdate) {
    const proj = update.projections.sigma_p;
    
    // Clear old projection
    this.projectionUpperSeries.setData([]);
    this.projectionMedianSeries.setData([]);
    this.projectionLowerSeries.setData([]);
    
    // Add new projection curves
    const baseTime = new Date(update.timestamp).getTime() / 1000;
    
    for (let i = 0; i < proj.q50_curve.length; i++) {
      const futureTime = baseTime + (i * 120);  // 2-min bars = 120 seconds
      
      this.projectionUpperSeries.update({
        time: futureTime,
        value: proj.q90_curve[i]
      });
      
      this.projectionMedianSeries.update({
        time: futureTime,
        value: proj.q50_curve[i]
      });
      
      this.projectionLowerSeries.update({
        time: futureTime,
        value: proj.q10_curve[i]
      });
    }
  }
}
```

### Chart Configuration

**Series Setup**:
```typescript
// Main chart (left Y-axis: price)
const chart = createChart(container, {
  layout: { background: { color: '#1E1E1E' } },
  leftPriceScale: { visible: true },
  rightPriceScale: { visible: true },
  timeScale: { timeVisible: true, secondsVisible: false }
});

// Candlestick series
const candlestickSeries = chart.addCandlestickSeries({
  priceScaleId: 'left'
});

// Stream overlay (right Y-axis: -1 to +1)
const streamSeries = chart.addLineSeries({
  color: '#F44336',
  lineWidth: 2,
  priceScaleId: 'right',
  title: 'Pressure (σ_P)'
});

// Projection bands (right Y-axis)
const projectionUpperSeries = chart.addLineSeries({
  color: 'rgba(255, 152, 0, 0.8)',
  lineWidth: 1,
  lineStyle: 2,  // Dashed
  priceScaleId: 'right',
  title: 'Projection (90th %ile)'
});

const projectionMedianSeries = chart.addLineSeries({
  color: 'rgba(255, 152, 0, 1.0)',
  lineWidth: 2,
  lineStyle: 2,
  priceScaleId: 'right',
  title: 'Projection (Median)'
});

const projectionLowerSeries = chart.addLineSeries({
  color: 'rgba(255, 152, 0, 0.8)',
  lineWidth: 1,
  lineStyle: 2,
  priceScaleId: 'right',
  title: 'Projection (10th %ile)'
});

// Configure right Y-axis for streams
chart.priceScale('right').applyOptions({
  scaleMargins: { top: 0.1, bottom: 0.1 },
  autoScale: false,
  minValue: -1.0,
  maxValue: 1.0
});
```

---

## Part 6: Putting It All Together

### Complete Workflow

#### Training Phase (Given: Silver data exists 2025-11-07 to 2025-12-15)

```bash
cd backend

# Constants
CANONICAL_VERSION="3.1.0"
TRAIN_START="2025-11-15"
TRAIN_END="2025-12-15"
DATASET_VERSION="v30s_20251115_20251215"

# Step 1: Compute normalization stats (30-day lookback)
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 30 \
  --end-date $TRAIN_END \
  --canonical-version $CANONICAL_VERSION \
  --output-name current

# Verify output
ls -lh data/gold/streams/normalization/current.json

# Step 2: Run Pentaview pipeline (generate 30s stream bars)
# IMPORTANT: --date is REQUIRED even with --start/--end
uv run python -m scripts.run_pentaview_pipeline \
  --date $TRAIN_START \
  --start $TRAIN_START \
  --end $TRAIN_END \
  --canonical-version $CANONICAL_VERSION \
  --bar-freq 30s

# Verify output
ls -d data/gold/streams/pentaview/version=$CANONICAL_VERSION/date=2025-11-*/
ls -d data/gold/streams/pentaview/version=$CANONICAL_VERSION/date=2025-12-*/

# Step 3: Build projection training dataset (16-bar lookback, 10-bar horizon)
uv run python -m scripts.build_projection_dataset \
  --start $TRAIN_START \
  --end $TRAIN_END \
  --canonical-version $CANONICAL_VERSION \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r \
  --lookback 16 \
  --horizon 10 \
  --version $DATASET_VERSION

# Verify output
ls -lh data/gold/training/projection_samples/projection_samples_*_${DATASET_VERSION}.npz

# Step 4: Train projection models (5 streams, 16-bar lookback)
uv run python -m scripts.train_projection_models \
  --stream all \
  --version $DATASET_VERSION \
  --lookback-bars 16 \
  --horizon-bars 10 \
  --epochs 200 \
  --learning-rate 0.05 \
  --max-depth 6

# Verify output
ls -lh data/ml/projection_models/projection_*_${DATASET_VERSION}.joblib

# Step 5: View training results in MLFlow
mlflow ui
# Open http://localhost:5000
# Navigate to: Experiments → stream_projection
# Check metrics: path_r2 > 0.30, coverage ~0.80
```

#### Replay Phase (2025-12-18)

```bash
cd backend

# Constants
REPLAY_DATE="2025-12-18"
CANONICAL_VERSION="3.1.0"

# Step 5: Prepare replay date artifacts
uv run python -m scripts.run_pipeline \
  --date $REPLAY_DATE \
  --canonical-version $CANONICAL_VERSION

uv run python -m scripts.run_pentaview_pipeline \
  --date $REPLAY_DATE \
  --canonical-version $CANONICAL_VERSION

# Step 6: Set environment variables
export REPLAY_DATE=$REPLAY_DATE
export REPLAY_SPEED=1.0
export REPLAY_USE_BRONZE_FUTURES=true
export REPLAY_FUTURES_SYMBOL=ES

# Step 7: Start Docker services
cd /Users/loganrobbins/research/qmachina/spymaster
docker-compose up -d nats lake gateway
docker-compose up -d --force-recreate core
sleep 5

# Verify health
curl http://localhost:8000/health

# Step 8a: Start frontend (Terminal 1)
cd frontend
npm run start
# Open http://localhost:4200

# Step 8b: Run replay engine (Terminal 2)
cd backend
REPLAY_SPEED=1.0 REPLAY_DATE=$REPLAY_DATE \
  uv run python -m src.ingestion.databento.replay

# Monitor (Terminal 3)
docker logs -f core | grep -E "(stream|projection)"

# Expected behavior:
# - Start ingestion at 09:22 AM (16 bars before RTH)
# - 2-min candlesticks appear starting at 9:30 AM ET
# - Stream overlays update every 30 seconds (values in [-1, +1])
# - First projection available at 09:30 AM (16 bars accumulated)
# - By 09:38 AM, have 32 bars of history
# - Projection bands extend 5 minutes (10 bars @ 30s) into future
# - Bands show q10 (lower), q50 (median), q90 (upper) curves
```

---

## Part 7: Answering Your Specific Questions

### Q1: How do I train Pentaview on data from Nov 15 - Dec 15?

**Answer:** Given that silver state tables already exist for 2025-11-07 through 2025-12-15, follow these exact steps:

```bash
cd backend

# Step 1: Compute normalization (30-day lookback)
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 30 \
  --end-date 2025-12-15 \
  --canonical-version 3.1.0 \
  --output-name current

# Step 2: Generate 30s stream bars (requires --date even with --start/--end)
uv run python -m scripts.run_pentaview_pipeline \
  --date 2025-11-15 \
  --start 2025-11-15 \
  --end 2025-12-15 \
  --canonical-version 3.1.0 \
  --bar-freq 30s

# Step 3: Build training dataset (16-bar lookback, 10-bar horizon)
uv run python -m scripts.build_projection_dataset \
  --start 2025-11-15 \
  --end 2025-12-15 \
  --canonical-version 3.1.0 \
  --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r \
  --lookback 16 \
  --horizon 10 \
  --version v30s_20251115_20251215

# Step 4: Train models (16-bar lookback)
uv run python -m scripts.train_projection_models \
  --stream all \
  --version v30s_20251115_20251215 \
  --lookback-bars 16 \
  --horizon-bars 10 \
  --epochs 200
```

**Key points:**
- Use 30s bar frequency (--bar-freq 30s)
- 16-bar lookback = 8 minutes @ 30s cadence
- 10-bar horizon = 5 minutes @ 30s cadence
- Version tag: `v30s_20251115_20251215`
- Output normalization to: `data/gold/streams/normalization/current.json`

### Q2: How do I use replay engine to play Dec 18 from RTH start?

**Answer:** The replay engine automatically filters to RTH start (9:30 AM ET) when you specify the date:

```bash
export REPLAY_DATE=2025-12-18
export REPLAY_SPEED=1.0
uv run python -m src.ingestion.databento.replay
```

Internally, `replay.py` calculates 9:30 AM ET in nanoseconds for the specified date and filters trades accordingly.

**⚠️ Critical Timing:**
- **Start at 09:22 AM** (16 bars × 30s = 8 minutes before RTH)
- **First projection at 09:30 AM** (RTH start, 16 bars accumulated)
- **32 bars accumulated by 09:38 AM** (16 before + 16 after RTH)
- PM_HIGH/PM_LOW levels established during 09:22-09:30 warmup
- SMA values warmed up during initial 8 minutes

### Q3: Should I call inference every 30 seconds?

**Answer: YES. Call inference every 30 SECONDS.**

**Cadence:**
- **State updates**: Every 30 seconds (Stage 16 output)
- **Stream computation**: Every 30 seconds (Pentaview processes each state sample)
- **Projection inference**: Every 30 seconds (models predict from latest 16-bar history)

**Implementation pattern:**
```python
async def on_stream_bar_complete(self, stream_bar: StreamBar):
    """Called every 30 seconds when new stream sample arrives."""
    
    # Check minimum history requirement
    if len(self.stream_history) < 16:
        logger.info(f"Need 16 bars, have {len(self.stream_history)}, waiting...")
        return
    
    # Run projection inference
    projections = self.compute_projections(stream_bar)
    await self.gateway.emit('projection_update', projections)
```

**Timing:**
- **09:22 AM**: Start ingestion
- **09:30 AM**: First projection (16 bars accumulated)
- **09:38 AM**: 32 bars accumulated
- **Every 30s thereafter**: New projection with updated history

### Q4: Update projection overlay on 2-min chart?

**Answer**: Yes. The projection overlay shows:

1. **Current Stream Value** (bright line): Current sigma_p, sigma_m, etc.
2. **Historical Stream** (solid line): Past stream values from previous 30s samples
3. **Projection Band** (dashed lines + shaded area):
   - **Upper bound** (q90): Optimistic forecast
   - **Median** (q50): Expected trajectory
   - **Lower bound** (q10): Conservative forecast
   - **Shaded area**: Uncertainty region between q10 and q90

**Update frequency**: Every 30 seconds (when new stream sample arrives)

**Chart appearance**:
```
Time: ──────────────────────────────────────────────────►
      9:30      9:32      9:34      9:36      9:38      9:40
      
Price (2-min candles):
 6900 ─────────────────────────────────────────────────
 6880 ─────────────────────────────────────────────────
 6860 ─────────────────────────────────────────────────

Stream (30s updates):
 +1.0 ─────────────────────────────────────────────────
 +0.5 ──────┐              ╱╲╲╲╲  ← Projection (5 min ahead)
  0.0 ──────│──────────────╱  ╲╲╲
 -0.5 ──────│          ╱╱╱    ╲╲
 -1.0 ──────┘       ╱╱╱      ╲
                ↑       ↑
          Last 16 bars  Current + 10-bar projection
          (8 min)       (5 min ahead)
```

---

## Part 8: Testing & Validation

### Validate Complete Pipeline

```bash
cd backend

# 1. Check bronze data
ls -lh data/bronze/futures/symbol=ES/date=2025-12-18/

# 2. Check silver state table
uv run python -m scripts.validate_stage_16_materialize_state_table --date 2025-12-18

# 3. Check Pentaview streams
uv run python -m scripts.validate_pentaview --date 2025-12-18

# 4. Test projection inference
uv run python -m scripts.demo_projection

# 5. Run replay in fast mode (for testing)
REPLAY_SPEED=0 REPLAY_DATE=2025-12-18 \
  uv run python -m src.ingestion.databento.replay
```

### Monitor Real-Time Pipeline

```bash
# Watch Core service logs
docker logs -f core

# Watch Gateway WebSocket messages
docker logs -f gateway | grep projection_update

# Monitor NATS messages
docker exec -it nats nats sub "market.futures.trades"

# Check MLFlow UI
cd backend && mlflow ui
# Open: http://localhost:5000
```

---

## Part 9: Troubleshooting

### Issue: "No Bronze futures trades found"

**Solution:**
```bash
# Verify data exists
ls backend/data/bronze/futures/symbol=ES/date=2025-12-18/

# If missing, run backfill
cd backend
uv run python -m scripts.backfill_bronze_futures --date 2025-12-18
```

### Issue: "Not enough data for projection (need at least 16 bars)"

**Solution:** Start ingestion at 09:22 AM (16 bars before RTH).

```python
if len(stream_history) < 16:
    logger.info(f"Need 16 bars for projection, currently have {len(stream_history)}...")
    return None  # Skip projection until enough history
```

**Why 16 bars?** Projection models use 16-bar history as input features (8 minutes @ 30s cadence).

**Configuration:** Start at 09:22 AM → First projection available at 09:30 AM RTH start.

### Issue: "Projection values outside [-1, +1]"

**Solution:** Check normalization stats are loaded:

```bash
# Verify normalization file exists (correct path!)
ls backend/data/gold/streams/normalization/current.json

# Re-compute if missing (use 30-day lookback)
cd backend
uv run python -m scripts.compute_stream_normalization \
  --lookback-days 30 \
  --end-date 2025-12-15 \
  --canonical-version 3.1.0 \
  --output-name current
```

### Issue: "WebSocket not receiving updates"

**Solution**:
```bash
# Check Gateway is running
curl http://localhost:8000/health

# Check Gateway logs
docker logs -f gateway

# Restart Gateway
docker-compose restart gateway
```

---

## Part 10: Summary Checklist

### Training Checklist

**Given:** Silver state tables exist for 2025-11-07 through 2025-12-15

- [ ] **Step 1:** Compute normalization statistics (30-day lookback, 2025-11-15 to 2025-12-15)
  - Command: `compute_stream_normalization --lookback-days 30 --end-date 2025-12-15 --canonical-version 3.1.0 --output-name current`
  - Output: `data/gold/streams/normalization/current.json`

- [ ] **Step 2:** Run Pentaview pipeline (generate 30s stream bars)
  - Command: `run_pentaview_pipeline --date 2025-11-15 --start 2025-11-15 --end 2025-12-15 --canonical-version 3.1.0 --bar-freq 30s`
  - Note: `--date` is REQUIRED even with `--start/--end`
  - Note: `--bar-freq 30s` matches Stage 16 state table cadence
  - Output: `data/gold/streams/pentaview/version=3.1.0/date=*/stream_bars.parquet` (~780 rows/date)

- [ ] **Step 3:** Build projection training dataset (16-bar lookback, 10-bar horizon)
  - Command: `build_projection_dataset --start 2025-11-15 --end 2025-12-15 --lookback 16 --horizon 10 --version v30s_20251115_20251215 --streams sigma_p,sigma_m,sigma_f,sigma_b,sigma_r`
  - Output: 5 `.npz` files with version `v30s_20251115_20251215`

- [ ] **Step 4:** Train projection models (5 streams, 16-bar lookback, 200 epochs)
  - Command: `train_projection_models --stream all --version v30s_20251115_20251215 --lookback-bars 16 --horizon-bars 10 --epochs 200`
  - Output: 5 `.joblib` models in `data/ml/projection_models/`
  - Validate: Check MLFlow UI for metrics (path_r2 > 0.30, coverage ~0.80)

### Replay Checklist (2025-12-18)

- [ ] **Step 5:** Prepare replay date
  - Verify bronze data: `ls data/bronze/futures/symbol=ES/date=2025-12-18/`
  - Generate/verify state table: `run_pipeline --date 2025-12-18 --canonical-version 3.1.0`
  - Generate Pentaview streams: `run_pentaview_pipeline --date 2025-12-18 --canonical-version 3.1.0`

- [ ] **Step 6:** Configure environment
  - Set `REPLAY_DATE=2025-12-18`, `REPLAY_SPEED=1.0`
  - Set `REPLAY_USE_BRONZE_FUTURES=true`, `REPLAY_FUTURES_SYMBOL=ES`

- [ ] **Step 7:** Start Docker services
  - `docker-compose up -d nats lake gateway`
  - `docker-compose up -d --force-recreate core`
  - Verify health: `curl http://localhost:8000/health`

- [ ] **Step 8:** Run replay engine
  - Command: `REPLAY_SPEED=1.0 REPLAY_DATE=2025-12-18 uv run python -m src.ingestion.databento.replay`
  - Monitor: `tail -f logs/replay.log`, `docker logs -f core`

- [ ] Start frontend: `cd frontend && npm run start` → http://localhost:4200

- [ ] **Verify real-time flow:**
  - Start at 09:22 AM (16 bars before RTH for warmup)
  - 2-min candles appear starting at 09:30 AM
  - Stream overlays update every 30s (values in [-1, +1])
  - First projection at 09:30 AM (16 bars accumulated)
  - 32 bars accumulated by 09:38 AM
  - Projection bands extend 5 minutes (10 bars @ 30s) into future

### Integration Checklist (Core Service)

- [ ] Load projection models at startup (`data/ml/projection_models/projection_*_v30s_20251115_20251215.joblib`)
- [ ] Maintain in-memory buffer of stream bars per level (minimum 16, grow to 32+)
- [ ] **Every 30 seconds** (when new state sample arrives):
  - [ ] Check history length >= 16 bars
  - [ ] Compute projections for all active streams
  - [ ] Generate q10, q50, q90 curves (10 bars = 5 minutes @ 30s)
  - [ ] Emit `projection_update` event via Gateway WebSocket
- [ ] Frontend subscribes to `projection_update` and renders chart overlays
- [ ] Start ingestion at 09:22 AM (16 bars before RTH)

### Critical Requirements

⚠️ **Inference Cadence:** Every 30 SECONDS (matching Stage 16 state table output)
⚠️ **Minimum History:** 16 bars (8 minutes @ 30s cadence)
⚠️ **Warmup Timing:** Start at 09:22 AM (16 bars before RTH) → First projection at 09:30 AM
⚠️ **Bar Frequency:** `--bar-freq 30s` in Pentaview pipeline
⚠️ **Lookback/Horizon:** 16 bars lookback, 10 bars horizon
⚠️ **Version Tag:** `v30s_20251115_20251215`
⚠️ **Paths:** Use versioned paths with `canonical-version=3.1.0`
⚠️ **Normalization:** Located at `data/gold/streams/normalization/current.json`

---

## Conclusion

This guide provides the **validated, canonical workflow** for:

1. **Training:** Building Pentaview projection models on 30 days of historical data (2025-11-15 through 2025-12-15)
2. **Replay:** Playing back bronze futures with real-time projection inference
3. **Integration:** Displaying 2-min candles + stream overlays + projection bands in frontend

### Key Configuration Parameters

**✓ Inference Cadence:** Every **30 SECONDS** (matches Stage 16 state table output)
**✓ Bar Frequency:** `--bar-freq 30s` (no aggregation, use state samples directly)
**✓ Lookback History:** **16 bars** (8 minutes @ 30s cadence)
**✓ Forecast Horizon:** **10 bars** (5 minutes @ 30s cadence)
**✓ Warmup Timing:** Start at **09:22 AM** (16 bars before RTH) → First projection at **09:30 AM**
**✓ Training Window:** **30 days** (2025-11-15 to 2025-12-15)
**✓ Version Naming:** `v30s_20251115_20251215` format
**✓ Normalization Path:** `data/gold/streams/normalization/current.json`
**✓ State Table Path:** `data/silver/state/es_level_state/version=3.1.0/`
**✓ run_pentaview_pipeline:** Requires `--date` flag even with `--start/--end` (script quirk)

### Timing Summary

- **09:22 AM**: Start ingestion (16 bars before RTH)
- **09:30 AM**: RTH start + first projection (16 bars accumulated)
- **09:38 AM**: 32 bars accumulated (16 before + 16 after RTH)
- **Every 30s thereafter**: New projection with growing history

### Next Steps

1. **Train models** (Steps 1-4):
   - Compute normalization (30-day lookback)
   - Generate 30s stream bars (--bar-freq 30s)
   - Build training dataset (16-bar lookback, 10-bar horizon)
   - Train projection models (5 streams, 200 epochs)

2. **Validate training:**
   - Check MLFlow UI for metrics
   - Verify path_r2 > 0.30, coverage ~0.80
   - Test with `demo_projection.py`

3. **Run replay** (Steps 5-8):
   - Prepare replay date artifacts
   - Start Docker services
   - Configure start time: 09:22 AM (16 bars before RTH)
   - Run replay engine
   - Monitor first projection at 09:30 AM RTH start

4. **Iterate:**
   - Adjust hyperparameters if needed
   - Expand training window as more data accumulates
   - Monitor 30s projection updates in frontend

**This guide provides the complete configuration for 30-second inference cadence with 16-bar lookback, enabling projections at RTH start via pre-warmup.**

