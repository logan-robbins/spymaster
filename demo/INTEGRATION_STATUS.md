# Pentaview Viewer Integration Status

## ✅ COMPLETED

### Demo Folder Rebuild
- ✓ Completely rebuilt `demo/` folder for Pentaview visualization
- ✓ Removed all legacy generic projection code
- ✓ Added WebSocket proxy to Gateway (`ws://localhost:8000/ws/stream`)
- ✓ TradingView Lightweight Charts with dual Y-axes
- ✓ Stream overlays ready: sigma_p, sigma_m, sigma_f, sigma_b, sigma_r
- ✓ Projection bands ready: q10, q50, q90
- ✓ Professional dark UI with status indicators
- ✓ Comprehensive README with troubleshooting

### Files Created/Modified
```
demo/
├── app.py                   [REBUILT] Flask WebSocket proxy
├── templates/index.html     [REBUILT] TradingView chart UI
├── requirements.txt         [UPDATED] Flask-Sock + websockets
├── run.sh                   [UPDATED] Launch script
├── README.md                [NEW] Complete documentation
└── INTEGRATION_STATUS.md    [NEW] This file
```

### Training Complete
- ✓ 5 projection models trained (Path R² 0.77-0.86)
- ✓ Models saved: `backend/data/ml/projection_models/projection_*_v30s_20251115_20251215.joblib`
- ✓ Normalization stats: `backend/data/gold/streams/normalization/current.json`
- ✓ 30s bar frequency configured in Pentaview pipeline
- ✓ Docker services running (NATS, Lake, Gateway, Core)

## ⚠️ MISSING: Gateway Stream Integration

The viewer is **ready to display data**, but the **Gateway is not yet emitting Pentaview streams**.

### What the Gateway Needs to Send

The viewer expects WebSocket messages with this structure:

```json
{
  "candles": [
    {"time": 1702900800, "open": 4500.0, "high": 4502.5, "low": 4498.0, "close": 4501.0}
  ],
  "streams": {
    "sigma_p": [{"time": 1702900800, "value": 0.45}],
    "sigma_m": [{"time": 1702900800, "value": -0.12}],
    "sigma_f": [{"time": 1702900800, "value": 0.23}],
    "sigma_b": [{"time": 1702900800, "value": 0.67}],
    "sigma_r": [{"time": 1702900800, "value": 0.34}]
  },
  "projections": {
    "q10": [{"time": 1702900830, "value": 0.40}, {"time": 1702900860, "value": 0.38}, ...],
    "q50": [{"time": 1702900830, "value": 0.45}, {"time": 1702900860, "value": 0.47}, ...],
    "q90": [{"time": 1702900830, "value": 0.50}, {"time": 1702900860, "value": 0.56}, ...]
  }
}
```

### Required Backend Work

According to the **PENTAVIEW_TRAINING_AND_REPLAY_GUIDE.md** (Part 4), you need to:

#### Option A: Core Service Auto-Projection (Recommended)

**File**: `backend/src/core/streaming_pipeline.py` or equivalent

```python
from src.ml.projection.stream_projector import StreamProjector

class StreamingPipeline:
    def __init__(self):
        # Load trained models
        self.projectors = {
            'sigma_p': StreamProjector.load('data/ml/projection_models/projection_sigma_p_v30s_20251115_20251215.joblib'),
            'sigma_m': StreamProjector.load('data/ml/projection_models/projection_sigma_m_v30s_20251115_20251215.joblib'),
            'sigma_f': StreamProjector.load('data/ml/projection_models/projection_sigma_f_v30s_20251115_20251215.joblib'),
            'sigma_b': StreamProjector.load('data/ml/projection_models/projection_sigma_b_v30s_20251115_20251215.joblib'),
            'sigma_r': StreamProjector.load('data/ml/projection_models/projection_sigma_r_v30s_20251115_20251215.joblib'),
        }
        self.stream_history = {}  # Buffer last 20 bars per level
    
    async def on_stream_update(self, stream_bar: StreamBar):
        """Called every 30 seconds when Pentaview emits a new stream sample."""
        
        # Update history buffer
        level_key = stream_bar.level_kind
        if level_key not in self.stream_history:
            self.stream_history[level_key] = []
        self.stream_history[level_key].append(stream_bar)
        
        # Keep last 20 bars
        if len(self.stream_history[level_key]) > 20:
            self.stream_history[level_key] = self.stream_history[level_key][-20:]
        
        # Check minimum history (16 bars = 8 minutes @ 30s)
        if len(self.stream_history[level_key]) < 16:
            logger.info(f"Waiting for history: {len(self.stream_history[level_key])}/16 bars")
            return
        
        # Run projections for all streams
        projections = {}
        for stream_name, projector in self.projectors.items():
            proj = projector.predict(
                stream_hist=self.get_history(level_key, stream_name),
                horizon=10  # 10 bars = 5 minutes @ 30s
            )
            projections[stream_name] = proj
        
        # Emit to Gateway
        await self.gateway.emit('pentaview_update', {
            'timestamp': stream_bar.timestamp,
            'level_kind': level_key,
            'candles': self.get_recent_candles(120),  # Last 2 minutes
            'streams': self.format_streams(stream_bar),
            'projections': self.format_projections(projections)
        })
```

#### Gateway Update

**File**: `backend/src/gateway/websocket_handler.py` or equivalent

```python
@websocket.on('pentaview_update')
async def handle_pentaview_update(data):
    """Forward Pentaview stream data to connected clients"""
    
    # Format for frontend
    payload = {
        'candles': data['candles'],
        'streams': {
            'sigma_p': data['streams']['sigma_p'],
            'sigma_m': data['streams']['sigma_m'],
            'sigma_f': data['streams']['sigma_f'],
            'sigma_b': data['streams']['sigma_b'],
            'sigma_r': data['streams']['sigma_r'],
        },
        'projections': {
            'q10': data['projections']['sigma_p']['q10'],
            'q50': data['projections']['sigma_p']['q50'],
            'q90': data['projections']['sigma_p']['q90'],
        }
    }
    
    # Broadcast to all WebSocket clients
    await broadcast(payload)
```

## Testing the Viewer

### 1. Install Dependencies

```bash
cd demo

# Using uv with Python 3.12
uv venv --python 3.12
uv pip sync

# Or just run the launch script (handles setup automatically)
./run.sh
```

### 2. Start the Viewer

```bash
./run.sh
```

### 3. Open Browser

Navigate to: **http://localhost:5000**

### 4. Expected Behavior

**Without Gateway Integration (current state):**
- ✓ UI loads correctly
- ✓ Status shows "Connecting..." or "Connected"
- ✗ No candles appear (Gateway not emitting data)
- ✗ Streams show "—" (waiting for data)
- ✗ No projections visible

**With Gateway Integration (after backend work):**
- ✓ UI loads correctly
- ✓ Status shows "Connected" (green dot)
- ✓ 2-minute candles appear progressively
- ✓ Stream lines update every 30 seconds
- ✓ Projection bands extend 5 minutes ahead
- ✓ Legend values update in real-time

## Next Actions

### Priority 1: Core Service Integration

1. **Check if StreamProjector exists:**
   ```bash
   cd backend
   find src -name "*projector*" -o -name "*projection*"
   ```

2. **Locate streaming pipeline:**
   ```bash
   cd backend
   find src/core -name "*.py" | grep -E "(stream|pipeline)"
   ```

3. **Implement projection inference** (see Option A above)

### Priority 2: Gateway WebSocket Publisher

1. **Find Gateway WebSocket handler:**
   ```bash
   cd backend
   find src/gateway -name "*.py"
   ```

2. **Add Pentaview stream emission** (see Gateway Update above)

### Priority 3: Test End-to-End

```bash
# Terminal 1: Start replay
cd backend
REPLAY_SPEED=1.0 REPLAY_DATE=2025-12-18 \
  REPLAY_USE_BRONZE_FUTURES=true REPLAY_FUTURES_SYMBOL=ES \
  uv run python -m src.ingestion.databento.replay

# Terminal 2: Monitor Core
docker logs -f spymaster-core | grep -E "(stream|projection)"

# Terminal 3: Start viewer
cd demo
./run.sh

# Browser: Open http://localhost:5000
```

## Files to Check/Modify

Based on project structure, likely files:

```
backend/src/
├── core/
│   ├── streaming_pipeline.py      [ADD projection inference]
│   └── market_state.py             [CHECK for stream buffer]
├── gateway/
│   ├── websocket_handler.py        [ADD pentaview publisher]
│   └── publishers/                 [CHECK for existing publishers]
├── ml/
│   └── projection/
│       └── stream_projector.py     [VERIFY exists]
└── pipeline/
    └── pipelines/
        └── pentaview_pipeline.py   [ALREADY UPDATED to 30s]
```

## Summary

**Viewer Status**: ✅ Ready  
**Models Status**: ✅ Trained  
**Backend Integration**: ⚠️ Not Yet Implemented  

The visualization is **fully functional and ready to display data**. You just need to connect the Core service projection inference to the Gateway WebSocket publisher.

**Estimated Integration Time**: 2-3 hours to implement Core→Gateway flow

---

**Last Updated**: 2025-12-31  
**Model Version**: v30s_20251115_20251215  
**Viewer Port**: http://localhost:5000  
**Gateway Port**: ws://localhost:8000/ws/stream
