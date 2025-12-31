# Pentaview Stream Viewer

Real-time visualization of Pentaview projection models with TradingView Lightweight Charts.

## Features

- **2-Minute OHLCV Candles** (left Y-axis): Price action from ES futures
- **Stream Overlays** (right Y-axis, -1 to +1): Live normalized streams updating every 30 seconds
  - `sigma_p` (Pressure): Red line
  - `sigma_m` (Momentum): Blue line  
  - `sigma_f` (Flow): Green line
  - `sigma_b` (Barrier): Orange line
  - `sigma_r` (Structure): Purple line
- **Projection Bands** (right Y-axis): 5-minute ahead forecasts (10 bars @ 30s cadence)
  - `q10`: Lower confidence bound (10th percentile)
  - `q50`: Median forecast (50th percentile)
  - `q90`: Upper confidence bound (90th percentile)
- **Real-Time Updates**: WebSocket connection to Gateway for live streaming data
- **Professional UI**: Dark TradingView theme with live status indicators

## Architecture

```
Gateway (ws://localhost:8000/ws/stream)
    ↓
Flask WebSocket Proxy (demo/app.py)
    ↓
Frontend (demo/templates/index.html)
    ↓
TradingView Lightweight Charts
```

The Flask app acts as a WebSocket proxy between the Gateway and the browser, forwarding stream data in real-time.

## Prerequisites

### 1. Gateway Service Running

```bash
# Gateway must be running on port 8000
docker ps | grep spymaster-gateway
# Should show: spymaster-gateway ... Up ... 0.0.0.0:8000->8000/tcp
```

### 2. Replay Engine Running (for historical playback)

```bash
cd backend
REPLAY_SPEED=1.0 REPLAY_DATE=2025-12-18 \
  REPLAY_USE_BRONZE_FUTURES=true REPLAY_FUTURES_SYMBOL=ES \
  uv run python -m src.ingestion.databento.replay
```

### 3. Trained Projection Models

Models should exist at:
```
backend/data/ml/projection_models/
├── projection_sigma_p_v30s_20251115_20251215.joblib
├── projection_sigma_m_v30s_20251115_20251215.joblib
├── projection_sigma_f_v30s_20251115_20251215.joblib
├── projection_sigma_b_v30s_20251115_20251215.joblib
└── projection_sigma_r_v30s_20251115_20251215.joblib
```

## Installation

```bash
cd demo

# Using uv (Python 3.12)
uv venv --python 3.12
uv pip sync

# Dependencies are managed via pyproject.toml
```

**Note**: This project uses `uv` for dependency management with Python 3.12. If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Start the Viewer

```bash
cd demo
./run.sh
```

Or manually:
```bash
uv run python app.py
```

Then open your browser to: **http://localhost:5000**

### Expected Behavior

1. **Connection**: Status indicator shows "Connected" (green dot)
2. **Price Candles**: 2-minute bars appear on the left Y-axis
3. **Stream Overlays**: 5 colored lines appear on the right Y-axis (-1 to +1 range)
4. **Projections**: Dashed orange lines extend 5 minutes into the future
5. **Updates**: Legend values update every 30 seconds as new stream samples arrive
6. **Stats**: Header shows bar count, update count, and model version

### Troubleshooting

**"Connecting..." never changes to "Connected"**
- Check Gateway is running: `curl http://localhost:8000/health`
- Check WebSocket endpoint: `docker logs -f spymaster-gateway`
- Verify replay engine is publishing data

**No candles appearing**
- Verify replay engine is running and publishing trades
- Check Core service logs: `docker logs -f spymaster-core`
- Confirm bronze data exists for replay date

**Streams show "—" values**
- Verify Pentaview pipeline is generating streams
- Check Gateway is emitting stream payloads
- Inspect browser console for WebSocket messages

**Projections not appearing**
- Verify projection models are loaded in Core service
- Check 16-bar minimum history requirement (need 8 minutes of warmup)
- Confirm inference is running every 30 seconds

## Data Flow

### Expected WebSocket Payload

The Gateway should emit messages with this structure:

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
    "q10": [{"time": 1702900830, "value": 0.40}, {"time": 1702900860, "value": 0.38}],
    "q50": [{"time": 1702900830, "value": 0.45}, {"time": 1702900860, "value": 0.47}],
    "q90": [{"time": 1702900830, "value": 0.50}, {"time": 1702900860, "value": 0.56}]
  }
}
```

### Integration Notes

**If Gateway doesn't emit this format:**

You'll need to adapt `src/gateway/` to publish Pentaview stream data. See the guide section "Part 4: Projection Inference at 30-Second Cadence" for implementation details.

The Core service should:
1. Generate stream bars every 30 seconds (from Stage 16 state table)
2. Compute Pentaview streams (sigma_p, sigma_m, etc.)
3. Run projection inference when history >= 16 bars
4. Emit to Gateway via NATS or direct call
5. Gateway forwards to WebSocket clients

## Configuration

Edit `demo/app.py` to change:

```python
# Gateway WebSocket URL
GATEWAY_WS_URL = "ws://localhost:8000/ws/stream"

# Projection model version
model_version = "v30s_20251115_20251215"
```

## Performance

- **Update Frequency**: 30 seconds (matches Pentaview inference cadence)
- **Data Volume**: ~5 KB per update (5 streams + projections)
- **Latency**: <50ms from Gateway to browser
- **Memory**: ~20 MB for 6.5 hours of data (390 bars × 30s)

## Technical Stack

- **Python**: 3.12 (managed via `uv`)
- **Backend**: Flask 3.0 + Flask-Sock (WebSocket)
- **Frontend**: Vanilla JavaScript + TradingView Lightweight Charts 4.1
- **WebSocket**: Real-time bidirectional communication
- **Chart Library**: Lightweight Charts (60 FPS, smooth zooming/panning)
- **Package Manager**: `uv` (fast, deterministic Python package installer)

## Next Steps

### For Development

1. **Add historical mode**: Load and display full day of historical streams
2. **Add controls**: Pause/resume, speed controls, time scrubbing
3. **Add metrics panel**: Show projection accuracy, stream statistics
4. **Add level markers**: Overlay PM_HIGH, OR_HIGH, SMA levels on chart

### For Production

1. **Add authentication**: Secure WebSocket connection
2. **Add error recovery**: Automatic reconnection with backoff
3. **Add data buffering**: Handle network interruptions gracefully
4. **Add performance monitoring**: Track update latency, frame drops

## Related Documentation

- **Training Guide**: `../PENTAVIEW_TRAINING_AND_REPLAY_GUIDE.md`
- **Pipeline Architecture**: `../backend/DATA_ARCHITECTURE.md`
- **Frontend Contract**: `../frontend/README.md`

## Support

For issues:
1. Check Gateway health: `curl http://localhost:8000/health`
2. Check Docker services: `docker ps | grep spymaster`
3. Check browser console for errors
4. Check Flask logs in terminal

---

**Model Version**: v30s_20251115_20251215  
**Last Updated**: 2025-12-31  
**Inference Cadence**: Every 30 seconds  
**Horizon**: 10 bars (5 minutes @ 30s cadence)
