# SPYMASTER - 0DTE Options Flow Monitor

Real-time SPY 0DTE options flow monitoring system with dynamic strike tracking and historical replay capabilities.

## Architecture

**Stack:**
- **Backend:** Python 3.13+ (FastAPI, AsyncIO)
- **Frontend:** Angular 21 (Signals Architecture)
- **Data Provider:** Polygon.io Options Advanced Plan
- **Storage:** Local Parquet cache + optional S3
- **Tooling:** `uv` for Python dependency management

**System:** Apple M4 Silicon / 128GB RAM optimized

## Quick Start

### Prerequisites

```bash
# Python 3.13+
python --version

# Node.js for frontend
node --version

# uv for Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup

Create `backend/.env`:

```bash
# Required
POLYGON_API_KEY=your_polygon_api_key_here

# Optional - for S3 persistence
S3_BUCKET=your_bucket_name
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Mode selection
REPLAY_MODE=false  # Set to 'true' for historical replay
```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd backend
uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install  # First time only
npm start
```

**Access:** http://localhost:4200

## Key Features

### 1. Dynamic Strike Management
- Monitors SPY price via options chain (`underlying_asset.price`)
- Tracks ATM ± 3 strikes (7 strikes total: 3 puts + ATM + 3 calls)
- Auto-subscribes/unsubscribes as price moves
- Updates every 60s or on >$0.50 deviation

### 2. Historical Data Cache
- **Fetch once, replay forever**
- Stores raw trades in `backend/data/historical/trades/YYYY-MM-DD/{ticker}.parquet`
- Intelligent caching: only fetches missing data
- Complete sessions cached; incomplete sessions refetched

### 3. Replay Engine
- Simulates WebSocket stream from historical data
- Configurable speed (1.0x = real-time)
- Respects market hours (9:30 AM - 4:00 PM ET)
- Seamlessly switches between live/replay modes

### 4. Real-time Flow Aggregation
- Cumulative volume & premium tracking
- Net delta/gamma flow calculations
- Greeks integration from chain snapshots
- 250ms broadcast interval to frontend

## Operating Modes

### Live Mode (Default)

```bash
cd backend
uv run fastapi dev src/main.py
```

- Connects to Polygon WebSocket
- Real-time options trade data
- Dynamic strike tracking
- Persists to Parquet for future replay

### Replay Mode

```bash
cd backend
REPLAY_MODE=true uv run fastapi dev src/main.py
```

- Uses local cache (no API calls if cached)
- Replays full trading day (9:30 AM - 4:00 PM ET)
- Adjustable speed via code (`speed=1.0`)
- Perfect for testing and development

## Project Structure

```
spymaster/
├── backend/
│   ├── src/
│   │   ├── main.py                    # FastAPI app & lifecycle
│   │   ├── replay_engine.py           # Historical replay logic
│   │   ├── stream_ingestor.py         # Live WebSocket client
│   │   ├── strike_manager.py          # Dynamic strike tracking
│   │   ├── flow_aggregator.py         # Trade → metrics aggregation
│   │   ├── greek_enricher.py          # Greeks from chain snapshots
│   │   ├── socket_broadcaster.py      # WebSocket output to frontend
│   │   ├── persistence_engine.py      # Parquet persistence
│   │   └── historical_cache.py        # Smart historical data cache
│   ├── data/
│   │   └── historical/trades/         # Cached Parquet files
│   └── pyproject.toml                 # uv dependencies
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.ts       # Root component
│   │   │   ├── data-stream.service.ts # WebSocket client
│   │   │   └── strike-grid/           # Real-time grid display
│   │   └── main.ts
│   └── package.json
│
├── APP.md                              # Detailed architecture spec
├── RULES.md                            # Development guidelines
└── README.md                           # This file
```

## Data Flow

```
Polygon API → Cache Check → Historical Cache (if exists)
                ↓ (if not cached)
        Live Stream / REST Fetch
                ↓
          Stream Ingestor
                ↓
          asyncio.Queue
                ↓
         Flow Aggregator ← Greek Enricher (60s)
                ↓
          State Store (in-memory)
                ↓
      Socket Broadcaster (250ms)
                ↓
          Frontend (WebSocket)
                ↓
         Strike Grid Display
```

## API Endpoints

**WebSocket:**
- `ws://localhost:8000/ws/stream` - Real-time flow data

**REST (Auto-generated):**
- `http://localhost:8000/docs` - Interactive API docs
- `http://localhost:8000/redoc` - ReDoc documentation

## Development Workflow

### Adding a New Feature

1. **Search first:** Check existing patterns before creating new files
2. **Plan:** Outline 3-5 concrete steps
3. **Implement:** Single canonical solution (no fallbacks)
4. **Verify:** Run with `uv run pytest` or manual test

### Testing Changes

```bash
# Backend tests
cd backend
uv run pytest

# Frontend tests
cd frontend
npm test

# Manual integration test
# Terminal 1: Start backend in replay mode
cd backend && REPLAY_MODE=true uv run fastapi dev src/main.py

# Terminal 2: Start frontend
cd frontend && npm start

# Browser: Open http://localhost:4200 and verify data flow
```

### Debugging

**Backend:**
```bash
# Check logs for errors
cd backend
uv run fastapi dev src/main.py

# Look for:
# - "✓ SPY Price from Options Chain" - Price fetching works
# - "📦 Cache: N files" - Cache statistics
# - "Loaded X historical trades" - Data loaded
# - "Replay Complete" - Replay finished
```

**Frontend:**
```bash
# Browser console shows:
# - "🔌 Connecting to 0DTE Stream..."
# - "✅ Connected to Stream"
# - Check Network tab for WebSocket connection
```

**Common Issues:**

1. **"NOT_AUTHORIZED" for SPY price**
   - ✅ Fixed: Now uses `list_snapshot_options_chain()` 
   - Included in Options Advanced plan

2. **No cached data**
   - First run fetches from API (takes 30-60s)
   - Subsequent runs use cache (< 1s)

3. **Frontend not showing data**
   - Check WebSocket connection in Network tab
   - Verify backend is broadcasting: look for "Loaded X trades"
   - Check browser console for errors

## Cache Management

**View cache stats:**
```python
from src.historical_cache import HistoricalDataCache
cache = HistoricalDataCache()
print(cache.get_cache_stats())
```

**Clear cache (force refetch):**
```bash
rm -rf backend/data/historical/trades/
```

**Cache location:**
```
backend/data/historical/trades/
└── YYYY-MM-DD/
    ├── O_SPY{date}C{strike}.parquet
    ├── O_SPY{date}P{strike}.parquet
    └── ...
```

## Polygon API Plan Requirements

**Minimum Required:**
- Polygon.io **Options Advanced** plan
- Includes: Options trades, quotes, chain snapshots, underlying price
- **Does NOT require** separate Stocks plan (underlying price included in chain)

**API Calls:**
- `list_snapshot_options_chain()` - SPY price + Greeks (every 60s)
- `list_trades()` - Historical options trades (cached locally)
- WebSocket - Real-time options trades (unlimited)

## Performance

**Typical Numbers:**
- Cache hit: < 1 second to load full day
- Cache miss: 30-60 seconds (one-time per day)
- Memory usage: ~500MB for full day replay
- Broadcast latency: < 250ms (real-time mode)

## Troubleshooting

### Backend won't start

```bash
cd backend
uv sync  # Reinstall dependencies
uv run fastapi dev src/main.py
```

### Frontend build errors

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### WebSocket disconnects

- Check if backend is running
- Verify no firewall blocking port 8000
- Check browser console for errors

### No data in replay mode

- Verify cache exists: `ls backend/data/historical/trades/2025-12-16/`
- Check logs for "Loaded X trades" message
- If no trades found, cache may be empty - will refetch

## Contributing

See `RULES.md` for development guidelines.

## License

Proprietary - Internal Use Only

