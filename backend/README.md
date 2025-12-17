# Backend - 0DTE Options Flow Monitor

FastAPI backend for real-time SPY 0DTE options flow tracking with intelligent historical replay.

## Quick Start

```bash
# Install dependencies
uv sync

# Run in live mode
uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000

# Run in replay mode
REPLAY_MODE=true uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

## Environment Configuration

Create `.env` file:

```bash
# Required
POLYGON_API_KEY=your_api_key

# Optional
S3_BUCKET=your_bucket
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=your_key
S3_SECRET_KEY=your_secret

# Mode
REPLAY_MODE=false
```

## Module Architecture

### `main.py`
- FastAPI application with lifespan management
- Coordinates all background tasks
- WebSocket endpoint `/ws/stream` for frontend
- Mode selection (live vs replay)

### `replay_engine.py`
- Historical data replay simulation
- Fetches from local cache (via `historical_cache.py`)
- Configurable speed and time windows
- Respects market hours (9:30 AM - 4:00 PM ET)

### `stream_ingestor.py`
- Live Polygon WebSocket client
- Options market data ingestion
- Dynamic subscription management
- Non-blocking message handling via `asyncio.Queue`

### `historical_cache.py`
- **Fetch once, replay forever**
- Local Parquet storage by date/ticker
- DuckDB-powered efficient queries
- Intelligent partial data handling

### `strike_manager.py`
- Dynamic ATM ± 3 strike tracking
- Calculates target strikes from SPY price
- Generates Polygon-format option tickers
- Diff logic for subscribe/unsubscribe

### `flow_aggregator.py`
- Consumes trade messages from queue
- Maintains in-memory state store
- Aggregates: volume, premium, delta flow
- Integrates Greeks from enricher

### `greek_enricher.py`
- Fetches options chain snapshots (60s interval)
- Caches Delta, Gamma, Theta, Vega
- Provides lookup for aggregator
- Single REST call per refresh cycle

### `socket_broadcaster.py`
- WebSocket connection manager
- Broadcasts state snapshots to frontend
- 250ms throttled updates
- Handles client connect/disconnect

### `persistence_engine.py`
- Parquet persistence for processed flow
- Buffered writes (5000 records or 60s)
- Date-partitioned storage
- S3 or local filesystem support

## Data Flow

```
┌─────────────────┐
│  Polygon API    │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Cache?  │
    └──┬───┬───┘
       │   │
    Yes│   │No
       │   │
       │   └──► Fetch & Cache
       │
       ▼
┌──────────────────┐
│ Stream Ingestor  │
│   OR             │
│ Replay Engine    │
└────────┬─────────┘
         │
         ▼
   asyncio.Queue
         │
         ▼
┌──────────────────┐
│ Flow Aggregator  │◄─── Greek Enricher (60s)
└────────┬─────────┘
         │
         ▼
  State Store (in-memory)
         │
         ▼
┌──────────────────┐
│Socket Broadcaster│──► Frontend
└──────────────────┘
```

## Key Patterns

### Async Execution
All I/O operations use `asyncio`:
- WebSocket handling
- Queue processing
- REST API calls (via `run_in_executor`)
- Background loops (Greek refresh, strike monitoring)

### Error Handling
Graceful degradation:
1. Try primary method (e.g., options chain for SPY price)
2. Fallback to secondary (e.g., previous close)
3. Hard fallback if all fail (e.g., $600.00)

### Cache Strategy
- Check local Parquet files first
- Fetch from API only if missing
- Only cache complete trading sessions
- Never refetch existing data

## Testing

```bash
# Run all tests
uv run pytest

# Run specific module
uv run pytest tests/test_cache.py

# With coverage
uv run pytest --cov=src --cov-report=html
```

## Debugging

### Enable verbose logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check cache stats
```python
from src.historical_cache import HistoricalDataCache
cache = HistoricalDataCache()
print(cache.get_cache_stats())
```

### Monitor WebSocket
Check for these log messages:
- `✓ SPY Price from Options Chain` - Price fetching works
- `Fetching history for X tickers` - Data fetch initiated
- `💾 Cached X trades` - Data cached successfully
- `Loaded X historical trades` - Replay ready

## Common Issues

### "NOT_AUTHORIZED" errors
**Fixed:** Now uses `list_snapshot_options_chain()` to get SPY price. This is included in Options Advanced plan.

### Slow first run
**Expected:** First replay fetches from API (30-60s). Subsequent runs use cache (< 1s).

### Missing imports
```bash
uv sync  # Reinstall all dependencies
```

## Dependencies

Managed via `pyproject.toml`:
- `fastapi[standard]` - Web framework with CLI
- `polygon-api-client` - Data provider SDK
- `duckdb` - Embedded analytics DB
- `pandas` - Data manipulation
- `python-dotenv` - Environment config
- `websockets` - WebSocket support
- `s3fs` - S3 filesystem (optional)

## Production Deployment

```bash
# Production server
uv run fastapi run src/main.py --host 0.0.0.0 --port 8000

# With workers
uv run uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000
```

**Note:** Current architecture uses in-memory state. For multi-worker deployments, add Redis for shared state.

