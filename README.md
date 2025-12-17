# SPYMASTER: 0DTE Options Flow Monitor
## Technical Specification for AI Coding Agents

**Table of Contents:**
- [AI Agent Development Rules](#ai-agent-development-rules)
- [System Purpose (WHY)](#system-purpose-why)
- [Architecture Overview (WHAT)](#architecture-overview-what)
- [Backend Components](#backend-components)
- [Frontend Components](#frontend-components)
- [Development Workflow](#development-workflow)
- [Replay Mode (CRITICAL)](#replay-mode-critical-understanding)
- [Extension Points](#extension-points-where-to-add-features)
- [File Structure Map](#file-structure-map)
- [Quick Reference Commands](#quick-reference-commands)
- [Critical Implementation Notes](#critical-implementation-notes)

---

## AI Agent Development Rules

**Environment:**
- Apple M4 Silicon, 128GB RAM
- Python tooling: `uv` exclusively (NOT pip, NOT raw python)
- `.venv` is single source of truth for dependencies

**Workflow for Every Task:**
1. **Discover:** Search codebase for existing patterns before creating new files
2. **Plan:** Outline 3-5 concrete steps, state verification method
3. **Implement:** Single canonical solution (no fallbacks, no legacy paths)
4. **Verify:** Run tests with `uv run <command>` or manual integration test

**Code Principles:**
- Search first, create second
- Fail fast with clear errors (no complex fallback logic)
- Atomic changes (don't break unrelated domains)
- One implementation per feature

---

## System Purpose (WHY)

**Product Intent:**
Traders run TradingView for price action and technical confluence. This system provides **real-time options flow sentiment** as a confirmation instrument for SPY 0DTE positions.

**Core Questions Answered:**
1. Where is flow concentrating? (strikes/side)
2. How fast is flow building? (velocity)
3. Is flow rate accelerating or decelerating? (2nd/3rd derivatives)
4. Is pressure building above or below ATM?

**Technical Goals:**
- Sub-250ms latency from Polygon WebSocket → Frontend display
- Dynamic strike tracking (ATM ± 3 strikes, auto-adjusts)
- Historical replay from local cache (zero API calls)
- Real-time Greek-enriched flow aggregation (delta/gamma notional impact)

**Not Goals:**
- Not replicating TradingView (no price charts, SMAs, confluence indicators)
- Not handling multi-asset (SPY 0DTE only)
- Not providing BBO/quote-based aggressor classification

---

## Architecture Overview (WHAT)

**Stack:**
- Backend: Python 3.13+, FastAPI, AsyncIO, Polygon WebSocket
- Frontend: Angular 21 Signals, TradingView Lightweight Charts v4
- Data: Parquet cache (local), optional S3 persistence
- Tooling: `uv` for Python dependencies

**Data Flow:**
```
Polygon WebSocket (options trades)
  ↓
StreamIngestor (async queue)
  ↓
FlowAggregator (in-memory state store)
  ← GreekEnricher (60s REST snapshots)
  ↓
SocketBroadcaster (250ms throttle)
  ↓
Frontend WebSocket (TradingView charts + strike ladder)
```

**Key Invariant:**
Backend broadcasts **cumulative** per-contract metrics, not per-trade. Frontend computes derivatives (velocity/accel/jerk) by differencing snapshots.

---

## Backend Components

### Environment Setup
```bash
# Required: backend/.env
POLYGON_API_KEY=your_key_here
REPLAY_MODE=false              # true = replay from cache
S3_BUCKET=optional_bucket       # optional
```

### Core Modules

#### `src/main.py`
FastAPI lifecycle. Spawns background tasks:
- `greek_enricher.start_snapshot_loop()` (60s interval)
- `stream_ingestor.run_async()` OR `replay_engine.run()` (mode-dependent)
- `processing_loop()` (consumes queue → aggregator)
- `broadcast_loop()` (250ms snapshot broadcast)
- `strike_monitor_loop()` (60s, updates subscriptions)

WebSocket endpoint: `/ws/stream`

#### `src/strike_manager.py`
**Purpose:** Determine active strikes based on SPY price.

**Logic:**
1. Fetch SPY price via `list_snapshot_options_chain()` (includes underlying price in options plan)
2. Calculate ATM = `round(spy_price)`
3. Generate 7 strikes: `[ATM-3, ATM-2, ATM-1, ATM, ATM+1, ATM+2, ATM+3]`
4. Format tickers: `O:SPY{YYMMDD}{C/P}{STRIKE_8_DIGIT}`
5. Diff current vs target → return (add_list, remove_list)

**Update Triggers:** Every 60s OR >$0.50 price deviation

#### `src/stream_ingestor.py`
**Purpose:** Polygon WebSocket client for live mode.

**Critical Implementation Details:**
- `market='options'`, subscribe to trade ticks only
- `on_message`: Immediately offload to `asyncio.Queue` (DO NOT block)
- Exposes `update_subs(add, remove)` for dynamic subscription
- Handles reconnection via Polygon client auto-reconnect

**Key Method:**
```python
def update_subs(self, add: List[str], remove: List[str]):
    if remove:
        self.ws.unsubscribe(*remove)
    if add:
        self.ws.subscribe(*add)
```

#### `src/replay_engine.py`
**Purpose:** Simulate WebSocket stream from cached Parquet files.

**Current Behavior (CRITICAL):**
- Replays **last 10 minutes** of most recent cached day (3:50-4:00 PM ET)
- Loads from `data/historical/trades/YYYY-MM-DD/*.parquet`
- Infers SPY price from cached strike filenames (median strike)
- Replays ALL cached tickers (not just ATM ± 3)
- Speed: 1.0x (configurable in code)

**Key Parameters:**
```python
async def run(self, minutes_back: int = 10, speed: float = 1.0)
```

**Data Loading:**
1. Find most recent date in cache: `cache.get_cache_stats()['cached_dates'][-1]`
2. Calculate time window: `market_close - timedelta(minutes=10)` → `market_close`
3. Load all Parquet files: `cache.get_cached_trades(ticker, date, start, end)`
4. Sort by timestamp, replay with timing delays

**Why This Matters:**
Replay mode gives you ~8,000-10,000 real trades to test frontend without API calls.

#### `src/historical_cache.py`
**Purpose:** Smart Parquet cache to avoid redundant Polygon API calls.

**Cache Strategy:**
- Complete trading sessions (9:30 AM - 4:00 PM ET): cache forever
- Incomplete sessions (during market hours): fetch fresh, don't cache
- File format: `O_SPY251216C00600000.parquet` (underscore replaces colon)

**Key Method:**
```python
def fetch_or_get_trades(client, ticker, start, end) -> List[Dict]:
    # Returns cached if exists, else fetches from Polygon REST API
```

**Cache Location:** `backend/data/historical/trades/YYYY-MM-DD/{ticker}.parquet`

#### `src/flow_aggregator.py`
**Purpose:** Transform raw trades into metrics.

**State Store:** `Dict[ticker: str, FlowMetrics]`

**FlowMetrics Schema:**
```python
{
    "cumulative_volume": int,
    "cumulative_premium": float,  # price * size * 100
    "net_delta_flow": float,      # Σ(volume * delta)
    "net_gamma_flow": float,      # Σ(volume * gamma)
    "last_price": float,
    "strike_price": float,
    "type": str,  # 'C' or 'P'
    "expiration": str
}
```

**Processing:**
1. Receive trade from queue
2. Lookup Greeks from `GreekEnricher` cache
3. Compute delta notional: `trade.size * delta`
4. Update cumulative metrics for that ticker

#### `src/greek_enricher.py`
**Purpose:** Provide Greeks for flow calculations.

**Background Loop (60s):**
1. Fetch `list_snapshot_options_chain("SPY")` for current expiry
2. Build lookup table: `Dict[ticker, {delta, gamma}]`
3. Used by `FlowAggregator` to enrich trades

**Why 60s:** 0DTE Greeks change fast. 60s is maximum staleness tolerance.

#### `src/socket_broadcaster.py`
**Purpose:** Push state snapshots to frontend.

**Broadcast Loop:**
- Interval: 250ms
- Payload: JSON snapshot of entire `FlowAggregator` state store
- All connected WebSocket clients receive same snapshot

#### `src/persistence_engine.py`
**Purpose:** Write trades to Parquet for future replay.

**Storage Strategy:**
- S3 path: `s3://{bucket}/flatfiles/data/raw/flow/year=YYYY/month=MM/day=DD/`
- Local path: `data/historical/trades/YYYY-MM-DD/`
- Format: Parquet (one file per ticker per day)

**Buffer Logic:**
Batches writes to avoid I/O on every trade.

### Running Backend

**Live Mode:**
```bash
cd backend
uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

**Replay Mode:**
```bash
cd backend
REPLAY_MODE=true uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

**Logs to Monitor:**
- "✓ SPY Reference Price: $XXX" - Strike selection working
- "📦 Found N cached trade files" - Cache loaded
- "Loaded X historical trades. Starting Replay..." - Replay active
- "INFO connection open" - Frontend connected

---

## Frontend Components

### Environment Setup
```bash
cd frontend
npm install
npm start  # Runs on http://localhost:4200
```

**Key Dependency:** `lightweight-charts@4.2.0` (TradingView library)

### Architecture

**State Management:** Angular Signals (no NgRx/Observables for local state)

**Component Tree:**
```
AppComponent
  └── FlowDashboardComponent (layout)
        ├── FlowChartComponent (TradingView charts)
        └── StrikeGridComponent (compact ladder)
```

**Services:**
- `DataStreamService`: WebSocket connection, exposes `flowData` signal
- `FlowAnalyticsService`: Computes derivatives (available but not visualized)

### Core Components

#### `src/app/data-stream.service.ts`
**Purpose:** WebSocket client.

**Key Implementation:**
```typescript
private ws: WebSocket;
public flowData = signal<FlowMap>({});

connect() {
  this.ws = new WebSocket('ws://localhost:8000/ws/stream');
  this.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    this.flowData.set(data);  // Triggers reactive update
  };
}
```

**Type Definition:**
```typescript
interface FlowMap {
  [ticker: string]: FlowMetrics;
}

interface FlowMetrics {
  cumulative_volume: number;
  cumulative_premium: number;
  net_delta_flow: number;
  net_gamma_flow: number;
  strike_price: number;
  type: 'C' | 'P';
  expiration: string;
  last_price: number;
}
```

#### `src/app/flow-chart/flow-chart.component.ts`
**Purpose:** TradingView Lightweight Charts integration.

**Chart Configuration:**
```typescript
createChart(container, {
  layout: { background: '#0f1419', textColor: '#a0aec0' },
  grid: { vertLines: '#1a1f2e', horzLines: '#1a1f2e' },
  timeScale: { timeVisible: true, secondsVisible: true }
});
```

**Series Types:**
1. **Area Series:** Cumulative premium flow (green gradient)
2. **Histogram Series:** Volume bars (colored by net delta: green = positive, red = negative)

**Data Update Pattern:**
```typescript
effect(() => {
  const data = this.dataService.flowData();
  
  // Aggregate all strikes
  let totalPremium = 0, totalVolume = 0, totalDelta = 0;
  for (const ticker in data) {
    totalPremium += data[ticker].cumulative_premium;
    totalVolume += data[ticker].cumulative_volume;
    totalDelta += data[ticker].net_delta_flow;
  }
  
  // Add to rolling buffer (keep last 500 points)
  this.dataPoints.set(now, {premium, volume, delta});
  
  // Update chart
  this.premiumSeries.setData(premiumData);
  this.volumeSeries.setData(volumeData);
});
```

#### `src/app/strike-grid/strike-grid.component.ts`
**Purpose:** Compact strike ladder with heat mapping.

**Layout:**
```
┌─────────┬────────┬─────────┐
│  Calls  │ Strike │  Puts   │
├─────────┼────────┼─────────┤
│ $12,345 │  682   │  $8,901 │
│   1,234 │        │    456  │
├─────────┼────────┼─────────┤
│  ...    │  ...   │   ...   │
```

**Heat Mapping:**
- Premium cells background: `rgba(34,197,94, alpha)` for calls, `rgba(244,63,94, alpha)` for puts
- Alpha intensity: `0.15 + (velocity / maxVelocity) * 0.55`
- Velocity computed by `FlowAnalyticsService.perStrikeVel()` signal

**Sorting:** High strike → low strike (descending)

#### `src/app/flow-analytics.service.ts`
**Purpose:** Compute derivatives from cumulative snapshots.

**What It Computes (but not currently visualized):**
- ATM strike: median of active strikes
- Buckets: call_above, call_below, put_above, put_below (relative to ATM)
- Derivatives: velocity (d1), acceleration (d2), jerk (d3)
- Timescales: 1s, 5s, 30s windows

**Usage in Code:**
```typescript
perStrikeVel = computed(() => {
  // Returns {[strike]: {call: {premium_vel}, put: {premium_vel}}}
});
```

**Why Not Visualized:**
Analytics are computed and ready, but current UI focuses on TradingView charts. Future UI can expose multi-timescale derivative panels.

### File Structure
```
frontend/src/app/
├── app.component.ts                 # Root component
├── data-stream.service.ts           # WebSocket client
├── flow-analytics.service.ts        # Derivative computations
├── flow-dashboard/                  # Main layout
│   └── flow-dashboard.component.ts
├── flow-chart/                      # TradingView charts
│   └── flow-chart.component.ts
└── strike-grid/                     # Strike ladder
    ├── strike-grid.component.ts
    └── strike-grid.component.html
```

### Styling
**No Tailwind:** Removed due to build issues. Uses custom CSS in `src/styles.css`.

**Theme:**
- Background: `#0f1419` (dark)
- Cards: `#1a1f2e` with `#2d3748` borders
- Green: `#48bb78` (calls, positive delta)
- Red: `#f56565` (puts, negative delta)

---

## Development Workflow

### Tooling Rules (STRICT)

**Python:**
- Use `uv` exclusively (not pip, not raw python)
- Commands: `uv run <script>`, `uv add <package>`, `uv sync`
- `.venv` is single source of truth

**Frontend:**
- Use `npm` for Node packages
- No Tailwind (removed)
- Lightweight Charts v4 (not v5 - API incompatible)

### Code Principles

1. **Search First:** Before creating files, search for existing patterns
2. **Single Implementation:** One canonical solution, no fallbacks
3. **Fail Fast:** Clear errors when prerequisites missing
4. **Atomic Changes:** Don't break unrelated domains

### Workflow Sequence

For non-trivial tasks:

1. **Discover:** Search codebase for existing patterns
2. **Plan:** Outline 3-5 concrete steps
3. **Implement:** Write direct implementation
4. **Verify:** Test with `uv run` or manual browser test

### Testing

**Backend:**
```bash
cd backend
uv run pytest                                    # Run tests
uv run fastapi dev src/main.py                   # Manual test
```

**Frontend:**
```bash
cd frontend
npm test                                         # Run tests (if any)
npm start                                        # Manual test at localhost:4200
```

**Integration Test:**
```bash
# Terminal 1
cd backend && REPLAY_MODE=true uv run fastapi dev src/main.py

# Terminal 2
cd frontend && npm start

# Browser: http://localhost:4200
# Check: Charts rendering, WebSocket connected, data flowing
```

### Common Issues

**"Loaded 0 historical trades"**
- Cache is empty. Run in live mode first to populate.
- Check: `ls backend/data/historical/trades/`

**"Charts not rendering"**
- Check browser console for errors
- Verify: `npm list lightweight-charts` shows v4.2.0
- Clear browser cache

**"WebSocket not connecting"**
- Check browser Network tab for `/ws/stream`
- Verify backend on port 8000: `lsof -i :8000`
- Check backend logs for "connection open"

**"Frontend showing old data"**
- Backend restarted (cumulative values reset)
- Frontend handles negative deltas as 0 (built-in guard)

---

## Replay Mode (CRITICAL UNDERSTANDING)

### Why Replay Mode Exists

**Problem:** Live market is 6.5 hours/day. Can't efficiently develop frontend during off-hours.

**Solution:** Cache real trades to Parquet, replay them at will.

### How It Works

**Activation:**
```bash
cd backend
REPLAY_MODE=true uv run fastapi dev src/main.py
```

**Replay Behavior:**
1. Finds most recent cached date: `data/historical/trades/YYYY-MM-DD/`
2. Loads **last 10 minutes** of that day: 3:50 PM - 4:00 PM ET
3. Loads ALL cached tickers (typically 14 files, ~8,000-10,000 trades)
4. Replays trades in timestamp order with timing delays (speed = 1.0x)
5. Broadcasts to frontend via same WebSocket endpoint

**Key Difference from Live:**
- No Polygon API calls (reads Parquet only)
- No dynamic strike management (uses whatever is cached)
- Repeatable (same data every run)

### Implementation Details

**File:** `src/replay_engine.py`

**Entry Point:**
```python
async def run(self, minutes_back: int = 10, speed: float = 1.0):
    # 1. Find most recent cached date
    cached_dates = self.cache.get_cache_stats()['cached_dates']
    most_recent_date = cached_dates[-1]
    
    # 2. Calculate time window
    market_close = datetime(..., hour=16, minute=0, tzinfo=ZoneInfo("America/New_York"))
    start_time = market_close - timedelta(minutes=minutes_back)
    end_time = market_close
    
    # 3. Load all cached tickers
    cache_dir = self.cache.cache_dir / most_recent_date
    cached_files = list(cache_dir.glob("*.parquet"))
    
    # 4. Load trades from cache
    all_trades = self._load_trades_from_cache_only(tickers, date, start, end)
    
    # 5. Replay with timing
    for trade in all_trades:
        offset_ms = trade['t'] - start_ts
        wait_ms = (offset_ms / speed) - elapsed_ms
        if wait_ms > 0:
            await asyncio.sleep(wait_ms / 1000.0)
        await self.queue.put(MockMessage(trade))
```

**Speed Parameter:**
- 1.0 = real-time (10 minutes of data takes 10 minutes to replay)
- 2.0 = 2x speed (10 minutes in 5 minutes)
- 0.5 = half speed (10 minutes in 20 minutes)

**Adjusting Replay Window:**
To replay different time ranges, edit `replay_engine.py`:
```python
# Change to replay full day:
start_time = market_open  # 9:30 AM
end_time = market_close    # 4:00 PM

# Change to replay lunch hour:
start_time = market_close.replace(hour=12, minute=0)
end_time = market_close.replace(hour=13, minute=0)
```

### Cache Management

**View Cache:**
```bash
ls backend/data/historical/trades/
ls backend/data/historical/trades/2025-12-16/
```

**Clear Cache (force refetch):**
```bash
rm -rf backend/data/historical/trades/
```

**Cache Stats (Python):**
```python
from src.historical_cache import HistoricalDataCache
cache = HistoricalDataCache()
print(cache.get_cache_stats())
# Output: {'total_files': 14, 'total_size_mb': 4.63, 'cached_dates': ['2025-12-16']}
```

---

## Extension Points (Where to Add Features)

### Backend Extensions

**Add New Metric to Flow:**
1. Update `FlowMetrics` schema in `flow_aggregator.py`
2. Compute in `FlowAggregator.process_message()`
3. Include in state store snapshot
4. Frontend will receive automatically via WebSocket

**Add New Data Source:**
1. Create new module following `stream_ingestor.py` pattern
2. Write to same `asyncio.Queue`
3. `FlowAggregator` handles agnostically

**Change Strike Selection Logic:**
1. Edit `StrikeManager.get_target_strikes()`
2. Example: ATM ± 5 instead of ± 3
3. Example: Weight by volume instead of uniform range

**Adjust Greek Refresh Rate:**
1. Edit `greek_enricher.py`: change `await asyncio.sleep(60)` to `30`
2. Trade-off: More API calls vs fresher Greeks

### Frontend Extensions

**Add Separate Call/Put Charts:**
```typescript
// In flow-chart.component.ts
this.callSeries = this.chart.addAreaSeries({...});
this.putSeries = this.chart.addAreaSeries({...});

// In updateChart():
for (const ticker in flowData) {
  if (flowData[ticker].type === 'C') {
    callPremium += flowData[ticker].cumulative_premium;
  } else {
    putPremium += flowData[ticker].cumulative_premium;
  }
}
```

**Visualize Analytics Service Data:**
```typescript
// In flow-dashboard.component.ts
<div>
  @for (bucket of ['call_above', 'put_below']; track bucket) {
    <div>{{ bucket }}: {{ analytics.getLatestByBucket()[bucket] }}</div>
  }
</div>
```

**Add Strike-Level Time Series:**
```typescript
// When user clicks a strike, show chart for just that strike
selectStrike(strike: number) {
  this.selectedStrikeData = this.dataService.flowData().filter(
    m => m.strike_price === strike
  );
  // Render in separate chart
}
```

**Add TradingView Indicators:**
Requires upgrading to TradingView Advanced Charts (not Lightweight).
1. Apply for access: https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/
2. Replace Lightweight Charts with Advanced Charts
3. Add custom studies using Study API

### Data Analysis Extensions

**Export to CSV:**
```python
# In backend, add REST endpoint
@app.get("/export/{date}")
def export_data(date: str):
    df = pd.read_parquet(f"data/historical/trades/{date}/*.parquet")
    return df.to_csv()
```

**Run DuckDB Queries:**
```python
import duckdb
con = duckdb.connect()
result = con.execute("""
    SELECT strike_price, SUM(cumulative_premium) as total_premium
    FROM 'data/historical/trades/**/*.parquet'
    WHERE type = 'C'
    GROUP BY strike_price
    ORDER BY total_premium DESC
    LIMIT 10
""").fetchdf()
```

---

## API Endpoints Reference

**WebSocket:**
- `ws://localhost:8000/ws/stream` - Real-time flow snapshots (250ms)

**REST (auto-generated by FastAPI):**
- `http://localhost:8000/docs` - Interactive Swagger UI
- `http://localhost:8000/redoc` - ReDoc documentation

**Payload Format (WebSocket):**
```json
{
  "O:SPY251216C00680000": {
    "cumulative_volume": 1234,
    "cumulative_premium": 123456.78,
    "net_delta_flow": 567.89,
    "net_gamma_flow": 12.34,
    "strike_price": 680.0,
    "type": "C",
    "expiration": "2025-12-16",
    "last_price": 2.45
  },
  "O:SPY251216P00680000": { ... }
}
```

---

## Performance Characteristics

**Backend:**
- Memory: ~500MB for full day replay
- Cache hit: < 1s to load 10 minutes of data
- Cache miss: 30-60s to fetch from Polygon (one-time)
- Broadcast latency: < 250ms

**Frontend:**
- Chart updates: < 16ms (60 FPS)
- WebSocket receive: < 10ms
- Rolling buffer: Last 500 points (~2 minutes at 250ms intervals)

**Polygon API Limits:**
- WebSocket: No strict limit (monitor bandwidth)
- REST snapshots: 5 requests/minute (for Greeks)
- REST historical: 10 requests/minute (for cache population)

---

## File Structure Map

```
spymaster/
├── backend/
│   ├── src/
│   │   ├── main.py                    # FastAPI lifecycle
│   │   ├── replay_engine.py           # Replay from cache
│   │   ├── stream_ingestor.py         # Live WebSocket
│   │   ├── strike_manager.py          # Strike selection
│   │   ├── flow_aggregator.py         # Metrics computation
│   │   ├── greek_enricher.py          # Greeks from REST snapshots
│   │   ├── socket_broadcaster.py      # WebSocket broadcast
│   │   ├── persistence_engine.py      # Parquet writes
│   │   └── historical_cache.py        # Cache management
│   ├── data/historical/trades/        # Parquet cache
│   ├── pyproject.toml                 # uv dependencies
│   └── .env                           # API keys
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.ts               # Root
│   │   │   ├── data-stream.service.ts         # WebSocket
│   │   │   ├── flow-analytics.service.ts      # Derivatives
│   │   │   ├── flow-dashboard/                # Layout
│   │   │   ├── flow-chart/                    # TradingView
│   │   │   └── strike-grid/                   # Ladder
│   │   ├── styles.css                         # Global styles
│   │   └── main.ts                            # Bootstrap
│   └── package.json                           # npm dependencies
│
└── README.md                                  # This file
```

---

## Quick Reference Commands

**Start Backend (Live):**
```bash
cd backend && uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

**Start Backend (Replay):**
```bash
cd backend && REPLAY_MODE=true uv run fastapi dev src/main.py --host 0.0.0.0 --port 8000
```

**Start Frontend:**
```bash
cd frontend && npm start
```

**Clear Cache:**
```bash
rm -rf backend/data/historical/trades/
```

**Add Python Package:**
```bash
cd backend && uv add package-name
```

**Add Frontend Package:**
```bash
cd frontend && npm install package-name
```

---

## Critical Implementation Notes

1. **WebSocket Blocking:** Never process logic in `on_message` callback. Always offload to queue.

2. **Cumulative Resets:** Backend restarts reset cumulative values. Frontend handles via `if (delta < 0) delta = 0`.

3. **Greek Staleness:** 0DTE Greeks change rapidly. 60s is maximum acceptable staleness.

4. **Parquet Partitioning:** One file per ticker per day. Do not append to existing files (Parquet is immutable).

5. **TradingView API:** v4 uses `addAreaSeries()`, v5 uses `addSeries({type: 'Area'})`. We use v4.

6. **Replay Window:** Last 10 minutes chosen because it's dense flow, tests all edge cases, loads fast.

7. **Angular Signals:** All reactive state uses `signal()` / `computed()`. No Observables for local state.

8. **No Tailwind:** Removed due to PostCSS issues. Use custom CSS only.

---

## End of Specification

This document contains all architectural knowledge needed to extend, debug, or refactor SPYMASTER. If you are an AI agent reading this: you now have complete context. When implementing changes, follow the development workflow, search existing patterns first, and verify implementations with replay mode before touching live mode.
