## Frontend Context (for AI Coding Agents)

### Product intent
This app is **NOT** trying to replicate TradingView or Pine-script confluence logic (PM/ORB/SMAs). Traders will typically run **TradingView** for price/action + confluence and keep **this webpage** open as a **flow confirmation / sentiment instrument** for SPY 0DTE.

The frontend’s job is to answer questions like:
- **Where is flow concentrating?** (which strikes / which side)
- **How fast is flow building?** (velocity)
- **Is the rate itself increasing/decreasing?** (2nd/3rd derivatives: acceleration/jerk)
- **Is pressure building above or below ATM?** (bucketed by strike relative to ATM)

### System context (how this ties to the backend)
Backend pipeline (high-level):
- Polygon streams options trades → backend aggregates **per-contract cumulative** metrics
- Backend broadcasts a **state snapshot** over WebSocket every ~250ms to the frontend

Frontend pipeline (high-level):
- Connect to backend WebSocket (`/ws/stream`) and receive snapshots (`FlowMap`)
- Maintain a small rolling history buffer
- Compute derivatives (velocity/accel/jerk) over selectable timescales
- Render: strike ladder + heat shading + “flow wave” panels + above/below pressure

Key backend detail: the stream payload is **cumulative** per contract, not per-trade. That’s deliberate: it keeps the frontend simple and makes derivative math stable (difference between snapshots).

### The stream data contract the frontend consumes
The WebSocket payload is a JSON object mapping **option ticker → metrics**.

The frontend type lives in `frontend/src/app/data-stream.service.ts`:
- **`FlowMap`**: `{ [ticker: string]: FlowMetrics }`
- **`FlowMetrics`** fields currently used by the UI/analytics:
  - `cumulative_volume`
  - `cumulative_premium`
  - `net_delta_flow`
  - `net_gamma_flow`
  - `strike_price`
  - `type` (`'C'` or `'P'`)
  - `expiration`

Important: The backend does **not** currently broadcast underlying price/spot. It uses REST snapshot chain internally for strike selection and Greek caching.

### Frontend “analytics layer” (what we compute and why)
Because the stream is cumulative, the frontend computes “how fast is it changing” by differencing snapshots.

#### ATM inference (current approach)
To bucket “above vs below” we need a reference strike. Since spot is not in the stream, we infer a stable proxy:
- **ATM strike ≈ median of the strikes present in the current snapshot**

This is implemented in `FlowAnalyticsService` and is intentionally simple/robust. If you later add spot to the backend payload, switch bucketing to use the true spot-derived ATM.

ATM handling:
- We treat **ATM as neutral** and exclude it from “above/below” buckets (to avoid “ATM drift” noise).

#### Buckets
We split the world into four buckets:
- `call_above`: Calls with strike > ATM
- `call_below`: Calls with strike < ATM
- `put_above`: Puts with strike > ATM
- `put_below`: Puts with strike < ATM

These buckets power the “Above/Below Pressure” panel and the wave panels.

#### Derivative definitions
For each bucket and each metric we compute:
- **Order 1 (velocity)**: \(\Delta\text{cum} / \Delta t\)
- **Order 2 (acceleration)**: \(\Delta\text{velocity} / \Delta t\)
- **Order 3 (jerk)**: \(\Delta\text{acceleration} / \Delta t\)

Metrics supported today:
- `premium` (uses `cumulative_premium`)
- `delta` (uses `net_delta_flow`)
- `gamma` (uses `net_gamma_flow`)

Timescales supported today:
- 1s, 5s, 30s windows

Implementation notes:
- We compute velocity over a lookback window (1s/5s/30s).
- Accel/jerk are computed from successive update ticks (websocket cadence).
- We guard against backend restarts by treating negative cumulative deltas as 0 (cumulative should not decrease during a session).

### “Flow wave” visualization (what it is)
The “wave” is just the selected derivative time-series rendered as a compact SVG line:
- It’s meant to be read like a **pulse/impulse meter**
- “Now” is visually emphasized (vertical marker), and the trace flows into it

This is implemented in `frontend/src/app/flow-wave/flow-wave.component.ts`.

### Strike ladder heat shading (what it is)
The strike grid is still the canonical “ladder” view, but now it has a lightweight heat signal:
- We compute **instantaneous premium velocity per strike/side** from snapshot-to-snapshot deltas
- We shade the premium cell with a green/red alpha intensity proportional to \(|velocity| / max|\velocity|\)
- We show a small “$/s” readout under the premium value for each strike side

This is intentionally the first “heatmap” step: it gives immediate signal without adding a full-blown matrix component yet.

### File map (what matters)

#### Data ingestion
- `frontend/src/app/data-stream.service.ts`
  - Connects to `ws://localhost:8000/ws/stream`
  - Maintains `flowData` as an Angular **signal** (the canonical live snapshot)

#### Analytics / derivatives engine
- `frontend/src/app/flow-analytics.service.ts`
  - Subscribes to `flowData` via `effect()`
  - Maintains a rolling history buffer (default ~2 minutes)
  - Computes:
    - ATM strike proxy
    - per-bucket derivatives (1st/2nd/3rd order, multi-timescale)
    - per-strike premium velocity for heat shading

If you add new “flow math,” it almost certainly belongs here (keep components dumb).

#### Dashboard layout
- `frontend/src/app/flow-dashboard/flow-dashboard.component.ts`
  - Top controls: metric (premium/delta/gamma), order (d1/d2/d3), timescale (1s/5s/30s)
  - Right panel: Above/Below pressure + waves
  - Left panel: Strike ladder

#### Wave component
- `frontend/src/app/flow-wave/flow-wave.component.ts`
  - Pure rendering component (input = series array)
  - Scales by max abs value in the series to keep it visually stable

#### Strike grid
- `frontend/src/app/strike-grid/strike-grid.component.ts`
- `frontend/src/app/strike-grid/strike-grid.component.html`
  - Renders the strike ladder
  - Uses `FlowAnalyticsService` for per-strike premium velocity + heat intensity

#### App entry
- `frontend/src/app/app.component.ts`
  - Mounts `<app-flow-dashboard>`

Note: `frontend/src/app/app.ts` is a leftover scaffold component and is not bootstrapped (the app bootstraps `AppComponent` in `frontend/src/main.ts`).

### Operational / dev notes
- Frontend expects backend at `ws://localhost:8000/ws/stream`.
- To run:
  - `cd frontend && npm start`
- To build:
  - `cd frontend && npm run build`

### Things to remember (common pitfalls)
- **Cumulative resets**: if the backend restarts, cumulative values can jump backwards; derivative logic should treat negative deltas as 0.
- **ATM approximation**: current “above/below” is relative to a derived ATM strike, not true spot. If you need precision, add underlying price (or spot-derived ATM) to the backend broadcast.
- **No BBO/quotes**: the system is not using bid/ask/quote-based aggressor classification. This UI is designed around **magnitude and derivatives**, not “who hit the bid/ask.”
- **Units**:
  - `premium` is in dollars (price * size * 100), so velocity is $/s, accel is $/s², etc.
  - `delta` and `gamma` flows are notionals per the backend (see backend `FlowAggregator` math), so derivatives are in “per second” versions of that unit.

### Next obvious extensions (where an AI agent should go next)
- **Add spot/underlying price to the broadcast**:
  - Best fix for above/below accuracy and “distance from spot” scaling in heatmaps.
- **Introduce a real strike heatmap matrix**:
  - Rows = strikes, columns = metric/timeframe/order (or calls vs puts), color = derivative magnitude.
- **Add multi-timescale “stacked slopes”**:
  - Show d1/d2/d3 simultaneously for 1s/5s/30s to visually see regime shifts.
- **Add robust normalization**:
  - Winsorize/clip extremes; consider EMA smoothing for derivatives so the wave reads cleanly during bursty tape.


