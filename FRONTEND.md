# Spymaster Frontend (Angular) — Current Implementation & Backend Contract

This file is the **single source of truth** for what the current Angular UI does and the **exact backend payloads** it expects.

---

## What’s implemented in the UI (today)

The app renders a “Command Center” layout:

- **Left: Price Ladder** (`PriceLadderComponent`)
  - Static “viewport” ladder: ticks and level markers stay anchored; **spot moves**.
  - Range is currently **±2.5** around the anchor price (see `CommandCenterComponent`).
  - Level markers show **barrier state** (WALL/VACUUM/ABSORPTION), **touch count**, and **tape pressure badge**.
  - Adds **MAGNET/CLIFF** markers derived from options **net gamma flow** within the visible ladder range.

- **Center: Strength + Attribution + Confluence**
  - **Strength Cockpit** (`StrengthCockpitComponent`): Break/Bounce meters, Call/Put success, **GO/WAIT/NO-GO**, mechanics tiles (tape velocity, approach speed, dealer gamma, gamma regime).
  - **Attribution** (`AttributionBarComponent`): tug-of-war view based on weighted “forces” (Barrier/Tape/Fuel/Approach/Confluence).
  - **Confluence Stack** (`ConfluenceStackComponent`): groups nearby levels into zones; shows **×N stack count**, **Δ from spot**, and includes an interactive **Band slider**.

- **Right: Options Activity** (`OptionsPanelComponent`)
  - **Strikes** (`StrikeGridComponent`): premium/volume + **GEX** (= net gamma flow aggregated by strike).
  - **Flow** (`FlowChartComponent`): bucketed flow velocity/acceleration charts (Lightweight Charts).

---

## Running the frontend

### Dev server
From the repo root:

```bash
cd frontend
npm run start
```

Angular defaults to port `4200`. If that port is in use:

```bash
cd frontend
npm run start -- --port 4300
```

### Live mode (default)
The frontend connects to the backend WebSocket at:
- `ws://localhost:8000/ws/stream`
  - Note: this URL is currently hard-coded in `frontend/src/app/data-stream.service.ts`.

### Static / mock mode (frontend-only)
Append `?mock=true` (or any `?mock` param) to the URL to run with generated dummy data:
- Example: `http://localhost:4300/?mock=true`

Mock mode is implemented in `frontend/src/app/data-stream.service.ts` and emits:
- **Levels** (with stable + drifting levels like VWAP)
- **Options flow** (11 strikes centered around spot, with OTM decay)

---

## Backend contract (what the frontend expects)

### Transport
**WebSocket**: `ws://localhost:8000/ws/stream`

The gateway should emit JSON frames containing **at least**:
- `levels` (required for ladder/cockpit/attribution/confluence)
- `flow` (required for options panel + magnet/cliff markers)

Additional keys are ignored by the frontend.

### Recommended cadence
- Target **~4 Hz** (250ms) updates for a smooth “command center” feel.
- Higher rates work, but keep frames lightweight (windowed strikes; small monitored level set).

### Frame shape (required keys)

```json
{
  "flow": { "<ticker>": { /* FlowMetrics */ } },
  "levels": { /* LevelsPayload */ }
}
```

### Snapshot semantics (important)
Each WebSocket frame is treated as a **full snapshot**:
- `flow` replaces the previous flow map (do not send partial patches).
- `levels.levels` replaces the previous level list (send the level universe you want visible; the frontend will filter by viewport).

---

## `flow`: options aggregates (required)

The frontend treats `flow` as a **state snapshot**: a map from ticker → cumulative metrics.

### `FlowMetrics` (required fields)

```ts
export interface FlowMetrics {
  cumulative_volume: number;      // contracts, monotonic increasing (session cumulative)
  cumulative_premium: number;     // USD, monotonic increasing (session cumulative)
  last_price: number;             // option last trade/mark (for display)

  net_delta_flow: number;         // signed, cumulative (can go up/down); used for flow analytics
  net_gamma_flow: number;         // signed, cumulative (can go up/down); used for ladder magnet/cliff + “GEX” display

  delta: number;                  // per-contract delta (optional for now)
  gamma: number;                  // per-contract gamma (optional for now)

  strike_price: number;           // numeric strike (ideally integer dollars for SPY)
  type: "C" | "P";                // call/put
  expiration: string;             // ISO date string

  last_timestamp: number;         // ms unix epoch; used for replay alignment and derivatives
}
```

### Backend expectations / invariants
- **Cumulative fields**:
  - `cumulative_volume` and `cumulative_premium` must be monotonic increasing (per session) for correct heatmapping.
  - `net_delta_flow` / `net_gamma_flow` should be **cumulative signed** flows. The frontend computes derivatives by differencing snapshots.
- **Universe / expirations**:
  - The UI currently assumes the `flow` snapshot is for a **single active expiry** (intended: 0DTE). If multiple expirations are mixed, strike aggregation will be misleading.
- **Windowing**:
  - Prefer emitting a **rolling strike window** around spot (default suggestion: **±10 strikes**) to keep payload size stable.
  - Include both call and put contracts for each strike when possible.

---

## `levels`: level physics snapshot (required)

### `LevelsPayload` (required)

```ts
export interface LevelsPayload {
  ts: number; // ms unix epoch
  spy: { spot: number | null; bid: number | null; ask: number | null };
  levels: LevelSignal[];
}
```

### `LevelSignal` (required fields)

These are the raw “market physics” signals. The frontend **does not** derive these from prices; it only scores/visualizes them.

```ts
export interface LevelSignal {
  id: string;                    // stable ID across time (used for hover, gamma velocity tracking)
  level_price: number;
  level_kind_name: string;       // e.g. PM_HIGH, VWAP, SMA_200, CALL_WALL...
  direction: "UP" | "DOWN";      // UP = resistance context, DOWN = support context
  distance: number;              // abs(spot - level_price) in dollars

  is_first_15m: boolean;
  bars_since_open: number;

  // Barrier physics
  barrier_state: string;         // enum-like: WALL | VACUUM | ABSORPTION | NEUTRAL | WEAK | CONSUMED
  barrier_delta_liq: number;     // signed; + tends to support bounce, - tends to support break (frontend uses sign)
  barrier_replenishment_ratio: number;
  wall_ratio: number;

  // Tape physics
  tape_imbalance: number;        // [-1..+1] normalized
  tape_velocity: number;         // signed; sign should align with "pressure into the level" for the given direction
  tape_buy_vol: number;
  tape_sell_vol: number;
  sweep_detected: boolean;

  // Fuel physics
  gamma_exposure: number;        // signed dealer gamma exposure (scale ~10k–50k for strong regimes)
  fuel_effect: "AMPLIFY" | "DAMPEN" | "NEUTRAL";

  // Approach context
  approach_velocity: number;     // signed (price speed), scale ~0–0.5
  approach_bars: number;
  approach_distance: number;
  prior_touches: number;

  // Display-only outputs
  break_score_raw: number;
  break_score_smooth: number;
  signal: "BREAK" | "BOUNCE" | "CHOP";
  confidence: "HIGH" | "MEDIUM" | "LOW";

  note?: string;
}
```

### `level_kind_name`: expected vocabulary
The frontend uses `level_kind_name` as a **semantic key** (weighting + labeling). Expected values include:

- Structural: `PM_HIGH`, `PM_LOW`, `OR_HIGH`, `OR_LOW`, `SESSION_HIGH`, `SESSION_LOW`
- Dynamic: `VWAP`, `SMA_200`, `SMA_400`
- Options-derived: `CALL_WALL`, `PUT_WALL`
- Utility: `ROUND`, `STRIKE`

Unknown kinds are accepted but get a default lower confluence weight.

### `barrier_state`: expected vocabulary
`barrier_state` is rendered and scored via enum-like strings. Expected values:
- `WALL`, `VACUUM`, `ABSORPTION`, `NEUTRAL`, `WEAK`, `CONSUMED`

### Minimal required fields (frontend fail-soft)
The frontend will **skip** a level if these are missing or non-finite:
- `level_price`, `distance`, `tape_velocity`, `gamma_exposure`, `barrier_delta_liq`

If `levels` is missing/invalid, the UI will enter a **Data unavailable** state.

---

## Frontend-derived computations (what the UI calculates)

The frontend currently computes UI-only derived signals in `LevelDerivedService`:

- **Break/Bounce strength**: weighted sum of normalized Barrier/Tape/Fuel/Approach plus a Confluence multiplier.
- **Gamma velocity (dealer positioning rate)**: \(d/dt\) of `gamma_exposure` per level ID.
- **Confluence groups**: clustering by price within `Band` (default 0.15) and level-kind weights.
- **MAGNET/CLIFF markers** (ladder): derived from net gamma flow aggregated by strike within the visible ladder viewport.

These do **not** replace backend engines; they’re presentation-grade summaries.

---

## Optional backend endpoint (future / not required for basic UI)

`FlowAnalyticsService` supports a user-defined “hotzone” anchor strike:
- `POST /api/config/hotzone` with `{ "strike": number | null }`

This is currently used by `FlowDashboardComponent` (not part of the main command-center layout).

---

## Known gaps / next work (frontend)

- Canvas-based fluid overlays (membrane/current/ribbon) are not implemented yet.
- A deep “Level Detail Drawer” for drilling into raw per-level metrics is not implemented yet.
