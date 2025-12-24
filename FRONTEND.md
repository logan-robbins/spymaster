# Frontend Enhancement Plan - SPY 0DTE Bounce/Break Command Center

## Current Frontend Evaluation
- The UI is centered on options flow (FlowChart, StrikeGrid) with a secondary LevelTable and small LevelStrip. Bounce/break logic is visible only as text rows and badges, not as an at-a-glance decision surface.
- Signals use mixed language (BREAK/REJECT/CONTESTED) while the ML schema is BREAK/BOUNCE. This makes it harder to build consistent strength and attribution cues.
- There is no per-dollar ladder view, no explicit key-level hierarchy (PMH/PML, ORH/ORL, SMA200/400), and no confluence grouping when levels stack within a tight band.
- Strength is not visualized as a continuous, fluid signal. There are no motion-driven indicators for tape pressure, dealer gamma velocity, or liquidity state.
- Design is functional but table-heavy, with system fonts and a static dark palette. It does not convey a high-frequency market-physics feel.

## Product Goal and Success Criteria
- Show, in one screen, whether a 0DTE call or put is likely to succeed at the next dollar level, with explicit BREAK vs BOUNCE strength.
- Visualize strength as continuous, fluid signals that update in real time.
- Highlight key levels and confluence zones, and show how much predictive strength is coming from dealer mechanics vs tape vs barrier dynamics.
- Preserve data transparency by keeping a drill-down view for raw metrics.

## Data Inventory and Required Fields
Use the existing merged stream (`/ws/stream`) and ensure all frontend-visible data originates from the NATS JetStream pipeline (ingestor → core → lake → gateway). The gateway should continue to publish a merged payload (flow + levels) over WebSocket, and the frontend should not compute any raw physics features that belong in the backend engines.

Required per-level fields in the live stream (from backend physics engines and vectorized pipeline):
- Level identity: `id`, `level_price`, `level_kind_name`, `direction` (UP/DOWN), `distance`
- Market context: `is_first_15m`, `bars_since_open`
- Barrier physics: `barrier_state`, `barrier_delta_liq`, `barrier_replenishment_ratio`, `wall_ratio`
- Tape physics: `tape_imbalance`, `tape_velocity`, `tape_buy_vol`, `tape_sell_vol`, `sweep_detected`
- Fuel physics: `gamma_exposure`, `fuel_effect`
- Approach context: `approach_velocity`, `approach_bars`, `approach_distance`, `prior_touches`
- Outcome fields for display only: `break_score_raw`, `break_score_smooth`, `signal`, `confidence`

Viewport scoring (optional, gated by `VIEWPORT_SCORING_ENABLED=true`):
- `viewport.targets[]` with: `level_id`, `level_kind_name`, `level_price`, `direction`, `distance`, `distance_signed`
- ML outputs: `p_tradeable_2`, `p_break`, `p_bounce`, `strength_signed`, `strength_abs`, `time_to_threshold`
- Retrieval summaries: `retrieval.p_break`, `retrieval.p_tradeable_2`, `retrieval.strength_signed_mean`
- Ranking + state: `utility_score`, `viewport_state`, `stage`, `pinned`, `relevance`

StrikeGrid requirements (flow stream):
- Continue streaming per-strike option aggregates for calls/puts (volume, premium, net delta/gamma flow, last price).
- Ensure the backend publishes strike-aligned buckets around spot so the grid can show activity above/at/below current price.
- If not already present, add a small "active strikes" window (e.g., +/- 10 strikes) to reduce payload size while preserving visibility near spot.

Gap handling:
- Add SMA_400 to the level universe (backend and stream). The UI should treat SMA_200 and SMA_400 as key levels even if only one is present.
- Normalize direction naming: map SUPPORT/RESISTANCE to DOWN/UP (or expose both) to keep bounce/break logic consistent.
- If any of the above fields are missing, the frontend should show "data unavailable" states and fall back to a reduced-strength visualization.

## Data Contract (Stream Payloads)
All payloads are produced by backend services and delivered via NATS JetStream to the gateway, which then emits WebSocket frames. The frontend should treat the WebSocket as the single source of truth and never fabricate physics values.

Merged stream (gateway output over `/ws/stream`):
```
{
  "flow": {
    "SPY241220C00585000": {
      "cumulative_volume": 1234,
      "cumulative_premium": 184523.50,
      "last_price": 2.15,
      "net_delta_flow": 21500.0,
      "net_gamma_flow": -7800.0,
      "delta": 0.42,
      "gamma": 0.08,
      "strike_price": 585.0,
      "type": "C",
      "expiration": "2025-12-20",
      "last_timestamp": 1765843200000
    }
  },
  "levels": {
    "ts": 1765843200000,
    "spy": { "spot": 585.12, "bid": 585.10, "ask": 585.14 },
    "levels": [
      {
        "id": "PM_HIGH_585.50",
        "level_price": 585.50,
        "level_kind_name": "PM_HIGH",
        "direction": "UP",
        "distance": 0.38,
        "is_first_15m": false,
        "barrier_state": "WALL",
        "barrier_delta_liq": 220.0,
        "barrier_replenishment_ratio": 1.8,
        "wall_ratio": 2.1,
        "tape_imbalance": 0.32,
        "tape_velocity": 44.5,
        "tape_buy_vol": 1240,
        "tape_sell_vol": 900,
        "sweep_detected": false,
        "gamma_exposure": -31500.0,
        "fuel_effect": "AMPLIFY",
        "approach_velocity": 0.18,
        "approach_bars": 3,
        "approach_distance": 1.20,
        "prior_touches": 2,
        "bars_since_open": 18,
        "break_score_raw": 72,
        "break_score_smooth": 68,
        "signal": "BREAK",
        "confidence": "HIGH"
      }
    ]
  },
  "viewport": {
    "ts": 1765843200000,
    "targets": [
      {
        "level_id": "OR_LOW",
        "level_kind_name": "OR_LOW",
        "level_price": 585.0,
        "direction": "DOWN",
        "distance": 0.12,
        "distance_signed": -0.12,
        "p_tradeable_2": 0.62,
        "p_break": 0.48,
        "p_bounce": 0.52,
        "strength_signed": -0.15,
        "strength_abs": 0.15,
        "time_to_threshold": {
          "t1": {"60": 0.22, "120": 0.31},
          "t2": {"60": 0.08, "120": 0.14}
        },
        "retrieval": {
          "p_break": 0.44,
          "p_bounce": 0.56,
          "p_tradeable_2": 0.59,
          "strength_signed_mean": -0.11,
          "strength_abs_mean": 0.18,
          "time_to_threshold_1_mean": 92.0,
          "time_to_threshold_2_mean": 188.0,
          "neighbors": []
        },
        "utility_score": 0.093,
        "viewport_state": "IN_MONITOR_BAND",
        "stage": "stage_a",
        "pinned": false,
        "relevance": 0.64
      }
    ]
  }
}
```

StrikeGrid windowing (backend output guidelines):
- Only emit strikes within a configurable band around spot (default +/- 10 strikes).
- Include at-spot strikes even if volume is low to preserve the visual ladder context.
- If the band shifts, send a full refresh for the grid to avoid partial state.

## Core UX Concept: Bounce/Break Command Center
One screen, three zones:

1. Price Ladder (center-left)
   - Vertical ladder showing every 1.00 SPY increment in a visible range (default +/- 6.00 from spot).
   - The active decision band (distance <= 0.25) is emphasized.
   - Key levels (PMH, PML, ORH, ORL, SMA200, SMA400, VWAP, CALL_WALL, PUT_WALL) have labeled chips and higher visual weight.

2. Strength Cockpit (center-right)
   - Dual strength meter: BREAK strength vs BOUNCE strength, always visible.
   - Call/Put success cards derived from the current closest level and approach direction.
   - Attribution bar showing percent contribution from Barrier, Tape, Fuel, and Confluence.

3. Confluence and Timeline (right panel)
   - Confluence stack list: clustered levels within a defined band, each with a confluence strength score.
   - Signal timeline: last N level touches with predicted outcome, strength, and actual resolution (if available).

## Strength Model and Attribution (Frontend Calculation)
The UI must surface two continuous scores (0-100):
- Break Strength
- Bounce Strength

Use a weighted sum of normalized inputs. Recommended weighting:
- Barrier state and wall ratio: 0.30
- Tape velocity and imbalance: 0.25
- Fuel effect and gamma exposure: 0.25
- Approach context (velocity, distance, prior touches): 0.10
- Confluence multiplier: 0.10

Directional logic:
- If direction is UP (approaching resistance):
  - Break strength increases with VACUUM, high tape velocity, AMPLIFY, fast approach.
  - Bounce strength increases with WALL/ABSORPTION, DAMPEN, slow approach.
- If direction is DOWN (approaching support):
  - Break strength increases with VACUUM, strong sell imbalance, AMPLIFY, fast approach.
  - Bounce strength increases with WALL/ABSORPTION, DAMPEN, slow approach.

Dealer mechanics velocity attribution:
- Track delta over time for `gamma_exposure` per level.
- Compute a short-window derivative (for example, 15-30 seconds) and map it to a velocity magnitude.
- The Fuel contribution to total strength should be reported as a percent share and displayed in the attribution bar.

Call/Put success mapping:
- For UP approaches: calls prefer BREAK strength, puts prefer BOUNCE strength.
- For DOWN approaches: puts prefer BREAK strength, calls prefer BOUNCE strength.
- Display call/put success as separate gauges to avoid confusion.

## Confluence Logic
Define a confluence band (default 0.10 to 0.20).
Cluster key levels within this band into a Confluence Group:
- Score = sum of level weights (PMH/PML and ORH/ORL highest, SMA200/400 medium, VWAP/ROUND lower).
- Boost score during first 15 minutes and when tape velocity is high.
- Assign a dominant bias based on the average Break vs Bounce strength of the group.

The ladder should show confluence zones as a single, emphasized node with stacked labels (not a list of thin lines).

## Fluid Indicators (Multiple, Continuous)
Implement at least three fluid indicators, all tied to live data:

1. Liquidity Membrane (Barrier Physics)
   - A shimmering band at each level that thickens with wall_ratio.
   - VACUUM cracks: micro gaps or tearing effect when barrier_delta_liq is strongly negative.
   - ABSORPTION shows a slow pulsing thickness without lateral drift.

2. Dealer Gamma Current (Fuel Physics)
   - A flowing gradient stream that moves up/down based on direction and gamma velocity.
   - AMPLIFY increases flow speed and turbulence; DAMPEN reduces motion and increases viscosity.
   - The current overlays the ladder and is strongest at the nearest active level.

3. Tape Momentum Ribbon (Tape Physics)
   - A ribbon wave that leans toward the dominant imbalance.
   - Velocity increases with tape_velocity; sweep detection adds a brief burst effect.

Optional fourth indicator for confluence:
- Confluence Bloom: a soft glow that expands and contracts based on confluence score and fades when groups split.

## Visual System (Non-Generic, Market-Physics Aesthetic)
- Typography: use a distinct headline font (Space Grotesk) and a numeric mono font (IBM Plex Mono or JetBrains Mono).
- Palette: deep navy and charcoal base, with electric teal for bounce, ember orange for break, and steel cyan for neutral.
- Background: subtle grid or topographic contour lines to imply price structure; avoid flat backgrounds.
- Motion: slow ambient motion for baseline, accelerated motion on high velocity or sweep detection.
- Avoid emojis and novelty icons; use minimal glyphs and directional arrows.

## Backend vs Frontend Responsibilities
Backend (ingestion + physics engines):
- All physics feature computation: barrier_state, tape_imbalance/velocity, fuel_effect, gamma_exposure, approach metrics.
- Level detection and labeling: level_kind, direction, distance, and any break/bounce raw scores.
- Confluence inputs if desired (e.g., per-level weights or key-level tags), but the final grouping can be frontend-driven.
- Strike-level flow aggregation for the StrikeGrid (calls/puts above/at/below spot, cumulative premium/volume, net delta/gamma flow).

Frontend (presentation + light aggregation):
- Normalize/format the streamed fields for display (units, colors, thresholds).
- Compute derived UI-only scores: break/bounce strength meters, attribution percentages, and confluence grouping.
- Maintain rolling windows for smoothing and motion (velocity of gamma_exposure, ribbon easing, UI interpolation).
- Provide user-driven filters, range changes, and hover/pin interaction state.

## Component Architecture (Angular)
Core components:
- AppShellComponent: layout, header, session status.
- PriceLadderComponent: vertical ladder, per-dollar lines, key-level chips.
- LevelMarkerComponent: single level line with fluid overlays and label.
- StrengthCockpitComponent: dual strength meters and call/put success cards.
- ConfluenceStackComponent: grouped level list with strength and bias.
- AttributionBarComponent: stacked contribution from Barrier/Tape/Fuel/Confluence.
- SignalTimelineComponent: chronological list of recent touches.
- LevelDetailDrawerComponent: deep metrics view for selected level.
- OptionsFlowPanelComponent: required. Keep the StrikeGrid visible (or one-click accessible) to monitor strike activity above/at spot; FlowChart can remain as a secondary tab.

Services and state:
- LevelStreamService: raw stream ingestion.
- LevelDerivedService: normalization, strength scores, confluence detection, dealer velocity.
- ViewStateService: selected level, range window, filters, pinned confluence group.

## Interaction Model
- Hover a level to preview strength and metrics; click to pin and open detail drawer.
- Toggle visibility by level type (PM, OR, SMA, VWAP, WALL).
- Range selector to widen/narrow ladder (for example +/- 3, 6, 10).
- Confluence threshold slider with live recomputation.
- Session indicator for first 15 minutes; visually increase volatility emphasis during this window.

## Performance and Rendering
- Use a canvas-backed layer for fluid indicators; keep text and labels in DOM for accessibility.
- Maintain a rolling time series buffer per level (60-120 seconds) for velocity and ribbon smoothing.
- Update UI at a fixed cadence (for example 10-20 fps) to avoid full reflow on each stream tick.
- Ensure mobile layout stacks the ladder above the strength cockpit and collapses the timeline.

## Phased Implementation Steps (No Code)
1. Normalize the stream schema in the frontend: update interfaces to match `backend/features.json`, add SMA_400 support, and map direction naming consistently. Confirm all data arrives via the NATS → gateway → WebSocket path.
2. Build LevelDerivedService to compute break/bounce strengths, confluence clusters, and dealer gamma velocity attribution using short rolling windows.
3. Implement the new layout shell and replace the current FlowDashboard in `AppComponent`.
4. Build PriceLadderComponent with per-dollar ticks, key-level chips, and hover/pin interactions.
5. Add the three fluid indicators (Liquidity Membrane, Dealer Gamma Current, Tape Momentum Ribbon) as layered canvas/SVG overlays.
6. Implement StrengthCockpit with dual meters and call/put success logic based on approach direction.
7. Add ConfluenceStack and AttributionBar, wired to the derived confluence groups and contribution weights.
8. Provide a LevelDetailDrawer that reveals full barrier/tape/fuel/approach context and prior touches.
9. Keep the StrikeGrid visible (or one-click) to track live strike activity near spot; keep FlowChart as a secondary tab if needed.
10. Final polish: typography, palette, motion tuning, and responsive layout verification.

---

## Implementation Status (2025-12-23 - Updated)

### Current runtime state (VERIFIED WORKING)
- **Backend stack is running** (docker-compose): NATS JetStream, MinIO, Core, Lake, Gateway, and Ingestor.
- **Gateway is healthy**: `GET /health` returns `{"service":"gateway","status":"healthy","connections":1}` and `/ws/stream` emits merged frames (`flow` + `levels`) at ~250ms intervals.
- **Live stream has real data**:
  - `levels.spy.spot` is populated (spot values: 678-687 range from ES futures / 10)
  - `levels.levels` is a real array (0-4 levels depending on MONITOR_BAND proximity)
  - All required fields present: `barrier_state`, `tape_velocity`, `gamma_exposure`, `approach_velocity`, etc.
- **Frontend is running**: Angular dev server at `http://localhost:4200/` with hot-reload enabled.
- **WebSocket verified working**: Test client successfully receives level signals with SPY data and level arrays.
- **Replay mode active**: Ingestor replaying DBN files at 1.0x realtime speed (configurable via `REPLAY_SPEED` env var).

### Fixes that are implemented in code (and now expected in runtime)
- **Frontend fail-soft stream handling**:
  - Hardened `DataStreamService` parsing so malformed/partial frames do not crash UI render paths.
  - Added an explicit “data unavailable” state when `levels` payload is invalid/missing.
  - Made `LevelDerivedService` fail-soft by skipping invalid levels instead of throwing.
- **Gateway contract alignment**:
  - `wall_ratio` is no longer incorrectly mapped from `barrier_replenishment_ratio` (it is derived from barrier depth where missing).
  - `approach_velocity`, `approach_bars`, `approach_distance`, `prior_touches` are emitted from the Core payload (no longer hardcoded zeros).
  - Direction naming is normalized (`SUPPORT/RESISTANCE` → `DOWN/UP`) and signal naming is normalized (`REJECT/CONTESTED` → `BOUNCE/CHOP`).
- **Core emits more required fields**:
  - Live payload now includes approach context fields and tracks prior touches.
  - Live level universe now supports structural levels (PM/OR) and **SMA_200/SMA_400** derived from the ES trade stream.
- **Replay-based local data feed**:
  - `docker-compose.yml` ingestor runs `src.ingestor.replay_publisher` (DBN → NATS) with a DBN volume mount so the stack produces non-empty spot/levels without external feeds.
  - NATS publishing now JSON-encodes Enums as `.value` so consumers can reconstruct messages correctly.
- **Docker caching / no-pull**:
  - Added `backend/.dockerignore` to prevent data/logs from invalidating Docker layers.
  - Compose is intended to be launched with `--pull never` (and Compose services have `pull_policy: never` for non-build images). Base Python images must already be present locally or the build will fail (by design).

### Status vs this plan (high level)
- **Step 1 (schema normalization)**: ✅ COMPLETE - Resilient to malformed frames with fail-soft parsing.
- **Step 2 (LevelDerivedService)**: ✅ COMPLETE - Computes break/bounce strengths, confluence, gamma velocity.
- **Step 3 (new layout shell)**: ✅ COMPLETE - CommandCenterComponent renders three-panel layout.
- **Step 4 (PriceLadder)**: ✅ COMPLETE - Vertical ladder with level markers, hover states working.
- **Step 5 (fluid indicators)**: ⚠️ PARTIAL - Dealer gamma bar indicator added; canvas-backed membrane/current/ribbon TBD.
- **Step 6 (StrengthCockpit)**: ✅ COMPLETE + ENHANCED - Dual meters + call/put success + NEW: velocity/gamma mechanics section.
- **Step 7 (Confluence + Attribution)**: ✅ COMPLETE - Attribution bar + confluence stack with grouping logic.
- **Step 8 (LevelDetailDrawer)**: ❌ NOT IMPLEMENTED - Deep metrics drill-down pending.
- **Step 9 (StrikeGrid visible/one-click)**: ✅ COMPLETE - Options panel with strike grid and flow chart.
- **Step 10 (final polish)**: ⚠️ IN PROGRESS - Typography/palette done; motion tuning ongoing.

### Latest Enhancements (2025-12-23)

**NEW: Velocity & Dealer Mechanics Visualization**
Added comprehensive mechanics section to `StrengthCockpitComponent` displaying:
1. **Tape Velocity** - Order flow acceleration with color-coded meter (blue gradient)
2. **Approach Speed** - Price velocity toward level with normalized bar (yellow gradient)
3. **Dealer Gamma Exposure** - Bidirectional indicator showing SHORT←→LONG positioning:
   - Visual bar centered at zero with left (SHORT/red) and right (LONG/green) ranges
   - Displays gamma in K-format (e.g., "31.5K" for 31,500)
   - Auto-scales based on magnitude
4. **Gamma Velocity** - Rate of dealer accumulation/exit with contextual hints:
   - "Dealers accumulating FAST" (>500/s)
   - "Dealers building position" (>100/s)
   - "Dealers reducing exposure" (<-100/s)
   - "Dealers exiting FAST" (<-500/s)
   - "Stable positioning" (neutral)

**Color Encoding:**
- 🟢 Green: Positive/bullish momentum
- 🔴 Red: Negative/bearish momentum
- 🔵 Blue: Tape flow activity
- 🟡 Yellow: Approach velocity
- ⚪ Gray: Neutral/no signal

### Remaining work (next actions)
- **Canvas-based fluid indicators**:
  - Liquidity Membrane (barrier physics shimmer/crack effects)
  - Dealer Gamma Current (flowing gradient stream overlay)
  - Tape Momentum Ribbon (wave with burst effects on sweeps)
- **LevelDetailDrawerComponent**: Deep metrics drill-down with full barrier/tape/fuel context
- **Interaction polish**: Validate hover/pin behavior, range selector, confluence threshold slider
- **Performance tuning**: Ensure 10-20fps UI cadence for fluid animations
- **Mobile responsive layout**: Stack panels vertically for mobile viewports

### Verification checklist (PASSING)
- ✅ **UI renders on live stream**: Ladder ticks visible, level markers render when within MONITOR_BAND
- ✅ **Cockpit metrics**: Break/Bounce strengths computed, attribution percentages present
- ✅ **Stream resiliency**: Fail-soft parsing prevents crashes on malformed frames
- ✅ **Contract correctness**: `wall_ratio` distinct from `barrier_replenishment_ratio`, direction uses `UP/DOWN`, signal uses `BREAK/BOUNCE/CHOP`
- ✅ **WebSocket connectivity**: Gateway broadcasts to connected clients at ~4 Hz
- ✅ **Velocity metrics**: Tape velocity, approach speed, gamma exposure all display real values
- ✅ **Gamma velocity tracking**: `LevelDerivedService` computes rate of change for dealer positioning

## Quick Start Guide (For AI Agents)

### Running the System

**1. Start Backend Services:**
```bash
cd /Users/loganrobbins/research/qmachina/spymaster
docker-compose up -d
```

This starts:
- NATS JetStream (messaging bus)
- MinIO (object storage)
- Ingestor (DBN replay → NATS at 1.0x realtime)
- Core (physics engines + level signals)
- Lake (data persistence)
- Gateway (WebSocket relay at port 8000)

**2. Start Frontend Dev Server:**
```bash
cd frontend
npm run start
```

Frontend runs at `http://localhost:4200` with hot-reload.

**3. Verify Services:**
```bash
# Check all containers running
docker ps

# Check gateway health
curl http://localhost:8000/health

# Check WebSocket (simple test)
cd backend
uv run python -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8000/ws/stream') as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print(f'SPY: {data[\"levels\"][\"spy\"][\"spot\"]}')
asyncio.run(test())
"
```

### Configuration

**Replay Speed (Ingestor):**
Controlled by `REPLAY_SPEED` environment variable in `docker-compose.yml`:
- `0` = Max speed (no pacing)
- `1.0` = Realtime (1x)
- `2.0` = 2x speed
- `0.5` = Half speed

Change and restart:
```bash
export REPLAY_SPEED=2.0
docker-compose up -d ingestor --force-recreate
```

**Monitor Band (Core):**
Levels only appear when `abs(spot - level_price) <= MONITOR_BAND`.
Default: `$0.25` (configurable in `backend/src/common/config.py`)

**Snap Interval (Core):**
Level signal computation frequency. Default: `250ms` (4 Hz)

### Key File Locations

**Frontend Components:**
- `frontend/src/app/command-center/command-center.component.ts` - Main layout
- `frontend/src/app/price-ladder/price-ladder.component.ts` - Vertical price ladder
- `frontend/src/app/strength-cockpit/strength-cockpit.component.ts` - Strength meters + mechanics
- `frontend/src/app/attribution-bar/attribution-bar.component.ts` - Contribution breakdown
- `frontend/src/app/confluence-stack/confluence-stack.component.ts` - Grouped levels
- `frontend/src/app/options-panel/options-panel.component.ts` - Strike grid

**Frontend Services:**
- `frontend/src/app/data-stream.service.ts` - WebSocket connection + stream parsing
- `frontend/src/app/level-derived.service.ts` - Strength computation + confluence detection

**Backend Services:**
- `backend/src/gateway/main.py` - WebSocket gateway entry point
- `backend/src/gateway/socket_broadcaster.py` - NATS → WebSocket relay
- `backend/src/core/service.py` - Physics orchestrator + snap loop
- `backend/src/core/level_signal_service.py` - Level signal generation
- `backend/src/ingestor/replay_publisher.py` - DBN replay → NATS

**Configuration:**
- `docker-compose.yml` - Service orchestration + environment variables
- `backend/src/common/config.py` - Backend configuration constants
- `backend/pyproject.toml` - Python dependencies (managed by `uv`)

### Debugging Tips

**No levels showing:**
- Check spot price is within valid range (typically 600-700 for SPY)
- Verify MONITOR_BAND allows levels to appear: `spot ± 0.25`
- Check core logs: `docker logs spymaster-core --tail 50`

**WebSocket not connecting:**
- Verify gateway is running: `curl http://localhost:8000/health`
- Check browser console for connection errors
- Ensure no firewall blocking port 8000
- Test with simple WebSocket client (see verification above)

**Frontend not updating:**
- Check gateway logs for broadcast messages: `docker logs spymaster-gateway --tail 30`
- Verify browser DevTools shows WebSocket connection in Network tab
- Refresh browser to force reconnection
- Check for JavaScript errors in browser console

**Data looks wrong:**
- Core publishes spot values from ES futures / 10 (so ES 6790 → SPY 679)
- Some weird spot values (5.76 instead of 681.76) indicate data parsing issues
- Check ingestor logs: `docker logs spymaster-ingestor --tail 50`
- Verify DBN files are present in `dbn-data/` directory
