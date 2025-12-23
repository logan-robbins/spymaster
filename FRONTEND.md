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

## Implementation Status (2025-12-23)

### Current runtime state (as of now)
- **Backend stack is running** (docker-compose): NATS JetStream, MinIO, Core, Lake, Gateway, and Ingestor.
- **Gateway is healthy**: `GET /health` works and `/ws/stream` emits merged frames (`flow` + `levels`) at snap cadence.
- **Live stream has real data**:
  - `levels.spy.spot` is populated (not `null`)
  - `levels.levels` is a real array (non-empty) and includes the required keys (`wall_ratio`, `approach_*`, etc.)
- **Frontend is being served** at `http://localhost:4200/` from `frontend/dist/frontend/` (static dist bundle).

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
- **Step 1 (schema normalization)**: Implemented enough to be resilient to malformed/partial frames (fail-soft path exists).
- **Step 2 (LevelDerivedService)**: Implemented and no longer blocked by stream-parse exceptions (invalid levels are skipped).
- **Step 3 (new layout shell)**: Implemented (`AppComponent` renders the command center).
- **Step 4 (PriceLadder)**: Implemented (basic ladder + level markers; hover/pin not yet validated).
- **Step 5 (fluid indicators)**: Not fully implemented (no canvas-backed membrane/current/ribbon yet).
- **Step 6 (StrengthCockpit)**: Implemented.
- **Step 7 (Confluence + Attribution)**: Implemented; requires UI validation with real stream data.
- **Step 8 (LevelDetailDrawer)**: Not implemented.
- **Step 9 (StrikeGrid visible/one-click)**: Implemented (Options panel has StrikeGrid + Flow tab).
- **Step 10 (final polish)**: Partially implemented (typography + palette are in place; motion/interaction polish TBD).

### Remaining work (next actions)
- **Browser validation (must-do)**:
  - Use the browser tool to validate the Command Center with live data: Ladder markers render, cockpit shows non-zero strengths, attribution sums to ~100, confluence groups appear, and Options panel updates.
  - Specifically confirm there is no runtime crash and that the “data unavailable” state is only shown when appropriate.
- **UX completion per spec**:
  - Implement the three “fluid indicators” (canvas-backed) and validate performance (10–20fps UI cadence).
  - Implement `LevelDetailDrawerComponent` drill-down view.
  - Validate hover/pin interactions + range selector + confluence threshold slider (if not already wired).
- **Deployment standardization (optional but clarifying)**:
  - Decide whether to keep serving `frontend/dist` separately on 4200 or serve it from the gateway for a single-port local deploy.

### Verification checklist (next)
- **UI renders on live stream**: Ladder ticks + at least one level marker; cockpit shows non-zero Break/Bounce; attribution shares sum to ~100; confluence stack non-empty.
- **Stream resiliency**: Malformed/partial frames do not crash; header status indicates “Data unavailable” when levels payload is invalid/missing.
- **Contract correctness**: `wall_ratio` is distinct from `barrier_replenishment_ratio`; approach metrics are non-zero when data supports it; direction is `UP/DOWN`; signal is `BREAK/BOUNCE/CHOP`.
