# Frontend Interfaces

**Module**: `frontend/`  
**Framework**: Angular 18  
**Role**: Real-time UI for SPY 0DTE break/bounce signals  
**Audience**: AI Coding Agents

---

## Module Purpose

Provides real-time visualization of SPY 0DTE level signals with physics attribution, strength meters, and dealer mechanics. Connects to Gateway WebSocket for live data.

---

## Input Interface

### WebSocket Connection

**URL**: `ws://localhost:8000/ws/stream`  
**Protocol**: WebSocket (JSON text frames)  
**Cadence**: ~250ms (driven by backend snap interval)

**Implementation**: `DataStreamService` (`src/app/data-stream.service.ts`)

```typescript
class DataStreamService {
  connect(): void
  disconnect(): void
  getLevelsStream(): Observable<LevelData>
  getConnectionStatus(): Observable<boolean>
}
```

---

### WebSocket Payload Schema

**Message Format** (from Gateway):
```typescript
interface StreamPayload {
  ts: number;  // Unix milliseconds
  spy: {
    spot: number;
    bid: number;
    ask: number;
  };
  levels: LevelSignal[];
  viewport?: {
    ts: number;
    targets: ViewportTarget[];
  };
}
```

---

### Level Signal Schema

```typescript
interface LevelSignal {
  // Identity
  id: string;                    // e.g., "STRIKE_687", "PM_HIGH"
  level_price: number;
  level_kind_name: string;       // "STRIKE" | "PM_HIGH" | "OR_LOW" | etc.
  direction: string;             // "UP" (resistance) | "DOWN" (support)
  distance: number;              // Dollars from spot
  
  // Context
  is_first_15m: boolean;
  bars_since_open: number;
  
  // Scores
  break_score_raw: number;       // 0-100
  break_score_smooth: number;    // 0-100 (EWMA smoothed)
  signal: string;                // "BREAK" | "BOUNCE" | "CHOP" | "NEUTRAL"
  confidence: string;            // "HIGH" | "MEDIUM" | "LOW"
  
  // Barrier physics
  barrier_state: string;         // "VACUUM" | "WALL" | "ABSORPTION" | etc.
  barrier_delta_liq: number;
  barrier_replenishment_ratio: number;
  wall_ratio: number;
  
  // Tape physics
  tape_imbalance: number;        // -1 to +1
  tape_velocity: number;         // $/sec
  tape_buy_vol: number;
  tape_sell_vol: number;
  sweep_detected: boolean;
  
  // Fuel physics
  gamma_exposure: number;
  fuel_effect: string;           // "AMPLIFY" | "DAMPEN" | "NEUTRAL"
  
  // Approach context
  approach_velocity: number;
  approach_bars: number;
  approach_distance: number;
  prior_touches: number;
}
```

---

### Viewport Target Schema (Optional ML)

```typescript
interface ViewportTarget {
  // Level identity
  level_id: string;
  level_kind_name: string;
  level_price: number;
  direction: string;
  distance: number;
  distance_signed: number;
  
  // ML predictions
  p_tradeable_2: number;         // 0-1 probability
  p_break: number;               // 0-1 probability (conditional on tradeable)
  p_bounce: number;              // 0-1 probability (1 - p_break)
  strength_signed: number;       // Expected movement magnitude
  strength_abs: number;
  time_to_threshold: {
    t1: { [horizon: string]: number };  // e.g., {"60": 0.22, "120": 0.31}
    t2: { [horizon: string]: number };
  };
  
  // Retrieval (kNN)
  retrieval: {
    p_break: number;
    p_bounce: number;
    p_tradeable_2: number;
    strength_signed_mean: number;
    strength_abs_mean: number;
    time_to_threshold_1_mean: number;
    time_to_threshold_2_mean: number;
    neighbors: any[];  // Optional neighbor metadata
  };
  
  // Scoring metadata
  utility_score: number;
  viewport_state: string;        // "IN_MONITOR_BAND" | "OUTSIDE_BAND"
  stage: string;                 // "stage_a" | "stage_b"
  pinned: boolean;
  relevance: number;
}
```

---

## Service Interfaces

### 1. DataStreamService

**Location**: `src/app/data-stream.service.ts`  
**Purpose**: WebSocket connection and stream parsing

**Interface**:
```typescript
class DataStreamService {
  // Connection management
  connect(): void;
  disconnect(): void;
  
  // Data streams (RxJS Observables)
  getLevelsStream(): Observable<LevelData>;
  getConnectionStatus(): Observable<boolean>;
  
  // Error handling
  getErrors(): Observable<string>;
}
```

**LevelData Type**:
```typescript
interface LevelData {
  timestamp: number;
  spot: number;
  bid: number;
  ask: number;
  levels: LevelSignal[];
  viewport?: ViewportData;
}
```

**Connection States**:
- `CONNECTING`: Initial connection attempt
- `CONNECTED`: Active WebSocket connection
- `DISCONNECTED`: Connection lost (triggers auto-reconnect)
- `ERROR`: Connection error (manual retry required)

---

### 2. LevelDerivedService

**Location**: `src/app/level-derived.service.ts`  
**Purpose**: Compute UI-specific derived metrics (strength scores, confluence)

**Interface**:
```typescript
class LevelDerivedService {
  // Strength computation
  computeBreakStrength(level: LevelSignal): number;    // 0-100
  computeBounceStrength(level: LevelSignal): number;   // 0-100
  
  // Attribution breakdown
  computeAttribution(level: LevelSignal): Attribution;
  
  // Confluence detection
  detectConfluenceGroups(
    levels: LevelSignal[],
    confluenceBand: number
  ): ConfluenceGroup[];
  
  // Gamma velocity
  computeGammaVelocity(
    currentExposure: number,
    levelId: string,
    timestamp: number
  ): number;
}
```

**Attribution Type**:
```typescript
interface Attribution {
  barrier: number;    // 0-100 (percent contribution)
  tape: number;       // 0-100
  fuel: number;       // 0-100
  confluence: number; // 0-100
}
```

**ConfluenceGroup Type**:
```typescript
interface ConfluenceGroup {
  levels: LevelSignal[];
  centerPrice: number;
  strengthScore: number;
  dominantBias: 'BREAK' | 'BOUNCE';
}
```

---

## Strength Computation Contract

### Break Strength Formula

```typescript
function computeBreakStrength(level: LevelSignal): number {
  const direction = level.direction; // 'UP' or 'DOWN'
  
  // Barrier contribution (0-100)
  let barrierScore = 0;
  if (level.barrier_state === 'VACUUM') barrierScore = 100;
  else if (level.barrier_state === 'WEAK') barrierScore = 75;
  else if (level.barrier_state === 'CONSUMED') barrierScore = 60;
  else if (level.barrier_state === 'NEUTRAL') barrierScore = 50;
  else barrierScore = 0; // WALL, ABSORPTION
  
  // Tape contribution (0-100)
  let tapeScore = 50;
  if (level.sweep_detected) {
    tapeScore = 100; // Sweep confirms direction
  } else {
    // Use velocity + imbalance alignment
    const velocityAligned = (
      (direction === 'UP' && level.tape_velocity > 0) ||
      (direction === 'DOWN' && level.tape_velocity < 0)
    );
    const imbalanceAligned = (
      (direction === 'UP' && level.tape_imbalance > 0) ||
      (direction === 'DOWN' && level.tape_imbalance < 0)
    );
    
    if (velocityAligned && imbalanceAligned) tapeScore = 75;
    else if (velocityAligned || imbalanceAligned) tapeScore = 60;
  }
  
  // Fuel contribution (0-100)
  let fuelScore = 50;
  if (level.fuel_effect === 'AMPLIFY') fuelScore = 100;
  else if (level.fuel_effect === 'DAMPEN') fuelScore = 0;
  
  // Weighted sum
  const breakStrength = 
    0.30 * barrierScore +
    0.25 * tapeScore +
    0.25 * fuelScore +
    0.10 * (level.approach_velocity > 0 ? 75 : 50) +
    0.10 * 50; // Confluence placeholder
  
  return Math.max(0, Math.min(100, breakStrength));
}
```

### Bounce Strength Formula

```typescript
function computeBounceStrength(level: LevelSignal): number {
  // Inverse of break strength logic
  const direction = level.direction;
  
  // Barrier contribution (0-100)
  let barrierScore = 0;
  if (level.barrier_state === 'WALL') barrierScore = 100;
  else if (level.barrier_state === 'ABSORPTION') barrierScore = 100;
  else if (level.barrier_state === 'NEUTRAL') barrierScore = 50;
  else barrierScore = 0; // VACUUM, WEAK, CONSUMED
  
  // Tape contribution (opposite of break)
  let tapeScore = 50;
  const velocityOpposed = (
    (direction === 'UP' && level.tape_velocity < 0) ||
    (direction === 'DOWN' && level.tape_velocity > 0)
  );
  if (velocityOpposed) tapeScore = 75;
  
  // Fuel contribution (opposite of break)
  let fuelScore = 50;
  if (level.fuel_effect === 'DAMPEN') fuelScore = 100;
  else if (level.fuel_effect === 'AMPLIFY') fuelScore = 0;
  
  // Weighted sum
  const bounceStrength = 
    0.30 * barrierScore +
    0.25 * tapeScore +
    0.25 * fuelScore +
    0.10 * (level.approach_velocity < 0 ? 75 : 50) +
    0.10 * 50;
  
  return Math.max(0, Math.min(100, bounceStrength));
}
```

---

## Component Interfaces

### 1. CommandCenterComponent

**Location**: `src/app/command-center/command-center.component.ts`  
**Purpose**: Main layout container

**Inputs**: None (subscribes to services)  
**Outputs**: None (renders child components)

**Template Structure**:
```html
<div class="command-center">
  <div class="left-panel">
    <app-price-ladder></app-price-ladder>
  </div>
  <div class="center-panel">
    <app-strength-cockpit></app-strength-cockpit>
  </div>
  <div class="right-panel">
    <app-confluence-stack></app-confluence-stack>
    <app-attribution-bar></app-attribution-bar>
    <app-options-panel></app-options-panel>
  </div>
</div>
```

---

### 2. PriceLadderComponent

**Location**: `src/app/price-ladder/price-ladder.component.ts`  
**Purpose**: Vertical price ladder with level markers

**Inputs**:
```typescript
@Input() levels: LevelSignal[];
@Input() spot: number;
@Input() rangeWindow: number = 6.0; // ±$6 from spot
```

**Outputs**:
```typescript
@Output() levelSelected = new EventEmitter<LevelSignal>();
@Output() levelHovered = new EventEmitter<LevelSignal | null>();
```

**Rendering Contract**:
- Display $1 increments as horizontal lines
- Emphasize key level kinds (PM_HIGH, OR_LOW, SMA_200/400) with chips
- Show level state via color coding:
  - VACUUM: Red border
  - WALL: Green border
  - NEUTRAL: Gray border
- Render distance labels (e.g., "+0.42" for resistance $0.42 above spot)

---

### 3. StrengthCockpitComponent

**Location**: `src/app/strength-cockpit/strength-cockpit.component.ts`  
**Purpose**: Break/bounce meters + dealer mechanics

**Inputs**:
```typescript
@Input() breakStrength: number;    // 0-100
@Input() bounceStrength: number;   // 0-100
@Input() nearestLevel: LevelSignal | null;
@Input() gammaExposure: number;
@Input() gammaVelocity: number;
@Input() tapeVelocity: number;
@Input() approachSpeed: number;
```

**Rendering Contract**:
- Dual meters: horizontal bars (break = orange, bounce = teal)
- Call/put success cards (derived from direction + strength)
- Dealer gamma bar: bidirectional (-SHORT → LONG+)
- Gamma velocity indicator with contextual hints:
  - ">500/s": "Dealers accumulating FAST"
  - ">100/s": "Dealers building position"
  - "<-100/s": "Dealers reducing exposure"
  - "<-500/s": "Dealers exiting FAST"
  - Neutral: "Stable positioning"

---

### 4. AttributionBarComponent

**Location**: `src/app/attribution-bar/attribution-bar.component.ts`  
**Purpose**: Physics contribution breakdown

**Inputs**:
```typescript
@Input() attribution: Attribution;  // {barrier, tape, fuel, confluence}
```

**Rendering Contract**:
- Stacked horizontal bar (100% width)
- Segments colored by physics type:
  - Barrier: Blue
  - Tape: Yellow
  - Fuel: Red
  - Confluence: Purple
- Show percentages on hover

---

### 5. ConfluenceStackComponent

**Location**: `src/app/confluence-stack/confluence-stack.component.ts`  
**Purpose**: Grouped level clusters

**Inputs**:
```typescript
@Input() confluenceGroups: ConfluenceGroup[];
@Input() confluenceThreshold: number = 0.15; // Default band
```

**Outputs**:
```typescript
@Output() groupSelected = new EventEmitter<ConfluenceGroup>();
```

**Rendering Contract**:
- List of confluence groups (sorted by proximity to spot)
- Each group shows:
  - Center price
  - Level count (stacked labels)
  - Strength score (0-100)
  - Dominant bias (BREAK/BOUNCE icon)

---

### 6. OptionsPanelComponent

**Location**: `src/app/options-panel/options-panel.component.ts`  
**Purpose**: Strike grid + flow chart

**Inputs**:
```typescript
@Input() flow: { [optionSymbol: string]: FlowData };
@Input() spot: number;
```

**FlowData Type**:
```typescript
interface FlowData {
  cumulative_volume: number;
  cumulative_premium: number;
  last_price: number;
  net_delta_flow: number;
  net_gamma_flow: number;
  delta: number;
  gamma: number;
  strike_price: number;
  type: 'C' | 'P';
  expiration: string;
  last_timestamp: number;
}
```

**Rendering Contract**:
- Strike grid: rows for strikes (calls left, puts right)
- Highlight ATM strike (nearest to spot)
- Color code by gamma flow (positive = green, negative = red)
- Flow chart: time series of net delta/gamma flow

---

## State Management

### Component State

**No global state management** (NgRx/NGRX not used).  
**Pattern**: Services with RxJS BehaviorSubjects.

**Example** (LevelDerivedService):
```typescript
class LevelDerivedService {
  private levelsSubject = new BehaviorSubject<ProcessedLevel[]>([]);
  levels$ = this.levelsSubject.asObservable();
  
  updateLevels(rawLevels: LevelSignal[]): void {
    const processed = rawLevels.map(level => ({
      ...level,
      breakStrength: this.computeBreakStrength(level),
      bounceStrength: this.computeBounceStrength(level),
      attribution: this.computeAttribution(level)
    }));
    this.levelsSubject.next(processed);
  }
}
```

---

## Styling Contract

### Color Palette

**Base Colors**:
- Background: `#0a0e27` (deep navy)
- Surface: `#1a1f3a` (charcoal)
- Text: `#e0e6ed` (light gray)

**Signal Colors**:
- Break: `#ff6b35` (ember orange)
- Bounce: `#00d9ff` (electric teal)
- Neutral: `#6c757d` (steel gray)

**Physics Colors**:
- Barrier: `#4a90e2` (blue)
- Tape: `#f5a623` (yellow)
- Fuel: `#e94b3c` (red)
- Confluence: `#9b59b6` (purple)

**Gamma Indicator**:
- Short (negative): `#e94b3c` (red)
- Long (positive): `#2ecc71` (green)

---

## Performance Requirements

### Rendering Cadence

**Update Frequency**: 250ms (4 Hz, matches backend snap interval)

**Optimization Strategy**:
- Use `OnPush` change detection for components
- Debounce WebSocket messages (no faster than 10 fps UI updates)
- Canvas rendering for fluid indicators (future)
- Virtual scrolling for long level lists (if >50 levels)

**Example** (Component):
```typescript
@Component({
  selector: 'app-price-ladder',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class PriceLadderComponent {
  // Inputs trigger change detection only when references change
}
```

---

## Error Handling Contract

### WebSocket Disconnection

**Behavior**:
1. Display "Disconnected" indicator (red dot)
2. Attempt reconnection every 2 seconds (exponential backoff: 2s, 4s, 8s, max 30s)
3. Show last known data with "stale" indicator
4. Clear stale data after 60 seconds of disconnection

**Implementation** (DataStreamService):
```typescript
private reconnect(): void {
  this.reconnectAttempts++;
  const delay = Math.min(2000 * Math.pow(2, this.reconnectAttempts), 30000);
  
  setTimeout(() => {
    this.connect();
  }, delay);
}
```

---

### Malformed Data

**Behavior**:
1. Log error to console
2. Skip malformed level (don't crash UI)
3. Display "data unavailable" state for affected components

**Implementation** (DataStreamService):
```typescript
private parseMessage(data: string): LevelData | null {
  try {
    const parsed = JSON.parse(data);
    // Validate required fields
    if (!parsed.spy || !parsed.levels) {
      console.error('Invalid payload structure', parsed);
      return null;
    }
    return parsed as LevelData;
  } catch (e) {
    console.error('Failed to parse WebSocket message', e);
    return null;
  }
}
```

---

## Testing Interfaces

### Component Testing

**Framework**: Jasmine + Karma

**Example Test**:
```typescript
describe('StrengthCockpitComponent', () => {
  it('should display break strength meter', () => {
    component.breakStrength = 85;
    fixture.detectChanges();
    
    const meter = fixture.debugElement.query(By.css('.break-meter'));
    expect(meter.nativeElement.style.width).toBe('85%');
  });
  
  it('should show gamma velocity hints', () => {
    component.gammaVelocity = 600; // Fast accumulation
    fixture.detectChanges();
    
    const hint = fixture.debugElement.query(By.css('.gamma-hint'));
    expect(hint.nativeElement.textContent).toContain('accumulating FAST');
  });
});
```

---

### Service Testing

**Example Test**:
```typescript
describe('LevelDerivedService', () => {
  it('should compute break strength from VACUUM state', () => {
    const level: LevelSignal = {
      id: 'TEST',
      barrier_state: 'VACUUM',
      fuel_effect: 'AMPLIFY',
      sweep_detected: true,
      direction: 'UP',
      // ... other required fields
    };
    
    const strength = service.computeBreakStrength(level);
    expect(strength).toBeGreaterThan(80); // High break strength
  });
});
```

---

## Critical Invariants

1. **WebSocket resilience**: Never crash UI on malformed frames
2. **Fail-soft parsing**: Skip invalid levels, continue rendering valid ones
3. **No compute logic**: All physics computed in backend; frontend only displays
4. **Direction normalization**: Backend sends "UP"/"DOWN", not "SUPPORT"/"RESISTANCE"
5. **Time units**: Backend sends Unix milliseconds for `ts` field
6. **OnPush optimization**: All data-bound components use OnPush change detection

---

## References

- Full frontend documentation: [FRONTEND.md](../FRONTEND.md)
- Backend interface: [backend/src/gateway/INTERFACES.md](../backend/src/gateway/INTERFACES.md)
- Level signals schema: [backend/src/core/INTERFACES.md](../backend/src/core/INTERFACES.md)
- WebSocket protocol: Gateway README §WebSocket Protocol

