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
@Injectable({ providedIn: 'root' })
class DataStreamService {
  // Writable Signals (public state)
  public flowData: WritableSignal<FlowMap> = signal({});
  public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
  public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
  public dataStatus: WritableSignal<'ok' | 'unavailable'> = signal('unavailable');
  public lastError: WritableSignal<string | null> = signal(null);

  // Connection management (connect is PRIVATE, called in constructor)
  private connect(): void;
  public disconnect(): void;
}
```

---

### WebSocket Payload Schema

**Message Format** (from Gateway):
```typescript
interface MergedPayload {
  flow: FlowMap;
  levels: LevelsPayload;
}
```

**FlowMap Type**:
```typescript
interface FlowMap {
  [ticker: string]: FlowMetrics;
}
```

**FlowMetrics Type**:
```typescript
interface FlowMetrics {
  cumulative_volume: number;
  cumulative_premium: number;
  last_price: number;
  net_delta_flow: number;
  net_gamma_flow: number;
  delta: number;
  gamma: number;
  strike_price: number;
  type: string;           // 'C' or 'P'
  expiration: string;
  last_timestamp: number; // ms Unix epoch
}
```

**LevelsPayload Type**:
```typescript
interface LevelsPayload {
  ts: number;             // Unix milliseconds
  spy: SpySnapshot;
  levels: LevelSignal[];
}

interface SpySnapshot {
  spot: number | null;
  bid: number | null;
  ask: number | null;
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
  direction: 'UP' | 'DOWN';      // 'UP' (resistance) | 'DOWN' (support)
  distance: number;              // Dollars from spot

  // Context
  is_first_15m: boolean;
  bars_since_open: number;

  // Scores
  break_score_raw: number;       // 0-100
  break_score_smooth: number;    // 0-100 (EWMA smoothed)
  signal: 'BREAK' | 'BOUNCE' | 'CHOP';
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  note?: string;

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

## Service Interfaces

### 1. DataStreamService

**Location**: `src/app/data-stream.service.ts`
**Purpose**: WebSocket connection and stream parsing using Angular Signals

**Interface**:
```typescript
@Injectable({ providedIn: 'root' })
class DataStreamService {
  // Public Writable Signals
  public flowData: WritableSignal<FlowMap> = signal({});
  public levelsData: WritableSignal<LevelsPayload | null> = signal(null);
  public connectionStatus: WritableSignal<'connecting' | 'connected' | 'disconnected'> = signal('disconnected');
  public dataStatus: WritableSignal<'ok' | 'unavailable'> = signal('unavailable');
  public lastError: WritableSignal<string | null> = signal(null);

  // Connection management
  private connect(): void;        // Called automatically in constructor
  public disconnect(): void;
}
```

**Connection States**:
- `connecting`: Initial connection attempt
- `connected`: Active WebSocket connection
- `disconnected`: Connection lost (triggers auto-reconnect with exponential backoff)

---

### 2. LevelDerivedService

**Location**: `src/app/level-derived.service.ts`
**Purpose**: Compute UI-specific derived metrics using Angular computed signals

**Interface**:
```typescript
@Injectable({ providedIn: 'root' })
class LevelDerivedService {
  // Computed Signals (read-only)
  public spy: Signal<SpySnapshot | null>;
  public levels: Signal<DerivedLevel[]>;
  public confluenceGroups: Signal<ConfluenceGroup[]>;
  public primaryLevel: Signal<DerivedLevel | null>;

  // Confluence band configuration
  public setConfluenceBand(value: number): void;
  public getConfluenceBand(): Signal<number>;
}
```

**DerivedLevel Type** (computed from LevelSignal):
```typescript
interface DerivedLevel {
  id: string;
  price: number;
  kind: string;
  direction: 'UP' | 'DOWN';
  distance: number;
  breakStrength: number;          // 0-100
  bounceStrength: number;         // 0-100
  bias: 'BREAK' | 'BOUNCE' | 'NEUTRAL';
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  signal: 'BREAK' | 'BOUNCE' | 'CHOP';
  barrier: {
    state: string;
    deltaLiq: number;
    wallRatio: number;
    replenishmentRatio: number;
  };
  tape: {
    imbalance: number;
    velocity: number;
    buyVol: number;
    sellVol: number;
    sweepDetected: boolean;
  };
  fuel: {
    effect: string;
    gammaExposure: number;
    gammaVelocity: number;
  };
  approach: {
    velocity: number;
    bars: number;
    distance: number;
    priorTouches: number;
    barsSinceOpen: number;
    isFirst15m: boolean;
  };
  contributions: {
    barrier: number;    // 0-100 (percent contribution)
    tape: number;       // 0-100
    fuel: number;       // 0-100
    approach: number;   // 0-100
    confluence: number; // 0-100
  };
  confluenceId?: string;
}
```

**ConfluenceGroup Type**:
```typescript
interface ConfluenceGroup {
  id: string;
  centerPrice: number;
  levels: DerivedLevel[];
  strength: number;               // 0-100 (normalized)
  bias: 'BREAK' | 'BOUNCE' | 'NEUTRAL';
  score: number;                  // Raw aggregated score
}
```

---

### 3. FlowAnalyticsService

**Location**: `src/app/flow-analytics.service.ts`
**Purpose**: Compute flow derivatives (velocity, acceleration) across buckets

**Interface**:
```typescript
@Injectable({ providedIn: 'root' })
class FlowAnalyticsService {
  // UI Selector Signals
  public selectedMetric: WritableSignal<FlowMetric>;
  public selectedOrder: WritableSignal<DerivativeOrder>;
  public selectedBarInterval: WritableSignal<BarIntervalMs>;

  // Computed State
  public atmStrike: Signal<number | null>;
  public anchorStrike: WritableSignal<number | null>;
  public perStrikeVel: Signal<StrikeVelocityMap>;

  // Methods
  public setAnchor(strike: number | null): void;
  public getSeries(bucket: BucketKey): Array<{ time: number, value: number }>;
  public getLatestByBucket(): LatestByBucket;
  public getLatestAbsMax(): number;
}

type FlowMetric = 'premium' | 'delta' | 'gamma';
type DerivativeOrder = 1 | 2 | 3;  // Velocity, Acceleration, Jerk
type BarIntervalMs = 5000 | 30000 | 60000;
type BucketKey = 'call_above' | 'call_below' | 'put_above' | 'put_below';
```

---

## Component Interfaces

### 1. CommandCenterComponent

**Location**: `src/app/command-center/command-center.component.ts`
**Purpose**: Main layout container with header and grid

**Template Structure**:
```html
<div class="command-center">
  <header class="command-header">
    <div class="brand">
      <div class="brand-title">SPYMASTER</div>
      <div class="brand-subtitle">0DTE BOUNCE/BREAK COMMAND</div>
    </div>
    <div class="status">
      <div class="status-pill">{{ statusText() }}</div>
      <div class="spy-price">SPY {{ spy()!.spot }}</div>
    </div>
  </header>

  <div class="command-grid">
    <section class="panel ladder-panel">
      <app-price-ladder [range]="6"></app-price-ladder>
    </section>
    <section class="panel cockpit-panel">
      <app-strength-cockpit></app-strength-cockpit>
      <app-attribution-bar></app-attribution-bar>
      <app-confluence-stack></app-confluence-stack>
    </section>
    <section class="panel options-panel">
      <app-options-panel></app-options-panel>
    </section>
  </div>
</div>
```

**Computed Signals**:
```typescript
class CommandCenterComponent {
  private dataStream = inject(DataStreamService);
  private derived = inject(LevelDerivedService);

  public spy = computed(() => this.derived.spy());
  public connectionStatus = this.dataStream.connectionStatus;
  public dataStatus = this.dataStream.dataStatus;
  public statusText = computed(() => /* status text logic */);
}
```

---

### 2. PriceLadderComponent

**Location**: `src/app/price-ladder/price-ladder.component.ts`
**Purpose**: Vertical price ladder with level markers

**Signal-Based Data** (via injected LevelDerivedService):
```typescript
class PriceLadderComponent {
  private derived = inject(LevelDerivedService);

  public spot = computed(() => this.derived.spy()?.spot ?? null);
  public ticks = computed(() => /* ladder tick positions */);
  public spotMarker = computed(() => /* spot position percentage */);
  public markers = computed(() => /* DerivedLevel[] with positions */);
}
```

---

### 3. StrengthCockpitComponent

**Location**: `src/app/strength-cockpit/strength-cockpit.component.ts`
**Purpose**: Break/bounce meters + dealer mechanics

**Signal-Based Data** (via injected LevelDerivedService):
```typescript
class StrengthCockpitComponent {
  private derived = inject(LevelDerivedService);

  public level = this.derived.primaryLevel;
  public callSuccess = computed(() => /* percentage */);
  public putSuccess = computed(() => /* percentage */);
  public tapeVelocity = computed(() => level?.tape.velocity ?? 0);
  public gammaExposure = computed(() => level?.fuel.gammaExposure ?? 0);
  public gammaVelocity = computed(() => level?.fuel.gammaVelocity ?? 0);
}
```

**Gamma Velocity Hints**:
- ">500/s": "Dealers accumulating FAST"
- ">100/s": "Dealers building position"
- "<-100/s": "Dealers reducing exposure"
- "<-500/s": "Dealers exiting FAST"
- Neutral: "Stable positioning"

---

### 4. AttributionBarComponent

**Location**: `src/app/attribution-bar/attribution-bar.component.ts`
**Purpose**: Physics contribution breakdown

**Signal-Based Data**:
```typescript
class AttributionBarComponent {
  private derived = inject(LevelDerivedService);

  public level = this.derived.primaryLevel;
  public slices = computed(() => /* AttributionSlice[] */);
}
```

**Rendering Contract**:
- Stacked horizontal bar (100% width)
- Segments colored by physics type:
  - Barrier: `#38bdf8` (blue)
  - Tape: `#f97316` (orange)
  - Fuel: `#f43f5e` (red)
  - Approach: `#a855f7` (purple)
  - Confluence: `#22c55e` (green)

---

### 5. ConfluenceStackComponent

**Location**: `src/app/confluence-stack/confluence-stack.component.ts`
**Purpose**: Grouped level clusters

**Signal-Based Data**:
```typescript
class ConfluenceStackComponent {
  private derived = inject(LevelDerivedService);

  public groups = computed(() => this.derived.confluenceGroups().slice(0, 5));
  public band = this.derived.getConfluenceBand();
}
```

---

### 6. OptionsPanelComponent

**Location**: `src/app/options-panel/options-panel.component.ts`
**Purpose**: Tab container for strike grid and flow chart

**Internal State**:
```typescript
class OptionsPanelComponent {
  public activeTab = signal<'strikes' | 'flow'>('strikes');
}
```

---

### 7. StrikeGridComponent

**Location**: `src/app/strike-grid/strike-grid.component.ts`
**Purpose**: Options strike grid with calls/puts by strike

**Signal-Based Data**:
```typescript
class StrikeGridComponent {
  private dataService = inject(DataStreamService);
  private analytics = inject(FlowAnalyticsService);

  flowData = this.dataService.flowData;
  perStrikeVel = this.analytics.perStrikeVel;
  rows = computed(() => /* Array<{ strike, call?, put? }> */);
}
```

---

### 8. FlowChartComponent

**Location**: `src/app/flow-chart/flow-chart.component.ts`
**Purpose**: Time series chart of flow derivatives

**Signal-Based Data**:
```typescript
class FlowChartComponent {
  private analytics = inject(FlowAnalyticsService);

  public selectedMetric = this.analytics.selectedMetric;
  public selectedOrder = this.analytics.selectedOrder;
  public selectedBarInterval = this.analytics.selectedBarInterval;
  public latestValues = computed(() => this.analytics.getLatestByBucket());
}
```

---

## State Management

### Pattern: Angular Signals

**No global state management** (NgRx not used).
**Pattern**: Services with Angular Signals (signal, computed, effect).

**Example** (LevelDerivedService):
```typescript
@Injectable({ providedIn: 'root' })
class LevelDerivedService {
  private dataStream = inject(DataStreamService);
  private confluenceBand = signal(0.15);
  private gammaVelocity = signal<Record<string, number>>({});

  constructor() {
    effect(() => {
      const payload = this.dataStream.levelsData();
      if (!payload) return;
      this.updateGammaVelocity(payload);
    });
  }

  public levels = computed(() => {
    const payload = this.dataStream.levelsData();
    if (!payload) return [];
    return validLevels.map(level => ({
      ...level,
      breakStrength: computeBreakStrength(level),
      bounceStrength: computeBounceStrength(level),
      contributions: computeContributions(level)
    }));
  });

  public primaryLevel = computed(() => {
    const levels = this.levels();
    return levels.length ? levels[0] : null;
  });
}
```

---

## Styling Contract

### Color Palette

**Base Colors**:
- Background: `#0b1120` (deep navy)
- Surface: `#0f172a` (charcoal)
- Border: `#233047` (subtle)
- Text: `#e2e8f0` (light gray)

**Signal Colors**:
- Break: `#f87171` (ember red)
- Bounce: `#22c55e` (green)
- Neutral: `#94a3b8` (steel gray)

**Physics Colors**:
- Barrier: `#38bdf8` (blue)
- Tape: `#f97316` (orange)
- Fuel: `#f43f5e` (red)
- Approach: `#a855f7` (purple)
- Confluence: `#22c55e` (green)

---

## Performance Requirements

### Rendering Cadence

**Update Frequency**: 250ms (4 Hz, matches backend snap interval)

**Optimization Strategy**:
- Use Angular Signals for fine-grained reactivity
- Components derive from computed signals (automatic memoization)
- Use `OnPush` change detection where applicable

---

## Error Handling Contract

### WebSocket Disconnection

**Behavior**:
1. Display "Offline" indicator (red pill)
2. Attempt reconnection with exponential backoff (3s, 4.5s, max 30s)
3. Show last known data with "Data unavailable" indicator
4. Reset backoff on successful reconnection

---

### Malformed Data

**Behavior**:
1. Log error to console
2. Skip malformed level (don't crash UI)
3. Set `dataStatus` signal to `'unavailable'`
4. Continue processing valid data

**Validation** (LevelDerivedService):
```typescript
function validateLevel(level: LevelSignal): boolean {
  const values = [level.level_price, level.distance, level.tape_velocity];
  return values.every((v) => typeof v === 'number' && Number.isFinite(v));
}
```

---

## Critical Invariants

1. **WebSocket resilience**: Never crash UI on malformed frames
2. **Fail-soft parsing**: Skip invalid levels, continue rendering valid ones
3. **No compute logic duplication**: All physics computed in backend; frontend only computes derived UI metrics
4. **Direction normalization**: Backend sends "UP"/"DOWN", not "SUPPORT"/"RESISTANCE"
5. **Time units**: Backend sends Unix milliseconds for `ts` field
6. **Signal-based reactivity**: All state changes flow through Angular Signals
7. **Computed memoization**: Derived values are automatically cached until dependencies change

---

## References

- Full frontend documentation: [frontend/README.md](../frontend/README.md)
- Backend interface: [backend/src/gateway/INTERFACES.md](../backend/src/gateway/INTERFACES.md)
- Level signals schema: [backend/src/core/INTERFACES.md](../backend/src/core/INTERFACES.md)
