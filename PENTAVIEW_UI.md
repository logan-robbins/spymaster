## Pentaview Stream Bars Schema

### Source File
**Location**: `data/gold/streams/pentaview/version=3.1.0/date=YYYY-MM-DD/stream_bars.parquet`

**Granularity**: One row per 2-minute bar per active level

**Update Frequency**: Real-time (30s state → 2min bar aggregation)

---

## Schema Definition

### Core Fields

```typescript
interface StreamBar {
  // ============================================================
  // METADATA (for context and filtering)
  // ============================================================
  timestamp: string;              // ISO 8601: "2024-12-16T10:32:00"
  level_kind: LevelKind;          // "PM_HIGH" | "PM_LOW" | "OR_HIGH" | "OR_LOW" | "SMA_90" | "EMA_20"
  direction: Direction;           // "UP" | "DOWN" (approach direction)
  spot: number;                   // Current underlying price (e.g., 6875.25)
  atr: number;                    // Current ATR value (e.g., 12.5)
  level_price: number;            // Actual level price (e.g., 6880.00)
  
  // ============================================================
  // CANONICAL STREAMS (Primary signals, all in [-1, +1])
  // ============================================================
  sigma_m: number;                // MOMENTUM: directional price dynamics
                                  // > 0 = upward momentum, < 0 = downward
  
  sigma_f: number;                // FLOW: order aggression
                                  // > 0 = net buying, < 0 = net selling
  
  sigma_b: number;                // BARRIER: liquidity dynamics (directional via dir_sign)
                                  // > 0 = barrier favors break up, < 0 = favors break down
  
  sigma_d: number;                // DEALER/GAMMA: non-directional amplifier
                                  // > 0 = amplification (fuel), < 0 = dampening (pin)
  
  sigma_s: number;                // SETUP: quality/confidence scalar
                                  // > +0.5 = high quality, < -0.5 = degraded
  
  // ============================================================
  // MERGED STREAMS (Composite signals, all in [-1, +1])
  // ============================================================
  sigma_p: number;                // PRESSURE: 0.55*sigma_m + 0.45*sigma_f
                                  // Primary directional signal
  
  sigma_r: number;                // STRUCTURE: 0.70*sigma_b + 0.30*sigma_s
                                  // Structural support for moves
  
  // ============================================================
  // COMPOSITE METRICS (Alignment indicators, in [-1, +1])
  // ============================================================
  alignment: number;              // Directional alignment (M+F+B weighted by S)
                                  // +1 = all streams agree bullish, -1 = all bearish
  
  divergence: number;             // Standard deviation of directional streams
                                  // 0 = perfect alignment, 1 = max divergence
  
  alignment_adj: number;          // Dealer-adjusted alignment
                                  // alignment * (1.0 + 0.35*sigma_d)
  
  // ============================================================
  // DERIVATIVES (TA-style acceleration indicators)
  // Available for: sigma_p, sigma_m, sigma_f, sigma_b
  // ============================================================
  
  // Pressure derivatives (primary for charting)
  sigma_p_smooth: number;         // EMA-smoothed pressure
  sigma_p_slope: number;          // 1st derivative (rate of change)
  sigma_p_curvature: number;      // 2nd derivative (acceleration)
  sigma_p_jerk: number;           // 3rd derivative (rate of acceleration)
  
  // Momentum derivatives
  sigma_m_smooth: number;
  sigma_m_slope: number;
  sigma_m_curvature: number;
  sigma_m_jerk: number;
  
  // Flow derivatives
  sigma_f_smooth: number;
  sigma_f_slope: number;
  sigma_f_curvature: number;
  sigma_f_jerk: number;
  
  // Barrier derivatives
  sigma_b_smooth: number;
  sigma_b_slope: number;
  sigma_b_curvature: number;
  sigma_b_jerk: number;
}

// Type definitions
type LevelKind = "PM_HIGH" | "PM_LOW" | "OR_HIGH" | "OR_LOW" | "SMA_90" | "EMA_20";
type Direction = "UP" | "DOWN";
```

---

## Chart Overlay Configuration

### Recommended Setup

**Chart Type**: Candlestick (OHLCV) + Line overlays

**Y-Axes**:
- **Left Axis**: Price scale (e.g., 6850-6900)
- **Right Axis**: Stream scale (-1.0 to +1.0)

**Data Alignment**: Match `timestamp` from stream bars to candle bar timestamps

---

### Option 1: Five Canonical Streams (Pure Signals)

```typescript
interface ChartData {
  timestamp: string;              // X-axis
  
  // Candlestick (left Y-axis)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Stream overlays (right Y-axis, -1 to +1)
  sigma_m: number;                // Line color: Blue (momentum)
  sigma_f: number;                // Line color: Green (flow)
  sigma_b: number;                // Line color: Orange (barrier)
  sigma_d: number;                // Line color: Purple (dealer)
  sigma_s: number;                // Line color: Gray (setup quality)
}
```

**Visual Style**:
```typescript
const streamStyles = {
  sigma_m: { color: '#2196F3', width: 1.5, name: 'Momentum' },
  sigma_f: { color: '#4CAF50', width: 1.5, name: 'Flow' },
  sigma_b: { color: '#FF9800', width: 1.5, name: 'Barrier' },
  sigma_d: { color: '#9C27B0', width: 1.5, name: 'Dealer' },
  sigma_s: { color: '#757575', width: 1.0, name: 'Setup', style: 'dashed' }
};
```

---

### Option 2: Merged Streams (Simplified View)

```typescript
interface ChartDataSimplified {
  timestamp: string;
  
  // Candlestick
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Primary overlays (right Y-axis)
  sigma_p: number;                // PRESSURE (primary directional)
  sigma_r: number;                // STRUCTURE (support/resistance)
  sigma_s: number;                // SETUP (quality gate)
  
  // Optional: Alignment metrics
  alignment: number;              // Consensus indicator
  divergence: number;             // Conflict/uncertainty indicator
}
```

**Visual Style**:
```typescript
const mergedStreamStyles = {
  sigma_p: { color: '#F44336', width: 2.0, name: 'Pressure' },     // Red/Green gradient
  sigma_r: { color: '#2196F3', width: 1.5, name: 'Structure' },    // Blue
  sigma_s: { color: '#FFC107', width: 1.0, name: 'Setup', style: 'dashed' }  // Yellow
};
```

---

### Option 3: Pressure + Derivatives (TA Style)

```typescript
interface ChartDataTAStyle {
  timestamp: string;
  
  // Candlestick
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Pressure stream (right Y-axis, -1 to +1)
  sigma_p: number;                // Primary line
  sigma_p_smooth: number;         // Smoothed line (optional)
  
  // Derivatives (separate panel or color-coded zones)
  sigma_p_slope: number;          // Histogram/bar chart
  sigma_p_curvature: number;      // Secondary indicator
  sigma_p_jerk: number;           // Tertiary indicator (optional)
}
```

---

## Sample Data (JSON)

### Single Bar Example

```json
{
  "timestamp": "2024-12-16T10:32:00",
  "level_kind": "OR_HIGH",
  "direction": "UP",
  "spot": 6875.25,
  "atr": 12.5,
  "level_price": 6880.00,
  
  "sigma_m": 0.67,
  "sigma_f": 0.55,
  "sigma_b": -0.42,
  "sigma_d": 0.28,
  "sigma_s": 0.71,
  
  "sigma_p": 0.62,
  "sigma_r": -0.08,
  
  "alignment": 0.45,
  "divergence": 0.58,
  "alignment_adj": 0.51,
  
  "sigma_p_smooth": 0.61,
  "sigma_p_slope": 0.08,
  "sigma_p_curvature": -0.02,
  "sigma_p_jerk": -0.005,
  
  "sigma_m_smooth": 0.66,
  "sigma_m_slope": 0.05,
  "sigma_m_curvature": -0.01,
  "sigma_m_jerk": 0.002,
  
  "sigma_f_smooth": 0.54,
  "sigma_f_slope": 0.11,
  "sigma_f_curvature": -0.03,
  "sigma_f_jerk": -0.008,
  
  "sigma_b_smooth": -0.41,
  "sigma_b_slope": -0.06,
  "sigma_b_curvature": 0.02,
  "sigma_b_jerk": 0.004
}
```

### Time Series Example

```json
[
  {
    "timestamp": "2024-12-16T10:30:00",
    "spot": 6873.50,
    "sigma_m": 0.62,
    "sigma_f": 0.48,
    "sigma_b": -0.38,
    "sigma_d": 0.25,
    "sigma_s": 0.68,
    "sigma_p": 0.56,
    "sigma_r": -0.05
  },
  {
    "timestamp": "2024-12-16T10:32:00",
    "spot": 6875.25,
    "sigma_m": 0.67,
    "sigma_f": 0.55,
    "sigma_b": -0.42,
    "sigma_d": 0.28,
    "sigma_s": 0.71,
    "sigma_p": 0.62,
    "sigma_r": -0.08
  },
  {
    "timestamp": "2024-12-16T10:34:00",
    "spot": 6877.00,
    "sigma_m": 0.71,
    "sigma_f": 0.61,
    "sigma_b": -0.45,
    "sigma_d": 0.31,
    "sigma_s": 0.74,
    "sigma_p": 0.67,
    "sigma_r": -0.09
  }
]
```

---

## Filtering & Aggregation

### Per-Level Filtering

Since each bar is per-level, you may want to filter:

```typescript
// Show streams only for the level closest to spot
const closestLevel = streamBars
  .filter(bar => bar.timestamp === currentTimestamp)
  .reduce((closest, bar) => {
    const dist = Math.abs(bar.spot - bar.level_price);
    const closestDist = Math.abs(closest.spot - closest.level_price);
    return dist < closestDist ? bar : closest;
  });

// Or show only specific level types
const orHighStreams = streamBars.filter(bar => bar.level_kind === 'OR_HIGH');
```

### Aggregation Across Levels

If you want a single "market-wide" stream value:

```typescript
// Weighted average by proximity (closer levels have more influence)
function aggregateStreams(bars: StreamBar[], spot: number): number {
  const weighted = bars.map(bar => {
    const distance = Math.abs(spot - bar.level_price);
    const weight = Math.exp(-distance / (2 * bar.atr)); // Exponential decay
    return { value: bar.sigma_p, weight };
  });
  
  const totalWeight = weighted.reduce((sum, w) => sum + w.weight, 0);
  return weighted.reduce((sum, w) => sum + w.value * w.weight, 0) / totalWeight;
}
```

---

## WebSocket Streaming Format

If streaming real-time updates:

```typescript
interface StreamUpdate {
  type: 'pentaview_update';
  timestamp: string;
  bars: StreamBar[];              // All active levels for this timestamp
}

// Example WebSocket message
{
  "type": "pentaview_update",
  "timestamp": "2024-12-16T10:32:00",
  "bars": [
    {
      "level_kind": "OR_HIGH",
      "direction": "UP",
      "spot": 6875.25,
      "sigma_p": 0.62,
      "sigma_m": 0.67,
      "sigma_f": 0.55,
      "sigma_b": -0.42,
      "sigma_d": 0.28,
      "sigma_s": 0.71,
      "sigma_r": -0.08,
      "alignment": 0.45,
      ...
    },
    {
      "level_kind": "PM_HIGH",
      "direction": "UP",
      "spot": 6875.25,
      "sigma_p": 0.58,
      ...
    }
  ]
}
```

---

## Chart Implementation Notes

### Y-Axis Scaling

**Right Axis** (streams): Fixed scale `-1.0` to `+1.0`
- Add horizontal lines at 0, ±0.35, ±0.70 for visual reference
- Color zones: 
  - `> +0.35`: Light green (bullish)
  - `-0.35 to +0.35`: Gray (neutral)
  - `< -0.35`: Light red (bearish)

### Zero Line

Draw bold horizontal line at `y=0` on stream axis (neutral/equilibrium)

### Color Gradients (Optional)

For `sigma_p` (pressure):
```typescript
function getPressureColor(value: number): string {
  if (value > 0.70) return '#1B5E20';      // Dark green (strong bull)
  if (value > 0.35) return '#4CAF50';      // Green (moderate bull)
  if (value > 0.00) return '#81C784';      // Light green (weak bull)
  if (value > -0.35) return '#EF9A9A';     // Light red (weak bear)
  if (value > -0.70) return '#F44336';     // Red (moderate bear)
  return '#B71C1C';                        // Dark red (strong bear)
}
```

### Update Frequency

- **Batch Mode**: Load full date range on page load
- **Real-Time Mode**: WebSocket updates every 30s (aggregated to 2min bars)

---

## Summary

**Final Output**: 32-column DataFrame with 2-minute granularity

**Core Fields for Chart**:
- `timestamp` (X-axis alignment)
- 5 canonical streams: `sigma_m`, `sigma_f`, `sigma_b`, `sigma_d`, `sigma_s` (all in [-1, +1])
- 2 merged streams: `sigma_p`, `sigma_r` (optional, simplified view)
- Metadata: `level_kind`, `spot`, `level_price` (for filtering/context)

**Recommended Chart**: Pressure (`sigma_p`) as primary overlay, with optional toggles for other streams