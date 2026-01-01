# Frontend API Specification

**Audience**: Frontend developers  
**Protocol**: WebSocket (JSON)  
**Endpoint**: `ws://localhost:8000/ws/stream`

---

## What You Consume

The frontend receives **two types of data** via a single WebSocket stream:

### 1. Live Market Data (For Charting)
Real-time ES futures price data from Polygon.io.

```json
{
  "spy": {
    "spot": 687.50,
    "bid": 687.49,
    "ask": 687.51
  }
}
```

**Usage**: Build your real-time price chart.

---

### 2. Level Predictions (For Trading Signals)
Our proprietary retrieval-based predictions for key price levels.

```json
{
  "levels": [
    {
      "id": "PM_HIGH_687.75",
      "level_price": 687.75,
      "level_kind_name": "PM_HIGH",
      "direction": "UP",
      "distance": 0.25,
      
      "signal": "BREAK",           // Prediction: BREAK, BOUNCE, or NO_TRADE
      "confidence": "HIGH",         // HIGH, MEDIUM, or LOW
      
      "ml_predictions": {
        "p_break": 0.73,            // Probability of breakout
        "p_bounce": 0.21,           // Probability of rejection
        "similarity": 0.89,         // How similar to historical examples
        "n_neighbors": 47           // Sample size
      }
    }
  ]
}
```

---

## How Predictions Work

We use **Geometry-Only kNN Retrieval** (32D DCT shape matching).

**What this means**:
- We match the current price chart *shape* against 10,000+ historical examples
- We only use the **trajectory geometry** (not order flow physics)
- This gives us 2.4% calibration error (very reliable)

**What you can trust**:
- ✅ `p_break` and `p_bounce` are **well-calibrated**
  - If we say 70% break → it breaks 70% of the time
- ✅ `confidence: "HIGH"` means high similarity to past examples
- ✅ Works best for structural levels (SMA, Open Range, Pre-Market)

**What to ignore** (for now):
- Physics features (`approach_velocity`, `tape_imbalance`, etc.) are provided for context but **not used in predictions**
- These will be integrated in Phase 5 (Transformer model)

---

## Message Schema

### Full WebSocket Message
```typescript
interface SpymasterMessage {
  ts: number;              // Timestamp (ms)
  spy: {
    spot: number;
    bid: number;
    ask: number;
  };
  levels: Level[];
}

interface Level {
  // Identity
  id: string;
  level_price: number;
  level_kind_name: string;  // PM_HIGH, OR_LOW, SMA_90, etc.
  direction: "UP" | "DOWN";
  distance: number;
  
  // Prediction (GEOMETRY-BASED)
  signal: "BREAK" | "BOUNCE" | "NO_TRADE";
  confidence: "HIGH" | "MEDIUM" | "LOW";
  
  ml_predictions: {
    p_break: number;        // [0, 1]
    p_bounce: number;       // [0, 1]
    similarity: number;     // [0, 1] - geometric similarity to neighbors
    n_neighbors: number;    // Sample size (higher = more reliable)
  };
  
  // Context (for display, not predictions)
  barrier_state: string;
  gamma_exposure: number;
  fuel_effect: "AMPLIFY" | "DAMPEN" | "NEUTRAL";
  confluence_level_name: string;
  
  // Optional
  note?: string;
}
```

---

## Update Frequency

**Cadence**: ~250ms (4 Hz)  
**Latency**: <10ms from market data to your browser

---

## Connection Example

### TypeScript (Angular/React)
```typescript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onmessage = (event) => {
  const msg: SpymasterMessage = JSON.parse(event.data);
  
  // Update chart with spot price
  updateChart(msg.spy.spot);
  
  // Update level signals
  msg.levels.forEach(level => {
    if (level.confidence === "HIGH") {
      displaySignal(level);
    }
  });
};
```

---

## Key Design Decisions

### Why Geometry Only?
Our research (Phase 4) proved that **shape matching outperforms physics matching** for these predictions:
- Geometry-Only: 71.8% accuracy, 2.4% calibration error
- Physics-Only: 69.1% accuracy, -17% calibration error (inverted!)

Physics features (velocity, order flow) are "adversarial" - high velocity can mean either a breakout OR a fakeout depending on context. Our current system can't distinguish, so we exclude them.

### Future: Transformer (Phase 5)
When we deploy the neural model, it will learn when to use physics:
- At **PM_HIGH**: Physics helps (+6.5% accuracy)
- At **SMA_90**: Physics hurts (-11.3% accuracy)

The transformer will add a `ml_predictions.transformer` field alongside the current `ml_predictions` kNN output.

---

## Complete Reference
- Full schema: `backend/src/gateway/INTERFACES.md`
- Gateway implementation: `backend/src/gateway/README.md`
- Research findings: `backend/RESEARCH.md`
