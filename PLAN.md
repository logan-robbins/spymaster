### 0) Scope & principles (What we are building right now)

- **Asset**: **SPY only** (underlying + SPY 0DTE options).
- **Problem**: For a **critical level** \(L\), continuously decide:
  - **BREAK THROUGH** (level fails; price crosses and runs) vs
  - **REJECT / BOUNCE** (level holds; price reverses away)
  - **Either direction** (support tests and resistance tests).
- **Method**: “Physics + Math”, no hindsight calibration. Purely real-time state:
  - **Barrier / Liquidity**: is displayed liquidity **evaporating** (vacuum) or **replenishing** (wall/absorption)?
  - **Tape / Momentum**: is tape aggression **confirming** the direction into the level?
  - **Fuel / Hedging**: are dealers likely to **amplify** or **dampen** the move (gamma regime at/near level)?
- **Cadence**:
  - ingestion remains **event-driven** (process every trade/quote update)
  - scoring + publish occurs on a fixed **snap tick** (100–250ms)
- **Smoothing**: compute both **raw** signals and a **smoothed** version (EWMA / robust rolling) for stability.

**Non-goals (v1)**:
- No “trained” calibration; constants are mechanical/tunable.
- No ES L3/MBO in v1 (but we keep an upgrade path).
- No full multi-asset generalization (SPY-only).

**SPY levels + ES liquidity (current implementation)** ✅ COMPLETE:
- **Levels are SPY prices** (strikes, rounds, VWAP) because we trade SPY 0DTE options.
- **Barrier physics uses ES MBP-10 + trades** as the liquidity source (ES has superior depth visibility vs SPY L1).
- **Price conversion** (`price_converter.py`): ES ≈ SPY × 10. When computing barrier state at SPY level $L$, we query ES depth at $L \times 10$ (dynamic ratio supported).
- **Fuel engine uses SPY options** (gamma flow from Polygon API).
- **Tape engine uses ES trades** (queries use ES-converted levels).
- **All outputs in SPY terms**: `get_spot()`, `get_bid_ask()`, defending quotes, level prices.
- **Time units**:
  - Vendor WS timestamps: **Unix ms**
  - Internal event-time: `ts_event_ns` **Unix ns (UTC)**
  - Internal receive-time: `ts_recv_ns` **Unix ns (UTC)**

This architecture gives us:
- Institutional-grade liquidity visibility from ES futures
- SPY option strike alignment for gamma/fuel analysis
- Unified SPY-denominated output for trading decisions

---

### 1) System architecture (“Snap Engine”)

We unify 3 asynchronous streams into a consistent event loop:

- **ES futures (liquidity source for barrier/tape physics)**
  - Trades stream: ES time-and-sales (from Databento DBN files or live feed)
  - Book stream: **MBP-10** (top 10 price levels per side)
  - Price conversion: ES prices are converted to SPY-equivalent using dynamic ratio (ES/SPY ≈ 10)
- **SPY options (fuel/gamma source)**
  - Trades stream: `T.O:SPY...` (Polygon WebSocket)
  - Greeks snapshots: cached via REST (Polygon API)
- **SPY spot price (for level generation)**
  - Derived from ES (ES_price / conversion_ratio) OR from SPY quotes if available

**Snap loop**:
- Every **100ms** (or 250ms to match current broadcaster), compute a `MarketState` using **LastKnownValue** from each stream.
- Compute `LevelSignals` for all active levels and publish over WS.
  - **Important**: this cadence is the *scoring/publish* cadence. **Book/trade ingestion remains event-driven** (ingest every MBP update + every trade), and engines compute windowed features from those event buffers. This avoids “churn blindness” from coarse sampling.

#### 1.1 High-level pipeline (institutional-grade stream → consume → store)

**Design goals (v1 on Apple M4, v2 in colocation)**:
- **Append-only, replayable, schema-versioned** raw capture (no lossy transforms).
- **Separation of concerns**: ingestion, normalization, stateful compute, storage are independent modules/processes.
- **Event-time first**: every record carries `ts_event_ns` (UTC) and `ts_recv_ns` (ingest).
- **Idempotency + recovery**: at-least-once inputs become exactly-once *storage* via deterministic keys + dedup.
- **ML-friendly**: columnar, typed, partitioned, joinable datasets; clean bronze/silver/gold tiers.
- **Upgrade path**: keep the same engine interfaces while swapping:
  - `asyncio.Queue` → **NATS JetStream / Redpanda (Kafka)** in colocation
  - local filesystem → **S3/object store** (same partitioning)
  - “plain Parquet directories” → **Iceberg** table metadata (optional, later)

**High-level pipeline**:
- **Feed adapters** (Massive WS/REST (Polygon legacy) now, direct feeds later) emit normalized events
- **Internal bus** carries events to consumers
- **Consumers**:
  - **Stateful realtime engines** (Barrier/Tape/Fuel/Score)
  - **Storage writer** (bronze micro-batches + curated silver/gold)
  - **WS/API server** (publishes live `LevelSignals`)