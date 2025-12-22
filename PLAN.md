## SPY Break/Reject Physics Engine (v1) — Parallel Agent Implementation Plan

### How to use this plan (for AI coding agents)

- This document is intentionally written **top-down**: scope → architecture → contracts → engines → integration → testing → acceptance → external vendor specs → **agent tasking**.
- **Do not change shared interfaces** (schemas, dataclasses, config keys) without coordinating; agents are meant to work in parallel using these stable contracts.
- **Time units**:
  - Vendor WS timestamps: **Unix ms**
  - Internal event-time: `ts_event_ns` **Unix ns (UTC)**
  - Internal receive-time: `ts_recv_ns` **Unix ns (UTC)**

### Table of contents

- §0 Scope & principles
- §1 System architecture (“Snap Engine”)
- §2 Canonical message envelope & dataset contracts (Bronze/Silver/Gold)
- §3 Definitions (levels, direction, monitoring zone)
- §4 Level universe (what levels exist)
- §5 Core engines (Barrier / Tape / Fuel / Score / Smoothing / Room-to-Run)
- §6 Backend integration plan (fit existing repo)
- §7 Frontend plan (Angular)
- §8 Testing & simulation plan
- §9 Configuration (single source of truth)
- §10 Acceptance criteria (v1)
- §11 Web-dependent vendor contracts (Dec 2025) — Massive + Databento
- §12 Parallel agent assignments (Agent A, Agent B, …)

---

### 0) Scope & principles (What we are building right now)

- **Asset**: **SPY only** (underlying + SPY 0DTE options).
- **Problem**: For each **critical level** \(L\), continuously decide:
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

#### 0.1 CD.md + GM.md synthesis (why the model looks like this)

CD.md provides the **state machine** and decision logic:
- **VACUUM**: liquidity disappears **without prints** → break is easy.
- **ABSORPTION / WALL**: liquidity gets hit/lifted **but replenishes** → reject/bounce likely.
- **CONSUMED**: liquidity is being eaten faster than replenished → contested → often resolves into break.
- Add **Room to Run**: after break/bounce, distance to next obstacle.

GM.md provides the **3-module decomposition** + **composite score**:
- **Barrier Engine (CAN it move?)**
- **Fuel Engine (WILL it move?)**
- **Tape Engine (IS it moving?)**
- Combine into a **Break Score 0–100**, updated on a fixed cadence (100–250ms).

**SPY-only adaptation (v1 reality)**:
- We do **not** have ES **L3/MBO** depth in v1; we implement Barrier Physics using **SPY quotes + trades** (L1/NBBO) as a first-order proxy.
- **Preferred upgrade (institutional)**: use **ES L2 (MBP-10) + ES trades** to infer **pulled vs filled vs replenished** near levels **without requiring L3** (see §5.1.1).
- The model is designed so we can **swap Barrier inputs** (SPY L1 → ES L2 → ES L3) **without changing** the Fuel/Tape/Score interfaces.

---

### 1) System architecture (“Snap Engine”)

We unify 3 asynchronous streams into a consistent event loop:

- **SPY underlying**
  - Trades stream: `T.SPY`
  - Quotes stream: `Q.SPY` (NBBO)
- **(Optional / preferred later) ES futures as liquidity source**
  - Trades stream: ES time-and-sales
  - Book stream: **MBP-10** (top 10 price levels per side)
- **SPY options**
  - Trades stream: `T.O:SPY...` (already subscribed dynamically)
  - Greeks snapshots: cached via REST (already implemented)

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

#### 1.2 Phased rollout (get running fast on M4 → evolve to servers)

**Phase 0 — Fast local replay on Apple M4 (single process, days not weeks)**
- **Bus**: in-process `asyncio.Queue` (already present)
- **Storage**:
  - Write **Bronze** Parquet immediately using canonical schemas + partitions
  - Write **Gold** `levels.signals` Parquet (optional but recommended early for debugging)
  - Skip Silver compaction initially (keep it simple)
- **Durability**: best-effort micro-batching (1–5s) + atomic file commits
- **Replay**:
  - Replay from Bronze partitions by time range (deterministic event order by `ts_event_ns`)
  - Goal: “press play” locally and reproduce the same `levels.signals` stream
- **Acceptance**:
  - You can stop/start and still replay the same session from disk
  - Output parity for deterministic configs (within rounding policy)

**Phase 1 — Local institutional hygiene (M4, 1–2 weeks)**
- **Durability**:
  - Add per-stream **WAL spool** to guarantee recovery of unflushed events
  - Add **run manifests** (`_meta/runs/.../manifest.json`) and schema snapshots
- **Silver**:
  - Add a periodic **compaction job** that produces Silver:
    - deterministic dedup on `event_id`
    - sort by event-time
    - optional as-of joins (quotes↔trades, greeks↔option trades) with explicit tolerances
- **Replay**:
  - Replay from Bronze or Silver (choose based on “raw parity” vs “cleaned training”)

**Phase 2 — Single-machine “real server” (low-latency, still simple)**
- **Process boundaries** (still 1 host):
  - `ingestor` (feed adapters) → `bus` → `engine` (Barrier/Tape/Fuel/Score) → `writer` (lake)
- **Bus**: optional single-node **NATS JetStream** (or Redis Streams) for persistence + fanout
- **Storage**: local NVMe + periodic sync to object store (optional)
- **Ops**: metrics, structured logs, backpressure policies, health endpoints

**Phase 3 — Colocation / institutional deployment (scale + resilience)**
- **Bus**: **NATS JetStream** or **Redpanda/Kafka** (multi-node)
  - retention policies, consumer groups, replay by offset/time
  - schema registry (Confluent-compatible) if Kafka/Redpanda is used
- **Storage**: object store (S3/MinIO) with Bronze/Silver/Gold
- **Table format**: add **Iceberg** metadata for schema evolution + scalable analytics
- **Compute**: separate services with independent scaling; strict SLOs and monitoring

---

### 2) Canonical message envelope & dataset contracts (bus + storage)

#### 2.1 Message envelope (internal contract)

All internal events and stored records follow a common envelope:
- **`schema_name`**: e.g. `stocks.trades.v1`, `stocks.quotes.v1`, `options.trades.v1`, `levels.signals.v1`
- **`schema_version`**: integer; backwards-compatible evolution rules documented
- **`source`**: `massive_ws`, `massive_rest`, `polygon_ws` (legacy), `polygon_rest` (legacy), `replay`, `sim`, `direct_feed`
- **`ts_event_ns`** (int64 UTC): event time (trade/quote time)
- **`ts_recv_ns`** (int64 UTC): time received by our system
- **`key`**: partition key for bus + storage (e.g., `SPY`, or `SPY|2025-12-22|exp=...`)
- **`seq`** (optional): monotonic per-connection sequence for ordering diagnostics
- **`payload`**: typed fields per schema

Schema registry options:
- **v1 (simple)**: `pyarrow.Schema` + Pydantic models stored in-repo (`backend/src/schemas/`)
- **v2 (colocation)**: external **Schema Registry** (Confluent-compatible) if Kafka/Redpanda is used

#### 2.2 Storage architecture: Bronze / Silver / Gold (lakehouse-ready)

We treat storage as a **data lake** with three tiers:

- **Bronze (raw, immutable)**:
  - Near-raw normalized events as received, minimal enrichment
  - Purpose: full replay, audits, vendor parity checks
- **Silver (clean, normalized, deduped)**:
  - Canonical typed tables with stable keys
  - Joins applied where safe (e.g., attach best-known quote to stock trades, attach greeks snapshot id to option trades)
- **Gold (derived analytics & features)**:
  - `levels.signals` time-series
  - feature tables for ML (offline training) + optionally online feature snapshots

**File format (best practice)**:
- **Apache Parquet** for all tiers (columnar, compressible, ML-friendly)
- **Compression**: **ZSTD** (default level 3–6), dictionary encoding on categorical columns
- **Sorting**: within file sort by `ts_event_ns` (and `symbol` where applicable)
- **File sizing**: target 256MB–1GB files; Parquet row groups 64–256MB

**Table format (optional upgrade)**:
- v1: Hive-style partitioned Parquet directories
- v2: Add **Apache Iceberg** metadata for ACID-ish operations, schema evolution, catalogs (works on S3/object store)

#### 2.3 Canonical datasets + partitioning (SPY-only, but scalable)

All datasets live under a configurable root:
- `DATA_ROOT` (env) default: `backend/data/lake/`

Partitioning rules (balance scan speed vs file explosion):
- Partition by **date** and **hour** for high-frequency event streams
- Avoid partitioning by ultra-high cardinality fields (e.g., full option symbol) in v1

Recommended layout:

```text
DATA_ROOT/
  bronze/
    stocks/trades/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
    stocks/quotes/symbol=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
    futures/trades/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
    futures/mbp10/symbol=ES/date=YYYY-MM-DD/hour=HH/part-*.parquet
    options/trades/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
    options/greeks_snapshots/underlying=SPY/date=YYYY-MM-DD/part-*.parquet
  silver/
    stocks/trades/...
    stocks/quotes/...
    futures/trades/...
    futures/mbp10/...
    options/trades_enriched/...
  gold/
    levels/signals/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
    features/level_features/underlying=SPY/date=YYYY-MM-DD/hour=HH/part-*.parquet
  _meta/
    runs/ingest_run_id=.../manifest.json
    schemas/*.json (or arrow schema dumps)
```

#### 2.4 Storage schemas (minimum required columns)

**`stocks.trades.v1`**
- `ts_event_ns` int64 (UTC)
- `ts_recv_ns` int64 (UTC)
- `symbol` utf8 (always `SPY` for v1)
- `price` float64 (or `price_e4` int64 scaled; choose one and standardize)
- `size` int32
- `exchange` int16? (if available)
- `conditions` list<int16> or utf8 (vendor field)

**`stocks.quotes.v1` (NBBO)**
- `ts_event_ns`, `ts_recv_ns`, `symbol`
- `bid_px`, `ask_px` float64 (or scaled ints)
- `bid_sz`, `ask_sz` int32 *(shares; as of 2025-11-03 Massive reports shares (not round lots) per SEC MDI rules)*
- `bid_exch`, `ask_exch` int16? (if available)

**`options.trades.v1`**
- `ts_event_ns`, `ts_recv_ns`
- `underlying` utf8 (`SPY`)
- `option_symbol` utf8 (vendor symbol)
- `exp_date` date
- `strike` float64
- `right` utf8 (`C`/`P`)
- `price`, `size`
- `opt_bid`, `opt_ask` (if available; else null — populate from last-known `Q.O:*` option quote if subscribed)
- `aggressor` int8 (BUY=+1, SELL=-1, MID=0) — based on option BBO if available

**`options.greeks_snapshots.v1`**
- `ts_event_ns` (snapshot time)
- `underlying`, `option_symbol`, `delta`, `gamma`, `theta`, `vega`
- *(optional but recommended)* `implied_volatility`, `open_interest`, `last_quote_ts_ns`, `last_trade_ts_ns`
- `source` (`massive_rest` / `polygon_rest` legacy) + `snapshot_id` (hash of content/time)

**`options.trades_enriched.v1` (Silver)**
- `options.trades.v1` +:
  - `greeks_snapshot_id`
  - `delta`, `gamma` (as-of join; record the join tolerance)
  - `delta_notional`, `gamma_notional` (contract multiplier applied)

**`levels.signals.v1` (Gold)**
- `ts_event_ns` (snap tick time)
- `underlying` (`SPY`)
- `spot`, `bid`, `ask`
- `level_id`, `level_kind`, `level_price`
- `direction`, `distance`
- barrier metrics (state + flows)
- tape metrics (imbalance/velocity/sweep)
- fuel metrics (effect + net dealer gamma)
- `break_score_raw`, `break_score_smooth`, `signal`, `confidence`
- `runway_next_level_id`, `runway_distance`, `runway_quality`

**`futures.trades.v1` (ES time-and-sales)** *(optional, for ES L2 barrier inference)*
- `ts_event_ns` int64 (UTC)
- `ts_recv_ns` int64 (UTC)
- `symbol` utf8 (e.g., `ES` or full contract `ESH6` — standardize later)
- `price` float64 (or scaled int policy)
- `size` int32
- `aggressor` int8? (BUY=+1, SELL=-1, MID=0) if vendor provides; else derive from MBP best bid/ask
- vendor passthrough: `exchange`/`venue`, `conditions`, `seq` as available

**`futures.mbp10.v1` (ES MBP-10)** *(optional, for ES L2 barrier inference)*
- `ts_event_ns`, `ts_recv_ns`, `symbol`
- 10 bid levels: `bid_px_1..10`, `bid_sz_1..10`
- 10 ask levels: `ask_px_1..10`, `ask_sz_1..10`
- Optional: `is_snapshot` boolean (if vendor differentiates snapshot vs incremental)

#### 2.5 Durability, backpressure, and replay correctness (institutional behaviors)

**Ingestion durability (M4)**:
- Phase 0–1 approach:
  - Use an **in-process bus** (`asyncio.Queue`) plus a **storage writer** that micro-batches to Parquet.
  - Add a lightweight **write-ahead log (WAL)** per stream in Phase 1:
    - append segments (`.arrow` IPC stream or `.jsonl.gz`) so a crash can replay unflushed events
    - flush Parquet on cadence (e.g., 1–5s) and rotate files by size/time

**Production durability (colocation)**:
- Phase 3 approach: replace in-process bus with **NATS JetStream** or **Redpanda/Kafka**:
  - partitions by `symbol` (and optionally by `expiry` for options)
  - consumer groups for engines vs storage vs API
  - retention policies and replay by offset/time

**Idempotency / dedup keys**:
- Prefer vendor-provided unique ids if available; otherwise:
  - build `event_id = hash(source, ts_event_ns, symbol/option_symbol, price, size, exchange, seq?)`
- Silver tier performs deterministic dedup on `(event_id)` within a time window.

**Out-of-order handling**:
- Engines process primarily in event-time with a small lateness buffer (e.g., 250–500ms)
- Storage keeps all raw, and silver compaction can re-order within partitions if needed

#### 2.6 Reuse for ML feature engineering (offline + online)

- Gold `levels.signals` is a **first-class supervised learning dataset**:
  - labels can be derived later (e.g., “break within N seconds” / “reject then move X”)
- Maintain a **feature dictionary** (YAML/markdown) with:
  - feature name, definition, window, units, and leakage notes
- Optional v2:
  - offline store = Parquet/Iceberg
  - online store = Redis/KeyDB (latest features by level_id) for low-latency serving

---

### 3) Definitions (shared language)

#### 3.1 Level \(L\)

A level is a real-valued price with metadata:
- `id`: stable identifier (`"VWAP"`, `"CALL_WALL"`, `"PUT_WALL"`, `"STRIKE_680"`, `"ROUND_680"`, `"HOTZONE_675"`, …)
- `price`: float
- `kind`: enum (VWAP | STRIKE | ROUND | SESSION_HIGH | SESSION_LOW | GAMMA_WALL | USER | …)

#### 3.2 Direction context at a level

Let `spot` be current SPY price.
- If `spot > L`: level is a **SUPPORT test** (approach from above)
  - Break direction: **DOWN**
  - Reject/bounce direction: **UP**
- If `spot < L`: level is a **RESISTANCE test** (approach from below)
  - Break direction: **UP**
  - Reject/bounce direction: **DOWN**

#### 3.3 Active Monitoring Zone

We only compute full signals if:
- `abs(spot - L) <= MONITOR_BAND` (e.g., 0.25–0.50)

---

### 4) Level universe (what levels exist, and where they come from)

We maintain an active set of levels that is updated on a cadence (e.g., 1–5s) or on important changes (e.g., call/put wall changes).

**Level sources (v1)**:
- **VWAP** (session VWAP or anchored to RTH open — decide in config)
- **Round numbers** (e.g., every $1, optionally every $0.50 near spot)
- **Option strikes** (near spot; use dynamic strike manager already present)
- **Flow-derived walls**:
  - Call Wall / Put Wall derived from net gamma flow in `flow_aggregator.py`
- **User-defined hotzones** (manual list in config)
- *(optional later)* session high/low, premarket levels, etc.

**Interfaces to define now**:
- `LevelUniverse.get_levels(market_state) -> list[Level]`
- Each `Level` must have: `id`, `price`, `kind`

---

### 5) Core engines (physics + scoring)

This section specifies deterministic computation modules and their inputs/outputs. Agents should implement against these interfaces.

#### 5.1 Barrier Engine (SPY L1 “book physics” proxy)

**Inputs**: rolling window of SPY quotes + trades, a level \(L\), and direction (SUPPORT/RESISTANCE).

We want to measure whether liquidity at the defending side is:
- **pulled** (canceled) vs
- **consumed** (filled) vs
- **replenished** (added).

Because we only have NBBO (L1), we define the “defending queue” as:
- SUPPORT: when `bid_price` is near \(L\), track `bid_size`
- RESISTANCE: when `ask_price` is near \(L\), track `ask_size`

**Event accounting (heuristic but mechanical)**:

Maintain rolling sums over window \(W_b\) (e.g., 5s or 10s) for the defending side:
- `added_size`
- `removed_size`
  - split into:
    - `filled_size` (removed concurrent with prints at/through the quote)
    - `canceled_size` (removed without such prints)

We compute:
- \( \Delta_{liq} = added\_size - canceled\_size - filled\_size \)
- **Replenishment Ratio**:
  - \( R = \dfrac{added\_size}{canceled\_size + filled\_size + \epsilon} \)

**State classification** (tunable constants; not trained):
- `VACUUM` if \( R < R_{vac} \) **and** \( \Delta_{liq} < -F_{thresh} \)
- `WALL` / `ABSORPTION` if \( R > R_{wall} \) **and** \( \Delta_{liq} > +F_{thresh} \)
- `CONSUMED` if \( \Delta_{liq} \ll 0 \) **and** `filled_size > canceled_size`
- `NEUTRAL` otherwise

**Optional “WEAK” state (GM.md)**:
- Track baseline `defending_size` distribution for last 30–60 minutes and label `WEAK` if current size is below the 20th percentile.

**Outputs** (`BarrierState`):
- `state`: VACUUM | WALL | ABSORPTION | CONSUMED | WEAK | NEUTRAL
- `delta_liq`: \( \Delta_{liq} \)
- `replenishment_ratio`: \(R\)
- `added_size`, `canceled_size`, `filled_size`
- `defending_quote`: {price, size}

##### 5.1.1 Preferred Barrier Physics (L2/MBP-10 + Trades): cancel-vs-fill inference (L3 not required)

**Validation**:
- The **physics framing is correct**: when displayed depth drops, it dropped because it was **filled**, **pulled**, or was **replaced** (absorbed).
- L2+trades can infer “pulled vs filled” **well enough for real-time classification**, but it is **not mathematically identical to L3** because L2 only shows **net displayed depth**.

**Key caveats (why it’s not identical to L3)**:
- **Churn blindness (if you downsample)**: if cancels and adds occur inside a coarse sampling window and net depth ends unchanged, a sampled L2 snapshot can miss that churn. Mitigation: treat MBP-10 as an **event stream** (process every update) and compute **gross** (absolute) depth changes over the window.
- **Iceberg/reserve/hidden** liquidity can make `traded_volume > displayed_depth_lost`.
- **Implied liquidity** (from spreads) can appear/disappear without simple add/cancel semantics.
- **Top-10 coverage**: you can only compute this reliably when the level is within MBP-10 range.
- **Time alignment**: trade and book updates are not perfectly synchronous → use exchange timestamps and a small lateness buffer.
- You must count only **passive fills on the defending side** (aggressor filtering), not all prints at that price.
- **Order modifications** are not explicitly labeled in L2; they appear as depth deltas. For break/bounce physics, treating size reductions as “pulled” is directionally correct, even if it was a modify.

**Inference (per update, or every 100–500ms tick)**:

Let:
- `depth(p, side, t)` = displayed size at absolute price `p` on `side` from MBP-10 state
- `Δdepth = depth(p, side, t1) - depth(p, side, t0)`
- `Vpassive(p, side, (t0,t1])` = executed volume at price `p` that consumed *passive* liquidity on `side`
  - BID side: **SELL-initiated** trades at `p`
  - ASK side: **BUY-initiated** trades at `p`

If `Δdepth < 0` (depth decreased):
- `depth_lost = -Δdepth`
- `inferred_filled = min(depth_lost, Vpassive + ε_fill)`
- `inferred_pulled = max(0, depth_lost - Vpassive - ε_pull)`

If `Δdepth >= 0` and `Vpassive` is meaningfully positive:
- depth was consumed but replenished → **ABSORPTION/WALL**

**Robustness recommendation**:
- Compute over a **zone** around the level (e.g., `L ± N ticks`) by summing `depth` and `Vpassive` across prices in the zone. This is materially more stable than a single-price calculation.
- If `Vpassive >> depth_lost`, interpret that excess as **hidden/iceberg or rapid replenish** and bias classification toward **ABSORPTION**, not “negative pulls.”
- Track **gross flows** in addition to net:
  - `gross_removed = Σ max(0, -Δdepth_update)` across MBP updates in the window
  - `gross_added = Σ max(0, +Δdepth_update)` across MBP updates in the window
  - `churn = gross_added + gross_removed`
  - Use `churn` as an **activity/confidence modifier**: stable depth with high churn usually still means “defended,” but it can also reflect spoof-like behavior (L3 needed for certainty), so we should reduce confidence rather than over-trusting the wall.

#### 5.2 Tape Engine (SPY prints confirm direction)

**Inputs**: SPY trades stream plus last-known quote (for aggressor classification), level \(L\).

**Aggressor** (from CD.md / GM.md):
- If `trade_price >= ask_price`: BUY (lift)
- Else if `trade_price <= bid_price`: SELL (hit)
- Else: MID/UNKNOWN (ignore or downweight)

**Imbalance near level** over window \(W_t\) (e.g., 3–10s) within band \(B_t\) (e.g., 0.05–0.10 around L):
- \( buy = \sum size \;\;(\text{BUY prints}) \)
- \( sell = \sum size \;\;(\text{SELL prints}) \)
- \( I = \dfrac{buy - sell}{buy + sell + \epsilon} \in [-1, +1] \)

**Velocity into level**:
- Compute slope of price over last \(W_v\) seconds (e.g., linear regression on \((t, price)\)):
  - \( v = \text{slope}(\text{price}(t)) \) in \$/sec

**Sweep detection** (simple, mechanical):
- Group trades into clusters with max gap \( \Delta t \le 50–100ms \)
- Trigger if:
  - total notional \( \sum price \cdot size \ge N_{min} \) AND
  - number of distinct venues >= `VENUE_MIN` (if available) AND/OR prints repeatedly cross the spread

**Outputs** (`TapeState`):
- `imbalance`: \(I\), plus raw `buy_vol`, `sell_vol`
- `velocity`: \(v\)
- `sweep`: {detected, notional, direction, venues?, window_ms}

#### 5.3 Fuel Engine (Dealer gamma impulse from SPY 0DTE options)

We estimate whether dealers will **amplify** or **dampen** a move near \(L\).

**Trade-level gamma transfer**:
- For each option trade \(k\):
  - customer buy/sell sign \( s_k \in \{+1, -1\} \)
    - Prefer BBO classification if available (option trade price vs option bid/ask)
    - Fallback: tick rule vs last option trade price
  - customer gamma bought:
    - \( g_k = s_k \cdot size_k \cdot \gamma_k \cdot 100 \)
  - dealer gamma change:
    - \( \Delta G^{dealer}_k = -g_k \)

**Net dealer gamma near level**:
- Select strikes \(K\) within range \( |strike - L| \le \Delta_K \) (e.g., 1–2 dollars)
- Window \(W_g\) (e.g., 30–300s)
- \( G^{dealer}(L) = \sum_{k \in K} \Delta G^{dealer}_k \)

**Gamma effect**:
- If \( G^{dealer}(L) < 0 \): dealers **SHORT gamma** → **AMPLIFY** (trend accelerant)
- Else: **DAMPEN** (mean reversion tendency)

**Critical gamma-derived levels** (v1 pragmatic versions):
- **Call Wall / Put Wall (flow-based)**:
  - `call_wall = argmax_strike(net_gamma_flow_calls)` in last \(W_{wall}\)
  - `put_wall = argmin_strike(net_gamma_flow_puts)` or largest magnitude put-side gamma flow
- **HVL / Gamma flip (approx)**:
  - Compute a cumulative “dealer gamma proxy” across strikes and find the strike where sign changes
  - (We can upgrade later to true OI-based GEX if we ingest open interest reliably)

**Outputs** (`FuelState`):
- `effect`: AMPLIFY | DAMPEN | NEUTRAL
- `net_dealer_gamma`: \(G^{dealer}(L)\)
- `call_wall`, `put_wall`, `hvl` (optional early versions)

#### 5.4 Composite decision logic (Break Score + triggers)

We compute a **Break Score** \(S \in [0, 100]\) for each monitored level \(L\), every tick.

##### 5.4.1 Component scores (GM.md mapping)

- **Liquidity Score \(S_L\)**:
  - VACUUM: 100
  - WEAK: 75
  - NEUTRAL/CONSUMED: 50 (or 60 if strongly negative delta_liq)
  - WALL/ABSORPTION: 0

- **Hedge Score \(S_H\)**:
  - AMPLIFY in the break direction: 100
  - DAMPEN: 0
  - NEUTRAL: 50

- **Confirmation Score \(S_T\)**:
  - If sweep detected in break direction: 100
  - Else: scale 0–50 from velocity magnitude and tape imbalance consistency

##### 5.4.2 Composite

- \( S = w_L S_L + w_H S_H + w_T S_T \)
- Default weights (GM.md): \(w_L=0.45, w_H=0.35, w_T=0.20\)

##### 5.4.3 Signal triggers (state machine)

We produce discrete events:
- **BREAK IMMINENT**:
  - \( S > 80 \) sustained for \(T_{hold}\) (e.g., 3s) while price is within `MONITOR_BAND`
- **REJECT / HOLD**:
  - \( S < 20 \) while price touches the level (within tight `TOUCH_BAND`)
- **CONTESTED**:
  - mid scores with high activity (CONSUMED + strong tape)

#### 5.5 Room to Run (runway)

Given a primary signal (BREAK or REJECT), compute:
- Direction of expected move (UP/DOWN)
- **Next obstacle** = nearest level in that direction from the active level set
- `runway = abs(next_obstacle.price - L)`
- Include `runway_quality`:
  - CLEAR if no strong walls in between
  - OBSTRUCTED otherwise

#### 5.6 Smoothing layer (required for v1 usability)

We will compute both:
- **Raw** metrics (per-window sums, ratios, slopes)
- **Smoothed** metrics:
  - EWMA for continuous series:
    - \( x_{smooth}(t) = \alpha x(t) + (1-\alpha)x_{smooth}(t-\Delta t) \)
    - \( \alpha = 1 - e^{-\Delta t / \tau} \) (configure \(\tau\) by half-life)
  - Optional robust smoother:
    - rolling median or Huberized mean for quote sizes / score spikes

We will smooth:
- `BreakScore`, `delta_liq`, `replenishment_ratio`, `velocity`, and `net_dealer_gamma`

---

### 6) Backend integration plan (fit the existing repo)

#### 6.1 Current state (repo reality)

- Options flow is processed in `backend/src/flow_aggregator.py` and broadcast via WS.
- SPY trades are subscribed for strike updates but currently **not routed** into processing.

#### 6.2 New backend modules (files to create)

- `backend/src/event_types.py`
  - normalized dataclasses: `StockTrade`, `StockQuote`, `OptionTrade`
- `backend/src/market_state.py`
  - `MarketState` store (last quote, last trade, ring buffers, per-strike option flow)
- `backend/src/level_universe.py`
  - produces critical levels (VWAP, round, strikes, walls, user hotzone)
- `backend/src/barrier_engine.py`
- `backend/src/tape_engine.py`
- `backend/src/fuel_engine.py`
- `backend/src/score_engine.py`
  - composite score + trigger state machine
- `backend/src/smoothing.py`
- `backend/src/room_to_run.py`
- `backend/src/level_signal_service.py`
  - orchestrates: `MarketState` → `LevelSignals` payload

#### 6.3 Streaming changes (minimal coupling)

- Extend `backend/src/stream_ingestor.py` to subscribe to:
  - `Q.SPY` (quotes)
  - continue `T.SPY` (trades)
- Route SPY trade+quote events into the same `msg_queue` as options trades, but with type tags so downstream can dispatch safely.

#### 6.4 Broadcast payload (new WS channel or merged)

Option A (fastest): **merge into current WS payload** as:
- `{ "flow": <existing flow snapshot>, "levels": <level signals> }`

Option B (cleaner): add a second endpoint:
- `/ws/levels` for level physics payload
- keep `/ws/stream` as flow snapshot

**Level physics payload (v1)**:

```json
{
  "ts": 1715629300123,
  "spy": {
    "spot": 545.42,
    "bid": 545.41,
    "ask": 545.43
  },
  "levels": [
    {
      "id": "STRIKE_545",
      "price": 545.0,
      "kind": "STRIKE",
      "direction": "SUPPORT",
      "distance": 0.42,
      "break_score_raw": 88,
      "break_score_smooth": 81,
      "signal": "BREAK",
      "confidence": "HIGH",
      "barrier": {
        "state": "VACUUM",
        "delta_liq": -8200,
        "replenishment_ratio": 0.15,
        "added": 3100,
        "canceled": 9800,
        "filled": 1500
      },
      "tape": {
        "imbalance": -0.45,
        "buy_vol": 120000,
        "sell_vol": 320000,
        "velocity": -0.08,
        "sweep": { "detected": true, "direction": "DOWN", "notional": 1250000 }
      },
      "fuel": {
        "effect": "AMPLIFY",
        "net_dealer_gamma": -185000,
        "call_wall": 548,
        "put_wall": 542,
        "hvl": 545
      },
      "runway": {
        "direction": "DOWN",
        "next_obstacle": { "id": "PUT_WALL", "price": 542 },
        "distance": 3.0,
        "quality": "CLEAR"
      },
      "note": "Vacuum + dealers chase; sweep confirms"
    }
  ]
}
```

---

### 7) Frontend plan (Angular)

We will add a small, focused UI slice:
- **Level Table**: sorted by |distance| with columns: level, score (raw/smoothed), barrier state, gamma effect, tape velocity, runway.
- **Level Strip** overlay on existing visuals: show nearest 3–5 levels and highlight “break/reject” triggers.

Implementation options:
- Extend `DataStreamService` to also hold `levels` payload (if merged).
- Or create `LevelStreamService` with separate WebSocket.

---

### 8) Testing & simulation plan

#### 8.1 Unit tests (backend)

Create synthetic sequences to force each classification:
- Barrier: vacuum vs absorption vs consumed (quote size drops without trades vs with trades vs replenishment).
- Tape: imbalance sign + sweep detection.
- Fuel: dealer gamma sign flipping and wall identification.
- Score: trigger timers (score sustained over N ticks).

#### 8.2 Replay mode

Extend replay to include:
- SPY trades + quotes alongside options trades, so level scoring can be tested deterministically.
- **Replay source of truth** becomes Bronze/Silver Parquet (not ad-hoc logs):
  - load `bronze/stocks/*` + `bronze/options/*`
  - optionally replay from `silver/*` when you want deduped/cleaned streams

---

### 9) Configuration (single source of truth)

Create `backend/src/config.py` (or similar) containing:
- window sizes: `W_b`, `W_t`, `W_g`, `W_v`
- bands: `MONITOR_BAND`, `TOUCH_BAND`
- thresholds: `R_vac`, `R_wall`, `F_thresh`, sweep thresholds
- weights: `w_L`, `w_H`, `w_T`
- smoothing: `tau_score`, `tau_velocity`, …

---

### 10) Acceptance criteria (v1)

- For any selected level \(L\), system emits:
  - barrier state + tape metrics + fuel effect + composite score (raw + smoothed)
  - discrete signal: BREAK / REJECT / CONTESTED / NEUTRAL
  - runway to next obstacle
- UI shows nearest levels with stable scores (not flickering) thanks to smoothing.
- Unit tests cover classification edges and trigger timing.
- **Data backbone**:
  - Bronze Parquet partitions exist for SPY trades/quotes and SPY options trades (+ greeks snapshots)
  - *(optional)* Bronze Parquet partitions exist for ES trades + ES MBP-10 when ES ingestion is enabled
  - Gold Parquet partitions exist for `levels.signals`
  - Replay produces byte-for-byte identical results given the same inputs + config (within floating/rounding policy)

---

### 11) Web-dependent vendor contracts (Dec 2025) — so agents don’t need to search

#### 11.1 Massive (Polygon legacy) — WebSocket (real-time)

- **Docs**: `https://massive.com/docs/websocket/stocks/trades`, `https://massive.com/docs/websocket/stocks/quotes`, `https://massive.com/docs/websocket/options/trades`, `https://massive.com/docs/websocket/options/quotes`
- **Important**: stocks quote sizes are reported in **shares (not round lots)** as of **2025-11-03** (SEC MDI) — `https://massive.com/blog/change-stocks-quotes-round-lots-to-shares/`
- **Endpoints**:
  - Stocks: `wss://socket.massive.com/stocks` (delayed: `wss://delayed.massive.com/stocks`)
  - Options: `wss://socket.massive.com/options` (delayed: `wss://delayed.massive.com/options`)
- **Subscribe params** (comma-separated lists supported):
  - Stocks trades: `T.SPY` (wildcard: `T.*`)
  - Stocks quotes: `Q.SPY` (wildcard: `Q.*`)
  - Options trades: `T.O:SPYYYMMDDC########` / `T.O:SPYYYMMDDP########`
  - Options quotes: `Q.O:SPYYYMMDDC########` / `Q.O:SPYYYMMDDP########`
- **Auth/status**: client examples wait for `ev='status'` with `status='auth_success'` before subscribing (client libs handle auth/handshake).
- **Wire schema (timestamps are Unix *milliseconds*)**:
  - **Stocks trade (`ev=T`)**: `ev`, `sym`, `x`, `i`, `z`, `p`, `s`, `c`, `t`, `q`, `trfi`, `trft`
    - `t`: SIP timestamp (Unix ms)
    - `trft`: TRF timestamp (Unix ms)
  - **Stocks quote (`ev=Q`)**: `ev`, `sym`, `bx`, `bp`, `bs`, `ax`, `ap`, `as`, `c`, `i`, `t`, `q`, `z`
    - `bs/as`: bid/ask size (shares)
  - **Options trade (`ev=T`)**: `ev`, `sym`, `x`, `p`, `s`, `c`, `t`, `q`
  - **Options quote (`ev=Q`)**: `ev`, `sym`, `bx`, `ax`, `bp`, `ap`, `bs`, `as`, `t`, `q`
- **Normalization rules**:
  - `ts_event_ns = t * 1_000_000`
  - `ts_recv_ns = time.time_ns()` at ingest
- **Option ticker parsing** (`sym` / contract tickers):
  - Format: `O:<UNDERLYING><YYMMDD><C|P><STRIKE8>`
  - Example: `O:SPY251216C00676000` → underlying `SPY`, exp `2025-12-16`, right `C`, strike `676.000` (decode strike as `int(STRIKE8)/1000`)

#### 11.2 Massive — REST options snapshots/greeks (base `https://api.massive.com`)

- **Docs**: `https://massive.com/docs/rest/options/snapshots/option-chain-snapshot`, `https://massive.com/docs/rest/options/snapshots/unified-snapshot`
- **Option chain snapshot (recommended for greeks cache)**:
  - `GET /v3/snapshot/options/{underlyingAsset}`
  - Query params:
    - `contract_type` = `call|put`
    - `expiration_date` (or range: `expiration_date.gt|gte|lt|lte`)
    - `strike_price` (or range: `strike_price.gt|gte|lt|lte`)
    - `sort` = `ticker|expiration_date|strike_price`, `order` = `asc|desc`, `limit` (max 250)
  - Response: `status`, `request_id`, optional `next_url`, and `results[]` with (plan-dependent availability):
    - `details.{ticker,expiration_date,strike_price,contract_type,shares_per_contract,exercise_style}`
    - `greeks.{delta,gamma,theta,vega}` *(may be absent for some contracts)*
    - `implied_volatility`, `open_interest`
    - `last_quote.{bid,ask,bid_size,ask_size,last_updated(ns),midpoint,timeframe}` *(only if plan includes quotes)*
    - `last_trade.{price,size,exchange,conditions,sip_timestamp(ns),timeframe}` *(only if plan includes trades)*
    - `fmv`, `fmv_last_updated(ns)` *(business plans)*, `underlying_asset.{ticker,price,last_updated(ns),...}`
  - Pagination: if `next_url` present, follow it verbatim.
- **Universal snapshot (for a specific contract list)**:
  - `GET /v3/snapshot`
  - Query params: `type=options` and either `ticker.any_of` (comma-separated, <=250) or `ticker.gte/gt/lte/lt` ranges; plus `order/limit/sort`

#### 11.3 Databento — ES futures L2 + trades (GLBX.MDP3)

- **Dataset**: `GLBX.MDP3` (CME Globex MDP 3.0)
- **Docs**: `https://databento.com/datasets/GLBX.MDP3`, `https://raw.githubusercontent.com/databento/databento-python/main/examples/live_smoke_test.py`, `https://raw.githubusercontent.com/databento/dbn/main/python/python/databento_dbn/_lib.pyi`
- **Approach**:
  - **v1 (now)**:  **Databento Historical DBN** files are in the dbn-data/ directory (and ingest/replay from DBN into our Bronze/engine pipeline).
  - **v2 (later)**: switch to **Databento Live** streaming subscriptions; keep the same record mapping + normalization.
- **Schemas** (from Databento dataset exports):
  - `trades`: tick-by-tick trades
  - `mbp-10`: **Level 2 (market-by-price)** top-10 depth updates (“Every trade and book update at the top 10 price levels.”)
  - **No MBO**: we are **not** ingesting `mbo` (Level 3 / market-by-order) data in v1.
- **Continuous symbology**:
  - Use `symbols='ES.c.0'` with `stype_in='continuous'` (front contract, rolls per Databento rules)
- **Live subscribe API (databento-python) — v2 later**:
  - `from databento import Live`
  - `client.subscribe(dataset='GLBX.MDP3', schema='mbp-10', symbols='ES.c.0', stype_in='continuous', start=None, snapshot=False)`
  - iterate: `for record in client: ...`
- **Record field mapping (databento_dbn, prices are fixed-point; use `pretty_*` helpers)**:
  - Timestamps are **nanoseconds**:
    - `record.ts_event` (ns), `record.ts_recv` (ns), optional `record.ts_out` (ns, live gateway send)
  - **Trades schema** (`TradeMsg`):
    - `record.pretty_price` (float), `record.size` (int)
    - `record.side` is aggressor: `A` (ask=sell aggressor), `B` (bid=buy aggressor), `N` (none)
  - **MBP-10 schema** (`MBP10Msg`):
    - `record.levels: list[BidAskPair]` (10 levels)
    - each level has: `pretty_bid_px`, `bid_sz`, `pretty_ask_px`, `ask_sz` (+ `bid_ct/ask_ct` counts)

---

### 12) Parallel agent assignments (clear tasks + ownership)

**Shared rule for all agents**
- **Do not** invent your own event/level/signal shapes. Implement against §2 (envelope + schemas) and §6.4 (WS payload).
- When you add a new file, keep it in `backend/src/` and add minimal docstrings.
- If you must change an interface, update this `PLAN.md` section and inform other agents.

#### Agent A — Shared contracts + config (foundation) ✅ COMPLETE

**Goal**: Define stable types/config that all other agents can depend on.

- **Status**: ✅ **COMPLETE** (commits: 9b6e38a)
- **Deliverables**
  - ✅ Create `backend/src/event_types.py` with `StockTrade`, `StockQuote`, `OptionTrade` (normalized; includes `ts_event_ns`, `ts_recv_ns`, symbol fields).
  - ✅ Create `backend/src/config.py` per §9.
  - ⏸️ (Optional but recommended) Create `backend/src/schemas/` with Arrow/Pydantic schema representations matching §2.4.
- **Interfaces to expose**
  - `Config` object (or module-level constants) with the keys listed in §9.
  - Dataclasses for normalized events.
- **Dependencies**
  - None (everyone else depends on this).
- **Files owned**
  - `backend/src/event_types.py`, `backend/src/config.py`, optional `backend/src/schemas/*`

#### Agent B — Ingestion + routing (SPY quotes/trades into the bus)

**Goal**: Ensure SPY `T.SPY` and `Q.SPY` are subscribed and routed into the internal queue with normalized event types.

- **Deliverables**
  - Modify `backend/src/stream_ingestor.py` to subscribe to `Q.SPY` and keep `T.SPY`.
  - Route incoming events into `msg_queue` with explicit type tags or dataclass instances.
- **Interfaces to consume**
  - `StockTrade`, `StockQuote` from Agent A.
- **Dependencies**
  - Agent A types.
- **Files owned**
  - `backend/src/stream_ingestor.py` (and any narrow helper you create under `backend/src/`)

#### Agent C — `MarketState` + ring buffers (state backbone) ✅ COMPLETE

**Goal**: Implement a single in-memory state store that engines can query on snap ticks.

- **Status**: ✅ **COMPLETE** (commits: 9b6e38a, 43c2428)
- **Deliverables**
  - ✅ Create `backend/src/market_state.py` with:
    - ✅ last-known SPY quote/trade
    - ✅ rolling buffers (timestamped) for quotes and trades
    - ✅ per-strike option flow aggregates (compatible with existing flow aggregator output)
  - ✅ Provide methods like:
    - ✅ `update_stock_trade(trade)`
    - ✅ `update_stock_quote(quote)`
    - ✅ `update_option_trade(opt_trade)` (or consume existing flow aggregator output)
    - ✅ window queries by `(ts_now, window_seconds, price_band, level_price)`
  - ✅ Verification: `backend/test_market_state.py` (all tests passing)
- **Interfaces to consume**
  - Event dataclasses (Agent A)
  - Option flow shapes from current `flow_aggregator.py` (coordinate with Agent E).
- **Dependencies**
  - Agent A ✅
  - Light coordination with Agent E (fuel inputs).
- **Files owned**
  - `backend/src/market_state.py`
  - `backend/test_market_state.py`

#### Agent D — Barrier + Tape engines (SPY L1 physics + tape momentum)

**Goal**: Implement deterministic barrier/tape states per §5.1 and §5.2.

- **Deliverables**
  - Create `backend/src/barrier_engine.py` implementing barrier accounting over rolling windows.
  - Create `backend/src/tape_engine.py` implementing aggressor classification, imbalance, velocity, and sweep detection.
- **Interfaces to consume**
  - `MarketState` queries (Agent C)
  - Config thresholds/windows (Agent A)
- **Dependencies**
  - Agent A + Agent C
- **Files owned**
  - `backend/src/barrier_engine.py`, `backend/src/tape_engine.py`

#### Agent E — Fuel engine (options gamma transfer + wall inference)

**Goal**: Implement dealer gamma proxy per §5.3 using options trades + greeks snapshots (existing greeks cache).

- **Deliverables**
  - Create `backend/src/fuel_engine.py` implementing:
    - trade-level gamma transfer using option trade aggressor sign
    - rolling net dealer gamma near level
    - flow-based call/put wall + optional HVL proxy
  - Define how to get `gamma_k` for each option trade:
    - use existing greeks cache/enricher mechanisms (coordinate with current `greek_enricher.py`)
- **Interfaces to consume**
  - Option trade normalization (Agent A)
  - Greeks snapshot access (existing repo + coordinate with Agent C state store)
  - Config windows/ranges (Agent A)
- **Dependencies**
  - Agent A + Agent C
- **Files owned**
  - `backend/src/fuel_engine.py` (and any small helper file)

#### Agent F — Level universe + room-to-run (levels + runway logic)

**Goal**: Provide a consistent set of levels and compute runway per §5.5.

- **Deliverables**
  - Create `backend/src/level_universe.py` that returns levels based on:
    - spot price (rounds, strikes)
    - flow-derived walls (call/put walls)
    - VWAP (if available) and user hotzones
  - Create `backend/src/room_to_run.py` that:
    - given `levels[]`, active level, and direction, finds nearest obstacle and computes runway quality
- **Interfaces to consume**
  - `MarketState` spot + flow-derived walls (Agent C)
  - Config settings (Agent A)
- **Dependencies**
  - Agent A + Agent C + Agent E (for walls)
- **Files owned**
  - `backend/src/level_universe.py`, `backend/src/room_to_run.py`

#### Agent G — Scoring + smoothing + WS payload orchestrator

**Goal**: Combine Barrier/Tape/Fuel into score + triggers, smooth it, and publish `levels` payload (merged or separate WS).

- **Deliverables**
  - Create `backend/src/score_engine.py` (component scores + composite + trigger timers).
  - Create `backend/src/smoothing.py` (EWMA + optional robust smoothing).
  - Create `backend/src/level_signal_service.py` to orchestrate:
    - build levels from Agent F
    - compute states from Agents D/E
    - compute score + smoothing + runway
    - emit payload shaped like §6.4
  - Integrate into broadcaster path (`socket_broadcaster.py` or equivalent) with chosen WS option (A or B).
- **Interfaces to consume**
  - Engines outputs (Agents D/E)
  - Level universe + runway (Agent F)
  - Config (Agent A)
- **Dependencies**
  - Agent A + C + D + E + F
- **Files owned**
  - `backend/src/score_engine.py`, `backend/src/smoothing.py`, `backend/src/level_signal_service.py`, and minimal edits to WS publisher code

#### Agent H — Frontend (Angular UI for levels)

**Goal**: Display level signals: table + strip overlay per §7.

- **Deliverables**
  - Add service updates:
    - if WS merged: extend `frontend/src/app/data-stream.service.ts`
    - if separate WS: create `frontend/src/app/level-stream.service.ts`
  - Add components:
    - Level Table view (near existing dashboard)
    - Optional “Level Strip” overlay for nearest levels
- **Interfaces to consume**
  - WS payload shape in §6.4
- **Dependencies**
  - Agent G finalizes whether WS is merged or separate (Option A vs B).
- **Files owned**
  - `frontend/src/app/*` new/modified files for levels UI

#### Agent I — Storage writer + replay correctness (Bronze/Gold integration)

**Goal**: Ensure ingestion writes canonical Parquet partitions and replay can reproduce level signals deterministically.

- **Deliverables**
  - Add/extend writer to emit Bronze:
    - `stocks.trades.v1`, `stocks.quotes.v1`, `options.trades.v1`, `options.greeks_snapshots.v1`
  - Add Gold writer for `levels.signals.v1` (snap tick output).
  - Extend replay engine to play back SPY quotes/trades + options trades in deterministic event-time order.
- **Interfaces to consume**
  - Schema fields in §2.4
  - Event envelope in §2.1
  - Level signals payload (Agent G)
- **Dependencies**
  - Agent A for schemas/types
  - Agent G for signal outputs
- **Files owned**
  - Likely `backend/src/persistence_engine.py`, `backend/src/replay_engine.py`, plus any new writer helpers


