# Architecture Specification: 0DTE Dynamic Flow Monitor (v1.0)

**Target Infrastructure:** Local Execution (Apple M4 Silicon / 128GB RAM)
**Data Provider:** Polygon.io (Advanced Tier - WebSocket enabled)
**Storage Format:** Parquet (via DuckDB)
**Backend Runtime:** Python 3.12+ (AsyncIO)
**Frontend Framework:** Angular 17+ (Signals Architecture)

IMPORTANT: RUN ALL CODE IN THE ORDER OF THE DOCUMENT. DO NOT SKIP ANY STEPS.
IMPORTANT: Use pyproject.toml and uv to manage dependencies. 

## 1. System Overview
This system is a low-latency market data engine that maintains a **Dynamic Sliding Window** of option contracts. It monitors real-time trade flow for SPY 0DTE options, specifically targeting the At-The-Money (ATM) strike $\pm$ 3 strikes.

## 2. Backend Engineering (Python)

### 2.1 Core Dependencies
* **Runtime:** `uv` for dependency management (leveraging M4 performance).
* **Web Framework:** `fastapi` (API), `uvicorn` (ASGI Server).
* **Data Client:** `polygon-api-client` (latest stable).
* **Async Logic:** `asyncio` (native event loop).
* **Data Structures:** `pydantic` (schema validation), `collections.deque` (rolling windows).

### 2.2 Module: `StrikeManager` (The Brain)
**Responsibility:** Determine which contracts are "Active" based on the underlying price.
1.  **Input:** Real-time SPY price (via Polygon `A` (Aggregate) or `T` (Trade) channel for `Stocks`).
2.  **Logic (Execute every 60 seconds OR on >$0.50 price deviation):**
    * Calculate `center_strike` = `round(current_spy_price)`.
    * Generate `target_strikes`: List of integers `[center_strike - 3` to `center_strike + 3]`.
    * Format Tickers: Construct Polygon-compliant tickers:
        * Format: `O:SPY{YYMMDD}{C/P}{STRIKE_8_DIGIT}`
        * *Example:* `O:SPY251216C00572000`
    * **Diff Logic:** Compare `current_subscriptions` set vs `target_strikes` set.
    * **Output:** Return `subscribe_list` and `unsubscribe_list`.

### 2.3 Module: `StreamIngestor` (The IO Layer)
**Responsibility:** Manage the WebSocket connection to Polygon without blocking.
1.  **Connection:** Initialize `polygon.WebSocketClient` with `market='options'` and `all_trade_updates=False`.
2.  **Dynamic Subscription:**
    * Expose a method `update_subs(add: list, remove: list)`.
    * Invoke `ws.subscribe(add)` and `ws.unsubscribe(remove)` dynamically at runtime.
3.  **Message Handling:**
    * Listen for `T` (Trade) events.
    * **Critical:** Offload processing immediately to an `asyncio.Queue` to prevent blocking the WebSocket keep-alive heartbeat. Do not process logic inside the `on_message` callback.

### 2.4 Module: `FlowAggregator` (The State Machine)
**Responsibility:** Process raw trade messages into consumable metrics.
1.  **Worker:** Consume from `asyncio.Queue`.
2.  **State Store:** In-memory Dictionary `Dict[Ticker, ContractMetrics]`.
    * `ContractMetrics` Schema:
        * `cumulative_volume`: Int
        * `cumulative_premium`: Float (Price * Size * 100)
        * `last_price`: Float
        * `net_aggressor`: Int (Buy side vs Sell side heuristic - *Note: Polygon trade ticks verify condition codes, exclude delayed/out-of-sequence trades*).
3.  **Greeks Mapping (Optimization):**
    * Since the Stream does not provide Delta/Gamma, run a parallel background task (interval: 60s) using `RESTClient.get_snapshot_option_chain`.
    * Map `delta` to the `ContractMetrics` store.
    * Compute `Net Delta Flow` = `Trade Volume * Contract Delta`.

### 2.5 Module: `SocketBroadcaster` (The Output)
**Responsibility:** Push updates to the Frontend.
1.  **Endpoint:** `WebSocket /ws/stream`.
2.  **Payload:** JSON object containing the `State Store` snapshot.
3.  **Frequency:** Throttle broadcasts to 250ms or 500ms (debounce) to avoid frontend rendering exhaustion.

### 2.6 Module: `GreekEnricher`
**Responsibility:** Attach Greeks to incoming trade flow *before* aggregation/storage.
1.  **Snapshot Loop:** Every 60 seconds (or upon strike list change), fetch the full Option Chain Snapshot from Polygon API.
2.  **Cache:** Store `Delta` and `Gamma` in a local Look-Up Table (Hash Map) keyed by `Ticker`.
    * *Note:* Since we are 0DTE, Greeks drift fast. 60s is the maximum staleness tolerance. Ideally 30s.
3.  **Enrichment:** When `FlowAggregator` processes a trade:
    * Lookup `Ticker` in Cache.
    * Append `Delta` and `Gamma` to the trade record.
    * Calculate `DeltaNotional` = `TradeSize * Delta * 100`.

### 2.7 Module: `PersistenceEngine` (New)
**Responsibility:** High-throughput, asynchronous logging of data for future ML training.
1.  **Tech Stack:** `DuckDB` (Python API).
2.  **Buffer Strategy:** Do NOT write to disk on every tick.
    * Maintain an in-memory `pandas.DataFrame` or `List[Dict]` buffer.
    * **Flush Triggers:** `Buffer Size > 5,000 rows` OR `Time > 60 seconds`.
3.  **Schema (The Training Set):**
    * `timestamp` (UTC High Precision)
    * `underlying_price` (At moment of trade)
    * `ticker` (Option Symbol)
    * `strike`
    * `type` (C/P)
    * `price` (Premium)
    * `size`
    * `aggressor_side` (Buy/Sell/Unknown)
    * `delta` (Snapshot value)
    * `gamma` (Snapshot value)
    * `net_delta_impact` (Calculated)
4.  **Storage Layout (Data Lake):**
    * Write strictly to **Parquet** files.
    * Partition Strategy: `data/raw/flow/year=2025/month=12/day=16/flow_part_001.parquet`
    * *Why:* Parquet provides heavy compression and is natively readable by PyTorch/TensorFlow for future "Prediction Models."

### 2.8 Module: `AnalyticsService` (Future Proofing)
**Responsibility:** Allow the frontend to request historical stats from the stored data.
1.  **Query Engine:** Use `duckdb.query("SELECT sum(net_delta_impact) FROM 'data/raw/flow/*/*.parquet' WHERE ...")`.
2.  **Performance:** DuckDB on M4 Silicon can scan millions of rows in milliseconds without needing a running server process.


---

## 3. Frontend Engineering (Angular)

### 3.1 Setup & Configuration
* **Version:** Angular 17+ (Strict Mode enabled).
* **State Management:** Use **Angular Signals** (`signal()`, `computed()`) exclusively. Do not use `NgRx` or `Observables` for local component state.
* **Styling:** TailwindCSS (for rapid layout).

### 3.2 Service: `DataStreamService`
1.  **Connection:** Establish `WebSocket` connection to `ws://localhost:8000/ws/stream`.
2.  **Resilience:** Implement auto-reconnect logic (exponential backoff) if the backend Python server restarts.
3.  **Signal:** Expose a `readonly flowData = signal<FlowMap>({})`.

### 3.3 Component: `StrikeGridComponent`
**Responsibility:** Render the sliding window.
1.  **Layout:** Vertical list sorted by Strike Price (High to Low).
2.  **Rows:**
    * Center Row: Current ATM Strike (Highlight Color).
    * Above: Call ITM / Put OTM.
    * Below: Call OTM / Put ITM.
3.  **Columns:**
    * `Strike`
    * `Call Vol` | `Call Premium` | `Call Net Delta`
    * `Put Vol` | `Put Premium` | `Put Net Delta`
4.  **Reactivity:** The component template must bind directly to the `flowData` signal. Use `changeDetection: OnPush`.

---

## 4. Implementation Steps (Execution Order)

1.  **Setup Backend venv:** Initialize Python environment, install `fastapi`, `polygon-api-client`, `uvicorn`.
2.  **Harness Data:** Create `test_stream.py`. hardcode 1 active ticker. Verify WebSocket connection and `T` message parsing.
3.  **Implement `StrikeManager`:** (Same as v1.0).
4.  **Implement `GreekEnricher`:** Create the background loop that fetches snapshots and builds the `Dict[Ticker, Greeks]` cache.
5.  **Implement `PersistenceEngine`:** Build the class that accepts a `TradeDict`, appends it to a list, and flushes to `.parquet` when full.
6.  **Build `FlowAggregator`:** Create the in-memory store and the loop to update volume/premium.
7.  **Build API:** FastAPI + WebSocket.
6.  **Setup Frontend:** Initialize Angular CLI project.
8.  **Connect:** Build the Angular Service to consume the JSON stream and log to console.
9.  **Render:** Build the Grid Table to visualize the data.

## 5. Technical Constraints & Guardrails
* **Rate Limits:** Respect Polygons API limits on the REST Snapshot calls (used for Greeks). The WebSocket stream has no strict limit but monitor bandwidth.
* **Memory Safety:** The `State Store` should be cleaned (pruned) daily or upon session reset.
* **Error Handling:** If the WebSocket disconnects, the system must attempt reconnect immediately without crashing the main process.
* **Precision:** Use `decimal` or high-precision float handling for Premium calculations to avoid floating-point drift.
* **Parquet File Size:** Aim for file chunks ~100MB. If they are too small (1KB), DuckDB read performance drops. Adjust "Flush Triggers" accordingly.
* **Concurrency:** DuckDB allows **one** writer and **multiple** readers. Ensure the `PersistenceEngine` is the *only* module with write access to the specific daily folder.
* **Disk I/O:** M4 SSD is fast, but async writing (`aiofiles` or running DuckDB insert in a `ThreadPoolExecutor`) is required to ensure the main WebSocket loop never blocks while saving data.

---

Here is the explanation of **Why** we need Greeks, followed by the **Updated Architecture Specification (v1.1)** which includes a high-performance Data Lake strategy for your M4 Silicon to handle storage and future ML training.

### Part 1: Why We Need Greeks (Specifically Delta & Gamma)

In 0DTE trading, "Volume" is a liar. The Greeks are the "Truth Serum."

**1. Delta ($\Delta$): The "True" Market Pressure**
* **The Problem:** Buying 1,000 contracts of a Cheap OTM Call (Delta 0.05) costs very little and forces the Market Maker to buy only 5,000 shares of SPY to hedge. Buying 1,000 contracts of an ITM Call (Delta 0.80) forces the Market Maker to buy 80,000 shares.
* **The Solution (Net Delta):** If you only track volume, both look like "1,000 Volume." If you track **Net Delta** (`Volume * Delta`), you see the second order is **16x** more powerful.
* **For Your Algo:** You want to filter for **High Delta Flow**. This indicates "Smart Money" or aggressive positioning, rather than just retail gambling on lottos.

**2. Gamma ($\Gamma$): The "Acceleration" Zones**
* **The Problem:** As price moves, Delta changes. Gamma tells you *how fast* Delta changes.
* **The Solution:** High Gamma strikes act as "magnets" or "repellents." If you know the "Gamma Exposure" (GEX) of the strikes you are watching, you can predict where price might get "stuck" or where it might explode (Gamma Squeeze).

---