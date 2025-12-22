# NEXT.md — Phase 2 Transition Plan (NATS + Services)

> **Audience**: AI Coding Agents (Parallel Execution)
> **Goal**: Transition from `asyncio.Queue` monolith to NATS-based microservices.
> **Status**: Refactoring complete. Directories established. `docker-compose.yml` ready. `bus.py` created.

---

## Architecture: The "Spymaster" Distributed System

We are moving to an event-driven architecture using **NATS JetStream** as the backbone.

### Data Flow
1.  **Ingestor** → `market.<type>.<symbol>` (NATS)
2.  **Core** ← `market.*` (NATS) → `levels.signals` (NATS)
3.  **Lake** ← `market.*` + `levels.signals` (NATS) → MinIO/S3 (Parquet)
4.  **Gateway** ← `levels.signals` (NATS) → WebSocket (Frontend)

### NATS Subjects (Contracts)
| Subject | Schema | Publisher | Consumer |
| :--- | :--- | :--- | :--- |
| `market.futures.trades` | `FuturesTrade` | Ingestor | Core, Lake |
| `market.futures.mbp10` | `MBP10` | Ingestor | Core, Lake |
| `market.options.trades` | `OptionTrade` | Ingestor | Core, Lake |
| `levels.signals` | `LevelSignals` | Core | Gateway, Lake |

---

## Agent Assignments

**CRITICAL**: Agents can work in parallel. Do not modify shared code (`src/common`) without coordination. Use `src/common/bus.py` for all NATS interaction.

### AGENT A: Ingestor Service (The Source) ✅ COMPLETE
**Goal**: Create a standalone process that reads feed data and publishes to NATS.

**Status**: ✅ **COMPLETE**

1.  **Modify `src/ingestor/stream_ingestor.py`**: ✅
    *   Remove `asyncio.Queue` dependency.
    *   Inject `NATSBus`.
    *   Publish normalized events to NATS subjects (e.g., `market.options.trades`).
2.  **Create `src/ingestor/main.py`**: ✅
    *   Initialize `NATSBus`.
    *   Initialize `StreamIngestor`.
    *   Run the event loop.
3.  **Update `DBNIngestor` (Replay)**: ✅
    *   Create a `ReplayPublisher` that reads DBN files and publishes to NATS at `REPLAY_SPEED`.
    *   This allows "Replay Mode" to just be a NATS publisher, so other services don't know the difference.

**Deliverables**:
- `src/ingestor/stream_ingestor.py` - Updated to publish to NATS
- `src/ingestor/main.py` - Service entry point
- `src/ingestor/replay_publisher.py` - DBN replay with configurable speed
- `src/common/config.py` - Added NATS_URL, S3 settings, REPLAY_SPEED

### AGENT B: Lake Service (The Memory) ✅ COMPLETE
**Goal**: Create a standalone process that archives everything from NATS to MinIO/S3.

1.  ✅ **Updated `BronzeWriter` (`src/lake/bronze_writer.py`)**:
    *   Removed `wal_manager` (NATS JetStream is the WAL).
    *   Subscribes to `market.*` via NATS.
    *   Writes Parquet to S3/MinIO or local filesystem (configurable).
2.  ✅ **Updated `GoldWriter` (`src/lake/gold_writer.py`)**:
    *   Subscribes to `levels.signals` via NATS.
    *   Writes Parquet to S3/MinIO or local filesystem.
3.  ✅ **Created `src/lake/main.py`**:
    *   Entry point with `LakeService` class.
    *   Graceful shutdown handling.

### AGENT C: Core Service (The Brain)
**Goal**: The physics engine. Consumes raw data, calculates state, emits signals.

1.  **Create `src/core/service.py`**:
    *   Wraps `MarketState`, `BarrierEngine`, `ScoreEngine`, etc.
    *   Subscribes to `market.*`.
    *   Updates `MarketState` on every message.
2.  **Implement Snap Loop**:
    *   Run a periodic task (every 100-250ms).
    *   Run logic: `LevelSignalService.compute_level_signals()`.
    *   Publish result to `levels.signals` on NATS.
3.  **Create `src/core/main.py`**:
    *   Initialize and run the service.

### AGENT D: Gateway Service (The Interface) ✅ COMPLETE
**Goal**: Serve the frontend via WebSockets.

**Status**: ✅ **COMPLETE**

1.  ✅ **Updated `SocketBroadcaster` (`src/gateway/socket_broadcaster.py`)**:
    *   Removed internal state computation - now pure NATS relay.
    *   Subscribes to `levels.signals` on NATS.
    *   Caches latest payload for new connections.
    *   Handles multiple concurrent WebSocket clients.
2.  ✅ **Created `src/gateway/main.py`**:
    *   FastAPI app with lifespan management.
    *   WebSocket endpoint `/ws/stream`.
    *   Health check endpoint `/health`.
    *   New clients receive cached state on connect.
    *   Configurable port via `GATEWAY_PORT` env var.

**Deliverables**:
- `src/gateway/socket_broadcaster.py` - NATS-based WebSocket relay
- `src/gateway/main.py` - Service entry point
- `tests/test_gateway_integration.py` - 6 integration tests (all passing)
- `docker-compose.yml` - Infrastructure services (NATS, MinIO)

### AGENT E: Infrastructure & Orchestration
**Goal**: Tie it all together with Docker.

1.  **Dockerize**:
    *   Create `Dockerfile` in `backend/`.
    *   Update `docker-compose.yml` to add services (`ingestor`, `core`, `lake`, `gateway`) building from that Dockerfile.
    *   Command overrides for each: `uv run python -m src.ingestor.main`, etc.
2.  **Validation**:
    *   Spin up the stack.
    *   Run a replay.
    *   Verify frontend receives data.

---

## Shared Configuration (`src/common/config.py`) ✅ COMPLETE
Ensure these env vars are supported:
- `NATS_URL`: default `nats://localhost:4222` ✅
- `S3_ENDPOINT`: default `http://localhost:9000` ✅
- `S3_BUCKET`: default `spymaster-lake` ✅
- `S3_ACCESS_KEY`: `minioadmin` ✅
- `S3_SECRET_KEY`: `minioadmin` ✅
- `REPLAY_SPEED`: default `1.0` ✅

## Development Workflow
To run a service locally without Docker (for debugging):
```bash
# Terminal 1: Infrastructure
docker-compose up nats minio

# Terminal 2: Core
export NATS_URL=nats://localhost:4222
uv run python -m src.core.main

# Terminal 3: Ingestor
export NATS_URL=nats://localhost:4222
uv run python -m src.ingestor.main
```
