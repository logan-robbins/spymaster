
# Market Wind Tunnel (MWT) — Critical Path Implementation Guide (V1)

**Audience**: Engineering execution team + AI/LLM coding agents (this is intentionally “machine-readable”, not polished prose).  
**Primary objective (V1)**: **Stream 1-second market “surfaces” and render them as a GPU fluid / aero-dynamics visualization** where *price behaves like a particle moving through a viscous medium with pressure forces and obstacles (resting liquidity walls, GEX walls, etc.)*.  
**Non-goals (explicit)**: do **not** implement forward prediction; do **not** address non-functional requirements (latency budgets, scale, HA, auth, etc.).  
**V1 definition of “done”**: A UE Niagara Fluids 2D simulation is live-updated from the backend’s 1-second HUD stream and is viewable locally; the same project can be run on an Azure GPU VM and accessed via Pixel Streaming.

## Task Plan (2026-01-06 RTH)
1. Inventory Unreal artifacts in repo + define safe cleanup targets. [done]
2. Add Unreal cleanup utility and ignore rules. [done]
3. Add Remote Control API client + UE control hooks. [done]
4. Update README/DOCS for Unreal cleanup + remote control usage. [done]
5. Verify minimal CLI smoke with `uv run`. [done]
6. Add Remote Control map utilities (list/open/delete). [done]
7. Update docs for single-map workflow. [done]
8. Verify CLI smoke with `uv run`. [done]
9. Create clean map with BP_MwtReceiver + prune other maps. [done]
10. Set EditorStartupMap to the single map. [done]

---

## 0. Source-of-truth Inputs (Do not deviate)

### 0.1 Existing backend + frontend stream contract (authoritative)

The backend already publishes the 1-second visualization data via:

- **Transport**: WebSocket  
- **Endpoint**: `ws://localhost:8000/v1/hud/stream`  
- **Format**: **Arrow IPC** (binary frames)  
- **Cadence**: **1 second windows**  
- **Framing**: a `batch_start` control frame per window and a `surface_header` control frame immediately preceding each binary Arrow payload.  
  :contentReference[oaicite:0]{index=0}

This means the fastest V1 is **not**: “read `backend_data.json` snapshots and invent a new protocol”.  
The fastest V1 is: **consume the existing HUD WebSocket stream**, then (optionally) bridge it into a simpler UE-friendly datagram format.

### 0.2 Tick-native coordinate system invariants (authoritative)

The visualization grid is defined as:

- **1 column = 1 second** (window cadence)  
- **1 row = 1 tick** ($0.25 for ES)  
- **Center row = spot price** (rel_ticks = 0)  
:contentReference[oaicite:1]{index=1}

Key anchor rule:

- `spot_ref_price_int` is the **master anchor** for all surfaces; it is tick-aligned and used for geometry anchoring. `mid_price` is cosmetic.  
:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

V1 must respect this: **all spatial fields are indexed by `rel_ticks` relative to `spot_ref_price_int`**.

### 0.3 What the backend already computes (authoritative formulas)

You must treat the following as the canonical baseline computations:

- **Snap**:  
  `mid_price = (best_bid + best_ask) * 0.5 * PRICE_SCALE`  
  `spot_ref_price_int = last_trade_price_int || (book_valid ? best_bid_price_int : 0)`  
  :contentReference[oaicite:4]{index=4}

- **Wall surface**:  
  `rel_ticks = round((price_int - spot_ref) / TICK_INT)`  
  `depth_qty_start = max(depth_end - add + pull + fill, 0)`  
  `d1_depth_qty = depth_end[t] - depth_end[t-1]`  
  `d2_depth_qty = d1[t] - d1[t-1]`  
  `d3_depth_qty = d2[t] - d2[t-1]`  
  :contentReference[oaicite:5]{index=5}

- **Vacuum surface**: composite normalized score 0..1:  
  `vacuum_score = (n1 + n2 + n3 + n4) / 4` with  
  - `n1 = norm(pull_add_log)`  
  - `n2 = norm(log1p(pull_intensity_rest))`  
  - `n3 = norm(log1p(wall_erosion / (depth_start + EPS)))`  
  - `n4 = norm(d2_pull_add_log)`  
  and normalization function  
  `norm(val) = clip((val - q05) / (q95 - q05), 0, 1)` using calibration dataset `gold.hud.physics_norm_calibration`.  
  :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

- **Physics surface**: directional ease (signed):  
  `ease = 0.5*vac + 0.35*erosion + 0.15*(1-wall)`  
  `physics_score_signed = +ease if side=A else -ease if side=B`  
  :contentReference[oaicite:8]{index=8}

- **GEX surface** (0DTE options):  
  `gamma = e^(-rT) * N'(d1) / (F * iv * √T)`  
  `gex = gamma * OI * 50`  
  `gex_abs = call + put`  
  `gex = call - put`  
  `gex_imbalance_ratio = gex / (gex_abs + EPS)`  
  :contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}

Constants currently used (must match backend + HUD expectations):

- `WINDOW_NS = 1,000,000,000`  
- `REST_NS = 500,000,000` (resting orders > 500ms)  
- `TICK_SIZE = 0.25`  
- `TICK_INT = 250,000,000`  
- `PRICE_SCALE = 1e-9`  
:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}

---

## 1. Architecture & Execution Strategy (Fastest V1)

### 1.1 Minimal end-to-end topology

**Existing**:  
Bronze → Silver surfaces → WebSocket HUD stream → Frontend WebGL  
:contentReference[oaicite:13]{index=13}

**Target V1 adds**:  
WebSocket HUD stream → *Bridge* (local Python) → Unreal Engine (Niagara Fluids) → (optional) Pixel Streaming → Browser

**Why a bridge exists** (critical):  
- Unreal Engine can receive UDP trivially using built-in `FUdpSocketReceiver`. :contentReference[oaicite:14]{index=14}  
- Parsing **Arrow IPC** inside UE (C++) is a large dependency cost (Arrow C++ build, ABI concerns on Mac+Windows). Arrow IPC decoding is trivial in Python via `pyarrow.ipc.open_stream`. :contentReference[oaicite:15]{index=15}  
=> For the **fastest path**, decode Arrow in Python, then send **simple sparse tick updates** to UE.

### 1.2 Local vs Cloud split (Apple M4 + Azure)

**Local (Apple M4 / macOS)**:
- Run backend and stream HUD WebSocket locally.  
- Run Bridge locally.  
- Run Unreal Editor locally (Metal).  
- Validate simulation logic and visual semantics quickly in-editor.

**Cloud (Azure GPU VM)**:
- Run packaged Unreal build on an NVads A10 v5 GPU VM (e.g., `Standard_NV6ads_A10_v5`). The NVadsA10v5-series is designed for GPU-accelerated graphics apps and provides NVIDIA A10 GPUs with partitioning options. :contentReference[oaicite:16]{index=16}  
- Host Pixel Streaming infrastructure + UE app for remote browser access. Pixel Streaming infrastructure provides the signalling/server components and is now distributed as a dedicated “Pixel Streaming Infrastructure” package/repo in modern UE versions. :contentReference[oaicite:17]{index=17}

---

## 2. Software Required (Install Immediately)

### 2.1 Backend / data pipeline (already in repo)

From the project’s launch instructions:

Backend:
```bash
cd backend
uv run python -m src.serving.main
# WebSocket: ws://localhost:8000/v1/hud/stream?symbol=ESH6&dt=2026-01-06
````

Frontend (optional for debugging):

```bash
cd frontend
npm run dev
# http://localhost:5173
```

【66:0†README.md†L3-L17】

Silver pipeline generation commands (for offline/recorded playback datasets):

```bash
cd backend
uv run python -m src.data_eng.runner --product-type future_mbo --layer silver --symbol ES --dates 2026-01-06 --workers 1
uv run python -m src.data_eng.runner --product-type future_option_mbo --layer silver --symbol ES --dates 2026-01-06 --workers 1
```

【66:0†READMEion scripts (useful for sanity before working on UE):

```bash
cd backend
uv run python scripts/test_data_integrity.py --dt 2026-01-06
uv run python scripts/test_physics_integrity.py --dt 2026-01-06
uv run python scripts/simulate_frontend.py
```

【66:0†README.md†L74-L81】

### 2.2 Bridge runtime (local + cloud)

* Python 3.10+ (match repo)
* `uv` (alrearow` for Arrow IPC decoding (required by stream format). ([Apache Arrow][1])
* A WebSocket client library (`websockets` or equivalent)

### 2.3 Unreal Engine (local authoring)

* Unreal Engine **5.4+** (use a modern UE5 branch; Niagara Fluid are actively maintained in newer UE versions).
* Xcode 15+ (macOS toolchain for C++ modules)
* UE Plugins enabled:

  * Niagara
  * Niagara Fluids (grid simulations) ([Epic Games Developers][2])
  * Networking (built-in)
  * (Later) Pixel Streaming ([Epic Games Developers][3])

### 2.4 Pixel Streaming servers (Azure)

* Node.js runtime (to run signalling/web server components)
* Pixel Streaming Infrastructure package/scripts (Signalling Server / SFU / frontend). ([Epic Games Developers][4])
* GPU driver stack (NVIDIA A10 on NVads A10 v5 Azure VMs) ([Microsoft Learn][5])

---

## 3. Data Inputs to Visualization (Surfaces)

### 3.1 Surfaces streamed per 1-second window

The backend streams multiple “surfaces” per window, each with its own Arrow payload, including:

* `snap` (1 row) — provides spot anchor and spot line
* `wall` (variable rows per window within ±400 ticks) — resting liquidity & dynamics
* `vacuum` (variable rows per window within ±400 ticks) — erosion / pull forces
* `physics` (variable rows, usually around spot) — signed directional ease
* `gex` (25 rows) — 0DTE gamma exposure grid
  【66:1†frontend_data.json†L30-L90】【66:1†frontend_data.json†L91-L97】

The frontend’s Z-order stack (useful for conceptual mapping to UE render layers):

* Physics (green above / red below)
* Vacuum (black overlay)
* Wall (blue asks / red bids)
* GEX (green calls / red puts)
* Spot line (cyan)
  【66:0†README.md†L52-L60】

### 3.2 Key field semantics for V1 physics mapping

**Snap**

* `spot_ref_price_int` = anchor
* `mid_price` = cosmetic
* `book_valid` = validity flag
  【66:1†frontend_data.json†L31-L58】

**Wall**

* `rel_ticks` = tick offset from spot
* `side` = 'A' (ask) vs 'B' (bid)
* `depth_qty_rest` = resting depth (>500ms old) used for intensity
* derivatives: `d1_depth_qty`, `d2_depth_qty` exist (optional usage)
  【66:1†frontend_data.json†L65-L113】【70:13†frontend_data.json†L1-L57】

**Vacuum**

* `vacuum_score` normalized 0..1
* optional `d2_p:contentReference[oaicite:26]{index=26}:contentReference[oaicite:27]{index=27}tend alpha mapping: `alpha = vacuum_score * 128` (informational)
  【66:1†frontend_data.json†L59-L63】【70:8†frontend_data.json†L3-L42】

**Physics**

* `physics_score_signed` meaning: + upward ease, - downward ease
* Used as directional ntend
  【66:1†frontend_data.json†L7-L23】【70:1†frontend_data.json†L1-L23】

**GEX**

* `gex_abs` magnitude, `gex_imbalance_ratio` in [-1, +1] for calls vs puts in 5-point steps (ES strikes), which implies **20 ticks** per strike step (5.00 / 0.25).
  Frontend warns: GEX must align exactly to 20-tick grid.
  【66:3†DOCS_FRONTEND.md†L13-L18】【66:1†frontend_data.json†L25-L89】

---

## 4. Vnt: UE should not parse Arrow in V1

Arrow IPC is a streaming binary protocol. It is designed to be read sequentially (stream reader), and Python has direct support for it. ([Apache Arrow][1])stone, but that is not V1-critical.

### 4.2 Bridge responsibilities (exact)

The Bridge must:

1. Connect to the HUD WebSocket endpoinor that `window_end_ts_ns`.
2. Decode each surface’s Arrow IPC payload → in-memory record batch/table.
3. Convert each surface into a **tick-native sparse representation** suitable for UE.
4. Emit per-surface **UDP datagrams** to UE on localhost (dev) or to UE built-in UDP receive utilities:

* `FUdpSocketReceiver` asynchronously receives data from a UDP socket. ([Epic Games Developers][6])
* `FUdpSocketBuilder` provides socket construction/bind configuration. ([Epic Games Developers][7])

This is simpler than integrating WebSocket + Arrow directly in UE.

---

## 5. Bridge-to-UE Datagram Contract (MWT-UDP v1)

### 5.1 Design principles

* **Tick-native**: send `rel_ticks` not float prices.
* **Sparse**: send only active levels (thresholded or top-K) to keep datagrams bounded.
* **Per-surface packets**: avoids MTU fragmentation; each packet remains < ~1400 bytes.
* **Idempotent**: each packet contains `window_end_ts_ns` so UE can discard stale updates.

### 5.2 Packet envelope (all surfaces)

All packets share a common header:

| Offset | Field                | Type   | Notes                                                                                    |
| -----: | -------------------- | ------ | ---------------------------------------------------------------------------------------- |
|   0..3 | `Magic`              | uint32 | ASCII “MWT1”                                                                             |
|   4..5 | `Version`            | uint16 | 1                                                                                        |
|   6..7 | `SurfaceId`          | uint16 | SNAP=1, WALL=2, VACUUM=3, PHYSICS=4, GEX=5                                               |
|  8..15 | `window_end_ts_ns`   | int64  | canonical window timestamp                                                               |
| 16..23 | `spot_ref_price_int` | int64  | anchor for `rel_ticks` (copy from snap; for non-snap surfaces, use their provided value) |
| 24..27 | `Count`              | uint32 | number of entries following                                                              |
| 28..31 | `Flags`              | uint32 | reserved (0 in v1)                                                                       |

**Why include `spot_ref_price_int` in every packet**: if the `snap` packet drops, the surface packets still remain spatially meaningful. The frontend spec also notes that `gex` includes underlying spot to use if snap missing. 【66:1†frontend_data.json†L44-L47】

### 5.3 Payload layout per SurfaceId

#### 5.3.1 SNAP payload

Exactly 1 entry (Count=1):

| Field        | Type    | Notes                               |
| ------------ | ------- | ----------------------------------- |
| `mid_price`  | float64 | cosmetic                            |
| `book_valid` | uint8   | 0/1                                 |
| `reserved`   | 7 bytes | pad to 16-byte alignment (optional) |

#### 5.3.2 WALL payload

Each entry:

| Field            | Type    | Notes                                                                                                    |
| ---------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `rel_ticks`      | int16   | [-400, +400] (expected)                                                                                  |
| `side`           | uint8   | 0=B (bid), 1=A (ask)                                                                                     |
| `wall_intensity` | float32 | **Bridge-computed** from `depth_qty_rest` using `log1p` mapping (see §6)                                 |
| `wall_erosion`   | float32 | optional (if present: `max(-d1_depth_qty,0)` or use backend `wall_erosion` if already available); else 0 |

This supports both “static obstacle strength” and “decay/erosion” visuals.

#### 5.3.3 VACUUM payload

Each entry:

| Field          | Type    | Notes                                                         |
| -------------- | ------- | ------------------------------------------------------------- |
| `rel_ticks`    | int16   |                                                               |
| `vacuum_score` | float32 | from backend (0..1)                                           |
| `turbulence`   | float32 | recommended = clamp(abs(d2_pull_add_log),0,…) or 0 if missing |

#### 5.3.4 PHYSICS payload

Each entry:

| Field                  | Type    | Notes                           |
| ---------------------- | ------- | ------------------------------- |
| `rel_ticks`            | int16   |                                 |
| `physics_score_signed` | float32 | from back#### 5.3.5 GEX payload |

Each entry:

| Field                 | Type    | Notes                               |
| --------------------- | ------- | ----------------------------------- |
| `rel_ticks`           | int16   | must align to 20 ticks (ES strikes) |
| `gex_abs`             | float32 | magnitude                           |
| `gex_imbalance_ratio` | float32 | [-1, +1]                            |

---

## 6. Bridge Transformations (Pre-calc for the Engine)

This is where we explicitly convert “market metrics” into “fluid knobs” **without touching the backend Silver pipeline**.

### 6.1 Shared tick window and dense grid baseline

Even though packets are sparse, the Bridge must operate against the canonical **±400 tick** window:

* `LAYER_HEIGHT_TICKS = 801` (±400 ticks around spot)
  【70:14†DOCS_FRONTEND.md†L21-L27】

Define:

* `T = 400`
* Dense index mapping: `idx = rel_ticks + T` → [0..800]

### 6.2 Wall intensity mapping (critical)

Frontend uses: `Intensity = log1p(depth_qty_rest)` for liquidity heatmap.
【70:13†frontend_data.json†L54-L57】

So Bridge must compute:

**Formula**:

* `wall_intensity_raw = log(1 + depth_qty_rest)`

Then compress to an engine-friendly normalized value:

* `wall_intensity_norm = clamp(wall_intensity_raw / W_LOG_MAX, 0, 1)`

Where `W_LOG_MAX` is chosen empirically (V1: choose a constant such that typical large walls saturate).

**Why**: Niagara needs bounded values; also you need stable mapping across sessions.

### 6.3 Wall erosion mapping

Backend already defines “wall_erosion” in vacuum derivations as:

* `wall_erosion = max(-d1_depth_qty, 0)` (from wall_surface)
  【70:12†backend_data.json†L60-L69】

If the `wall` stream does not include `wall_erosion` explicitly (frontend schema does not require it), the Bridge must compute:

* `wall_erosion_raw = max(-d1_depth_qty, 0)`

Then normalize similarly:

* `wall_erosion_norm = clamp(log(1 + wall_erosion_raw / (depth_qty_start + EPS)) / E_LOG_MAX, 0, 1)`

(Consistent with baion/(depth_start+EPS) )` as an input component for vacuum scoring.)
【70:12†backend_data.json†L91-L95】

### 6.4 Vacuum score (already normalized)

The backend provides `vacuum_score` in [0,1] as cond_data.json†L87-L98】【70:8†frontend_data.json†L21-L37】

Bridge rule:

* Use `vacuum_score` directly.
* If `vacuum_score` missing for a tick: treat as 0.

### 6.5 Physics signed score (already computed)

The physics stream’s `physics_score_signed` is derived from ease, with sign depending on side:

* `physics_score_signed = +ease if A else -ease if B`
  【70:0†backend_data.json†L91-L99】

Bridge rule:

* Use it directly.
* If physics stream is sparse around spot (likely), this is acceptable.

### 6.6 GEX mapping (engine consumption)

GEX fields used for visualizati `gex_imbalance_ratio` for color (call-heavy green / put-heavy red)
【66:1†frontend_data.json†L62-L77】

Bridge rule:

* Use as-is.
* Validate `rel_ticks % 20 == 0` (do not “fix” it; if not aligned, treat as data error).

### 6.7 Temporal decay / dissipation (V1 must mimic existing semantics)

Frontend uses temporal decay for “pressure hanging”:

* Physics: τ=5s
* Vacuum: τ=5s
* Wall: τ=0 (instant clear)
* Rule: `newen apply new writes
  【70:14†DOCS_FRONTEND.md†L28-L33】

For V1, implement this decay in **UE-side state** (preferred) inside the Bridge and keeps Bridge stateless.

---

## 7. Unreal Engine Implementation Plan (V1)

### 7.1 UE project creation

Project name suggestion: `MarketWindTunnel` (separate from existing frontend).
Type: **Games → Blank → C++**.

### 7.2 Enable required plugins

Enable:

* Niagara
* Niagaates) ([Epic Games Developers][2])
* Pixel Streaming (only needed for cloud demo) ([Epic Games Developers][3])

### 7.3 Add a UDP Receiver Actor (C++)

**Actor**: `AMwtUdpReceiver`

Responsibilities:

* Bind UDP socket on configured port (default 7777).
* Use `FUdpSocketRe worker thread. ([Epic Games Developers][6])
* Parse header + payload into a small POD struct per surface type.
* Push surface updates into a thread-safe queue.
* On Tick (game thread): apply updates to Niagara input parameters (arrays, textures, or data channels).

**Why C++ not Blueprint**:

* binary parsing + queue + array updates are easier, deterministic, and less error-prone.

#required)

Maintain persistent per-surface state for the **±400 tick window** (801 slots):

* `SpotRefPriceInt` (int64)
* `MidPrice` (double) cosmetic
* Arrays (float):

  * `WallAsk[801]`, `WallBid[801]` (or single array with sign)
  * `WallErosion[801]`
  * `Vacuum[801]`
  * `PhysicsSigned[801]`
  * `GexAbs[801]`
  * `GexImbalance[801]`

Update rules:

* When a new SNAP arrives: update anchors.
* When a new WALL arrives for `window_end_ts_ns`: **clear** Wall arrays first (τ=0), then write sparse entries.
* When a new VACUUM arrives: decay Vacuum array by exp(-Δt/5s) before applying new writes.
* When a new PHYSICS arrives: decay Physics array by exp(-Δt/5s) before applying new writes.
* When a new GEX arrives: either no decay or a slow decay (V1 can treat as “replace per window”).

### 7.5 Feeding Niagara (choose 1 ingestion path)

V1 choice is about speed of implementation, not elegance.

**Path A (fastest): Niagara User Parameter Arrays**

* Create Niagara System user parameters:

  * `User.WallAsk` (Array Float)
  * `User.WallBid` (Array Float)
  * `User.Vacuum`
  * `User.PhysicsSigned`
  * `User.GexAbs`
  * `User.GexImbalance`
  * `User.SpotRelTick` (float) = 0 (spot is center)
* UE receiver writes arrays each update tick.

**Path B (more scalable later): Niagara Data Channels**

* Use Niagara Data Channels to push sparse per-tick events.
* Better long-term, but more setup.

For V1, implement **Path A**.

---

## 8. Niagara Fluids System (The Wind Tunnel)

### 8.1 What Niagara Fluids gives you

Niagara Fluids gas sims are grid-based: each cell stores values like density and velocity; resolution trades off quality vs compute. ([Epic Games Developers][8])

For V1 we will use a **2D Gas** template (not liquid/FLIP) because:

* It already solves a Navier–Stokes-like advection + pressure projection pipeline
* It is sufficient to represent “pressure fields”, “viscosity obstacles”, and “tracer dye”

### 8.2 Create the Niagara system from template

Create: Niagara System → template **Grid 2D Gas** (choose a smoke/fire template if needed, then strip visuals). Niagara Fluids includes templates for smoke/fire and other grid sims. ([Epic Games Developers][2])

Name: `NS_MarketWindTunnel`.

### 8.3 Grid resolution & mapping to tick space

Set grid resolution to align with tick-native window:

* **Y resolution**: 801 (one cell per tick across ±400 ticks)
* **X resolution**: start at 256 (history “width”)
* World size:

  * Y world size = 801 * `TickWorldScale`
  * X world size = `HistoryWorldWidth`

Set `TickWorldScale`:

* V1 recommended: `TickWorldScale = 0.25` world units per tick (mirrors frontend’s tick-native scale concept).
  The frontend explicitly uses tick-native integer math and defines tick space and world/GL unit mappings; its goal is to eliminate drift.
  【66:2†DOCS_FRONTEND.md†L25-L38】

### 8.4 “Wind tunnel” flow direction

We want the tracer to advect **to the right** (history accumulation). Implement:

* constant horizontal velocity field `Vx = +WindSpeed`
* no need for actual physical correctness; it just maps time into X.

Set `WindSpeed` so that 1 second of simulation produces a visible shift. (Exact value can be tuned visually; not a functional requirement.)

### 8.5 Price tracer injection (the “particle”)

Even though price is not simulated in V1, we **visualize** price as a tracer source:

* inject **density** (dye) at:

  * X = left boundary (or near-left)
  * Y = `rel_ticks = 0` (centerline anchored to spot)
* BUT: spot moves each second; we represent this as the entire grid being spot-relative, so tracer injection stays at center.

This matches tick-native architecture: the spot anchor is the moving reference; everything is drawn relative to it.
【66:2†DOCS_FRONTEND.md†L3-L12】【66:2†DOCS_FRONTEND.md†L25-L36】

### 8.6 Walls as viscosity / obstacles

Map Wall arrays to a field that resists flow.

For each tick row `y`:

* compute `WallResistance(y)` from `WallAsk[y]` and `WallBid[y]`

V1 recommended:

* `WallResistance = clamp(WallAsk + WallBid, 0, 1)`

Apply to the grid:

* Increase viscosity / damping in those cells:

  * `Velocity *= (1 - WallResistance * WallDampingGain)`
* Optionally add density “clumps” at wall levels to act like visible obstacles.

**Why this matches your metaphor**:

* Resting depth is “debris / barrier mass”
* High viscosity zones cause the “flow” to detour around them

### 8.7 Vacuum as pressure sink / suction

Vacuum score is already 0..1.

At each tick row:

* `VacuumForce(y) = Vacuum[y] * VacuumGain`

Apply as:

* pressure sink: `Pressure -=:contentReference[oaicite:56]{index=56}eleration: `VelocityY += sign * VacuumForce` (if you choose a directional model)

V1 recommended: implement as **pressure perturbation** because the solver will convert that into velocity gradients naturally.

### 8.8 Physics signed score as vertical bias field

Physics provides a signed ease-of-movement:

* positive means “easy to move up”
* negative means “easy to move down”
  【66:3†DOCS_FRONTEND.md†L15-L16】【70:1†frontend_data.json†L14-L23】

Map:

* `ExternalForceY(y) = PhysicsSigned[y] * PhysicsForceGain`

Apply as:

* `VelocityY += ExternalForceY`

This gives immediate interpretability: the flow field tilts upward above spot if physics indicates upward ease.

### 8.9 GEX as large-scale boundary constraint

GEX is lower-resolution (25 strikes) but meaningful as “macro obstacles”:

* Convert gex samples into tick rows:

  * Each gex entry provides `rel_ticks` (multiples of 20 tiAbs[y] / GEX_MAX, 0, 1)`
  * `GexSign(y) = GexImbalance[y]` in [-1,+1]

Use in sim:

* Add static viscosity component:

  * `TotalResistance(y) = WallResistance(y) + GexBarrier(y) * GexResistanceGain`
* (Optional) Add directional bias:

  * if `GexSign > 0` treat as “upward pressure” near that strike; if `<0`, downward.

This meets your conceptual model: GEX walls act like macro structural debris.

---

## 9. Rendering Semantics (What must be visible in V1)

V1 must show **distinctly**:

1. **Spot / Price trace**: continuous tracer/dye line
2. **Walls**: visible obstacles / dense bands; ask vs bid distinguished (blue/red in frontend)
3. **Vacuum**: “erosion overlay” effect (darkening or suction vortices)
4. **Physics**: directional layer (green above / red below in frontend)
5. **GEX**: macro heatmap overlay (green calls / red puts)

Frontend’s layer mapping is explicit; use it to validate semantics.
【66:0†README.md†L52-L60】

Implementation notes:

* Use different render passes or material tints for each field.
* Do not attempt to perfectly match frontend aesthetics; match meaning.

---

## 10. Local Runbook (Apple M4)

### 10.1 Start the backend stream

````bash
cd:contentReference[oaicite:59]{index=59}:contentReference[oaicite:60]{index=60} is available (you can also run the existing frontend for validation):
```bash
cd frontend
npm run dev
````

【66:0†README.md†L3-L17】

### 10.2 Run the Bridge (new component)

Bridge must connect to:

* `ws://localhost:8000/v1/hud/stream?...` (same as frontend)
  【66:0†README.md†L5-L10】

Bridge must send UDP to:

* `127.0.0.1:7777` (default)

### 10.3 Run Unreal Editor

* Open `MarketWindTunnel.uproject`
* Press Play in Editor
* Confirm UDP receiver logs show packets for all surfaces at ~1Hz

---

## 11. Azure Runbook (Pixel Streaming Demo)

### 11.1 VM selection (GPU)

Use an NVads A10 v5 VM size series. Microsoft documents the NVadsA10v5-series as NVIDIA A10 GPU-powered VMs suitable for GPU-accelerated graphics, with options from fractional to full GPU allocations. ([Microsoft Learn][5])

Suggested starting point:

* `Standard_NV6ads_A10_v5` (A10-backed) ([Microsoft Learn][5])

### 11.2 Pixel Streaming infrastructure

Modern UE docs describe:

* Pixel Streaming Infrastructure contains the servers and frontend components (Signalling Server, Matchmaker, SFU) and is designed to be modified/extended. ([Epic Games Developers][4])
* Getting started guidance includes using LAN/VPN first and then STUN/TURN for broader internet access. ([Epic Games Developers][9])
* Pixel Streaming Reference notes supported OS/hardware encoders and testing on macOS Ventura and Ubuntu versions, etc. ([Epic Games Developers][10])ckaged build + signalling server on the same VM.
* Expose required ports (as per Pixel Streaming docs). ([Epic Games Developers][9])

### 11.3 Where the Bridge runs in cloud

Two valid V1 modes:

**Mode A (simplest)**: Bridge runs on the same VM as UE

* Bridge connects to backend stream (wherever backend runs)
* Bridge sends UDP to `127.0.0.1:7777`

**Mode B**: Bridge runs near backend, sends UDP over netwoUDP routing and firewall rules (functional but more moving parts)

V1 default: Mode A.

---

## 12. Implementation Checklist (Striirm inputs

* [x] Backend can stream WS HUD at 1Hz.
* [x] `snap`, `wall`, `vacuum`, `physics`, `gex` surfaces appear in the stream.
* [x] Spot anchor (`spot_ref_price_int`) is non-zero and stable.

### Phase 1 — Bridge

* [x] Bridge decodes Arrow IPC via Python stream reader (`pyarrow.ipc.open_stream`).
* [x] Bridge emits UDP packets per surface with MWT header + sparse payload (as defined).
* [x] Bridge computes `wall_intensity_raw = log1p(depth_qty_rest)` for wall.
* [x] Bridge forwards `vacuum_score` as-is (0..1).
* [x] Bridge forwards `physics_score_signed` as-is.
* [x] Bridge forwards `gex_abs` + `gex_imbalance_ratio`.

### Phase 2 — UE ingestion

* [x] UDP receiver uses `FUdpSocketReceiver`.
* [x] Receiver updates per-surface arrays (801 ticks) anchored to spot.
* [x] Receiver applies dissipation (τ=5s) for physics + vacuum; clears wall each update.

### Phase 3 — Niagara

* [ ] Create Niagara Fluids Grid 2D system. ([Epic Games Developers][2])
* [ ] User parameter arrays drive wall/vacuum/physics/gex fields.
* [ ] Visual output shows:

  * tracer (price)
  * obstacles (walls)
  * suction/pressure (vacuum)
  * directional bias (physics)
  * macro overlay (gex)

### Phase 4 — Azure + Pixel Streaming (optional for V1 demo)

* [ ] Package UE build for Windows
* [ ] Deploy to NVads A10 v5 VM ([Microsoft Learn][5])
* [ ] Run Pixel Streaming infrastructure scripts ([Epic Games Developers][4])
* [ ] Browser can connect and view stream ([Epic Games Developers][3])

---

## 13. Reserved “Future Prediction” Hook (Do not implement in V1)

To support “predict n seconds ahead” later, reserve these additions in the UDP header/packet schema:

* `PredictionHorizonNs` (int64)
* `PredictedSpotRelTicks[]` (sparse polyline)
* `PredictedPressureField[]` (coarse)

No logic required now; just keep schema extensible.

---

## 14. Appendices

### Appendix A — Why Niagara Fluids for this problem

Niagara Fluids provides grid-based gas simulations where each cell contains values like density and velocity; increasing grid resolution increases quality at compute cost. ([Epic Games Developers][8])
This maps directly to your conceptual model:

* density = tracer / price dye
* viscosity/ gex walls
* pressure sinks = vacuum
* external forcing = physics signed ease

#ocs pointers (operational)

* Pixel Streaming Infrastructure overv([Epic Games Developers][4])
* Pixel Streaming getting-started guidance (LAN/VPN first, STUN/TURN later) ([Epic Games Developers][9])
* Pixel Streaming Reference on supported OS/hardware encoders ([Epic Games Developers][10])

---

## References (repo artifacts)



::contentReference[oaicite:90]{index=90}
:contentReference[oaicite:91]{index=91}:contentReference[oaicite:92]{index=92}:contentReference[oaicite:93]{index=93}:contentReference[oaicite:94]{index=94}:contentReference[oaicite:95]{index=95}:contentReference[oaicite:96]{index=96}

[1]: https://arrow.apache.org/docs/python/ipc.html?utm_source=chatgpt.com "Streaming, Serialization, and IPC — Apache Arrow v23.0.0"
[2]: https://dev.epicgames.com/community/learning/paths/mZ/unreal-engine-niagara-fluids?utm_source=chatgpt.com "Niagara Fluids | Epic Developer Community"
[3]: https://dev.epicgames.com/documentation/en-us/unreal-engine/pixel-streaming-in-unreal-engine?utm_source=chatgpt.com "Pixel Streaming in Unreal Engine"
[4]: https://dev.epicgames.com/documentation/en-us/unreal-engine/pixel-streaming-infrastructure?utm_source=chatgpt.com "Pixel Streaming Infrastructure | Unreal Engine 5.7 ..."
[5]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series?utm_source=chatgpt.com "NVadsA10_v5 size series - Azure Virtual Machines"
[6]: https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Networking/FUdpSocketReceiver?utm_source=chatgpt.com "FUdpSocketReceiver | Unreal Engine 5.7 Documentation"
[7]: https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Networking/FUdpSocketBuilder?utm_source=chatgpt.com "FUdpSocketBuilder | Unreal Engine 5.7 Documentation"
[8]: https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine?utm_source=chatgpt.com "Niagara Fluids Reference in Unreal Engine"
[9]: https://dev.epicgames.com/documentation/en-us/unreal-engine/getting-started-with-pixel-streaming-in-unreal-engine?utm_source=chatgpt.com "Getting Started with Pixel Streaming in Unreal Engine"
[10]: https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-pixel-streaming-reference?utm_source=chatgpt.com "Unreal Engine Pixel Streaming Reference"
