Yep. You’re describing a **price particle** moving through a **learned, time‑varying medium**: currents (drift), viscosity (damping), pressure vacuums (attractors), and soft obstacles (repellers/reflectors with memory). And you want it to run *live*, predict **t+3/t+4 (500ms steps)**, learn weights online, and visualize predicted vs actual in a “physics lab” UI.

Below is a **technical implementation document** (what + why; minimal how; no code) for a **simple working model first** that’s architected to absorb a *huge* feature set later without rewrites.

---

# Market Particle Physics Engine

## Technical Implementation Document (Swift 6.2 + Metal + SwiftUI)

### Target platforms

* **macOS first** (faster iteration, bigger canvas, easier GPU debugging), then iOS with the same engine.
* **Swift 6.2 toolchain** for modern concurrency/performance and long-term support. ([Swift.org][1])

---

## 1) Core idea and success criteria

### What we are building

A real-time “physics” simulator where:

* The **spot price** is a **single particle** with position/velocity in 1D price space.
* The “market state” is a **medium** represented as a small set of **canonical fields** defined over price levels:

  1. **Potential field** (pressure vacuum / attraction)
  2. **Current field** (preferred drift velocity)
  3. **Damping field** (viscosity / friction)
  4. **Obstacle field** (soft walls, barriers with memory)
  5. Optional **diffusion/noise** (stochastic volatility)

The simulator runs each 500ms tick and generates a **multi-step forecast** (H=4 by default: t+1..t+4).

### Why this abstraction works

Your features may be huge, but almost everything you described can be expressed as:

* “This region pulls price toward it” → **potential**
* “Flow pushes price through here” → **current**
* “Market resists motion here” → **damping**
* “This level blocks/reflects until it erodes” → **obstacle with memory**

That means the model scales by **adding influence producers**, not by changing the physics core.

### Definition of “simple working model”

Day‑1 deliverable is **not** a perfect market model. It is:

* A stable 1D particle sim
* A live UI that shows:

  * actual price
  * predicted path (t+1..t+4)
  * a visual of the medium (potential + obstacle)
* A parameter vector (weights) that can be tuned online, even if starting with just 3–6 weights.

---

## 2) Technology stack (final choices)

### What

* **SwiftUI** for app shell + controls (sliders, toggles, replay controls)
* **Swift Charts** for time-series overlays (actual vs predicted per horizon). ([Apple Developer][2])
* **Metal** for compute + rendering
* **MetalKit (MTKView)** as the Metal-backed rendering surface embedded in SwiftUI. ([Apple Developer][3])
* Metal compute dispatch via `MTLComputeCommandEncoder` (authoritative GPU pipeline). ([Apple Developer][4])

### Why

* This is **agent-friendly**: explicit buffers in/out, explicit passes, minimal hidden engine state.
* It gives you a clean “math core” that isn’t entangled with a game engine scene graph.
* Metal + MTKView is the “native” route for deterministic GPU compute + custom visualization. ([Apple Developer][3])

---

## 3) Coordinate system and timebase (non-negotiable)

### What

* Time step: **Δt = 0.5 seconds** (your 500ms feature cadence)
* Prediction horizon: **H = 4** steps (2.0 seconds forward)
* Spatial axis: **ticks**, with your natural bucket size:

  * **Δx = 2 ticks** per spatial cell
  * Grid is centered on the current spot/BBO mid at time t:

    * indices i ∈ [-N, …, +N]
    * x(i) = i * Δx ticks

### Why

* You already think in “buckets above/below spot.” Keep the world aligned to that.
* Centering the grid on spot reduces dynamic range problems and keeps your fields always “in view.”

---

## 4) Data contracts (the engine doesn’t care where features come from)

### What

At each tick t (every 500ms) the simulator receives a **MarketFrame**:

* `timestamp`
* `spotPrice` (or mid)
* `tickSize` (or “ticks” already)
* `features` in one of two shapes:

  * **Scalar channels** (global features for this 500ms)
  * **Profile channels** over the grid (per-bucket features above/below)

Crucially: the physics core consumes **canonical influence fields**, not raw features.

### Why

This keeps step‑1 small: you can feed only 1–3 channels initially, then scale to dozens/hundreds without refactoring the simulator.

---

## 5) Canonical influence fields (the “medium”)

### What we maintain per tick

All fields are defined over the same grid i ∈ [-N..N].

1. **Potential field U(i)**

   * Scalar energy landscape.
   * Price particle accelerates “downhill”: attraction toward minima.
   * Used to model “pressure vacuums suck price toward them.”

2. **Current field C(i)**

   * Preferred local drift velocity.
   * Used to model persistent order-flow “wind.”

3. **Damping field Γ(i)**

   * Viscosity / friction coefficient.
   * Higher Γ means motion is resisted.

4. **Obstacle field O(i)** + **Obstacle memory state**

   * A barrier energy term and/or local reflect/repel behavior.
   * Backed by a stateful representation so walls can **erode or thicken**.

5. Optional: **Diffusion D(i) or Dscalar**

   * Controls stochasticity / uncertainty envelope.

### Why this set

These are the minimum building blocks that map directly onto your mental model (currents, viscosity, vacuums, walls) while staying mathematically stable and easy to extend.

---

## 6) The physics model (1D particle with learned force mixing)

### State variables

* Position: x(t) in ticks (continuous)
* Velocity: v(t) in ticks/second (continuous)

### Dynamics

We model acceleration as a weighted sum of force components:

* **Force from potential:** F_U(x) = −∂U/∂x
* **Force toward current:** F_C(x,v) = (C(x) − v)
* **Viscous damping:** F_Γ(v) = −v (or −Γ(x)*v if spatially varying)
* **Obstacle force:** F_O(x) = −∂O/∂x
* Optional: stochastic term from diffusion

Total:

* a = wU*F_U + wC*F_C + wΓ*F_Γ + wO*F_O + b

Where:

* wU, wC, wΓ, wO, b are **learned parameters** (start small: 5–8 parameters total)

### Why this specific form

* It is **physics-legible** (each term is interpretable).
* It is **stable** with damping.
* It is **learnable** because parameters enter linearly (weights on known force components).

---

## 7) Minimal working model configuration (few inputs, maximal extensibility)

### What the first model uses (on purpose)

Start with exactly these three influences:

1. **Global directional pressure scalar** → produces a simple constant “push” term
   (Represents your directional imbalance signal in its simplest form.)

2. **One soft wall** (single obstacle) → one Gaussian-like barrier term

   * defined by {center offset in ticks, strength, width}
   * strength can be “static for now” or driven by one scalar

3. **Constant viscosity** Γ (single scalar)

Everything else is “plug-in ready” but not needed for v1.

### Why

* Lets you validate: loop timing, stability, UI, replay, prediction bookkeeping.
* If this scaffold is right, adding your rich bucketed features is incremental.

---

## 8) Prediction pipeline (t+1 … t+4 every tick)

### What happens per tick (deterministic sequence)

For each 500ms MarketFrame at time t:

1. **Compose fields**

   * Build U(i), C(i), Γ(i), O(i) from the current MarketFrame and any persistent obstacle state.

2. **Initialize particle**

   * x₀ = 0 (since grid is centered at spot at time t)
   * v₀ = estimated from last tick’s motion (or start at 0 initially)

3. **Roll out H steps**

   * Run the simulator forward H steps using the same fields (zero-order hold) for v1.
   * Output predicted positions: x̂(t+1..t+H)

4. **Emit predictions**

   * Convert x̂ to absolute predicted price levels.
   * Send to UI + store in a “prediction ledger” keyed by timestamp and horizon.

5. **When actual arrives**

   * As future ticks arrive, resolve the ledger:

     * compare predicted at (t+h) to actual at (t+h)
     * accumulate losses for online learning

### Why

* This is the simplest closed loop that produces meaningful visuals and gives you a learning signal.
* Zero-order hold is intentionally crude, but it makes the core loop correct and testable.

---

## 9) Online learning: tuning weights for forward propagation

### Objective

Minimize **multi-horizon prediction error**:

* Loss(t) = Σ_{h=1..H} α_h * ρ( x_actual(t+h) − x_predicted(t+h) )

Where:

* α_h increases with horizon (or is flat; pick one and keep it stable)
* ρ is a robust loss (Huber-style) to avoid a single jump blowing up the weights

### What we learn

* The weight vector θ = [wU, wC, wΓ, wO, b, …]

### Update strategy (online, stable, research-grade)

* Use **regularized online regression** on the *effective force contributions* observed during rollout.
* Maintain:

  * θ (weights)
  * a small curvature/scale estimate (for stable step sizes)
  * constraints/bounds to prevent unstable negative damping, etc.

### Why this approach

* You get a **continuous calibration loop** without needing a full differentiable simulator on day one.
* You preserve interpretability: when weights move, you know what changed (pressure vs walls vs drift).

### “No lookahead” guarantee

Learning only updates θ when the relevant future actuals have arrived; predictions at time t are produced from information available at time t.

---

## 10) Obstacle memory model (soft walls that erode/thicken)

### What

Represent walls as persistent entities with internal state:

* WallState:

  * center (absolute price or relative offset)
  * strength S(t)
  * width σ
  * decay/erosion rate
  * reinforcement term from new inputs

Each tick:

* S(t+1) = decay(S(t)) + reinforcement(inputs at t)

Obstacle field O(i) is generated from the current WallState set.

### Why

This captures your “GEX walls that erode or thicken” requirement and decouples:

* *detection* of walls (from features)
  from
* *dynamics* of walls (memory + decay + reinforcement)

Even with a single wall in v1, this architecture doesn’t change later.

---

## 11) GPU execution model (Metal) and why it’s shaped this way

### What runs on GPU

* Field sampling / interpolation
* Multi-step rollout (H steps, optionally with substeps)
* Generating render-ready buffers/textures for the UI

Metal compute is encoded with `MTLComputeCommandEncoder`, and the kernels are dispatched across the grid/particles. ([Apple Developer][4])

### Why GPU at all for v1

* Even if v1 is small, the end state is “many inputs → many fields,” and you want this to scale cleanly.
* GPU is also ideal for the visualization pipeline (heatmaps/field lines).

---

## 12) Visualization: “physics lab” UI (predicted vs actual + medium view)

### Required views

1. **Trajectory panel (time series)**

* Plot:

  * actual spot
  * predicted spot for each horizon (t+1..t+4)
  * optional confidence band per horizon
* Use Swift Charts for fast iteration and clean interaction. ([Apple Developer][2])

2. **Medium panel (the “physics view”)**
   A Metal-rendered view (MTKView embedded in SwiftUI) showing, at the current tick:

* x-axis = price offset (ticks)
* A curve or heat strip for U(x) (pressure landscape)
* Overlay for obstacle O(x) (wall bumps)
* The particle at x=0 and the predicted path x̂(t+1..t+H) drawn forward

MTKView is the canonical Metal-backed view for drawing on screen. ([Apple Developer][3])

3. **Debug overlay**
   Real-time text for:

* current weights θ
* per-horizon errors
* stability checks (NaN detection, weight bounds)

### Why this specific UI split

* The Charts panel answers: “Is the forecast good?”
* The Medium panel answers: “Why did it move (in this model’s language)?”

---

## 13) Correctness, stability, and testability (research-lab standards)

### Determinism

* Every tick produces the same output given the same inputs and weight snapshot.
* All random diffusion (if enabled) is driven by a seeded RNG recorded in the replay stream.

### Replay-first design

* Every MarketFrame is persisted to a log format that can be replayed offline.
* The engine can run:

  * in **live mode** (streaming)
  * in **replay mode** (fixed dataset)
  * in **unit test mode** (synthetic fields)

### Invariants and guardrails

* Weight bounds:

  * damping weight must never create “negative viscosity”
* Field sanity:

  * U and O should be finite and within expected normalized ranges
* Integrator sanity:

  * cap max acceleration/velocity per tick to avoid explosions during early learning

### Why

If you don’t do these early, you’ll never trust the model enough to iterate fast.

---

## 14) Extensibility: adding your rich feature set without touching physics

### The plug-in contract

Every new feature group must implement exactly one of these transformations:

* Feature → ΔU(i) (adds an attractor/repeller)
* Feature → ΔC(i) (adds drift/current)
* Feature → ΔΓ(i) (adds friction/absorption)
* Feature → ΔO(i) and/or WallState update (adds/modifies obstacles)

Then the engine:

* sums contributions
* applies weights
* runs the rollout

### Why

This prevents “feature spaghetti” and keeps the physics model stable while you add complexity.

---

## 15) Future-proof learning acceleration (GPU training when you want it)

You don’t need this for v1, but the architecture should keep the door open to **batched calibration**.

Apple’s **MPSGraph** exists specifically to “build, compile, and execute compute graphs” across GPU/CPU/Neural Engine, and Apple provides training-oriented patterns for it. ([Apple Developer][5])

### Why mention it

If/when you decide to tune a larger parameterization (many weights, nonlinearities), you’ll want a path that doesn’t require leaving the Apple stack.

---

# Implementation sequence (step-by-step “what + why”)

### Step 1 — Build the app harness (UI + render loop) [COMPLETE]

**What:** SwiftUI app with:

* a Chart panel (actual vs predicted)
* an MTKView panel (medium view)
* a control panel (start/stop/replay/seed/weight display)

**Why:** You need immediate visual feedback while you iterate on the model.

---

### Step 2 — Define the canonical fields and the minimal simulation state [COMPLETE]

**What:** Data structures for:

* Grid config (Δx, N)
* Fields: U, C, Γ, O
* Particle state: x, v
* Parameter vector θ
* Prediction ledger (stores predictions until actual arrives)

**Why:** This is the spine of the system; keep it stable while features evolve.

---

### Step 3 — Implement the minimal model with 3 influences [COMPLETE]

**What:** Compose fields from:

* directional push scalar
* single wall obstacle
* constant viscosity

Run rollout to H=4 and visualize.

**Why:** Proves the full loop: ingest → simulate → predict → display.

---

### Step 4 — Add prediction scoring + online weight updates [COMPLETE]

**What:** Once actual future values arrive, compute multi-horizon loss and update θ.

**Why:** This is “step two” in your plan—learning the propagation weights so the simulator becomes predictive rather than illustrative.

---

### Step 5 — Expand from scalar inputs to profile inputs [COMPLETE]

**What:** Replace the single directional push with a per-bucket profile to generate U(i) and/or C(i).

**Why:** This is where your existing bucketed feature tensor starts to matter, but the physics core doesn’t change.

---

### Step 6 — Add wall memory dynamics [COMPLETE]

**What:** Introduce persistent WallState updates (erosion/thickening), even if still only 1–3 walls.

**Why:** Adds the “soft obstacle” behavior you care about (and it’s visually obvious when it works).

---

[1]: https://swift.org/blog/swift-6.2-released/?utm_source=chatgpt.com "Swift 6.2 Released"
[2]: https://developer.apple.com/documentation/Charts?utm_source=chatgpt.com "Swift Charts | Apple Developer Documentation"
[3]: https://developer.apple.com/documentation/metalkit/mtkview?utm_source=chatgpt.com "MTKView | Apple Developer Documentation"
[4]: https://developer.apple.com/documentation/Metal/MTLComputeCommandEncoder?utm_source=chatgpt.com "MTLComputeCommandEncoder"
[5]: https://developer.apple.com/documentation/metalperformanceshadersgraph?utm_source=chatgpt.com "Metal Performance Shaders Graph"

---

# Medium Scope Visual Encoding Spec

*A “physics lab instrument panel” for your market‑as‑medium model (1D price particle + fields), built for live streaming + replay + rapid iteration.*

## Tooling baseline [COMPLETE]

* **Swift 6.2** app layer (UI + orchestration) ([Swift.org][1])
* **Metal + MetalKit MTKView** as the realtime visualization surface ([Apple Developer][2])
* **Swift Charts** for time-series truth vs forecast overlays (and it’s actively evolving; Apple publishes update notes) ([Apple Developer][3])
* Compute/encoding model uses standard Metal command encoders (and can align with newer Metal evolution like Metal 4) ([Apple Developer][4])

---

## 1) Purpose and “read in 2 seconds” goals [COMPLETE]

This view must answer, instantly:

1. **Where are the vacuums / attractors?** (potential minima)
2. **Where are the soft walls / obstacles?** (barriers, erosion state)
3. **What direction does the medium push?** (current/drift)
4. **What will the particle do next (t+1..t+H)?** (ghost forecast)
5. **How wrong was the last resolved forecast?** (continuous reality check)

This is not a decorative viz. It’s an *instrument*.

---

## 2) Canvas layout (single MTKView scene, fixed bands) [COMPLETE]

Inside the MTKView, render three horizontal bands with shared X axis:

### Band A — **Medium Field (≈65% height)**

Shows the “terrain” of forces: potential + obstacles + viscosity cues + current arrows.

### Band B — **Forecast Ladder (≈20% height)**

Shows the current tick’s predicted positions for **h=1..H** as a stacked ladder (so time doesn’t fight with the terrain’s Y axis).

### Band C — **Resolved Error Ladder (≈15% height)**

Always shows the **most recently fully-resolved origin frame** (e.g., t−H) with predicted vs actual per horizon, so you get a constant “prediction vs reality” comparison even in live mode.

> Why this layout: you avoid the classic problem of trying to use one Y-axis for both “energy height” and “time forward.” The ladder bands make horizons readable without compromising the field terrain.

---

## 3) Coordinate system (hard rule) [COMPLETE]

**X axis is always price offset in ticks**, centered on “spot/mid at origin time of the frame.”

* X range: [-NΔx, +NΔx] ticks (N is grid half-width; Δx = bucket size in ticks)
* The vertical bands **never rescale X** when volatility spikes; only user zoom changes X scale.

### Grid styling (instrument feel)

* Major vertical gridlines every **10 buckets**
* Minor gridlines every **1 bucket**
* Centerline at **x = 0** is thicker and labeled “Spot (t0)”
* Two thin dashed lines at ±(1 bucket) to visually anchor “near touch”

---

## 4) Visual encodings (exact mappings) [COMPLETE]

### 4.1 Potential / Vacuum field `U(x)` → “Terrain” [COMPLETE]

**What it should look like:** a valley/hill landscape.

**Encode as a signed height field** in Band A:

* Compute a display scalar `V(x)` that is the **effective potential** you want the human to “believe” (for v1: `V = normalize(wU·U + wO·O)`).
* Map to a terrain height:

  * `height(x) = midlineY − k * V(x)`
* Interpretation rule:

  * **Downward valleys = attraction/vacuum**
  * **Upward ridges = repulsion/barrier pressure**

**Fill + contour:**

* Fill under the terrain curve with subtle opacity (so it reads at a glance).
* Add **contour lines** (3–7 levels) so structure is visible even when the fill is faint.

**Why:** Valleys are the most intuitive visual metaphor for “pressure vacuum sucks price.”

---

### 4.2 Obstacle field `O(x)` → “Soft walls” [COMPLETE]

Obstacles must read as *distinct objects with identity*, not just “part of the terrain.”

**Render obstacles as vertical “gates” over the terrain:**

* Gate center at wall price level (in ticks)
* Gate width = wall σ (or bucket span)
* Gate intensity/opacity = current wall strength S(t)
* Gate edge style:

  * Hard, crisp edges (instrument-like)
  * Slight glow only when S(t) changes rapidly (wall thickening/erosion event)

**Erosion/thickening indicator (must-have):**

* Draw a small “wear meter” on each gate:

  * A thin inset bar whose fill represents **S(t) / S_max**
  * When S(t) is decaying, animate the bar subtly downward (no flashing)

**Why:** Your “GEX walls that erode or thicken” are *entities*. They need persistent visual semantics.

---

### 4.3 Current / Drift field `C(x)` → “Flow arrows” [COMPLETE]

Currents must be readable without clutter.

**Glyph rule:**

* Draw arrows along a fixed baseline inside Band A (not on the terrain surface).
* Arrow direction = sign(C(x))
* Arrow length = |C(x)| (clamped to avoid huge spikes)
* Arrow density:

  * One arrow per bucket (or every 2 buckets if crowded)
* Arrow alpha scales with confidence (if you have it later), otherwise constant.

**Why:** This reads like wind in a wind tunnel—your “market flow.”

---

### 4.4 Viscosity / Damping `Γ(x)` → “Drag haze” [COMPLETE]

Viscosity is a *resistance field*; it should feel like “thicker medium.”

**Encode as a translucent haze overlay in Band A:**

* Higher Γ(x) → higher local haze opacity
* Haze texture is subtle “striations” aligned horizontally (implies resistance to motion)

**Constraint:** Never let viscosity hide the terrain; cap haze opacity.

**Why:** Humans instantly read haze as “thicker / harder to move through.”

---

### 4.5 Particle state (spot) → “Probe” [COMPLETE]

The particle is your probe moving through the medium.

**Render:**

* A bright circle at x=0 (spot at origin time t0)
* A velocity arrow attached to the circle:

  * Direction = sign(v)
  * Length = |v| (clamped)
* A tiny “force vector” arrow (optional but recommended):

  * Direction = sign(a_net)
  * Length = |a_net|
  * This makes the model’s instantaneous intent visible even before price moves.

**Why:** This makes the sim legible: you see whether motion is inertia-driven vs force-driven.

---

## 5) Forecast encoding (no ambiguity) [COMPLETE]

### 5.1 Forecast Ladder (Band B): current tick predictions [COMPLETE]

Band B is a discrete, horizon-indexed instrument readout.

For each horizon step **h = 1..H**:

* Allocate one row.
* Draw:

  * A small label: `+0.5s`, `+1.0s`, … (derived from Δt)
  * A **predicted marker** at x = x̂(t0+h)
  * A faint vertical “drop line” from the marker to the band baseline (helps alignment)

Connect the predicted markers across horizons with a thin line so the forecast reads as a path.

**Time encoding rule:**

* Near horizons are higher opacity / thicker marker.
* Far horizons are lower opacity / smaller marker.
* This is the only “depth cue” you need.

**Why:** It makes t+1..t+H readable without turning the main field into a spaghetti plot.

---

### 5.2 Resolved Error Ladder (Band C): always show last resolved frame [COMPLETE]

Band C uses the same ladder layout, but shows **both predicted and actual** for one origin time that’s now fully known.

Per horizon row:

* Predicted marker at x̂
* Actual marker at x_actual
* A horizontal error bar between them
* An error number (ticks) that appears only if |error| exceeds a small threshold

**Why:** You get constant calibration feedback even in live streaming. The eye quickly learns whether your model is systematically early/late, underreacting, or overreacting.

---

## 6) Interaction and controls (instrument-grade) [COMPLETE]

These interactions are mandatory because they accelerate iteration:

1. **Hover/crosshair readout**

* When pointer hovers at X:

  * show U(X), O(X), C(X), Γ(X)
  * show ∂V/∂x (the local force direction)
* Also snap-highlight the nearest wall gate if within its width.

2. **Freeze / Step / Replay scrub**

* Freeze pauses the live stream **but keeps rendering** (so you can inspect).
* Scrub chooses a historical t0 and shows:

  * Band A fields at that t0
  * Band B forecast from that t0
  * Band C resolved comparison for that t0 (if available)

3. **Zoom X**

* Zoom is symmetric around x=0 (maintains the “spot-centered instrument” vibe)

---

## 7) Rendering order (layer stack) [COMPLETE]

Back → front, in Band A:

1. Background grid + axis labels
2. V(x) fill + contour lines (terrain)
3. Obstacle gates + wear meters
4. Viscosity haze
5. Current arrows
6. Particle + velocity + force arrow
7. Debug text overlay (toggle)

Band B:

* ladder grid → predicted markers → connecting line → labels

Band C:

* ladder grid → predicted+actual markers → error bars → labels

---

## 8) Extensibility rule (so you can add 100 signals later) [COMPLETE]

You will add many influences later; the viz must not collapse.

**Rule:** New inputs do not add new visual primitives by default.
They only change one of:

* terrain (V)
* gates (walls)
* flow arrows (C)
* haze (Γ)
* forecast path (x̂)

If a new factor needs visibility, it gets a **debug-only overlay channel**, not a permanent new glyph.

**Why:** Keeps the instrument readable forever.

---

[1]: https://swift.org/blog/swift-6.2-released/?utm_source=chatgpt.com "Swift 6.2 Released"
[2]: https://developer.apple.com/documentation/metalkit/mtkview?utm_source=chatgpt.com "MTKView | Apple Developer Documentation"
[3]: https://developer.apple.com/documentation/Charts?utm_source=chatgpt.com "Swift Charts | Apple Developer Documentation"
[4]: https://developer.apple.com/documentation/Metal/MTLComputeCommandEncoder?utm_source=chatgpt.com "MTLComputeCommandEncoder"

---

# ✅ What you added for “true real-time GEX flow” (and where it currently lives)

1. **Silver now has real options book microstructure outputs** (these are your “true flow” feeds):

* `silver.future_option_mbo.book_flow_1s` (mapped as stream key `options_flow`)

  * fields include `instrument_id`, `side`, `price_int`, `add_qty`, `pull_qty`, `fill_qty`, `window_end_ts_ns` 
* `silver.future_option_mbo.book_wall_1s` (mapped as stream key `options_wall`)

  * adds `depth_total` and `pull_rest_qty` on top of add/pull/fill-type fields 

2. **Silver’s `gex_surface_1s` already contains “flow-like” derivatives** (cheap to stream, fixed 25 rows/window):

* `gex_call_abs`, `gex_put_abs`
* `d1/d2/d3` for `gex_abs`, `gex`, `gex_imbalance_ratio` 

3. **Your current WebSocket HUD stream does *not* include any of the new flow streams**

* `frontend_data.json` defines only: `snap`, `wall`, `vacuum`, `physics`, `gex`, `bucket_radar` 
* DOCS also flags `options_wall` / `options_flow` as “TBD” in the frontend layer list 

So: the pipeline can produce real-time options flow, but the serving layer currently doesn’t expose it, and the client contracts don’t acknowledge it yet.

---

## A) Extend the Silver data pipeline (this is required)

**All computations that involve joins, aggregation, state, or normalization must live in Silver (or Gold), not in the serving layer.**
Serving should not become a second analytics engine.

### A1) Make `SilverComputeGexSurface1s` *actually emit* the new datasets every window [COMPLETE]

Even if the contract is written, treat it as “not done” until verified on disk.

**Implementation steps**

1. In `backend/src/data_eng/stages/silver/` locate the stage implementation for **`SilverComputeGexSurface1s`** (path called out in your README). 
2. Ensure this stage writes **three outputs per 1s window**:

   * `silver.future_option_mbo.gex_surface_1s`
   * `silver.future_option_mbo.book_wall_1s`
   * `silver.future_option_mbo.book_flow_1s` 
3. Ensure `book_wall_1s` includes **`depth_total` and `pull_rest_qty`** (these are essential for “wall memory”: thickness + erosion). 
4. Ensure `book_flow_1s` includes **`add_qty`, `pull_qty`, `fill_qty`** keyed by `(window_end_ts_ns, instrument_id, side, price_int)` (or tighter if you already do it; the point is: the dataset must exist and be consistent window-to-window). 

**Why pipeline (not serving):** this requires per-window aggregation over raw option MBO events; doing it in serving would create inconsistent results across replays and make performance unpredictable.

---

### A2) Add a strike-aligned *flow surface* for the underlying (this is the part that makes it usable by the physics app) [COMPLETE]

Right now `options_flow` / `options_wall` are **option-premium price space** (`price_int` = option price), keyed by `instrument_id`. That’s not directly drawable or usable as an obstacle field in **underlying tick space**.

Your physics app needs a **fixed, strike-grid surface** exactly like `gex_surface_1s` (25 rows per window, rel_ticks in multiples of 20 for ES). 

**Implementation steps**

1. Create a new Silver dataset (new file in `backend/src/data_eng/contracts/silver/` + new stage output) called something like:

   * `silver.future_option_mbo.gex_flow_surface_1s` (name it whatever, but it must be strike-grid aligned). 
2. Build it inside the **same stage** `SilverComputeGexSurface1s` (do not create a second pass unless you must), because that stage already has the inputs you need: instrument defs, futures spot anchor, and strike grid logic. 
3. For each 1s window:

   * Join `book_flow_1s` to `instrument_definitions` to get:

     * underlying strike (strike_price_int / strike_points)
     * right (call/put)
   * Aggregate flows to strike:

     * `add_qty_sum`, `pull_qty_sum`, `fill_qty_sum` per strike
     * also compute: **`reinforce = add_qty_sum - pull_qty_sum - fill_qty_sum`**
   * Join futures `spot_ref_price_int` and compute:

     * `rel_ticks = (strike_price_int - spot_ref_price_int) / TICK_INT`
     * enforce `rel_ticks % 20 == 0` for ES-style $5 strikes 
   * Emit exactly the 25 strike rows (same strike grid as `gex_surface_1s`); missing strikes must be present with zeros.

**Why this is non-negotiable:**
Without this surface, the Swift physics sim will be forced to do heavyweight joining/aggregation itself (instrument definitions, strike mapping, flow aggregation), and you’ll lose determinism across clients and replays.

---

### A3) Extend `gold.hud.physics_norm_calibration` to include the new flow metrics [COMPLETE]

You already normalize key physics metrics via `gold.hud.physics_norm_calibration`. 
Do the same for flow or your “wall thickening vs erosion” will be dominated by raw scale changes.

**Implementation steps**

1. Update the calibration builder (wherever you compute q05/q95 bands) to add quantile ranges for:

   * `flow_abs = add_qty_sum + pull_qty_sum + fill_qty_sum`
   * `flow_reinforce = add_qty_sum - pull_qty_sum - fill_qty_sum`
   * optionally `pull_rest_intensity = pull_rest_qty / (depth_total + eps)` aggregated to strike using `book_wall_1s` 
2. Recompute calibration on a representative period (same way you did for `gex_abs`, `wall_strength_log`, etc.). 
3. In the new `gex_flow_surface_1s`, output normalized versions (0..1 or -1..1) using the exact same `clip((val-q05)/(q95-q05))` pattern you already use. 

**Why pipeline:** calibration is inherently global/stateful. Serving must not own it.

---

## B) Update the serving layer (streaming only; no heavy math)

Now that Silver produces usable surfaces, serving’s job is to **expose** them safely.

### B1) Add surface gating via query param (required to not break the existing HUD) [COMPLETE]

Your frontend renderer has synchronization logic that assumes a known set of surfaces per batch. If you suddenly add more surfaces to every batch, you risk stalling the ring-buffer advance or breaking “allSurfacesReceived()”-style logic. 

**Implementation steps**

1. In `src.serving.main` (or wherever the WebSocket endpoint is implemented) add support for a query param like:

   * `?surfaces=snap,wall,vacuum,physics,gex,bucket_radar,gex_flow`
2. Default behavior must remain **exactly** the current 6 surfaces defined in `frontend_data.json`:
   `snap, wall, vacuum, physics, gex, bucket_radar` 
3. When `surfaces` is specified:

   * Only include those in the `batch_start.surfaces` list
   * Only send `surface_header` + Arrow payloads for those surfaces 

This allows:

* Web HUD stays stable (unchanged defaults)
* Swift Physics Lab requests the extra surfaces it needs

---

### B2) Stream the new strike-aligned flow surface [COMPLETE]

**Implementation steps**

1. Register the new Silver dataset (`gex_flow_surface_1s`) in the server’s dataset registry / stream mapping (where you map dataset → stream name).
   Your backend contract already uses stream mapping patterns like this. 
2. Implement streaming for that surface exactly like `gex`:

   * fixed row count per window (25)
   * Arrow IPC payload
   * keyed by `window_end_ts_ns`

---

### B3) Extend the existing `gex` stream payload to include the derivative fields (do this now) [COMPLETE]

This is basically free bandwidth (25 rows), and it gives you “instant momentum of the obstacle field.”

**Implementation steps**

1. In the serving transform for the `gex` surface, include these additional columns in the Arrow table:

   * `gex_call_abs`, `gex_put_abs`
   * `d1_gex_abs`, `d2_gex_abs`, `d3_gex_abs`
   * `d1_gex`, `d2_gex`, `d3_gex`
   * `d1_gex_imbalance_ratio`, `d2_*`, `d3_*` 
2. Do not change the existing column meanings; you’re only adding fields already produced by Silver.

**Why serving is OK here:** it’s not computing anything, just exposing already-computed Silver columns.

---

## C) Update the contracts + docs (yes, do it; this prevents future breakage) [COMPLETE]

You said “don’t write the schema,” so I’m not dumping JSON—but the engineer must update these artifacts.

**Implementation steps**

1. Update `frontend_data.json` to include:

   * the new optional surface(s) (at least the strike-level flow surface)
   * the extended `gex` fields list
     so the contract matches reality. 
2. Update `DOCS_FRONTEND.md`:

   * remove “TBD” for the new surfaces you’re actually streaming
   * document the `surfaces=` query param contract and default surface set
     (this doc is treated as “law” by engineers, and it’s currently inconsistent). 

---

## D) Verification steps (must be done before calling it “implemented”) [PARTIAL]

Use your repo’s own verification flow.

**Implementation steps**

1. Run the Silver pipelines to generate outputs:

   * futures + options pipelines as documented in README 
2. Run stream verification (`verify_websocket_stream_v1.py`) and integrity tests (`test_integrity_v2.py`) mentioned in README. 
3. Validate invariants:

   * `window_end_ts_ns` strictly increasing
   * `spot_ref_price_int > 0` when `book_valid = true`
   * For strike-aligned surfaces: `rel_ticks % 20 == 0` (ES $5 grid) 
4. Validate surface gating:

   * Default connection returns only the 6 original surfaces (so web HUD remains identical). 
   * Physics Lab connection with `surfaces=` receives the additional flow surface(s).

Status: `test_integrity_v2.py` PASS. `verify_websocket_stream_v1.py` connected but closed without a close frame (port 8000 already in use); rerun after restarting backend if needed.

---

## What this unlocks immediately in the physics app

With **one new strike-aligned flow surface** plus the **extended gex derivatives**, you can now implement:

* **Soft obstacle thickening/erosion**: obstacle strength update driven by `flow_reinforce_norm` (and/or `d1_gex_abs`) at each strike rel_ticks.
* **Memory**: wall persists and changes, rather than being static “heatmap wallpaper.”
* **Stable bandwidth**: still fixed-row, tick-native surfaces (no giant per-instrument streams needed for the sim).
