Below is a concrete, step‑by‑step implementation guide for an LLM coding agent to extend the existing **future_mbo** and **future_option_mbo** pipelines from “1‑second net liquidity velocity” into a **spatiotemporal, multi‑scale physics field** with **obstacles (walls / permeability / viscosity)** and **pressure (pressure gradient force)**, plus **short‑horizon lookahead** suitable for real‑time streaming.

This is written to align with your current architecture (Bronze → Silver → Gold, Arrow streaming, Frontend2 grids).  

---

## 1) Why the pipelines are built in 3 stages (and why that matters for “physics”)

### Bronze = “canonical raw events”

**Why:** Bronze exists to make raw MBO usable and reproducible:

* cast types once (int64), fix NULL_PRICE edge cases, filter to a fixed session window.
* keep it “as close to exchange feed as possible” so you can rebuild everything deterministically.

This is crucial for physics/ML because you need **stable semantics** and you don’t want “model features” leaking into ingestion rules. 

### Silver = “stateful reconstruction + physically meaningful primitives”

**Why:** Physics requires state (books), coordinates (space), and conserved quantities (depth/flow). Silver is where you:

* reconstruct book state each 1s window
* compute:

  * **book_snapshot_1s** (best bid/ask, mid, last trade, spot anchor)
  * **depth_and_flow_1s** (depth start/end, adds/pulls/fills, resting/persistent liquidity)
* produce **stable coordinate systems**:

  * `rel_ticks` (spot-anchored)
  * `rel_ticks_side` (side-anchored)

This is exactly analogous to converting raw sensor readings into a **field on a grid**.

### Gold = “derived fields (normalized), designed for streaming”

**Why:** Gold turns Silver primitives into normalized signals that:

* are comparable across price levels and time
* are robust to depth scale changes
* stream cleanly as a surface (grid)

You currently compute normalized intensities and:

* `liquidity_velocity = add_intensity - pull_intensity - fill_intensity`
  which is a **net local source/sink of liquidity**. That is already “physics-like” because it is a signed field on a spatial grid. 

---

## 2) What features exist today (explicitly) and what they mean in “physics terms”

### Future (underlying) today

**Silver depth_and_flow_1s (per 1s × price level × side)**

* `depth_qty_start`, `depth_qty_end`
* `add_qty`, `pull_qty_total`, `fill_qty`
* `depth_qty_rest` and `pull_qty_rest` (persistence proxy, >500ms)
* coordinates: `rel_ticks`, `rel_ticks_side`

**Gold physics_surface_1s**

* `add_intensity = add_qty / (depth_qty_start + 1)`
* `pull_intensity = pull_qty_total / (depth_qty_start + 1)`
* `fill_intensity = fill_qty / (depth_qty_start + 1)`
* `liquidity_velocity = add_intensity - pull_intensity - fill_intensity`

**Interpretation:**

* Think of each price level as a cell on a 1D lattice.
* `liquidity_velocity(x,t)` is the **signed “rate of change” of obstacle mass** (building vs eroding) normalized by local depth.
* It is not price velocity; it’s **medium reconfiguration velocity**.

### Options today

Options Silver/Gold mirror futures but on a **strike bucket grid**:

* `strike_price_int` bucketed by $5, mapped to `rel_ticks` (multiples of 20)
* `depth_qty_*`, adds/pulls/fills per strike bucket
* Gold computes the same normalized `liquidity_velocity`
* streaming currently aggregates C/P and bid/ask into a single net velocity per strike level. 

**Interpretation:**

* Options surface is a coarse “external structure field” on top of the underlying lattice.

---

## 3) Define the physics vocabulary you want (standard terms) and map them to your data

You asked for **standard fluid/aero terms**. Use these as your canonical terms:

### 3.1 Spatial coordinate

* **x** = `rel_ticks` (spot-anchored ticks)
* Grid spacing: Δx = 1 tick ($0.25)

### 3.2 Fields you will maintain (and stream)

1. **Liquidity density** (analogous to mass / density in a medium)

   * Symbol: **ρ(x,t)**
   * Source: depth at level
   * Candidate: `ρ = log1p(depth_qty_end)` (compress heavy tails)

2. **Permeability** (porous-media standard term)

   * Symbol: **κ(x,t)**
   * Intuition: “how easily price can pass through this level”
   * Candidate: `κ = 1 / (1 + ρ)` or `κ = exp(-ρ)`

3. **Viscosity** (kinematic viscosity, standard symbol ν)

   * Symbol: **ν(x,t)** (nu)
   * Intuition: “friction / damping produced by obstacles”
   * Source: density + persistence + strengthening
   * Candidate:

     * persistence fraction **φ(x,t)** = `depth_qty_rest / (depth_qty_end + 1)`
     * strengthening signal **s(x,t)** = `EMA_slow(liquidity_velocity)` clipped
     * `ν = ν0 + νρ*ρ + νφ*φ + νs*max(0, s)`

4. **Pressure** (standard symbol p) / **pressure gradient force** (−∂p/∂x)

   * In fluids: acceleration is driven by pressure gradients.
   * In your market mapping: **pressure is the directional “push” produced by *changes* in obstacles** (velocity of liquidity building/eroding), not the obstacle mass itself.
   * You will compute a **signed pressure gradient field** directly from liquidity_velocity with the correct market sign convention:

     * For **bids below spot**, building liquidity is supportive → pushes price up.
     * For **asks above spot**, building liquidity is resistive → pushes price down.
   * Define:

     * `g(x,t) = pressure_gradient(x,t)`
     * If you have both side and rel_ticks:

       * `g = +u` for bid side
       * `g = -u` for ask side
     * Equivalent in a single formula if x is signed:

       * `g(x,t) = -sign(x) * u(x,t)` (u is liquidity_velocity on spot-anchored x)

5. **Force on the price particle** (external force term)

   * Symbol: **F(t)**
   * Compute as a **distance‑weighted integral of pressure gradient** near the price:

     * `F(t) = Σ_x w(|x|) * g(x,t)`
     * w(|x|) decays with distance (Gaussian or exponential)

6. **Drag / damping** (standard)

   * Symbol: **D(t)** or γ
   * Derived from local viscosity near x=0:

     * `ν_local(t) = Σ_{|x|≤K} wν(|x|) * ν(x,t)`

This gives you a clean “physics API”:

* **ρ(x,t)** obstacle mass
* **κ(x,t)** permeability (inverse obstacle)
* **ν(x,t)** viscosity (damping)
* **g(x,t)** pressure gradient
* **F(t)** net pressure force
* **ν_local(t)** local damping

---

## 4) The core upgrade: from single‑frame velocity to spatiotemporal awareness + derivatives

Your target is: “not just u(x,t), but how it evolves across time scales and across x”.

### 4.1 Multi‑timeframe (temporal) representation (causal, streaming friendly)

Implement **cascaded EMAs** (or IIR filters) per cell (x,side):

* Choose 3–5 time constants (seconds): e.g. τ = {2, 8, 32, 128}
* Maintain:

  * `u_ema_2`, `u_ema_8`, `u_ema_32`, `u_ema_128`

Then define **band‑pass / wave components** (this is your “wave propagation” feel):

* `u_band_fast = u_ema_2 - u_ema_8`
* `u_band_mid  = u_ema_8 - u_ema_32`
* `u_band_slow = u_ema_32 - u_ema_128`

Define **wave energy**:

* `E = sqrt(u_band_fast^2 + u_band_mid^2 + u_band_slow^2)`
* This highlights “sudden structural shifts” vs noise.

**Why this is the right first step:**

* It’s incremental (perfect for real-time).
* It’s interpretable (fast vs slow “liquidity wave”).
* It gives you a causal lookback structure that naturally supports short-horizon extrapolation.

### 4.2 Temporal derivatives (per cell)

Compute first/second temporal derivatives (discrete):

* `du_dt = u(t) - u(t-1)`
* `d2u_dt2 = du_dt(t) - du_dt(t-1)`

Also compute derivatives for the band components:

* `d(u_band_fast)/dt`, etc.
  These derivatives are “shock detectors”.

### 4.3 Spatial derivatives (across x)

On each 1s frame, after you have the array u(x) (for each side or in combined signed x):

* **gradient:** `du_dx(x) = (u(x+1) - u(x-1)) / 2`
* **curvature / Laplacian:** `d2u_dx2(x) = u(x+1) - 2u(x) + u(x-1)`

Do the same for obstacle density ρ(x,t) and/or viscosity ν(x,t):

* `dν_dx`, `d2ν_dx2`

**Why:**
Spatial gradients distinguish:

* a “uniform build” (less predictive)
* a “front” (liquidity cliff approaching the touch)
* a “wall” (local maxima in ρ and ν)

### 4.4 Feature extraction: “walls”, “voids”, and “fronts”

Make these explicit, because traders think this way and it maps to physics obstacles.

Define per frame:

* **Wall strength** (obstacle):

  * `O(x,t) = ρ(x,t) * (1 + λφ*φ(x,t)) * (1 + λs*max(0, u_ema_32(x,t)))`
* **Void strength**:

  * `V(x,t) = -min(0, u_ema_8(x,t))` (eroding zones)
  * and/or `κ(x,t)` (high permeability)

Then detect:

* nearest top‑N walls above and below spot
* nearest top‑N voids above and below spot
* a “front” if `|du_dx|` is large near touch

These become both:

* additional heatmap overlays (spatial awareness)
* inputs into the forecast engine (lookahead)

---

## 5) Forward projection: a simple, explainable “price particle” simulator (no ML required)

You want “some lookahead based on momentum” with an understandable API.

### 5.1 Define the price particle state

At each second t:

* current spot coordinate: `x0(t) = 0` in rel_ticks (by definition)
* but you track actual spot tick index separately in the UI (already done).

Define particle velocity state:

* `v_p(t)` (this is **price velocity** in ticks/sec), computed from spot changes:

  * `v_p(t) = spot_tick(t) - spot_tick(t-1)`

### 5.2 Compute net force from pressure gradient field

Use your pressure gradient definition:

* `g(x,t) = +u_ema_8(x,t)` for bids, `g = -u_ema_8(x,t)` for asks
  (or `g = -sign(x)*u_ema_8` if you have signed x)

Then net force:

* `F(t) = Σ_{|x|≤Xmax} w(|x|) * g(x,t)`
* Use two radii to get “near” vs “far” forces:

  * `F_near` with σ=8 ticks
  * `F_far`  with σ=32 ticks

### 5.3 Compute local damping from viscosity

* `ν_local(t) = Σ_{|x|≤K} wν(|x|) * ν(x,t)`
* Ensure ν_local is bounded away from 0.

### 5.4 Discrete-time dynamics (interpretable)

Use a damped driven system (standard):

* `v_p(t+1) = (1 - γ*ν_local) * v_p(t) + β * F(t)`
* `x_p(t+1) = x_p(t) + v_p(t+1)`

Where:

* γ is damping strength
* β is force coupling
* clamp `(1 - γ*ν_local)` into [0,1] for stability

### 5.5 Produce a forecast for horizons 1–30s

Run the recurrence forward N steps using:

* either “frozen field” assumption: u_ema stays constant
* or a simple extrapolation:

  * `u_ema_8_pred(h) = u_ema_8 + h * du_ema_8_dt` (clipped)
  * recompute F each step

Output:

* `predicted_spot_tick_delta[h]` for h=1..H
* `predicted_direction[h] = sign(delta)`
* `confidence[h] = |F| / (ν_local + ε)` (simple, explainable)

**This is the minimum viable lookahead engine**:

* It is physics‑inspired
* It is causal and incremental
* It gives a right‑margin prediction path immediately

---

## 6) Where options fit: “external obstacles” and “pinning viscosity”

Your options surface is currently **coarse strike buckets**. Use it as an **external potential / obstacle field** rather than pretending it directly predicts direction.

### 6.1 Add missing primitives to options Silver (if needed)

To make options contribute to viscosity consistently, you want the same base primitives as futures:

* `depth_qty_rest` (resting quantity >500ms) per strike bucket
  (today you have `pull_qty_rest` but not necessarily `depth_qty_rest` in the JSON doc)

**Why:**
Options “walls” are only meaningful if they’re persistent, not flickering.

### 6.2 Compute options obstacle strength on the strike lattice

For each strike bucket x (multiples of 20 ticks):

* `ρ_opt = log1p(depth_qty_end_opt)`
* `φ_opt = depth_qty_rest_opt / (depth_qty_end_opt + 1)`
* `u_opt_slow = EMA_32(liquidity_velocity_opt)`

Define:

* `O_opt(x,t) = ρ_opt * (1 + λφ_opt*φ_opt) * (1 + λs_opt*max(0,u_opt_slow))`

### 6.3 Couple options obstacles into the futures viscosity field

Map strike buckets into the futures tick lattice by “spreading”:

* For each strike x_s:

  * distribute O_opt(x_s) into nearby ticks with a kernel (e.g., σ=6 ticks)
* Then:

  * `ν_total(x,t) = ν_fut(x,t) + λ_opt * O_opt_spread(x,t)`

**Interpretation:**
Options liquidity walls create “macro viscosity bumps” at round strikes → **pinning / slowing**.

### 6.4 (Optional later) directional options pressure

If/when you stop aggregating away right/side, you can try:

* call‑heavy build vs put‑heavy build as directional bias
  But don’t make this v1; keep v1 as “options = obstacles, not thrust.”

---

## 7) Concrete implementation plan (what files to touch, what to add, in what order)

This is written so an LLM coding agent can execute it mechanically.

### Step 0 — Lock the schema decisions first

**Goal:** decide exactly which new columns you will stream.

**Add to futures gold `physics_surface_1s` (or create a new dataset if you prefer):**

* Base:

  * `rho` (float32)
  * `phi_rest` (float32)
  * `permeability_kappa` (float32)
  * `viscosity_nu` (float32)
* Multi-scale:

  * `u_ema_2`, `u_ema_8`, `u_ema_32` (float32)
  * `u_band_fast`, `u_band_mid` (float32)
  * `u_wave_energy` (float32)
* Derivatives:

  * `du_dt`, `d2u_dt2` (float32)
  * `du_dx`, `d2u_dx2` (float32)
* Pressure gradient:

  * `pressure_grad` (float32)  // g(x,t)

**Add to options gold similarly (at strike buckets):**

* at least: `rho_opt`, `phi_rest_opt`, `viscosity_nu_opt`, `u_ema_8_opt`, `pressure_grad_opt` (even if pressure isn’t used yet)

**Why schema first:**
Everything downstream (Avro, datasets.yaml, futures_data.json, streaming, frontend parsing) depends on it. 

---

### Step 1 — Update Avro contracts + datasets.yaml + futures_data.json

**Agent actions:**

1. Modify:

* `backend/src/data_eng/contracts/gold/future_mbo/physics_surface_1s.avsc`
* `backend/src/data_eng/contracts/gold/future_option_mbo/physics_surface_1s.avsc`

2. Ensure `backend/src/data_eng/config/datasets.yaml` matches column set.

3. Update `futures_data.json` to accurately describe:

* which fields exist in each layer
* transformations (including new ones)

4. Update `README.md` “Data products” and “Streaming protocol” sections if the stream changes.  

**Acceptance check:**
Pipeline runs without schema mismatch and produces Arrow tables with the new columns.

---

### Step 2 — Extend Silver if you need more primitives (especially for options)

**Agent actions:**

* Confirm options Silver outputs enough to compute `phi_rest_opt` (rest fraction).
* If not present, add `depth_qty_rest` to options `depth_and_flow_1s` (same concept as futures).
* If you add it, remember: update Avro + datasets.yaml + futures_data.json + README.

**Why:**
You can’t define viscosity/obstacles without persistence.

---

### Step 3 — Implement the new Gold computations (futures + options)

**Agent actions (futures):**

* In `backend/src/data_eng/stages/gold/future_mbo/compute_physics_surface_1s.py`:

  1. Compute base fields: ρ, φ, κ, ν
  2. Compute causal EMAs per (rel_ticks, side):

     * store last EMA state (offline: via groupby+rolling; online: via dict/array state)
  3. Compute band components and energy
  4. Compute du_dt (needs previous u for each cell)
  5. Compute spatial derivatives per frame:

     * convert rows to a dense array for each side
     * compute du_dx, d2u_dx2
     * write back to rows

**Agent actions (options):**

* Mirror logic but on 41 strike buckets.
* Keep it cheap; it’ll be tiny.

**Critical constraint:**
Make the implementation incremental‑friendly:

* Even if your gold batch job is offline today, write it so it can be reused in a real-time path.

---

### Step 4 — Add a real-time “forecast” computation in the stream server

Your forecasts should be computed **in the server at stream time**, not baked into gold, because:

* you may tune parameters live
* you want the same logic in real-time and replay

**Agent actions:**

* In `backend/src/serving/velocity_streaming.py`:

  * Maintain last spot tick index and last price velocity `v_p`
  * From the current frame’s gold fields, compute:

    * `pressure_grad` field (or use the one you stream)
    * `F_near`, `F_far`
    * `ν_local`
  * Run the discrete dynamics forward for horizons (e.g., 5s, 10s, 30s):

    * output an Arrow table `forecast` with:

      * `window_end_ts_ns`
      * `horizon_s`
      * `predicted_spot_tick`
      * `predicted_spot_price` (optional)
      * `confidence`

**Streaming protocol update:**

* Add a new surface name: `"forecast"`
* Or add a JSON message type for forecast (but stick to Arrow for speed/consistency)

Update README streaming protocol accordingly. 

---

### Step 5 — Frontend: render new fields + render forecast in the prediction margin

Your frontend already reserves a right margin (`PREDICTION_MARGIN`) but doesn’t draw predictions.

**Agent actions:**

1. Update `frontend2/src/ws-client.ts`:

* Add parsing for new columns in `VelocityRow` (or define a richer type).
* Add `onForecast` callback.

2. Add a simple UI “field selector”:

* velocity (u)
* wave energy (E)
* viscosity (ν)
* pressure gradient (g)

3. Update `VelocityGrid` to support multiple channels:

* Either:

  * store multiple fields in RGBA channels (R=u, G=E, B=ν, A=1)
  * OR allocate multiple textures (cleaner but more code)

4. Implement a `PredictionLine` similar to `SpotLine`:

* draw predicted path starting at “now” and extending into the right margin

5. (Optional but powerful) show predicted zones:

* if you also forecast `u_band_fast`, draw a faint heatmap into the right margin.

---

## 8) Calibration/training plan (uses your 20 days, stays interpretable)

You said regimes don’t matter much; good. Treat calibration as fitting a small number of physics parameters.

### 8.1 Parameters to fit (keep it small)

* EMA time constants (or just choose fixed τ)
* weights:

  * β (force coupling)
  * γ (damping coupling)
  * ν weights (νρ, νφ, νs)
  * λ_opt (options viscosity coupling)
* kernel widths (σ near, σ far)

### 8.2 Targets

For horizons h ∈ {1,2,5,10,30} seconds:

* classification: sign(Δspot_tick(h))
* regression: Δspot_tick(h)

### 8.3 Fit method (fast + robust)

* Start with ridge regression / logistic regression on:

  * F_near, F_far, ν_local, wave_energy_near, wall_distance_near, etc.
* Or fit β and γ directly by minimizing forecast error (simple grid search).

### 8.4 Evaluation

* Use walk-forward by day:

  * train on N-1 days, test on the held-out day
* Track:

  * directional hit rate
  * MAE in ticks
  * “confidence calibration” (do high-confidence forecasts hit more often?)

This keeps the product explainable while still data‑driven.

---

## 9) “Native physics engine” option (if you really want an engine API)

If the goal is **an intuitive simulation and API**, not necessarily best prediction, then yes: you can adapt a standard physics engine.

### Swift-native suggestion (Apple-friendly)

* Use **SpriteKit** physics:

  * price = a dynamic circular body
  * obstacles = static thin rectangles at price levels (walls)
  * viscosity = linear damping on the particle (SpriteKit has this)
  * pressure = force applied each timestep

**Why it’s attractive:**

* trivial API
* great visuals
* runs extremely fast on Apple Silicon

**Caution:**

* rigid-body collision physics is not the same as a calibrated forecast model.
* treat this as a visualization/sandbox, while your “forecast engine” remains deterministic math.

### Web suggestion (if staying Angular)

* Matter.js or planck.js (Box2D) can do the same.
* Same caution applies.

---

## 10) Practical acceptance criteria (what “done” looks like)

The agent should ship this in milestones, each with clear pass/fail:

### Milestone A — Spatiotemporal fields exist and stream

* Gold surface includes ν, pressure_grad, EMAs, band components.
* Frontend can toggle and view at least:

  * liquidity_velocity
  * wave_energy
  * viscosity

### Milestone B — Forecast surface exists and renders

* Server sends `forecast` each second.
* Frontend renders predicted spot path into right margin.
* Forecast updates smoothly in replay.

### Milestone C — Basic calibration works

* A small offline script can fit β/γ (or regression weights) from 20-day lookback (first 3 hours).
* The server loads these weights from config.

