# Spymaster Upgrade Spec: Spatiotemporal Liquidity Physics + Lookahead

This is the **single source of truth** for upgrading the existing working pipelines (**future_mbo**, **future_option_mbo**) from a single-frame `liquidity_velocity` heatmap into a **spatiotemporal physics system** with:

- **Obstacles** expressed in standard fluid / porous-media terms: **density ρ**, **permeability κ**, **hydraulic resistance R**, and **effective viscosity ν**.
- **Pressure-gradient force** expressed as a signed field **g(x,t) = −∂p/∂x**, aligned to market direction.
- **Spatial awareness** across **±N ticks** so each cell responds to what is happening above/below it (collapse / vacuum / barrier growth).
- **Causal lookahead** (1–30s) computed from these fields and streamed in real time.

**No code belongs in this document.** It is written for an LLM coding agent to implement **exactly**. Formulas, columns, and ordering are non‑negotiable.

---

## 0) Ground truth: why the pipeline is Bronze → Silver → Gold (and why it must stay that way)

The current system is correct because each layer has a distinct invariance contract.

### Bronze = canonical raw events (no “model semantics”)
- Purpose: deterministic ingest of MBO events with stable types.
- Allowed transformations: casting, NULL_PRICE edge handling, session-window filtering.
- Disallowed transformations: any feature engineering, smoothing, normalization, or “physics.”

**Reason:** if features leak into Bronze, replay and auditability die. Every downstream model becomes irreproducible.

### Silver = state reconstruction + physically meaningful primitives
- Purpose: reconstruct book state and compute the primitives that correspond to “measurable quantities” on a grid.
- Silver outputs are the **state variables** we need for physics:
  - Depth at price level at window start/end (`depth_qty_start`, `depth_qty_end`)
  - Flux components (`add_qty`, `pull_qty_total`, `fill_qty`)
  - Persistence proxies (`depth_qty_rest`, `pull_qty_rest`)
  - Stable coordinates: `rel_ticks` (spot anchored), `side` (A/B)

**Reason:** obstacles, viscosity, pressure require **state**, not just events.

### Gold = normalized fields designed for streaming + downstream inference
- Purpose: map Silver primitives to dimensionless, comparable fields on a lattice.
- Gold already computes:
  - `u(x,t) = liquidity_velocity = add_intensity − pull_intensity − fill_intensity`

Gold is where the physics fields live (multi-scale, derivatives, obstacle/pressure fields).

---

## 1) Coordinate system and sign conventions (fixed)

### 1.1 Space
- Spatial coordinate: **x = rel_ticks** (spot anchored)
- Grid spacing: **Δx = 1 tick** (ES: $0.25)
- Window cadence: **Δt = 1 second**

### 1.2 Side and direction
- `side = 'B'` is bid book.
- `side = 'A'` is ask book.

**Market direction convention:**
- Positive force means **upward price pressure**.
- Negative force means **downward price pressure**.

### 1.3 Compression vs rarefaction (gas dynamics terminology)
Define the base field:
- **u(x,t)** = `liquidity_velocity(x,t)` (already computed in Gold)

Interpretation:
- **u > 0** = liquidity **compression** (book thickening / wall building)
- **u < 0** = liquidity **rarefaction** (book thinning / vacuum opening)

---

## 2) Temporal awareness: multi-scale causal “liquidity waves” (mandatory)

All temporal filtering is **causal** (real-time safe) and defined per cell `(x, side)`.

### 2.1 Four fixed EMA time constants
Maintain four exponential moving averages of `u`:
- `u_ema_2`   (τ = 2s)
- `u_ema_8`   (τ = 8s)
- `u_ema_32`  (τ = 32s)
- `u_ema_128` (τ = 128s)

EMA update rule:
- `EMA_τ(t) = EMA_τ(t−1) + α_τ * (u(t) − EMA_τ(t−1))`
- `α_τ = 1 − exp(−Δt / τ)` with `Δt = 1s`

### 2.2 Wave (band) decomposition
Define three band components:
- `u_band_fast = u_ema_2  − u_ema_8`
- `u_band_mid  = u_ema_8  − u_ema_32`
- `u_band_slow = u_ema_32 − u_ema_128`

Define wave energy (scalar intensity of structural change):
- `u_wave_energy = sqrt(u_band_fast² + u_band_mid² + u_band_slow²)`

This is the required replacement for “just showing u.” It encodes how liquidity is changing across time scales.

### 2.3 Temporal derivatives (computed from the smoothed field, not raw)
Compute temporal derivatives from `u_ema_2`:
- `du_dt     = u_ema_2(t) − u_ema_2(t−1)`
- `d2u_dt2   = du_dt(t) − du_dt(t−1)`

---

## 3) Spatial awareness: neighborhood coupling across ±N ticks (mandatory)

This section implements your requirement:

- A level at +$0.50 matters because it is strengthening while +$0.25 is depleting.
- We must detect whether depletion extends further above (collapse / vacuum) or terminates into a strengthening wall.

This is done by enforcing **two spatial scales** and by computing **local prominence** (relative strengthening) and **coherent rarefaction corridors**.

### 3.1 Two fixed spatial neighborhoods
Everything spatial uses **spot-anchored x** and is computed per side per frame.

Define two neighborhoods:
- **Near neighborhood:** `N_near = 16` ticks (±16)
- **Far neighborhood:**  `N_far  = 64` ticks (±64)

### 3.2 Fixed Gaussian kernels (normalized to sum=1)
Use Gaussian kernels on integer ticks with truncation at ±N.

- Near kernel σ: `σ_near = 6`
- Far  kernel σ: `σ_far  = 24`

Compute:
- `u_near(x,t) = GaussianSmooth(u_ema_8(·,t),  σ_near, ±N_near) evaluated at x`
- `u_far(x,t)  = GaussianSmooth(u_ema_32(·,t), σ_far,  ±N_far)  evaluated at x`

### 3.3 Spatial derivatives (from the near-smoothed field)
Compute the spatial gradient and curvature on `u_near`:
- `du_dx(x,t)   = (u_near(x+1,t) − u_near(x−1,t)) / 2`
- `d2u_dx2(x,t) = u_near(x+1,t) − 2*u_near(x,t) + u_near(x−1,t)`

### 3.4 Local prominence = “relative strengthening vs neighborhood”
Define a spatial prominence (high-pass) field:
- `u_prom(x,t) = u_near(x,t) − u_far(x,t)`

Interpretation:
- `u_prom > 0`: this level is strengthening **relative to its neighborhood**
- `u_prom < 0`: this level is depleting **relative to its neighborhood**

This is the formal definition of “+$0.50 is an obstacle because it is strengthening while +$0.25 is depleting.”

### 3.5 Coherent rarefaction corridor (collapse detection)
Define rarefaction magnitude:
- `r(x,t) = max(0, −u_near(x,t))`

Define fixed exponential distance weights:
- `w_k = exp(−k / 16)` and renormalize within each sum.

Upward corridor (asks above spot):
- `corridor_up(t)   = Σ_{k=1..N_far} w_k * r(+k,t)`

Downward corridor (bids below spot):
- `corridor_down(t) = Σ_{k=1..N_far} w_k * r(−k,t)`

These are mandatory per-second scalars. They answer:
- “Is depletion extending further above (vacuum)?” → high `corridor_up`
- “Is depletion extending further below?” → high `corridor_down`


---

## 4) Obstacles and viscosity: porous-media mapping (fixed definitions)

This section defines the “obstacle” in standard terms and makes it explicitly depend on:
1) depth (mass), 2) persistence (resting), and 3) reinforcement vs neighborhood.

### 4.1 Liquidity density (ρ)
Per cell `(x, side)`:
- `rho(x,t) = log(1 + depth_qty_end(x,t))`

`log(1+·)` is mandatory to compress the heavy-tailed depth distribution.

### 4.2 Persistence fraction (φ)
Per cell `(x, side)`:
- `phi_rest(x,t) = depth_qty_rest(x,t) / (depth_qty_end(x,t) + 1)`

This is the persistence proxy: what fraction of depth is “resting” (>500ms).

### 4.3 Persistence-weighted liquidity wave (institutional weighting)
Define persistence-weighted velocities:
- `u_p(x,t)      = phi_rest(x,t) * u_ema_8(x,t)`
- `u_p_slow(x,t) = phi_rest(x,t) * u_ema_32(x,t)`

These are the only `u` variants used to define obstacles and pressure. They explicitly de-emphasize flickering liquidity.

### 4.4 Obstacle strength (Ω): “wall mass × reinforcement”
Define obstacle strength:
- `Omega(x,t) = rho(x,t) * (0.5 + 0.5*phi_rest(x,t)) * (1 + max(0, u_p_slow(x,t)))`

Interpretation:
- High depth + high persistence yields a high baseline obstacle.
- If that obstacle is **being reinforced** (positive slow persistence-weighted velocity), it strengthens further.

### 4.5 Obstacle prominence (Ω_prom): “this wall is stronger than neighbors”
Compute the same near/far smoothing on `Omega`:
- `Omega_near(x,t) = GaussianSmooth(Omega(·,t), σ_near=6,  ±N_near=16)`
- `Omega_far(x,t)  = GaussianSmooth(Omega(·,t), σ_far=24,   ±N_far=64)`

Define prominence:
- `Omega_prom(x,t) = Omega_near(x,t) − Omega_far(x,t)`

This is the formal definition of a “wall” / “obstacle” in this system. It captures *relative* strengthening in space.

### 4.6 Effective viscosity (ν), resistance (R), and permeability (κ)
Define **hydraulic resistance**:
- `R(x,t) = 1 + Omega_near(x,t) + 2*max(0, Omega_prom(x,t))`

Define **effective viscosity**:
- `nu(x,t) = R(x,t)`  (ν is resistance/viscosity in this 1D medium)

Define **permeability**:
- `kappa(x,t) = 1 / nu(x,t)`

This is porous-media standard form: high resistance ↔ low permeability.

---

## 5) Pressure-gradient force: market-aligned g(x,t) = −∂p/∂x (fixed)

Pressure gradient must push price **in the correct direction**.

### 5.1 Side-specific pressure gradient (g_side)
Using the persistence-weighted mid-scale velocity `u_p`:

- Bid side:
  - `pressure_grad(x,t) = +u_p(x,t)` for rows with `side='B'`

- Ask side:
  - `pressure_grad(x,t) = −u_p(x,t)` for rows with `side='A'`

Interpretation:
- Building bids (positive u) increases upward pressure.
- Building asks (positive u) increases downward pressure (therefore sign flip).

### 5.2 Directional field on spot-anchored x (g_dir)
For force and forecasting, construct a directional field on signed x:

- For x > 0 (above spot): `g_dir(x,t) = pressure_grad(+x, side='A')`
- For x < 0 (below spot): `g_dir(x,t) = pressure_grad(−|x|, side='B')`
- For x = 0: exclude from force integrals.

This makes `g_dir > 0` mean upward pressure everywhere on the lattice.

---

## 6) The collapse vs wall diagnostic (mandatory)

This produces a deterministic answer to your mental model:

- If +$0.25 is depleting and +$0.50 is strengthening → move into +$0.50 then stall.
- If +$0.25 and +$0.50 and higher levels are depleting → full collapse / vacuum → continuation likely.

### 6.1 Find the first meaningful wall above and below
Compute within ±N_far (64 ticks):

Above spot (asks, x=+1..+64):
- `W_up = max_{k=1..64} Omega_prom(+k,t)`
- Threshold: `T_up = 0.60 * W_up`
- First wall distance:
  - `D_up(t) = min{k ≥ 1 : Omega_prom(+k,t) ≥ T_up}`
  - If none meets it, set `D_up = 64`

Below spot (bids, x=−1..−64):
- `W_down = max_{k=1..64} Omega_prom(−k,t)`
- Threshold: `T_down = 0.60 * W_down`
- First wall distance:
  - `D_down(t) = min{k ≥ 1 : Omega_prom(−k,t) ≥ T_down}`
  - If none meets it, set `D_down = 64`

### 6.2 Vacuum mass into the first wall (rarefaction integral)
Use the same fixed weights `w_k = exp(−k/16)` renormalized.

Above:
- `Vacuum_up(t) = Σ_{k=1..D_up−1} w_k * max(0, −u_near(+k,t))`

Below:
- `Vacuum_down(t) = Σ_{k=1..D_down−1} w_k * max(0, −u_near(−k,t))`

### 6.3 Wall reinforcement at the first wall
Above:
- `Reinforce_up(t) = max(0, u_p_slow(+D_up,t))`

Below:
- `Reinforce_down(t) = max(0, u_p_slow(−D_down,t))`

### 6.4 Run propensity score (the scalar explanation a trader understands)
Compute:
- `RunScore_up(t)   = Vacuum_up(t)   − Reinforce_up(t)`
- `RunScore_down(t) = Vacuum_down(t) − Reinforce_down(t)`

Interpretation:
- Large positive `RunScore_up`: vacuum corridor is strong and the first wall is not strengthening → continuation likely.
- Negative `RunScore_up`: the first wall is being reinforced → stall/mean-revert near that wall.

These two scalars MUST be produced every second and streamed with the forecast for explainability.


---

## 7) Lookahead: price particle in a viscous medium (mandatory, causal, stream-time)

This is the production lookahead engine. It is deterministic and derived from the fields above.

### 7.1 State variables
Maintain per second:
- Spot tick index `S(t)` from the snap surface.
- Price velocity in ticks/sec: `v(t) = S(t) − S(t−1)`.

Particle position in **relative ticks** during forecast:
- `δ(0) = 0` (start at current spot)
- `δ(h)` evolves in ticks for h=1..H.

### 7.2 Force and damping are evaluated at shifted coordinates (critical)
During forecast, the particle moves through the spatial field. Therefore force and viscosity MUST be evaluated at **shifted coordinates**.

Fixed kernels:
- Force radius: `X_force = 64`
- Force weights: `w_force(k) = exp(−k² / (2*12²))`, normalized over k=1..64.
- Damping radius: `X_damp = 16`
- Damping weights: `w_damp(k) = exp(−k² / (2*6²))`, normalized over k=1..16.

At forecast step h with position δ(h):

Force:
- `F(h) = Σ_{x ∈ {−64..−1,+1..+64}} w_force(|x|) * kappa(x−δ(h), t) * g_dir(x−δ(h), t)`

Local viscosity:
- `NuLocal(h) = Σ_{x ∈ {−16..−1,+1..+16}} w_damp(|x|) * nu(x−δ(h), t)`

This is what makes “wall at +$0.50 strengthening” behave like a wall as the particle approaches it.

### 7.3 Discrete-time dynamics (1s step)
Forecast horizon: `H = 30` seconds.

Dynamics:
- `a(h) = β * F(h) / (1 + NuLocal(h))`
- `v(h+1) = (1 − γ) * v(h) + a(h)`
- `δ(h+1) = δ(h) + v(h+1)`

Hard stability constraints:
- Clamp `v(h)` to `[-8, +8]` ticks/sec.
- Clamp `δ(h)` to `[-80, +80]` ticks.

### 7.4 Forecast outputs (streamed every second)
For each horizon `h ∈ {1..30}` output:
- `predicted_tick_delta(h) = round(δ(h))`
- `predicted_spot_tick(h) = S(t) + predicted_tick_delta(h)`
- `confidence(h)` (below)

### 7.5 Confidence (explainable and tied to collapse/wall diagnostics)
Define:
- `C0 = tanh( |F(0)| / (1 + NuLocal(0)) )`

Directional gating using run scores:
- If `predicted_tick_delta(h) > 0`: `Gate = tanh(max(0, RunScore_up(t)))`
- If `predicted_tick_delta(h) < 0`: `Gate = tanh(max(0, RunScore_down(t)))`
- Else: `Gate = 0`

Final confidence:
- `confidence(h) = C0 * Gate`

High confidence only occurs when:
- force exists AND
- there is a vacuum corridor in the predicted direction AND
- the first wall is not being reinforced.

---

## 8) Options: how they enter the physics (strictly as viscosity, not thrust)

Options liquidity is treated as an **external obstacle field** that increases viscosity near round strikes (pinning). It does **not** directly contribute to pressure-gradient thrust in this upgrade.

### 8.1 Required Silver change (options)
The options Silver `depth_and_flow_1s` MUST include:
- `depth_qty_rest` (resting quantity >500ms)

Without `depth_qty_rest`, options obstacles cannot be persistence-weighted and are not usable as “real walls.”

### 8.2 Options obstacle field on strike lattice
On the options strike buckets (rel_ticks multiples of 20), compute:

- `rho_opt = log(1 + depth_qty_end_opt)`
- `phi_rest_opt = depth_qty_rest_opt / (depth_qty_end_opt + 1)`
- `u_opt_ema_8`, `u_opt_ema_32` using the same EMA rules (per strike bucket)
- `u_opt_p_slow = phi_rest_opt * u_opt_ema_32`

Define:
- `Omega_opt = rho_opt * (0.5 + 0.5*phi_rest_opt) * (1 + max(0, u_opt_p_slow))`

### 8.3 Spread options obstacles onto futures tick lattice
For each strike bucket located at x_s (multiple of 20 ticks), spread into tick space:

Fixed kernel:
- spread radius: ±24 ticks
- `σ_spread = 8`
- `K_spread(d) = exp(−d² / (2*8²))`, normalized over d=−24..+24

Compute:
- `Omega_opt_spread(x,t) = Σ_s Omega_opt(x_s,t) * K_spread(x − x_s)`

### 8.4 Combine into total obstacle and viscosity
Define:
- `Omega_total = Omega_fut + 0.70 * Omega_opt_spread`

Then recompute:
- `Omega_total_near`, `Omega_total_far`, `Omega_total_prom`
- `nu_total`, `kappa_total`

All lookahead computations use the total fields (`nu_total`, `kappa_total`, `Omega_total_prom`).


---

## 9) Data products and schema: exactly what must be added

This upgrade adds columns. It does **not** create alternate datasets or parallel “v2” tables. The existing Gold tables are extended.

### 9.1 Futures Gold: `gold.future_mbo.physics_surface_1s` (add these columns)
Grain: `(window_end_ts_ns, rel_ticks, side)`.

**Existing (keep):**
- `window_end_ts_ns`, `spot_ref_price_int`, `rel_ticks`, `side`
- `add_intensity`, `pull_intensity`, `fill_intensity`
- `liquidity_velocity` (u)

**Add (mandatory):**

Temporal multi-scale:
- `u_ema_2`, `u_ema_8`, `u_ema_32`, `u_ema_128`
- `u_band_fast`, `u_band_mid`, `u_band_slow`
- `u_wave_energy`
- `du_dt`, `d2u_dt2`

Spatial awareness:
- `u_near`, `u_far`, `u_prom`
- `du_dx`, `d2u_dx2`

Obstacle / viscosity:
- `rho`, `phi_rest`
- `u_p`, `u_p_slow`
- `Omega`, `Omega_near`, `Omega_far`, `Omega_prom`
- `nu`, `kappa`

Pressure:
- `pressure_grad` (side-specific; directional assembly happens in the server)

### 9.2 Options Gold: `gold.future_option_mbo.physics_surface_1s` (add these columns)
Grain: current options Gold grain + required persistence columns.

**Existing (keep):**
- `window_end_ts_ns`, `spot_ref_price_int`, `rel_ticks`, `liquidity_velocity`

**Add (mandatory):**
- `rho_opt`, `phi_rest_opt`
- `u_opt_ema_8`, `u_opt_ema_32`
- `u_opt_p_slow`
- `Omega_opt`

This requires options Silver to provide `depth_qty_end` and `depth_qty_rest` at the strike bucket level.

### 9.3 New streaming surface: `forecast`
Add a new streamed surface called `forecast` (Arrow IPC) emitted once per second.

Grain: `(window_end_ts_ns, horizon_s)`.

Rows:
- Horizons 1..30:
  - `horizon_s` (int)
  - `predicted_spot_tick` (int)
  - `predicted_tick_delta` (int)
  - `confidence` (float)

- A required horizon 0 diagnostic row (same schema):
  - `horizon_s = 0`
  - `RunScore_up`, `RunScore_down` (float)
  - `D_up`, `D_down` (int)

---

## 10) Calibration (mandatory): fit β and γ from 20 days / first 3 hours

The lookahead model has two scalar parameters:
- `β` (force coupling)
- `γ` (velocity damping)

These are fit from data and then loaded by the stream server.

### 10.1 Training window and data
- Use the last **20 trading days** of MBO.
- Use only **RTH 09:30–12:30 ET** (first 3 hours).
- Do not regime-split. Fit globally.

### 10.2 Fit method (deterministic linear regression)
For every second t:
- Spot tick index: `S(t)`
- Velocity: `v(t) = S(t) − S(t−1)`
- Compute `F(0)` and `NuLocal(0)` at δ=0 using the definitions above.
- Features:
  - `x1 = v(t)`
  - `x2 = F(0) / (1 + NuLocal(0))`
- Target:
  - `y = v(t+1)`

Fit:
- `y ≈ a*x1 + b*x2`

Map to physics parameters:
- `γ = 1 − clip(a, 0, 1)`
- `β = max(0, b)`

### 10.3 Evaluation (required)
Evaluate on held-out days:
- Horizons: 1s, 2s, 5s, 10s, 20s, 30s
- Metrics:
  - Hit rate on sign(ΔS)
  - MAE in ticks
  - Confidence calibration: bucket by `confidence`, verify monotonic hit rate

The product is considered launchable only if confidence is monotonic on held-out days.

---

## 11) Non-goals (strictly out of scope until this spec is complete)

These are not implemented until the above is finished and validated:

1) Using a rigid-body physics engine (SpriteKit / Matter.js / Box2D) as the forecasting mechanism.  
   - Production lookahead is the deterministic model defined here.

2) Regime classifiers, volatility-conditioned parameter sets, or “if VIX then …” logic.  
   - You stated regimes do not matter materially; do not add complexity.

3) Any model that uses non-causal features (future leakage).  
   - Everything here is strictly causal.

---

## 12) Implementation order (strict) and acceptance criteria

An LLM coding agent MUST implement in this order. Do not reorder.

### Phase A — Schema + contracts first (must compile end-to-end)
1. Update Avro contracts for the modified Gold tables and the new forecast surface.
2. Update `datasets.yaml` to match the contracts.
3. Update `futures_data.json` so it describes the new columns and transformations.
4. Update `README.md` streaming protocol section to include the `forecast` surface.

**Acceptance:** pipelines run without schema mismatch and the stream server can send Arrow IPC for all surfaces.

### Phase B — Options Silver persistence (must exist before options viscosity)
1. Add `depth_qty_rest` to options Silver `depth_and_flow_1s`.
2. Propagate the new field through options Gold.

**Acceptance:** options Gold can compute `phi_rest_opt` and `Omega_opt`.

### Phase C — Futures Gold physics fields (must match formulas exactly)
1. Implement temporal EMAs and wave fields.
2. Implement spatial smoothing (near/far), prominence, and derivatives.
3. Implement obstacle fields (rho, phi_rest, Omega, Omega_prom, nu, kappa).
4. Implement pressure_grad sign conventions.

**Acceptance:** Gold table contains every required column with correct shapes and no NaNs.

### Phase D — Stream-time forecasting (must be real-time safe)
1. In the stream server, assemble directional fields (`g_dir`) and viscosity fields.
2. Compute `D_up`, `D_down`, `RunScore_up`, `RunScore_down`.
3. Run the 1..30s particle forecast with shifted-coordinate sampling.
4. Stream the `forecast` surface every second.

**Acceptance:** frontend receives forecasts without stalling; predicted path reacts to walls and vacuums.

### Phase E — Frontend visualization (must communicate the story)
1. Add a field selector for rendering:
   - `liquidity_velocity` (u)
   - `u_wave_energy`
   - `nu` (viscosity)
   - `pressure_grad`
2. Render the forecast path into the right margin.
3. Display `RunScore_up/down` and `D_up/down` on-screen as a compact debug HUD.

**Acceptance:** a user can point at a run and explain it using vacuum + first-wall distance + reinforcement.

### Phase F — Calibration + validation (must be repeatable)
1. Implement the β/γ regression fit using 20 days, 09:30–12:30 ET.
2. Store parameters in config and load them in the stream server.
3. Validate monotonic confidence on held-out days.

**Acceptance:** higher confidence forecasts hit more often; low confidence forecasts are explicitly de-emphasized.



---

## 13) Physics-engine visualization sandbox (after launch, not used for forecast)

This is the only allowed way to “feed the field into a physics engine” without contaminating the forecast logic.

### Engine choice (fixed)
- Use **Matter.js** in the browser (same runtime as the existing Three.js UI).
- Do not use the engine for prediction. It is a **visual sandbox** driven by the deterministic fields.

### Mapping from Spymaster fields → Matter.js parameters (fixed)
Represent the system in 2D where Y is price (ticks) and X is time (already in the UI):

1) **Price particle**
- Body: circle.
- Vertical position: current spot tick.
- Vertical velocity: `v(t)` from the snap surface.
- Air friction (damping): set `frictionAir ∝ (γ + NuLocal(0) / (1 + NuLocal(0)))`.

2) **Walls (obstacles)**
- For each tick level x in the rendered range:
  - Create a thin static rectangle at that Y level.
  - Rectangle “hardness” is proportional to `Omega_prom(x,t)`:
    - restitution = 0 (no bounce)
    - friction = clamp(Omega_prom / (1 + Omega_prom), 0, 1)
  - Collision filtering:
    - Walls above spot use ask-side `Omega_prom`.
    - Walls below spot use bid-side `Omega_prom`.

3) **Pressure-gradient force**
- Each render frame, apply a vertical force to the particle:
  - `F_engine ∝ F(0)` (the same net force scalar from the deterministic model)
- Do not apply per-wall forces; the deterministic net force already integrates the field.

### Update cadence (fixed)
- Physics engine step: 60 Hz.
- Field refresh: once per second (on each streamed tick).
- Between ticks: hold fields constant.

This sandbox exists only to make the “price particle in viscous medium with growing/eroding obstacles” intuition immediately legible.
