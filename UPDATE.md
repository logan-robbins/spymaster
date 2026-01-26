# Bucket‑Native Market Physics Radar
*A technically exhaustive architecture + specification (WHAT + WHY, with formulas where needed).*

## 0) North Star

We want a **real‑time “fluid medium” visualization** over a standard TradingView candle/line chart where:

- **Price is a particle moving through a medium**.
- The medium is defined by **futures microstructure** (resting depth, cancellations, erosion) plus **options boundary forces** (GEX).
- The visualization is **bucket‑native** (e.g., 2‑tick / $0.50 buckets), because that matches what traders can actually read in real time.
- We always show **above and below spot simultaneously**, because:
  - while price is rising, **vacuum/erosion below** may be forming a “return channel,” and vice versa.
- Over time, we see the **medium evolve** (build, erode, stiffen) rather than just flashing independent snapshots.
- Direction/“forward projection” is ultimately driven by **ML outputs**, but we allow a **temporary placeholder gate** (derivative heuristic) that can be replaced without changing the UI math.

This builds on your current tick‑native system: 1 column = 1 second, anchored to a spot reference, streamed via WebSocket as Arrow IPC and rendered in WebGL. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}


---

## 1) Data Provenance (What comes from what)

### 1.1 Futures MBO → Medium (microstructure)
The “medium” is futures‑only:

- **`wall_surface_1s`**: resting depth + book dynamics per tick level (side, depth_qty_rest, add/pull/fill, derivatives). :contentReference[oaicite:2]{index=2}
- **`vacuum_surface_1s`**: **computed from `wall_surface_1s`** + normalization calibration. This is *not* options‑derived. :contentReference[oaicite:3]{index=3}
- **`physics_surface_1s`**: computed from `wall_surface_1s + vacuum_surface_1s` as an “ease of movement” score (0..1) and signed directional variant. :contentReference[oaicite:4]{index=4}

### 1.2 Futures Options MBO + OI → External Force (macro boundary conditions)
The “external field” is options‑derived:

- **`gex_surface_1s`**: computed from 0DTE options MBO + open interest + instrument definitions, aligned to the futures spot reference. Grid is **±60 points in 5‑point steps** (ES: every 20 ticks). :contentReference[oaicite:5]{index=5}

### 1.3 Streaming contract
Frontend currently receives the following streams:

- `snap` (spot anchor + mid line)
- `wall` (depth_qty_rest)
- `vacuum` (vacuum_score)
- `physics` (physics_score, physics_score_signed)
- `gex` (gex_abs, gex_imbalance_ratio, rel_ticks multiples of 20) :contentReference[oaicite:6]{index=6}

**Important constraint**: the current HUD stream omits several “internal” wall/vacuum fields (e.g., depth derivatives, pull/add, erosion). That affects what can be computed purely client‑side. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}


---

## 2) Rename / Mental Model Taxonomy (So “everything is physics” stays clear)

Instead of naming each stream as “physics,” use a hierarchy:

### 2.1 “Physics Engine” (top)
**Market Physics Engine = Frame + Medium + Forces + Predictors**

- **Frame (Reference)**: where we are in price/time coordinates.
- **Medium (Futures)**: what the fluid is made of (depth, erosion).
- **Forces (Options)**: external potential (gamma landscape).
- **Predictors (ML)**: forward probability / path expectation.

### 2.2 First‑order “primitive fields” (measured/derived but still interpretable)
These are the primitives we bucketize:

1) **Obstacle Mass** (aka *Liquidity Density*): from futures resting depth (`depth_qty_rest`). :contentReference[oaicite:9]{index=9}  
2) **Erosion / Withdrawal** (aka *Vacuum*): `vacuum_score` (0..1) computed from pull/add, pull intensity, wall erosion, pull/add acceleration. :contentReference[oaicite:10]{index=10}  
3) **Ease / Permeability** (optional primitive): `physics_score` or a custom ease derived from (1) and (2). :contentReference[oaicite:11]{index=11}  
4) **External Stiffness / Potential**: from options GEX (`gex_abs`, `gex_imbalance_ratio`). :contentReference[oaicite:12]{index=12}

### 2.3 Second‑order “UI fields” (what you render)
UI fields are not “new data,” they’re **structured views** of primitives:

- **Resistance / Blockedness Field**: “how solid is this bucket”
- **Vacuum / Cavitation Field**: “how hollow / unstable is this bucket”
- **Mobility / Flow Ease Field**: “how easily can price traverse”
- **External Potential Field**: “macro pin/repel stiffness”
- **Directional Gate**: “how much to emphasize above vs below” (ML‑driven)

This gives a stable mental model:  
**Medium primitives → UI fields → rendering + ML gating**.


---

## 3) Bucket Space Definition (The canonical grid)

### 3.1 Parameters (configurable, but choose sane defaults)
- **Tick size**: ES = 0.25. :contentReference[oaicite:13]{index=13}  
- **Bucket size**: `BUCKET_TICKS = 2` (=$0.50)  
- **History length**: `T = 60` seconds (or more)  
- **Vertical extent**: `H = 31` buckets (recommended)  
  - center row is `bucket_rel = 0` (spot bucket)
  - 15 buckets above, 15 below

> Note: you originally said “30 buckets high.” Using **31** avoids ambiguity about the “middle row.” If you must use 30, treat center as a seam between two rows; the math is still possible, but readability is worse.

### 3.2 Coordinate conversions (formulas only)

#### Tick index
Let:
- `TICK_INT = 250000000` (scaled integer per tick for ES). :contentReference[oaicite:14]{index=14}  
- `spot_ref_tick = spot_ref_price_int / TICK_INT`
- each wall/vacuum/physics row provides `rel_ticks`

Then absolute tick index for a row:
```

abs_tick = spot_ref_tick + rel_ticks

```

#### Bucket index
```

bucket_abs = floor(abs_tick / BUCKET_TICKS)

```

#### Anchor bucket (spot row)
To ensure **middle row tracks the real spot**, the best anchor is `mid_price_int` (not currently streamed, but exists in backend snapshot schema). :contentReference[oaicite:15]{index=15}

If/when `mid_price_int` is available:
```

mid_tick    = mid_price_int / TICK_INT
spot_bucket = floor(mid_tick / BUCKET_TICKS)
bucket_rel  = bucket_abs - spot_bucket

```

Temporary fallback (less ideal) if only `mid_price` is available:
```

mid_tick ≈ mid_price / TICK_SIZE
spot_bucket = floor(mid_tick / BUCKET_TICKS)

```

**WHY**: your current renderer treats `spot_ref_price_int` as canonical for alignment stability, but the bucket grid’s *semantic center* should be the **true mid/spot line**, otherwise “center row” and “spot line” drift apart. :contentReference[oaicite:16]{index=16}


---

## 4) Bucketization (Turning tick‑native surfaces into bucket‑native fields)

We bucketize each primitive field into a fixed `H` rows around spot.

### 4.1 General bucket aggregation operator

For a raw per‑tick value `x(rel_ticks)`:

Let `B(b)` be the set of ticks that fall into bucket `b` (two ticks wide).

Unweighted aggregation:
```

X_bucket(b) = mean_{r ∈ B(b)} x(r)

```

Depth‑weighted aggregation (often better for vacuum):
```

X_bucket(b) = ( Σ_{r ∈ B(b)} w(r) * x(r) ) / ( Σ_{r ∈ B(b)} w(r) + ε )

```

Common choice:
- `w(r) = log1p(depth_qty_rest(r))` so vacuums inside “real walls” matter more.

### 4.2 Optional vertical convolution (recommended)
Even with 2‑tick buckets, the surface can still be noisy. Apply a small symmetric kernel `K` across neighboring buckets:

```

X_smooth(b) = ( Σ_{k=-m..m} K(k) * X_bucket(b+k) ) / ( Σ K(k) )

```

Good first kernels:
- triangular: K = [1,2,3,2,1]
- small gaussian‑like: K = [1,4,6,4,1]

**WHY**:
- makes the “medium” look like a continuum
- prevents “checkerboard micro flicker”
- solves sparse grids like GEX (section 7)

---

## 5) Primitive Fields (WHAT they mean and WHAT they become)

### 5.1 Obstacle Mass Field (from `wall`)
**Input**: `depth_qty_rest` (per tick level), rendered today as intensity = log1p(depth_qty_rest). :contentReference[oaicite:17]{index=17}

Bucket mass:
```

W(b) = log1p( Σ_{r ∈ B(b)} depth_qty_rest(r) )

```

Normalize to 0..1 (use calibration quantiles if done server‑side):
```

W_norm = clip((W - q05_W) / (q95_W - q05_W), 0, 1)

```

**WHY**:
- A wall is “mass/terrain.”
- For “solid boxes,” mass is your strongest base signal.

### 5.2 Vacuum / Cavitation Field (from `vacuum_score`)
**Input**: `vacuum_score` in [0,1], where higher means more “liquidity eroding.” :contentReference[oaicite:18]{index=18}

**True definition (important mental model)**: vacuum is futures‑only and is computed from the futures wall dynamics:

From backend:
- `pull_intensity_rest = pull_qty_rest / (depth_start + EPS)`
- `pull_add_log = log((pull_rest + EPS) / (add + EPS))`
- `wall_erosion = max(-d1_depth_qty, 0)`
- `d2_pull_add_log` (acceleration)
- then `vacuum_score = mean(norm(pull_add_log), norm(log1p(pull_intensity_rest)), norm(log1p(wall_erosion/(depth_start+EPS))), norm(d2_pull_add_log))` :contentReference[oaicite:19]{index=19}

Bucket vacuum:
```

V(b) = mean_{r ∈ B(b)} vacuum_score(r)

```

**WHY**:
- Vacuum is the *mechanism* by which “fake walls” vanish.
- Vacuum appearing on the opposite side (e.g., below while price rises) is exactly the “reversal channel formation” you care about.

### 5.3 External Potential / Stiffness Field (from `gex`)
**Input**: `gex_abs` and `gex_imbalance_ratio`, available only at 5‑point strike grid (ES: 20 ticks). :contentReference[oaicite:20]{index=20}

Bucket mapping:
- one GEX row corresponds to every **10 buckets** (because 20 ticks / (2 ticks per bucket) = 10 buckets)

Normalize (server‑side preferred; calibration includes gex_abs). :contentReference[oaicite:21]{index=21}

Bucket GEX stiffness:
```

G_raw(b) = gex_abs_norm at strike buckets; 0 elsewhere
G(b)     = conv(G_raw, K_gex)   // vertical convolution spreads influence

```

**WHY**:
- GEX should behave like a “macro stiffness ridge,” not sparse dots.
- We want it to shape corridors and barriers even between strikes.

---

## 6) Second‑Order UI Fields (the bucket matrices you actually render)

We define **three core UI matrices** (each `T × H`), plus an ML gate.

### 6.1 Resistance / Blockedness Matrix (solidness)
This is the field you described as “blocked 0..5.”

Start with base resistance:
```

R0(b) = clamp( a_w * W_norm(b) + a_g * G(b), 0, 1 )

```

Then apply destructive erosion by vacuum:
```

R_eff(b) = R0(b) * (1 - V(b))^γ

```

Typical behavior:
- if vacuum is high, walls “hollow out” quickly
- if vacuum is low, walls stay solid

Quantize to 0..5 for display:
```

BlockedLevel(b) = round(5 * R_eff(b))   // 0..5
BlockedAlpha(b) = BlockedLevel(b) / 5

```

**WHY**:
- “Solid boxes should be impassable” is implemented by making **solidness** a function of mass + stiffness, and requiring vacuum to erode it before traversal becomes likely.
- The exponent `γ` is a tunable “fragility” parameter: higher γ = vacuum punches cleaner corridors.

### 6.2 Vacuum / Cavitation Matrix (explicitly visible)
Even though vacuum erodes resistance, you explicitly want to *see* it.

We render vacuum as its own channel:
```

Cavitation(b) = V(b)^p

```

Where `p` controls contrast (e.g., p=1.5 makes only strong vacuum “glow”).

**WHY**:
- If vacuum is only implicit (just “less red”), you lose the ability to read “vacuum forming below to pull price back down.”
- Making cavitation visible preserves both sides’ information simultaneously.

### 6.3 Mobility / Corridor Matrix (optional but powerful)
Mobility is the “ease of movement,” which can be derived from resistance:

```

Mobility(b) = 1 - R_eff(b)

```

Or, as a baseline, reuse the backend’s physics score bucketed:
- `physics_score` already combines vacuum + erosion + inverse wall strength. :contentReference[oaicite:22]{index=22}

**WHY**:
- Mobility highlights corridors directly (“where price can actually flow”).
- It’s also the natural feature target for ML outputs later (predicting where the corridor will form next).

### 6.4 Directional Gate (ML‑driven emphasis, not hiding)
We need the “colors must SWITCH” behavior without losing the opposite side.

Define a **gate** that controls *emphasis* (brightness/opacity scaling), not the existence of the field:

- `GateUp(t)` and `GateDown(t)` in [0..1]
- constraint: `GateUp + GateDown = 1` (or close)

Render scaling:
- for buckets **above** spot (`bucket_rel > 0`): multiply intensities by `S_above = minVis + (1-minVis) * GateUp`
- for buckets **below** spot (`bucket_rel < 0`): multiply intensities by `S_below = minVis + (1-minVis) * GateDown`

Where `minVis` is a floor like 0.25–0.40 so the “other side” never disappears.

**WHY**:
- This produces the intuitive “switching” you want when direction flips, but still lets you see contrarian formation (vacuum below while moving up).
- It also provides a clean interface to replace heuristics with ML.

**Temporary placeholder** (explicitly a placeholder):
- until ML exists, you can derive `GateUp/GateDown` from a smoothed spot velocity sign.
- the UI math does not change when ML replaces it.

**ML target interface** (final):
- a model produces `P(up_next)` and `P(down_next)` for the next Δt window (1s or 500ms)
- set `GateUp = P(up_next)`, `GateDown = P(down_next)`

This aligns with your existing non‑visual ML feature vector dataset `radar_vacuum_1s` (~196 features) that already computes `approach_dir` and derivatives for microstructure inference. :contentReference[oaicite:23]{index=23}


---

## 7) Rendering Semantics (Color + alpha that matches the physics)

We want:
- black background
- “solid” cells visually block price
- vacuum visible as “low pressure”
- direction emphasis switchable without losing info

### 7.1 Color system (pressure‑style)
Use **pressure conventions**:

- **High pressure / high resistance** → warm (red/orange)
- **Low pressure / vacuum / cavitation** → cool (blue/cyan)
- Opacity encodes solidity (0..5)

Conceptually per cell (bucket b, time t):
- `alpha = BlockedAlpha(b,t) * S_side(t)`
- `red   = alpha` (resistance)
- `blue  = Cavitation(b,t) * S_side(t)` (vacuum glow)
- final color is an additive/over blend depending on your renderer (implementation detail)

**WHY**:
- Red = “compressed / blocked”
- Blue = “cavitated / unstable”
- Both can coexist: you can have a red wall with a blue edge (vacuum forming inside a wall).

### 7.2 “Spot line should only move through black corridors”
This becomes a calibration statement:

- A bucket at BlockedLevel 4–5 means: *“statistically unlikely to be traversed without an erosion/break event.”*
- When it *is* traversed, that event becomes a **training label** (“breakthrough”), not a failure.

**WHY**:
- In live markets, “impossible barriers” do break.
- Your system becomes stronger if those breaks are explicit anomalies that feed model training and parameter tuning.

### 7.3 Above + below are always shown
- We render both sides every frame.
- Gate only changes emphasis.

**WHY**:
- This preserves the “reversal channel formation” readability you described.


---

## 8) Time Evolution (How the medium evolves across the 60s matrix)

You explicitly want to “show the evolution over time” and have layers destructively affect each other.

There are **two valid evolution models**; you can support both:

### 8.1 Snapshot history (simplest)
Each second, compute `R_eff`, `V`, `G`, etc for that second and store as a column.
- The 60×H matrix is a pure historical record.

**WHY**:
- maximum faithfulness
- easier debugging

### 8.2 Stateful medium (recommended for “solidness” intuition)
Maintain a latent “material field” `Z` that persists and evolves:

Let:
- `Z_t(b)` = stored solidity state at time t (0..1)
- `R0_t(b)` = base resistance input at time t (mass + gex)
- `V_t(b)` = vacuum at time t

Update each window (formulas only):
```

Build:   Z'  = (1-α) * Z_{t-1} + α * R0_t
Erode:   Z'' = Z' * (1 - β * V_t)^γ
Clamp:   Z_t = clip(Z'' , 0, 1)

```

Then:
- `BlockedLevel = round(5 * Z_t)`
- `Cavitation = V_t^p`

**WHY**:
- “Walls persist unless eroded” matches trader intuition.
- Prevents unrealistic flicker where a wall appears/disappears instantly.
- Makes corridor formation visually continuous (fluid medium behavior).

This complements your existing “dissipation/persistence” concept used in the renderer. :contentReference[oaicite:24]{index=24}


---

## 9) Directionality (Hard part) — solved by separating “physics” from “attention”

The hardest conceptual trap is conflating:
- **what is physically present** (medium state above/below)
with
- **what is currently relevant** (where price is likely to go next)

This spec solves it by separation:

1) **Physics fields** (`R_eff`, `V`, `G`) are always bidirectional and always drawn.
2) **Directional attention** is a single scalar gate that can be ML‑driven:
   - it controls emphasis and “switching”
   - it never deletes information

**WHY**:
- As you said: while moving up, the downward vacuum formation matters.
- The attention model is the correct home for ML, not the raw physics.

---

## 10) ML Roadmap Integration (end goal: micro‑trading, 500ms)

### 10.1 Model inputs (already present)
You already have an ML‑oriented feature vector dataset:
- `radar_vacuum_1s` (~196 features), including directional features + d1/d2/d3 derivatives and `approach_dir`. :contentReference[oaicite:25]{index=25}

You also have:
- per‑window GEX values with d1/d2/d3 in backend (even if not streamed). :contentReference[oaicite:26]{index=26}

### 10.2 Model outputs (define the contract now)
To future‑proof the UI, define a minimal model output schema:

**A) Directional gate**
- `p_up_next`, `p_down_next` (sum to 1)

**B) Optional path distribution (advanced, later)**
- `p_bucket_next[b]` over a limited bucket range (e.g., -5..+5)
- or “expected signed move” `E[Δbucket]`

**WHY**:
- The UI can evolve from simple “emphasis gating” to actual “predicted flow field” without rethinking bucket physics.

### 10.3 Higher cadence (500ms)
Your architecture is already discrete‑column based (“never use continuous UV scrolling”). :contentReference[oaicite:27]{index=27}

To go 500ms:
- halve the window (`Δt = 500ms`)
- double the columns per minute of history
- keep bucketization identical

**WHY**:
- bucket view stays readable even as cadence increases
- ML can target microstructure impulses more precisely

---

## 11) What to add/change in the pipeline (conceptual, not code)

### 11.1 Add one new *bucket‑native* surface stream (recommended)
Create a server‑side derived dataset (call it conceptually):

- **`bucket_radar_surface_Δt`** (Δt = 1s now, 500ms later)

It emits fixed bucket rows:
- `window_end_ts_ns`
- `bucket_rel` (instead of rel_ticks)
- `blocked_level` (0..5) or `blocked_alpha` (0..1)
- `cavitation` (0..1)
- optional `gex_field` (0..1)
- optional `mobility` (0..1)

**WHY**:
- The backend has access to full fidelity fields and calibration.
- The frontend stays a renderer, not a feature engineering engine.
- It guarantees consistent results for training vs live view.

### 11.2 Add `mid_price_int` to `snap` stream (strongly recommended)
Backend snapshot dataset already defines it, but HUD stream currently doesn’t include it. :contentReference[oaicite:28]{index=28} :contentReference[oaicite:29]{index=29}

**WHY**:
- exact bucket anchoring
- removes float drift and rounding ambiguity

---

## 12) Diagnostics (How we know it’s correct)

You already rely on strict tick‑space alignment rules and ring buffer rectification. :contentReference[oaicite:30]{index=30}  
Extend the same philosophy to buckets:

1) **Bucket gridlines**
- draw at every bucket boundary (2 ticks)
- also draw heavy lines at every 10 buckets (=$5) to verify GEX alignment

2) **Sanity invariants**
- GEX rel_ticks must map to exactly every 10 buckets (ES) :contentReference[oaicite:31]{index=31}
- bucket center must follow spot line (if mid_price_int present)

3) **Breakthrough events**
- log when spot moves through BlockedLevel 5 without prior erosion in the preceding columns
- treat as “break label” for ML training and tuning

---

## 13) Summary (the new “clear mental model”)

### Physics, clearly:
- **Obstacle Mass (wall)**: where the medium is “thick.” (futures)
- **Vacuum / Erosion (vacuum)**: where thickness is being removed. (futures, derived from wall dynamics)
- **External Potential (gex)**: macro stiffness ridges. (options)
- **Blockedness**: mass + stiffness, destroyed by vacuum (your “solid boxes”)
- **Cavitation**: visible vacuum (your “reversal preparation” signal)
- **Directional switching**: not hard-coded; it’s **attention gating** driven by ML probabilities (with a temporary heuristic placeholder)

This gives you the bucket‑native “weather radar” overlay you described, while preserving bidirectional information and making ML integration a first-class design element. :contentReference[oaicite:32]{index=32} :contentReference[oaicite:33]{index=33} :contentReference[oaicite:34]{index=34} :contentReference[oaicite:35]{index=35}