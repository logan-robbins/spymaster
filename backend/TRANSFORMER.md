# TRANSFORMER.md: Phase 5 Implementation Guide

**Target Audience:** AI Coding Agent
**Objective:** Replace the "Geometry Only" DCT vector with a learned **Neural Time-Series Embedding** (PatchTST/Transformer) to maximize context retrieval precision.
**Status:** PLAN / READY FOR IMPLEMENTATION

---

## 1. Context & Motivation

**Current State (Phase 4.5):**
- We use **DCT (Discrete Cosine Transform)** to encode the 20-minute approach trajectory.
- **Ablation Result (Dec 31):**
    - **Geometry (Shape):** 71.8% Accuracy, **+19.6% Calibration** (Robust/Linear).
    - **Physics (Velocity/Force):** 69.1% Accuracy, **-17.0% Calibration** (Inverted/Non-Linear).
    - **Combined (149D):** 69.4% Accuracy (Pre-Tide). Physics "dilutes" Geometry.
- **The "Context Discovery":**
    - Physics helps for *Regime* levels (PM High: +6.5%).
    - Physics hurts for *Structural* levels (SMA 200: -11.3%).

**The Problem:**
- A linear model (kNN) cannot switch between "Follow Momentum" (Regime) and "Fade Momentum" (structure). It averages them, leading to failure.
- We need an **Attention Mechanism** to learn: *"If Level=SMA, Ignore Velocity. If Level=PM_HIGH, Attend Velocity."*

**The Solution:**
- Train a **Time-Series Transformer (PatchTST)** to encode the raw 20-minute trajectory.
- **Crucial Upgrade:** Input must include **BOTH** Geometry (Shape) and Physics (Velocity) channels so the model can learn the interaction.

---

## 2. Implementation Roadmap

### Step 1: Data Engineering (The Missing Link)
**Critical Blocker:** Stage 17 (`construct_episodes.py`) currently computes DCT coefficients but discards the raw trajectory. The current sequence saver (added recently) only saves 4 channels (`d_atr`, `ofi`, `barrier`, `tape`).

**Task:**
1.  Modify `src/ml/episode_vector.py`:
    - Update `construct_episodes_from_events` sequence formatting.
    - **New Shape:** `[N, 40, 7]` (40 steps = 20 mins @ 30s cadence).
    - **Channels (C=7):**
        1.  `distance_signed_atr` (Geometry)
        2.  `ofi_60s` (Order Flow)
        3.  `barrier_delta_liq_log` (Limit Book)
        4.  `tape_imbalance` (Aggression)
        5.  **`velocity_1min`** (Physics - Speed)
        6.  **`acceleration_1min`** (Physics - Force Outcome)
        7.  **`jerk_1min`** (Physics - Change in Force)
2.  **Backfill:** Re-run Stage 17 for the full date range to regenerate `sequences.npy` with these 7 channels.

### Step 2: The Neural Architecture (Physics-Aware PatchTST)
**File:** `src/ml/models/market_transformer.py`

**Specs:**
- **Input:** `(Batch, 40, 7)`
- **Conditioning:**
    - `level_kind` (Categorical, 15 types) -> Embedding (Dim 8).
    - Concatenate this embedding to the Transformer output (or use as a CLS token modifier).
    - **Why:** To explicitly tell the model "We are at an SMA" vs "We are at PM High", enabling the conditional logic found in the ablation.
- **Patching:**
    - Patch Length: 8 steps (4 mins).
    - Stride: 4 steps.
- **Backbone:**
    - `TransformerEncoder`.
    - Layers: 2.
    - Heads: 4.
    - Model Dim: 128.
- **Projection Head:** Linear -> `(Batch, 32)` (Output Dimension).

### Step 3: Training Pipeline (Supervised Contrastive)
**Desire:** "Similar Setup + Similar Context = Similar Outcome."

**Script:** `scripts/train_transformer.py`
1.  Load `sequences.npy` (X) and `metadata.parquet` (y_outcome, y_level).
2.  **Loss:** `SupConLoss` (Supervised Contrastive).
    - Pull together same-outcome pairs.
    - Push apart opposite-outcome pairs.
3.  **Result:** A 32D embedding that clusters by *Dynamics-Adjusted Outcome*.

### Step 4: Integration (149D Hybrid)
**Strategy:**
- We do **NOT** replace the whole 149D vector.
- we replace **Section F (Trajectory Basis, 32 dims)** with the **Neural Embedding (32 dims)**.
- **Result:**
    - Sections A-E (Regime, Physics, History, **Market Tide**): Explicit Human-Readable Features (117 dims).
    - Section F (Trajectory): Neural "Black Box" encoding of Shape+Physics interaction (32 dims).
    - Total: 149D.

---

## 3. Step-by-Step Instructions

### Phase 5.1: Fix the Data (Priority 1)
1.  Edit `src/ml/episode_vector.py`:
    - Add `velocity_1min`, `acceleration_1min`, `jerk_1min` to `seq_array`.
    - Ensure `nan` handling (fillna 0).
2.  Run `scripts/backfill_sequences.py` (or pipeline) to update `gold`.

### Phase 5.2: Build & Train
1.  Implement `MarketTransformer` (with Level embedding).
2.  Train on Nov-Dec data.

### Phase 5.3: Deployment
1.  Update `construct_episodes.py` to allow loading the model and inferring Section F on the fly (optional) OR build a separate "Neural Indexer".

---

## 4. Context for the "Why"
The ablation proved that **Physics is Context-Dependent**.
- SMA 200: High Velocity = FAKEOUT (Rejection).
- PM High: High Velocity = BREAKOUT.
We cannot code this rule manually (too complex). The Transformer will learn it via the `level_kind` token and the interaction between Channel 1 (Distance) and Channel 5 (Velocity).
