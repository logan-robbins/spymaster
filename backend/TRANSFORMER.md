# TRANSFORMER.md: Phase 5 Implementation Guide

**Target Audience:** AI Coding Agent
**Objective:** Replace the "Geometry Only" DCT vector with a learned **Neural Time-Series Embedding** (PatchTST/Transformer) to maximize context retrieval precision.
**Status:** PLAN / READY FOR IMPLEMENTATION

---

## 1. Context & Motivation

**Current State (Phase 4):**
- We use **DCT (Discrete Cosine Transform)** to encode the 20-minute approach trajectory (40 samples @ 30s cadence).
- This is "Rigid Geometry." It captures the *frequency* components of the shape.
- **Success:** Reduced ECE from 21% (Physics) to 2.4% (Geometry), proving that **Shape > Physics**.

**The Problem:**
- DCT is lossy and mathematically predetermined. It cannot learn non-linear patterns (e.g., "A specific double-bottom followed by an OFI spike").
- We want to learn the "Geometry of Liquidity" directly from the data.

**The Solution:**
- Train a **Time-Series Transformer (PatchTST)** to encode the raw 20-minute trajectory into a dense vector.
- Use this Neural Vector for similarity search.

---

## 2. Implementation Roadmap

### Step 1: Data Engineering (The Missing Link)
**Critical Blocker:** Currently, Stage 17 (`construct_episodes.py`) computes DCT coefficients but **discards** the raw trajectory data. You cannot train a Transformer on DCT coefficients.

**Task:**
1.  Modify `src/ml/episode_vector.py`:
    *   Update `construct_episodes_from_events` to extract and return `raw_sequences` (Shape: `[N, 40, C]`).
    *   Channels (`C=4`): `d_atr`, `ofi_60s`, `barrier_delta_liq_log`, `tape_imbalance`.
2.  Modify `src/pipeline/stages/construct_episodes.py`:
    *   Capture `raw_sequences` from the vector constructor.
    *   Save to `gold/episodes/es_level_episodes/version=X/sequences/date=YYYY-MM-DD/sequences.npy`.

### Step 2: The Neural Architecture (PatchTST Encoder)
**File:** `src/ml/models/market_transformer.py`

**Specs:**
- **Input:** `(Batch, 40, 4)`
    - Length: 40 steps (20 mins).
    - Channels: 4 features.
- **Patching:**
    - Patch Length: 8 steps (4 mins).
    - Stride: 4 steps (50% overlap).
    - Patches per Series: ~9 patches.
- **Backbone:**
    - `TransformerEncoder` (PyTorch).
    - Layers: 2 (Keep it shallow to prevent overfitting).
    - Heads: 4.
    - Model Dim: 64 or 128.
- **Pooling:** `CLS` token or Mean Pooling over patches.
- **Projection Head:** Linear -> `(Batch, 32)` (Output Dimension).

### Step 3: Training Pipeline (Supervised Contrastive)
**Desire:** We want "Similar Setups have Similar Outcomes."
**Loss Function:** **Supervised Contrastive Loss (SupCon)** or Triplet Loss is superior to simple Cross-Entropy here.
- **Positive Pair:** Two episodes with the same Outcome (e.g., both `BREAK`).
- **Negative Pair:** Episodes with different Outcomes.
- **Goal:** Learn an embedding space where `BREAK` clusters are distinct from `REJECT` clusters.

**Script:** `scripts/train_transformer.py`
1.  Load `sequences.npy` (X) and `metadata.parquet` (y = `outcome_4min`).
2.  Split Train/Val (Temporal Split!).
3.  Train with `SupConLoss`.
4.  Save best model to `data/ml/models/market_transformer_v1.pt`.

### Step 4: Integration (The Vector Compressor)
**Task:**
- Update `src/ml/vector_compressor.py`.
- Add strategy `neural_transformer`.
- Logic:
    - Load `market_transformer_v1.pt`.
    - Instead of `Vector[112:144]` (DCT), we ignore the vector input.
    - We take the `raw_sequence` input (need to plumbing this through).
    - **Wait:** `VectorCompressor` currently takes the 144D vector.
    - **Refactor:** `VectorCompressor` needs access to the raw data OR we replace the 144D vector entirely in Stage 17.
    - **Simplification Plan:**
        - Keep `VectorCompressor` as a "post-processor" for now.
        - **Better:** Use the Transformer *inside* Stage 17 to generate the vector in the first place?
        - **No:** We want to keep the "Source of Truth" (144D) and "Neural View" aligned.
        - **Decision:** The `VectorCompressor` will be updated to load the `sequences.npy` (matched by ID) if available, OR we simply rebuild the indices using a new script `scripts/build_neural_indices.py` that utilizes the model.

---

## 3. Step-by-Step Implementation Instructions (For Agent)

### Phase 5.1: Save the Data (Do this NOW)
1.  Edit `src/ml/episode_vector.py`.
    - Function: `construct_episodes_from_events`.
    - Logic: Inside the loop, collect `raw_series` (the 40x4 array used for DCT).
    - Return: `Tuple[vectors, metadata, sequences]`.
2.  Edit `src/pipeline/stages/construct_episodes.py`.
    - receive `sequences`.
    - Save `np.save(output_dir / 'sequences.npy', sequences)`.
3.  **Run Pipeline Stage 17** for the full date range.
    - This will backfill the raw training data.

### Phase 5.2: Build the Model
1.  Create `src/ml/models/market_transformer.py`.
    - Implement `PatchTSTEncoder`.
2.  Create `src/ml/datasets/sequence_dataset.py`.
    - `torch.utils.data.Dataset` that loads `sequences.npy` and `metadata.parquet`.

### Phase 5.3: Train
1.  Create `scripts/train_market_encoder.py`.
    - Implement training loop.
    - Use `pytorch_metric_learning` losses if available, or implement simple Triplet Loss.

### Phase 5.4: Deploy
1.  Create `scripts/build_neural_indices.py`.
    - Loads the trained model.
    - Loads all `sequences.npy`.
    - Infer embeddings (N, 32).
    - Builds FAISS index from these embeddings.
    - Saves to `gold/indices/neural_indices/...`

---

## 4. Context for the "Why"
The user (Human) is downloading massive amounts of data in the background. They want to ensure that when that data arrives, the **Systems** are ready to ingest it into a Transformer.
We are moving from **Feature Engineering** (DCT) to **Representation Learning** (Transformer).
This is the standard evolution of ML systems. We have validated the *signal* exists (Physics/Geometry), now we build the optimal *sensor* for it.
