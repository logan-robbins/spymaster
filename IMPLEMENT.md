# Implementation Plan: The "Radar" Architecture

## Objective
Transition the platform from a "Sniper" architecture (event-based, static level dependencies) to a "Radar" architecture (continuous, rolling physics visualization). This enables:
1.  **Visualization**: Real-time overlay of "Physics" (Vacuum/Walls/GEX) on the price chart.
2.  **Modeling**: Generating generalized "physics features" (gradients, obstacles) that describe market behavior as a fluid medium ("drop of water") interacting with obstacles (walls).

---

## 1. Phase A: Futures (The Continuous Vacuum)

**Goal:** Calculate slope/vacuum physics for every 5-second window, relative to the *current* price, creating a continuous "surface" of pressure.

### A.1. Data Contract: `silver.future_mbo.continuous_vacuum.avsc`
*   **Location:** `backend/src/data_eng/contracts/silver/future_mbo/continuous_vacuum.avsc`
*   **Action:** Create new schema based on `mbo_level_vacuum_5s.avsc`.
*   **Key Changes:**
    *   **Remove:** `P_ref`, `P_REF_INT`.
    *   **Add:** `ref_price` (double) - The snapshot Mid-Price used as the anchor for this window's buckets.
    *   **Retain:** All `f1_...`, `f2_...`, `u1_...` features. The *names* can stay the same, but their *meaning* changes from "Relative to P_ref" to "Relative to Instantaneous Price".
    *   **Why:** We need a schema that supports a continuous stream without needing a static pre-market level.

### A.2. Stage Implementation: `SilverComputeContinuousVacuum5s`
*   **Location:** `backend/src/data_eng/stages/silver/future_mbo/compute_continuous_vacuum_5s.py`
*   **Action:** Create this file by cloning `compute_level_vacuum_5s.py`.
*   **Function-by-Function Changes:**

    1.  **`transform(df, dt)`**:
        *   **Remove:** `_load_p_ref(df, dt)`.
        *   **Update:** Call `compute_continuous_vacuum_5s(df)` without passing `p_ref`.

    2.  **`compute_continuous_vacuum_5s(df)`**:
        *   **Logic:** Iterate through the 5s windows.
        *   **New Logic:** Inside the window loop, calculate `current_mid_price` at the *start* of the window (using the snapshot of the order book).
        *   **Pass:** This `current_mid_price` (converted to int) becomes the `p_ref_int` for *that specific window*.
        *   **Why:** This ensures the buckets (`At`, `Near`, `Far`) are always centered on the price *right now*.

    3.  **`_bucket_for(side, price_int, p_ref_int)`**:
        *   **No Change Needed:** logic works if `p_ref_int` is passed correctly (it just calculates distance).
        *   **Note:** Ensure `p_ref_int` is updated every window in the main loop.

    4.  **`_snapshot(...)`**:
        *   **Update:** Ensure it accepts the dynamic `p_ref_int`.

    5.  **`_load_p_ref(...)`**:
        *   **Delete:** This function is no longer needed.

### A.3. Pipeline Registration
*   **Location:** `backend/src/data_eng/pipeline.py`
*   **Action:**
    *   Import `SilverComputeContinuousVacuum5s`.
    *   In `build_pipeline("future_mbo", "silver")`, add this new stage.
    *   **Note:** Keep the old `SilverComputeMboLevelVacuum5s` for now (legacy compatibility), but run the new one alongside it.

---

## 2. Phase B: Options (The Visible Walls)

**Goal:** Visualize the "Brick Walls" (Liquidity Depth) and their erosion (`d1`, `d2`) to model them as physical obstacles.

### B.1. Data Contract: `silver.future_option_mbo.gex_5s.avsc`
*   **Location:** `backend/src/data_eng/contracts/silver/future_option_mbo/gex_5s.avsc`
*   **Action:** Add fields for Wall Size.
*   **New Fields:**
    *   `wall_call_above_[1-5]` (double)
    *   `wall_put_above_[1-5]` (double)
    *   `wall_call_below_[1-5]` (double)
    *   `wall_put_below_[1-5]` (double)
    *   **Derivatives:** `d1_wall_...`, `d2_wall_...`, `d3_wall_...` (for all the above).
    *   **Why:** To model "erosion" (d1 < 0) and "collapse" (d2 < 0).

### B.2. Stage Logic: `SilverComputeGex5s`
*   **Location:** `backend/src/data_eng/stages/silver/future_option_mbo/compute_gex_5s.py`
*   **Function:** `_compute_gex_features`
*   **Action:**
    *   **Current:** `call_depth = strike_depth.get(...)`.
    *   **Change:** Store `call_depth` and `put_depth` into the `features` dictionary as `wall_call_above_i`, etc.
    *   **Why:** We are already paying the compute cost to look up the depth; just save it.

### B.3. Feature Definition List
*   **Location:** `compute_gex_5s.py` (top of file)
*   **Action:** Update `BASE_GEX_FEATURES` or create `WALL_FEATURES` list to include the new field names so `_add_derivatives` automatically picks them up.

---

## 3. Phase C: Modeling Standardization (The "Drop of Water")

**Goal:** Ensure features are "Stationary" and "Normalized" so a model trained on 2024 data works on 2026 data.

### C.1. Normalization Strategy (Future Work / Check)
*   **Context:** `wall_call_above_1` = 500. Is that big?
*   **Task:** We need a "Relative Size" metric eventually.
*   **Immediate Action:** Add `avg_daily_volume` or `avg_layer_depth` to the metadata or as a normalization factor?
*   **Decision:** For this Sprint ("Overlay"), raw size is fine because the *human* trader sees the context. For *Modeling*, we will likely post-process these features (e.g., `Z-Score(wall_size)`).
*   **Code Action:** None required for this implementation, but keep features raw (un-scaled) to preserve information.

---

## 4. Verification & Testing

### Test Case 1: Continuous Vacuum
*   **Input:** 1 hour of MBO data where price moves from 5800 to 5820.
*   **Check:**
    *   At 5800: Buckets are calculated around 5800.
    *   At 5810: Buckets are calculated around 5810.
    *   **Verify:** No "gaps" in the output dataframe (previously, data would disappear if far from P_ref).

### Test Case 2: Wall Erosion
*   **Input:** Synthetic option order book where a 1000-lot wall at Strike X is cancelled over 1 minute.
*   **Check:**
    *   `wall_call_above_1`: Drops from 1000 -> 0.
    *   `d1_wall_call_above_1`: Shows negative values (e.g., -100/sec).
    *   `d2_wall_call_above_1`: Shows acceleration of the cancel.
