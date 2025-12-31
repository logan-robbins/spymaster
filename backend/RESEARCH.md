# Research & Development: AI Context for "Spymaster"

> [!IMPORTANT]
> **TARGET AUDIENCE: AI CODING AGENTS**
> This document defines the *intent*, *history*, and *scientific goals* of the project. Use this to understand *why* the code exists, not just *how* it works. Prioritize the **Glass Box** philosophy over "Black Box" optimization.

---

## 1. Core Thesis: "Contextual Intelligence over Oracle Prediction"

**Primary Goal:** Transform the system from a directional predictor ("Buy/Sell") into a **Context Retrieval Engine** ("Here is what happened last time").

**The Problem (The "Alpha Fallacy"):**
*   We initially hypothesized that microstructure physics (Order Flow, Velocity) could deterministically predict Break/Reject outcomes at key levels.
*   **Result:** FAILED. High-fidelity physics matching (Phase 2) resulted in 60% accuracy vs 88% baseline (Inverse Correlation). The "physically identical" setups often produced opposite outcomes, indicating that microstructure at this scale is dominated by noise or adversarial HFT mechanics.

**The Solution (The "Glass Box" Pivot):**
*   We pivoted to providing **Historical Base Rates** and **Volatility Scenarios**.
*   Instead of guessing the future, we retrieve the *past*.
*   **Value Proposition:** "In 98% of geometrically similar setups (N=50), this level held. The average adverse excursion was 2.5 points." -> This is actionable, defensive intelligence for a human trader.

---

## 2. Research Phases & Findings

### Phase 1: Statistical Expansion (The Failure)
*   **Hypothesis:** Increasing sample size (N=45 -> N=361) and data density will validate the high (73%) initial accuracy.
*   **Outcome:** **NEGATIVE RESULT.** Accuracy dropped to 68.4%, significantly underperforming the "Always Reject" baseline (87.8%).
*   **Discovery:** The "Similarity Inversion" Anomaly. The *closest* generic neighbors (Q4) were *less* predictive than distant ones (Q1). The vector space was fitting to noise.

### Phase 2: Mechanism Validation (The Insight)
*   **Hypothesis:** The "Inversion" is caused by specific features (Physics/Noise) overwhelming the signal (Geometry/Structure).
*   **Experiment:** Comparison of "Physics Only" vs "Geometry Only" vectors.
*   **Outcome:** **VALIDATED.**
    *   **Physics Only:** Strong Inversion (-16.9% Delta). High similarity = Low Accuracy.
    *   **Geometry Only:** Positive Correlation (+11.9% Delta). High similarity = High Accuracy ("History Repeats").
*   **Conclusion:** The Retrieval Mechanism is sound (Geometry works), but the Feature Engineering for Physics was "over-fitting to noise."

### Phase 3: "Glass Box" Implementation (Current Standard)
*   **Goal:** Operationalize Phase 2 insights.
*   **Implementation:**
    *   **Vector:** 144D composite (Geometry + Physics). We keep Physics because we *want* to see the velocity context, even if it predicts poorly.
    *   **Output:** Expose `context_metrics` (Base Rates, Volatility) directly to the user.
    *   **Architecture:** `SimilarityQueryEngine` is the Source of Truth.
*   **Status:** **LIVE / DEPLOYED (v3.1.0).**

---

## 3. The Vector Architecture (The "Snapshot")

The 144-dimensional vector is the core "genetic code" of a market setup. It is designed to capture **Invisible Physics**.

| Section | Dims | Purpose | Agent Note |
|:---|:---|:---|:---|
| **A: Context** | 25 | Regime (Time of day, active levels, nearby structure) | "Where are we?" |
| **B: Dynamics** | 37 | **Physics**. Velocity, Acceleration, Jerk, OFI (multi-scale). | "How fast are we moving?" (The noise source) |
| **C: History** | 35 | Micro-history (last 5 bars). Log-transformed Delta Liq. | "What just happened?" |
| **D: Derived** | 11 | **Force/Mass**. Ratio of Order Flow (Force) to Limit Book (Mass). | "How 'heavy' is the move?" |
| **E: Trends** | 4 | Online trend slopes. | "Is pressure building?" |
| **F: Trajectory** | 32 | **Geometry**. DCT Coefficients of Price/OFI shapes. | "What does the path look like?" (The robust signal) |

**Critical Constraint:**
*   **Do not change the vector dimensions** without retraining the indices. FAISS is rigid (144D).
*   **DCT Basis:** Section F uses discrete cosine transform to encode shape frequency. This is our most robust feature set.

---

## 4. Current Research Questions (Phase 4 Roadmap)

We are now optimizing the **Quality of the Mirror**.

**Q1: Vector Calibration (The "Better Mirror")**
*   **Hypothesis:** We can create a "Better Vector" by reducing dimensionality (PCA/Autoencoder) to filter out the "Physics Noise" while keeping the "Physics Signal."
*   **Metric:** **Calibration Error.** If the retrieved neighbors imply 70% Reject, does it reject 70% of the time?
*   **Status:** **PLANNED.**

**Q2: The "Zone" Mechanics**
*   **Hypothesis:** Retrieval quality decays as the "Approach" gets stale.
*   **Investigation:** Verify if `MONITOR_BAND` (currently 5.0 pts) is optimal. Should it be dynamic (ATR-based)?

**Q3: Human-AI Alignment**
*   **Hypothesis:** Users trust "Visual Similarity" (Geometry) over "Invisible Similarity" (OFI).
*   **Goal:** Ensure the retrieved neighbors *look* like the current chart. If they don't, the user will reject the AI's advice.

---

## 5. Deployment & Infrastructure Rules

1.  **Source of Truth:** `src/ml/episode_vector.py` defines the vector. `src/ml/retrieval_engine.py` defines the logic.
2.  **Indices:** Stored in `data/gold/indices/es_level_indices`. Must be rebuilt if Vector logic changes.
3.  **Partitions:** We strictly partition by `Level Kind / Direction / Time Bucket`. **NEVER** cross-pollinate (e.g. comparing Open Range directly to Premarket High).
4.  **Fail Safe:** If Retrieval fails (missing index), the system must degrade gracefully (return 0.5 probability or "No Analogs"), not crash the `CoreService`.

---

> [!TIP]
> **Summary for Context Injection:**
> When starting a new task, remember: We are building a **Historical Context Engine**, not a Crystal Ball. Evaluate all code changes against: "Does this help the user understand the *historical precedents* of the current moment?"
