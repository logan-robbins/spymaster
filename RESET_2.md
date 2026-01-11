## 16) Build the **UPWARD PUSH / BREAK-UP** feature vector (price pushes **up through** (P_{ref}))

You will **not** change any raw processing, shadow-book logic, bucketing, or windowing from the previous spec. You will compute the original 17 features (Sections 11–12) exactly as written, then deterministically transform them into a new 17-feature vector whose **polarity is “UP”**.

### 16.1 Naming

* Let the original (already emitted) features be `f1..f7` exactly as defined.
* Create a second set of features with prefix `u_` (UP-oriented) and emit them alongside (plus their d1/d2/d3, Section 16.4).

### 16.2 UP-oriented feature transforms (explicit 1-to-1 mapping)

**Ask-side (resistance above level) — keep the “ghosting” mechanics**
These features already represent *resistance retreat/decay*, which is supportive of an upward push. Keep them, except where a sign must be flipped so “more ghosting” is numerically larger.

1. `u1_ask_com_disp_log = f1_ask_com_disp_log`

2. `u2_ask_slope_convex_log = f1_ask_slope_convex_log`

3. `u3_ask_near_share_decay = -f1_ask_near_share_delta`
   Reason: for an UP push, “near ask share goes down” is supportive; flipping makes “more decay” larger.

4. `u4_ask_reprice_away_share_rest = f1_ask_reprice_away_share_rest`
   (“Away” = asks repricing higher away from the level → supportive)

5. `u5_ask_pull_add_log_rest = f2_ask_pull_add_log_rest`

6. `u6_ask_pull_intensity_rest = f2_ask_pull_intensity_rest`

7. `u7_ask_near_pull_share_rest = f2_ask_near_pull_share_rest`

---

**Bid-side (support below level) — invert from “evaporation” to “support building”**
Your base bid features (`f3`, `f4`) were oriented so positive meant **bids moving down/pulling** (support loss). For an UP push you need the opposite: bids **move up toward** the level and **adds dominate** pulls.

8. `u8_bid_com_approach_log = -f3_bid_com_disp_log`
   (Approach toward level = distance decreases)

9. `u9_bid_slope_support_log = -f3_bid_slope_convex_log`
   (Base was `log(far/near)`; support building makes near larger → invert)

10. `u10_bid_near_share_rise = f3_bid_near_share_delta`
    (Already aligned: near share increasing is supportive)

11. `u11_bid_reprice_toward_share_rest = 1 - f3_bid_reprice_away_share_rest`
    (Base “away” = bids repricing lower; invert to “toward” = repricing higher)

12. `u12_bid_add_pull_log_rest = -f4_bid_pull_add_log_rest`
    (Base was `log(pull/add)`; invert to represent `log(add/pull)`)

13. **Replace the “pull intensity” concept with its true inverse: add intensity**
    Compute using the same internal intermediates already required by the base engine (`bid_add_qty`, `bid_depth_total_start`):

* `u13_bid_add_intensity = bid_add_qty / (bid_depth_total_start + EPS_QTY)`

14. **Invert the location of any remaining pulls** (if pulls exist, you prefer them NOT to be near the level):

* `u14_bid_far_pull_share_rest = 1 - f4_bid_near_pull_share_rest`

---

**UP composites (must be recomputed from the UP-oriented components, not from base f5–f7):**
15) `u15_up_expansion_log = u1_ask_com_disp_log + u8_bid_com_approach_log`

16. `u16_up_flow_log = u5_ask_pull_add_log_rest + u12_bid_add_pull_log_rest`

17. `u17_up_total_log = u15_up_expansion_log + u16_up_flow_log`

### 16.3 Output rules (still enforced)

* Do **not** output raw quantities.
* Output only `u1..u17`, plus their derivatives in Section 16.4.

### 16.4 Derivatives for UP features

For every `uX` (X=1..17), compute:

* `d1_uX[k] = uX[k] - uX[k-1]`
* `d2_uX[k] = d1_uX[k] - d1_uX[k-1]`
* `d3_uX[k] = d2_uX[k] - d2_uX[k-1]`
  Missing prior values → set to `0`.

---

## 17) Use the engine for **price coming DOWN toward (P_{ref})** (support test)

Nothing in the raw feature computation changes. What changes is **which oriented vector corresponds to “rejection” vs “breakthrough”**.

### 17.1 Define two canonical outcome vectors (always computed every window)

You will always compute both:

* **DOWN outcome vector** = the original base features (already defined in prior spec):
  `DOWN_VECTOR = { f1..f7 and their d1/d2/d3 }`
  Interpretation: “Support below is receding/evaporating” (and concurrent above ghosting per your base spec), which is consistent with **downward continuation away from the level**.

* **UP outcome vector** = the UP-oriented features (Section 16):
  `UP_VECTOR = { u1..u17 and their d1/d2/d3 }`
  Interpretation: “Support below is strengthening while resistance above is ghosting,” consistent with **upward movement away from / through the level**.

You will not conditionally compute one or the other. Compute both every window.

---

## 18) **Approach-direction mapping** (how to interpret “rejection” vs “breakthrough”)

Define `approach_dir` externally from your price stream (last trade / mid / 2m candle trend). This spec does not dictate how you compute it; it only dictates how to **map outcomes** once you have it.

### 18.1 If price is coming **UP** toward (P_{ref}) (resistance test)

* **Rejection (down from level)** → use `DOWN_VECTOR` (`f*`)
* **Breakthrough (up through level)** → use `UP_VECTOR` (`u*`)

### 18.2 If price is coming **DOWN** toward (P_{ref}) (support test)

* **Rejection (up from level / bounce)** → use `UP_VECTOR` (`u*`)
* **Breakthrough (down through level / continuation)** → use `DOWN_VECTOR` (`f*`)

This is the only “flip” required for the second permutation: **the label “rejection” swaps which outcome-vector it points to when the approach direction flips.**

---

## 19) Final emit requirement for multi-permutation use

For every 5s window, emit a single row containing:

* the original base feature set: `f1..f7`, `d1_f*`, `d2_f*`, `d3_f*`
* the UP-oriented feature set: `u1..u17`, `d1_u*`, `d2_u*`, `d3_u*`

Downstream systems (retrieval / classification / visualization) must select:

* `UP_VECTOR` vs `DOWN_VECTOR` according to Sections 18.1–18.2 based on approach direction and the semantic meaning of “reject” vs “break.”
