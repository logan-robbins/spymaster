# Stage 2 — Episode Extraction by Level Band

## Overview

This stage filters the continuous 5-second bar stream into discrete **episodes** — time windows around key level tests that will be used for setup characterization (Stage 3) and similarity retrieval (Stage 4+).

**Input:** Continuous 5-second bars with 233 base features (Stage 1 output)

**Output:** Four partitioned tables, one per level type:
- `market_by_price_10_pm_high_episodes`
- `market_by_price_10_pm_low_episodes`
- `market_by_price_10_or_high_episodes`
- `market_by_price_10_or_low_episodes`

Each episode contains:
- 15 minutes of lookback bars (180 bars)
- Variable forward window (minimum 8 minutes, extends on re-touches)
- Outcome label
- Metadata for grouping and retrieval

---

## Section 0 — Definitions and Constants

### 0.1 Time Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BAR_DURATION_NS` | 5,000,000,000 | 5 seconds in nanoseconds |
| `LOOKBACK_BARS` | 180 | 15 minutes of history |
| `LOOKBACK_NS` | 900,000,000,000 | 15 minutes in nanoseconds |
| `MIN_FORWARD_BARS` | 96 | 8 minutes minimum forward |
| `MIN_FORWARD_NS` | 480,000,000,000 | 8 minutes in nanoseconds |
| `EXTENSION_BARS` | 96 | 8 minutes extension on re-touch |
| `OUTCOME_WINDOW_1_BARS` | 48 | First 4 minutes for outcome eval |
| `OUTCOME_WINDOW_2_START` | 48 | 4 minutes mark |
| `OUTCOME_WINDOW_2_END` | 72 | 6 minutes mark |
| `COOLDOWN_BARS` | 24 | 2 minutes between episode triggers |
| `SESSION_END_OFFSET_NS` | 14,400,000,000,000 | 4 hours after open |

### 0.2 Distance Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `TRIGGER_BAND_PTS` | 1.0 | Trigger when trade within ±1 pt of level |
| `OUTCOME_BAND_1_PTS` | 1.0 | First outcome threshold |
| `OUTCOME_BAND_2_PTS` | 2.0 | Second outcome threshold |
| `POINT` | 1.0 | One index point |

### 0.3 Session Timing

```
MARKET_OPEN = 09:30:00 ET
SESSION_START = MARKET_OPEN
SESSION_END = MARKET_OPEN + 4 hours = 13:30:00 ET
```

For OR (Opening Range) levels:
```
OR_FORMATION_PERIOD = first 30 minutes (09:30 - 10:00 ET)
OR_HIGH = max(high) during OR_FORMATION_PERIOD
OR_LOW = min(low) during OR_FORMATION_PERIOD
```

OR levels are only valid for testing AFTER the OR formation period ends.

### 0.4 Level Definitions

| Level | Source | When Valid |
|-------|--------|------------|
| `PM_HIGH` | Prior session high (from schema) | From session start |
| `PM_LOW` | Prior session low (from schema) | From session start |
| `OR_HIGH` | Max price in first 30 min | After 10:00 ET |
| `OR_LOW` | Min price in first 30 min | After 10:00 ET |

---

## Section 1 — Trigger Detection

### 1.1 Microprice Computation

For each bar, compute microprice from end-of-bar book state:

```
microprice = (ask_px_00 * bid_sz_00 + bid_px_00 * ask_sz_00) / (bid_sz_00 + ask_sz_00 + EPSILON)
```

If book is empty on one side, use midpoint:
```
microprice = (ask_px_00 + bid_px_00) / 2
```

### 1.2 Trigger Condition

A **trigger** occurs when:

```
|microprice - level_price| <= TRIGGER_BAND_PTS
```

AND at least one of:
- `bar5s_trade_cnt_sum > 0` (trades occurred in this bar)
- Previous bar was outside the band (fresh entry into band)

### 1.3 Approach Direction

Determine approach direction by looking at the **prior 12 bars (1 minute)** before trigger:

```
pre_trigger_microprice = mean(microprice[t-12 : t-1])
approach_direction = sign(level_price - pre_trigger_microprice)
```

- `approach_direction = +1` → Price approaching from BELOW (microprice < level)
- `approach_direction = -1` → Price approaching from ABOVE (microprice > level)

**Edge case:** If pre-trigger bars are also within the band (already testing), use the earliest bar in the lookback window to determine direction.

### 1.4 Level-Direction Alignment

For outcome labeling, we need to know if this is a "breakout" or "bounce" scenario:

| Level Type | Approach From | Expected Outcome Direction |
|------------|---------------|---------------------------|
| PM_HIGH | Below (+1) | Break = continue up, Bounce = reverse down |
| PM_HIGH | Above (-1) | Break = continue down (re-break), Bounce = hold above |
| PM_LOW | Above (-1) | Break = continue down, Bounce = reverse up |
| PM_LOW | Below (+1) | Break = continue up (re-break), Bounce = hold below |
| OR_HIGH | Below (+1) | Break = continue up, Bounce = reverse down |
| OR_HIGH | Above (-1) | Break = continue down, Bounce = hold above |
| OR_LOW | Above (-1) | Break = continue down, Bounce = reverse up |
| OR_LOW | Below (+1) | Break = continue up, Bounce = hold below |

Store as:
```
level_polarity = +1 if level_type in {PM_HIGH, OR_HIGH} else -1
is_standard_approach = (approach_direction == -level_polarity)
```

`is_standard_approach = True` means:
- Testing a HIGH from below, or
- Testing a LOW from above

---

## Section 2 — Episode Window Construction

### 2.1 Initial Window

When a trigger is detected at bar `t_trigger`:

```
lookback_start = t_trigger - LOOKBACK_BARS  (180 bars = 15 min before)
forward_end = t_trigger + MIN_FORWARD_BARS  (96 bars = 8 min after)
```

### 2.2 Forward Extension on Re-Touch

Scan forward from `t_trigger`. If microprice **re-enters** the trigger band after having left it:

```
new_forward_end = max(forward_end, re_touch_bar + EXTENSION_BARS)
```

Continue scanning until no more re-touches or session end.

**Re-touch definition:**
1. Price must have exited the band (|microprice - level| > TRIGGER_BAND_PTS for at least 6 bars = 30 seconds)
2. Price re-enters the band (|microprice - level| <= TRIGGER_BAND_PTS)

### 2.3 Episode Boundaries

Final episode contains:

```
episode_start = lookback_start
episode_end = min(forward_end, session_end_bar)
trigger_bar_index = LOOKBACK_BARS  (bar 180 in 0-indexed, or bar 181 in 1-indexed)
```

### 2.4 Episode ID Generation

Each episode gets a unique identifier:

```
episode_id = f"{date}_{symbol}_{level_type}_{trigger_bar_ts}"
```

Example: `2024-01-15_ES_PM_HIGH_1705332600000000000`

---

## Section 3 — Outcome Labeling

### 3.1 Reference Point

All outcome distances are measured from `level_price`, not from trigger microprice.

### 3.2 Signed Distance Convention

```
signed_dist = (microprice - level_price) * approach_direction
```

This normalizes so that:
- **Positive** signed_dist = price moved in the "break" direction (through the level)
- **Negative** signed_dist = price moved in the "bounce" direction (away from level)

### 3.3 Outcome Windows

| Window | Bars After Trigger | Duration |
|--------|-------------------|----------|
| Window 1 | bars 1-48 | 0-4 minutes |
| Window 2 | bars 49-72 | 4-6 minutes |

### 3.4 Trade-Weighted Position

For each outcome window, compute the **volume-weighted average signed distance**:

```
window_trades = bars in window where trade_cnt > 0
vwap_signed_dist = sum(signed_dist * trade_vol) / sum(trade_vol)
```

If no trades in window, use time-weighted average of microprice:
```
twa_signed_dist = mean(signed_dist) over window bars
```

### 3.5 Outcome Classification

| Outcome | Window 1 Condition | Window 2 Condition |
|---------|-------------------|-------------------|
| `STRONG_BREAK` | vwap_signed_dist > +1.0 pt | vwap_signed_dist > +2.0 pt |
| `SOFT_BREAK` | vwap_signed_dist > +1.0 pt | +1.0 pt < vwap_signed_dist ≤ +2.0 pt |
| `CHOP_BREAK` | vwap_signed_dist > +1.0 pt | vwap_signed_dist ≤ +1.0 pt |
| `CHOP` | -1.0 pt ≤ vwap_signed_dist ≤ +1.0 pt | (any) |
| `CHOP_BOUNCE` | vwap_signed_dist < -1.0 pt | vwap_signed_dist ≥ -1.0 pt |
| `SOFT_BOUNCE` | vwap_signed_dist < -1.0 pt | -2.0 pt < vwap_signed_dist ≤ -1.0 pt |
| `STRONG_BOUNCE` | vwap_signed_dist < -1.0 pt | vwap_signed_dist < -2.0 pt |

**Simplified 5-class version (recommended for initial retrieval):**

| Outcome | Condition |
|---------|-----------|
| `STRONG_BREAK` | W1 > +1 pt AND W2 > +2 pt |
| `WEAK_BREAK` | W1 > +1 pt AND W2 ≤ +2 pt |
| `CHOP` | -1 pt ≤ W1 ≤ +1 pt |
| `WEAK_BOUNCE` | W1 < -1 pt AND W2 ≥ -2 pt |
| `STRONG_BOUNCE` | W1 < -1 pt AND W2 < -2 pt |

### 3.6 Outcome Confidence Score

Compute a continuous score for retrieval ranking:

```
outcome_score = (vwap_signed_dist_w1 + vwap_signed_dist_w2) / 2
```

Range: typically [-5, +5] points

---

## Section 4 — Cooldown and Overlap Handling

### 4.1 Cooldown Period

After a trigger at bar `t`, the next trigger for the **same level** cannot occur until:

```
next_valid_trigger_bar >= t + COOLDOWN_BARS  (24 bars = 2 minutes)
```

### 4.2 Cross-Level Triggers

Different levels can trigger independently. If PM_HIGH and OR_HIGH are both at 6800 (same price), treat as **one trigger** with:
- `level_type = "PM_HIGH+OR_HIGH"` (compound)
- Or choose based on priority: PM levels take precedence over OR levels

If PM_HIGH is at 6800 and OR_HIGH is at 6802, both can trigger independently with separate episodes.

### 4.3 Overlapping Episodes

Episodes for **different levels** may overlap in time. This is expected and allowed.

For the **same level**, the cooldown prevents immediate re-triggers, but episodes can still overlap if:
- Episode 1 triggers at t=100, extends to t=300
- Episode 2 triggers at t=130 (after cooldown), extends to t=350
- Bars 130-300 appear in both episodes

This is acceptable — they represent different "attempts" at the level.

### 4.4 Minimum Lookback Requirement

If a trigger occurs within the first 15 minutes of the session, the lookback is truncated:

```
available_lookback = t_trigger - session_start_bar
if available_lookback < LOOKBACK_BARS:
    lookback_start = session_start_bar
    lookback_bars_actual = available_lookback
```

Mark these episodes with `is_truncated_lookback = True`.

---

## Section 5 — Output Schema

### 5.1 Row Structure

Each row is one 5-second bar. An episode contains multiple rows sharing the same `episode_id`.

### 5.2 New Columns (in addition to 233 base features)

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | string | Unique episode identifier |
| `level_type` | string | PM_HIGH, PM_LOW, OR_HIGH, OR_LOW |
| `level_price` | float64 | Actual price of the level (points) |
| `trigger_bar_ts` | uint64 | Timestamp of the trigger bar (ns) |
| `bar_index_in_episode` | int32 | 0-indexed position within episode |
| `bars_to_trigger` | int32 | Negative = before trigger, 0 = trigger bar, positive = after |
| `is_pre_trigger` | bool | True if bar is in lookback period |
| `is_trigger_bar` | bool | True if this is the trigger bar |
| `is_post_trigger` | bool | True if bar is after trigger |
| `approach_direction` | int8 | +1 = from below, -1 = from above |
| `is_standard_approach` | bool | True if testing HIGH from below or LOW from above |
| `microprice_eob` | float64 | Microprice at end of bar |
| `dist_to_level_pts` | float64 | Signed distance: microprice - level_price |
| `signed_dist_pts` | float64 | Normalized: dist_to_level * approach_direction |
| `outcome` | string | STRONG_BREAK, WEAK_BREAK, CHOP, WEAK_BOUNCE, STRONG_BOUNCE |
| `outcome_score` | float64 | Continuous outcome measure |
| `is_truncated_lookback` | bool | True if lookback < 15 min |
| `is_extended_forward` | bool | True if forward window was extended |
| `extension_count` | int32 | Number of re-touch extensions |

### 5.3 Output Partitioning

```
output_path = f"{base_path}/market_by_price_10_{level_type}_episodes/date={date}/"
file_pattern = f"part-{sequence:05d}.parquet"
```

Each date partition contains all episodes for that date and level type.

---

## Section 6 — Processing Logic

### 6.1 Algorithm Pseudocode

```python
def process_date(date, bars_df, levels):
    """
    bars_df: DataFrame with continuous 5-second bars (Stage 1 output)
    levels: Dict with {PM_HIGH, PM_LOW, OR_HIGH, OR_LOW} prices
    """
    
    # Add microprice column
    bars_df['microprice_eob'] = compute_microprice(bars_df)
    
    # Session boundaries
    session_start = get_market_open_bar(date)
    session_end = session_start + (4 * 60 * 60 / 5)  # 4 hours in bars
    or_valid_start = session_start + (30 * 60 / 5)   # 30 min in bars
    
    # Process each level independently
    for level_type, level_price in levels.items():
        
        # Skip if level_price is NaN/missing
        if pd.isna(level_price):
            continue
            
        # Determine when this level becomes valid
        if level_type in ['OR_HIGH', 'OR_LOW']:
            valid_start = or_valid_start
        else:
            valid_start = session_start
        
        # Initialize state
        episodes = []
        last_trigger_bar = -COOLDOWN_BARS  # Allow first trigger immediately
        
        # Scan for triggers
        for t in range(valid_start, session_end):
            
            # Check cooldown
            if t < last_trigger_bar + COOLDOWN_BARS:
                continue
            
            # Check trigger condition
            microprice = bars_df.loc[t, 'microprice_eob']
            dist = abs(microprice - level_price)
            
            if dist > TRIGGER_BAND_PTS:
                continue
                
            if bars_df.loc[t, 'bar5s_trade_cnt_sum'] == 0:
                # Also check if fresh entry
                if t > 0:
                    prev_dist = abs(bars_df.loc[t-1, 'microprice_eob'] - level_price)
                    if prev_dist <= TRIGGER_BAND_PTS:
                        continue  # Not a fresh entry, skip
            
            # TRIGGER DETECTED at bar t
            
            # Compute approach direction
            approach_direction = compute_approach_direction(bars_df, t, level_price)
            
            # Determine lookback window
            lookback_start = max(session_start, t - LOOKBACK_BARS)
            
            # Determine forward window (with extensions)
            forward_end = compute_forward_end(bars_df, t, level_price, session_end)
            
            # Compute outcome
            outcome, outcome_score = compute_outcome(
                bars_df, t, level_price, approach_direction
            )
            
            # Extract episode bars
            episode_bars = bars_df.loc[lookback_start:forward_end].copy()
            
            # Add episode metadata
            episode_bars['episode_id'] = f"{date}_{level_type}_{bars_df.loc[t, 'bar_ts']}"
            episode_bars['level_type'] = level_type
            episode_bars['level_price'] = level_price
            episode_bars['trigger_bar_ts'] = bars_df.loc[t, 'bar_ts']
            episode_bars['bar_index_in_episode'] = range(len(episode_bars))
            episode_bars['bars_to_trigger'] = episode_bars['bar_index_in_episode'] - (t - lookback_start)
            episode_bars['is_pre_trigger'] = episode_bars['bars_to_trigger'] < 0
            episode_bars['is_trigger_bar'] = episode_bars['bars_to_trigger'] == 0
            episode_bars['is_post_trigger'] = episode_bars['bars_to_trigger'] > 0
            episode_bars['approach_direction'] = approach_direction
            episode_bars['dist_to_level_pts'] = episode_bars['microprice_eob'] - level_price
            episode_bars['signed_dist_pts'] = episode_bars['dist_to_level_pts'] * approach_direction
            episode_bars['outcome'] = outcome
            episode_bars['outcome_score'] = outcome_score
            # ... other metadata columns
            
            episodes.append(episode_bars)
            
            # Update cooldown
            last_trigger_bar = t
        
        # Write episodes to partition
        if episodes:
            write_partition(episodes, level_type, date)


def compute_approach_direction(bars_df, trigger_bar, level_price):
    """Determine if approaching from above or below."""
    lookback_bars = 12  # 1 minute
    start = max(0, trigger_bar - lookback_bars)
    
    pre_microprice = bars_df.loc[start:trigger_bar-1, 'microprice_eob'].mean()
    
    if pre_microprice < level_price:
        return +1  # Approaching from below
    else:
        return -1  # Approaching from above


def compute_forward_end(bars_df, trigger_bar, level_price, session_end):
    """Compute forward window end, with extensions for re-touches."""
    forward_end = trigger_bar + MIN_FORWARD_BARS
    
    # Track if price has exited the band
    exited = False
    exit_duration = 0
    
    for t in range(trigger_bar + 1, min(session_end, trigger_bar + 500)):  # Cap at ~40 min
        microprice = bars_df.loc[t, 'microprice_eob']
        dist = abs(microprice - level_price)
        
        if dist > TRIGGER_BAND_PTS:
            exit_duration += 1
            if exit_duration >= 6:  # 30 seconds outside band
                exited = True
        else:
            if exited:
                # Re-touch detected
                forward_end = max(forward_end, t + EXTENSION_BARS)
                exited = False
            exit_duration = 0
        
        # Stop if we've gone past the current forward_end and no re-touch
        if t > forward_end and exited:
            break
    
    return min(forward_end, session_end)


def compute_outcome(bars_df, trigger_bar, level_price, approach_direction):
    """Compute outcome label and score."""
    
    # Window 1: bars 1-48 (0-4 minutes after trigger)
    w1_start = trigger_bar + 1
    w1_end = trigger_bar + 48
    
    # Window 2: bars 49-72 (4-6 minutes after trigger)
    w2_start = trigger_bar + 49
    w2_end = trigger_bar + 72
    
    # Compute volume-weighted signed distance for each window
    def vwap_signed_dist(start, end):
        window = bars_df.loc[start:end]
        vol = window['bar5s_trade_vol_sum']
        microprice = window['microprice_eob']
        dist = (microprice - level_price) * approach_direction
        
        if vol.sum() > 0:
            return (dist * vol).sum() / vol.sum()
        else:
            return dist.mean()
    
    w1_dist = vwap_signed_dist(w1_start, w1_end)
    w2_dist = vwap_signed_dist(w2_start, w2_end)
    
    # Classify outcome
    if w1_dist > 1.0:
        if w2_dist > 2.0:
            outcome = 'STRONG_BREAK'
        else:
            outcome = 'WEAK_BREAK'
    elif w1_dist < -1.0:
        if w2_dist < -2.0:
            outcome = 'STRONG_BOUNCE'
        else:
            outcome = 'WEAK_BOUNCE'
    else:
        outcome = 'CHOP'
    
    # Continuous score
    outcome_score = (w1_dist + w2_dist) / 2
    
    return outcome, outcome_score
```

---

## Section 7 — Edge Cases

### 7.1 Missing Bars

If the input has gaps (missing bars), episodes containing those gaps should:
- Be flagged with `has_gap = True`
- Still be output, but with caution flag for Stage 3

### 7.2 Empty Book

If microprice cannot be computed (empty book on both sides):
- Use last valid microprice
- Flag with `has_empty_book = True`

### 7.3 Level at Zero or NaN

If a level price is 0, NaN, or clearly invalid:
- Skip that level entirely for that date
- Log warning

### 7.4 Very Close Levels

If two levels are within 1 point of each other (e.g., PM_HIGH = 6800, OR_HIGH = 6800.5):
- Create separate episodes for each
- Consider adding a `nearby_level` column indicating the other level

### 7.5 Trigger at Session End

If a trigger occurs with insufficient forward bars:
- Truncate forward window at session end
- Flag with `is_truncated_forward = True`
- Outcome may be `INCOMPLETE` if windows 1/2 can't be computed

---

## Section 8 — Validation

### 8.1 Per-Episode Checks

- `bar_index_in_episode` is sequential (0, 1, 2, ...)
- Exactly one bar has `is_trigger_bar = True`
- `bars_to_trigger` is consistent: negative before, 0 at trigger, positive after
- `outcome` is one of the 5 valid values
- `level_price` is constant across all bars in episode

### 8.2 Per-Date Checks

- No duplicate `episode_id` values
- Episodes respect cooldown (trigger bars are at least COOLDOWN_BARS apart)
- All episodes are within session boundaries

### 8.3 Cross-Date Checks

- Episode counts are reasonable (typically 5-50 per level per day)
- Outcome distribution is roughly balanced (not all CHOP)

---

## Section 9 — Output Statistics

After processing each date, log:

```
{
  "date": "2024-01-15",
  "level_type": "PM_HIGH",
  "level_price": 6800.00,
  "episodes_count": 12,
  "outcome_distribution": {
    "STRONG_BREAK": 2,
    "WEAK_BREAK": 3,
    "CHOP": 4,
    "WEAK_BOUNCE": 2,
    "STRONG_BOUNCE": 1
  },
  "avg_lookback_bars": 178.5,
  "avg_forward_bars": 102.3,
  "extensions_triggered": 3,
  "truncated_lookback_count": 1,
  "total_bars_output": 3456
}
```

---

## Section 10 — Integration with Stage 3

### 10.1 Required Columns for Stage 3

Stage 3 expects these columns (verified present):

| Column | Stage 2 Provides |
|--------|-----------------|
| `bar_ts` | ✓ From Stage 1 |
| `symbol` | ✓ From Stage 1 |
| `touch_id` / `episode_id` | ✓ Generated |
| `level_type` | ✓ Generated |
| `level_price` | ✓ Generated |
| `bar_index_in_touch` | ✓ = `bar_index_in_episode` |
| `is_pre_touch` | ✓ = `is_pre_trigger` |
| `is_post_touch` | ✓ = `is_post_trigger` |
| 233 base features | ✓ Passed through |

### 10.2 Column Aliasing

For compatibility with Stage 3 spec, add aliases:

```python
df['touch_id'] = df['episode_id']
df['bar_index_in_touch'] = df['bar_index_in_episode']
df['is_pre_touch'] = df['is_pre_trigger']
df['is_post_touch'] = df['is_post_trigger']
```

---

## Appendix A — Column List (New in Stage 2)

```
# Episode identification
episode_id                    # string, unique per episode
touch_id                      # string, alias for episode_id

# Level information  
level_type                    # string: PM_HIGH, PM_LOW, OR_HIGH, OR_LOW
level_price                   # float64: actual level price in points

# Temporal position
trigger_bar_ts                # uint64: trigger bar timestamp (ns)
bar_index_in_episode          # int32: 0-indexed position in episode
bar_index_in_touch            # int32: alias for bar_index_in_episode
bars_to_trigger               # int32: negative=before, 0=trigger, positive=after
is_pre_trigger                # bool
is_pre_touch                  # bool: alias for is_pre_trigger
is_trigger_bar                # bool
is_post_trigger               # bool
is_post_touch                 # bool: alias for is_post_trigger

# Approach characterization
approach_direction            # int8: +1=from below, -1=from above
is_standard_approach          # bool: HIGH from below or LOW from above

# Price relative to level
microprice_eob                # float64: microprice at bar end
dist_to_level_pts             # float64: microprice - level_price
signed_dist_pts               # float64: normalized by approach direction

# Outcome
outcome                       # string: STRONG_BREAK, WEAK_BREAK, CHOP, WEAK_BOUNCE, STRONG_BOUNCE
outcome_score                 # float64: continuous measure [-5, +5]

# Quality flags
is_truncated_lookback         # bool: lookback < 15 min
is_truncated_forward          # bool: forward cut short by session end
is_extended_forward           # bool: forward window was extended
extension_count               # int32: number of re-touch extensions
has_gap                       # bool: missing bars in episode
has_empty_book                # bool: empty book condition occurred
```

Total new columns: 24

---

## Appendix B — File Output Example

```
/data/silver/market_by_price_10_pm_high_episodes/
  date=2024-01-15/
    part-00000.parquet    # Episodes 1-10
    part-00001.parquet    # Episodes 11-20
    ...
  date=2024-01-16/
    part-00000.parquet
    ...
```

Each parquet file contains complete episodes (not split across files).

---

## Appendix C — Outcome Label Distribution Expectations

Based on market microstructure:

| Outcome | Expected Frequency |
|---------|-------------------|
| STRONG_BREAK | 10-20% |
| WEAK_BREAK | 15-25% |
| CHOP | 30-40% |
| WEAK_BOUNCE | 15-25% |
| STRONG_BOUNCE | 10-20% |

If actual distribution differs significantly:
- Review trigger/outcome thresholds
- Check level price accuracy
- Verify approach direction logic
