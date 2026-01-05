# Relative Volume Feature Engineering

## Overview

This specification adds **relative volume features** that compare current activity to historical baselines. These features answer: "Is this setup seeing more or less activity than typical for this time of day?"

**Why This Matters:**
- A 500-lot trade at 9:31am (high volume period) is normal
- A 500-lot trade at 11:45am (lunch lull) is significant
- Raw volume features don't capture this — relative volume does

**Integration Point:** These features are computed in Stage 3 and included in the Stage 5 setup vector.

---

## Section 1 — Historical Profile Construction

### 1.1 Profile Granularity

We build historical profiles at **5-minute bucket** granularity (not 5-second bars) for stability.

| Bucket | Time Range | Bucket ID |
|--------|------------|-----------|
| 0 | 09:30:00 - 09:34:59 | 0 |
| 1 | 09:35:00 - 09:39:59 | 1 |
| 2 | 09:40:00 - 09:44:59 | 2 |
| ... | ... | ... |
| 47 | 13:25:00 - 13:29:59 | 47 |

Total: 48 buckets for the 4-hour session (09:30 - 13:30).

### 1.2 Metrics to Profile

For each 5-minute bucket, compute historical averages of:

#### Trade Metrics
| Metric | Description |
|--------|-------------|
| `trade_vol` | Total trade volume |
| `trade_cnt` | Total trade count |
| `trade_aggbuy_vol` | Aggressive buy volume |
| `trade_aggsell_vol` | Aggressive sell volume |
| `trade_signed_vol` | Net signed volume (aggbuy - aggsell) |

#### Flow Metrics (by side)
| Metric | Description |
|--------|-------------|
| `flow_add_vol_bid` | Total bid-side add volume |
| `flow_add_vol_ask` | Total ask-side add volume |
| `flow_rem_vol_bid` | Total bid-side remove volume |
| `flow_rem_vol_ask` | Total ask-side remove volume |
| `flow_net_vol_bid` | Net bid-side flow (add - remove) |
| `flow_net_vol_ask` | Net ask-side flow (add - remove) |

#### Activity Metrics
| Metric | Description |
|--------|-------------|
| `msg_cnt` | Total message count |
| `add_cnt` | Add event count |
| `cancel_cnt` | Cancel event count |
| `modify_cnt` | Modify event count |

#### Derived Metrics
| Metric | Description |
|--------|-------------|
| `trade_imbal` | trade_signed_vol / trade_vol |
| `flow_imbal` | (flow_net_bid - flow_net_ask) / (flow_add_bid + flow_add_ask) |

### 1.3 Rolling Window Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LOOKBACK_DAYS` | 7 | Captures recent regime, adapts to changes |
| `MIN_DAYS` | 3 | Minimum days required for valid profile |
| `EXCLUDE_CURRENT` | True | Don't include current day in historical calc |

### 1.4 Profile Schema

```sql
CREATE TABLE volume_profiles (
    symbol          TEXT NOT NULL,
    profile_date    DATE NOT NULL,      -- Date this profile is valid FOR
    bucket_id       INTEGER NOT NULL,   -- 0-47 (5-min buckets)
    bucket_start    TIME NOT NULL,      -- e.g., '09:30:00'
    
    -- Trade metrics (mean and std over lookback window)
    trade_vol_mean          REAL,
    trade_vol_std           REAL,
    trade_cnt_mean          REAL,
    trade_cnt_std           REAL,
    trade_aggbuy_vol_mean   REAL,
    trade_aggbuy_vol_std    REAL,
    trade_aggsell_vol_mean  REAL,
    trade_aggsell_vol_std   REAL,
    trade_signed_vol_mean   REAL,
    trade_signed_vol_std    REAL,
    
    -- Flow metrics
    flow_add_vol_bid_mean   REAL,
    flow_add_vol_bid_std    REAL,
    flow_add_vol_ask_mean   REAL,
    flow_add_vol_ask_std    REAL,
    flow_rem_vol_bid_mean   REAL,
    flow_rem_vol_bid_std    REAL,
    flow_rem_vol_ask_mean   REAL,
    flow_rem_vol_ask_std    REAL,
    flow_net_vol_bid_mean   REAL,
    flow_net_vol_bid_std    REAL,
    flow_net_vol_ask_mean   REAL,
    flow_net_vol_ask_std    REAL,
    
    -- Activity metrics
    msg_cnt_mean            REAL,
    msg_cnt_std             REAL,
    
    -- Metadata
    days_in_lookback        INTEGER,    -- How many days contributed
    
    PRIMARY KEY (symbol, profile_date, bucket_id)
);
```

### 1.5 Profile Build Process

```python
def build_volume_profile(symbol: str, for_date: date, lookback_days: int = 7) -> pd.DataFrame:
    """
    Build volume profile for a specific date using prior N days.
    
    Args:
        symbol: Instrument symbol
        for_date: Date the profile will be used FOR (not included in calc)
        lookback_days: Number of prior days to include
    
    Returns:
        DataFrame with 48 rows (one per bucket) and mean/std for each metric
    """
    
    # Get prior trading days (exclude weekends, holidays)
    lookback_dates = get_prior_trading_days(for_date, lookback_days)
    
    if len(lookback_dates) < MIN_DAYS:
        raise InsufficientDataError(f"Only {len(lookback_dates)} days available")
    
    # Load 5-second bars for lookback period
    bars = load_bars(symbol, lookback_dates)
    
    # Assign each bar to a 5-minute bucket
    bars['bucket_id'] = bars['bar_ts'].apply(get_bucket_id)
    
    # Aggregate to 5-minute level per day
    daily_buckets = bars.groupby(['date', 'bucket_id']).agg({
        'bar5s_trade_vol_sum': 'sum',
        'bar5s_trade_cnt_sum': 'sum',
        'bar5s_trade_aggbuy_vol_sum': 'sum',
        'bar5s_trade_aggsell_vol_sum': 'sum',
        'bar5s_trade_signed_vol_sum': 'sum',
        'bar5s_flow_add_vol_bid_*': 'sum',  # Sum across all bands
        'bar5s_flow_add_vol_ask_*': 'sum',
        'bar5s_flow_rem_vol_bid_*': 'sum',
        'bar5s_flow_rem_vol_ask_*': 'sum',
        'bar5s_flow_net_vol_bid_*': 'sum',
        'bar5s_flow_net_vol_ask_*': 'sum',
        'bar5s_meta_msg_cnt_sum': 'sum',
    }).reset_index()
    
    # Compute mean and std across days for each bucket
    profile = daily_buckets.groupby('bucket_id').agg({
        col: ['mean', 'std'] for col in metric_columns
    })
    
    profile['days_in_lookback'] = len(lookback_dates)
    
    return profile


def get_bucket_id(ts_ns: int) -> int:
    """Convert nanosecond timestamp to 5-minute bucket ID (0-47)."""
    dt = pd.Timestamp(ts_ns, unit='ns', tz='America/New_York')
    minutes_since_open = (dt.hour - 9) * 60 + dt.minute - 30
    bucket_id = minutes_since_open // 5
    return max(0, min(47, bucket_id))
```

---

## Section 2 — Relative Volume Feature Computation

### 2.1 Feature Types

#### Type 1: Ratio Features
```
ratio = current_value / historical_mean
```
- Interpretation: 1.0 = average, 2.0 = double average, 0.5 = half average
- Range: [0, ∞), typically [0.1, 10]
- Good for: Multiplicative comparisons

#### Type 2: Z-Score Features
```
zscore = (current_value - historical_mean) / historical_std
```
- Interpretation: 0 = average, +2 = 2 std above, -2 = 2 std below
- Range: (-∞, +∞), typically [-5, +5]
- Good for: Anomaly detection, handles varying scales

#### Type 3: Percentile Features
```
percentile = rank(current_value) / n_historical_observations
```
- Interpretation: 0.5 = median, 0.95 = top 5%
- Range: [0, 1]
- Good for: Non-parametric comparison, robust to outliers

### 2.2 Computation Granularity

Compute relative volume at two levels:

#### Bar-Level (5-second)
Compare current 5-second bar to historical 5-minute bucket average (scaled).
```
bar_ratio = bar_value / (bucket_mean / 60)  # 60 bars per 5-min bucket
```

#### Rolling Window Level (1-minute, 5-minute)
Compare rolling sum to historical bucket.
```
rolling_1min = sum(last 12 bars)
rolling_ratio_1min = rolling_1min / (bucket_mean / 5)  # 5 minutes worth
```

### 2.3 Relative Volume Feature List

#### Trade Relative Volume (14 features)
| Feature | Computation | Type |
|---------|-------------|------|
| `rvol_trade_vol_ratio` | trade_vol / hist_mean | Ratio |
| `rvol_trade_vol_zscore` | (trade_vol - hist_mean) / hist_std | Z-Score |
| `rvol_trade_cnt_ratio` | trade_cnt / hist_mean | Ratio |
| `rvol_trade_cnt_zscore` | (trade_cnt - hist_mean) / hist_std | Z-Score |
| `rvol_trade_aggbuy_ratio` | aggbuy_vol / hist_mean | Ratio |
| `rvol_trade_aggsell_ratio` | aggsell_vol / hist_mean | Ratio |
| `rvol_trade_aggbuy_zscore` | z-score of aggbuy_vol | Z-Score |
| `rvol_trade_aggsell_zscore` | z-score of aggsell_vol | Z-Score |
| `rvol_trade_imbal_vs_hist` | current_imbal - hist_mean_imbal | Difference |
| `rvol_trade_aggbuy_pct_vs_hist` | (aggbuy/total) - hist_pct | Difference |
| `rvol_trade_vol_ratio_1min` | 1-min rolling trade vol ratio | Ratio |
| `rvol_trade_vol_ratio_5min` | 5-min rolling trade vol ratio | Ratio |
| `rvol_trade_vol_zscore_1min` | 1-min rolling z-score | Z-Score |
| `rvol_trade_vol_zscore_5min` | 5-min rolling z-score | Z-Score |

#### Flow Relative Volume (16 features)
| Feature | Computation | Type |
|---------|-------------|------|
| `rvol_flow_add_bid_ratio` | add_vol_bid / hist_mean | Ratio |
| `rvol_flow_add_ask_ratio` | add_vol_ask / hist_mean | Ratio |
| `rvol_flow_add_bid_zscore` | z-score | Z-Score |
| `rvol_flow_add_ask_zscore` | z-score | Z-Score |
| `rvol_flow_rem_bid_ratio` | rem_vol_bid / hist_mean | Ratio |
| `rvol_flow_rem_ask_ratio` | rem_vol_ask / hist_mean | Ratio |
| `rvol_flow_rem_bid_zscore` | z-score | Z-Score |
| `rvol_flow_rem_ask_zscore` | z-score | Z-Score |
| `rvol_flow_net_bid_ratio` | net_vol_bid / hist_mean | Ratio |
| `rvol_flow_net_ask_ratio` | net_vol_ask / hist_mean | Ratio |
| `rvol_flow_net_bid_zscore` | z-score | Z-Score |
| `rvol_flow_net_ask_zscore` | z-score | Z-Score |
| `rvol_flow_add_total_ratio` | (add_bid + add_ask) / hist_mean | Ratio |
| `rvol_flow_add_total_zscore` | z-score | Z-Score |
| `rvol_flow_imbal_vs_hist` | current_flow_imbal - hist_mean | Difference |
| `rvol_flow_bid_ask_ratio_vs_hist` | (bid/ask) / hist_ratio | Ratio |

#### Activity Relative Volume (6 features)
| Feature | Computation | Type |
|---------|-------------|------|
| `rvol_msg_cnt_ratio` | msg_cnt / hist_mean | Ratio |
| `rvol_msg_cnt_zscore` | z-score | Z-Score |
| `rvol_msg_cnt_ratio_1min` | 1-min rolling ratio | Ratio |
| `rvol_msg_cnt_ratio_5min` | 5-min rolling ratio | Ratio |
| `rvol_add_cancel_ratio_vs_hist` | (add/cancel) / hist_ratio | Ratio |
| `rvol_activity_intensity` | composite activity score | Composite |

#### Cumulative Deviation (8 features)
Track cumulative deviation from expected since market open:
| Feature | Computation |
|---------|-------------|
| `rvol_cumul_trade_vol_dev` | Σ(actual - expected) from open |
| `rvol_cumul_trade_vol_dev_pct` | cumul_dev / cumul_expected |
| `rvol_cumul_aggbuy_dev` | Cumulative aggbuy deviation |
| `rvol_cumul_aggsell_dev` | Cumulative aggsell deviation |
| `rvol_cumul_flow_bid_dev` | Cumulative bid flow deviation |
| `rvol_cumul_flow_ask_dev` | Cumulative ask flow deviation |
| `rvol_cumul_imbal_dev` | Cumulative imbalance deviation |
| `rvol_cumul_msg_dev` | Cumulative message count deviation |

**Total: 44 new features**

### 2.4 Asymmetric Relative Volume (Bid vs Ask)

Key insight: It's not just "is volume high?" but "is volume high on which side?"

| Feature | Computation | Interpretation |
|---------|-------------|----------------|
| `rvol_bid_ask_add_asymmetry` | (add_bid_zscore - add_ask_zscore) | + = bid adds unusually high |
| `rvol_bid_ask_rem_asymmetry` | (rem_bid_zscore - rem_ask_zscore) | + = bid removes unusually high |
| `rvol_bid_ask_net_asymmetry` | (net_bid_zscore - net_ask_zscore) | + = bid flow unusually positive |
| `rvol_aggbuy_aggsell_asymmetry` | (aggbuy_zscore - aggsell_zscore) | + = buying unusually aggressive |

**Additional: 4 features**

---

## Section 3 — Lookback Profile Features

### 3.1 Profile Over Approach Window

For the 15-minute lookback window, compute relative volume aggregations:

| Feature | Computation |
|---------|-------------|
| `rvol_lookback_trade_vol_mean_ratio` | mean(bar_ratios) over lookback |
| `rvol_lookback_trade_vol_max_ratio` | max(bar_ratios) over lookback |
| `rvol_lookback_trade_vol_trend` | (late_ratio - early_ratio) |
| `rvol_lookback_flow_imbal_mean_zscore` | mean flow imbalance z-score |
| `rvol_lookback_activity_peak_ratio` | max activity ratio in lookback |
| `rvol_lookback_activity_peak_bar` | which bar had peak (0-180) |
| `rvol_lookback_elevated_bars` | count of bars with ratio > 1.5 |
| `rvol_lookback_depressed_bars` | count of bars with ratio < 0.5 |

**Additional: 8 features**

### 3.2 Recent vs Lookback Comparison

Compare the last 1-minute to the full 15-minute lookback:

| Feature | Computation |
|---------|-------------|
| `rvol_recent_vs_lookback_ratio` | recent_1min_ratio / lookback_mean_ratio |
| `rvol_recent_vs_lookback_trade` | Is recent volume accelerating vs approach avg? |
| `rvol_recent_vs_lookback_flow_bid` | Bid flow acceleration |
| `rvol_recent_vs_lookback_flow_ask` | Ask flow acceleration |

**Additional: 4 features**

---

## Section 4 — Implementation

### 4.1 Pipeline Integration

```
[Stage 1: 5s Bars]
        ↓
[Profile Builder] ← Runs daily after market close
        ↓              Creates volume_profiles table for next day
[Stage 2: Episode Extraction]
        ↓
[Stage 3: Approach Features + Relative Volume]
        ↓              Joins with volume_profiles
[Stage 5: Vectorization]
```

### 4.2 Profile Builder (New Component)

```python
class VolumeProfileBuilder:
    """Builds and maintains historical volume profiles."""
    
    def __init__(self, lookback_days: int = 7, min_days: int = 3):
        self.lookback_days = lookback_days
        self.min_days = min_days
    
    def build_profile_for_date(self, symbol: str, for_date: date) -> VolumeProfile:
        """Build profile for a specific date using prior data."""
        
        lookback_dates = self._get_lookback_dates(for_date)
        
        if len(lookback_dates) < self.min_days:
            return self._get_fallback_profile(symbol)
        
        # Load historical bars
        bars = self._load_bars(symbol, lookback_dates)
        
        # Aggregate to 5-minute buckets per day
        daily_buckets = self._aggregate_to_buckets(bars)
        
        # Compute statistics across days
        profile = self._compute_profile_stats(daily_buckets)
        
        return profile
    
    def update_profiles_daily(self, symbols: List[str], as_of_date: date):
        """Update profiles for all symbols after market close."""
        
        for symbol in symbols:
            profile = self.build_profile_for_date(symbol, as_of_date + timedelta(days=1))
            self._store_profile(symbol, as_of_date + timedelta(days=1), profile)
    
    def _aggregate_to_buckets(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 5-second bars to 5-minute buckets."""
        
        bars['bucket_id'] = bars['bar_ts'].apply(get_bucket_id)
        bars['date'] = bars['bar_ts'].apply(lambda x: pd.Timestamp(x, unit='ns').date())
        
        # Sum metrics within each bucket
        agg_dict = {
            'bar5s_trade_vol_sum': 'sum',
            'bar5s_trade_cnt_sum': 'sum',
            'bar5s_trade_aggbuy_vol_sum': 'sum',
            'bar5s_trade_aggsell_vol_sum': 'sum',
            'bar5s_trade_signed_vol_sum': 'sum',
            'bar5s_meta_msg_cnt_sum': 'sum',
        }
        
        # Add flow columns (sum across all bands)
        for side in ['bid', 'ask']:
            for flow_type in ['add', 'rem', 'net']:
                for band in ['p0_1', 'p1_2', 'p2_3', 'p3_5', 'p5_10']:
                    col = f'bar5s_flow_{flow_type}_vol_{side}_{band}_sum'
                    agg_dict[col] = 'sum'
        
        buckets = bars.groupby(['date', 'bucket_id']).agg(agg_dict).reset_index()
        
        # Compute total flow by side
        for side in ['bid', 'ask']:
            for flow_type in ['add', 'rem', 'net']:
                band_cols = [f'bar5s_flow_{flow_type}_vol_{side}_{band}_sum' 
                            for band in ['p0_1', 'p1_2', 'p2_3', 'p3_5', 'p5_10']]
                buckets[f'flow_{flow_type}_{side}_total'] = buckets[band_cols].sum(axis=1)
        
        return buckets
    
    def _compute_profile_stats(self, daily_buckets: pd.DataFrame) -> pd.DataFrame:
        """Compute mean/std for each bucket across days."""
        
        metrics = [
            'bar5s_trade_vol_sum', 'bar5s_trade_cnt_sum',
            'bar5s_trade_aggbuy_vol_sum', 'bar5s_trade_aggsell_vol_sum',
            'bar5s_trade_signed_vol_sum', 'bar5s_meta_msg_cnt_sum',
            'flow_add_bid_total', 'flow_add_ask_total',
            'flow_rem_bid_total', 'flow_rem_ask_total',
            'flow_net_bid_total', 'flow_net_ask_total',
        ]
        
        profile = daily_buckets.groupby('bucket_id').agg({
            metric: ['mean', 'std', 'count'] for metric in metrics
        }).reset_index()
        
        # Flatten column names
        profile.columns = ['_'.join(col).strip('_') for col in profile.columns]
        
        return profile
```

### 4.3 Feature Computation (Stage 3 Addition)

```python
class RelativeVolumeFeatureComputer:
    """Computes relative volume features for episode bars."""
    
    def __init__(self, profile_store: VolumeProfileStore):
        self.profile_store = profile_store
    
    def compute_features(self, episode: pd.DataFrame) -> pd.DataFrame:
        """Add relative volume features to episode bars."""
        
        # Get profile for this date
        episode_date = episode['date'].iloc[0]
        symbol = episode['symbol'].iloc[0]
        profile = self.profile_store.get_profile(symbol, episode_date)
        
        # Add bucket_id to each bar
        episode['bucket_id'] = episode['bar_ts'].apply(get_bucket_id)
        
        # Join with profile
        episode = episode.merge(
            profile, 
            on='bucket_id', 
            how='left',
            suffixes=('', '_hist')
        )
        
        # Compute ratio features
        episode['rvol_trade_vol_ratio'] = (
            episode['bar5s_trade_vol_sum'] / 
            (episode['trade_vol_mean'] / 60 + EPSILON)  # Scale to bar level
        )
        
        episode['rvol_trade_vol_zscore'] = (
            (episode['bar5s_trade_vol_sum'] - episode['trade_vol_mean'] / 60) /
            (episode['trade_vol_std'] / np.sqrt(60) + EPSILON)
        )
        
        # ... compute all other features
        
        # Compute rolling features
        episode['rvol_trade_vol_ratio_1min'] = (
            episode['bar5s_trade_vol_sum'].rolling(12).sum() /
            (episode['trade_vol_mean'] / 5 + EPSILON)
        )
        
        # Compute cumulative deviation
        episode['expected_trade_vol_cumul'] = (
            episode['trade_vol_mean'] / 60
        ).cumsum()
        
        episode['actual_trade_vol_cumul'] = (
            episode['bar5s_trade_vol_sum']
        ).cumsum()
        
        episode['rvol_cumul_trade_vol_dev'] = (
            episode['actual_trade_vol_cumul'] - 
            episode['expected_trade_vol_cumul']
        )
        
        # Compute asymmetry features
        episode['rvol_bid_ask_add_asymmetry'] = (
            episode['rvol_flow_add_bid_zscore'] - 
            episode['rvol_flow_add_ask_zscore']
        )
        
        return episode
```

### 4.4 Trigger-Bar Extraction for Setup Vector

```python
def extract_rvol_features_for_vector(trigger_bar: pd.Series, lookback: pd.DataFrame) -> np.ndarray:
    """Extract relative volume features for the setup vector."""
    
    features = []
    
    # Snapshot features (at trigger bar)
    snapshot_cols = [
        'rvol_trade_vol_ratio',
        'rvol_trade_vol_zscore', 
        'rvol_trade_aggbuy_ratio',
        'rvol_trade_aggsell_ratio',
        'rvol_flow_add_bid_zscore',
        'rvol_flow_add_ask_zscore',
        'rvol_flow_net_bid_zscore',
        'rvol_flow_net_ask_zscore',
        'rvol_bid_ask_add_asymmetry',
        'rvol_bid_ask_net_asymmetry',
        'rvol_aggbuy_aggsell_asymmetry',
        'rvol_cumul_trade_vol_dev_pct',
        'rvol_cumul_imbal_dev',
    ]
    features.extend(trigger_bar[snapshot_cols].values)
    
    # Lookback profile features
    features.append(lookback['rvol_trade_vol_ratio'].mean())
    features.append(lookback['rvol_trade_vol_ratio'].max())
    features.append(lookback['rvol_trade_vol_ratio'].iloc[-12:].mean() - 
                   lookback['rvol_trade_vol_ratio'].iloc[:12].mean())  # trend
    features.append((lookback['rvol_trade_vol_ratio'] > 1.5).sum())  # elevated bars
    features.append((lookback['rvol_trade_vol_ratio'] < 0.5).sum())  # depressed bars
    features.append(lookback['rvol_bid_ask_net_asymmetry'].mean())
    
    # Recent vs lookback
    recent = lookback.iloc[-12:]  # Last minute
    features.append(recent['rvol_trade_vol_ratio'].mean() / 
                   (lookback['rvol_trade_vol_ratio'].mean() + EPSILON))
    
    return np.array(features)
```

---

## Section 5 — Feature Summary

### 5.1 New Features for Setup Vector

| Category | Features | Dims |
|----------|----------|------|
| Trade relative volume (snapshot) | ratio, zscore, by aggressor | 8 |
| Flow relative volume (snapshot) | by side, zscore, asymmetry | 10 |
| Cumulative deviation | trade, flow, imbalance | 4 |
| Asymmetry features | bid/ask, aggbuy/sell | 4 |
| Lookback profile | mean, max, trend, elevated/depressed | 6 |
| Recent vs lookback | acceleration | 2 |
| **Total New** | | **34** |

### 5.2 Updated Setup Vector

| Component | Before | After |
|-----------|--------|-------|
| Original features | 151 | 151 |
| Relative volume | 0 | 34 |
| **Total** | **151** | **185** |

Or with optimized v2 feature set:

| Component | Before (v2) | After (v2) |
|-----------|-------------|------------|
| Optimized features | 27 | 27 |
| Relative volume | 0 | 34 |
| **Total** | **27** | **61** |

---

## Section 6 — Expected Impact

### 6.1 Hypothesis

Relative volume features should improve retrieval because:

1. **Time normalization**: A "hot" setup at 11:30am should match other hot setups at 11:30am, not 9:35am (which is always hot)

2. **Regime detection**: High relative volume indicates "something is happening" — institutional activity, news reaction, etc.

3. **Side asymmetry**: Elevated bid-side relative volume during approach = more bullish than elevated ask-side

4. **Cumulative deviation**: If volume has been running 30% above average all morning, the current state is different from a normally quiet day

### 6.2 Features Most Likely to Help

Based on market microstructure theory:

| Feature | Expected Impact | Rationale |
|---------|-----------------|-----------|
| `rvol_bid_ask_net_asymmetry` | High | Captures directional flow imbalance |
| `rvol_aggbuy_aggsell_asymmetry` | High | Captures aggressive trader sentiment |
| `rvol_cumul_imbal_dev` | Medium | Cumulative pressure building |
| `rvol_lookback_trend` | Medium | Is activity accelerating into the level? |
| `rvol_trade_vol_ratio` | Low-Medium | General activity level |

### 6.3 Validation Plan

After implementing:

1. **Ablation test**: Add relative volume features to v2 set, measure accuracy change
2. **Feature importance**: Which relative volume features contribute most?
3. **Correlation check**: Are any relative volume features redundant with existing features?
4. **Time-of-day analysis**: Does retrieval improve specifically for non-open times?

---

## Section 7 — Edge Cases

### 7.1 Insufficient Historical Data

If `days_in_lookback < MIN_DAYS`:
- Use a global average profile (across all available data)
- Flag with `rvol_profile_quality = 'fallback'`

### 7.2 Zero Historical Volume

If `hist_mean = 0` for a bucket (rare, early morning edge):
- Set ratio features to 1.0 (neutral)
- Set zscore features to 0.0 (neutral)

### 7.3 First Days of Data

For dates where 7-day lookback isn't available:
- Use whatever is available (down to 3 days)
- Flag with `rvol_days_available < 7`

### 7.4 Half Days / Early Closes

If historical data includes half days:
- Exclude half days from profile computation
- Or: compute separate profiles for half days

---

## Section 8 — Storage Requirements

### 8.1 Profile Table Size

```
48 buckets × ~40 metrics × 8 bytes = ~15 KB per symbol per day
1 symbol × 252 trading days = ~4 MB per year
```

Negligible storage impact.

### 8.2 Feature Column Additions

```
34 new features × 8 bytes × ~280 bars per episode × ~50 episodes/day
= ~4 MB per day additional
= ~1 GB per year additional
```

Manageable.

---

## Appendix A — Full Feature List

```python
RELATIVE_VOLUME_FEATURES = [
    # Trade relative volume - snapshot (8)
    'rvol_trade_vol_ratio',
    'rvol_trade_vol_zscore',
    'rvol_trade_cnt_ratio',
    'rvol_trade_cnt_zscore',
    'rvol_trade_aggbuy_ratio',
    'rvol_trade_aggsell_ratio',
    'rvol_trade_aggbuy_zscore',
    'rvol_trade_aggsell_zscore',
    
    # Flow relative volume - snapshot (10)
    'rvol_flow_add_bid_ratio',
    'rvol_flow_add_ask_ratio',
    'rvol_flow_add_bid_zscore',
    'rvol_flow_add_ask_zscore',
    'rvol_flow_net_bid_ratio',
    'rvol_flow_net_ask_ratio',
    'rvol_flow_net_bid_zscore',
    'rvol_flow_net_ask_zscore',
    'rvol_flow_add_total_ratio',
    'rvol_flow_add_total_zscore',
    
    # Cumulative deviation (4)
    'rvol_cumul_trade_vol_dev',
    'rvol_cumul_trade_vol_dev_pct',
    'rvol_cumul_flow_imbal_dev',
    'rvol_cumul_msg_dev',
    
    # Asymmetry (4)
    'rvol_bid_ask_add_asymmetry',
    'rvol_bid_ask_rem_asymmetry',
    'rvol_bid_ask_net_asymmetry',
    'rvol_aggbuy_aggsell_asymmetry',
    
    # Lookback profile (6)
    'rvol_lookback_trade_vol_mean_ratio',
    'rvol_lookback_trade_vol_max_ratio',
    'rvol_lookback_trade_vol_trend',
    'rvol_lookback_elevated_bars',
    'rvol_lookback_depressed_bars',
    'rvol_lookback_asymmetry_mean',
    
    # Recent vs lookback (2)
    'rvol_recent_vs_lookback_vol_ratio',
    'rvol_recent_vs_lookback_asymmetry',
]

assert len(RELATIVE_VOLUME_FEATURES) == 34
```

---

## Appendix B — Profile Builder SQL (Alternative)

If using SQL-based profile computation:

```sql
-- Build 7-day rolling volume profile for a given date
WITH lookback_bars AS (
    SELECT 
        date,
        (EXTRACT(HOUR FROM bar_ts) - 9) * 12 + 
        EXTRACT(MINUTE FROM bar_ts) / 5 AS bucket_id,
        bar5s_trade_vol_sum,
        bar5s_trade_aggbuy_vol_sum,
        bar5s_trade_aggsell_vol_sum,
        bar5s_meta_msg_cnt_sum
    FROM silver.market_by_price_10_5s_bars
    WHERE symbol = 'ES'
      AND date BETWEEN :for_date - INTERVAL '10 days' AND :for_date - INTERVAL '1 day'
      AND EXTRACT(DOW FROM date) NOT IN (0, 6)  -- Exclude weekends
),
daily_buckets AS (
    SELECT 
        date,
        bucket_id,
        SUM(bar5s_trade_vol_sum) AS trade_vol,
        SUM(bar5s_trade_aggbuy_vol_sum) AS aggbuy_vol,
        SUM(bar5s_trade_aggsell_vol_sum) AS aggsell_vol,
        SUM(bar5s_meta_msg_cnt_sum) AS msg_cnt
    FROM lookback_bars
    GROUP BY date, bucket_id
)
SELECT 
    bucket_id,
    AVG(trade_vol) AS trade_vol_mean,
    STDDEV(trade_vol) AS trade_vol_std,
    AVG(aggbuy_vol) AS aggbuy_vol_mean,
    STDDEV(aggbuy_vol) AS aggbuy_vol_std,
    AVG(aggsell_vol) AS aggsell_vol_mean,
    STDDEV(aggsell_vol) AS aggsell_vol_std,
    AVG(msg_cnt) AS msg_cnt_mean,
    STDDEV(msg_cnt) AS msg_cnt_std,
    COUNT(*) AS days_in_lookback
FROM daily_buckets
GROUP BY bucket_id
ORDER BY bucket_id;
```
