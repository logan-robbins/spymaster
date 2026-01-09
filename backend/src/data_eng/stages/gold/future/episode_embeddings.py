from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

LOOKBACK_BARS = 120

METADATA_COLS = {
    "bar_ts",
    "symbol",
    "episode_id",
    "touch_id",
    "level_type",
    "level_price",
    "trigger_bar_ts",
    "bar_index_in_episode",
    "bar_index_in_touch",
    "bars_to_trigger",
    "is_pre_trigger",
    "is_pre_touch",
    "is_trigger_bar",
    "is_post_trigger",
    "is_post_touch",
    "approach_direction",
    "is_standard_approach",
    "dist_to_level_pts",
    "signed_dist_pts",
    "outcome",
    "outcome_score",
    "is_truncated_lookback",
    "is_truncated_forward",
    "is_extended_forward",
    "extension_count",
}


# TRUE level-relative features: computed using level_price in the calculation
# These are the ONLY features valid for similarity search across different levels/contracts
TRUE_LEVEL_RELATIVE_PREFIXES = (
    "bar5s_lvldepth_",   # depth above/below level price
    "bar5s_lvlflow_",    # flow above/below level price
    "bar5s_lvlwall_",    # wall detection at level price
    "bar5s_lvl_",        # distance to level, crosses, etc.
    "bar5s_approach_",   # approach direction relative to level
)


def is_level_relative_feature(col: str) -> bool:
    """Return True only for features computed relative to level_price."""
    if col in METADATA_COLS:
        return False

    for prefix in TRUE_LEVEL_RELATIVE_PREFIXES:
        if col.startswith(prefix):
            return True

    return False


def load_feature_columns_from_contract(contract_path: Path) -> List[str]:
    with open(contract_path, "r") as f:
        schema = json.load(f)
    return [
        field["name"]
        for field in schema["fields"]
        if is_level_relative_feature(field["name"])
    ]


def get_feature_columns(df: pd.DataFrame, contract_path: Path = None) -> List[str]:
    if contract_path is not None and contract_path.exists():
        contract_features = load_feature_columns_from_contract(contract_path)
        return [c for c in contract_features if c in df.columns]
    return [c for c in df.columns if is_level_relative_feature(c)]


def extract_episode_tensor(
    df_episode: pd.DataFrame,
    feature_cols: List[str],
    lookback_bars: int = LOOKBACK_BARS,
) -> Tuple[np.ndarray, dict]:
    trigger_mask = df_episode["is_trigger_bar"] == True
    if not trigger_mask.any():
        return None, {}

    trigger_idx = df_episode[trigger_mask].index[0]
    df_sorted = df_episode.sort_values("bar_ts")
    trigger_pos = df_sorted.index.get_loc(trigger_idx)

    lookback_start = max(0, trigger_pos - lookback_bars)
    df_lookback = df_sorted.iloc[lookback_start:trigger_pos + 1]

    tensor = df_lookback[feature_cols].values.astype(np.float32)

    if len(tensor) < lookback_bars + 1:
        pad_rows = (lookback_bars + 1) - len(tensor)
        padding = np.zeros((pad_rows, len(feature_cols)), dtype=np.float32)
        tensor = np.vstack([padding, tensor])

    trigger_row = df_episode[trigger_mask].iloc[0]
    metadata = {
        "episode_id": trigger_row["episode_id"],
        "level_type": trigger_row["level_type"],
        "level_price": trigger_row["level_price"],
        "trigger_bar_ts": trigger_row["trigger_bar_ts"],
        "approach_direction": trigger_row["approach_direction"],
        "outcome": trigger_row["outcome"],
        "outcome_score": trigger_row["outcome_score"],
        "is_truncated_lookback": trigger_row["is_truncated_lookback"],
    }

    return tensor, metadata


def extract_all_episode_tensors(
    df: pd.DataFrame,
    lookback_bars: int = LOOKBACK_BARS,
    contract_path: Path = None,
) -> Tuple[np.ndarray, List[dict], List[str]]:
    feature_cols = get_feature_columns(df, contract_path)
    episode_ids = df["episode_id"].unique()

    tensors = []
    metadata_list = []

    for ep_id in episode_ids:
        df_ep = df[df["episode_id"] == ep_id]
        tensor, metadata = extract_episode_tensor(df_ep, feature_cols, lookback_bars)
        if tensor is not None:
            tensors.append(tensor)
            metadata_list.append(metadata)

    if not tensors:
        return np.array([]), [], feature_cols

    stacked = np.stack(tensors, axis=0)

    return stacked, metadata_list, feature_cols


def flatten_tensors(tensors: np.ndarray) -> np.ndarray:
    n_episodes = tensors.shape[0]
    return tensors.reshape(n_episodes, -1)


def fit_scaler_pca(
    tensors_flat: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 2048,
) -> Tuple[StandardScaler, PCA, dict]:
    """Fit StandardScaler then PCA on flattened episode tensors."""
    tensors_flat = np.nan_to_num(tensors_flat, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score standardization across episodes (critical for multi-scale features)
    scaler = StandardScaler()
    tensors_scaled = scaler.fit_transform(tensors_flat)

    n_samples, n_features = tensors_scaled.shape
    n_components = min(n_samples - 1, n_features, max_components)

    pca_full = PCA(n_components=n_components)
    pca_full.fit(tensors_scaled)

    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_threshold = np.searchsorted(cumsum, variance_threshold) + 1

    stats = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components_fit": n_components,
        "n_components_for_95pct": int(n_for_threshold),
        "variance_explained_curve": cumsum.tolist()[:100],
        "variance_at_256": float(cumsum[min(255, len(cumsum) - 1)]),
        "variance_at_512": float(cumsum[min(511, len(cumsum) - 1)]),
        "variance_at_1024": float(cumsum[min(1023, len(cumsum) - 1)]) if len(cumsum) > 1023 else None,
    }

    return scaler, pca_full, stats


def fit_pca(
    tensors_flat: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 2048,
) -> Tuple[PCA, dict]:
    """Legacy: fit PCA without standardization. Use fit_scaler_pca instead."""
    tensors_flat = np.nan_to_num(tensors_flat, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = tensors_flat.shape
    n_components = min(n_samples - 1, n_features, max_components)

    pca_full = PCA(n_components=n_components)
    pca_full.fit(tensors_flat)

    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_threshold = np.searchsorted(cumsum, variance_threshold) + 1

    stats = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components_fit": n_components,
        "n_components_for_95pct": int(n_for_threshold),
        "variance_explained_curve": cumsum.tolist()[:100],
        "variance_at_256": float(cumsum[min(255, len(cumsum) - 1)]),
        "variance_at_512": float(cumsum[min(511, len(cumsum) - 1)]),
        "variance_at_1024": float(cumsum[min(1023, len(cumsum) - 1)]) if len(cumsum) > 1023 else None,
    }

    return pca_full, stats


def create_embeddings_scaled(
    tensors_flat: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    n_components: int,
) -> np.ndarray:
    """Create embeddings with standardization then PCA projection."""
    tensors_flat = np.nan_to_num(tensors_flat, nan=0.0, posinf=0.0, neginf=0.0)
    tensors_scaled = scaler.transform(tensors_flat)
    transformed = pca.transform(tensors_scaled)
    return transformed[:, :n_components]


def create_embeddings(
    tensors_flat: np.ndarray,
    pca: PCA,
    n_components: int,
) -> np.ndarray:
    """Legacy: create embeddings without standardization."""
    tensors_flat = np.nan_to_num(tensors_flat, nan=0.0, posinf=0.0, neginf=0.0)
    transformed = pca.transform(tensors_flat)
    return transformed[:, :n_components]


# =============================================================================
# TEMPORAL-AWARE EMBEDDINGS (preserves derivative information)
# =============================================================================

def add_temporal_derivatives(tensors: np.ndarray) -> np.ndarray:
    """
    Add velocity (1st derivative) and acceleration (2nd derivative) features.

    Input: (n_episodes, n_bars, n_features)
    Output: (n_episodes, n_bars, n_features * 3)  # original + velocity + accel
    """
    n_episodes, n_bars, n_features = tensors.shape

    # Velocity: diff along time axis, pad first row with zeros
    velocity = np.diff(tensors, axis=1, prepend=tensors[:, :1, :])

    # Acceleration: diff of velocity
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])

    # Concatenate along feature axis
    augmented = np.concatenate([tensors, velocity, acceleration], axis=2)

    return augmented


def compute_temporal_summary(tensor: np.ndarray, n_segments: int = 6) -> np.ndarray:
    """
    Compress time axis while preserving temporal structure.

    Splits lookback into segments and computes summary stats per segment per feature.
    This preserves "how did feature X evolve over time" information.

    Input: (n_bars, n_features)
    Output: (n_segments * 4 * n_features,) - 4 stats per segment per feature
    """
    n_bars, n_features = tensor.shape
    segment_size = n_bars // n_segments

    summaries = []
    for seg_idx in range(n_segments):
        start = seg_idx * segment_size
        end = start + segment_size if seg_idx < n_segments - 1 else n_bars
        segment = tensor[start:end, :]

        # 4 summary stats per feature: mean, std, min-to-max range, end-start delta
        seg_mean = np.mean(segment, axis=0)
        seg_std = np.std(segment, axis=0)
        seg_range = np.max(segment, axis=0) - np.min(segment, axis=0)
        seg_delta = segment[-1, :] - segment[0, :]

        summaries.extend([seg_mean, seg_std, seg_range, seg_delta])

    return np.concatenate(summaries)


def create_temporal_embeddings(
    tensors: np.ndarray,
    n_segments: int = 6,
    add_derivatives: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Create embeddings that preserve temporal/derivative structure.

    Instead of flattening blindly, this:
    1. Optionally adds velocity/acceleration features
    2. Splits time into segments
    3. Computes summary stats per segment per feature

    This preserves "the feature was rising in segment 1, peaked in segment 3, fell in segment 5"

    Input: (n_episodes, n_bars, n_features)
    Output: (n_episodes, n_segments * 4 * n_features_augmented)
    """
    if add_derivatives:
        tensors = add_temporal_derivatives(tensors)

    tensors = np.nan_to_num(tensors, nan=0.0, posinf=0.0, neginf=0.0)

    n_episodes, n_bars, n_features = tensors.shape

    embeddings = []
    for i in range(n_episodes):
        emb = compute_temporal_summary(tensors[i], n_segments)
        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)

    stats = {
        "n_episodes": n_episodes,
        "n_bars": n_bars,
        "n_features_per_bar": n_features,
        "n_segments": n_segments,
        "embedding_dim": embeddings.shape[1],
        "add_derivatives": add_derivatives,
    }

    return embeddings, stats


def fit_robust_pca(
    embeddings: np.ndarray,
    max_components: int = None,
) -> Tuple[RobustScaler, PCA, dict]:
    """
    Fit RobustScaler (median/IQR) then PCA.

    RobustScaler is better for market data with outliers.
    """
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    n_samples, n_features = embeddings_scaled.shape
    if max_components is None:
        max_components = min(n_samples - 1, n_features)
    n_components = min(n_samples - 1, n_features, max_components)

    pca = PCA(n_components=n_components)
    pca.fit(embeddings_scaled)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_for_95 = np.searchsorted(cumsum, 0.95) + 1
    n_for_99 = np.searchsorted(cumsum, 0.99) + 1

    stats = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components_fit": n_components,
        "n_components_for_95pct": int(n_for_95),
        "n_components_for_99pct": int(n_for_99),
        "variance_curve": cumsum.tolist(),
    }

    return scaler, pca, stats


def apply_robust_pca(
    embeddings: np.ndarray,
    scaler: RobustScaler,
    pca: PCA,
    n_components: int = None,
) -> np.ndarray:
    """Apply fitted RobustScaler + PCA to embeddings."""
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    embeddings_scaled = scaler.transform(embeddings)
    transformed = pca.transform(embeddings_scaled)

    if n_components is not None:
        transformed = transformed[:, :n_components]

    return transformed


def load_episodes_from_lake(
    lake_path: Path,
    symbol: str,
    level_type: str,
    dates: List[str],
) -> pd.DataFrame:
    all_dfs = []
    table_name = f"market_by_price_10_{level_type.lower()}_approach"

    for dt in dates:
        path = lake_path / f"silver/product_type=future/symbol={symbol}/table={table_name}/dt={dt}"
        if path.exists():
            parquet_files = list(path.glob("*.parquet"))
            for f in parquet_files:
                df = pd.read_parquet(f)
                df["dt"] = dt
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def build_episode_embeddings_pipeline(
    lake_path: Path,
    symbol: str,
    level_type: str,
    dates: List[str],
    variance_threshold: float = 0.95,
    target_components: int = None,
) -> dict:
    print(f"Loading episodes for {symbol} {level_type}...")
    df = load_episodes_from_lake(lake_path, symbol, level_type, dates)

    if len(df) == 0:
        return {"error": "No episodes found"}

    print(f"Loaded {df['episode_id'].nunique()} episodes, {len(df)} total bars")

    print("Extracting episode tensors...")
    tensors, metadata_list, feature_cols = extract_all_episode_tensors(df)

    if len(tensors) == 0:
        return {"error": "No valid tensors extracted"}

    print(f"Tensor shape: {tensors.shape}")
    n_episodes, n_bars, n_features = tensors.shape

    print("Flattening tensors...")
    tensors_flat = flatten_tensors(tensors)
    print(f"Flattened shape: {tensors_flat.shape}")

    print("Fitting PCA...")
    pca, pca_stats = fit_pca(tensors_flat, variance_threshold)
    print(f"PCA stats: {pca_stats['n_components_for_95pct']} components for 95% variance")

    if target_components is None:
        target_components = pca_stats["n_components_for_95pct"]

    target_components = min(target_components, pca_stats["n_components_fit"])

    print(f"Creating embeddings with {target_components} components...")
    embeddings = create_embeddings(tensors_flat, pca, target_components)
    print(f"Embeddings shape: {embeddings.shape}")

    return {
        "embeddings": embeddings,
        "metadata": metadata_list,
        "feature_cols": feature_cols,
        "tensor_shape": (n_episodes, n_bars, n_features),
        "pca_stats": pca_stats,
        "pca_model": pca,
        "target_components": target_components,
    }


if __name__ == "__main__":
    lake_path = Path(__file__).parents[5] / "lake"

    dates = ["2025-12-18"]
    symbol = "ESH6"
    level_type = "PM_HIGH"

    result = build_episode_embeddings_pipeline(
        lake_path=lake_path,
        symbol=symbol,
        level_type=level_type,
        dates=dates,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\n=== RESULTS ===")
        print(f"Episodes: {len(result['metadata'])}")
        print(f"Tensor shape: {result['tensor_shape']}")
        print(f"Embedding shape: {result['embeddings'].shape}")
        print(f"\nPCA Stats:")
        print(f"  Components for 95% var: {result['pca_stats']['n_components_for_95pct']}")
        print(f"  Variance at 256 dims: {result['pca_stats']['variance_at_256']:.1%}")
        print(f"  Variance at 512 dims: {result['pca_stats']['variance_at_512']:.1%}")
        if result['pca_stats']['variance_at_1024']:
            print(f"  Variance at 1024 dims: {result['pca_stats']['variance_at_1024']:.1%}")

        print("\nSample outcomes:")
        for m in result['metadata'][:5]:
            print(f"  {m['episode_id']}: {m['outcome']} (score={m['outcome_score']:.2f})")
