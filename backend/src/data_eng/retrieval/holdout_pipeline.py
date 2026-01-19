from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from ..utils import expand_date_range
from .index_builder import _filter_rth, _load_selection_rows, _load_vectors, build_indices
from .normalization import apply_robust_scaling, fit_robust_stats, l2_normalize
from .query import TriggerVectorRetriever

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss-cpu or faiss-gpu required") from exc

try:
    import igraph as ig
    import leidenalg as la
except ImportError as exc:
    raise ImportError("leidenalg and igraph required") from exc


@dataclass(frozen=True)
class EvalConfig:
    split_date: str
    horizon: int
    k_raw: int
    k_values: List[int]
    tau: float
    tod_minutes: int
    seed: int
    cluster_m_values: List[int]
    cluster_assign_k: int
    theta_values: List[float]
    resample_count: int
    resample_frac: float


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_int_list(raw: str) -> List[int]:
    items = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Empty int list")
    return sorted(set(items))


def _parse_float_list(raw: str) -> List[float]:
    items = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Empty float list")
    return sorted(set(items))


def _break_reject_labels(approach_dir: str) -> Tuple[str, str]:
    if approach_dir == "approach_up":
        return "BREAK_UP", "REJECT_DOWN"
    if approach_dir == "approach_down":
        return "BREAK_DOWN", "REJECT_UP"
    raise ValueError(f"Unexpected approach_dir: {approach_dir}")


def _label_target(df: pd.DataFrame, approach_dir: str, horizon: int) -> pd.Series:
    label_col = f"true_outcome_h{horizon}"
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    break_label, reject_label = _break_reject_labels(approach_dir)
    allowed = {break_label, reject_label, "CHOP", "WHIPSAW"}
    labels = df[label_col].astype(str)
    bad = sorted(set(labels.unique()) - allowed)
    if bad:
        raise ValueError(f"Unexpected labels: {bad}")
    return (labels == break_label).astype(int)


def _tod_bins(df: pd.DataFrame, bin_minutes: int) -> pd.Series:
    if "ts_end_ns" not in df.columns or "session_date" not in df.columns:
        raise ValueError("Missing ts_end_ns or session_date for time bins")
    ts = pd.to_datetime(df["ts_end_ns"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    minute = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    start_min = 9 * 60 + 30
    total_min = 3 * 60
    if np.any(minute < start_min) or np.any(minute > start_min + total_min):
        raise ValueError("Timestamp outside RTH window")
    minute = np.minimum(minute, start_min + total_min - 1)
    offset = minute - start_min
    bin_start = (offset // bin_minutes) * bin_minutes + start_min
    bin_end = bin_start + bin_minutes
    labels = [
        f"{int(s // 60):02d}:{int(s % 60):02d}-{int(e // 60):02d}:{int(e % 60):02d}"
        for s, e in zip(bin_start, bin_end)
    ]
    return pd.Series(labels, index=df.index, dtype=str)


def _weights(similarities: np.ndarray, tau: float) -> np.ndarray:
    s1 = float(similarities[0])
    return np.exp((similarities - s1) / tau)


def _neighbor_stats(similarities: np.ndarray, y_vals: np.ndarray, tau: float) -> Dict[str, float]:
    w = _weights(similarities, tau)
    w_sum = float(np.sum(w))
    if w_sum == 0:
        raise ValueError("Zero weight sum")
    p = float(np.sum(w * y_vals) / w_sum)
    w_sq = np.sum(w * w)
    n_eff = float(w_sum * w_sum / w_sq) if w_sq > 0 else 0.0
    p0 = 1.0 - p
    entropy = 0.0
    if p > 0:
        entropy -= p * float(np.log(p))
    if p0 > 0:
        entropy -= p0 * float(np.log(p0))
    return {
        "p_hat": p,
        "n_eff": n_eff,
        "s1": float(similarities[0]),
        "sK": float(similarities[-1]),
        "sim_gap": float(similarities[0] - similarities[-1]),
        "entropy": entropy,
    }


def _calibration_bins(y: np.ndarray, p: np.ndarray, bins: int) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        count = int(np.sum(mask))
        if count == 0:
            acc = float("nan")
            conf = float("nan")
        else:
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "bin_acc": acc,
                "bin_conf": conf,
                "bin_count": count,
            }
        )
    return rows


def _ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    rows = _calibration_bins(y, p, bins)
    total = float(len(y))
    if total == 0:
        return 0.0
    ece = 0.0
    for row in rows:
        if np.isnan(row["bin_acc"]):
            continue
        ece += abs(row["bin_acc"] - row["bin_conf"]) * (row["bin_count"] / total)
    return float(ece)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _lift(y: np.ndarray, p: np.ndarray, base: float) -> float:
    if len(y) == 0:
        return 0.0
    thresh = np.quantile(p, 0.9)
    mask = p >= thresh
    if not np.any(mask):
        return 0.0
    top_rate = float(np.mean(y[mask]))
    return top_rate / base if base > 0 else 0.0


def _metrics_rows(
    y: np.ndarray,
    p: np.ndarray,
    base: float,
    model: str,
    k: int | None,
    target_id: str,
    level_id: str,
    approach_dir: str,
    split_id: str,
    group: str,
    tod_bin: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    metrics = {
        "log_loss": _log_loss(y, p),
        "brier": _brier(y, p),
        "auc": _auc(y, p),
        "ece": _ece(y, p, 10),
        "lift_top_decile": _lift(y, p, base),
    }
    for name, value in metrics.items():
        rows.append(
            {
                "target_id": target_id,
                "level_id": level_id,
                "approach_dir": approach_dir,
                "split_id": split_id,
                "model": model,
                "k": k if k is not None else -1,
                "group": group,
                "tod_bin": tod_bin,
                "metric": name,
                "value": float(value) if value is not None else float("nan"),
                "n": int(len(y)),
                "bin_lower": float("nan"),
                "bin_upper": float("nan"),
                "bin_acc": float("nan"),
                "bin_conf": float("nan"),
                "bin_count": -1,
            }
        )
    calib = _calibration_bins(y, p, 10)
    for row in calib:
        rows.append(
            {
                "target_id": target_id,
                "level_id": level_id,
                "approach_dir": approach_dir,
                "split_id": split_id,
                "model": model,
                "k": k if k is not None else -1,
                "group": group,
                "tod_bin": tod_bin,
                "metric": "calibration_bin",
                "value": float("nan"),
                "n": int(len(y)),
                "bin_lower": row["bin_lower"],
                "bin_upper": row["bin_upper"],
                "bin_acc": row["bin_acc"],
                "bin_conf": row["bin_conf"],
                "bin_count": row["bin_count"],
            }
        )
    return rows


def _collect_metrics(
    df: pd.DataFrame,
    base: float,
    model: str,
    k: int | None,
    target_id: str,
    level_id: str,
    approach_dir: str,
    split_id: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    y = df["y_true"].to_numpy(dtype=float)
    p = df["p_hat"].to_numpy(dtype=float)
    rows.extend(
        _metrics_rows(
            y=y,
            p=p,
            base=base,
            model=model,
            k=k,
            target_id=target_id,
            level_id=level_id,
            approach_dir=approach_dir,
            split_id=split_id,
            group="overall",
            tod_bin="",
        )
    )
    for tod_bin, group in df.groupby("tod_bin"):
        y_g = group["y_true"].to_numpy(dtype=float)
        p_g = group["p_hat"].to_numpy(dtype=float)
        rows.extend(
            _metrics_rows(
                y=y_g,
            p=p_g,
            base=base,
            model=model,
            k=k,
            target_id=target_id,
            level_id=level_id,
            approach_dir=approach_dir,
            split_id=split_id,
                group="tod_bin",
                tod_bin=str(tod_bin),
            )
        )
    return rows


def _build_knn_predictions(
    retriever: TriggerVectorRetriever,
    test_df: pd.DataFrame,
    approach_dir: str,
    level_id: str,
    horizon: int,
    k_raw: int,
    k_values: List[int],
    tau: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    k_max = max(k_values)
    break_label, _ = _break_reject_labels(approach_dir)
    label_key = f"true_outcome_h{horizon}"
    for row in test_df.itertuples(index=False):
        vector = np.array(getattr(row, "vector"), dtype=np.float64)
        query_id = int(getattr(row, "vector_id"))
        session_date = str(getattr(row, "session_date"))
        neighbors = retriever.find_similar(level_id, approach_dir, vector, k=k_raw)
        filtered = []
        for n in neighbors:
            meta = n.metadata
            if int(meta.get("vector_id")) == query_id:
                continue
            n_date = str(meta.get("session_date"))
            if n_date >= session_date:
                continue
            filtered.append(n)
        if len(filtered) < k_max:
            continue
        sims = np.array([n.similarity for n in filtered], dtype=np.float64)
        ids = [int(n.metadata.get("vector_id")) for n in filtered]
        dates = [str(n.metadata.get("session_date")) for n in filtered]
        for k in k_values:
            sims_k = sims[:k]
            ids_k = ids[:k]
            dates_k = dates[:k]
            label_vals = [n.metadata.get(label_key) for n in filtered[:k]]
            if any(v is None for v in label_vals):
                raise ValueError("Missing neighbor labels")
            y_vals = np.array([1 if str(v) == break_label else 0 for v in label_vals], dtype=np.float64)
            stats = _neighbor_stats(sims_k, y_vals, tau)
            rows.append(
                {
                    "vector_id": query_id,
                    "session_date": session_date,
                    "symbol": str(getattr(row, "symbol")),
                    "level_id": level_id,
                    "approach_dir": approach_dir,
                    "ts_end_ns": int(getattr(row, "ts_end_ns")),
                    "P_ref": float(getattr(row, "P_ref")),
                    "tod_bin": str(getattr(row, "tod_bin")),
                    "k": int(k),
                    "p_hat": float(stats["p_hat"]),
                    "y_true": int(getattr(row, "y_true")),
                    "n_eff": float(stats["n_eff"]),
                    "s1": float(stats["s1"]),
                    "sK": float(stats["sK"]),
                    "sim_gap": float(stats["sim_gap"]),
                    "entropy": float(stats["entropy"]),
                    "neighbor_ids": ids_k,
                    "neighbor_dates": dates_k,
                    "neighbor_sims": sims_k.tolist(),
                }
            )
    return rows


def _sample_null(
    rng: np.random.Generator,
    train_by_bin: Dict[str, np.ndarray],
    tod_bin: str,
    k: int,
) -> float:
    labels = train_by_bin.get(tod_bin)
    if labels is None:
        raise ValueError(f"Missing train bin: {tod_bin}")
    if len(labels) < k:
        raise ValueError("Insufficient train labels for null sample")
    pick = rng.choice(labels, size=k, replace=False)
    return float(np.mean(pick))


def _build_null_predictions(
    rng: np.random.Generator,
    test_df: pd.DataFrame,
    k_values: List[int],
    train_by_bin: Dict[str, np.ndarray],
) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {k: {} for k in k_values}
    for row in test_df.itertuples(index=False):
        vid = int(getattr(row, "vector_id"))
        tod_bin = str(getattr(row, "tod_bin"))
        for k in k_values:
            out[k][vid] = _sample_null(rng, train_by_bin, tod_bin, k)
    return out


def _build_baseline_predictions(
    test_df: pd.DataFrame,
    base_rate: float,
    base_by_bin: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    p_all = np.full(len(test_df), base_rate, dtype=np.float64)
    p_tod = np.array([base_by_bin[str(b)] for b in test_df["tod_bin"].tolist()], dtype=np.float64)
    return p_all, p_tod


def _target_id(approach_dir: str, horizon: int) -> str:
    break_label, reject_label = _break_reject_labels(approach_dir)
    return f"true_outcome_h{horizon}:{break_label}_vs_{reject_label}"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _fit_temperature(p_train: np.ndarray, y_train: np.ndarray) -> float:
    logits = _logit(p_train)
    temps = np.logspace(-2, 2, 41)
    best_t = float(temps[0])
    best_loss = float("inf")
    for t in temps:
        p_scaled = _sigmoid(logits / t)
        loss = _log_loss(y_train, p_scaled)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def _calibration_table(
    train_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    k_values: List[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for k in k_values:
        train_k = train_pred.loc[train_pred["k"] == k].copy()
        test_k = test_pred.loc[test_pred["k"] == k].copy()
        if len(train_k) == 0 or len(test_k) == 0:
            raise ValueError("Missing calibration data")
        y_train = train_k["y_true"].to_numpy(dtype=int)
        p_train = train_k["p_hat"].to_numpy(dtype=float)
        y_test = test_k["y_true"].to_numpy(dtype=int)
        p_test = test_k["p_hat"].to_numpy(dtype=float)
        rows.append(
            {
                "k": int(k),
                "method": "knn_raw",
                "log_loss": _log_loss(y_test, p_test),
                "ece": _ece(y_test, p_test, 10),
                "auc": _auc(y_test, p_test),
            }
        )
        temp = _fit_temperature(p_train, y_train)
        p_temp = _sigmoid(_logit(p_test) / temp)
        rows.append(
            {
                "k": int(k),
                "method": "knn_temp",
                "log_loss": _log_loss(y_test, p_temp),
                "ece": _ece(y_test, p_temp, 10),
                "auc": _auc(y_test, p_temp),
            }
        )
        logit_model = LogisticRegression(max_iter=200, solver="liblinear")
        logit_model.fit(_logit(p_train).reshape(-1, 1), y_train)
        p_platt = logit_model.predict_proba(_logit(p_test).reshape(-1, 1))[:, 1]
        rows.append(
            {
                "k": int(k),
                "method": "knn_platt",
                "log_loss": _log_loss(y_test, p_platt),
                "ece": _ece(y_test, p_platt, 10),
                "auc": _auc(y_test, p_platt),
            }
        )
        if len(np.unique(p_train)) < 2:
            raise ValueError("Isotonic needs more variance")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train, y_train)
        p_iso = iso.predict(p_test)
        rows.append(
            {
                "k": int(k),
                "method": "knn_iso",
                "log_loss": _log_loss(y_test, p_iso),
                "ece": _ece(y_test, p_iso, 10),
                "auc": _auc(y_test, p_iso),
            }
        )
    return pd.DataFrame(rows)


def _pivot_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[(metrics["group"] == "overall") & (metrics["metric"] != "calibration_bin")].copy()
    idx_cols = ["target_id", "split_id", "level_id", "approach_dir", "model", "k"]
    pivot = base.pivot_table(index=idx_cols, columns="metric", values="value", aggfunc="first").reset_index()
    n_vals = base.groupby(idx_cols)["n"].first().reset_index()
    pivot = pivot.merge(n_vals, on=idx_cols, how="left")
    return pivot


def _best_per_model(pivot: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in sorted(pivot["model"].unique().tolist()):
        model_df = pivot.loc[pivot["model"] == model].copy()
        if len(model_df) == 0:
            continue
        min_ll = model_df.loc[model_df["log_loss"].idxmin()]
        max_auc = model_df.loc[model_df["auc"].idxmax()]
        if int(min_ll["k"]) == int(max_auc["k"]):
            rows.append({**min_ll.to_dict(), "selection": "min_log_loss|max_auc"})
        else:
            rows.append({**min_ll.to_dict(), "selection": "min_log_loss"})
            rows.append({**max_auc.to_dict(), "selection": "max_auc"})
    out = pd.DataFrame(rows)
    return out


def _add_deltas(best_df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    out = best_df.copy()
    delta_cols = ["auc", "log_loss", "lift_top_decile", "ece"]
    for base_model in ["base_rate", "null_knn"]:
        deltas = []
        for row in out.itertuples(index=False):
            mask = (
                (pivot["model"] == base_model)
                & (pivot["k"] == row.k)
                & (pivot["target_id"] == row.target_id)
                & (pivot["split_id"] == row.split_id)
                & (pivot["level_id"] == row.level_id)
                & (pivot["approach_dir"] == row.approach_dir)
            )
            base_row = pivot.loc[mask]
            if len(base_row) != 1:
                raise ValueError("Baseline row not found")
            base_row = base_row.iloc[0]
            deltas.append(
                {
                    f"delta_auc_vs_{base_model}": float(row.auc - base_row.auc),
                    f"delta_log_loss_vs_{base_model}": float(row.log_loss - base_row.log_loss),
                    f"delta_lift_top_decile_vs_{base_model}": float(row.lift_top_decile - base_row.lift_top_decile),
                    f"delta_ece_vs_{base_model}": float(row.ece - base_row.ece),
                }
            )
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)
    return out


def _leaderboard(pivot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    knn_df = pivot.loc[pivot["model"] == "knn"].copy()
    if len(knn_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (level_id, approach_dir), group in knn_df.groupby(["level_id", "approach_dir"]):
        best_auc = group.loc[group["auc"].idxmax()]
        best_ll = group.loc[group["log_loss"].idxmin()]
        if int(best_ll["k"]) == int(best_auc["k"]):
            rows.append({**best_auc.to_dict(), "selection": "max_auc|min_log_loss"})
        else:
            rows.append({**best_auc.to_dict(), "selection": "max_auc"})
            rows.append({**best_ll.to_dict(), "selection": "min_log_loss"})
    lb = pd.DataFrame(rows)
    lb = _add_deltas(lb, pivot)
    auc_rows = lb.loc[lb["selection"].str.contains("max_auc")].copy()
    auc_rows = auc_rows.sort_values("delta_auc_vs_null_knn", ascending=False)
    top = auc_rows.head(10)
    bottom = auc_rows.tail(10).sort_values("delta_auc_vs_null_knn")
    return top, bottom


def _metric_summary(knn_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "log_loss", "ece", "lift_top_decile"]
    rows = []
    for metric in metrics:
        vals = knn_df[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "min": float(np.nanmin(vals)),
                "median": float(np.nanmedian(vals)),
                "max": float(np.nanmax(vals)),
            }
        )
    return pd.DataFrame(rows)


def _pareto_set(knn_df: pd.DataFrame) -> pd.DataFrame:
    rows = knn_df.copy().reset_index(drop=True)
    if len(rows) == 0:
        return rows
    auc = rows["auc"].to_numpy(dtype=float)
    loss = rows["log_loss"].to_numpy(dtype=float)
    ece = rows["ece"].to_numpy(dtype=float)
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        if not keep[i]:
            continue
        better_or_equal = (auc >= auc[i]) & (loss <= loss[i]) & (ece <= ece[i])
        strictly_better = (auc > auc[i]) | (loss < loss[i]) | (ece < ece[i])
        dominated = np.any(better_or_equal & strictly_better)
        if dominated:
            keep[i] = False
    return rows.loc[keep].copy()


def _format_header(
    target_id: str,
    split_id: str,
    level_id: str,
    approach_dir: str,
    train_range: str,
    test_range: str,
    norm_stats_source: str,
    faiss_index_hash: str,
) -> str:
    return (
        f"target_id={target_id} | split_id={split_id} | level_id={level_id} | "
        f"approach_dir={approach_dir} | train_range={train_range} | "
        f"test_range={test_range} | norm_stats_source={norm_stats_source} | "
        f"faiss_index_hash={faiss_index_hash}"
    )


def _write_summary_chat(
    output_dir: Path,
    header: str,
    knobs: Dict[str, object],
    metrics: pd.DataFrame,
    base_rate: float,
    n_eval: int,
    calib_table: pd.DataFrame,
) -> None:
    pivot = _pivot_metrics(metrics)
    knn_df = pivot.loc[pivot["model"] == "knn"].copy()
    best = _best_per_model(pivot)
    best = _add_deltas(best, pivot)
    top, bottom = _leaderboard(pivot)
    summary = _metric_summary(knn_df)
    pareto = _pareto_set(knn_df)
    lines: List[str] = []
    lines.append(header)
    lines.append("knobs=" + json.dumps(knobs, sort_keys=True))
    lines.append("Table A — Best k per model")
    lines.append(header)
    lines.append(best.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("Table B — Leaderboard by delta_auc vs null_knn")
    lines.append(header)
    if len(top) == 0:
        lines.append("no rows")
    else:
        lines.append("Top 10")
        lines.append(header)
        lines.append(top.round(6).to_markdown(index=False))
        lines.append("")
        lines.append("Bottom 10")
        lines.append(header)
        lines.append(bottom.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("Shape diagnostics")
    lines.append(header)
    lines.append(f"base_rate={base_rate:.6f} n={int(n_eval)}")
    lines.append(summary.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("Pareto set (AUC high, log_loss low, ECE low)")
    lines.append(header)
    lines.append(pareto.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("Table C — Calibration fixes")
    lines.append(header)
    lines.append(calib_table.round(6).to_markdown(index=False))
    (output_dir / "summary_chat.md").write_text("\n".join(lines))


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _build_graph(vectors: np.ndarray, m: int) -> Tuple[ig.Graph, np.ndarray]:
    if vectors.ndim != 2:
        raise ValueError("Expected 2D vectors")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    sims, idxs = index.search(vectors.astype(np.float32), m + 1)
    edge_map: Dict[Tuple[int, int], float] = {}
    for i in range(vectors.shape[0]):
        for j, sim in zip(idxs[i], sims[i]):
            if j < 0 or j == i:
                continue
            a = int(min(i, j))
            b = int(max(i, j))
            key = (a, b)
            current = edge_map.get(key)
            if current is None or sim > current:
                edge_map[key] = float(sim)
    edges = list(edge_map.keys())
    weights = [edge_map[e] for e in edges]
    graph = ig.Graph(n=vectors.shape[0], edges=edges, directed=False)
    graph.es["weight"] = weights
    return graph, np.array(weights, dtype=np.float64)


def _leiden_labels(graph: ig.Graph) -> np.ndarray:
    part = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
    )
    return np.array(part.membership, dtype=int)


def _cluster_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[int, float]:
    clusters_a: Dict[int, set] = {}
    clusters_b: Dict[int, set] = {}
    for idx, c in enumerate(labels_a.tolist()):
        clusters_a.setdefault(int(c), set()).add(idx)
    for idx, c in enumerate(labels_b.tolist()):
        clusters_b.setdefault(int(c), set()).add(idx)
    out: Dict[int, float] = {}
    for c_id, a_set in clusters_a.items():
        best = 0.0
        for b_set in clusters_b.values():
            inter = len(a_set & b_set)
            union = len(a_set | b_set)
            if union > 0:
                best = max(best, inter / union)
        out[c_id] = float(best)
    return out


def _merge_jaccard(base: Dict[int, float], new: Dict[int, float], count: Dict[int, int]) -> None:
    for k, v in new.items():
        base[k] = base.get(k, 0.0) + v
        count[k] = count.get(k, 0) + 1


def _benjamini_hochberg(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    qvals = np.zeros(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        p = pvals[idx]
        q = min(prev, p * n / (n - rank + 1))
        qvals[idx] = q
        prev = q
    return qvals.tolist()


def _cluster_enrichment(
    labels: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> pd.DataFrame:
    base_rate = float(np.mean(y)) if len(y) else 0.0
    rows: List[Dict[str, object]] = []
    for c_id in sorted(set(labels.tolist())):
        mask = labels == c_id
        size = int(np.sum(mask))
        if size == 0:
            continue
        rate = float(np.mean(y[mask]))
        diff = rate - base_rate
        perm_diffs = []
        for _ in range(permutations):
            y_perm = rng.permutation(y)
            perm_diffs.append(float(np.mean(y_perm[mask]) - base_rate))
        pval = float((np.sum(np.abs(perm_diffs) >= abs(diff)) + 1) / (permutations + 1))
        rows.append(
            {
                "cluster_id": int(c_id),
                "cluster_size": size,
                "cluster_rate": rate,
                "base_rate": base_rate,
                "rate_diff": diff,
                "p_value": pval,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["q_value"] = _benjamini_hochberg(df["p_value"].tolist())
    return df


def _cluster_medoids(
    labels: np.ndarray,
    vectors: np.ndarray,
    meta: pd.DataFrame,
    top_n: int,
) -> Dict[int, List[Dict[str, object]]]:
    medoids: Dict[int, List[Dict[str, object]]] = {}
    for c_id in sorted(set(labels.tolist())):
        idxs = np.where(labels == c_id)[0]
        if idxs.size == 0:
            continue
        centroid = np.mean(vectors[idxs], axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid = centroid / norm
        sims = vectors[idxs] @ centroid
        order = np.argsort(-sims)[:top_n]
        rows: List[Dict[str, object]] = []
        for rank in order.tolist():
            row = meta.iloc[int(idxs[rank])]
            rows.append(
                {
                    "vector_id": int(row["vector_id"]),
                    "session_date": str(row["session_date"]),
                    "symbol": str(row["symbol"]),
                    "ts_end_ns": int(row["ts_end_ns"]),
                    "P_ref": float(row["P_ref"]),
                }
            )
        medoids[int(c_id)] = rows
    return medoids


def _assign_clusters(
    retriever: TriggerVectorRetriever,
    test_df: pd.DataFrame,
    level_id: str,
    approach_dir: str,
    k_assign: int,
    cluster_map: Dict[int, int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for row in test_df.itertuples(index=False):
        vector = np.array(getattr(row, "vector"), dtype=np.float64)
        session_date = str(getattr(row, "session_date"))
        neighbors = retriever.find_similar(level_id, approach_dir, vector, k=k_assign)
        totals: Dict[int, float] = {}
        total_weight = 0.0
        for n in neighbors:
            n_date = str(n.metadata.get("session_date"))
            if n_date >= session_date:
                continue
            cid = cluster_map.get(int(n.metadata.get("vector_id")), -1)
            if cid < 0:
                continue
            weight = float(n.similarity)
            totals[cid] = totals.get(cid, 0.0) + weight
            total_weight += weight
        if not totals or total_weight == 0.0:
            rows.append(
                {
                    "vector_id": int(getattr(row, "vector_id")),
                    "session_date": session_date,
                    "symbol": str(getattr(row, "symbol")),
                    "level_id": level_id,
                    "approach_dir": approach_dir,
                    "cluster_id": -1,
                    "cluster_conf": 0.0,
                }
            )
            continue
        best_cluster = max(totals.items(), key=lambda kv: kv[1])[0]
        conf = float(totals[best_cluster] / total_weight)
        rows.append(
            {
                "vector_id": int(getattr(row, "vector_id")),
                "session_date": session_date,
                "symbol": str(getattr(row, "symbol")),
                "level_id": level_id,
                "approach_dir": approach_dir,
                "cluster_id": int(best_cluster),
                "cluster_conf": conf,
            }
        )
    return pd.DataFrame(rows)


def _run_backtest(
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    theta_values: List[float],
    base_rate: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pred_by_k = {k: group.copy() for k, group in predictions.groupby("k")}
    test_df = test_df.copy()
    for k, pred_df in pred_by_k.items():
        merged = pred_df.merge(
            test_df[["vector_id", "vol_regime"]],
            on="vector_id",
            how="left",
        )
        for theta in theta_values:
            p = merged["p_hat"].to_numpy(dtype=float)
            y = merged["y_true"].to_numpy(dtype=int)
            side = np.zeros(len(p), dtype=int)
            side[p > theta] = 1
            side[p < (1.0 - theta)] = -1
            trade = side != 0
            hits = (side == 1) & (y == 1) | (side == -1) & (y == 0)
            traded = int(np.sum(trade))
            hit_rate = float(np.sum(hits) / traded) if traded > 0 else 0.0
            pnl = np.where(trade, np.where(hits, 1.0, -1.0), 0.0)
            avg_pnl = float(np.mean(pnl[trade])) if traded > 0 else 0.0
            rows.append(
                {
                    "k": int(k),
                    "theta": float(theta),
                    "group": "overall",
                    "tod_bin": "",
                    "vol_regime": "",
                    "trades": traded,
                    "trade_rate": float(traded / len(p)) if len(p) else 0.0,
                    "hit_rate": hit_rate,
                    "avg_pnl": avg_pnl,
                    "base_rate": float(base_rate),
                }
            )
            for tod_bin, group in merged.groupby("tod_bin"):
                p_g = group["p_hat"].to_numpy(dtype=float)
                y_g = group["y_true"].to_numpy(dtype=int)
                side_g = np.zeros(len(p_g), dtype=int)
                side_g[p_g > theta] = 1
                side_g[p_g < (1.0 - theta)] = -1
                trade_g = side_g != 0
                hits_g = (side_g == 1) & (y_g == 1) | (side_g == -1) & (y_g == 0)
                traded_g = int(np.sum(trade_g))
                hit_rate_g = float(np.sum(hits_g) / traded_g) if traded_g > 0 else 0.0
                pnl_g = np.where(trade_g, np.where(hits_g, 1.0, -1.0), 0.0)
                avg_pnl_g = float(np.mean(pnl_g[trade_g])) if traded_g > 0 else 0.0
                rows.append(
                    {
                        "k": int(k),
                        "theta": float(theta),
                        "group": "tod_bin",
                        "tod_bin": str(tod_bin),
                        "vol_regime": "",
                        "trades": traded_g,
                        "trade_rate": float(traded_g / len(p_g)) if len(p_g) else 0.0,
                        "hit_rate": hit_rate_g,
                        "avg_pnl": avg_pnl_g,
                        "base_rate": float(base_rate),
                    }
                )
            for vol_regime, group in merged.groupby("vol_regime"):
                p_g = group["p_hat"].to_numpy(dtype=float)
                y_g = group["y_true"].to_numpy(dtype=int)
                side_g = np.zeros(len(p_g), dtype=int)
                side_g[p_g > theta] = 1
                side_g[p_g < (1.0 - theta)] = -1
                trade_g = side_g != 0
                hits_g = (side_g == 1) & (y_g == 1) | (side_g == -1) & (y_g == 0)
                traded_g = int(np.sum(trade_g))
                hit_rate_g = float(np.sum(hits_g) / traded_g) if traded_g > 0 else 0.0
                pnl_g = np.where(trade_g, np.where(hits_g, 1.0, -1.0), 0.0)
                avg_pnl_g = float(np.mean(pnl_g[trade_g])) if traded_g > 0 else 0.0
                rows.append(
                    {
                        "k": int(k),
                        "theta": float(theta),
                        "group": "vol_regime",
                        "tod_bin": "",
                        "vol_regime": str(vol_regime),
                        "trades": traded_g,
                        "trade_rate": float(traded_g / len(p_g)) if len(p_g) else 0.0,
                        "hit_rate": hit_rate_g,
                        "avg_pnl": avg_pnl_g,
                        "base_rate": float(base_rate),
                    }
                )
    return pd.DataFrame(rows)


def _cluster_workflow(
    train_df: pd.DataFrame,
    vectors_norm: np.ndarray,
    level_id: str,
    approach_dir: str,
    output_dir: Path,
    cfg: EvalConfig,
) -> Tuple[pd.DataFrame, Dict[int, int], pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    m_values = cfg.cluster_m_values
    base_m = m_values[len(m_values) // 2]
    graph, _ = _build_graph(vectors_norm, base_m)
    base_labels = _leiden_labels(graph)
    edge_list = graph.get_edgelist()
    edge_weights = np.array(graph.es["weight"], dtype=np.float64)
    cluster_edge_weights: Dict[int, List[float]] = {}
    for (u, v), w in zip(edge_list, edge_weights):
        c_u = int(base_labels[u])
        c_v = int(base_labels[v])
        if c_u != c_v:
            continue
        cluster_edge_weights.setdefault(c_u, []).append(float(w))
    cluster_median_sim = {
        cid: float(np.median(vals)) if vals else float("nan")
        for cid, vals in cluster_edge_weights.items()
    }
    jaccard_sum: Dict[int, float] = {}
    jaccard_count: Dict[int, int] = {}
    stability_rows: List[Dict[str, object]] = []
    for m in m_values:
        graph_m, _ = _build_graph(vectors_norm, m)
        labels_m = _leiden_labels(graph_m)
        ari = float(adjusted_rand_score(base_labels, labels_m))
        nmi = float(normalized_mutual_info_score(base_labels, labels_m))
        stability_rows.append(
            {
                "level_id": level_id,
                "approach_dir": approach_dir,
                "variant": f"m_{m}",
                "ari": ari,
                "nmi": nmi,
            }
        )
        j_scores = _cluster_jaccard(base_labels, labels_m)
        _merge_jaccard(jaccard_sum, j_scores, jaccard_count)
    session_dates = sorted(train_df["session_date"].astype(str).unique())
    sample_size = max(1, int(len(session_dates) * cfg.resample_frac))
    for i in range(cfg.resample_count):
        sample_dates = rng.choice(session_dates, size=sample_size, replace=True).tolist()
        mask = train_df["session_date"].astype(str).isin(sample_dates).to_numpy()
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        vec_subset = vectors_norm[idxs]
        graph_s, _ = _build_graph(vec_subset, base_m)
        labels_s = _leiden_labels(graph_s)
        labels_base_subset = base_labels[idxs]
        ari = float(adjusted_rand_score(labels_base_subset, labels_s))
        nmi = float(normalized_mutual_info_score(labels_base_subset, labels_s))
        stability_rows.append(
            {
                "level_id": level_id,
                "approach_dir": approach_dir,
                "variant": f"resample_{i}",
                "ari": ari,
                "nmi": nmi,
            }
        )
        j_scores = _cluster_jaccard(labels_base_subset, labels_s)
        _merge_jaccard(jaccard_sum, j_scores, jaccard_count)
    jaccard_avg = {
        k: (jaccard_sum.get(k, 0.0) / jaccard_count.get(k, 1)) for k in jaccard_sum
    }
    jaccard_values = np.array(list(jaccard_avg.values()), dtype=np.float64)
    thresh = float(np.nanmedian(jaccard_values)) if jaccard_values.size else 0.0
    stable_clusters = {cid for cid, val in jaccard_avg.items() if val >= thresh}
    meta_cols = ["vector_id", "session_date", "symbol", "ts_end_ns", "P_ref"]
    meta = train_df[meta_cols].reset_index(drop=True)
    medoids = _cluster_medoids(base_labels, vectors_norm, meta, top_n=10)
    y = train_df["y_true"].to_numpy(dtype=int)
    enrich = _cluster_enrichment(base_labels, y, rng, permutations=200)
    enrich["jaccard"] = enrich["cluster_id"].map(jaccard_avg).astype(float)
    enrich["median_sim"] = enrich["cluster_id"].map(cluster_median_sim).astype(float)
    enrich["stable"] = enrich["cluster_id"].isin(stable_clusters)
    enrich["medoids"] = enrich["cluster_id"].map(medoids)
    enrich = enrich.loc[enrich["stable"]].reset_index(drop=True)
    cluster_map: Dict[int, int] = {}
    for idx, row in train_df.reset_index(drop=True).iterrows():
        cid = int(base_labels[idx])
        if cid not in stable_clusters:
            continue
        cluster_map[int(row["vector_id"])] = cid
    clusters_train = train_df.drop(columns=["vector"]).copy()
    clusters_train["cluster_id"] = [int(base_labels[i]) for i in range(len(train_df))]
    clusters_train["stable"] = clusters_train["cluster_id"].isin(stable_clusters)
    clusters_train = clusters_train.loc[clusters_train["stable"]].reset_index(drop=True)
    stability_df = pd.DataFrame(stability_rows)
    _write_parquet(output_dir / "cluster_stability.parquet", stability_df)
    return clusters_train, cluster_map, enrich


def _run_pair(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    index_df: pd.DataFrame,
    lineage: List[Dict[str, str]],
    output_dir: Path,
    repo_root: Path,
    cfg: EvalConfig,
    level_id: str,
    approach_dir: str,
    split_id: str,
) -> None:
    pair_train = df_train[(df_train["level_id"] == level_id) & (df_train["approach_dir"] == approach_dir)].copy()
    pair_test = df_test[(df_test["level_id"] == level_id) & (df_test["approach_dir"] == approach_dir)].copy()
    if len(pair_train) == 0 or len(pair_test) == 0:
        raise ValueError("Empty train or test set for pair")
    target_id = _target_id(approach_dir, cfg.horizon)
    pair_train["y_true"] = _label_target(pair_train, approach_dir, cfg.horizon)
    pair_test["y_true"] = _label_target(pair_test, approach_dir, cfg.horizon)
    pair_train["tod_bin"] = _tod_bins(pair_train, cfg.tod_minutes)
    pair_test["tod_bin"] = _tod_bins(pair_test, cfg.tod_minutes)
    train_by_bin = {
        str(k): group["y_true"].to_numpy(dtype=int)
        for k, group in pair_train.groupby("tod_bin")
    }
    k_max = max(cfg.k_values)
    for k, vals in train_by_bin.items():
        if len(vals) < k_max:
            raise ValueError(f"Insufficient train samples in bin {k}")
    base_rate = float(pair_train["y_true"].mean())
    base_by_bin = {str(k): float(v) for k, v in pair_train.groupby("tod_bin")["y_true"].mean().items()}
    stats = fit_robust_stats(np.array(pair_train["vector"].tolist(), dtype=np.float64))
    index_dir = output_dir / "index"
    build_indices(index_df, index_dir, stats, repo_root, lineage)
    retriever = TriggerVectorRetriever(index_dir)
    pred_rows = _build_knn_predictions(
        retriever=retriever,
        test_df=pair_test,
        approach_dir=approach_dir,
        level_id=level_id,
        horizon=cfg.horizon,
        k_raw=cfg.k_raw,
        k_values=cfg.k_values,
        tau=cfg.tau,
    )
    predictions = pd.DataFrame(pred_rows)
    if len(predictions) == 0:
        raise ValueError("No predictions produced")
    leak_rows = predictions.loc[
        predictions.apply(
            lambda r: any(d >= r["session_date"] for d in r["neighbor_dates"]), axis=1
        )
    ]
    if len(leak_rows) > 0:
        raise ValueError("Leakage detected in neighbor dates")
    eval_ids = sorted(set(predictions["vector_id"].tolist()))
    eval_test = pair_test.loc[pair_test["vector_id"].isin(eval_ids)].reset_index(drop=True)
    missing_bins = sorted(set(eval_test["tod_bin"].astype(str).unique()) - set(base_by_bin.keys()))
    if missing_bins:
        raise ValueError(f"Missing train bins for eval: {missing_bins}")
    predictions["target_id"] = target_id
    predictions["split_id"] = split_id
    train_pred_rows = _build_knn_predictions(
        retriever=retriever,
        test_df=pair_train,
        approach_dir=approach_dir,
        level_id=level_id,
        horizon=cfg.horizon,
        k_raw=cfg.k_raw,
        k_values=cfg.k_values,
        tau=cfg.tau,
    )
    train_predictions = pd.DataFrame(train_pred_rows)
    if len(train_predictions) == 0:
        raise ValueError("No train predictions produced")
    train_y = pair_train["y_true"].to_numpy(dtype=int)
    if len(np.unique(train_y)) < 2:
        raise ValueError("Train labels lack both classes")
    train_vectors = np.array(pair_train["vector"].tolist(), dtype=np.float64)
    scaled_train = apply_robust_scaling(train_vectors, stats)
    logit = LogisticRegression(max_iter=200, solver="liblinear")
    logit.fit(scaled_train, train_y)
    eval_vectors = np.array(eval_test["vector"].tolist(), dtype=np.float64)
    scaled_eval = apply_robust_scaling(eval_vectors, stats)
    p_logit = logit.predict_proba(scaled_eval)[:, 1]
    logit_map = {int(vid): float(p) for vid, p in zip(eval_test["vector_id"], p_logit)}
    metrics_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(cfg.seed)
    null_preds = _build_null_predictions(rng, eval_test, cfg.k_values, train_by_bin)
    for k in cfg.k_values:
        pred_k = predictions.loc[predictions["k"] == k].copy()
        if len(pred_k) == 0:
            continue
        metrics_rows.extend(
            _collect_metrics(
                df=pred_k,
                base=base_rate,
                model="knn",
                k=k,
                target_id=target_id,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
        )
        null_df = pred_k.copy()
        null_df["p_hat"] = null_df["vector_id"].map(null_preds[k]).astype(float)
        metrics_rows.extend(
            _collect_metrics(
                df=null_df,
                base=base_rate,
                model="null_knn",
                k=k,
                target_id=target_id,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
        )
        eval_subset = eval_test.loc[eval_test["vector_id"].isin(pred_k["vector_id"])].copy()
        base_pred, tod_pred = _build_baseline_predictions(eval_subset, base_rate, base_by_bin)
        base_df = eval_subset[["vector_id", "tod_bin"]].copy()
        base_df["p_hat"] = base_pred
        base_df["y_true"] = eval_subset["y_true"].to_numpy(dtype=int)
        metrics_rows.extend(
            _collect_metrics(
                df=base_df,
                base=base_rate,
                model="base_rate",
                k=k,
                target_id=target_id,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
        )
        tod_df = eval_subset[["vector_id", "tod_bin"]].copy()
        tod_df["p_hat"] = tod_pred
        tod_df["y_true"] = eval_subset["y_true"].to_numpy(dtype=int)
        metrics_rows.extend(
            _collect_metrics(
                df=tod_df,
                base=base_rate,
                model="tod_rate",
                k=k,
                target_id=target_id,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
        )
        logit_df = eval_subset[["vector_id", "tod_bin"]].copy()
        logit_df["p_hat"] = logit_df["vector_id"].map(logit_map).astype(float)
        logit_df["y_true"] = eval_subset["y_true"].to_numpy(dtype=int)
        metrics_rows.extend(
            _collect_metrics(
                df=logit_df,
                base=base_rate,
                model="logit",
                k=k,
                target_id=target_id,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
        )
    metrics = pd.DataFrame(metrics_rows)
    pivot = _pivot_metrics(metrics)
    knn_pivot = pivot.loc[pivot["model"] == "knn"].copy()
    if len(knn_pivot) == 0:
        raise ValueError("Missing knn metrics")
    best_k_auc = int(knn_pivot.loc[knn_pivot["auc"].idxmax(), "k"])
    best_k_ll = int(knn_pivot.loc[knn_pivot["log_loss"].idxmin(), "k"])
    calib_k_values = sorted(set([best_k_auc, best_k_ll]))
    calib_table = _calibration_table(train_predictions, predictions, calib_k_values)
    break_label, reject_label = _break_reject_labels(approach_dir)
    spec = {
        "target_id": target_id,
        "level_id": level_id,
        "approach_dir": approach_dir,
        "split_id": split_id,
        "split_date": cfg.split_date,
        "label_col": f"true_outcome_h{cfg.horizon}",
        "break_label": break_label,
        "reject_label": reject_label,
        "horizon": cfg.horizon,
        "bar_seconds": 120,
        "threshold_ticks": 8,
        "k_raw": cfg.k_raw,
        "k_values": cfg.k_values,
        "tau": cfg.tau,
        "tod_minutes": cfg.tod_minutes,
        "train_rows": int(len(pair_train)),
        "test_rows": int(len(pair_test)),
        "base_rate": base_rate,
        "base_by_bin": base_by_bin,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_spec.json").write_text(json.dumps(spec, sort_keys=True))
    _write_parquet(output_dir / "metrics.parquet", metrics)
    _write_parquet(output_dir / "metrics_full.parquet", metrics)
    _write_parquet(output_dir / "predictions.parquet", predictions)
    sample_pool = predictions.loc[predictions["k"] == best_k_auc].copy()
    if len(sample_pool) == 0:
        raise ValueError("No predictions for sample")
    sample_size = min(200, len(sample_pool))
    predictions_sample = sample_pool.sample(n=sample_size, random_state=cfg.seed)
    _write_parquet(output_dir / "predictions_sample.parquet", predictions_sample)
    vectors = np.array(pair_train["vector"].tolist(), dtype=np.float64)
    scaled = apply_robust_scaling(vectors, stats)
    vectors_norm, valid = l2_normalize(scaled)
    if not np.all(valid):
        raise ValueError("Zero norm vectors after scaling")
    pair_train = pair_train.reset_index(drop=True)
    clusters_train, cluster_map, enrich = _cluster_workflow(
        train_df=pair_train,
        vectors_norm=vectors_norm,
        level_id=level_id,
        approach_dir=approach_dir,
        output_dir=output_dir,
        cfg=cfg,
    )
    _write_parquet(output_dir / "clusters_train.parquet", clusters_train)
    test_assign = _assign_clusters(
        retriever=retriever,
        test_df=eval_test,
        level_id=level_id,
        approach_dir=approach_dir,
        k_assign=cfg.cluster_assign_k,
        cluster_map=cluster_map,
    )
    _write_parquet(output_dir / "clusters_test_assignments.parquet", test_assign)
    train_cluster_rates = clusters_train.groupby("cluster_id")["y_true"].mean().to_dict()
    test_cluster_rates = (
        test_assign.merge(eval_test[["vector_id", "y_true"]], on="vector_id", how="left")
        .groupby("cluster_id")["y_true"]
        .mean()
        .to_dict()
    )
    enrich["train_rate"] = enrich["cluster_id"].map(train_cluster_rates)
    enrich["test_rate"] = enrich["cluster_id"].map(test_cluster_rates)
    _write_parquet(output_dir / "cluster_enrichment_report.parquet", enrich)
    vol_day = eval_test.groupby("session_date")[["mfe_up_ticks", "mfe_down_ticks"]].sum()
    vol_proxy = (vol_day["mfe_up_ticks"] + vol_day["mfe_down_ticks"]).to_dict()
    vol_series = eval_test["session_date"].map(vol_proxy).astype(float)
    try:
        vol_bins = pd.qcut(vol_series, 3, labels=["low", "mid", "high"])
    except ValueError as exc:
        raise ValueError("Vol regime binning failed") from exc
    eval_test = eval_test.copy()
    eval_test["vol_regime"] = vol_bins.astype(str)
    backtest = _run_backtest(
        predictions=predictions,
        test_df=eval_test,
        theta_values=cfg.theta_values,
        base_rate=base_rate,
    )
    _write_parquet(output_dir / "backtest_metrics.parquet", backtest)
    train_range = f"{pair_train['session_date'].min()}:{pair_train['session_date'].max()}"
    test_range = f"{eval_test['session_date'].min()}:{eval_test['session_date'].max()}"
    norm_stats_path = index_dir / "norm_stats.json"
    norm_stats_source = f"{norm_stats_path}:{_hash_file(norm_stats_path)}"
    index_path = index_dir / level_id / f"{approach_dir}.index"
    faiss_index_hash = _hash_file(index_path)
    header = _format_header(
        target_id=target_id,
        split_id=split_id,
        level_id=level_id,
        approach_dir=approach_dir,
        train_range=train_range,
        test_range=test_range,
        norm_stats_source=norm_stats_source,
        faiss_index_hash=faiss_index_hash,
    )
    knobs = {
        "tau": cfg.tau,
        "k_raw": cfg.k_raw,
        "k_values": cfg.k_values,
        "tod_minutes": cfg.tod_minutes,
        "similarity_metric": "inner_product",
        "normalization": "robust_l2",
        "horizon": cfg.horizon,
    }
    _write_summary_chat(
        output_dir=output_dir,
        header=header,
        knobs=knobs,
        metrics=metrics,
        base_rate=base_rate,
        n_eval=int(len(eval_test)),
        calib_table=calib_table,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dates", required=True)
    parser.add_argument("--split-date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-raw", type=int, default=500)
    parser.add_argument("--k-values", default="5,10,20,50,100")
    parser.add_argument("--tau", type=float, default=0.03)
    parser.add_argument("--tod-minutes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--cluster-m-values", default="30,40,50")
    parser.add_argument("--cluster-assign-k", type=int, default=50)
    parser.add_argument("--theta-values", default="0.55,0.6,0.65,0.7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resample-count", type=int, default=3)
    parser.add_argument("--resample-frac", type=float, default=0.8)
    parser.add_argument("--selection-path", type=Path)
    args = parser.parse_args()

    dates = expand_date_range(dates=args.dates)
    if not dates:
        raise ValueError("No dates provided")
    if args.selection_path is None:
        args.selection_path = args.repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    selection = _load_selection_rows(args.selection_path, dates)
    df_all, lineage_all = _load_vectors(args.repo_root, selection)
    if len(df_all) == 0:
        raise ValueError("No vectors loaded from lake")
    df_all = _filter_rth(df_all)
    split_date = args.split_date
    train_mask = df_all["session_date"].astype(str) <= split_date
    test_mask = df_all["session_date"].astype(str) > split_date
    df_train = df_all.loc[train_mask].reset_index(drop=True)
    df_test = df_all.loc[test_mask].reset_index(drop=True)
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Empty train or test split")
    lineage_train = [row for row in lineage_all if str(row.get("session_date")) <= split_date]
    cfg = EvalConfig(
        split_date=split_date,
        horizon=int(args.horizon),
        k_raw=int(args.k_raw),
        k_values=_parse_int_list(args.k_values),
        tau=float(args.tau),
        tod_minutes=int(args.tod_minutes),
        seed=int(args.seed),
        cluster_m_values=_parse_int_list(args.cluster_m_values),
        cluster_assign_k=int(args.cluster_assign_k),
        theta_values=_parse_float_list(args.theta_values),
        resample_count=int(args.resample_count),
        resample_frac=float(args.resample_frac),
    )
    level_ids = sorted(df_all["level_id"].astype(str).unique().tolist())
    approach_dirs = sorted(df_all["approach_dir"].astype(str).unique().tolist())
    for level_id in level_ids:
        for approach_dir in approach_dirs:
            base_dir = args.output_dir / level_id / approach_dir
            pair_train = df_train[
                (df_train["level_id"] == level_id) & (df_train["approach_dir"] == approach_dir)
            ].copy()
            pair_all = df_all[
                (df_all["level_id"] == level_id) & (df_all["approach_dir"] == approach_dir)
            ].copy()
            split_id = f"split_{split_date}"
            walk_id = f"walk_{split_date}"
            out_dir = base_dir / split_id
            _run_pair(
                df_train=df_train,
                df_test=df_test,
                index_df=pair_train,
                lineage=lineage_train,
                output_dir=out_dir,
                repo_root=args.repo_root,
                cfg=cfg,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=split_id,
            )
            out_dir_walk = base_dir / walk_id
            _run_pair(
                df_train=df_train,
                df_test=df_test,
                index_df=pair_all,
                lineage=lineage_all,
                output_dir=out_dir_walk,
                repo_root=args.repo_root,
                cfg=cfg,
                level_id=level_id,
                approach_dir=approach_dir,
                split_id=walk_id,
            )


if __name__ == "__main__":
    main()
