from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..config import load_config
from ..contracts import enforce_contract, load_avro_contract
from ..io import is_partition_complete, partition_ref, read_partition
from .index_builder import (
    APPROACH_DIRS,
    FEATURE_NAMES,
    INVARIANT_DIM,
    METADATA_FIELDS,
    METADATA_SCHEMA_HASH,
    METADATA_TYPES,
    _filter_rth,
    build_indices,
)
from .mbo_contract_day_selector import load_selection
from .normalization import RobustStats
from .query import TriggerVectorRetriever

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss-cpu or faiss-gpu required") from exc


@dataclass(frozen=True)
class PartitionPaths:
    level_id: str
    approach_dir: str
    index_path: Path
    meta_path: Path
    vectors_path: Path
    manifest_path: Path


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_feature_list(indices_dir: Path) -> Tuple[List[str], str]:
    path = indices_dir / "feature_list.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_list.json in {indices_dir}")
    payload = json.loads(path.read_text())
    vector_dim = int(payload.get("vector_dim", 0))
    features = payload.get("features", [])
    if vector_dim != INVARIANT_DIM:
        raise ValueError("Feature list dim mismatch")
    if features != FEATURE_NAMES:
        raise ValueError("Feature list mismatch")
    return features, _hash_file(path)


def _load_norm_stats(indices_dir: Path) -> Tuple[RobustStats, str]:
    path = indices_dir / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing norm_stats.json in {indices_dir}")
    payload = json.loads(path.read_text())
    median = np.array(payload.get("median", []), dtype=np.float64)
    mad = np.array(payload.get("mad", []), dtype=np.float64)
    if median.size != INVARIANT_DIM or mad.size != INVARIANT_DIM:
        raise ValueError("Norm stats dim mismatch")
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(mad)):
        raise ValueError("Non-finite norm stats")
    return RobustStats(median=median, mad=mad), _hash_file(path)


def _discover_partitions(indices_dir: Path) -> List[PartitionPaths]:
    partitions: List[PartitionPaths] = []
    for level_dir in sorted(p for p in indices_dir.iterdir() if p.is_dir()):
        for approach_dir in APPROACH_DIRS:
            partitions.append(
                PartitionPaths(
                    level_id=level_dir.name,
                    approach_dir=approach_dir,
                    index_path=level_dir / f"{approach_dir}.index",
                    meta_path=level_dir / f"metadata_{approach_dir}.npz",
                    vectors_path=level_dir / f"vectors_{approach_dir}.npy",
                    manifest_path=level_dir / f"manifest_{approach_dir}.json",
                )
            )
    return partitions


def _expected_level_files() -> List[str]:
    files: List[str] = []
    for approach_dir in APPROACH_DIRS:
        files.extend(
            [
                f"{approach_dir}.index",
                f"metadata_{approach_dir}.npz",
                f"vectors_{approach_dir}.npy",
                f"manifest_{approach_dir}.json",
            ]
        )
    return sorted(files)


def _check_level_dir(level_dir: Path) -> List[str]:
    failures: List[str] = []
    expected = set(_expected_level_files())
    actual = {p.name for p in level_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            failures.append(f"Missing files in {level_dir.name}: {missing}")
        if extra:
            failures.append(f"Extra files in {level_dir.name}: {extra}")
    return failures


def _load_selection_map(selection_path: Path) -> Dict[str, str]:
    df = load_selection(selection_path)
    df = df.loc[df["selected_symbol"] != ""].copy()
    df["session_date"] = df["session_date"].astype(str)
    df["selected_symbol"] = df["selected_symbol"].astype(str)
    return {row.session_date: row.selected_symbol for row in df.itertuples(index=False)}


def _rth_window_ns(session_date: str) -> Tuple[int, int]:
    start = pd.Timestamp(f"{session_date} 09:30:00", tz="America/New_York")
    end = pd.Timestamp(f"{session_date} 12:30:00", tz="America/New_York")
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


def _validate_metadata_types(meta: Dict[str, np.ndarray], failures: List[str]) -> None:
    rng = np.random.default_rng(7)
    for name, kind in METADATA_TYPES.items():
        values = meta.get(name)
        if values is None:
            failures.append(f"Missing metadata field: {name}")
            continue
        sample_count = min(64, len(values))
        if sample_count == 0:
            continue
        idx = rng.choice(len(values), size=sample_count, replace=False)
        sample = values[idx]
        for val in sample:
            if val is None:
                if kind != "int_null":
                    failures.append(f"Null in field {name}")
                continue
            if isinstance(val, np.generic):
                val = val.item()
            if kind == "int":
                if not isinstance(val, (int, np.integer)):
                    failures.append(f"Type mismatch in field {name}")
                    break
            elif kind == "float":
                if not isinstance(val, (float, np.floating)):
                    failures.append(f"Type mismatch in field {name}")
                    break
            elif kind == "str":
                if not isinstance(val, str):
                    failures.append(f"Type mismatch in field {name}")
                    break
            elif kind == "int_null":
                if not isinstance(val, (int, np.integer)):
                    failures.append(f"Type mismatch in field {name}")
                    break


def _check_time_sanity(meta: Dict[str, np.ndarray], failures: List[str]) -> None:
    session_dates = np.asarray(meta["session_date"], dtype=object)
    ts_end = np.asarray(meta["ts_end_ns"], dtype=np.int64)
    for session_date in sorted(set(session_dates.tolist())):
        mask = session_dates == session_date
        if not np.any(mask):
            continue
        start_ns, end_ns = _rth_window_ns(session_date)
        ts_values = ts_end[mask]
        if np.any(ts_values < start_ns) or np.any(ts_values > end_ns):
            failures.append(f"Timestamp out of RTH window for {session_date}")
        ts_local = pd.to_datetime(ts_values, utc=True).tz_convert("America/New_York")
        local_dates = np.array([d.strftime("%Y-%m-%d") for d in ts_local])
        if np.any(local_dates != session_date):
            failures.append(f"Timezone mismatch for {session_date}")


def _check_constant_dims(vectors: np.ndarray, mad: np.ndarray, failures: List[str]) -> None:
    zero_idx = np.where(mad == 0.0)[0]
    if len(zero_idx) != 116:
        failures.append(f"Constant dim count mismatch: {len(zero_idx)}")
        return
    if vectors.size == 0:
        return
    max_abs = np.max(np.abs(vectors[:, zero_idx]))
    if max_abs > 1e-6:
        failures.append("Constant dims not zero after scaling")


def _check_norms(vectors: np.ndarray, failures: List[str]) -> None:
    if vectors.size == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        failures.append("Zero norm vectors in index")
    if np.any(norms > 1.001):
        failures.append("Vector norms exceed 1.001")
    delta = np.abs(norms - 1.0)
    bad = int(np.sum(delta > 1e-4))
    if bad > max(1, int(math.ceil(0.0001 * len(norms)))):
        failures.append("L2 norm tolerance exceeded")


def _check_similarity_range(vectors: np.ndarray, failures: List[str]) -> None:
    if vectors.size == 0:
        return
    rng = np.random.default_rng(11)
    n = len(vectors)
    pairs = min(1000, n)
    idx_a = rng.integers(0, n, size=pairs)
    idx_b = rng.integers(0, n, size=pairs)
    sims = np.sum(vectors[idx_a] * vectors[idx_b], axis=1)
    if np.any(sims > 1.001) or np.any(sims < -1.001):
        failures.append("Similarity range violation")


def _check_id_alignment(
    index: faiss.Index,
    vectors: np.ndarray,
    failures: List[str],
) -> None:
    if vectors.size == 0:
        return
    rng = np.random.default_rng(13)
    n = len(vectors)
    sample = min(200, n)
    ids = rng.choice(n, size=sample, replace=False)
    for idx in ids:
        try:
            v_faiss = index.reconstruct(int(idx))
        except Exception:
            failures.append("Index reconstruct failed")
            return
        v_store = vectors[int(idx)]
        sim = float(np.dot(v_faiss, v_store))
        if sim < 0.999999:
            failures.append("Index id alignment mismatch")
            return


def _duplicate_rate(vectors: np.ndarray) -> float:
    if vectors.size == 0:
        return 0.0
    view = np.ascontiguousarray(vectors).view(
        np.dtype((np.void, vectors.dtype.itemsize * vectors.shape[1]))
    )
    unique = np.unique(view).shape[0]
    return 1.0 - (unique / float(len(vectors)))


def _near_duplicate_rate(index: faiss.Index, vectors: np.ndarray) -> float:
    if vectors.size == 0:
        return 0.0
    rng = np.random.default_rng(17)
    n = len(vectors)
    sample = min(200, n)
    ids = rng.choice(n, size=sample, replace=False)
    query = vectors[ids]
    sims, idxs = index.search(query, 2)
    near = 0
    for i in range(sample):
        if idxs[i][0] == ids[i]:
            sim = sims[i][1] if idxs[i][1] >= 0 else -1.0
        else:
            sim = sims[i][0]
        if sim > 0.99999:
            near += 1
    return near / float(sample)


def _retrieval_sanity(
    index: faiss.Index,
    vectors: np.ndarray,
    failures: List[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if vectors.size == 0:
        return metrics
    rng = np.random.default_rng(19)
    n = len(vectors)
    sample = min(200, n)
    ids = rng.choice(n, size=sample, replace=False)
    query = vectors[ids]
    sims, idxs = index.search(query, min(11, n))
    self_hits = 0
    top1 = []
    top10_mean = []
    gap = []
    for i in range(sample):
        if idxs[i][0] == ids[i]:
            self_hits += 1
        if np.any(np.diff(sims[i]) > 1e-6):
            failures.append("Similarity order violation")
            break
        filt = [s for j, s in enumerate(sims[i]) if idxs[i][j] != ids[i]]
        if not filt:
            continue
        top1.append(float(filt[0]))
        top10_mean.append(float(np.mean(filt[: min(10, len(filt))])))
        gap.append(float(filt[0] - np.mean(filt[: min(10, len(filt))])))
    metrics["self_hit_rate"] = self_hits / float(sample)
    if self_hits != sample:
        failures.append("Self-query mismatch")
    if top1:
        metrics["top1_mean"] = float(np.mean(top1))
        metrics["top1_std"] = float(np.std(top1))
    if top10_mean:
        metrics["top10_mean"] = float(np.mean(top10_mean))
        metrics["top10_std"] = float(np.std(top10_mean))
    if gap:
        metrics["gap_mean"] = float(np.mean(gap))
        metrics["gap_std"] = float(np.std(gap))
        if metrics["gap_mean"] < 1e-5:
            failures.append("Similarity gap collapse")
    return metrics


def _approx_recall(
    index: faiss.Index,
    vectors: np.ndarray,
    failures: List[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if vectors.size == 0:
        return metrics
    if isinstance(index, faiss.IndexFlat):
        return metrics
    exact = faiss.IndexFlatIP(vectors.shape[1])
    exact.add(vectors)
    rng = np.random.default_rng(23)
    n = len(vectors)
    sample = min(300, n)
    ids = rng.choice(n, size=sample, replace=False)
    query = vectors[ids]
    k_values = [10, 50]
    for k in k_values:
        k_eff = min(k, n)
        sims_a, idx_a = index.search(query, k_eff)
        sims_b, idx_b = exact.search(query, k_eff)
        hit = 0
        total = 0
        for i in range(sample):
            set_a = set(idx_a[i].tolist())
            set_b = set(idx_b[i].tolist())
            total += len(set_b)
            hit += len(set_a.intersection(set_b))
        recall = hit / float(total) if total else 0.0
        metrics[f"recall_{k}"] = recall
    if metrics.get("recall_10", 1.0) < 0.95 or metrics.get("recall_50", 1.0) < 0.9:
        failures.append("Approx recall below threshold")
    return metrics


def _load_vectors_from_partitions(
    repo_root: Path, input_partitions: List[Dict[str, str]]
) -> pd.DataFrame:
    cfg = load_config(repo_root, repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
    if not input_partitions:
        raise ValueError("Missing input partitions for rebuild")
    dataset_key = input_partitions[0]["dataset"]
    contract_path = repo_root / cfg.dataset(dataset_key).contract
    contract = load_avro_contract(contract_path)
    frames = []
    for part in input_partitions:
        symbol = str(part["symbol"])
        session_date = str(part["session_date"])
        ref = partition_ref(cfg, dataset_key, symbol, session_date)
        if not is_partition_complete(ref):
            raise FileNotFoundError(f"Missing partition: {dataset_key} symbol={symbol} dt={session_date}")
        df = read_partition(ref)
        if len(df) == 0:
            continue
        df = enforce_contract(df, contract)
        frames.append(df)
    if not frames:
        raise ValueError("No vectors loaded for rebuild")
    df = pd.concat(frames, ignore_index=True)
    sort_cols = ["level_id", "approach_dir", "session_date", "symbol", "vector_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df = _filter_rth(df)
    if len(df) == 0:
        raise ValueError("No vectors inside RTH window")
    return df


def _compare_rebuild(
    indices_dir: Path,
    repo_root: Path,
    stats: RobustStats,
    partitions: List[PartitionPaths],
    failures: List[str],
) -> None:
    if not partitions:
        return
    first_manifest = json.loads(partitions[0].manifest_path.read_text())
    input_partitions = first_manifest.get("input_partitions", [])
    df = _load_vectors_from_partitions(repo_root, input_partitions)
    with tempfile.TemporaryDirectory(dir=indices_dir) as tmp:
        rebuild_dir = Path(tmp) / "rebuild"
        build_indices(df, rebuild_dir, stats, repo_root, input_partitions)
        for part in partitions:
            orig_meta = part.meta_path
            orig_vec = part.vectors_path
            rebuild_meta = rebuild_dir / part.level_id / f"metadata_{part.approach_dir}.npz"
            rebuild_vec = rebuild_dir / part.level_id / f"vectors_{part.approach_dir}.npy"
            if _hash_file(orig_meta) != _hash_file(rebuild_meta):
                failures.append("Metadata hash mismatch on rebuild")
                break
            if _hash_file(orig_vec) != _hash_file(rebuild_vec):
                failures.append("Vectors hash mismatch on rebuild")
                break
            orig_index = faiss.read_index(str(part.index_path))
            new_index = faiss.read_index(str(rebuild_dir / part.level_id / f"{part.approach_dir}.index"))
            vectors = np.load(orig_vec)
            if vectors.size == 0:
                continue
            rng = np.random.default_rng(29)
            n = len(vectors)
            sample = min(100, n)
            ids = rng.choice(n, size=sample, replace=False)
            query = vectors[ids]
            k = min(10, n)
            sims_a, idx_a = orig_index.search(query, k)
            sims_b, idx_b = new_index.search(query, k)
            matches = 0
            for i in range(sample):
                if np.array_equal(idx_a[i], idx_b[i]) and np.allclose(
                    sims_a[i], sims_b[i], atol=1e-6
                ):
                    matches += 1
            rate = matches / float(sample)
            if rate < 0.95:
                failures.append("Rebuild neighbor mismatch")
                break


def _cold_load_worker(indices_dir: str, queue: mp.Queue) -> None:
    try:
        retriever = TriggerVectorRetriever(Path(indices_dir))
        first_vec = None
        level_id = None
        approach_dir = None
        for level_dir in sorted(Path(indices_dir).iterdir()):
            if not level_dir.is_dir():
                continue
            for approach in APPROACH_DIRS:
                vec_path = level_dir / f"vectors_{approach}.npy"
                if vec_path.exists():
                    vecs = np.load(vec_path)
                    if vecs.size > 0:
                        first_vec = vecs[0]
                        level_id = level_dir.name
                        approach_dir = approach
                        break
            if first_vec is not None:
                break
        if first_vec is None:
            queue.put("No vectors available for cold load")
            return
        retriever.find_similar(level_id, approach_dir, first_vec, k=5)
        queue.put("")
    except Exception as exc:
        queue.put(str(exc))


def _concurrent_queries(retriever: TriggerVectorRetriever, indices_dir: Path) -> str:
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for level_dir in sorted(indices_dir.iterdir()):
            if not level_dir.is_dir():
                continue
            for approach in APPROACH_DIRS:
                vec_path = level_dir / f"vectors_{approach}.npy"
                if not vec_path.exists():
                    continue
                vecs = np.load(vec_path)
                if vecs.size == 0:
                    continue
                for v in vecs[:8]:
                    tasks.append(
                        pool.submit(
                            retriever.find_similar, level_dir.name, approach, v, 5
                        )
                    )
        for fut in as_completed(tasks):
            try:
                fut.result()
            except Exception as exc:
                return str(exc)
    return ""


def _denylist_check(failures: List[str]) -> None:
    deny = set(METADATA_FIELDS + ["vector", "vector_dim"])
    for name in FEATURE_NAMES:
        if name in deny:
            failures.append(f"Feature list contains forbidden field: {name}")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run index QA checks.")
    parser.add_argument("--indices-dir", type=Path, required=True)
    parser.add_argument("--selection-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    indices_dir = args.indices_dir
    selection_path = args.selection_path
    report_path = args.report_path or indices_dir / "index_qa_report.json"

    report: Dict[str, object] = {
        "indices_dir": str(indices_dir),
        "run_time": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "partitions": {},
        "global_failures": [],
    }

    features, feature_hash = _load_feature_list(indices_dir)
    stats, stats_hash = _load_norm_stats(indices_dir)
    if len(features) != INVARIANT_DIM:
        report["global_failures"].append("Feature list dim mismatch")
    _denylist_check(report["global_failures"])

    selection_map = _load_selection_map(selection_path)
    partitions = _discover_partitions(indices_dir)

    level_dirs = {p.index_path.parent for p in partitions}
    for level_dir in level_dirs:
        report["global_failures"].extend(_check_level_dir(level_dir))

    all_session_dates: List[str] = []
    all_true_outcome: List[str] = []
    seen_pairs: Dict[Tuple[str, str], int] = {}

    for part in partitions:
        key = f"{part.level_id}:{part.approach_dir}"
        part_report: Dict[str, object] = {"failures": [], "metrics": {}}

        for path in [part.index_path, part.meta_path, part.vectors_path, part.manifest_path]:
            if not path.exists():
                part_report["failures"].append(f"Missing artifact: {path.name}")

        if part_report["failures"]:
            report["partitions"][key] = part_report
            continue

        manifest = json.loads(part.manifest_path.read_text())
        if manifest.get("level_id") != part.level_id:
            part_report["failures"].append("Manifest level_id mismatch")
        if manifest.get("approach_dir") != part.approach_dir:
            part_report["failures"].append("Manifest approach_dir mismatch")
        if int(manifest.get("vector_dim", 0)) != INVARIANT_DIM:
            part_report["failures"].append("Manifest dim mismatch")
        if manifest.get("norm_stats_sha256") != stats_hash:
            part_report["failures"].append("Norm stats hash mismatch")
        if manifest.get("feature_list_sha256") != feature_hash:
            part_report["failures"].append("Feature list hash mismatch")
        if manifest.get("metadata_schema_sha256") != METADATA_SCHEMA_HASH:
            part_report["failures"].append("Metadata schema hash mismatch")

        index = faiss.read_index(str(part.index_path))
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            part_report["failures"].append("Index metric mismatch")
        if index.d != INVARIANT_DIM:
            part_report["failures"].append("Index dim mismatch")

        meta = np.load(part.meta_path, allow_pickle=True)
        meta_dict = {k: meta[k] for k in meta.files}
        expected_fields = ["id"] + METADATA_FIELDS
        if sorted(meta_dict.keys()) != sorted(expected_fields):
            part_report["failures"].append("Metadata fields mismatch")

        vectors = np.load(part.vectors_path)
        if vectors.ndim != 2 or vectors.shape[1] != INVARIANT_DIM:
            part_report["failures"].append("Vectors shape mismatch")
        if vectors.dtype != np.float32:
            part_report["failures"].append("Vectors dtype mismatch")

        n_faiss = int(index.ntotal)
        n_meta = int(len(meta_dict.get("id", [])))
        n_vec = int(len(vectors))
        part_report["metrics"]["n_faiss"] = n_faiss
        part_report["metrics"]["n_meta"] = n_meta
        part_report["metrics"]["n_vec"] = n_vec

        if n_faiss != n_meta or n_faiss != n_vec:
            part_report["failures"].append("Count mismatch")
        if int(manifest.get("count", -1)) != n_faiss:
            part_report["failures"].append("Manifest count mismatch")

        if n_meta > 0:
            level_vals = np.unique(meta_dict["level_id"])
            approach_vals = np.unique(meta_dict["approach_dir"])
            if not (len(level_vals) == 1 and level_vals[0] == part.level_id):
                part_report["failures"].append("Level id impurity")
            if not (len(approach_vals) == 1 and approach_vals[0] == part.approach_dir):
                part_report["failures"].append("Approach dir impurity")

        if n_meta > 0:
            ids = meta_dict["vector_id"]
            if len(np.unique(ids)) != len(ids):
                part_report["failures"].append("Duplicate vector_id in partition")

        _validate_metadata_types(meta_dict, part_report["failures"])
        _check_time_sanity(meta_dict, part_report["failures"])
        _check_constant_dims(vectors, stats.mad, part_report["failures"])
        _check_norms(vectors, part_report["failures"])
        _check_similarity_range(vectors, part_report["failures"])
        _check_id_alignment(index, vectors, part_report["failures"])

        dup_rate = _duplicate_rate(vectors)
        near_dup = _near_duplicate_rate(index, vectors)
        part_report["metrics"]["dup_rate"] = dup_rate
        part_report["metrics"]["near_dup_rate"] = near_dup
        if dup_rate > 0.1:
            part_report["failures"].append("High duplicate rate")
        if near_dup > 0.2:
            part_report["failures"].append("High near-duplicate rate")

        for name, value in _retrieval_sanity(index, vectors, part_report["failures"]).items():
            part_report["metrics"][name] = value
        for name, value in _approx_recall(index, vectors, part_report["failures"]).items():
            part_report["metrics"][name] = value

        all_session_dates.extend(meta_dict.get("session_date", []).tolist())
        all_true_outcome.extend(meta_dict.get("true_outcome", []).tolist())
        for s, sym in zip(meta_dict.get("session_date", []), meta_dict.get("symbol", [])):
            key_pair = (str(s), str(sym))
            seen_pairs[key_pair] = seen_pairs.get(key_pair, 0) + 1

        report["partitions"][key] = part_report

    for session_date, symbol in seen_pairs:
        expected_symbol = selection_map.get(session_date)
        if expected_symbol != symbol:
            report["global_failures"].append(
                f"Selection purity mismatch: {session_date} {symbol}"
            )
            break

    if all_session_dates:
        date_counts: Dict[str, int] = {}
        for d in all_session_dates:
            date_counts[str(d)] = date_counts.get(str(d), 0) + 1
        report["date_counts"] = date_counts
        expected_dates = set(selection_map.keys())
        missing = sorted(expected_dates - set(date_counts.keys()))
        if missing:
            report["missing_dates"] = missing
        counts = np.array(list(date_counts.values()), dtype=np.float64)
        median = float(np.median(counts))
        if median > 0:
            spikes = [d for d, c in date_counts.items() if c > 5 * median]
            if spikes:
                report["global_failures"].append(f"Date count spikes: {spikes}")

    if all_true_outcome:
        labels, counts = np.unique(np.array(all_true_outcome, dtype=object), return_counts=True)
        total = float(np.sum(counts))
        if total > 0:
            max_share = float(np.max(counts) / total)
            report["label_max_share"] = max_share
            if max_share > 0.98:
                report["global_failures"].append("Label dominance too high")

    rebuild_failures: List[str] = []
    _compare_rebuild(indices_dir, args.repo_root, stats, partitions, rebuild_failures)
    report["global_failures"].extend(rebuild_failures)

    queue: mp.Queue = mp.Queue()
    proc = mp.get_context("spawn").Process(
        target=_cold_load_worker, args=(str(indices_dir), queue)
    )
    proc.start()
    proc.join(timeout=120)
    if proc.is_alive():
        proc.terminate()
        report["global_failures"].append("Cold load timeout")
    else:
        msg = queue.get() if not queue.empty() else ""
        if msg:
            report["global_failures"].append(f"Cold load failure: {msg}")

    retriever_error = ""
    try:
        retriever = TriggerVectorRetriever(indices_dir)
        retriever_error = _concurrent_queries(retriever, indices_dir)
    except Exception as exc:
        retriever_error = str(exc)
    if retriever_error:
        report["global_failures"].append(f"Concurrent query failure: {retriever_error}")

    passed = not report["global_failures"] and all(
        not part["failures"] for part in report["partitions"].values()
    )
    report["passed"] = passed
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    status = "PASS" if passed else "FAIL"
    print(f"Index QA {status}: partitions={len(partitions)} report={report_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
