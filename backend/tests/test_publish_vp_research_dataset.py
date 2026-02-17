from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import scripts.publish_vp_research_dataset as publish_vp_research_dataset


def _write_source_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source_cache"
    source.mkdir(parents=True, exist_ok=False)

    bins_rows = [
        {
            "ts_ns": 1_000_000_000,
            "bin_seq": 0,
            "bin_start_ns": 900_000_000,
            "bin_end_ns": 1_000_000_000,
            "bin_event_count": 10,
            "event_id": 100,
            "mid_price": 100.0,
            "spot_ref_price_int": 100_000,
            "best_bid_price_int": 99_999,
            "best_ask_price_int": 100_001,
            "book_valid": True,
        },
        {
            "ts_ns": 1_100_000_000,
            "bin_seq": 1,
            "bin_start_ns": 1_000_000_000,
            "bin_end_ns": 1_100_000_000,
            "bin_event_count": 12,
            "event_id": 112,
            "mid_price": 100.5,
            "spot_ref_price_int": 100_500,
            "best_bid_price_int": 100_499,
            "best_ask_price_int": 100_501,
            "book_valid": True,
        },
    ]
    bins_tbl = pa.Table.from_pylist(bins_rows)
    pq.write_table(bins_tbl, source / "bins.parquet", compression="zstd")

    bucket_rows = [
        {
            "ts_ns": 1_000_000_000,
            "bin_seq": 0,
            "bin_start_ns": 900_000_000,
            "bin_end_ns": 1_000_000_000,
            "bin_event_count": 10,
            "event_id": 100,
            "mid_price": 100.0,
            "spot_ref_price_int": 100_000,
            "best_bid_price_int": 99_999,
            "best_ask_price_int": 100_001,
            "book_valid": True,
            "k": -1,
            "spectrum_state_code": -1,
            "last_event_id": 101,
            "pressure_variant": 1.0,
            "vacuum_variant": 2.0,
            "add_mass": 3.0,
            "pull_mass": 4.0,
            "fill_mass": 5.0,
            "rest_depth": 6.0,
            "v_add": 7.0,
            "v_pull": 8.0,
            "v_fill": 9.0,
            "v_rest_depth": 10.0,
            "a_add": 11.0,
            "a_pull": 12.0,
            "a_fill": 13.0,
            "a_rest_depth": 14.0,
            "j_add": 15.0,
            "j_pull": 16.0,
            "j_fill": 17.0,
            "j_rest_depth": 18.0,
            "spectrum_score": -0.2,
            "proj_score_h250": -0.1,
            "proj_score_h500": -0.05,
        },
        {
            "ts_ns": 1_000_000_000,
            "bin_seq": 0,
            "bin_start_ns": 900_000_000,
            "bin_end_ns": 1_000_000_000,
            "bin_event_count": 10,
            "event_id": 100,
            "mid_price": 100.0,
            "spot_ref_price_int": 100_000,
            "best_bid_price_int": 99_999,
            "best_ask_price_int": 100_001,
            "book_valid": True,
            "k": 0,
            "spectrum_state_code": 0,
            "last_event_id": 102,
            "pressure_variant": 2.0,
            "vacuum_variant": 1.0,
            "add_mass": 3.0,
            "pull_mass": 4.0,
            "fill_mass": 5.0,
            "rest_depth": 6.0,
            "v_add": 7.0,
            "v_pull": 8.0,
            "v_fill": 9.0,
            "v_rest_depth": 10.0,
            "a_add": 11.0,
            "a_pull": 12.0,
            "a_fill": 13.0,
            "a_rest_depth": 14.0,
            "j_add": 15.0,
            "j_pull": 16.0,
            "j_fill": 17.0,
            "j_rest_depth": 18.0,
            "spectrum_score": 0.1,
            "proj_score_h250": 0.2,
            "proj_score_h500": 0.3,
        },
    ]
    bucket_tbl = pa.Table.from_pylist(bucket_rows)
    pq.write_table(bucket_tbl, source / "buckets.parquet", compression="zstd")

    (source / "manifest.json").write_text(
        json.dumps({"rows": {"bins": 2, "buckets": 2}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return source


def test_publish_dataset_splits_and_freezes_base_data(tmp_path: Path) -> None:
    source = _write_source_dataset(tmp_path)
    research_root = tmp_path / "research"

    result = publish_vp_research_dataset.publish_dataset(
        source_dir=source,
        dataset_id="ds1",
        research_root=research_root,
        agents=["alpha", "beta"],
    )

    immutable = Path(result["immutable_dataset_dir"])
    experiment = Path(result["experiment_dataset_dir"])

    assert immutable.exists()
    assert experiment.exists()

    clean_schema = pq.read_schema(immutable / "grid_clean.parquet").names
    assert "proj_score_h250" not in clean_schema
    assert "proj_score_h500" not in clean_schema
    assert "spectrum_score" in clean_schema

    projection_schema = pq.read_schema(experiment / "projection_seed.parquet").names
    assert projection_schema == ["ts_ns", "bin_seq", "k", "proj_score_h250", "proj_score_h500"]

    for path in [
        immutable,
        immutable / "bins.parquet",
        immutable / "grid_clean.parquet",
        immutable / "manifest.json",
        immutable / "checksums.json",
    ]:
        mode = path.stat().st_mode
        assert (mode & stat.S_IWUSR) == 0
        assert (mode & stat.S_IWGRP) == 0
        assert (mode & stat.S_IWOTH) == 0

    for agent in ("alpha", "beta"):
        workspace = experiment / "agents" / agent
        base_link = workspace / "data" / "base_immutable"
        projection_copy = workspace / "data" / "projection_experiment.parquet"
        assert workspace.exists()
        assert base_link.is_symlink()
        assert base_link.resolve() == immutable.resolve()
        assert projection_copy.exists()


def test_add_agents_appends_new_workspaces(tmp_path: Path) -> None:
    source = _write_source_dataset(tmp_path)
    research_root = tmp_path / "research"

    publish_vp_research_dataset.publish_dataset(
        source_dir=source,
        dataset_id="ds2",
        research_root=research_root,
        agents=[],
    )

    result = publish_vp_research_dataset.add_agents(
        dataset_id="ds2",
        research_root=research_root,
        agents=["gamma", "delta"],
    )

    experiment = Path(result["experiment_dataset_dir"])
    assert (experiment / "agents" / "gamma").exists()
    assert (experiment / "agents" / "delta").exists()

    manifest_payload = json.loads(
        (experiment / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest_payload["agent_workspaces"]) == 2
