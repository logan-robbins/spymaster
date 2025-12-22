"""
Tests for Run Manifest Manager.

Verifies Phase 1 institutional hygiene per PLAN.md §1.2:
- Run manifest creation with metadata
- Config snapshot capture
- Schema version tracking
- Bronze/Gold file tracking
- Run status lifecycle (started → completed/crashed)
- Historical manifest loading
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from src.run_manifest_manager import (
    RunManifestManager, RunManifest, RunStatus, RunMode
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data lake."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def manifest_manager(temp_data_dir):
    """Create manifest manager with temp directory."""
    return RunManifestManager(data_root=temp_data_dir, mode=RunMode.LIVE)


def test_manifest_manager_initialization(manifest_manager, temp_data_dir):
    """Test manifest manager creates _meta/runs directory."""
    meta_dir = os.path.join(temp_data_dir, '_meta', 'runs')
    assert os.path.exists(meta_dir)
    assert os.path.isdir(meta_dir)


def test_start_run_creates_manifest(manifest_manager, temp_data_dir):
    """Test starting a run creates manifest.json."""
    run_id = manifest_manager.start_run()
    
    assert run_id is not None
    assert manifest_manager.current_manifest is not None
    assert manifest_manager.current_manifest.run_id == run_id
    
    # Verify run directory exists
    run_dir = os.path.join(temp_data_dir, '_meta', 'runs', run_id)
    assert os.path.exists(run_dir)
    
    # Verify manifest.json exists
    manifest_path = os.path.join(run_dir, 'manifest.json')
    assert os.path.exists(manifest_path)


def test_run_id_format(manifest_manager):
    """Test run ID has correct format."""
    run_id = manifest_manager.start_run()
    
    # Format: YYYY-MM-DD_HHMMSS_microseconds_{mode}_{hash}
    parts = run_id.split('_')
    assert len(parts) >= 5
    
    # Check date part
    date_part = parts[0]
    assert len(date_part) == 10  # YYYY-MM-DD
    
    # Check time part
    time_part = parts[1]
    assert len(time_part) == 6  # HHMMSS
    
    # Check microseconds part
    microseconds_part = parts[2]
    assert len(microseconds_part) == 6  # microseconds
    
    # Check mode
    mode_part = parts[3]
    assert mode_part == 'live'
    
    # Check hash
    hash_part = parts[4]
    assert len(hash_part) == 6


def test_manifest_initial_status(manifest_manager):
    """Test manifest starts with STARTED status."""
    manifest_manager.start_run()
    
    manifest = manifest_manager.current_manifest
    assert manifest.status == RunStatus.STARTED.value
    assert manifest.start_time is not None
    assert manifest.end_time is None


def test_config_snapshot_created(manifest_manager, temp_data_dir):
    """Test config snapshot is written to disk."""
    run_id = manifest_manager.start_run()
    
    config_path = os.path.join(
        temp_data_dir, '_meta', 'runs', run_id, 'config_snapshot.json'
    )
    assert os.path.exists(config_path)
    
    # Verify it's valid JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check some expected config keys
    assert 'W_b' in config
    assert 'MONITOR_BAND' in config
    assert 'w_L' in config


def test_schema_snapshot_created(manifest_manager, temp_data_dir):
    """Test schema versions are captured."""
    run_id = manifest_manager.start_run()
    
    schema_dir = os.path.join(
        temp_data_dir, '_meta', 'runs', run_id, 'schemas'
    )
    assert os.path.exists(schema_dir)
    
    # Schema registry integration is optional, so check if file exists
    versions_path = os.path.join(schema_dir, 'versions.json')
    if os.path.exists(versions_path):
        with open(versions_path, 'r') as f:
            schemas = json.load(f)
        
        # Should have schema information
        assert isinstance(schemas, dict)


def test_track_bronze_file(manifest_manager):
    """Test tracking Bronze file paths."""
    manifest_manager.start_run()
    
    file_path = os.path.join(
        manifest_manager.data_root,
        'bronze/futures/trades/symbol=ES/date=2025-12-22/hour=14/part-001.parquet'
    )
    
    manifest_manager.track_bronze_file(file_path)
    
    assert len(manifest_manager.current_manifest.bronze_files) == 1
    
    # Should be relative path
    tracked_path = manifest_manager.current_manifest.bronze_files[0]
    assert tracked_path.startswith('bronze/')


def test_track_multiple_bronze_files(manifest_manager):
    """Test tracking multiple Bronze files."""
    manifest_manager.start_run()
    
    files = [
        'bronze/futures/trades/part-001.parquet',
        'bronze/futures/mbp10/part-001.parquet',
        'bronze/options/trades/part-001.parquet'
    ]
    
    for file in files:
        full_path = os.path.join(manifest_manager.data_root, file)
        manifest_manager.track_bronze_file(full_path)
    
    assert len(manifest_manager.current_manifest.bronze_files) == 3


def test_track_gold_file(manifest_manager):
    """Test tracking Gold file paths."""
    manifest_manager.start_run()
    
    file_path = os.path.join(
        manifest_manager.data_root,
        'gold/levels/signals/date=2025-12-22/part-001.parquet'
    )
    
    manifest_manager.track_gold_file(file_path)
    
    assert len(manifest_manager.current_manifest.gold_files) == 1


def test_update_event_count(manifest_manager):
    """Test event count tracking."""
    manifest_manager.start_run()
    
    manifest_manager.update_event_count('futures.trades', 1000)
    manifest_manager.update_event_count('futures.mbp10', 500)
    manifest_manager.update_event_count('futures.trades', 200)  # increment
    
    counts = manifest_manager.current_manifest.event_counts
    assert counts['futures.trades'] == 1200
    assert counts['futures.mbp10'] == 500


def test_complete_run(manifest_manager, temp_data_dir):
    """Test marking run as completed."""
    run_id = manifest_manager.start_run()
    manifest_manager.complete_run(RunStatus.COMPLETED)
    
    manifest = manifest_manager.current_manifest
    assert manifest.status == RunStatus.COMPLETED.value
    assert manifest.end_time is not None
    
    # Verify written to disk
    manifest_path = os.path.join(
        temp_data_dir, '_meta', 'runs', run_id, 'manifest.json'
    )
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    assert data['status'] == RunStatus.COMPLETED.value
    assert data['end_time'] is not None


def test_mark_crashed(manifest_manager, temp_data_dir):
    """Test marking run as crashed with error."""
    run_id = manifest_manager.start_run()
    error_msg = "Connection timeout to Polygon API"
    
    manifest_manager.mark_crashed(error_msg)
    
    manifest = manifest_manager.current_manifest
    assert manifest.status == RunStatus.CRASHED.value
    assert manifest.error_message == error_msg
    
    # Verify written to disk
    manifest_path = os.path.join(
        temp_data_dir, '_meta', 'runs', run_id, 'manifest.json'
    )
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    assert data['status'] == RunStatus.CRASHED.value
    assert data['error_message'] == error_msg


def test_load_manifest(manifest_manager):
    """Test loading manifest by run ID."""
    run_id = manifest_manager.start_run()
    manifest_manager.track_bronze_file(
        os.path.join(manifest_manager.data_root, 'bronze/test.parquet')
    )
    manifest_manager.complete_run()
    
    # Load from disk
    loaded = manifest_manager.load_manifest(run_id)
    
    assert loaded is not None
    assert loaded.run_id == run_id
    assert loaded.status == RunStatus.COMPLETED.value
    assert len(loaded.bronze_files) == 1


def test_load_nonexistent_manifest(manifest_manager):
    """Test loading nonexistent manifest returns None."""
    loaded = manifest_manager.load_manifest('nonexistent_run_id')
    assert loaded is None


def test_list_runs(manifest_manager):
    """Test listing all runs."""
    # Create multiple runs
    run_ids = []
    for i in range(3):
        run_id = manifest_manager.start_run()
        run_ids.append(run_id)
        manifest_manager.complete_run()
        
        # Need new manager for each run
        manifest_manager.current_manifest = None
    
    # List all runs
    runs = manifest_manager.list_runs()
    
    assert len(runs) >= 3
    # Should be sorted by start_time descending (most recent first)
    for i in range(len(runs) - 1):
        assert runs[i].start_time >= runs[i + 1].start_time


def test_list_runs_filtered_by_status(manifest_manager):
    """Test filtering runs by status."""
    # Create runs with different statuses
    manifest_manager.start_run()
    manifest_manager.complete_run(RunStatus.COMPLETED)
    manifest_manager.current_manifest = None
    
    manifest_manager.start_run()
    manifest_manager.mark_crashed("Test error")
    manifest_manager.current_manifest = None
    
    manifest_manager.start_run()
    manifest_manager.complete_run(RunStatus.STOPPED)
    manifest_manager.current_manifest = None
    
    # Filter by completed
    completed = manifest_manager.list_runs(status=RunStatus.COMPLETED)
    assert len([r for r in completed if r.status == RunStatus.COMPLETED.value]) >= 1
    
    # Filter by crashed
    crashed = manifest_manager.list_runs(status=RunStatus.CRASHED)
    assert len([r for r in crashed if r.status == RunStatus.CRASHED.value]) >= 1


def test_list_runs_filtered_by_mode(manifest_manager):
    """Test filtering runs by mode."""
    manifest_manager.start_run()
    manifest_manager.complete_run()
    
    # Filter by live mode
    live_runs = manifest_manager.list_runs(mode=RunMode.LIVE)
    assert all(r.mode == RunMode.LIVE.value for r in live_runs)


def test_get_crashed_runs(manifest_manager):
    """Test getting only crashed runs."""
    manifest_manager.start_run()
    manifest_manager.mark_crashed("Test crash")
    manifest_manager.current_manifest = None
    
    crashed = manifest_manager.get_crashed_runs()
    assert len(crashed) >= 1
    assert all(r.status == RunStatus.CRASHED.value for r in crashed)


def test_config_hash(manifest_manager):
    """Test config hash is computed."""
    manifest_manager.start_run()
    
    config_hash = manifest_manager.current_manifest.config_hash
    assert config_hash is not None
    assert len(config_hash) == 32  # MD5 hash


def test_git_info_captured(manifest_manager):
    """Test git info is captured if available."""
    manifest_manager.start_run()
    
    manifest = manifest_manager.current_manifest
    # Git info may or may not be available depending on environment
    # Just check fields exist
    assert hasattr(manifest, 'code_commit')
    assert hasattr(manifest, 'code_branch')
    assert hasattr(manifest, 'code_dirty')


def test_get_current_run_id(manifest_manager):
    """Test getting current run ID."""
    assert manifest_manager.get_current_run_id() is None
    
    run_id = manifest_manager.start_run()
    assert manifest_manager.get_current_run_id() == run_id


def test_get_run_directory(manifest_manager, temp_data_dir):
    """Test getting full path to run directory."""
    run_id = manifest_manager.start_run()
    
    run_dir = manifest_manager.get_run_directory(run_id)
    expected = os.path.join(temp_data_dir, '_meta', 'runs', run_id)
    
    assert run_dir == expected


def test_replay_mode_in_run_id(temp_data_dir):
    """Test replay mode appears in run ID."""
    manager = RunManifestManager(data_root=temp_data_dir, mode=RunMode.REPLAY)
    run_id = manager.start_run()
    
    assert 'replay' in run_id
    assert manager.current_manifest.mode == RunMode.REPLAY.value


def test_no_duplicate_file_tracking(manifest_manager):
    """Test that duplicate file paths aren't tracked twice."""
    manifest_manager.start_run()
    
    file_path = os.path.join(
        manifest_manager.data_root,
        'bronze/futures/trades/part-001.parquet'
    )
    
    # Track same file twice
    manifest_manager.track_bronze_file(file_path)
    manifest_manager.track_bronze_file(file_path)
    
    # Should only appear once
    assert len(manifest_manager.current_manifest.bronze_files) == 1

