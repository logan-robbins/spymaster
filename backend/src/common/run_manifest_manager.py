"""
Run Manifest Manager for Phase 1 institutional hygiene.

Creates metadata for each ingestion run per PLAN.md §1.2 / §2.5:
- Unique run ID with timestamp
- Config snapshot (exact parameters used)
- Schema versions (from registry)
- Bronze file tracking
- Run status (started/completed/crashed)

Enables:
- Reproducibility: "re-run with exact same config/code"
- Debugging: "what changed between these two runs?"
- ML dataset provenance: "which run produced this training data?"
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from src.common.config import CONFIG


class RunStatus(Enum):
    """Run status states."""
    STARTED = "started"
    COMPLETED = "completed"
    CRASHED = "crashed"
    STOPPED = "stopped"


class RunMode(Enum):
    """Ingestion mode."""
    LIVE = "live"
    REPLAY = "replay"
    SIM = "sim"


@dataclass
class RunManifest:
    """
    Manifest for an ingestion run.
    
    Tracks metadata, config, schemas, and output files for reproducibility.
    """
    run_id: str
    start_time: str  # ISO 8601 UTC
    end_time: Optional[str] = None  # ISO 8601 UTC
    status: str = RunStatus.STARTED.value
    mode: str = RunMode.LIVE.value
    
    # Code version (git commit if available)
    code_commit: Optional[str] = None
    code_branch: Optional[str] = None
    code_dirty: bool = False
    
    # Config hash for quick comparison
    config_hash: str = ""
    
    # Output tracking
    bronze_files: List[str] = None
    gold_files: List[str] = None
    event_counts: Dict[str, int] = None
    
    # Error tracking
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.bronze_files is None:
            self.bronze_files = []
        if self.gold_files is None:
            self.gold_files = []
        if self.event_counts is None:
            self.event_counts = {}


class RunManifestManager:
    """
    Manages run manifests for ingestion sessions.
    
    Features:
    - Generate unique run IDs
    - Capture config + schema snapshots
    - Track output files
    - Mark run completion/failure
    - Load historical manifests
    """
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        mode: RunMode = RunMode.LIVE
    ):
        """
        Initialize manifest manager.
        
        Args:
            data_root: Root directory for data lake
            mode: Run mode (live, replay, sim)
        """
        self.data_root = data_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'lake'
        )
        self.mode = mode
        
        # Manifest storage
        self.meta_root = os.path.join(self.data_root, '_meta', 'runs')
        os.makedirs(self.meta_root, exist_ok=True)
        
        # Current run
        self.current_manifest: Optional[RunManifest] = None
        self.current_run_dir: Optional[str] = None
        
        print(f"📋 Run manifest manager initialized: {self.meta_root}")
    
    def _generate_run_id(self) -> str:
        """
        Generate unique run ID.
        
        Format: {timestamp}_{mode}_{short_hash}
        Example: 2025-12-22_143015_123456_live_abc123
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime('%Y-%m-%d_%H%M%S')
        microseconds = now.strftime('%f')  # Add microseconds for uniqueness
        short_hash = hashlib.md5(
            f"{timestamp}{microseconds}{os.getpid()}".encode()
        ).hexdigest()[:6]
        
        return f"{timestamp}_{microseconds}_{self.mode.value}_{short_hash}"
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git commit/branch info if available."""
        git_info = {
            'commit': None,
            'branch': None,
            'dirty': False
        }
        
        try:
            # Get current commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                git_info['commit'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check if repo is dirty
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                git_info['dirty'] = bool(result.stdout.strip())
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Git not available or timeout
            pass
        
        return git_info
    
    def _compute_config_hash(self) -> str:
        """Compute hash of current config for quick comparison."""
        config_dict = asdict(CONFIG)
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_json.encode()).hexdigest()
    
    def start_run(self) -> str:
        """
        Start a new run and create manifest.
        
        Returns:
            run_id
        """
        # Generate run ID and create directory
        run_id = self._generate_run_id()
        run_dir = os.path.join(self.meta_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Get git info
        git_info = self._get_git_info()
        
        # Create manifest
        self.current_manifest = RunManifest(
            run_id=run_id,
            start_time=datetime.now(timezone.utc).isoformat(),
            status=RunStatus.STARTED.value,
            mode=self.mode.value,
            code_commit=git_info['commit'],
            code_branch=git_info['branch'],
            code_dirty=git_info['dirty'],
            config_hash=self._compute_config_hash()
        )
        
        self.current_run_dir = run_dir
        
        # Write initial manifest
        self._write_manifest()
        
        # Write config snapshot
        self._write_config_snapshot()
        
        # Write schema snapshot
        self._write_schema_snapshot()
        
        print(f"  Run manifest created: {run_id}")
        return run_id
    
    def _write_manifest(self) -> None:
        """Write current manifest to disk."""
        if not self.current_manifest or not self.current_run_dir:
            return
        
        manifest_path = os.path.join(self.current_run_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(asdict(self.current_manifest), f, indent=2)
    
    def _write_config_snapshot(self) -> None:
        """Write config snapshot to disk."""
        if not self.current_run_dir:
            return
        
        config_path = os.path.join(self.current_run_dir, 'config_snapshot.json')
        config_dict = asdict(CONFIG)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _write_schema_snapshot(self) -> None:
        """Write schema versions to disk."""
        if not self.current_run_dir:
            return
        
        schema_dir = os.path.join(self.current_run_dir, 'schemas')
        os.makedirs(schema_dir, exist_ok=True)
        
        # Try to import schema registry
        try:
            from .schemas.base import SCHEMA_REGISTRY
            
            schemas_info = {}
            for schema_name in SCHEMA_REGISTRY.list_schemas():
                schema_obj = SCHEMA_REGISTRY.get_schema(schema_name)
                if schema_obj:
                    schemas_info[schema_name] = {
                        'name': schema_obj.name,
                        'version': schema_obj.version,
                        'full_name': schema_obj.full_name(),
                        'description': schema_obj.description or "",
                    }
            
            schema_path = os.path.join(schema_dir, 'versions.json')
            with open(schema_path, 'w') as f:
                json.dump(schemas_info, f, indent=2)
        
        except ImportError:
            # Schema registry not available (e.g., minimal setup)
            pass
    
    def track_bronze_file(self, file_path: str) -> None:
        """Track a Bronze file written during this run."""
        if not self.current_manifest:
            return
        
        # Make path relative to data_root for portability
        rel_path = os.path.relpath(file_path, self.data_root)
        
        if rel_path not in self.current_manifest.bronze_files:
            self.current_manifest.bronze_files.append(rel_path)
    
    def track_gold_file(self, file_path: str) -> None:
        """Track a Gold file written during this run."""
        if not self.current_manifest:
            return
        
        # Make path relative to data_root for portability
        rel_path = os.path.relpath(file_path, self.data_root)
        
        if rel_path not in self.current_manifest.gold_files:
            self.current_manifest.gold_files.append(rel_path)
    
    def update_event_count(self, schema_name: str, count: int) -> None:
        """Update event count for a schema."""
        if not self.current_manifest:
            return
        
        if schema_name in self.current_manifest.event_counts:
            self.current_manifest.event_counts[schema_name] += count
        else:
            self.current_manifest.event_counts[schema_name] = count
    
    def complete_run(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """
        Mark run as completed.
        
        Args:
            status: Final status (COMPLETED or STOPPED)
        """
        if not self.current_manifest:
            return
        
        self.current_manifest.end_time = datetime.now(timezone.utc).isoformat()
        self.current_manifest.status = status.value
        
        # Write final manifest
        self._write_manifest()
        
        print(f"  Run manifest finalized: {self.current_manifest.run_id} ({status.value})")
    
    def mark_crashed(self, error_message: str) -> None:
        """Mark run as crashed with error message."""
        if not self.current_manifest:
            return
        
        self.current_manifest.end_time = datetime.now(timezone.utc).isoformat()
        self.current_manifest.status = RunStatus.CRASHED.value
        self.current_manifest.error_message = error_message
        
        # Write final manifest
        self._write_manifest()
        
        print(f"  Run manifest marked as crashed: {self.current_manifest.run_id}")
    
    def get_current_run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self.current_manifest.run_id if self.current_manifest else None
    
    def load_manifest(self, run_id: str) -> Optional[RunManifest]:
        """
        Load a manifest by run ID.
        
        Args:
            run_id: Run ID to load
        
        Returns:
            RunManifest or None if not found
        """
        manifest_path = os.path.join(self.meta_root, run_id, 'manifest.json')
        
        if not os.path.exists(manifest_path):
            return None
        
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            
            return RunManifest(**data)
        
        except Exception as e:
            print(f"  Error loading manifest {run_id}: {e}")
            return None
    
    def list_runs(
        self,
        mode: Optional[RunMode] = None,
        status: Optional[RunStatus] = None
    ) -> List[RunManifest]:
        """
        List all runs, optionally filtered.
        
        Args:
            mode: Filter by mode (live/replay/sim)
            status: Filter by status
        
        Returns:
            List of RunManifest objects, sorted by start_time descending
        """
        manifests = []
        
        if not os.path.exists(self.meta_root):
            return manifests
        
        for run_dir in os.listdir(self.meta_root):
            manifest = self.load_manifest(run_dir)
            if manifest:
                # Apply filters
                if mode and manifest.mode != mode.value:
                    continue
                if status and manifest.status != status.value:
                    continue
                
                manifests.append(manifest)
        
        # Sort by start_time descending (most recent first)
        manifests.sort(key=lambda m: m.start_time, reverse=True)
        
        return manifests
    
    def get_crashed_runs(self) -> List[RunManifest]:
        """Get all runs that crashed (for recovery)."""
        return self.list_runs(status=RunStatus.CRASHED)
    
    def get_run_directory(self, run_id: str) -> str:
        """Get full path to run directory."""
        return os.path.join(self.meta_root, run_id)

