"""Click CLI for the qMachina experiment harness.

Provides commands for running experiments, comparing results, generating
immutable datasets, computing gold features, promoting experiment winners,
and listing available signals and datasets.

Entry point::

    uv run python -m src.experiment_harness.cli run path/to/experiment_spec.yaml
    uv run python -m src.experiment_harness.cli compare --signal ads
    uv run python -m src.experiment_harness.cli list-signals
    uv run python -m src.experiment_harness.cli list-datasets
    uv run python -m src.experiment_harness.cli generate path/to/pipeline_spec.yaml
    uv run python -m src.experiment_harness.cli generate-gold path/to/pipeline_spec.yaml
    uv run python -m src.experiment_harness.cli promote path/to/experiment_spec.yaml --run-id abc12345 --alias vp_main
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import click

from ..qmachina.stage_schema import SILVER_COLS
from .dataset_registry import DatasetRegistry
from .results_db import ResultsDB
from .runner import ExperimentRunner
from .signals import SIGNAL_REGISTRY, ensure_signals_loaded

logger = logging.getLogger(__name__)

# Default lake root: backend/lake relative to this file
# cli.py is at backend/src/experiment_harness/cli.py
# parents[0] = experiment_harness/, [1] = src/, [2] = backend/
_DEFAULT_LAKE_ROOT: Path = Path(__file__).resolve().parents[2] / "lake"


def _resolve_lake_root(lake_root_arg: str | None) -> Path:
    """Resolve lake root from CLI argument or default.

    Args:
        lake_root_arg: User-provided lake root path, or None for default.

    Returns:
        Resolved absolute Path to the lake root directory.

    Raises:
        click.BadParameter: If the resolved path does not exist.
    """
    if lake_root_arg is not None:
        resolved: Path = Path(lake_root_arg).resolve()
    else:
        resolved = _DEFAULT_LAKE_ROOT

    if not resolved.exists():
        raise click.BadParameter(
            f"Lake root does not exist: {resolved}",
            param_hint="--lake-root",
        )
    return resolved


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _sha256_file(p: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_dataset(lake_root: Path, pipeline_spec: Any) -> str:
    """Generate an immutable dataset from a resolved PipelineSpec.

    Streams events through the VP engine, writes bins.parquet,
    grid_clean.parquet, manifest.json, and checksums.json.
    Idempotent: returns immediately if the dataset already exists.

    Args:
        lake_root: Root path of the data lake.
        pipeline_spec: Validated PipelineSpec instance.

    Returns:
        The deterministic dataset_id string.
    """
    import pandas as pd

    from ..models.vacuum_pressure.stream_pipeline import stream_events

    dataset_id: str = pipeline_spec.dataset_id()
    output_dir: Path = lake_root / "research" / "datasets" / dataset_id
    bins_path: Path = output_dir / "bins.parquet"

    if bins_path.exists():
        click.echo(f"Dataset exists: {dataset_id}")
        return dataset_id

    config = pipeline_spec.resolve_runtime_config()

    end_time_et = pd.Timestamp(
        f"{pipeline_spec.capture.dt} {pipeline_spec.capture.end_time}:00",
        tz="America/New_York",
    ).tz_convert("UTC")
    end_time_ns: int = int(end_time_et.value)

    click.echo("Streaming events...")
    t_start: float = time.monotonic()

    all_bin_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    n_bins: int = 0

    for bin_grid in stream_events(
        lake_root=lake_root,
        config=config,
        dt=pipeline_spec.capture.dt,
        start_time=pipeline_spec.capture.start_time,
    ):
        ts_ns: int = int(bin_grid["ts_ns"])
        if ts_ns > end_time_ns:
            break

        bin_seq: int = int(bin_grid["bin_seq"])
        n_bins += 1

        all_bin_rows.append(
            {
                "bin_seq": bin_seq,
                "ts_ns": ts_ns,
                "bin_start_ns": int(bin_grid["bin_start_ns"]),
                "bin_end_ns": int(bin_grid["bin_end_ns"]),
                "mid_price": float(bin_grid["mid_price"]),
                "event_id": int(bin_grid["event_id"]),
                "bin_event_count": int(bin_grid["bin_event_count"]),
                "book_valid": bool(bin_grid["book_valid"]),
                "best_bid_price_int": int(bin_grid["best_bid_price_int"]),
                "best_ask_price_int": int(bin_grid["best_ask_price_int"]),
                "spot_ref_price_int": int(bin_grid["spot_ref_price_int"]),
            }
        )

        cols = bin_grid["grid_cols"]
        n_rows = len(cols["k"])
        for row_idx in range(n_rows):
            row: dict[str, Any] = {"bin_seq": bin_seq}
            for col_name, col_values in cols.items():
                if col_name not in SILVER_COLS:
                    continue
                value = col_values[row_idx]
                if hasattr(value, "item"):
                    value = value.item()
                row[col_name] = value
            all_bucket_rows.append(row)

    elapsed: float = time.monotonic() - t_start
    click.echo(f"Collected {n_bins} bins ({len(all_bucket_rows)} rows) in {elapsed:.2f}s")

    if not all_bin_rows:
        raise click.ClickException(
            "No bins emitted. Check date/time window and data availability."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bins_df: pd.DataFrame = pd.DataFrame(all_bin_rows)
    bins_df.to_parquet(bins_path, index=False)
    click.echo(f"  bins.parquet: {len(bins_df)} rows")

    if not all_bucket_rows:
        raise click.ClickException(
            "No grid rows emitted. Check date/time window and data availability."
        )

    df: pd.DataFrame = pd.DataFrame(all_bucket_rows)
    numeric_cols: list[str] = [
        c for c in df.columns
        if c not in ("k", "bin_seq") and df[c].dtype.kind in ("f", "i", "u", "b")
    ]
    if numeric_cols:
        mask = df[numeric_cols].abs().sum(axis=1) > 0
    else:
        mask = pd.Series(True, index=df.index)
    df_clean: pd.DataFrame = df[mask].reset_index(drop=True)
    grid_clean_path: Path = output_dir / "grid_clean.parquet"
    df_clean.to_parquet(grid_clean_path, index=False)
    click.echo(f"  grid_clean.parquet: {len(df_clean)} rows")

    manifest: dict[str, Any] = {
        "pipeline_spec_name": pipeline_spec.name,
        "dataset_id": dataset_id,
        "product_type": pipeline_spec.capture.product_type,
        "symbol": pipeline_spec.capture.symbol,
        "dt": pipeline_spec.capture.dt,
        "start_time": pipeline_spec.capture.start_time,
        "end_time": pipeline_spec.capture.end_time,
        "cell_width_ms": config.cell_width_ms,
        "n_bins": n_bins,
        "code_version": pipeline_spec.pipeline_code_version,
        "source_manifest": {
            "product_type": pipeline_spec.capture.product_type,
            "symbol": pipeline_spec.capture.symbol,
            "dt": pipeline_spec.capture.dt,
            "capture_start_et": pipeline_spec.capture.start_time,
            "capture_end_et": pipeline_spec.capture.end_time,
            "stream_start_time_hhmm": pipeline_spec.capture.start_time.replace(":", ""),
        },
    }
    manifest_path: Path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    click.echo("  manifest.json written")

    checksums: dict[str, str] = {
        "bins.parquet": _sha256_file(bins_path),
        "grid_clean.parquet": _sha256_file(grid_clean_path),
    }
    checksums_path: Path = output_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2))
    click.echo("  checksums.json written")

    click.echo(f"Dataset generated: {dataset_id}")
    return dataset_id


@click.group()
def cli() -> None:
    """qMachina Experiment Harness -- signal evaluation and comparison."""
    _setup_logging()


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory. Default: backend/lake/",
)
def run(config_path: str, lake_root: str | None) -> None:
    """Run an experiment from an ExperimentSpec YAML.

    Resolves serving -> pipeline, auto-generates the dataset if missing,
    expands sweep axes, evaluates, and persists results.
    """
    from ..qmachina.experiment_config import ExperimentSpec

    resolved_lake: Path = _resolve_lake_root(lake_root)
    spec: ExperimentSpec = ExperimentSpec.from_yaml(Path(config_path))

    click.echo(f"Experiment: {spec.name}")
    click.echo(f"Serving:    {spec.serving}")
    click.echo(f"Lake root:  {resolved_lake}")
    click.echo()

    serving_spec = spec.resolve_serving(resolved_lake)
    pipeline_spec = serving_spec.resolve_pipeline(resolved_lake)
    _generate_dataset(resolved_lake, pipeline_spec)

    harness_dict: dict[str, Any] = spec.to_runner_config(resolved_lake)

    # ExperimentConfig is the internal runner schema — not user-facing.
    from .config_schema import ExperimentConfig

    config: ExperimentConfig = ExperimentConfig.model_validate(harness_dict)

    click.echo(f"Datasets:   {config.datasets}")
    click.echo(f"Signals:    {config.signals}")
    click.echo()

    runner = ExperimentRunner(
        lake_root=resolved_lake,
        feature_store_config=spec.feature_store if spec.feature_store.enabled else None,
    )
    run_ids: list[str] = runner.run(config)

    click.echo()
    click.echo(f"Completed {len(run_ids)} runs.")
    for rid in run_ids:
        click.echo(f"  {rid}")


@cli.command()
@click.option("--signal", default=None, help="Filter by signal name.")
@click.option("--dataset-id", default=None, help="Filter by dataset ID.")
@click.option(
    "--sort",
    default="tp_rate",
    type=click.Choice(["tp_rate", "mean_pnl_ticks", "events_per_hour"]),
    help="Sort column for ranking.",
)
@click.option(
    "--min-signals",
    default=5,
    type=int,
    help="Minimum signals for a result to be shown.",
)
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory.",
)
def compare(
    signal: str | None,
    dataset_id: str | None,
    sort: str,
    min_signals: int,
    lake_root: str | None,
) -> None:
    """Compare experiment results across signals and datasets.

    Queries the ResultsDB for best-threshold results per run, optionally
    filtered by signal and dataset, sorted by the chosen metric.
    """
    resolved_lake: Path = _resolve_lake_root(lake_root)
    results_db = ResultsDB(resolved_lake / "research" / "harness" / "results")

    best = results_db.query_best(
        signal=signal, dataset_id=dataset_id, min_signals=min_signals
    )

    if best.empty:
        click.echo("No results found matching filters.")
        return

    if sort in best.columns:
        best = best.sort_values(sort, ascending=False).reset_index(drop=True)

    click.echo(
        f"{'Rank':<5} {'Signal':<22} {'Dataset':<32} "
        f"{'TP%':<8} {'N':<7} {'PnL':<9} {'Thr':<8} {'Evt/hr':<10}"
    )
    click.echo("-" * 102)

    for rank, (_, row) in enumerate(best.iterrows(), start=1):
        sig_name: str = str(row.get("signal_name", "?"))
        ds_id: str = str(row.get("dataset_id", "?"))
        tp_rate: float = float(row.get("tp_rate", 0))
        n_sig: int = int(row.get("n_signals", 0))
        mean_pnl: float = float(row.get("mean_pnl_ticks", 0))
        threshold: float = float(row.get("threshold", 0))
        evt_hr: float = float(row.get("events_per_hour", 0))

        click.echo(
            f"{rank:<5} {sig_name:<22} {ds_id:<32} "
            f"{tp_rate * 100:>5.1f}%  {n_sig:<7} {mean_pnl:>+7.2f}  "
            f"{threshold:<8.3f} {evt_hr:>8.1f}"
        )


@cli.command("online-sim")
@click.option("--signal", required=True, type=str, help="Signal name to simulate.")
@click.option("--dataset-id", required=True, type=str, help="Dataset ID to simulate.")
@click.option(
    "--signal-params-json",
    default="{}",
    type=str,
    help="JSON object of signal constructor params.",
)
@click.option(
    "--bin-budget-ms",
    default=100.0,
    type=float,
    help="Per-bin budget in milliseconds.",
)
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory.",
)
def online_sim(
    signal: str,
    dataset_id: str,
    signal_params_json: str,
    bin_budget_ms: float,
    lake_root: str | None,
) -> None:
    """Run online simulation for one signal/dataset combo."""
    from .online_simulator import OnlineSimulator

    resolved_lake: Path = _resolve_lake_root(lake_root)
    try:
        params: dict[str, Any] = json.loads(signal_params_json)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(
            f"Invalid --signal-params-json: {exc}"
        ) from exc

    sim = OnlineSimulator(resolved_lake)
    result = sim.simulate(
        dataset_id=dataset_id,
        signal_name=signal,
        signal_params=params,
        bin_budget_ms=bin_budget_ms,
    )

    click.echo(f"signal={result.signal_name} dataset={dataset_id}")
    click.echo(
        f"p99_total_us={result.total_latency_us.get('p99', 0.0):.2f} "
        f"budget_pct={result.p99_budget_pct:.2f}% "
        f"retrain_count={result.retrain_count}"
    )


@cli.command("list-signals")
def list_signals() -> None:
    """List all registered signals in the harness."""
    ensure_signals_loaded()

    if not SIGNAL_REGISTRY:
        click.echo("No signals registered.")
        return

    click.echo(f"{'Name':<22} {'Type':<15} {'Class':<40}")
    click.echo("-" * 77)

    for name in sorted(SIGNAL_REGISTRY.keys()):
        cls = SIGNAL_REGISTRY[name]
        from .signals.base import MLSignal, StatisticalSignal

        if issubclass(cls, MLSignal):
            sig_type = "ML"
        elif issubclass(cls, StatisticalSignal):
            sig_type = "Statistical"
        else:
            sig_type = "Unknown"

        click.echo(f"{name:<22} {sig_type:<15} {cls.__name__:<40}")


@cli.command("list-datasets")
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory.",
)
def list_datasets(lake_root: str | None) -> None:
    """List all available datasets in the data lake."""
    resolved_lake: Path = _resolve_lake_root(lake_root)
    registry = DatasetRegistry(resolved_lake)
    datasets: list[str] = registry.list_datasets()

    if not datasets:
        click.echo("No datasets found.")
        return

    click.echo(f"Found {len(datasets)} datasets:")
    for ds_id in datasets:
        click.echo(f"  {ds_id}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory. Default: backend/lake/",
)
def generate(config_path: str, lake_root: str | None) -> None:
    """Generate an immutable VP dataset from a PipelineSpec YAML.

    Loads CONFIG_PATH as a PipelineSpec, resolves runtime config, streams
    events through the VP engine, and writes to datasets/{dataset_id}/.
    """
    from ..qmachina.pipeline_config import PipelineSpec

    resolved_lake: Path = _resolve_lake_root(lake_root)
    spec: PipelineSpec = PipelineSpec.from_yaml(Path(config_path))

    click.echo(f"Pipeline:   {spec.name}")
    click.echo(f"Dataset ID: {spec.dataset_id()}")
    click.echo(f"Symbol:     {spec.capture.symbol}")
    click.echo(f"Date:       {spec.capture.dt}")
    click.echo(f"Window:     {spec.capture.start_time} - {spec.capture.end_time}")
    click.echo()

    dataset_id = _generate_dataset(resolved_lake, spec)
    if spec.feature_store.enabled:
        from .feature_store.writer import sync_dataset_to_feature_store

        registry = DatasetRegistry(resolved_lake)
        paths = registry.resolve(dataset_id)
        sync_dataset_to_feature_store(dataset_id, paths, resolved_lake, spec.feature_store)
        click.echo(f"Feature store synced: {dataset_id}")


@cli.command("generate-gold")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory. Default: backend/lake/",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute gold features even if gold_grid.parquet already exists.",
)
def generate_gold(config_path: str, lake_root: str | None, force: bool) -> None:
    """Compute gold features for an existing dataset.

    Loads CONFIG_PATH as a PipelineSpec, resolves the dataset, and runs
    the gold feature builder to produce gold_grid.parquet alongside the
    existing silver grid_clean.parquet. Idempotent unless --force is given.
    """
    from ..qmachina.gold_config import GoldFeatureConfig
    from ..qmachina.pipeline_config import PipelineSpec
    from .gold_builder import generate_gold_dataset

    resolved_lake: Path = _resolve_lake_root(lake_root)
    spec: PipelineSpec = PipelineSpec.from_yaml(Path(config_path))
    dataset_id = spec.dataset_id()

    click.echo(f"Pipeline:   {spec.name}")
    click.echo(f"Dataset ID: {dataset_id}")
    click.echo()

    registry = DatasetRegistry(resolved_lake)
    paths = registry.resolve(dataset_id)

    config = spec.resolve_runtime_config()
    gold_cfg = GoldFeatureConfig.from_runtime_config(config)
    click.echo(f"Gold config hash: {gold_cfg.config_hash()}")

    gold_path = generate_gold_dataset(paths, gold_cfg, force=force)
    click.echo(f"Gold features written: {gold_path}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--run-id",
    required=True,
    type=str,
    help="Run ID from experiment results to promote.",
)
@click.option(
    "--alias",
    required=True,
    type=str,
    help="Serving alias used at runtime (URL: ?serving=<alias>).",
)
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory. Default: backend/lake/",
)
def promote(
    config_path: str,
    run_id: str,
    alias: str,
    lake_root: str | None,
) -> None:
    """Promote a winning experiment run to an immutable serving version.

    Loads CONFIG_PATH as an ExperimentSpec, extracts the winning parameters
    from RUN_ID, resolves a full runtime snapshot, writes immutable serving
    version YAML, and updates the alias registry mapping.
    """
    from datetime import datetime, timezone

    from ..qmachina.config import build_config_with_overrides
    from ..qmachina.experiment_config import ExperimentSpec
    from ..qmachina.serving_config import (
        PublishedServingSource,
        PublishedServingSpec,
        ServingSpec,
    )
    from ..qmachina.serving_registry import ServingRegistry

    resolved_lake: Path = _resolve_lake_root(lake_root)
    spec: ExperimentSpec = ExperimentSpec.from_yaml(Path(config_path))

    results_db_root: Path = resolved_lake / "research" / "harness" / "results"
    results_db = ResultsDB(results_db_root)

    click.echo(f"Experiment: {spec.name}")
    click.echo(f"Run ID:     {run_id}")
    alias_clean = alias.strip().lower()
    click.echo(f"Alias:      {alias_clean}")
    click.echo()

    promoted: ServingSpec = spec.extract_winning_serving(
        run_id=run_id,
        results_db_root=results_db_root,
        lake_root=resolved_lake,
    )
    pipeline_spec = promoted.resolve_pipeline(resolved_lake)
    runtime_base = pipeline_spec.resolve_runtime_config()
    serving_runtime_fields: dict[str, Any] = promoted.to_runtime_fields(
        cell_width_ms=runtime_base.cell_width_ms
    )
    effective_runtime = build_config_with_overrides(runtime_base, serving_runtime_fields)

    meta_df = results_db.query_runs(run_id=run_id)
    if meta_df.empty:
        raise click.ClickException(
            f"Run '{run_id}' was not found in ResultsDB metadata at {results_db_root}"
        )
    meta_row = meta_df.iloc[0].to_dict()
    config_hash = str(meta_row.get("config_hash", "")).strip()
    if not config_hash:
        raise click.ClickException(
            f"Run '{run_id}' is missing config_hash in ResultsDB metadata."
        )

    promoted_at_utc = datetime.now(tz=timezone.utc).isoformat()
    registry = ServingRegistry(resolved_lake)
    serving_id = registry.build_serving_id(
        experiment_name=spec.name,
        run_id=run_id,
        config_hash=config_hash,
    )

    runtime_snapshot = effective_runtime.to_dict()
    runtime_snapshot["stream_dt"] = pipeline_spec.capture.dt
    runtime_snapshot["stream_start_time"] = pipeline_spec.capture.start_time

    published = PublishedServingSpec(
        serving_id=serving_id,
        description=promoted.description,
        runtime_snapshot=runtime_snapshot,
        source=PublishedServingSource(
            run_id=run_id,
            experiment_name=spec.name,
            config_hash=config_hash,
            promoted_at_utc=promoted_at_utc,
            serving_spec_name=promoted.name,
            signal_name=promoted.signal.name if promoted.signal is not None else None,
        ),
    )
    result = registry.promote(alias=alias_clean, spec=published, actor="cli.promote")

    click.echo(f"Serving ID: {result.serving_id}")
    click.echo(f"Alias:      {result.alias}")
    click.echo(f"Spec:       {result.spec_path}")
    click.echo(f"Reused:     {result.reused_existing}")
    click.echo()

    click.echo("Runtime snapshot identity:")
    click.echo(f"  config_version: {effective_runtime.config_version}")
    click.echo(f"  config_hash:    {config_hash}")
    click.echo(
        "WebSocket URL: "
        f"ws://localhost:8002/v1/stream?serving={result.alias}"
    )
    click.echo(
        "Frontend URL:  "
        f"http://localhost:5174/vp-stream.html?serving={result.alias}"
    )


if __name__ == "__main__":
    cli()
