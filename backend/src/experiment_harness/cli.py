"""Click CLI for the VP experiment harness.

Provides commands for running experiments, comparing results, importing
legacy data, and listing available signals and datasets.

Entry point::

    uv run python -m src.experiment_harness.cli run path/to/config.yaml
    uv run python -m src.experiment_harness.cli compare --signal ads
    uv run python -m src.experiment_harness.cli list-signals
    uv run python -m src.experiment_harness.cli list-datasets
    uv run python -m src.experiment_harness.cli import-legacy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

from .config_schema import ExperimentConfig
from .dataset_registry import DatasetRegistry
from .grid_generator import GridGenerator
from .online_simulator import OnlineSimulator
from .results_db import ResultsDB
from .runner import ExperimentRunner
from .signals import SIGNAL_REGISTRY, ensure_signals_loaded

logger = logging.getLogger(__name__)

# Default lake root: backend/lake relative to this file
# cli.py is at backend/src/experiment_harness/cli.py
# parents[0] = experiment_harness/, [1] = src/, [2] = backend/
_DEFAULT_LAKE_ROOT: Path = Path(__file__).resolve().parents[2] / "lake"

# Legacy experiment base path
_LEGACY_EXPERIMENTS_DIR: str = "research/vp_experiments"


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


@click.group()
def cli() -> None:
    """VP Experiment Harness -- signal evaluation and comparison."""
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
    """Run an experiment from a YAML config file.

    Loads CONFIG_PATH, expands the parameter grid, evaluates all
    signal/dataset/param combinations, and persists results to the
    ResultsDB under lake/research/vp_harness/results/.
    """
    resolved_lake: Path = _resolve_lake_root(lake_root)
    config: ExperimentConfig = ExperimentConfig.from_yaml(Path(config_path))

    click.echo(f"Experiment: {config.name}")
    click.echo(f"Datasets:   {config.datasets}")
    click.echo(f"Signals:    {config.signals}")
    click.echo(f"Lake root:  {resolved_lake}")
    click.echo()

    runner = ExperimentRunner(resolved_lake)
    run_ids: list[str] = runner.run(config)

    click.echo()
    click.echo(f"Completed {len(run_ids)} runs.")
    for rid in run_ids:
        click.echo(f"  {rid}")


@cli.command("generate-grid")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory. Default: backend/lake/",
)
def generate_grid(config_path: str, lake_root: str | None) -> None:
    """Generate grid variants from a config's grid_variant block."""
    resolved_lake: Path = _resolve_lake_root(lake_root)
    config: ExperimentConfig = ExperimentConfig.from_yaml(Path(config_path))
    if config.grid_variant is None:
        raise click.BadParameter(
            f"Config '{config_path}' does not include a grid_variant block."
        )

    specs = ExperimentRunner._expand_grid_variant_specs(config.grid_variant)
    generator = GridGenerator(resolved_lake)
    click.echo(f"Generating {len(specs)} variant(s) at {resolved_lake} ...")
    for idx, spec in enumerate(specs, start=1):
        variant_id = generator.generate(spec)
        click.echo(f"  [{idx}/{len(specs)}] {variant_id}")


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
    results_db = ResultsDB(resolved_lake / "research" / "vp_harness" / "results")

    best = results_db.query_best(
        signal=signal, dataset_id=dataset_id, min_signals=min_signals
    )

    if best.empty:
        click.echo("No results found matching filters.")
        return

    # Sort by chosen column
    if sort in best.columns:
        best = best.sort_values(sort, ascending=False).reset_index(drop=True)

    # Display as formatted table
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


@cli.command("import-legacy")
@click.option(
    "--lake-root",
    default=None,
    type=str,
    help="Override lake root directory.",
)
def import_legacy(lake_root: str | None) -> None:
    """Import legacy results.json files from Round 1+2 experiments.

    Scans vp_experiments/*/agents/*/outputs/results.json, converts each
    file to the ResultsDB append format, and persists them.
    """
    resolved_lake: Path = _resolve_lake_root(lake_root)
    results_db = ResultsDB(resolved_lake / "research" / "vp_harness" / "results")

    legacy_root: Path = resolved_lake / _LEGACY_EXPERIMENTS_DIR
    if not legacy_root.exists():
        click.echo(f"Legacy experiments directory not found: {legacy_root}")
        return

    results_files: list[Path] = sorted(legacy_root.glob("**/agents/*/outputs/results.json"))
    if not results_files:
        click.echo("No legacy results.json files found.")
        return

    click.echo(f"Found {len(results_files)} legacy result files.")
    imported_count: int = 0

    for results_path in results_files:
        try:
            raw: dict[str, Any] = json.loads(results_path.read_text())

            experiment_name: str = raw.get("experiment_name", "unknown")
            agent_name: str = raw.get("agent_name", "unknown")
            dataset_id: str = raw.get("dataset_id", "unknown")
            params: dict[str, Any] = raw.get("params", {})

            threshold_results: list[dict[str, Any]] = raw.get(
                "results_by_threshold", []
            )
            if not threshold_results:
                logger.warning(
                    "No threshold results in %s, skipping", results_path
                )
                continue

            meta: dict[str, Any] = {
                "experiment_name": f"legacy_{experiment_name}",
                "dataset_id": dataset_id,
                "signal_name": agent_name,
                "signal_params_json": json.dumps(params, sort_keys=True, default=str),
                "grid_variant_id": "immutable",
                "eval_tp_ticks": 8,
                "eval_sl_ticks": 4,
                "eval_max_hold_bins": 1200,
                "elapsed_seconds": 0.0,
                "config_hash": "legacy_import",
            }

            run_id: str = results_db.append_run(meta, threshold_results)
            imported_count += 1
            click.echo(f"  Imported {agent_name} -> run_id={run_id}")

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse %s: %s", results_path, exc)
            continue

    click.echo(f"Imported {imported_count} legacy runs.")


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
        # Determine type from base classes
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


if __name__ == "__main__":
    cli()
