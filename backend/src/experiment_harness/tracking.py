"""Tracking adapter for experiment runs.

MLflow is the canonical backend. Optionally mirrors summary metrics to W&B.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config_schema import TrackingConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Canonical experiment tracker.

    Logs each run to MLflow and optionally mirrors summary metrics/artifacts
    to Weights & Biases when enabled.
    """

    def __init__(self, config: TrackingConfig, default_experiment_name: str) -> None:
        self._cfg = config
        self._enabled: bool = config.backend == "mlflow"
        self._mlflow: Any | None = None

        if not self._enabled:
            return

        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise ImportError(
                "MLflow backend requested but mlflow is not installed."
            ) from exc

        self._mlflow = mlflow
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        exp_name = config.experiment_name or f"qmachina/{default_experiment_name}"
        mlflow.set_experiment(exp_name)
        logger.info("Tracking enabled: mlflow experiment='%s'", exp_name)

    def log_run(
        self,
        *,
        run_id_local: str,
        config_name: str,
        config_hash: str,
        meta: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        """Log a single harness run to MLflow (+ optional W&B mirror)."""
        if not self._enabled or self._mlflow is None:
            return

        mlflow = self._mlflow
        df = pd.DataFrame(results)

        if df.empty:
            best_row: dict[str, Any] = {}
        else:
            if "tp_rate" in df.columns:
                idx = df["tp_rate"].fillna(float("-inf")).idxmax()
                best_row = df.loc[int(idx)].to_dict()
            else:
                best_row = df.iloc[0].to_dict()

        signal_name = str(meta.get("signal_name", "unknown"))
        dataset_id = str(meta.get("dataset_id", "unknown"))
        run_prefix = self._cfg.run_name_prefix or config_name
        run_name = f"{run_prefix}:{signal_name}:{dataset_id}:{run_id_local[:8]}"

        tags = {
            "campaign": config_name,
            "config_hash": config_hash,
            "local_run_id": run_id_local,
            "signal_name": signal_name,
            "dataset_id": dataset_id,
            "grid_variant_id": str(meta.get("grid_variant_id", "immutable")),
        }
        tags.update(self._cfg.tags)

        params = {
            "campaign": config_name,
            "dataset_id": dataset_id,
            "signal_name": signal_name,
            "signal_params_json": str(meta.get("signal_params_json", "{}")),
            "eval_tp_ticks": str(meta.get("eval_tp_ticks", "")),
            "eval_sl_ticks": str(meta.get("eval_sl_ticks", "")),
            "eval_max_hold_bins": str(meta.get("eval_max_hold_bins", "")),
            "grid_variant_id": str(meta.get("grid_variant_id", "immutable")),
        }

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(tags)
                mlflow.log_params(params)

                mlflow.log_metric("n_threshold_rows", float(len(df)))
                if "elapsed_seconds" in meta:
                    mlflow.log_metric(
                        "elapsed_seconds", float(meta["elapsed_seconds"])
                    )

                for metric in (
                    "tp_rate",
                    "sl_rate",
                    "timeout_rate",
                    "mean_pnl_ticks",
                    "events_per_hour",
                    "n_signals",
                    "threshold",
                    "cooldown_bins",
                    "median_time_to_outcome_ms",
                ):
                    if metric in best_row and pd.notna(best_row[metric]):
                        mlflow.log_metric(metric, float(best_row[metric]))

                state_dist = self._parse_json_dict(
                    meta.get("state5_distribution_json")
                )
                micro_dist = self._parse_json_dict(
                    meta.get("micro9_distribution_json")
                )
                transition = self._parse_json_value(
                    meta.get("state5_transition_matrix_json")
                )
                labels = self._parse_json_value(meta.get("state5_labels_json"))

                for code, count in state_dist.items():
                    try:
                        mlflow.log_metric(f"state5_count_{code}", float(count))
                    except (TypeError, ValueError):
                        continue
                for code, count in micro_dist.items():
                    try:
                        mlflow.log_metric(f"micro9_count_{code}", float(count))
                    except (TypeError, ValueError):
                        continue

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    (tmp / "meta.json").write_text(
                        json.dumps(meta, indent=2, default=str) + "\n",
                        encoding="utf-8",
                    )
                    (tmp / "results_by_threshold.json").write_text(
                        json.dumps(results, indent=2, default=str) + "\n",
                        encoding="utf-8",
                    )
                    if not df.empty:
                        df.to_parquet(
                            tmp / "results_by_threshold.parquet",
                            index=False,
                            engine="pyarrow",
                        )
                    if state_dist or micro_dist or transition is not None:
                        diag = {
                            "taxonomy_version": meta.get("taxonomy_version"),
                            "state5_distribution": state_dist,
                            "micro9_distribution": micro_dist,
                            "state5_transition_matrix": transition,
                            "state5_labels": labels,
                        }
                        (tmp / "permutation_diagnostics.json").write_text(
                            json.dumps(diag, indent=2, default=str) + "\n",
                            encoding="utf-8",
                        )
                    mlflow.log_artifacts(str(tmp), artifact_path="harness")

                self._mirror_wandb(
                    run_name=run_name,
                    config_name=config_name,
                    meta=meta,
                    best_row=best_row,
                    df=df,
                )
        except Exception:
            logger.exception(
                "MLflow logging failed for local run_id=%s signal=%s dataset=%s",
                run_id_local,
                signal_name,
                dataset_id,
            )

    @staticmethod
    def _parse_json_value(raw: Any) -> Any | None:
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw
        if not isinstance(raw, str):
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @classmethod
    def _parse_json_dict(cls, raw: Any) -> dict[str, Any]:
        parsed = cls._parse_json_value(raw)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _mirror_wandb(
        self,
        *,
        run_name: str,
        config_name: str,
        meta: dict[str, Any],
        best_row: dict[str, Any],
        df: pd.DataFrame,
    ) -> None:
        """Optional W&B mirror (best-effort)."""
        if not self._cfg.wandb_mirror:
            return
        if not self._cfg.wandb_project:
            raise ValueError(
                "tracking.wandb_mirror=true requires tracking.wandb_project."
            )

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B mirroring requested but wandb is not installed."
            ) from exc

        run = wandb.init(
            project=self._cfg.wandb_project,
            entity=self._cfg.wandb_entity,
            name=run_name,
            config={
                "campaign": config_name,
                "dataset_id": meta.get("dataset_id"),
                "signal_name": meta.get("signal_name"),
                "signal_params_json": meta.get("signal_params_json"),
                "grid_variant_id": meta.get("grid_variant_id"),
                "eval_tp_ticks": meta.get("eval_tp_ticks"),
                "eval_sl_ticks": meta.get("eval_sl_ticks"),
            },
            reinit=True,
        )
        try:
            metrics = {
                k: float(v)
                for k, v in best_row.items()
                if isinstance(v, (int, float))
            }
            if metrics:
                wandb.log(metrics)
            if not df.empty:
                wandb.log({"threshold_results": wandb.Table(dataframe=df)})
        finally:
            run.finish()
