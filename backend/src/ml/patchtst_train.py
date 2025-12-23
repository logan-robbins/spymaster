"""
PatchTST multi-task trainer for SPY break/bounce modeling.

Trains a shared PatchTST backbone with:
- classification head (BREAK vs BOUNCE)
- regression head (strength_signed)

Requires sequence datasets produced by sequence_dataset_builder.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTModel
import mlflow
import wandb


@dataclass
class DatasetBundle:
    X: np.ndarray
    mask: np.ndarray
    y_break: np.ndarray
    y_strength: np.ndarray
    static: np.ndarray
    seq_feature_names: List[str]
    static_feature_names: List[str]


class SequenceDataset(Dataset):
    def __init__(
        self,
        bundle: DatasetBundle,
        use_static: bool,
    ) -> None:
        self.X = bundle.X.astype(np.float32)
        self.mask = bundle.mask.astype(np.float32)
        self.y_break = bundle.y_break.astype(np.int64)
        self.y_strength = bundle.y_strength.astype(np.float32)
        self.static = bundle.static.astype(np.float32)
        self.use_static = use_static

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.X[idx]
        mask = self.mask[idx]
        if self.use_static:
            static = self.static[idx]
            static_seq = np.repeat(static[None, :], x.shape[0], axis=0)
            x = np.concatenate([x, static_seq], axis=-1)

        return {
            "past_values": torch.tensor(x, dtype=torch.float32),
            "past_observed_mask": torch.tensor(mask, dtype=torch.float32),
            "y_break": torch.tensor(self.y_break[idx], dtype=torch.long),
            "y_strength": torch.tensor(self.y_strength[idx], dtype=torch.float32),
        }


class PatchTSTMultiTask(nn.Module):
    def __init__(self, config: PatchTSTConfig, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = PatchTSTModel(config)
        self.classifier = nn.Linear(config.d_model, num_classes)
        self.regressor = nn.Linear(config.d_model, 1)

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        if past_observed_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            time_mask = past_observed_mask[..., 0].float()
            denom = time_mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            pooled = (hidden * time_mask.unsqueeze(-1)).sum(dim=1) / denom

        logits = self.classifier(pooled)
        strength = self.regressor(pooled).squeeze(-1)
        return logits, strength


def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _parse_date_from_filename(path: Path) -> str:
    name = path.stem
    if "sequence_dataset_" not in name:
        raise ValueError(f"Unexpected dataset filename: {path.name}")
    return name.split("sequence_dataset_")[-1]


def _resolve_dataset_files(data_dir: Path, dates: List[str]) -> List[Path]:
    files = sorted(data_dir.glob("sequence_dataset_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sequence_dataset_*.npz files in {data_dir}")
    if dates:
        files = [f for f in files if _parse_date_from_filename(f) in dates]
    if not files:
        raise ValueError("No datasets matched the requested dates")
    return files


def _load_dataset_bundle(
    files: List[Path],
    use_static: bool,
) -> DatasetBundle:
    bundles = []
    seq_feature_names = None
    static_feature_names = None

    for file_path in files:
        payload = _load_npz(file_path)
        bundles.append(payload)

        if seq_feature_names is None:
            seq_feature_names = payload["seq_feature_names"].tolist()
        if static_feature_names is None:
            static_feature_names = payload["static_feature_names"].tolist()

    def concat(field: str, dtype: np.dtype) -> np.ndarray:
        return np.concatenate([b[field].astype(dtype) for b in bundles], axis=0)

    X = concat("X", np.float32)
    mask = concat("mask", np.float32)
    y_break = concat("y_break", np.int64)
    y_strength = concat("y_strength", np.float32)
    static = concat("static", np.float32) if use_static else np.zeros((len(X), 0), dtype=np.float32)

    if seq_feature_names is None:
        seq_feature_names = []
    if static_feature_names is None:
        static_feature_names = []

    return DatasetBundle(
        X=X,
        mask=mask,
        y_break=y_break,
        y_strength=y_strength,
        static=static,
        seq_feature_names=seq_feature_names,
        static_feature_names=static_feature_names,
    )


def _filter_break_bounce(bundle: DatasetBundle) -> DatasetBundle:
    keep = (bundle.y_break == 0) | (bundle.y_break == 1)
    if not np.any(keep):
        raise ValueError("No BREAK/BOUNCE samples found after filtering")

    return DatasetBundle(
        X=bundle.X[keep],
        mask=bundle.mask[keep],
        y_break=bundle.y_break[keep],
        y_strength=bundle.y_strength[keep],
        static=bundle.static[keep],
        seq_feature_names=bundle.seq_feature_names,
        static_feature_names=bundle.static_feature_names,
    )


def _split_dates(all_dates: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    if not all_dates:
        return [], []
    unique_dates = sorted(set(all_dates))
    split_idx = max(1, int(len(unique_dates) * (1.0 - val_ratio)))
    return unique_dates[:split_idx], unique_dates[split_idx:]


def _get_all_dates(data_dir: Path) -> List[str]:
    return [_parse_date_from_filename(p) for p in data_dir.glob("sequence_dataset_*.npz")]


def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total if total > 0 else 0.0

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_files(paths: List[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _load_features_version(backend_root: Path) -> str:
    features_path = backend_root / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found at {features_path}")
    payload = json.loads(features_path.read_text())
    version = payload.get("version")
    if not version:
        raise ValueError("features.json missing version field")
    return version


def _require_wandb_config() -> None:
    if os.getenv("WANDB_API_KEY"):
        return
    if os.getenv("WANDB_MODE") == "offline":
        return
    raise EnvironmentError("WANDB_API_KEY not set and WANDB_MODE is not 'offline'")


def _resolve_run_dates(train_dates: List[str], val_dates: List[str]) -> Tuple[str, str]:
    all_dates = sorted(set(train_dates + val_dates))
    if not all_dates:
        raise ValueError("No dataset dates available for run naming")
    return all_dates[0], all_dates[-1]


def train(args: argparse.Namespace) -> int:
    torch.set_float32_matmul_precision("high")
    _seed_everything(args.seed)
    device = _resolve_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_root = Path(__file__).resolve().parents[2]

    all_dates = _get_all_dates(data_dir)
    if args.train_dates or args.val_dates:
        train_dates = [d.strip() for d in args.train_dates.split(",") if d.strip()] if args.train_dates else []
        val_dates = [d.strip() for d in args.val_dates.split(",") if d.strip()] if args.val_dates else []
    else:
        train_dates, val_dates = _split_dates(all_dates, args.val_ratio)

    use_static = not args.no_static
    train_files = _resolve_dataset_files(data_dir, train_dates)
    val_files = _resolve_dataset_files(data_dir, val_dates) if val_dates else []
    if not val_files:
        raise ValueError("Validation datasets required; provide --val-dates or ensure at least two dates for --val-ratio")

    train_bundle = _filter_break_bounce(
        _load_dataset_bundle(train_files, use_static=use_static)
    )
    val_bundle = _filter_break_bounce(
        _load_dataset_bundle(val_files, use_static=use_static)
    ) if val_files else None

    train_ds = SequenceDataset(train_bundle, use_static=use_static)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = None
    if val_bundle is not None:
        val_ds = SequenceDataset(val_bundle, use_static=use_static)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    num_channels = train_bundle.X.shape[-1]
    if use_static:
        num_channels += train_bundle.static.shape[-1]

    config = PatchTSTConfig(
        num_input_channels=num_channels,
        context_length=train_bundle.X.shape[1],
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        num_hidden_layers=args.num_hidden_layers,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        ffn_dim=args.ffn_dim,
        attention_dropout=args.dropout,
        positional_dropout=args.dropout,
        path_dropout=args.dropout,
        ff_dropout=args.dropout,
        head_dropout=args.dropout,
        use_cls_token=False,
    )

    model = PatchTSTMultiTask(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.SmoothL1Loss()

    _require_wandb_config()
    schema_version = _load_features_version(backend_root)
    run_start, run_end = _resolve_run_dates(train_dates, val_dates)
    run_name = f"patchtst-v{schema_version}-{run_start}_{run_end}-{args.seed}"
    dataset_hash = _hash_files(train_files + val_files)

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "spymaster_patchtst"))
    mlflow_tags = {
        "schema_version": schema_version,
        "level_universe": "spy_options",
        "sma_mean_reversion": "true",
        "dealer_velocity": "true",
        "confluence": "true",
        "thresholds": "1_and_2",
    }
    wandb_tags = [
        "level_universe",
        "sma_mean_reversion",
        "dealer_velocity",
        "confluence",
        "thresholds",
    ]

    params = {
        "data_dir": str(data_dir),
        "train_dates": ",".join(train_dates),
        "val_dates": ",".join(val_dates),
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "num_hidden_layers": args.num_hidden_layers,
        "d_model": args.d_model,
        "num_attention_heads": args.num_attention_heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "classification_weight": args.classification_weight,
        "regression_weight": args.regression_weight,
        "num_channels": num_channels,
        "context_length": train_bundle.X.shape[1],
        "seed": args.seed,
        "dataset_hash": dataset_hash,
    }

    run_metadata = {
        "schema_version": schema_version,
        "dataset_hash": dataset_hash,
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "train_samples": len(train_bundle.X),
        "val_samples": len(val_bundle.X) if val_bundle is not None else 0,
        "seq_feature_names": train_bundle.seq_feature_names,
        "static_feature_names": train_bundle.static_feature_names,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2))

    features_path = backend_root / "features.json"

    best_val = float("inf")
    best_model_path = None

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(mlflow_tags)
        mlflow.log_params(params)
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(features_path))

        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "spymaster_patchtst"),
            name=run_name,
            config=params,
            tags=wandb_tags,
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                past_values = batch["past_values"].to(device)
                mask = batch["past_observed_mask"].to(device)
                labels = batch["y_break"].to(device)
                strength = batch["y_strength"].to(device)

                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, past_values.shape[-1]).bool()

                logits, strength_pred = model(past_values, mask_expanded)
                loss_cls = classification_loss(logits, labels)
                loss_reg = regression_loss(strength_pred, strength)
                loss = args.classification_weight * loss_cls + args.regression_weight * loss_reg

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            wandb.log({"train_loss": avg_loss}, step=epoch)
            print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f}")

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    past_values = batch["past_values"].to(device)
                    mask = batch["past_observed_mask"].to(device)
                    labels = batch["y_break"].to(device)
                    strength = batch["y_strength"].to(device)

                    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, past_values.shape[-1]).bool()
                    logits, strength_pred = model(past_values, mask_expanded)

                    loss_cls = classification_loss(logits, labels)
                    loss_reg = regression_loss(strength_pred, strength)
                    loss = args.classification_weight * loss_cls + args.regression_weight * loss_reg

                    val_loss += loss.item()
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())

            avg_val_loss = val_loss / max(1, len(val_loader))
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            metrics = _compute_metrics(logits_cat, labels_cat)

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_precision", metrics["precision"], step=epoch)
            mlflow.log_metric("val_recall", metrics["recall"], step=epoch)
            mlflow.log_metric("val_f1", metrics["f1"], step=epoch)
            wandb.log(
                {
                    "val_loss": avg_val_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                    "val_f1": metrics["f1"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d} | val_loss={avg_val_loss:.4f} "
                f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}"
            )

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_model_path = output_dir / "patchtst_multitask.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "seq_features": train_bundle.seq_feature_names,
                        "static_features": train_bundle.static_feature_names,
                    },
                    best_model_path,
                )

        if best_model_path is None:
            raise RuntimeError("Validation did not produce a saved model checkpoint")

        mlflow.log_metric("best_val_loss", best_val)
        mlflow.log_artifact(str(best_model_path))

        model_artifact = wandb.Artifact(
            "patchtst_model",
            type="model",
            metadata={"best_val_loss": best_val, "schema_version": schema_version},
        )
        model_artifact.add_file(str(best_model_path))
        model_artifact.add_file(str(metadata_path))
        model_artifact.add_file(str(features_path))
        wandb_run.log_artifact(model_artifact)
        wandb.finish()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PatchTST multi-task trainer (MPS compatible)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with sequence_dataset_*.npz")
    parser.add_argument("--output-dir", type=str, default="data/models/patchtst", help="Output directory")
    parser.add_argument("--train-dates", type=str, default="", help="Comma-separated training dates")
    parser.add_argument("--val-dates", type=str, default="", help="Comma-separated validation dates")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio if dates not provided")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--no-static", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
