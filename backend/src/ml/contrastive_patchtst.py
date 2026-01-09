from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTModel
from sklearn.preprocessing import RobustScaler

from src.data_eng.stages.gold.future.episode_embeddings import (
    load_episodes_from_lake,
    extract_all_episode_tensors,
)


LOOKBACK_BARS = 180
OUTCOME_TO_IDX = {
    "STRONG_BREAK": 0,
    "WEAK_BREAK": 1,
    "CHOP": 2,
    "WEAK_BOUNCE": 3,
    "STRONG_BOUNCE": 4,
}
DIRECTION_TO_IDX = {
    "BREAK": 0,
    "BOUNCE": 1,
    "CHOP": 2,
}


def outcome_to_direction(outcome: str) -> str:
    if outcome in ("STRONG_BREAK", "WEAK_BREAK"):
        return "BREAK"
    elif outcome in ("STRONG_BOUNCE", "WEAK_BOUNCE"):
        return "BOUNCE"
    return "CHOP"


@dataclass
class ContrastiveConfig:
    num_input_channels: int
    context_length: int
    patch_length: int = 30
    patch_stride: int = 15
    d_model: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    embedding_dim: int = 256
    temperature: float = 0.07
    use_direction_labels: bool = True


class EpisodeDataset(Dataset):
    def __init__(
        self,
        tensors: np.ndarray,
        metadata: List[dict],
        scaler: RobustScaler = None,
        use_direction: bool = True,
    ):
        self.tensors = tensors.astype(np.float32)
        self.metadata = metadata
        self.scaler = scaler
        self.use_direction = use_direction

        if use_direction:
            self.labels = np.array([
                DIRECTION_TO_IDX[outcome_to_direction(m["outcome"])]
                for m in metadata
            ], dtype=np.int64)
        else:
            self.labels = np.array([
                OUTCOME_TO_IDX[m["outcome"]]
                for m in metadata
            ], dtype=np.int64)

        if scaler is not None:
            n_ep, n_bars, n_feat = self.tensors.shape
            flat = self.tensors.reshape(-1, n_feat)
            flat_scaled = scaler.transform(flat)
            self.tensors = flat_scaled.reshape(n_ep, n_bars, n_feat).astype(np.float32)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": torch.tensor(self.tensors[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ContrastivePatchTST(nn.Module):
    def __init__(self, config: ContrastiveConfig):
        super().__init__()

        patchtst_config = PatchTSTConfig(
            num_input_channels=config.num_input_channels,
            context_length=config.context_length,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            d_model=config.d_model,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            ffn_dim=config.ffn_dim,
            attention_dropout=config.dropout,
            positional_dropout=config.dropout,
            path_dropout=config.dropout,
            ff_dropout=config.dropout,
            head_dropout=config.dropout,
            use_cls_token=False,
        )

        self.backbone = PatchTSTModel(patchtst_config)
        self.projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.embedding_dim),
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(past_values=x, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=[1, 2])
        embeddings = self.projector(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss


def fit_feature_scaler(tensors: np.ndarray) -> RobustScaler:
    n_ep, n_bars, n_feat = tensors.shape
    flat = tensors.reshape(-1, n_feat)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler()
    scaler.fit(flat)
    return scaler


def load_level_data(
    lake_path: Path,
    level_type: str,
    symbols: List[str] = ["ESU5", "ESZ5", "ESH6"],
) -> Tuple[np.ndarray, List[dict], List[str]]:
    all_dfs = []

    for symbol in symbols:
        table_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type.lower()}_episodes"
        if table_path.exists():
            dates = sorted([
                d.name.replace("dt=", "")
                for d in table_path.iterdir()
                if d.name.startswith("dt=")
            ])
            df = load_episodes_from_lake(lake_path, symbol, level_type, dates)
            if len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No data found for {level_type}")

    df = pd.concat(all_dfs, ignore_index=True)
    tensors, metadata, feature_cols = extract_all_episode_tensors(df)

    tensors = np.nan_to_num(tensors, nan=0.0, posinf=0.0, neginf=0.0)

    return tensors, metadata, feature_cols


def train_val_split(
    tensors: np.ndarray,
    metadata: List[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, List[dict], np.ndarray, List[dict]]:
    n = len(tensors)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_ratio))

    train_idx = indices[:split]
    val_idx = indices[split:]

    return (
        tensors[train_idx],
        [metadata[i] for i in train_idx],
        tensors[val_idx],
        [metadata[i] for i in val_idx],
    )


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_retrieval_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    import faiss

    emb_f32 = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(emb_f32)

    index = faiss.IndexFlatIP(emb_f32.shape[1])
    index.add(emb_f32)

    max_k = max(k_values) + 1
    _, all_indices = index.search(emb_f32, max_k)

    results = {}
    for k in k_values:
        precisions = []
        for i in range(len(labels)):
            neighbors = [j for j in all_indices[i] if j != i][:k]
            matches = sum(1 for j in neighbors if labels[j] == labels[i])
            precisions.append(matches / k)
        results[f"P@{k}"] = np.mean(precisions)

    label_counts = np.bincount(labels)
    n = len(labels)
    baseline = sum((c / n) ** 2 for c in label_counts)
    results["baseline"] = baseline
    results["lift@10"] = results["P@10"] / baseline if baseline > 0 else 0

    return results


def train_epoch(
    model: ContrastivePatchTST,
    dataloader: DataLoader,
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        labels = batch["label"].to(device)

        embeddings = model(sequences)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: ContrastivePatchTST,
    dataloader: DataLoader,
    criterion: SupConLoss,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        labels = batch["label"].to(device)

        embeddings = model(sequences)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return avg_loss, embeddings, labels


def train_level_model(
    lake_path: Path,
    level_type: str,
    output_dir: Path,
    config: ContrastiveConfig = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Training {level_type} model")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device()
    print(f"Device: {device}")

    print("Loading data...")
    tensors, metadata, feature_cols = load_level_data(lake_path, level_type)
    print(f"Loaded {len(tensors)} episodes with {len(feature_cols)} features")
    print(f"Tensor shape: {tensors.shape}")

    train_tensors, train_meta, val_tensors, val_meta = train_val_split(
        tensors, metadata, val_ratio, seed
    )
    print(f"Train: {len(train_tensors)}, Val: {len(val_tensors)}")

    print("Fitting scaler on training data...")
    scaler = fit_feature_scaler(train_tensors)

    if config is None:
        config = ContrastiveConfig(
            num_input_channels=tensors.shape[2],
            context_length=tensors.shape[1],
        )

    train_ds = EpisodeDataset(train_tensors, train_meta, scaler, config.use_direction_labels)
    val_ds = EpisodeDataset(val_tensors, val_meta, scaler, config.use_direction_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Config: d_model={config.d_model}, embed_dim={config.embedding_dim}, "
          f"patch={config.patch_length}/{config.patch_stride}")

    model = ContrastivePatchTST(config).to(device)
    criterion = SupConLoss(temperature=config.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_embeddings, val_labels = evaluate(model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == 1:
            metrics = compute_retrieval_metrics(val_embeddings, val_labels)
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"P@10={metrics['P@10']:.1%} lift={metrics['lift@10']:.2f}x")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                best_epoch = epoch

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "scaler_center": scaler.center_,
                    "scaler_scale": scaler.scale_,
                    "feature_cols": feature_cols,
                    "level_type": level_type,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, output_dir / f"contrastive_{level_type.lower()}.pt")

    print(f"\nBest epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"Best metrics: P@10={best_metrics['P@10']:.1%}, lift={best_metrics['lift@10']:.2f}x")

    return {
        "level_type": level_type,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
    }


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device = None,
) -> Tuple[ContrastivePatchTST, RobustScaler, ContrastiveConfig]:
    if device is None:
        device = resolve_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ContrastiveConfig(**checkpoint["config"])
    model = ContrastivePatchTST(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = RobustScaler()
    scaler.center_ = checkpoint["scaler_center"]
    scaler.scale_ = checkpoint["scaler_scale"]

    return model, scaler, config


@torch.no_grad()
def encode_episodes(
    model: ContrastivePatchTST,
    tensors: np.ndarray,
    scaler: RobustScaler,
    device: torch.device = None,
    batch_size: int = 64,
) -> np.ndarray:
    if device is None:
        device = resolve_device()

    model.eval()

    n_ep, n_bars, n_feat = tensors.shape
    flat = tensors.reshape(-1, n_feat)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat_scaled = scaler.transform(flat)
    tensors_scaled = flat_scaled.reshape(n_ep, n_bars, n_feat).astype(np.float32)

    all_embeddings = []
    for i in range(0, n_ep, batch_size):
        batch = torch.tensor(tensors_scaled[i:i + batch_size], dtype=torch.float32).to(device)
        embeddings = model(batch)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train contrastive PatchTST for episode embeddings")
    parser.add_argument("--level", type=str, default="all", help="Level type or 'all'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-length", type=int, default=30)
    parser.add_argument("--patch-stride", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use-outcome", action="store_true", help="Use 5-class outcomes instead of 3-class direction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/models/contrastive")

    args = parser.parse_args()

    lake_path = Path(__file__).parents[2] / "lake"
    output_dir = Path(__file__).parents[2] / args.output_dir

    levels = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"] if args.level == "all" else [args.level]

    results = []
    for level in levels:
        tensors, _, _ = load_level_data(lake_path, level)

        config = ContrastiveConfig(
            num_input_channels=tensors.shape[2],
            context_length=tensors.shape[1],
            d_model=args.d_model,
            embedding_dim=args.embed_dim,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            temperature=args.temperature,
            use_direction_labels=not args.use_outcome,
        )

        result = train_level_model(
            lake_path=lake_path,
            level_type=level,
            output_dir=output_dir,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['level_type']}: P@10={r['best_metrics']['P@10']:.1%}, lift={r['best_metrics']['lift@10']:.2f}x")

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
