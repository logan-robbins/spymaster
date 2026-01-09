from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTModel
from sklearn.preprocessing import RobustScaler
import faiss

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from data_eng.stages.gold.future.episode_embeddings import (
    load_episodes_from_lake,
    extract_all_episode_tensors,
    LOOKBACK_BARS,
)


DIRECTION_TO_IDX = {"BREAK": 0, "BOUNCE": 1, "CHOP": 2}


def outcome_to_direction(outcome: str) -> str:
    if outcome in ("STRONG_BREAK", "WEAK_BREAK"):
        return "BREAK"
    elif outcome in ("STRONG_BOUNCE", "WEAK_BOUNCE"):
        return "BOUNCE"
    return "CHOP"


class EpisodeDataset(Dataset):
    def __init__(self, tensors: np.ndarray, labels: np.ndarray):
        self.tensors = torch.tensor(tensors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"sequence": self.tensors[idx], "label": self.labels[idx]}


class ContrastiveEncoder(nn.Module):
    def __init__(self, n_features: int, context_length: int, d_model: int = 128, embed_dim: int = 128):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=n_features,
            context_length=context_length,
            patch_length=30,
            patch_stride=15,
            d_model=d_model,
            num_hidden_layers=3,
            num_attention_heads=4,
            ffn_dim=d_model * 2,
            dropout=0.1,
            attention_dropout=0.1,
            use_cls_token=False,
        )

        self.backbone = PatchTSTModel(config)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embed_dim),
        )
        self.d_model = d_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(past_values=x)
        pooled = out.last_hidden_state.mean(dim=[1, 2])
        emb = self.projector(pooled)
        return F.normalize(emb, p=2, dim=1)


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    device = embeddings.device

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    diag_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * diag_mask

    exp_sim = torch.exp(similarity) * diag_mask
    log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

    mask_sum = mask.sum(1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

    loss = -(mask * log_prob).sum(1) / mask_sum
    return loss.mean()


def load_data(lake_path: Path, level_type: str) -> Tuple[np.ndarray, List[dict], List[str]]:
    symbols = ["ESU5", "ESZ5", "ESH6"]
    all_dfs = []

    contract_path = Path(__file__).parents[2] / "src/data_eng/contracts/silver/future/level_relative_features.avsc"

    for symbol in symbols:
        table_path = lake_path / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type.lower()}_approach"
        if table_path.exists():
            dates = sorted([d.name.replace("dt=", "") for d in table_path.iterdir() if d.name.startswith("dt=")])
            df = load_episodes_from_lake(lake_path, symbol, level_type, dates)
            if len(df) > 0:
                all_dfs.append(df)
                print(f"  {symbol}: {df['episode_id'].nunique()} episodes")

    if not all_dfs:
        raise ValueError(f"No data found for {level_type}")

    df = pd.concat(all_dfs, ignore_index=True)
    tensors, metadata, feature_cols = extract_all_episode_tensors(
        df, lookback_bars=LOOKBACK_BARS, contract_path=contract_path
    )
    tensors = np.nan_to_num(tensors, nan=0.0, posinf=0.0, neginf=0.0)

    return tensors, metadata, feature_cols


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray, k_values: List[int] = [5, 10, 20]) -> Dict:
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


def train_level(
    lake_path: Path,
    level_type: str,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    d_model: int = 128,
    embed_dim: int = 128,
    lr: float = 1e-4,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Training {level_type}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data...")
    tensors, metadata, feature_cols = load_data(lake_path, level_type)
    n_episodes, n_bars, n_features = tensors.shape
    print(f"Total: {n_episodes} episodes, {n_bars} bars, {n_features} features (contract-derived)")

    labels = np.array([DIRECTION_TO_IDX[outcome_to_direction(m["outcome"])] for m in metadata], dtype=np.int64)
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")

    print("Scaling features...")
    flat = tensors.reshape(-1, n_features)
    scaler = RobustScaler()
    flat_scaled = scaler.fit_transform(flat)
    tensors_scaled = flat_scaled.reshape(n_episodes, n_bars, n_features).astype(np.float32)

    indices = np.random.permutation(n_episodes)
    split = int(n_episodes * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = EpisodeDataset(tensors_scaled[train_idx], labels[train_idx])
    val_ds = EpisodeDataset(tensors_scaled[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"Creating model (d_model={d_model}, embed_dim={embed_dim})...")
    model = ContrastiveEncoder(n_features, n_bars, d_model, embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_lift = 0.0
    best_epoch = 0
    best_metrics = None

    print("Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, y = batch["sequence"], batch["label"]

            emb = model(x)
            loss = supcon_loss(emb, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_embs = []
            val_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    emb = model(batch["sequence"])
                    val_embs.append(emb.numpy())
                    val_labels.append(batch["label"].numpy())

            val_embs = np.concatenate(val_embs)
            val_labels = np.concatenate(val_labels)

            metrics = compute_metrics(val_embs, val_labels)
            print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | P@10={metrics['P@10']:.1%} | lift={metrics['lift@10']:.2f}x")

            if metrics["lift@10"] > best_lift:
                best_lift = metrics["lift@10"]
                best_epoch = epoch
                best_metrics = metrics

                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "d_model": d_model,
                    "embed_dim": embed_dim,
                    "n_features": n_features,
                    "n_bars": n_bars,
                    "scaler_center": scaler.center_,
                    "scaler_scale": scaler.scale_,
                    "level_type": level_type,
                    "epoch": epoch,
                    "metrics": metrics,
                    "feature_cols": feature_cols,
                    "lookback_bars": n_bars,
                }
                torch.save(checkpoint, output_dir / f"contrastive_{level_type.lower()}.pt")

    print(f"\nBest: epoch {best_epoch}, lift={best_lift:.2f}x, P@10={best_metrics['P@10']:.1%}")
    return {"level_type": level_type, "best_epoch": best_epoch, "best_lift": best_lift, "best_metrics": best_metrics}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lake_path = Path(__file__).parents[2] / "lake"
    output_dir = Path(__file__).parents[2] / "data/models/contrastive"

    levels = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"] if args.level == "all" else [args.level]

    results = []
    for level in levels:
        result = train_level(
            lake_path=lake_path,
            level_type=level,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            embed_dim=args.embed_dim,
            lr=args.lr,
            seed=args.seed,
        )
        results.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['level_type']}: P@10={r['best_metrics']['P@10']:.1%}, lift={r['best_lift']:.2f}x")

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
