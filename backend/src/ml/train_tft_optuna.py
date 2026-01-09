"""
Optuna hyperparameter search for TFT-style contrastive encoder.
"""

import sys
sys.path.insert(0, 'src')

import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from data_eng.stages.gold.future.episode_embeddings import (
    load_episodes_from_lake,
    extract_all_episode_tensors,
)

DIRECTION_TO_IDX = {'BREAK': 0, 'BOUNCE': 1, 'CHOP': 2}


def outcome_to_direction(outcome: str) -> str:
    if outcome in ('STRONG_BREAK', 'WEAK_BREAK'):
        return 'BREAK'
    elif outcome in ('STRONG_BOUNCE', 'WEAK_BOUNCE'):
        return 'BOUNCE'
    return 'CHOP'


class EpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> dict:
        return {'seq': self.X[i], 'label': self.y[i]}


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.layer_norm(residual + gate * out)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim // 4, hidden_dim, dropout)
            for _ in range(n_features)
        ])
        self.softmax_grn = GatedResidualNetwork(n_features * hidden_dim, hidden_dim, n_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_feat = x.shape

        processed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]
            processed.append(self.feature_grns[i](feat_i))

        processed = torch.stack(processed, dim=-1)

        flat = processed.view(batch, seq_len, -1)
        weights = F.softmax(self.softmax_grn(flat), dim=-1)

        selected = (processed * weights.unsqueeze(2)).sum(dim=-1)
        return selected


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TFTEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

        self.d_model = d_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vsn(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_grn(x)
        x = self.projector(x)
        return F.normalize(x, dim=1)


def supcon_loss(emb: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    sim = torch.matmul(emb, emb.T) / temperature
    mask = torch.eq(y.view(-1, 1), y.view(1, -1)).float()
    diag = 1 - torch.eye(len(y), device=emb.device)
    exp_sim = torch.exp(sim) * diag
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
    mask_sum = (mask * diag).sum(1).clamp(min=1)
    return -((mask * diag * log_prob).sum(1) / mask_sum).mean()


def load_level_data(lake_path: Path, level_type: str) -> tuple:
    all_tensors = []
    all_metadata = []

    for symbol in ['ESU5', 'ESZ5', 'ESH6']:
        table_path = lake_path / f'silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type.lower()}_episodes'
        if not table_path.exists():
            continue

        dates = sorted([d.name.replace('dt=', '') for d in table_path.iterdir() if d.name.startswith('dt=')])

        for i in range(0, len(dates), 20):
            batch_dates = dates[i:i + 20]
            df = load_episodes_from_lake(lake_path, symbol, level_type, batch_dates)
            if len(df) == 0:
                continue

            tensors, metadata, _ = extract_all_episode_tensors(df)
            if len(tensors) > 0:
                tensors = np.nan_to_num(tensors, nan=0.0, posinf=0.0, neginf=0.0)
                all_tensors.append(tensors)
                all_metadata.extend(metadata)
            del df

    return np.concatenate(all_tensors, axis=0), all_metadata


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    nn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn_model.fit(embeddings)
    _, all_idx = nn_model.kneighbors(embeddings)

    p10 = 0.0
    for i in range(len(labels)):
        neighbors = [j for j in all_idx[i] if j != i][:10]
        p10 += sum(1 for j in neighbors if labels[j] == labels[i]) / 10
    p10 /= len(labels)

    baseline = sum((np.sum(labels == c) / len(labels)) ** 2 for c in range(3))
    return {'P@10': p10, 'baseline': baseline, 'lift': p10 / baseline}


class TFTObjective:
    def __init__(self, lake_path: Path, level_type: str, device: torch.device):
        self.device = device
        self.level_type = level_type

        print(f'Loading {level_type} data...')
        tensors, metadata = load_level_data(lake_path, level_type)
        self.n_ep, self.n_bars, self.n_feat = tensors.shape
        print(f'  {self.n_ep} episodes, {self.n_bars} bars, {self.n_feat} features')

        labels = np.array([DIRECTION_TO_IDX[outcome_to_direction(m['outcome'])] for m in metadata], dtype=np.int64)
        print(f'  Labels: {Counter(labels)}')

        flat = tensors.reshape(-1, self.n_feat)
        self.scaler = RobustScaler()
        flat_scaled = self.scaler.fit_transform(flat)
        tensors_scaled = flat_scaled.reshape(self.n_ep, self.n_bars, self.n_feat).astype(np.float32)

        np.random.seed(42)
        indices = np.random.permutation(self.n_ep)
        split = int(self.n_ep * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]

        self.train_tensors = tensors_scaled[train_idx]
        self.train_labels = labels[train_idx]
        self.val_tensors = tensors_scaled[val_idx]
        self.val_labels = labels[val_idx]

        print(f'  Train: {len(self.train_tensors)}, Val: {len(self.val_tensors)}')

    def __call__(self, trial: Trial) -> float:
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        nhead = trial.suggest_categorical('nhead', [2, 4])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        temperature = trial.suggest_float('temperature', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        if d_model % nhead != 0:
            raise optuna.TrialPruned()

        torch.manual_seed(42)
        np.random.seed(42)

        train_ds = EpDataset(self.train_tensors, self.train_labels)
        val_ds = EpDataset(self.val_tensors, self.val_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = TFTEncoder(
            n_features=self.n_feat,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            embed_dim=embed_dim,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=lr / 100)

        best_lift = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(1, 81):
            model.train()
            for batch in train_loader:
                x = batch['seq'].to(self.device)
                y = batch['label'].to(self.device)

                emb = model(x)
                loss = supcon_loss(emb, y, temperature)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            if epoch % 5 == 0:
                model.eval()
                val_embs = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['seq'].to(self.device)
                        emb = model(x)
                        val_embs.append(emb.cpu().numpy())

                val_embs = np.concatenate(val_embs)
                metrics = compute_metrics(val_embs, self.val_labels)

                if metrics['lift'] > best_lift:
                    best_lift = metrics['lift']
                    patience_counter = 0
                else:
                    patience_counter += 1

                trial.report(metrics['lift'], epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                if patience_counter >= patience // 5:
                    break

        return best_lift


def run_search(level_type: str, n_trials: int = 50):
    lake_path = Path('lake')
    output_dir = Path('data/models/contrastive/optuna')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    objective = TFTObjective(lake_path, level_type, device)

    study = optuna.create_study(
        direction='maximize',
        study_name=f'tft_{level_type}',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n{"="*60}')
    print(f'TFT Best trial for {level_type}:')
    print(f'  Lift: {study.best_trial.value:.4f}')
    print(f'  Params: {study.best_trial.params}')

    results = {
        'level_type': level_type,
        'architecture': 'TFT',
        'best_lift': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
    }

    with open(output_dir / f'tft_{level_type.lower()}_optuna.json', 'w') as f:
        json.dump(results, f, indent=2)

    return study


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='PM_HIGH')
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()

    run_search(args.level, args.n_trials)


if __name__ == '__main__':
    main()
