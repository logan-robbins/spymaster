import sys
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import math
import json

np.random.seed(42)

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

def outcome_to_direction(outcome):
    if outcome in ('STRONG_BREAK', 'WEAK_BREAK'): return 'BREAK'
    elif outcome in ('STRONG_BOUNCE', 'WEAK_BOUNCE'): return 'BOUNCE'
    return 'CHOP'

class EpDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {'seq': self.X[i], 'label': self.y[i]}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalEncoder(nn.Module):
    def __init__(self, n_features, d_model=256, nhead=8, num_layers=4, embed_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.projector(x)
        return F.normalize(x, dim=1)

def supcon_loss(emb, y, temp=0.07):
    sim = torch.matmul(emb, emb.T) / temp
    mask = torch.eq(y.view(-1, 1), y.view(1, -1)).float()
    diag = 1 - torch.eye(len(y), device=emb.device)
    exp_sim = torch.exp(sim) * diag
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
    mask_sum = (mask * diag).sum(1).clamp(min=1)
    return -((mask * diag * log_prob).sum(1) / mask_sum).mean()

def load_level_data(lake_path, level_type):
    all_tensors = []
    all_metadata = []

    for symbol in ['ESU5', 'ESZ5', 'ESH6']:
        table_path = lake_path / f'silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type.lower()}_episodes'
        if not table_path.exists():
            continue

        dates = sorted([d.name.replace('dt=', '') for d in table_path.iterdir() if d.name.startswith('dt=')])

        for i in range(0, len(dates), 20):
            batch_dates = dates[i:i+20]
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

def compute_metrics(embeddings, labels):
    nn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn_model.fit(embeddings)
    _, all_idx = nn_model.kneighbors(embeddings)

    p10 = 0.0
    for i in range(len(labels)):
        neighbors = [j for j in all_idx[i] if j != i][:10]
        p10 += sum(1 for j in neighbors if labels[j] == labels[i]) / 10
    p10 /= len(labels)

    baseline = sum((np.sum(labels == c) / len(labels))**2 for c in range(3))
    return {'P@10': p10, 'baseline': baseline, 'lift': p10 / baseline}

def train_level(lake_path, level_type, output_dir, device, epochs=50, batch_size=64, lr=1e-4):
    print(f'\n{"="*60}')
    print(f'Training {level_type}')
    print(f'{"="*60}')

    torch.manual_seed(42)
    np.random.seed(42)

    tensors, metadata = load_level_data(lake_path, level_type)
    n_ep, n_bars, n_feat = tensors.shape
    print(f'Loaded {n_ep} episodes')

    labels = np.array([DIRECTION_TO_IDX[outcome_to_direction(m['outcome'])] for m in metadata], dtype=np.int64)
    print(f'Labels: {Counter(labels)}')

    flat = tensors.reshape(-1, n_feat)
    scaler = RobustScaler()
    flat_scaled = scaler.fit_transform(flat)
    tensors_scaled = flat_scaled.reshape(n_ep, n_bars, n_feat).astype(np.float32)

    indices = np.random.permutation(n_ep)
    split = int(n_ep * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = EpDataset(tensors_scaled[train_idx], labels[train_idx])
    val_ds = EpDataset(tensors_scaled[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TemporalEncoder(n_features=n_feat, d_model=256, nhead=8, num_layers=4, embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_lift = 0
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch['seq'].to(device)
            y = batch['label'].to(device)

            emb = model(x)
            loss = supcon_loss(emb, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_embs = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['seq'].to(device)
                    emb = model(x)
                    val_embs.append(emb.cpu().numpy())
                    val_labels.append(batch['label'].numpy())

            val_embs = np.concatenate(val_embs)
            val_labels = np.concatenate(val_labels)

            metrics = compute_metrics(val_embs, val_labels)

            marker = ' *' if metrics['lift'] > best_lift else ''
            if metrics['lift'] > best_lift:
                best_lift = metrics['lift']
                best_metrics = metrics
                best_epoch = epoch

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'd_model': 256,
                    'embed_dim': 256,
                    'n_features': n_feat,
                    'n_bars': n_bars,
                    'epoch': epoch,
                    'metrics': {k: float(v) for k, v in metrics.items()},
                    'scaler_center': scaler.center_,
                    'scaler_scale': scaler.scale_,
                    'level_type': level_type,
                }, output_dir / f'{level_type.lower()}.pt')

            print(f'Epoch {epoch:2d} | loss={total_loss/len(train_loader):.4f} | P@10={metrics["P@10"]:.1%} | lift={metrics["lift"]:.2f}x{marker}')

    print(f'Best: epoch {best_epoch}, P@10={best_metrics["P@10"]:.1%}, lift={best_lift:.2f}x')
    return {'level_type': level_type, 'n_episodes': n_ep, 'best_epoch': best_epoch, 'best_metrics': best_metrics}

def main():
    lake_path = Path('lake')
    output_dir = Path('data/models/contrastive')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    results = []
    for level_type in ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW']:
        result = train_level(lake_path, level_type, output_dir, device)
        results.append(result)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"Level":<12} {"Episodes":<10} {"P@10":<10} {"Baseline":<10} {"Lift":<8}')
    print('-'*50)
    for r in results:
        m = r['best_metrics']
        print(f'{r["level_type"]:<12} {r["n_episodes"]:<10} {m["P@10"]:.1%}      {m["baseline"]:.1%}      {m["lift"]:.2f}x')

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print('\nDone!')

if __name__ == '__main__':
    main()
