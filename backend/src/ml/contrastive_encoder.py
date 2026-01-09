from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler


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


class ContrastiveTemporalEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embed_dim),
        )

        self.d_model = d_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.projector(x)
        return F.normalize(x, dim=1)


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    device = embeddings.device

    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
    diag_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * diag_mask

    exp_sim = torch.exp(similarity) * diag_mask
    log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

    mask_sum = mask.sum(1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

    loss = -(mask * log_prob).sum(1) / mask_sum
    return loss.mean()


def load_encoder(checkpoint_path: Path, device: torch.device = None) -> Tuple[ContrastiveTemporalEncoder, RobustScaler, Dict]:
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ContrastiveTemporalEncoder(
        n_features=checkpoint['n_features'],
        d_model=checkpoint.get('d_model', 256),
        embed_dim=checkpoint.get('embed_dim', 256),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = RobustScaler()
    scaler.center_ = checkpoint['scaler_center']
    scaler.scale_ = checkpoint['scaler_scale']

    metadata = {
        'level_type': checkpoint['level_type'],
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'n_features': checkpoint['n_features'],
        'n_bars': checkpoint['n_bars'],
        'feature_cols': checkpoint.get('feature_cols', []),
        'lookback_bars': checkpoint.get('lookback_bars', checkpoint['n_bars']),
    }

    return model, scaler, metadata


@torch.no_grad()
def encode_episodes(
    model: ContrastiveTemporalEncoder,
    tensors: np.ndarray,
    scaler: RobustScaler,
    device: torch.device = None,
    batch_size: int = 64,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

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
