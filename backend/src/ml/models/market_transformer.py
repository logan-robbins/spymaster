
import torch
import torch.nn as nn
import math
from typing import Optional

class PatchTSTEncoder(nn.Module):
    """
    Time-Series Transformer with Patching and Level Conditioning.
    
    Architecture:
    1. Patching: Input (B, 40, 7) -> (B, N_patches, Patch_dim)
    2. Embedding: Linear Projection + Positional Encoding
    3. Conditioning: Level Kind Embedding added to sequence
    4. Transformer: Standard Encoder Layers
    5. Head: Pooling -> Linear -> Output (32D)
    """
    def __init__(
        self,
        c_in: int = 7,             # Channels (Vel, Acc, Jerk, etc.)
        d_model: int = 128,        # Latent dimension
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        patch_len: int = 8,
        stride: int = 4,
        seq_len: int = 40,
        n_levels: int = 16,        # Number of unique level kinds
        output_dim: int = 32       # Final embedding size
    ):
        super().__init__()
        
        # Patching Params
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = int((seq_len - patch_len) / stride) + 2 # Approx padding logic
        
        # Input Projection (Patch -> d_model)
        # We flatten patch: (c_in * patch_len) -> d_model
        combined_dim = c_in * patch_len
        self.input_projection = nn.Linear(combined_dim, d_model)
        
        # Positional Encoding (Learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1, 20, d_model)) # Max 20 patches
        
        # Level Conditioning
        self.level_embedding = nn.Embedding(n_levels, d_model)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
        
    def forward(self, x: torch.Tensor, level_kind_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 40, 7) Raw Time Series
            level_kind_idx: (B,) Integer indices of level types
            
        Returns:
            (B, 32) Metric Embedding
        """
        B, L, C = x.shape
        
        # 1. Patching
        # Unfold: (B, C, L) -> (B, C, N_patches, Patch_len) ?
        # Easier manual unfolding for control
        # Ideal: (B, N_patches, C*Patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride) 
        # Shape: (B, N_patches, C, Patch_len)
        
        # Flatten patches
        B, N_patches, C, P_len = patches.shape
        patches = patches.permute(0, 1, 2, 3).reshape(B, N_patches, C * P_len)
        
        # 2. Projection + Positional
        # (B, N, combined_dim) -> (B, N, d_model)
        x_emb = self.input_projection(patches)
        x_emb = x_emb + self.positional_encoding[:, :N_patches, :]
        
        # 3. Conditioning (Prepend Level Token or Add?)
        # Let's Prepend as a CLS token modifier
        lvl_emb = self.level_embedding(level_kind_idx).unsqueeze(1) # (B, 1, d_model)
        
        # Concatenate: [Level, Patch1, Patch2...]
        x_seq = torch.cat([lvl_emb, x_emb], dim=1)
        
        # 4. Transformer
        x_out = self.encoder(x_seq)
        
        # 5. Pooling (Take the Level Token output as CLS)
        cls_token = x_out[:, 0, :]
        
        # 6. Head
        embedding = self.head(cls_token)
        return embedding

