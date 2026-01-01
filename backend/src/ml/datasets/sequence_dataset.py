
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class SequenceDataset(Dataset):
    """
    Dataset for Market Transformer.
    Loads raw time-series sequences and metadata labels.
    """
    def __init__(
        self,
        vectors_dir: str,  # Base 'gold/episodes/...' dir
        dates: List[str],
        level_map: Optional[Dict[str, int]] = None,
        outcome_window: str = 'outcome_4min'
    ):
        self.sequences = []
        self.metadatas = []
        self.outcome_col = outcome_window
        
        # Load Data
        vectors_path = Path(vectors_dir)
        valid_dates = []
        
        for date_str in dates:
            seq_file = vectors_path / 'sequences' / f'date={date_str}' / 'sequences.npy'
            meta_file = vectors_path / 'metadata' / f'date={date_str}' / 'metadata.parquet'
            
            if seq_file.exists() and meta_file.exists():
                seq_chunk = np.load(seq_file) # (N, 40, 7)
                meta_chunk = pd.read_parquet(meta_file)
                
                if len(seq_chunk) == len(meta_chunk):
                    self.sequences.append(seq_chunk)
                    self.metadatas.append(meta_chunk)
                    valid_dates.append(date_str)
        
        if not self.sequences:
            print(f"Warning: No data found for dates {dates}")
            self.total_len = 0
            self.X = np.empty((0, 40, 7), dtype=np.float32)
            self.Y_level = np.empty((0,), dtype=np.int64)
            self.Y_outcome = np.empty((0,), dtype=np.int64)
            return

        # Concatenate
        self.X = np.vstack(self.sequences).astype(np.float32) # (Total_N, 40, 7)
        self.meta_df = pd.concat(self.metadatas, ignore_index=True)
        self.total_len = len(self.X)
        
        # Level encoding
        if level_map is None:
            # Build map if not provided
            levels = sorted(self.meta_df['level_kind'].unique())
            self.level_map = {lvl: i for i, lvl in enumerate(levels)}
        else:
            self.level_map = level_map
            
        self.Y_level = self.meta_df['level_kind'].map(self.level_map).fillna(0).astype(int).values
        
        # Outcome encoding (BREAK=1, REJECT=0, CHOP=-1?)
        # For SupCon, we need class labels.
        # Let's map BREAK->0, REJECT->1, CHOP->2
        outcome_map = {'BREAK': 0, 'REJECT': 1, 'BOUNCE': 1, 'CHOP': 2}
        self.Y_outcome = self.meta_df[self.outcome_col].map(outcome_map).fillna(2).astype(int).values
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (40, 7) Tensor
            level_idx: (1,) LongTensor
            outcome_label: (1,) LongTensor
        """
        x = torch.from_numpy(self.X[idx])
        level_idx = torch.tensor(self.Y_level[idx], dtype=torch.long)
        label = torch.tensor(self.Y_outcome[idx], dtype=torch.long)
        return x, level_idx, label

