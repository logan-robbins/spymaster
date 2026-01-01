
import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.ml.models.market_transformer import PatchTSTEncoder
from src.ml.datasets.sequence_dataset import SequenceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, level_idx, label) in enumerate(loader):
        x = x.to(device)
        level_idx = level_idx.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        embedding = model(x, level_idx) # (B, 32)
        
        # Temporary: Using Model output directly as logits for 3 classes
        # We need to add a temporary projection to classes for this test script
        if not hasattr(model, 'temp_classifier'):
             model.temp_classifier = nn.Linear(32, 3).to(device)
        
        logits = model.temp_classifier(embedding)
        
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Run with dummy data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    # 0. WandB Setup
    use_wandb = False
    try:
        import wandb
        # Only init if API key is present OR if we are doing a real run
        # For dummy verification, we usually skip unless forced, but user asked for "Integration" check.
        # If API key is missing, this handles gracefully.
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(project="spymaster-transformer", config=vars(args))
            use_wandb = True
    except ImportError:
        logger.warning("wandb not installed, skipping logging")

    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Data Setup
    if args.dummy:
        logger.info("Generating DUMMY data for verification...")
        N = 1000
        # (N, 40, 7) random checks inputs
        X = torch.randn(N, 40, 7)
        # Levels 0-15
        Level = torch.randint(0, 16, (N,))
        # Outcomes 0-2
        Y = torch.randint(0, 3, (N,))
        
        train_ds = TensorDataset(X, Level, Y)
        val_ds = TensorDataset(X[:100], Level[:100], Y[:100])
        n_levels = 16
    else:
        # Load real data (Not implemented in this check script yet)
        logger.error("Real data loading requires path args. Run with --dummy for check.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # 2. Model Setup
    model = PatchTSTEncoder(
        c_in=7, 
        d_model=64, 
        n_levels=n_levels, 
        output_dim=32
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    logger.info("Starting Training Loop...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.1%}")
        
        if use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch})
        
    logger.info("✅ Training Loop Completed Successfully.")
    logger.info("Artifacts saved to: (Memory only for dummy run)")

if __name__ == "__main__":
    main()
