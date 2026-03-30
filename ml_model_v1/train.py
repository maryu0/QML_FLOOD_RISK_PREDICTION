# train.py
# Basic training script for flood risk models.
#
# Usage:
#   python train.py --model gcn   --epochs 30
#   python train.py --model qgnn  --epochs 10
#
# TODO: add proper validation loop with early stopping
# TODO: save best checkpoint (by val AUC) instead of last epoch
# TODO: add ROC-AUC metric -- accuracy alone is misleading on imbalanced data
# TODO: add learning-rate scheduler
# TODO: plot training curves

import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data  import FloodDataset
from model import GCN, QGNN


def get_model(name, in_channels):
    if name == 'gcn':
        return GCN(in_channels=in_channels)
    elif name == 'qgnn':
        return QGNN(in_channels=in_channels)
    else:
        raise ValueError(f'Unknown model: {name}')


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out  = model(batch.x, batch.edge_index)   # (N_total, 1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        preds  = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().flatten()
        labels = batch.y.long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      default='gcn',  choices=['gcn', 'qgnn'])
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cpu')   # quantum sim runs on CPU only

    # ---- Dataset ----
    ds    = FloodDataset(root='data')
    n     = len(ds)
    t_end = int(n * 0.70)

    # TODO: use the temporal_split() helper from data.py once it's added
    train_ds = ds[:t_end]
    val_ds   = ds[t_end:]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ---- Model ----
    in_ch = ds[0].x.shape[1]
    model = get_model(args.model, in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model}  params: {n_params:,}')

    # ---- Loss ----
    # pos_weight handles class imbalance (flood is rare)
    # TODO: compute this dynamically from training label distribution
    pos_weight = torch.tensor([5.0])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- Optimiser ----
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1 = evaluate(model, val_loader, device)
        print(f'epoch {epoch:3d}/{args.epochs}  '
              f'loss={loss:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}')

    # ---- Save checkpoint ----
    os.makedirs('results', exist_ok=True)
    ckpt = f'results/{args.model}_last.pt'
    torch.save(model.state_dict(), ckpt)
    print(f'Saved -> {ckpt}')


if __name__ == '__main__':
    main()
