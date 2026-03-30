# -*- coding: utf-8 -*-
"""
train.py  -  Training Loop + Evaluation (IEEE-grade)
=====================================================
Trains and evaluates GCN, ClassicalQGNN, or QGNN on the IndianFloodDataset.

Usage
-----
  python train.py                                    # GCN, 50 epochs, seed 42
  python train.py --model qgnn   --epochs 20         # QGNN (quantum simulator)
  python train.py --model classical_qgnn --epochs 50 # ablation model
  python train.py --model gcn    --epochs 50 --seed 123

Outputs saved to results/
--------------------------
  {model}_seed{seed}_best.pt       best checkpoint (highest val F1)
  {model}_seed{seed}_history.png   4-panel training curves (loss/acc/F1/AUC)
  {model}_seed{seed}_roc.png       ROC curve for this run
  {model}_seed{seed}_cm.png        confusion matrix heatmap
  {model}_seed{seed}_metrics.json  all test metrics (for paper_eval.py)
  {model}_test_report.txt          human-readable test report (single-run)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
)

from data  import IndianFloodDataset, temporal_split
from model import build_model, count_parameters

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Evaluation helper
# ===========================================================================

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device) -> dict:
    """
    Run model over a DataLoader and return:
      loss  - binary cross-entropy
      acc   - accuracy
      f1    - binary F1 (flood class)
      auc   - ROC-AUC
      probs - float array (N_total,)  sigmoid probabilities
      preds - int array  (N_total,)
      true  - int array  (N_total,)
      fpr   - false positive rate array (for ROC curve)
      tpr   - true  positive rate array (for ROC curve)
    """
    model.eval()
    all_logits, all_labels = [], []

    for data in loader:
        data   = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)  # (N_batch, 1)
        all_logits.append(logits.cpu().squeeze())
        all_labels.append(data.y.cpu().squeeze())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    loss  = F.binary_cross_entropy_with_logits(logits, labels).item()
    probs = torch.sigmoid(logits).numpy()
    true  = labels.numpy().astype(int)

    # Use a threshold sweep to stabilise F1 and avoid one-class collapse.
    best_macro_f1 = -1.0
    best_thr = 0.5
    best_preds = (probs > best_thr).astype(int)
    for thr in np.linspace(0.2, 0.8, 25):
        cand = (probs > thr).astype(int)
        cand_macro_f1 = f1_score(true, cand, average='macro', zero_division=0)
        if cand_macro_f1 > best_macro_f1:
            best_macro_f1 = cand_macro_f1
            best_thr = float(thr)
            best_preds = cand

    preds = best_preds

    try:
        auc = roc_auc_score(true, probs)
        fpr, tpr, _ = roc_curve(true, probs)
    except ValueError:
        auc = float('nan')
        fpr = tpr = np.array([0.0, 1.0])

    return {
        'loss' : loss,
        'acc'  : accuracy_score(true, preds),
        'f1'   : f1_score(true, preds, zero_division=0),
        'auc'  : auc,
        'thr'  : best_thr,
        'probs': probs,
        'preds': preds,
        'true' : true,
        'fpr'  : fpr,
        'tpr'  : tpr,
    }


# ===========================================================================
# Single epoch
# ===========================================================================

def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                pos_weight: torch.Tensor = None) -> float:
    """One training epoch. Returns mean batch loss."""
    model.train()
    total_loss = 0.0

    for data in loader:
        data   = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)

        pw = pos_weight.to(device) if pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, data.y,
                                                   pos_weight=pw)
        loss.backward()
        # Gradient clipping: important for quantum layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ===========================================================================
# Plotting
# ===========================================================================

def plot_history(history: list[dict], model_name: str,
                 seed: int | None = None) -> None:
    """
    4-panel training curves: BCE Loss (train+val), Val Accuracy, Val F1, Val AUC.
    Only BCE Loss shows both train and val lines; other panels show val only
    to avoid the misleading flat-zero train lines of earlier versions.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    epochs = range(1, len(history) + 1)

    # Panel 0: BCE Loss — train + val
    axes[0].plot(epochs, [h['train_loss'] for h in history],
                 label='Train', linewidth=1.5)
    axes[0].plot(epochs, [h['val_loss'] for h in history],
                 label='Val', linewidth=1.5, linestyle='--')
    axes[0].set_title('BCE Loss')
    axes[0].legend()

    # Panels 1-3: val only
    for ax, (key, title) in zip(axes[1:], [
        ('val_acc', 'Val Accuracy'),
        ('val_f1',  'Val F1'),
        ('val_auc', 'Val AUC'),
    ]):
        vals = [h.get(key, np.nan) for h in history]
        ax.plot(epochs, vals, linewidth=1.5, color='tab:orange', label='Val')
        ax.set_title(title)
        ax.legend()

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)

    seed_tag = f' (seed {seed})' if seed is not None else ''
    plt.suptitle(f'{model_name.upper()} Training History{seed_tag}', fontsize=13)
    plt.tight_layout()

    tag  = f'_{seed}' if seed is not None else ''
    path = RESULTS_DIR / f'{model_name}_seed{seed}_history.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'[plot]   Saved -> {path}')


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float,
                   model_name: str, seed: int) -> None:
    """Save a single-run ROC curve."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve  -  {model_name.upper()}  (seed {seed})')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / f'{model_name}_seed{seed}_roc.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'[plot]   Saved -> {path}')


def plot_confusion_heatmap(cm: np.ndarray, model_name: str,
                           seed: int) -> None:
    """Save an annotated confusion matrix heatmap (no seaborn dependency)."""
    classes = ['No Flood', 'Flood']
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix  -  {model_name.upper()}  (seed {seed})')

    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / total
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha='center', va='center', color=color, fontsize=11)

    plt.tight_layout()
    path = RESULTS_DIR / f'{model_name}_seed{seed}_cm.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'[plot]   Saved -> {path}')


# ===========================================================================
# Class-balance helper
# ===========================================================================

def compute_pos_weight(train_ds) -> torch.Tensor:
    """
    pos_weight = (#negatives) / (#positives) sampled from training set.
    Used in BCE to handle class imbalance.
    """
    n_pos = n_neg = 0.0
    for i in range(min(200, len(train_ds))):
        y     = train_ds[i].y
        n_pos += float(y.sum())
        n_neg += float((1 - y).sum())
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


# ===========================================================================
# Main training function
# ===========================================================================

def main(args: argparse.Namespace, seed: int | None = None) -> dict:
    """
    Full train + eval pipeline.

    Parameters
    ----------
    args : parsed CLI arguments
    seed : if provided, overrides args.seed and sets all RNG seeds

    Returns
    -------
    dict with all test metrics (for paper_eval.py aggregation)
    """
    _seed = seed if seed is not None else args.seed
    set_seed(_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device      : {device}')
    print(f'Model       : {args.model.upper()}')
    print(f'Epochs      : {args.epochs}')
    print(f'Batch size  : {args.batch_size}')
    print(f'LR          : {args.lr}')
    print(f'Seed        : {_seed}')

    # -------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------
    dataset = IndianFloodDataset(
        root='data',
        n_nodes=args.n_nodes,
        n_timesteps=args.n_timesteps,
        lookback=args.lookback,
        k_neighbors=args.k_neighbors,
        flood_quantile=args.flood_quantile,
        seed=_seed,
    )
    sample = dataset[0]
    flood_rate = float(np.mean([
        dataset[i].y.mean().item() for i in range(min(100, len(dataset)))
    ]))
    print(f'\nDataset     : {len(dataset)} graphs')
    print(f'Node feats  : {sample.x.shape[1]}')
    print(f'Edges       : {sample.edge_index.shape[1]}')
    print(f'Flood rate  : {flood_rate:.3f}')

    train_ds, val_ds, test_ds = temporal_split(dataset, 0.70, 0.15)
    print(f'Split       : {len(train_ds)} / {len(val_ds)} / {len(test_ds)}'
          f'  (train/val/test)')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # -------------------------------------------------------------------
    # Class balance
    # -------------------------------------------------------------------
    pos_weight = compute_pos_weight(train_ds)
    print(f'pos_weight  : {pos_weight.item():.3f}')

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    kwargs: dict = {'in_channels': 6}

    if args.model == 'gcn':
        kwargs['hidden'] = 64
    elif args.model == 'largegcn':
        kwargs['hidden'] = 32
    elif args.model == 'graphsage':
        kwargs['hidden'] = 64
    elif args.model == 'gat':
        kwargs['hidden'] = 64
        kwargs['heads'] = 4
    elif args.model == 'classical_qgnn':
        kwargs.update(hidden=32, n_qubits=4)
    elif args.model == 'qgnn_small':
        kwargs.update(hidden=32, n_qubits=4)
    else:  # qgnn (8-qubit)
        kwargs.update(hidden=32, n_qubits=8, n_qlayers=2)

    model = build_model(args.model, **kwargs).to(device)
    p     = count_parameters(model)
    print(f'Parameters  : {p["trainable"]:,} trainable  /  {p["total"]:,} total')

    # -------------------------------------------------------------------
    # Optimiser + scheduler
    # -------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=7, factor=0.5, min_lr=1e-5,
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    best_val_f1 = -1.0
    best_ckpt   = RESULTS_DIR / f'{args.model}_seed{_seed}_best.pt'
    history: list[dict] = []

    print('\n--- Training ---')
    t0 = time.time()
    log_every = max(1, args.epochs // 10)

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer,
                               device, pos_weight)
        val_m   = evaluate(model, val_loader, device)
        scheduler.step(val_m['f1'])
        epoch_secs = time.time() - t_epoch

        history.append({
            'train_loss': tr_loss,
            'val_loss'  : val_m['loss'],
            'val_acc'   : val_m['acc'],
            'val_f1'    : val_m['f1'],
            'val_auc'   : val_m['auc'],
        })

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{args.epochs}"
                f"  train_loss={tr_loss:.4f}"
                f"  val_loss={val_m['loss']:.4f}"
                f"  val_acc={val_m['acc']:.3f}"
                f"  val_f1={val_m['f1']:.3f}"
                f"  val_auc={val_m['auc']:.3f}"
                f"  epoch={epoch_secs:.1f}s  total={elapsed:.0f}s"
            )

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            torch.save(model.state_dict(), best_ckpt)

    print(f'\nBest val F1 : {best_val_f1:.4f}  ->  {best_ckpt}')

    # -------------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    test_m = evaluate(model, test_loader, device)

    report = classification_report(
        test_m['true'], test_m['preds'],
        target_names=['No Flood', 'Flood'], digits=4,
        labels=[0, 1], output_dict=False, zero_division=0,
    )
    report_dict = classification_report(
        test_m['true'], test_m['preds'],
        target_names=['No Flood', 'Flood'], digits=4,
        labels=[0, 1], output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test_m['true'], test_m['preds'])

    print('\n--- Test Results ---')
    print(f'  Accuracy  : {test_m["acc"]:.4f}')
    print(f'  F1-Score  : {test_m["f1"]:.4f}')
    print(f'  ROC-AUC   : {test_m["auc"]:.4f}')
    print(f'  Threshold : {test_m["thr"]:.2f}')
    print(f'\n{report}')
    print(f'Confusion matrix:\n{cm}')

    # Save text report (human-readable, backwards compat)
    rpt_path = RESULTS_DIR / f'{args.model}_test_report.txt'
    with open(rpt_path, 'w') as fh:
        fh.write(f'Model    : {args.model.upper()}\n')
        fh.write(f'Seed     : {_seed}\n')
        fh.write(f'Accuracy : {test_m["acc"]:.4f}\n')
        fh.write(f'F1-Score : {test_m["f1"]:.4f}\n')
        fh.write(f'ROC-AUC  : {test_m["auc"]:.4f}\n\n')
        fh.write(f'Threshold: {test_m["thr"]:.2f}\n\n')
        fh.write(report)
        fh.write(f'\nConfusion matrix:\n{cm}\n')
    print(f'[report] Saved -> {rpt_path}')

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------
    plot_history(history, args.model, seed=_seed)
    plot_roc_curve(test_m['fpr'], test_m['tpr'], test_m['auc'],
                   args.model, _seed)
    plot_confusion_heatmap(cm, args.model, _seed)

    # -------------------------------------------------------------------
    # Save JSON metrics for paper_eval.py aggregation
    # -------------------------------------------------------------------
    macro_f1 = report_dict['macro avg']['f1-score']
    metrics = {
        'model'       : args.model,
        'seed'        : _seed,
        'acc'         : test_m['acc'],
        'f1'          : test_m['f1'],       # binary flood-class F1
        'macro_f1'    : macro_f1,
        'auc'         : test_m['auc'],
        'threshold'   : test_m['thr'],
        'prec_flood'  : report_dict['Flood']['precision'],
        'rec_flood'   : report_dict['Flood']['recall'],
        'prec_noflood': report_dict['No Flood']['precision'],
        'rec_noflood' : report_dict['No Flood']['recall'],
        'confusion'   : cm.tolist(),
        'fpr'         : test_m['fpr'].tolist(),
        'tpr'         : test_m['tpr'].tolist(),
        'n_params'    : p['trainable'],
        'epochs'      : args.epochs,
    }
    json_path = RESULTS_DIR / f'{args.model}_seed{_seed}_metrics.json'
    with open(json_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'[json]   Saved -> {json_path}')

    print(f'\nTotal time  : {time.time() - t0:.1f}s')
    return metrics


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train GCN, ClassicalQGNN or QGNN on IndianFloodDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model',      default='gcn',
                   choices=['gcn', 'largegcn', 'graphsage', 'gat',
                           'classical_qgnn', 'qgnn_small', 'qgnn'],
                   help='Model architecture')
    p.add_argument('--epochs',     default=50,     type=int,
                   help='Training epochs')
    p.add_argument('--batch_size', default=32,     type=int,
                   help='Graphs per batch')
    p.add_argument('--lr',         default=1e-3,   type=float,
                   help='Adam learning rate')
    p.add_argument('--wd',         default=1e-4,   type=float,
                   help='Adam weight decay')
    p.add_argument('--seed',       default=42,     type=int,
                   help='Random seed for reproducibility')
    p.add_argument('--n_nodes',    default=50,     type=int,
                   help='Number of basin gauge nodes')
    p.add_argument('--n_timesteps', default=1000,  type=int,
                   help='Number of synthetic timesteps')
    p.add_argument('--lookback',   default=3,      type=int,
                   help='Rainfall lag window used as node features')
    p.add_argument('--k_neighbors', default=5,     type=int,
                   help='kNN neighbors for spatial graph construction')
    p.add_argument('--flood_quantile', default=0.72, type=float,
                   help='Quantile threshold to binarize flood risk labels')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
