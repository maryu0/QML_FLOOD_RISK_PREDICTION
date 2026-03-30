# Hybrid Flood Risk Prediction using Quantum GNN

Hybrid quantum-classical graph learning for flood-risk prediction on a synthetic Indian river-basin dataset.

This repository implements and compares:
- Classical GNN baselines: `GCN`, `LargeGCN`, `GraphSAGE`, `GAT`
- Quantum-inspired models: `ClassicalQGNN` (ablation), `QGNN` (4-qubit and 8-qubit variants)
- Multi-seed evaluation pipeline for paper-ready figures and tables

## Project Highlights

- Synthetic spatio-temporal graph dataset generator (`data.py`)
- Chronological train/val/test splitting (temporal leakage aware)
- Reproducible experiments via seed control
- Per-run artifacts: checkpoint, metrics JSON, ROC, confusion matrix, history curves
- Aggregated paper artifacts: LaTeX table, summary report, comparison figures

## Repository Structure

- `data.py`: Synthetic flood graph dataset generation and processing
- `model.py`: All model architectures and model factory
- `train.py`: Single-model training and evaluation pipeline
- `paper_eval.py`: Multi-model, multi-seed benchmarking and paper artifacts
- `fisher_analysis.py`: Expressiveness/Fisher information utilities
- `FLOOD_MODEL_EXPLANATION.md`: Detailed model explanation
- `presentation.html`, `presentation_ieee.html`, `presentation_aqi.html`: Presentation source files
- `html_to_pptx.py`: HTML-to-PowerPoint conversion utility
- `results/`: Saved model checkpoints, metrics, plots, and summary tables
- `ml_model_v1/`: Earlier version kept for reference

## Requirements

Recommended:
- Python 3.10 or 3.11
- PyTorch with a matching CUDA build (or CPU build)

Main Python dependencies used by the training/eval pipeline:
- `torch`
- `torch-geometric`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pennylane`

Optional (presentation tooling):
- `python-pptx`
- `beautifulsoup4`

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torch-geometric numpy scikit-learn matplotlib pennylane
python -m pip install python-pptx beautifulsoup4
```

## Quick Start

Train a default baseline (`GCN`, 50 epochs, seed 42):

```powershell
python train.py
```

Train a specific model:

```powershell
python train.py --model qgnn --epochs 25 --seed 42
python train.py --model classical_qgnn --epochs 50 --seed 123
python train.py --model gat --epochs 50 --seed 7
```

Available model names:
- `gcn`
- `largegcn`
- `graphsage`
- `gat`
- `classical_qgnn`
- `qgnn_small`
- `qgnn`

## Reproduce Paper-Style Evaluation

Run all configured models across seeds and generate aggregate outputs:

```powershell
python paper_eval.py
```

Skip runs already completed (based on existing metrics JSON):

```powershell
python paper_eval.py --skip_done
```

## Outputs

Single-run outputs (in `results/`):
- `{model}_seed{seed}_best.pt`
- `{model}_seed{seed}_metrics.json`
- `{model}_seed{seed}_history.png`
- `{model}_seed{seed}_roc.png`
- `{model}_seed{seed}_cm.png`
- `{model}_test_report.txt`

Paper/evaluation outputs (in `results/`):
- `paper_roc_comparison.png`
- `paper_confusion_matrices.png`
- `paper_param_efficiency.png`
- `paper_table.tex`
- `paper_summary.txt`

## Notes

- Dataset files are generated/loaded under `data/raw/` and `data/processed/`.
- Default data configuration uses 50 nodes and 1000 timesteps.
- The current code uses `np.ptp(...)` for NumPy 2.x compatibility.

## Git: Commit and Push

If you only changed the README:

```powershell
git add README.md
git commit -m "Add project README with setup and usage"
git push origin main
```

If you want to include all tracked/untracked changes:

```powershell
git add .
git commit -m "Update project documentation"
git push origin main
```
