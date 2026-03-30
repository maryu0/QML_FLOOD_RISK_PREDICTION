# paper_eval.py
# Quick multi-seed evaluation to compare GCN vs QGNN.
# Runs each model 3 times with different seeds, prints a summary table.
#
# Usage:
#   python paper_eval.py
#
# TODO: add ClassicalQGNN ablation once model.py has it
# TODO: save LaTeX table
# TODO: generate ROC comparison figure
# TODO: generate confusion matrix grid figure
# TODO: skip already-done runs (--skip_done flag)

import subprocess
import sys
import json
from pathlib import Path

import numpy as np

SEEDS  = [42, 123, 7]
MODELS = {
    'gcn' : 30,   # epochs
    'qgnn': 10,   # fewer epochs -- quantum sim is slower
    # TODO: 'classical_qgnn': 30,  -- add once implemented
}

RESULTS = Path('results')


def run_training(model, epochs, seed):
    """
    Launch train.py as a subprocess for a given model/epochs/seed.
    Returns the metrics dict loaded from results JSON, or None on failure.

    TODO: call train.main() directly once train.py has a proper
          importable main(args, seed) function
    """
    print(f'\n  Running: {model}  epochs={epochs}  seed={seed}')
    # TODO: pass --seed once train.py supports it  (currently hardcoded to cpu)
    cmd = [
        sys.executable, 'train.py',
        '--model',  model,
        '--epochs', str(epochs),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f'  WARNING: training failed for {model} seed={seed}')
        return None

    # train.py doesn't save JSON yet -- read last-epoch checkpoint metrics
    # TODO: update train.py to export metrics JSON, then load it here
    return None   # placeholder


def main():
    print('Multi-seed evaluation')
    print(f'Models : {list(MODELS.keys())}')
    print(f'Seeds  : {SEEDS}')

    # For now just print what we'd collect
    # TODO: actually collect metrics once train.py saves JSON per run
    for model, epochs in MODELS.items():
        print(f'\n=== {model.upper()} ===')
        for seed in SEEDS:
            run_training(model, epochs, seed)

    print('\n--- Summary ---')
    print('(TODO: once train.py exports metrics JSON, aggregate mean +/- std here)')
    print('Expected columns: Model | Params | Accuracy | F1 | AUC')


if __name__ == '__main__':
    main()
