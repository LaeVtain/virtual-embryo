"""Evaluate an expression prediction with the historical heldout metrics."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from ..paths import task_assets
from .metrics import reference_statistics, sample_metrics
from .predict import load_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path, default=task_assets('t1'))
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe = joblib.load(args.probe or args.model_dir/'prepared/probe.joblib')
    target = load_input(args.target)
    if hasattr(target, 'toarray'):
        target = target.toarray()
    truth = reference_statistics(target, probe['svd'], probe['clusters'])
    metrics = sample_metrics(np.load(args.prediction, mmap_mode='r'), truth,
                             probe['svd'], probe['clusters'], probe['bandwidth'])
    result = dict(evaluator='historical_cell_holdout', metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
