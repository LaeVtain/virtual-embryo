"""Disk-backed prediction for every T1 source/target-time route."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .model_loader import load_model
from .stream_predict import predict


def load_input(path):
    if path.suffix.lower() == '.h5ad':
        import h5py
        from anndata._io.specs import read_elem
        from scipy import sparse
        with h5py.File(path, 'r') as handle:
            return sparse.csr_matrix(read_elem(handle['X']))
    if path.suffix.lower() == '.npy':
        return np.load(path, mmap_mode='r')
    with np.load(path, allow_pickle=False) as data:
        if 'x' in data:
            return data['x'].astype('float32')
    from scipy import sparse
    return sparse.load_npz(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--source-time', type=float, required=True)
    parser.add_argument('--target-time', type=float, required=True)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    x = load_input(args.input)
    model = load_model(args.model_dir)
    began = time.monotonic()
    y, audit = predict(model, x, args.source_time, args.target_time, args.seed, args.output_dir)
    audit.update(source_time=args.source_time, target_time=args.target_time, seed=args.seed,
                 shape=list(y.shape), wall_seconds=time.monotonic()-began)
    (args.output_dir/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    shutil.copyfile(model.model_dir/'genes.json', args.output_dir/'genes.json')
    print(args.output_dir/'candidate.npy')


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
