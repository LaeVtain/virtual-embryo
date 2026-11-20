"""Transport observed cells toward the neural and historical pseudobulk forecasts."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from .model_loader import load_model
from .stream_predict import sample


def read_stage(path, genes=None):
    data = ad.read_h5ad(path, backed="r")
    try:
        names = data.var_names
        if not names.is_unique or (genes is not None and set(names) != set(genes)):
            raise ValueError("Stage has duplicate or inconsistent gene names")
        matrix = data.X.to_memory() if hasattr(data.X, "to_memory") else data.X[:]
        matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        if genes is not None and not names.equals(genes):
            matrix = matrix[:, names.get_indexer(genes)].tocsr()
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Expression must be finite and nonnegative")
        matrix.eliminate_zeros()
        return matrix, names if genes is None else genes
    finally:
        data.file.close()


def gene_mean(matrix):
    return np.asarray(matrix.mean(axis=0, dtype=np.float64)).ravel()


def transport(previous, source, donors, neural, horizon):
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("The forecast horizon must be positive")
    before = gene_mean(previous)
    present = gene_mean(source)
    # Shrink changes for rarely observed genes and bound a two-point extrapolation.
    historical = np.log((present + 0.05) / (before + 0.05))
    slope = np.clip(historical, -np.log(3), np.log(3))
    trend = present * np.exp(horizon / (1 + horizon) * slope)
    target = 0.5 * (trend + gene_mean(neural))
    result = sparse.csr_matrix(donors, dtype=np.float64, copy=True)
    result.eliminate_zeros()
    sizes = np.diff(result.indptr)
    if np.any(sizes == 0):
        raise ValueError("Source contains a cell with an empty RNA library")
    mass = np.expm1(result.data)
    genes = result.shape[1]
    for _ in range(8):
        mass *= np.repeat(10000 / np.add.reduceat(mass, result.indptr[:-1]), sizes)
        logged = np.log1p(mass)
        residual = result.shape[0] * target - np.bincount(result.indices, weights=logged, minlength=genes)
        gradient = np.bincount(result.indices, weights=mass / (1 + mass), minlength=genes)
        # A half step regularizes the correction where the donor sample has few detecting cells.
        adjustment = residual / (gradient + 4)
        mass *= np.exp(0.5 * np.clip(adjustment, -1, 1))[result.indices]
    mass *= np.repeat(10000 / np.add.reduceat(mass, result.indptr[:-1]), sizes)
    result.data = np.log1p(mass)
    return result.astype(np.float32)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--previous-time", required=True, type=float)
    parser.add_argument("--source-time", required=True, type=float)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.previous_time < args.source_time < args.target_time:
        parser.error("Provide a new output file and strictly increasing observation and target times")
    model = load_model(args.model_dir)
    genes = pd.Index(json.loads((model.model_dir / "genes.json").read_text(encoding="utf-8")))
    panel = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines()) if args.panel else genes
    if not panel.is_unique or set(panel) != set(genes):
        parser.error("The output panel must contain the model genes exactly once")
    started = time.perf_counter()
    previous, _ = read_stage(args.previous, genes)
    source, _ = read_stage(args.source, genes)
    if not 1 <= args.cells <= source.shape[0]:
        parser.error("The requested number of cells must lie within the source population")
    selection = np.random.default_rng(args.seed).choice(
        source.shape[0], args.cells, replace=False, p=np.full(source.shape[0], 1 / source.shape[0]))
    donors = source[selection]
    horizon = (args.target_time - args.source_time) / (args.source_time - args.previous_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2), tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        net = model.model("fullELBO30")
        neural, timing = sample(net, donors, args.source_time, args.target_time,
                                args.seed, Path(scratch) / "decoder.npy")
        correction_start = time.perf_counter()
        try:
            prediction = transport(previous, source, donors, neural, horizon)
        finally:
            neural._mmap.close()
        correction_seconds = time.perf_counter() - correction_start
    prediction = prediction[:, genes.get_indexer(panel)].tocsr()
    ad.AnnData(X=prediction, var=pd.DataFrame(index=panel)).write_h5ad(args.output, compression="gzip")
    print(json.dumps(dict(output=str(args.output), rows=prediction.shape[0], genes=prediction.shape[1],
                          previous_rows=previous.shape[0], source_rows=source.shape[0],
                          neural_seconds=timing["seconds"], correction_seconds=correction_seconds,
                          end_to_end_seconds=time.perf_counter() - started), ensure_ascii=False))


if __name__ == "__main__":
    main()
