"""Project a stage-specific whole-embryo expression trend onto observed T1 cells."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .transport import read_stage


def gene_multipliers(trend, genes, horizon, interval):
    if not (0 < horizon <= interval):
        raise ValueError("The forecast horizon must lie within the measured trend interval")
    if not trend.index.is_unique or not np.isfinite(trend.to_numpy()).all():
        raise ValueError("Expected one finite log fold change per gene")
    log_ratio = trend.reindex(genes).fillna(0).clip(-np.log(8), np.log(8))
    return np.exp(log_ratio.to_numpy() * (horizon / interval))


def adjust_cells(expression, multiplier):
    linear = sparse.csr_matrix(expression, dtype=np.float64, copy=True)
    linear.data = np.expm1(linear.data) * multiplier[linear.indices]
    totals = np.asarray(linear.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Expected finite, nonempty log-normalized source cells")
    linear.data = np.log1p(linear.data * np.repeat(10000 / totals, np.diff(linear.indptr)))
    linear.eliminate_zeros()
    return linear.astype(np.float32)


def forecast(source, multiplier, cells, seed, reweight=False):
    if not 0 < cells <= source.shape[0] or len(multiplier) != source.shape[1]:
        raise ValueError("Check the number of source cells and aligned genes")
    weights = np.ones(source.shape[0])
    if reweight:
        for start in range(0, len(weights), 256):
            block = source[start:start + 256].astype(np.float64)
            block.data = np.expm1(block.data)
            totals = np.asarray(block.sum(axis=1)).ravel()
            if np.any(totals <= 0):
                raise ValueError("Expected nonempty source cells")
            weights[start:start + len(totals)] = (block @ multiplier) / totals
    rows = np.random.default_rng(seed).choice(len(weights), cells, replace=False, p=weights / weights.sum())
    return adjust_cells(source[rows], multiplier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--trend", type=Path, required=True, help="CSV indexed by gene, with log_ratio in natural-log units")
    parser.add_argument("--interval", type=float, required=True, help="Days between the two reference profiles")
    parser.add_argument("--horizon", type=float, required=True, help="Days from source cells to forecast")
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    genes = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    if not genes.is_unique:
        parser.error("The gene panel contains duplicate names")
    source, genes = read_stage(args.source, genes)
    coefficients = pd.read_csv(args.trend, index_col=0)["log_ratio"]
    multiplier = gene_multipliers(coefficients, genes, args.horizon, args.interval)
    result = forecast(source, multiplier, args.cells, args.seed, args.reweight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=result, var=pd.DataFrame(index=genes)).write_h5ad(args.output, compression="gzip")


if __name__ == "__main__":
    main()
