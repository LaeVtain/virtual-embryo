"""Apply a nuclear reference's expression marginals to joint ranks from a whole-cell forecast."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .chamber_projection import match_means
from .lineage_response import mean


def exchange_marginals(whole, nuclear):
    """Match per-gene quantiles, using local expression to resolve equal whole-cell ranks."""
    n_cells, n_genes = whole.shape
    if nuclear.shape != whole.shape or min(n_cells, n_genes) < 2:
        raise ValueError("Both forecasts must have the same cell and gene counts, each at least two.")
    variance = mean(whole.power(2)) - mean(whole) ** 2
    variable = np.argsort(variance)[-1500:]
    components = min(20, n_cells - 1, len(variable) - 1)
    coordinates = TruncatedSVD(components, random_state=631).fit_transform(whole[:, variable])
    neighbors = NearestNeighbors(n_neighbors=min(16, n_cells), n_jobs=1).fit(coordinates)
    indices = neighbors.kneighbors(coordinates, return_distance=False)
    rows = np.repeat(np.arange(n_cells), indices.shape[1])
    weights = np.full(indices.size, 1 / indices.shape[1], dtype=np.float32)
    adjacency = sparse.csr_matrix((weights, (rows, indices.ravel())), shape=(n_cells, n_cells))

    source = whole.tocsc()
    target = nuclear.tocsc()
    blocks = []
    for start in range(0, n_genes, 128):
        end = min(n_genes, start + 128)
        original = source[:, start:end].tocsr()
        local_mean = (adjacency @ original).toarray()
        expression = original.toarray()
        sorted_values = target[:, start:end].toarray()
        sorted_values.sort(axis=0)
        rng = np.random.default_rng(41771 + start)
        tie_break = rng.random(expression.shape, dtype=np.float32)
        order = np.lexsort((tie_break, local_mean, expression), axis=0)
        assigned = np.empty_like(sorted_values)
        np.put_along_axis(assigned, order, sorted_values, axis=0)
        blocks.append(sparse.csr_matrix(assigned))

    result = sparse.hstack(blocks, format="csr")
    lengths = np.diff(result.indptr)
    if np.any(lengths == 0):
        raise ValueError("Marginal assignment produced an empty cell.")
    mass = np.expm1(result.data)
    totals = np.add.reduceat(mass, result.indptr[:-1])
    mass *= np.repeat(10000 / totals, lengths)
    result.data = np.log1p(mass)
    desired = mean(nuclear)
    for _ in range(4):
        result = match_means(result, desired)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-prediction", type=Path, required=True)
    parser.add_argument("--nuclear-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path.")
    whole = ad.read_h5ad(args.whole_prediction)
    nuclear = ad.read_h5ad(args.nuclear_prediction)
    if not whole.var_names.equals(nuclear.var_names) or not whole.var_names.is_unique:
        parser.error("Both forecasts must use the same unique gene names and order.")
    with threadpool_limits(limits=2):
        result = exchange_marginals(sparse.csr_matrix(whole.X), sparse.csr_matrix(nuclear.X))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(result, var=pd.DataFrame(index=whole.var_names.copy())).write_h5ad(args.output, compression="gzip")
    print(args.output)


if __name__ == "__main__":
    main()
