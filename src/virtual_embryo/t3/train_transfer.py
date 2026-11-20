"""Build T3 cross-gene model assets from wild-type RNA and a spatial donor pair."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve


def read_panel(path, genes):
    with h5py.File(path) as data:
        names = {gene: index for index, gene in enumerate(data["var/_index"].asstr()[:])}
        if genes[-1] not in names:
            raise ValueError(f"RNA file has no column for target {genes[-1]}: {path}")
        if data["X"].attrs["encoding-type"] != "csc_matrix":
            raise ValueError("Expected an official RNA H5AD file with CSC expression data")
        n = data["X"].attrs["shape"][0]
        pointers = data["X/indptr"][:]
        matrix = np.zeros((n, len(genes)), dtype=np.float32)
        for col, gene in enumerate(genes):
            if gene not in names:
                continue
            source = names[gene]
            start, end = pointers[source:source+2]
            matrix[data["X/indices"][start:end], col] = data["X/data"][start:end]
        return matrix, data["obs/celltype/codes"][:]


def gene_signature(files, genes, gene):
    """Ridge on smoothed cell-type expression; exclude the target's measured column."""
    genotype = np.asarray(genes).astype(str)
    designs, labels = [], []
    for path in files:
        matrix, types = read_panel(path, list(genotype) + [gene])
        x, y = matrix[:, :-1], matrix[:, -1]
        x[:, genotype == gene] = 0
        x = (x - x.mean(0)) / np.maximum(x.std(0), 0.1)
        y = (y - y.mean()) / max(float(y.std()), 0.1)
        for identity in np.unique(types):
            mask = types == identity
            if mask.sum() >= 30:
                designs.append(x[mask].mean(0))
                labels.append(y[mask].mean())
    design = np.array(designs, dtype=float)
    kernel = design @ design.T
    # Penalize coefficients by 10% of the mean squared sample norm.
    kernel.flat[::len(kernel)+1] += 0.1 * np.diag(kernel).mean()
    return design.T @ solve(kernel, labels, assume_a="pos")


def make_bank(wt, ko):
    rng = np.random.default_rng(2048)
    wt_rows = rng.choice(len(wt), 6400, replace=len(wt) < 6400)
    ko_rows = rng.choice(len(ko), 1600, replace=len(ko) < 1600)
    return np.vstack((wt[wt_rows], ko[ko_rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt", type=Path, required=True)
    parser.add_argument("--ko", type=Path, required=True)
    parser.add_argument("--rna", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.wt) as wt, np.load(args.ko) as ko:
        genes = wt["genes"].astype(str)
        order = ko["genes"].astype(str).tolist()
        bank = make_bank(wt["x"], ko["x"][:, [order.index(gene) for gene in genes]])
    signature = gene_signature(args.rna, genes, "Ctnnb1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, genes=genes, bank=bank, signature=signature)


if __name__ == "__main__":
    main()
