"""Predict an unseen perturbation using stage-selected WT references and WT RNA."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .mixture import predict_mixture
from .train_transfer import gene_signature, make_bank, read_panel


def within_type_response(rna_files, genes, gene):
    """Estimate signed association after removing cell type and panel depth."""
    estimates = []
    for path in rna_files:
        matrix, types = read_panel(path, list(genes) + [gene])
        x = matrix[:, :-1].copy()
        y = matrix[:, -1].copy()
        x[:, genes == gene] = 0
        for identity in np.unique(types):
            mask = types == identity
            x[mask] -= x[mask].mean(0)
            y[mask] -= y[mask].mean()
        depth = x.sum(1)
        y -= depth * ((y @ depth) / max(depth @ depth, 1e-7))
        estimates.append(-(x.T @ y) / max(y @ y, 1e-7))
    estimate = np.mean(estimates, axis=0)
    return estimate / max(0.01, np.abs(estimate).max())


def _read_bank(path, genes, sample_size=None):
    def align(profiles, names):
        order = {name: index for index, name in enumerate(names)}
        if len(names) != len(order) or set(names) != set(genes):
            raise ValueError(f"Expected one column for each spatial panel gene in {path}")
        if sample_size is not None:
            rows = np.random.default_rng(618).choice(profiles.shape[0], sample_size, replace=False)
            profiles = profiles[rows]
        if sparse.issparse(profiles):
            profiles = profiles.toarray()
        return profiles[:, [order[g] for g in genes]]

    if Path(path).suffix.lower() == ".h5ad":
        data = ad.read_h5ad(path, backed="r")
        try:
            return align(data.layers["log1p"].astype(np.float32), data.var_names)
        finally:
            data.file.close()
    with np.load(path, allow_pickle=False) as data:
        return align(data["x"], data["genes"].astype(str))


def reference_banks(wt_path, ko_path, atlas_paths, genes):
    bank = make_bank(_read_bank(wt_path, genes), _read_bank(ko_path, genes))
    references = [_read_bank(path, genes, sample_size=6400) for path in atlas_paths]
    return bank, references


def fit(wt_path, ko_path, atlas_paths, rna_files, gene="Ctnnb1"):
    with np.load(wt_path, allow_pickle=False) as data:
        genes = data["genes"].astype(str)
    bank, references = reference_banks(wt_path, ko_path, atlas_paths, genes)
    return dict(genes=genes, bank=bank, atlas=np.vstack(references),
                atlas_means=np.stack([ref.mean(0) for ref in references]),
                signature=gene_signature(rna_files, genes, gene),
                rna_response=within_type_response(rna_files, genes, gene))


def _select_reference_bank(x, model, seed):
    bank = model["bank"].copy()
    atlas = model["atlas"]
    means = model["atlas_means"]
    recipient_mean = x.mean(0)
    direction = bank[:6400].mean(0) - recipient_mean
    progress = (means - recipient_mean) @ direction / max(float(direction @ direction), 1e-7)
    age_weights = np.clip(progress, 0.05, 1)
    weights = np.repeat(age_weights, len(atlas) // len(age_weights)).astype(float)
    weights /= weights.sum()
    atlas_rows = np.random.default_rng(911).choice(len(atlas), 4800, replace=False, p=weights)
    bank[1600:6400] = atlas[atlas_rows]

    # Ridge approximation to matching a small shift in per-gene population means.
    scale = np.maximum(bank.std(0), 0.2)
    design = (bank - bank.mean(0)) / scale
    covariance = design.T @ design / len(design)
    covariance.flat[::len(covariance)+1] += 1
    coefficients = solve(covariance, model["rna_response"] / scale, assume_a="pos")
    log_weights = (0.75 * np.clip(design @ coefficients, -2, 2)).astype(np.float64)
    probabilities = softmax(log_weights)
    rng = np.random.default_rng(seed + 73)
    order = rng.permutation(len(bank))
    cdf = np.cumsum(probabilities[order])
    cdf[-1] = 1
    positions = (rng.random() + np.arange(len(bank))) / len(bank)
    counts = np.bincount(order[np.searchsorted(cdf, positions)], minlength=len(bank))
    removed = np.flatnonzero(counts == 0)
    repeats = np.repeat(np.arange(len(bank)), np.maximum(counts - 1, 0))
    bank[removed] = bank[rng.permutation(repeats)]
    return bank


def predict(x, coords, model, seed=101):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bank = _select_reference_bank(x, model, seed)
    prediction = predict_mixture(x, coords, bank, seed, fraction=0.5,
                                  signature=model["signature"], sensitivity=0.5)
    expression = prediction["x"].astype(float)
    difference = expression - x
    shift = difference.mean(0)
    sampling_variance = difference.var(0) / len(x)
    reliability = shift**2 / np.maximum(shift**2 + sampling_variance, 1e-12)
    derivative = (-np.expm1(-expression)).mean(0)
    log_gain = np.clip(0.6 * reliability * shift / np.maximum(derivative, 0.02), -0.8, 0.8)
    abundance = np.expm1(expression)
    total = abundance.sum(1)
    abundance *= np.exp(log_gain)
    abundance *= np.divide(total, abundance.sum(1), out=np.ones_like(total), where=total>0)[:, None]
    prediction["x"] = np.log1p(abundance).astype(np.float32)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt", type=Path, required=True)
    parser.add_argument("--ko", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, nargs=3, required=True,
                        help="Spatial WT NPZs in E8.25, E8.5, E10.5 order")
    parser.add_argument("--rna", type=Path, nargs=2, required=True, help="WT E8.5 and E9.5 RNA H5ADs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        model = fit(args.wt, args.ko, args.atlas, args.rna)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **model)


if __name__ == "__main__":
    main()
