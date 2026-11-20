"""Model residual cell-state variation using a disjoint set of genes for inference and response."""
import numpy as np
from sklearn.utils.extmath import randomized_svd

from .state_dispersion import restore_marginals, state_coordinates


def residualize(features, coordinates):
    design = np.column_stack([np.ones(len(coordinates)), coordinates])
    coefficients = np.linalg.lstsq(design, features, rcond=None)[0]
    residual = features - design @ coefficients
    return residual / np.maximum(.1, residual.std(axis=0))


def independent_axes(cells, coordinates, genes, mean, strength):
    selected = np.flatnonzero(genes & (mean > .05))
    selected = selected[np.argsort(strength[selected])[-700:]]
    dimensions = min(8, min(len(selected), cells.shape[0]) - 1)
    if dimensions < 1:
        return None
    expression = cells[:, selected].toarray().astype(np.float64)
    expression -= expression.mean(axis=0)
    expression /= np.sqrt(.15 + mean[selected])
    residual = residualize(expression, coordinates)
    vectors, scale, _ = randomized_svd(residual, n_components=dimensions, n_iter=4, random_state=9171)
    return residualize(vectors * scale, coordinates)


def adjust_cross_gene_dispersion(cells, genes):
    """Use factors from one gene set to adjust the other, retaining support and gene means."""
    cells = cells.tocsr()
    coordinates = state_coordinates(cells, genes)
    first = np.random.default_rng(192878).random(len(genes)) < .5
    mean = np.asarray(cells.mean(axis=0, dtype=np.float64)).ravel()
    second = np.asarray(cells.power(2).mean(axis=0, dtype=np.float64)).ravel()
    strength = np.maximum(0, second - mean**2) / (.15 + mean)
    changed = cells.astype(np.float64, copy=True)
    for training in (first, ~first):
        factors = independent_axes(cells, coordinates, training, mean, strength)
        if factors is None:
            continue
        n = len(factors)
        covariance = factors.T @ factors / n + .25 * np.eye(factors.shape[1])
        response = np.linalg.solve(covariance, np.asarray(cells.T @ factors).T / n) / (.15 + mean)
        response[:, training] = 0
        for start in range(0, n, 64):
            end = min(start + 64, n)
            left, right = changed.indptr[[start, end]]
            rows = np.repeat(np.arange(end - start), np.diff(changed.indptr[start:end + 1]))
            conditional = factors[start:end] @ response
            shift = .35 * np.clip(conditional[rows, changed.indices[left:right]], -1.2, 1.2)
            changed.data[left:right] = np.log1p(np.expm1(changed.data[left:right]) * np.exp(shift))
    return restore_marginals(changed, cells)
