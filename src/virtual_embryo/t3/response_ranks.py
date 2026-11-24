"""Combine mutant and wild-type cell-state ranks at the predicted gene marginals."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def _joint_ranks(expression, seed):
    cells, genes = expression.shape
    model = PCA(min(16, cells - 1, genes), svd_solver="randomized", random_state=473)
    embedded = model.fit_transform(expression)
    neighbors = cKDTree(embedded).query(embedded, k=min(16, cells))[1]
    local = np.zeros_like(expression)
    for rows in neighbors.T:
        local += expression[rows]
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(expression.shape, dtype=np.float32), local, expression), axis=0)
    ranks = np.empty_like(expression)
    values = np.broadcast_to(np.arange(cells)[:, None], order.shape)
    np.put_along_axis(ranks, order, values, axis=0)
    return ranks / cells


def couple_response(wild_type, predicted, seed):
    if len(predicted) < 2:
        return predicted.copy()
    mutant_ranks = _joint_ranks(predicted, seed + 71)
    control_ranks = _joint_ranks(wild_type, seed + 71)
    order = np.argsort(.5 * mutant_ranks + .5 * control_ranks, axis=0, kind="stable")
    assigned = np.empty_like(predicted)
    np.put_along_axis(assigned, order, np.sort(predicted, axis=0), axis=0)
    mass = np.expm1(assigned.astype(float))
    totals = np.expm1(predicted).sum(axis=1, dtype=float)
    empty = (mass.sum(axis=1) == 0) & (totals > 0)
    mass[empty] = np.expm1(predicted[empty])
    desired = predicted.mean(axis=0, dtype=float)
    for _ in range(8):
        sizes = mass.sum(axis=1)
        mass *= np.divide(totals, sizes, out=np.ones_like(totals), where=sizes > 0)[:, None]
        error = desired - np.log1p(mass).mean(axis=0)
        sensitivity = (mass / (1 + mass)).mean(axis=0)
        mass *= np.exp(np.clip(error / np.maximum(sensitivity, .008), -.45, .45))
    sizes = mass.sum(axis=1)
    mass *= np.divide(totals, sizes, out=np.ones_like(totals), where=sizes > 0)[:, None]
    return np.log1p(mass).astype(np.float32)
