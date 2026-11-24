"""Interpolate per-gene ranks of cells sampled from two observed developmental stages."""
import numpy as np
from scipy.spatial import cKDTree


def _ranks(expression, neighborhood, noise):
    order = np.lexsort((noise, neighborhood, expression), axis=0)
    result = np.empty_like(expression)
    quantiles = (np.arange(len(order), dtype=np.float32) + .5) / len(order)
    np.put_along_axis(result, order, np.broadcast_to(quantiles[:, None], order.shape), axis=0)
    return result


def balance_margins(expression, source, target_mean=None):
    """Preserve cell RNA totals and iteratively match the requested gene means in log space."""
    reference = np.expm1(source.astype(np.float64))
    abundance = np.expm1(expression.astype(np.float64))
    sizes = reference.sum(1)
    empty = (abundance.sum(1) == 0) & (sizes > 0)
    abundance[empty] = reference[empty]
    if target_mean is None:
        target_mean = source.mean(0, dtype=np.float64)
    for _ in range(6):
        current = abundance.sum(1)
        abundance *= np.divide(sizes, current, out=np.ones_like(current), where=current > 0)[:, None]
        difference = target_mean - np.log1p(abundance).mean(0)
        derivative = np.maximum((abundance / (1 + abundance)).mean(0), .008)
        abundance *= np.exp(np.clip(difference / derivative, -.45, .45))
    current = abundance.sum(1)
    abundance *= np.divide(sizes, current, out=np.ones_like(current), where=current > 0)[:, None]
    return np.log1p(abundance).astype(np.float32)


def interpolate_ranks(expression, coordinates, from_later, seed):
    source = np.asarray(expression, dtype=np.float32)
    if not len(source) or len(np.unique(from_later)) < 2:
        return source.copy()
    n_neighbors = min(24, len(source))
    neighbors = cKDTree(coordinates).query(coordinates, k=n_neighbors)[1].reshape(len(source), n_neighbors)
    local = np.zeros_like(source)
    for neighbor in neighbors.T:
        local += source[neighbor]
    noise = np.random.default_rng(seed + 71417).random(source.shape, dtype=np.float32)
    pooled = _ranks(source, local, noise)
    within_stage = np.empty_like(pooled)
    for side in (False, True):
        rows = from_later == side
        within_stage[rows] = _ranks(source[rows], local[rows], noise[rows])
    strength = .70
    order = np.argsort((1-strength)*pooled + strength*within_stage, axis=0, kind="stable")
    result = np.empty_like(source)
    np.put_along_axis(result, order, np.sort(source, axis=0), axis=0)
    return balance_margins(result, source)
