"""Stratified sampling of detected molecules in the spatial cardiac forecast."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.stats import binom, poisson


def molecule_counts(x):
    linear = np.expm1(x)
    unit = np.where(linear > 0, linear, np.inf).min(axis=1)
    return np.rint(linear / unit[:, None]).astype(np.int32)


def donor_adjustment(expression, weights, rows, observed_counts):
    """Estimate the source gene fractions expected in draws without replacement."""
    if len(expression) == len(rows):
        return np.zeros(expression.shape[1])
    # A size-biased sample without replacement orders independent exponential clocks.
    # Approximate the stopping time by the time at which the expected sample size is reached.
    def excess(time):
        return -np.expm1(-time * weights).sum() - len(rows)

    upper = float(len(expression))
    while excess(upper) < 0:
        upper *= 2
    stopping_time = brentq(excess, 0., upper)
    inclusion = -np.expm1(-stopping_time * weights)
    target = np.zeros(expression.shape[1], dtype=np.float64)
    for start in range(0, len(expression), 2048):
        counts = molecule_counts(expression[start:start + 2048])
        target += inclusion[start:start + len(counts)] @ counts
    target /= target.sum()
    actual = observed_counts.sum(axis=0, dtype=np.float64)
    actual /= actual.sum()
    return np.clip(np.log((target + 2e-5) / (actual + 2e-5)), -.4, .4)


def stratified_counts(observed, ratio, coordinates, seed):
    """Use conditional binomial/Poisson draws, balancing detection across each marker."""
    ratio = np.broadcast_to(ratio, observed.shape)
    neighbours = cKDTree(coordinates).query(coordinates, k=min(12, len(observed)))[1]
    local = np.zeros(observed.shape, dtype=np.float32)
    for neighbour in neighbours.T:
        local += observed[neighbour]
    increase = (.5 * observed + .5 * local / neighbours.shape[1]) * np.maximum(ratio - 1, 0)
    survival = np.minimum(ratio, 1.)
    with np.errstate(divide="ignore", invalid="ignore"):
        present = np.where(observed > 0, -np.expm1(observed * np.log1p(-survival) - increase),
                           -np.expm1(-increase))
    generated = np.zeros_like(observed, dtype=np.int32)
    rng = np.random.default_rng(seed + 40277)
    for gene in range(observed.shape[1]):
        possible = np.flatnonzero(present[:, gene] > 0)
        shuffled = possible[rng.permutation(len(possible))]
        boundaries = np.floor(np.cumsum(present[shuffled, gene]) + rng.random())
        positive = np.zeros(len(observed), dtype=bool)
        positive[shuffled[np.diff(boundaries, prepend=0) > 0]] = True
        hit = np.flatnonzero(positive)
        growing = ratio[hit, gene] >= 1
        for add in (False, True):
            cells = hit[growing == add]
            if not len(cells):
                continue
            # Given detection, draw the count from the upper tail beyond the zero mass.
            quantile = (rng.permutation(len(cells)) + rng.random(len(cells))) / len(cells)
            probability = present[cells, gene]
            quantile = np.clip(1 - probability + quantile * probability,
                               np.nextafter(0., 1.), np.nextafter(1., 0.))
            original = observed[cells, gene]
            if add:
                generated[cells, gene] = original + poisson.ppf(quantile, increase[cells, gene]).astype(np.int32)
            else:
                generated[cells, gene] = binom.ppf(quantile, original, survival[cells, gene]).astype(np.int32)
    return generated
