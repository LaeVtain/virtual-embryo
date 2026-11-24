"""Refine a mutant forecast with normal developmental and lineage expression summaries."""
import numpy as np

from .response_ranks import _joint_ranks


def _merge_marginals(wild_type, baseline, reference_quantiles, seed):
    cells, genes = wild_type.shape
    quantile_count = len(reference_quantiles)
    probabilities = np.concatenate((np.full(cells, .3 / cells),
                                    np.full(quantile_count, .7 / quantile_count)))
    grid = (np.arange(cells) + .5) / cells
    scores = .5 * (_joint_ranks(wild_type, seed + 983) + _joint_ranks(baseline, seed + 983))
    order = np.argsort(scores, axis=0, kind="stable")
    output = np.empty_like(baseline)
    for column in range(genes):
        values = np.concatenate((baseline[:, column], reference_quantiles[:, column]))
        indices = np.argsort(values, kind="stable")
        cdf = np.cumsum(probabilities[indices])
        cdf[-1] = 1.
        output[order[:, column], column] = values[indices[np.searchsorted(cdf, grid)]]
    return output


def _fit_gene_means(expression, baseline, desired):
    mass = np.expm1(expression.astype(np.float64))
    baseline_mass = np.expm1(baseline.astype(np.float64))
    totals = baseline_mass.sum(axis=1)
    empty = (mass.sum(axis=1) == 0) & (totals > 0)
    mass[empty] = baseline_mass[empty]
    occupied = mass > 0

    for _ in range(24):
        sizes = mass.sum(axis=1)
        mass *= np.divide(totals, sizes, out=np.ones_like(totals), where=sizes > 0)[:, None]
        error = desired - np.log1p(mass).mean(axis=0)
        if np.max(np.abs(error)) < 2e-5:
            break
        # Changing the ratio of high to low counts within a cell controls its
        # total log expression while preserving its original molecule total.
        slope = mass / (1. + mass)
        log_mass = np.zeros_like(mass)
        np.log(mass, out=log_mass, where=occupied)
        center = (log_mass * mass).sum(axis=1) / np.maximum(totals, 1.)
        centered = log_mass - center[:, None]
        shape_slope = slope * centered
        shape_derivative = min(float(shape_slope.sum() / len(mass)), -1.)
        shape_gain = np.clip(error.sum() / shape_derivative, -.25, .25)
        gene_gain = (error - shape_gain * shape_slope.mean(axis=0)) / np.maximum(slope.mean(axis=0), .003)
        mass *= np.exp(np.clip(shape_gain * centered + np.clip(gene_gain, -.7, .7), -2., 2.))
    sizes = mass.sum(axis=1)
    mass *= np.divide(totals, sizes, out=np.ones_like(totals), where=sizes > 0)[:, None]
    return np.log1p(mass).astype(np.float32)


def refine_response(wild_type, baseline, genes, model, seed=101):
    """Refine a v3 expression matrix in its input gene order using a fitted NPZ asset."""
    wild_type = np.asarray(wild_type, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if wild_type.ndim != 2 or wild_type.shape != baseline.shape:
        raise ValueError("Wild-type cells and v3 predictions must have the same matrix shape")
    gene_names = np.asarray(genes).astype(str)
    if len(gene_names) != wild_type.shape[1] or not np.array_equal(gene_names, model["genes"]):
        raise ValueError("Developmental response asset and input panel must have the same gene order")
    if len(wild_type) < 2:
        return baseline.copy()

    center = baseline.mean(axis=0, dtype=np.float64)
    source = wild_type.mean(axis=0, dtype=np.float64)
    effect_size = np.sqrt(np.mean(np.square(center - source)))
    gate = np.clip(4. * float(model["lineage_reliability"]), 0., 1.)
    developmental_mean = center + .65 * (model["developmental_mean"] - center)
    lineage_mean = (center + .35 * (model["depth_matched_mean"] - center)
                    + .35 * effect_size * model["lineage_response"])
    desired = (1. - gate) * developmental_mean + gate * lineage_mean
    marginals = _merge_marginals(wild_type, baseline, model["reference_quantiles"], seed)
    return _fit_gene_means(marginals, baseline, np.maximum(0., desired))
