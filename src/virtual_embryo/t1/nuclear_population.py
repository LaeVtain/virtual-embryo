"""Calibrate a cardiac cell forecast from measured, depth-matched nuclear populations."""
import numpy as np
from scipy import sparse
from scipy.special import expit, logit

from .lineage_response import mean
from .reference_marginals import exchange_marginals


def _depth_weights(measurement, forecast_depth):
    boundaries = np.unique(np.quantile(forecast_depth, np.linspace(0, 1, 9)[1:-1]))
    observed_group = np.searchsorted(boundaries, np.diff(measurement.indptr))
    predicted_group = np.searchsorted(boundaries, forecast_depth)
    counts = np.bincount(observed_group, minlength=len(boundaries) + 1)
    weights = np.bincount(predicted_group, minlength=len(counts)) / len(forecast_depth)
    observed = np.flatnonzero(counts)
    for missing in np.flatnonzero(counts == 0):
        closest = observed[np.argmin(np.abs(observed - missing))]
        weights[closest] += weights[missing]
    return weights[observed_group] / counts[observed_group]


def _depth_sample(measurement, requested_depth, rng):
    depth = np.diff(measurement.indptr)
    order = np.argsort(depth, kind="stable")
    closest = np.searchsorted(depth[order], requested_depth)
    radius = max(10, round(len(order) / 100))
    lower = np.maximum(0, closest - radius)
    upper = np.minimum(len(order), closest + radius + 1)
    return measurement[order[rng.integers(lower, upper)]]


def _draw_at_time(source, late, forecast_depth, fraction):
    seed = 74891
    requested_depth = np.random.default_rng(seed).permutation(forecast_depth)
    rng = np.random.default_rng(seed)
    use_late = rng.random(len(requested_depth)) < fraction
    earlier = _depth_sample(source, requested_depth, rng)
    later = _depth_sample(late, requested_depth, rng)
    stacked = sparse.vstack([earlier[~use_late], later[use_late]], format="csr")
    order = np.r_[np.flatnonzero(~use_late), np.flatnonzero(use_late)]
    return stacked[np.argsort(order)]


def restore_empirical_cells(prediction, source, late_nuclei, fraction, seed, measured_fraction=.5):
    """Preserve intact, measured cell profiles in part of the population at forecast gene means."""
    if not 0 <= fraction <= 1 or source.shape[1] != prediction.shape[1] or late_nuclei.shape[1] != prediction.shape[1]:
        raise ValueError("Measured cells and prediction require a shared gene panel and a fraction between 0 and 1")
    if not 0 < measured_fraction <= 1:
        raise ValueError("The measured fraction must be positive and at most one")
    desired = np.asarray(prediction.mean(axis=0, dtype=np.float64)).ravel()
    native = _draw_at_time(source, late_nuclei, np.diff(prediction.indptr), fraction)
    permutation = np.random.default_rng(seed + 631).permutation(prediction.shape[0])
    keep = len(permutation) - int(measured_fraction * len(permutation))
    retained, measured = permutation[:keep], permutation[keep:]
    template = sparse.vstack((prediction[retained], native[measured]), format="csr")
    restored = template[np.argsort(np.r_[retained, measured])]
    return _balance(restored, desired, 6)


def _balance(template, desired, iterations):
    """Fit gene log means while preserving a normalized molecular mass per cell."""
    result = template.copy()
    cells, genes = result.shape
    lengths = np.diff(result.indptr)
    mass = np.empty(len(result.data), dtype=np.float64)
    desired = np.maximum(desired, 0)
    wanted_sum = desired.sum()
    # CSR row blocks contain all genes in a cell. This permits normalization
    # without several dense buffers the size of a 4,096-cell prediction.
    blocks = []
    for first in range(0, cells, 64):
        end = min(first + 64, cells)
        values = slice(result.indptr[first], result.indptr[end])
        rows = slice(first, end)
        starts = result.indptr[rows] - result.indptr[first]
        blocks.append((values, rows, starts))
        mass[values] = np.expm1(result.data[values].astype(np.float64))
        row_totals = np.add.reduceat(mass[values], starts)
        mass[values] *= np.repeat(10000 / row_totals, lengths[rows])
    for _ in range(iterations):
        observed = np.zeros(genes)
        gradient = np.zeros(genes)
        derivative = 0.
        # Uniform rescaling of all molecules disappears when the cell is
        # normalized. Adjusting concentration also fits its total log expression.
        for values, rows, starts in blocks:
            current = mass[values]
            genes_in_block = result.indices[values]
            observed += np.bincount(genes_in_block, np.log1p(current), minlength=genes)
            response = current / (1 + current)
            gradient += np.bincount(genes_in_block, response, minlength=genes)
            log_mass = np.log(np.maximum(current, 1e-12))
            centroid = np.add.reduceat(current * log_mass, starts) / 10000
            derivative += np.sum(response * (log_mass - np.repeat(centroid, lengths[rows])))
        observed /= cells
        gradient /= cells
        derivative /= cells
        total_error = wanted_sum - observed.sum()
        power = np.clip(total_error / min(-1., derivative), -.12, .12)
        residual = desired - observed - total_error * gradient / max(1., gradient.sum())
        log_gain = np.clip(residual / (gradient + 2 / cells), -.5, .5)
        for values, rows, starts in blocks:
            current = mass[values]
            log_mass = np.log(np.maximum(current, 1e-12))
            current *= np.exp(log_gain[result.indices[values]] + power * log_mass)
            row_totals = np.add.reduceat(current, starts)
            current *= np.repeat(10000 / row_totals, lengths[rows])
    for values, _, _ in blocks:
        result.data[values] = np.log1p(mass[values])
    return result.astype(np.float32, copy=False)


def _moments(matrix, weights, include_detection):
    """Accumulate weighted moments directly into panel-length arrays in CSR order."""
    expression = np.zeros(matrix.shape[1])
    detected = np.zeros(matrix.shape[1]) if include_detection else None
    unweighted = np.zeros(matrix.shape[1])
    for first in range(0, matrix.shape[0], 128):
        end = min(first + 128, matrix.shape[0])
        begin, stop = matrix.indptr[first], matrix.indptr[end]
        indices = matrix.indices[begin:stop]
        values = matrix.data[begin:stop]
        expanded = np.repeat(weights[first:end], np.diff(matrix.indptr[first:end + 1]))
        np.add.at(expression, indices, values * expanded)
        np.add.at(unweighted, indices, values)
        if include_detection:
            np.add.at(detected, indices, expanded)
    return expression, detected, unweighted / matrix.shape[0]


def project_population(prediction, source, late_nuclei, source_time, target_time, method="rank"):
    """Refine a nuclear-trajectory forecast using all eligible cells at its source age.

    Parameters are CSR, log(1 + counts normalized to 10,000), with the same
    genes in the same order. The later measured nuclear population is E14.5.
    The source may contain a different number of cells than the forecast.
    """
    if not source_time < target_time < 14.5:
        raise ValueError("The source and E14.5 reference must bracket the target age.")
    if source.shape[1] != prediction.shape[1] or late_nuclei.shape[1] != prediction.shape[1]:
        raise ValueError("The measured nuclei and forecast need the same gene panel.")
    if method not in {"rank", "odds"}:
        raise ValueError("Choose a nuclear population method of rank or odds.")
    fraction = (target_time - source_time) / (14.5 - source_time)
    depth = np.diff(prediction.indptr)
    source_weight = _depth_weights(source, depth)
    late_weight = _depth_weights(late_nuclei, depth)
    before, source_detection, source_mean = _moments(source, source_weight, method == "odds")
    after, late_detection, _ = _moments(late_nuclei, late_weight, method == "odds")
    baseline = mean(prediction)
    if method == "rank":
        geometric = np.exp((1 - fraction) * np.log(before + .015) +
                           fraction * np.log(after + .015)) - .015
        forecast = .3 * baseline + .7 * geometric
        difference = baseline - source_mean
        reordered = np.zeros_like(difference)
        groups = np.array_split(np.argsort(source_mean, kind="stable"), 24)
        for indices in groups:
            ranked = np.argsort((forecast - source_mean)[indices], kind="stable")
            reordered[indices[ranked]] = np.sort(difference[indices])
        desired = source_mean + reordered
        iterations = 6
    else:
        minimum = .25 / prediction.shape[0]
        source_logit = logit(np.clip(source_detection, minimum, 1 - minimum))
        late_logit = logit(np.clip(late_detection, minimum, 1 - minimum))
        detection = expit((1 - fraction) * source_logit + fraction * late_logit)
        early_value = np.divide(before, source_detection, out=np.zeros_like(before), where=source_detection > 0)
        late_value = np.divide(after, late_detection, out=np.zeros_like(after), where=late_detection > 0)
        positive = np.where(source_detection == 0, late_value, np.where(
            late_detection == 0, early_value, (1 - fraction) * early_value + fraction * late_value))
        desired = .3 * baseline + .7 * detection * positive
        iterations = 8
    measured = _draw_at_time(source, late_nuclei, depth, fraction)
    ranked_population = exchange_marginals(prediction, measured)
    return _balance(ranked_population, desired, iterations)
