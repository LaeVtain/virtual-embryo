"""Apply a measured within-chamber developmental change to a T1 cell population."""
import numpy as np
from scipy import sparse

from .lineage_response import marker_classes, mean


def match_means(matrix, desired):
    result = matrix.astype(np.float64, copy=True)
    mass = np.expm1(result.data)
    n_cells, n_genes = result.shape
    row_lengths = np.diff(result.indptr)
    for _ in range(5):
        observed = np.bincount(result.indices, weights=np.log1p(mass), minlength=n_genes)
        derivative = np.bincount(result.indices, weights=mass / (1 + mass), minlength=n_genes)
        update = np.clip((n_cells * desired - observed) / (derivative + 4), -0.45, 0.45)
        mass *= np.exp(update)[result.indices]
        mass *= np.repeat(10000 / np.add.reduceat(mass, result.indptr[:-1]), row_lengths)
    result.data = np.log1p(mass)
    return result.astype(np.float32)


def redistribute(cells, predicted_types, source_types, composition):
    labels = sorted(set(predicted_types))
    current = np.array([(predicted_types == label).mean() for label in labels])
    previous = np.array([(source_types == label).mean() for label in labels])
    expected = previous * np.exp([composition.get(label, 0) for label in labels])
    expected /= expected.sum()
    counts = np.rint(len(predicted_types) * (0.65 * current + 0.35 * expected)).astype(int)
    counts[np.argmax(counts)] += len(predicted_types) - counts.sum()
    # A fixed second draw preserves the reference experiment across upstream donor seeds.
    rng = np.random.default_rng(8822)
    selected = []
    for label, count in zip(labels, counts):
        pool = np.flatnonzero(predicted_types == label)
        if count <= len(pool):
            rows = rng.choice(pool, count, replace=False)
        else:
            rows = np.concatenate([pool, rng.choice(pool, count - len(pool), replace=True)])
        selected.extend(rows)
    return cells[rng.permutation(selected)].tocsr()


def project_chambers(source, baseline, panel, reference, source_time, target_time):
    horizon = target_time - source_time
    with np.load(reference, allow_pickle=False) as stored:
        early, late = stored["times"]
        if not np.array_equal(stored["genes"], np.asarray(panel)):
            raise ValueError("The chamber reference must use the requested gene panel and its order.")
        if not 0 <= source_time - early <= horizon or target_time in (early, late):
            return baseline
        labels = stored["lineages"]
        fraction = horizon / (late - early)
        changes = np.clip(stored["log_change"] * fraction, -1.2, 1.2)
        trends = {key:(effect, weight) for key, effect, weight in zip(labels, changes, stored["reliability"])}
        composition = dict(zip(stored["composition_lineages"],
                               np.clip(stored["composition_log_change"] * fraction, -0.6, 0.6)))
    source_types = marker_classes(source, panel)
    predicted_types = marker_classes(baseline, panel)
    parts, order = [], []
    for label in sorted(set(predicted_types)):
        rows = np.flatnonzero(predicted_types == label)
        initial = np.flatnonzero(source_types == label)
        current = baseline[rows]
        if label in trends and min(len(rows), len(initial)) >= 15:
            previous = mean(source[initial])
            change, reliability = trends[label]
            expected = np.maximum(0, previous + (previous + 0.05) * np.expm1(change))
            observed = mean(current)
            desired = observed + 0.65 * reliability * (expected - observed)
            current = match_means(current, desired)
        parts.append(current)
        order.extend(rows)
    cells = sparse.vstack(parts, format="csr")[np.argsort(order)]
    return redistribute(cells, predicted_types, source_types, composition)
