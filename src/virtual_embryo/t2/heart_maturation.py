"""Forecast cardiac anatomy and spatial detection using an independent nuclear RNA forecast."""
import numpy as np
from scipy.special import expit, logit
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from .heart_anatomy import aligned_atlas, rms_radius
from .local_expression import place_local_profiles


def nuclear_reference(genes, before, after, observed, source_time, target_time):
    """Summarize an RNA forecast in the ordered spatial panel, using only source and predicted cells."""
    genes = np.asarray(genes, dtype=str)
    observed = np.asarray(observed, dtype=bool)
    if before.shape[1] != len(genes) or after.shape[1] != len(genes) or observed.shape != genes.shape:
        raise ValueError("RNA matrices, observation mask and spatial gene panel must agree")
    initial = np.log(before.mean(axis=0, dtype=np.float64) + .012)
    terminal = np.log(after.mean(axis=0, dtype=np.float64) + .012)
    change = np.clip(terminal - initial, -2., 2.)
    change[~observed] = 0.
    return dict(genes=genes, change=change, source_time=source_time, target_time=target_time)


def _quantiles(values, probabilities):
    sorted_values = np.sort(values, axis=0)
    sample = np.linspace(0., 1., len(values))
    return np.stack([np.interp(probabilities, sample, column) for column in sorted_values.T], axis=1)


def developmental_fraction(recent_time, target_time):
    return np.log((target_time - 7.5) / (recent_time - 7.5)) / np.log((14.5 - 7.5) / (recent_time - 7.5))


def surrounding_tissues(labels):
    names = np.char.lower(np.asarray(labels, dtype=str))
    surrounding = ("caudal epiblast", "neural tube", "som", "allantois", "v-cse", "d-cse", "pse",
                   "hindgut", "forebrain", "exe-endoderm", "gut endoderm")
    return np.isin(names, surrounding) | np.char.startswith(names, "exem")


def dissection_weights(weights, labels, cells, reference):
    """Downweight surrounding tissue in source collections that contain a substantial whole-embryo field."""
    outside = surrounding_tissues(labels)
    threshold = float(reference["dissected_max_surrounding"])
    retention = float(reference["surrounding_retention"])
    if not (0 <= threshold <= 1 and 0 < retention <= 1):
        raise ValueError("The dissection threshold and retention must be valid proportions")
    if outside.mean() < threshold or (~outside).sum() < cells:
        return weights
    probabilities = weights.copy()
    probabilities[outside] *= retention
    return probabilities / probabilities.sum()


def interpolate_shape(coordinates, recent_coordinates, atlas, recent_time, target_time, endpoint_shift=0.):
    """Interpolate observed and late-heart quantiles on relative developmental time in 15 frames."""
    if not 7.5 < recent_time < target_time <= 14.5:
        raise ValueError("Cardiac anatomical interpolation requires E7.5 < source < target <= E14.5")
    baseline = np.asarray(coordinates, dtype=np.float64)
    initial, terminal, axes = aligned_atlas(baseline, atlas)
    _, latest, _ = aligned_atlas(baseline, recent_coordinates)
    rank = (np.arange(len(initial)) + .5) / len(initial)
    fraction = developmental_fraction(recent_time, target_time)
    fraction += (1 - fraction) * endpoint_shift
    rng = np.random.default_rng(7194)
    transported = initial.copy()
    for step in range(15):
        frame = np.eye(3) if step == 0 else np.linalg.qr(rng.normal(size=(3, 3)))[0]
        current = transported @ frame
        order = np.argsort(current, axis=0, kind="stable")
        wanted = (1 - fraction) * _quantiles(latest @ frame, rank) + fraction * _quantiles(terminal @ frame, rank)
        delta = np.zeros_like(current)
        np.put_along_axis(delta, order, wanted - np.take_along_axis(current, order, axis=0), axis=0)
        transported += .65 * delta @ frame.T
    transported = transported @ axes.T
    transported -= transported.mean(axis=0)
    scale = rms_radius(baseline) / rms_radius(transported)
    return (transported * scale + baseline.mean(axis=0)).astype(np.float32)


def _rank_cells(expression, coordinates, seed):
    """Break equal expression ranks by local molecular state, location, then a reproducible draw."""
    components = min(12, len(expression) - 1, expression.shape[1])
    state = PCA(components, svd_solver="randomized", random_state=1277).fit_transform(expression)
    neighbors = min(24, len(expression))
    local_expression = cKDTree(state).query(state, k=neighbors)[1]
    local_space = cKDTree(coordinates).query(coordinates, k=neighbors)[1]
    neighborhood_sum = np.zeros_like(expression)
    for group in (local_expression, local_space):
        for rows in group.T:
            neighborhood_sum += expression[rows]
    rng = np.random.default_rng(seed + 917)
    return np.lexsort((rng.random(expression.shape, dtype=np.float32), neighborhood_sum, expression), axis=0)


def _detection_rates(log_odds, expected_count):
    low, high = -10., 10.
    for _ in range(30):
        intercept = (low + high) / 2
        if expit(log_odds + intercept).sum() < expected_count:
            low = intercept
        else:
            high = intercept
    return expit(log_odds + (low + high) / 2)


def _redistribute_expression(expression, order, probabilities):
    cells, genes = expression.shape
    counts = np.clip(np.rint(probabilities * cells).astype(int), 0, cells)
    raw = np.zeros_like(expression)
    for gene in range(genes):
        count = counts[gene]
        if count == 0:
            continue
        observed = expression[order[:, gene], gene]
        positive = observed[observed > 0]
        if not len(positive):
            continue
        positions = (np.arange(count) + .5) * len(positive) / count - .5
        values = np.interp(positions, np.arange(len(positive)), positive)
        raw[order[-count:, gene], gene] = np.expm1(values)
    empty = raw.sum(axis=1) == 0
    raw[empty] = np.expm1(expression[empty])
    library = np.expm1(np.asarray(expression, dtype=np.float64)).sum(axis=1)
    return np.log1p(raw * (library / np.maximum(raw.sum(axis=1), 1e-9))[:, None]).astype(np.float32)


def mature_heart(prediction, recent_coordinates, atlas, recent_time, target_time, reference, seed,
                 *, source_labels=None, atlas_radius=None):
    """Return a complete heart forecast from a source-derived spatial prediction and nuclear trajectory."""
    if float(reference["source_time"]) > recent_time or float(reference["target_time"]) != target_time:
        raise ValueError("Nuclear reference dates must match the observed and forecast spatial ages")
    genes = np.asarray(prediction["genes"], dtype=str)
    ref_genes = np.asarray(reference["genes"], dtype=str)
    index = {gene: idx for idx, gene in enumerate(ref_genes)}
    if len(index) != len(genes) or set(index) != set(genes):
        raise ValueError("Nuclear reference and spatial gene panel must match")
    physical = bool(reference.get("physical_geometry", False))
    endpoint_shift = 0.
    change_key = "change"
    if physical:
        if source_labels is None or np.shape(source_labels) != (len(recent_coordinates),):
            raise ValueError("Physical cardiac growth requires one tissue label per observed source cell")
        if atlas_radius is None or not np.isfinite(atlas_radius) or atlas_radius <= 0:
            raise ValueError("Physical cardiac growth requires a positive late-heart radius")
        surrounding = surrounding_tissues(source_labels)
        endpoint_shift = 1 - np.exp(-2 * surrounding.mean())
        if "dissected_change" in reference and surrounding.mean() < float(reference["dissected_max_surrounding"]):
            change_key = "dissected_change"
    change = np.asarray(reference[change_key], dtype=float)
    if change.shape != ref_genes.shape or not np.isfinite(change).all():
        raise ValueError("Expected one finite RNA change per reference gene")
    change = change[[index[gene] for gene in genes]]
    coords = interpolate_shape(prediction["c"], recent_coordinates, atlas, recent_time, target_time, endpoint_shift)
    expression = place_local_profiles(prediction["x"], coords)
    order = _rank_cells(expression, coords, seed)
    baseline = (np.count_nonzero(expression, axis=0) + .5) / (len(expression) + 1.)
    represented = _detection_rates(logit(baseline) + change, baseline.sum())
    strength = .35 * (target_time - recent_time)
    target = _detection_rates((1 - strength) * logit(baseline) + strength * logit(represented), baseline.sum())
    if physical:
        tissue = recent_coordinates[~surrounding]
        source_radius = rms_radius(tissue) if len(tissue) >= 3 else rms_radius(recent_coordinates)
        fraction = developmental_fraction(recent_time, target_time)
        radius = (1 - fraction) * source_radius + fraction * atlas_radius
        center = coords.mean(axis=0, dtype=float)
        scale = radius / rms_radius(coords.astype(float))
        coords = ((coords - center) * scale + center).astype(np.float32)
    return dict(prediction, c=coords, x=_redistribute_expression(expression, order, target))
