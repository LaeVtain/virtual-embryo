"""Transfer a permitted single-cell developmental reference onto the existing spatial heart donors."""
import numpy as np
from scipy.special import expit, logit
from scipy.spatial import cKDTree


def _lineages(labels):
    groups = np.full(len(labels), "other", dtype="<U16")
    for label in np.unique(labels):
        name = label.lower()
        if name.endswith("-cm") or name in ("cardiomyocyte", "cardiomyocytes"):
            lineage = "myocardial"
        elif name == "endo" or "endoth" in name or name.endswith("-endo") or name in ("bec", "early-vec"):
            lineage = "endothelial"
        elif ("phm" in name or "pam" in name or
              name in ("peri", "proepi", "epi", "cp", "fhf", "shf", "jcf", "st", "dmp")):
            lineage = "stromal"
        else:
            continue
        groups[labels == label] = lineage
    return groups


def _paths(reference, lineage, stage):
    left = reference[lineage + "_left"]
    right = reference[lineage + "_right"]
    transition = reference[lineage + "_flow"]
    start, end = np.nonzero(transition > 1e-4 * transition.max())
    fraction = np.clip((stage - reference["early_age"]) / (reference["late_age"] - reference["early_age"]), 0., 1.)
    position = (1 - fraction) * left[start] + fraction * right[end]
    initial = reference[lineage + "_p0"][start]
    terminal = reference[lineage + "_p1"][end]
    propensity = expit((1 - fraction) * logit(initial) + fraction * logit(terminal))
    mass = transition[start, end]
    return position, propensity, mass / mass.sum()


def _path_weights(donors, centers, mass):
    center = mass @ centers
    scale = np.sqrt(mass @ (centers - center) ** 2 + 1)
    local = donors - np.median(donors, axis=0)
    projected = center + local * scale / np.sqrt(np.mean(local ** 2, axis=0) + 1)
    count = min(32, len(centers))
    distances, indices = cKDTree(centers).query(projected, k=count)
    if count == 1:
        return np.ones((len(donors), 1), dtype=np.float32), indices[:, None]
    bandwidth = max(np.median(distances[:, -1] ** 2) / 8, .01)
    distance_term = -(distances ** 2 - distances[:, [0]] ** 2) / (2 * bandwidth)
    weight = np.exp(np.maximum(distance_term, -35)) * mass[indices]
    weight /= weight.sum(axis=1, keepdims=True)
    return weight.astype(np.float32), indices


def _log_odds(probabilities):
    return logit(np.clip(probabilities, .001, .999))


def _regional_variance(expression, coordinates):
    axial_order = np.argsort(coordinates[:, 2], kind="stable")
    blocks = np.array_split(axial_order, min(6, max(1, len(axial_order) // 25)))
    rates = [(np.sum(expression[block] > 0, axis=0) + .5) / (len(block) + 1) for block in blocks]
    return np.var(logit(np.clip(rates, .0005, .9995)), axis=0) / len(blocks)


def _change(reference, lineage, latent, source, baseline, coordinates, ages, strength):
    first, recent, target = ages
    centers, p0, mass = _paths(reference, lineage, recent)
    _, p1, _ = _paths(reference, lineage, target)
    weights, indices = _path_weights(latent, centers, mass)
    start = np.zeros(source.shape, dtype=np.float32)
    end = np.zeros_like(start)
    second_moment = np.zeros_like(start)
    for branch, weight in zip(indices.T, weights.T):
        start += weight[:, None] * p0[branch]
        end += weight[:, None] * p1[branch]
        if strength == "evidence":
            change = _log_odds(p1[branch]) - _log_odds(p0[branch])
            second_moment += weight[:, None] * change ** 2
    expected = _log_odds(end) - _log_odds(start)
    native = ((source > 0).sum(axis=0) + .5) / (len(source) + 1)
    forecast = ((baseline > 0).sum(axis=0) + .5) / (len(source) + 1)
    represented = logit(forecast) - logit(native)
    direction = expected.mean(axis=0)
    precision = forecast * (1 - forecast)
    coefficient = np.sum(direction * represented * precision) / (np.sum(direction ** 2 * precision) + 1e-6)
    residual = expected.copy()
    residual -= np.clip(coefficient, 0, 1) * direction
    if strength != "fixed":
        duration = max(0., 1 - (recent - first) / (target - recent))
        if strength == "evidence":
            spread = np.maximum(0., second_moment - expected ** 2)
            noise = _regional_variance(source, coordinates)
            missing = np.maximum(0., np.abs(expected) - np.maximum(0., np.sign(expected) * represented))
            confidence = missing ** 2 / (missing ** 2 + spread + noise + 1e-7)
            residual *= duration * confidence
        else:
            residual *= duration
    return np.clip(residual, -1.5, 1.5)


def _local_detection(positive, coordinates, depth):
    center = np.median(coordinates, axis=0)
    scale = np.maximum(np.std(coordinates, axis=0), .1)
    features = np.column_stack([(coordinates - center) / scale, 2 * np.log1p(depth)])
    indices = cKDTree(features).query(features, k=min(24, len(features)))[1]
    total = np.zeros_like(positive, dtype=np.float32)
    for neighbors in indices.T:
        total += positive[neighbors]
    return (total + 12 * positive.mean(axis=0) + .5) / (indices.shape[1] + 13)


def _update_detection(expression, local, shift, rng):
    positive = expression > 0
    target = expit(logit(local) + shift)
    gain = np.maximum(0, target - local) / np.maximum(1e-7, 1 - local)
    loss = np.maximum(0, local - target) / np.maximum(1e-7, local)
    output = expression.copy()
    for gene in range(expression.shape[1]):
        observed = np.flatnonzero(positive[:, gene])
        if not len(observed):
            continue
        for eligible, probability, add in ((~positive[:, gene], gain[:, gene], True),
                                            (positive[:, gene], loss[:, gene], False)):
            pool = np.flatnonzero(eligible & (probability > 1e-7))
            if not len(pool):
                continue
            shuffled = rng.permutation(pool)
            quota = np.floor(np.cumsum(probability[shuffled], dtype=float) + rng.random())
            selected = shuffled[np.diff(quota, prepend=0) > 0]
            if add:
                output[selected, gene] = rng.choice(expression[observed, gene], size=len(selected), replace=True)
            else:
                output[selected, gene] = 0
    return output


def transfer(prediction, observed, labels, depth, ages, seed, reference, strength="evidence"):
    """Update the predicted expression of exactly the donors selected by the spatial model."""
    reference_ages = (float(reference["early_age"]), float(reference["late_age"]))
    if any(8.25 < age < 8.75 or 9.5 < age <= 13.5 for age in reference_ages):
        raise ValueError("The external reference contains a protected T2 heart stage")
    first, recent, target = ages
    if strength == "animal-horizon":
        if target - recent < float(reference["minimum_supported_days"]):
            return prediction
        strength = "fixed"
    if strength not in ("fixed", "duration", "evidence"):
        raise ValueError("Unknown state-bridge strength")
    if strength != "fixed" and target - recent <= recent - first:
        return prediction
    genes = prediction["genes"].astype(str).tolist()
    expected = reference["genes"].astype(str).tolist()
    if len(genes) != len(expected) or set(genes) != set(expected):
        raise ValueError("The heart state reference requires the same genes as the prediction panel")
    observed_features = np.asarray(reference["present"], dtype=bool)
    canonical = genes == expected
    indices = [genes.index(gene) for gene in expected]
    original = np.ascontiguousarray(observed if canonical else observed[:, indices])
    baseline = np.ascontiguousarray(prediction["x"] if canonical else prediction["x"][:, indices])
    result = baseline.copy()
    groups = _lineages(np.asarray(labels).astype(str))
    rng = np.random.default_rng(seed + 91177)
    linear = np.expm1(original[:, observed_features]).astype(np.float32, copy=False)
    fractions = linear / np.maximum(linear.sum(axis=1, keepdims=True), 1e-9)
    latent = (np.log1p(1000 * fractions) - reference["center"]) @ reference["axes"].T
    for lineage in ("myocardial", "endothelial", "stromal"):
        rows = np.flatnonzero(groups == lineage)
        if len(rows) < 25:
            continue
        source = original[rows][:, observed_features]
        forecast = baseline[rows][:, observed_features]
        coordinates = prediction["c"][rows]
        shift = _change(reference, lineage, latent[rows], source, forecast, coordinates, ages, strength)
        local = _local_detection(forecast > 0, coordinates, depth[rows])
        result[np.ix_(rows, observed_features)] = _update_detection(forecast, local, shift, rng)
    linear = np.expm1(result)
    wanted = np.expm1(baseline).sum(axis=1)
    total = linear.sum(axis=1)
    empty = total <= 0
    linear[empty] = np.expm1(baseline[empty])
    total[empty] = wanted[empty]
    normalized = np.log1p(linear * (wanted / np.maximum(total, 1))[:, None])
    if not canonical:
        normalized = normalized[:, np.argsort(indices)]
    return {**prediction, "x": np.ascontiguousarray(normalized, dtype=np.float32)}
