"""Estimate tissue abundances and use matched donors to realize them without altering measured RNA."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog
from sklearn.cluster import AgglomerativeClustering

from .temporal import _align_population


def tissue_groups(stages, earlier, later):
    labels = {age: np.asarray(pop["labels"]).astype(str) for age, pop in stages.items()}
    if set(labels[earlier]) == set(labels[later]):
        names = sorted(set().union(*(set(names) for names in labels.values())))
        return {age: np.searchsorted(names, stage_labels) for age, stage_labels in labels.items()}

    # Adjacent stages can use different names for related tissue. Compare each annotated region's
    # expression against the mean of its own specimen before clustering its marker directions.
    regions = []
    profiles = []
    panel = list(stages[earlier]["genes"])
    for age, pop in sorted(stages.items()):
        expression, _ = _align_population(pop, panel)
        mean = expression.mean(axis=0, dtype=float)
        for label in np.unique(labels[age]):
            tissue_mean = expression[labels[age] == label].mean(axis=0, dtype=float)
            contrast = np.log((tissue_mean + .07) / (mean + .07))
            regions.append((age, label))
            profiles.append(contrast / max(.01, np.linalg.norm(contrast)))
    profiles = np.stack(profiles)
    supported = np.linalg.norm(profiles, axis=1) > 1e-10
    groups = np.zeros(len(regions), dtype=int)
    if supported.sum() >= 2:
        cluster = AgglomerativeClustering(n_clusters=min(8, int(supported.sum())), linkage="average", metric="cosine")
        groups[supported] = cluster.fit_predict(profiles[supported])
    if supported.any():
        groups[~supported] = groups[supported].max() + 1
    region_index = dict(zip(regions, groups))
    return {age: np.array([region_index[(age, name)] for name in names]) for age, names in labels.items()}


def tissue_distribution(groups, target):
    dates = sorted(groups)
    number = max(ids.max() for ids in groups.values()) + 1
    fractions = np.stack([np.bincount(groups[age], minlength=number) / len(groups[age]) for age in dates])
    curve = PchipInterpolator(dates, np.sqrt(fractions), axis=0)
    predicted = np.maximum(curve(target), 0) ** 2
    return predicted / predicted.sum()


def assign_donors(left, right, initial, desired):
    """Match tissue counts, keeping the clock's total number of later-stage cells fixed."""
    pairs, inverse = np.unique(np.stack([left, right], axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    pair_count, tissue_count = len(pairs), len(desired)
    capacity = np.bincount(inverse, minlength=pair_count)
    original = np.bincount(inverse[initial], minlength=pair_count)
    change = np.zeros((tissue_count, pair_count))
    change[pairs[:, 1], np.arange(pair_count)] += 1
    change[pairs[:, 0], np.arange(pair_count)] -= 1
    all_left = np.bincount(left, minlength=tissue_count)
    target = len(initial) * desired

    # Each extra right-hand donor removes one left tissue and adds one right tissue. Minimize
    # absolute abundance error with a secondary cost for reassigning an otherwise viable pair.
    residual_end = pair_count + tissue_count
    constraints = np.zeros((2*tissue_count + 2*pair_count, residual_end + pair_count))
    constraints[:tissue_count, :pair_count] = change
    constraints[tissue_count:2*tissue_count, :pair_count] = -change
    constraints[:tissue_count, pair_count:residual_end] = -np.eye(tissue_count)
    constraints[tissue_count:2*tissue_count, pair_count:residual_end] = -np.eye(tissue_count)
    bound = 2*tissue_count + pair_count
    constraints[2*tissue_count:bound, :pair_count] = np.eye(pair_count)
    constraints[bound:, :pair_count] = -np.eye(pair_count)
    constraints[2*tissue_count:bound, residual_end:] = -np.eye(pair_count)
    constraints[bound:, residual_end:] = -np.eye(pair_count)
    rhs = np.r_[target-all_left, all_left-target, original, -original]
    costs = np.r_[np.zeros(pair_count), 1/np.sqrt(np.maximum(desired, .01)), np.full(pair_count, .1)]
    total = np.r_[np.ones(pair_count), np.zeros(tissue_count+pair_count)][None, :]
    bounds = [(0, int(n)) for n in capacity] + [(0, None)]*(pair_count+tissue_count)
    solution = linprog(costs, A_ub=constraints, b_ub=rhs, A_eq=total,
                       b_eq=[initial.sum()], bounds=bounds, method="highs")
    if not solution.success:
        raise ValueError(f"Could not assign matched tissue donors: {solution.message}")
    ideal = solution.x[:pair_count]
    counts = np.floor(ideal+1e-8).astype(int)
    missing = int(initial.sum() - counts.sum())
    available = np.flatnonzero(counts < capacity)
    priority = available[np.argsort((ideal-counts)[available])[::-1]]
    counts[priority[:missing]] += 1
    selected = initial.copy()
    rng = np.random.default_rng(58411)
    for group, delta in enumerate(counts-original):
        if delta:
            candidates = np.flatnonzero((inverse == group) & (initial == (delta < 0)))
            selected[rng.choice(candidates, abs(delta), replace=False)] = delta > 0
    return selected
