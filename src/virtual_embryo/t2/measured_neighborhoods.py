"""Place whole expression profiles using the neighborhoods of their measured source cells."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def measured_neighborhoods(expression, coordinates, parents, donor_ids, from_later, source_weight):
    cells = len(expression)
    if cells < 4:
        return expression.copy()
    model = PCA(min(12, cells-1, expression.shape[1]), svd_solver="randomized", random_state=7501)
    model.fit(expression)
    scale = np.sqrt(np.maximum(model.explained_variance_, .1))
    scores = model.transform(expression) / scale
    nearest = cKDTree(coordinates).query(coordinates, k=min(32, cells))[1]
    neighborhood = scores[nearest].mean(axis=1)
    target = neighborhood.copy()
    for side, parent in enumerate(parents):
        rows = np.flatnonzero(from_later == side)
        if len(rows) < 4:
            continue
        native = model.transform(np.ascontiguousarray(parent["x"])) / scale
        locations = parent["c"][donor_ids[rows]]
        matches = cKDTree(parent["c"]).query(locations, k=min(32, len(native)))[1]
        observed = native[matches].mean(axis=1)
        local = neighborhood[rows]
        # Transfer relative spatial variation; baseline cells supply the target age's means.
        normalized = (observed - observed.mean(0)) / np.maximum(observed.std(0), .1)
        projected = local.mean(0) + normalized * local.std(0)
        target[rows] = (1-source_weight)*local + source_weight*projected
    geometry = (coordinates-coordinates.mean(0)) / np.maximum(coordinates.std(0), 1.)
    assigned = np.arange(cells)
    pending = [assigned.copy()]
    while pending:
        rows = pending.pop()
        sites = geometry[rows]
        if len(rows) > 90:
            axis = np.argmax(sites.var(0))
            pending.extend(np.array_split(rows[np.argsort(sites[:,axis], kind="stable")], 2))
            continue
        distance = cdist(sites, sites, "sqeuclidean")
        penalty = target[rows].var(0).sum() / max(distance.mean(), .001)
        position, donor = linear_sum_assignment(cdist(target[rows], scores[rows], "sqeuclidean") + penalty*distance)
        assigned[rows[position]] = rows[donor]
    return expression[assigned]
