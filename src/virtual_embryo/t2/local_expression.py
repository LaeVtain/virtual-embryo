"""Match complete expression profiles to their local tissue neighborhoods."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def place_local_profiles(expression, coordinates):
    cells = len(expression)
    if cells < 2:
        return expression.copy()
    model = PCA(min(12, cells - 1, expression.shape[1]), svd_solver="randomized", random_state=7501)
    model.fit(expression)
    scores = model.transform(expression) / np.sqrt(np.maximum(model.explained_variance_, .1))
    neighbors = cKDTree(coordinates).query(coordinates, k=min(24, cells))[1]
    target = scores[neighbors].mean(axis=1)
    axes = (coordinates - coordinates.mean(0)) / np.maximum(coordinates.std(0), 1.)
    assigned = np.arange(cells)
    pending = [assigned.copy()]
    while pending:
        rows = pending.pop()
        geometry = axes[rows]
        if len(rows) > 48:
            axis = np.argmax(geometry.var(0))
            ordered = rows[np.argsort(geometry[:, axis], kind="stable")]
            pending.extend(np.array_split(ordered, 2))
            continue
        molecular = cdist(target[rows], scores[rows], "sqeuclidean")
        distance = cdist(geometry, geometry, "sqeuclidean")
        tissue_variance = np.mean(np.var(target[rows], axis=0)) * scores.shape[1]
        penalty = tissue_variance / max(.001, np.mean(distance))
        positions, donors = linear_sum_assignment(molecular + penalty * distance)
        assigned[rows[positions]] = rows[donors]
    return expression[assigned]
