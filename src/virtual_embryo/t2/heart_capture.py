"""Project spatial assay detection complexity within cardiac tissues, at fixed molecule total."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .heart_clean_reference import encode_tissue


def project_capture(prediction, recent, rows, seed):
    expression = prediction["x"]
    groups = encode_tissue(recent["labels"])[rows]
    after_types = encode_tissue(recent["labels"])
    after = (recent["x"] > 0).sum(1)
    detected = (expression > 0).sum(1)
    wanted = detected.copy()
    for identity in np.unique(groups):
        chosen = np.flatnonzero(groups == identity)
        late = after[after_types == identity]
        if min(len(chosen), len(late)) < 20:
            continue
        percentiles = (rankdata(detected[chosen]) - .5)/len(chosen)
        present = np.quantile(late, percentiles)
        target_count = np.maximum(1., present)
        wanted[chosen] = np.rint(.5*detected[chosen] + .5*target_count).astype(int)

    return match_detection_counts(expression, prediction["c"], wanted, seed)


def match_detection_counts(expression, coordinates, wanted, seed):
    detected = np.count_nonzero(expression, axis=1)
    molecular = np.expm1(expression.astype(float))
    totals = molecular.sum(1)
    community = np.zeros_like(expression)
    neighbors = cKDTree(coordinates).query(coordinates, k=min(24, len(expression)))[1].reshape(len(expression), -1)
    for neighbor in neighbors.T:
        community += expression[neighbor]
    community /= neighbors.shape[1]
    rng = np.random.default_rng(seed + 4679)
    for i in np.flatnonzero((wanted != detected) & (totals > 0)):
        change = wanted[i] - detected[i]
        nonzero = np.flatnonzero(molecular[i] > 0)
        if change < 0:
            priority = expression[i, nonzero] + .2*community[i, nonzero] + .001*rng.random(len(nonzero))
            molecular[i, nonzero[np.argsort(priority)[:-change]]] = 0.
        else:
            eligible = np.flatnonzero((molecular[i] == 0) & (community[i] > 0))
            if len(eligible):
                # Exponential waiting times give a weighted draw without replacement.
                priority = rng.exponential(size=len(eligible)) / np.square(community[i, eligible])
                added = eligible[np.argsort(priority)[:change]]
                molecular[i, added] = np.quantile(molecular[i, nonzero], .1)
    scales = np.divide(totals, molecular.sum(1), out=np.ones_like(totals), where=molecular.sum(1)>0)
    return np.log1p(molecular * scales[:, None]).astype(np.float32)
