"""Map an empirical KO reference into a matched WT population's spatial frame."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.decomposition import PCA


def predict_mixture(x, c, knockout, seed, fraction=0.5, signature=None, sensitivity=0.0):
    """Replace a fraction of WT profiles with matched donor profiles in WT positions."""
    x = np.asarray(x)
    knockout = np.asarray(knockout)
    rng = np.random.default_rng(seed)
    size = round(fraction * len(x))
    slot_probabilities = None
    if signature is not None and sensitivity:
        scaled = (x - x.mean(0)) / np.maximum(x.std(0), 0.1)
        scores = scaled @ signature
        scores = (scores - scores.mean()) / max(scores.std(), 1e-6)
        slot_probabilities = softmax(sensitivity * np.clip(scores, -3, 3))
    slots = rng.choice(len(x), size, replace=False, p=slot_probabilities)
    donors = rng.choice(len(knockout), size, replace=size > len(knockout))
    result = x.copy()
    if size:
        source = x[slots].astype(np.float32)
        donor = knockout[donors].astype(np.float32)
        basis = PCA(min(12, *x.shape), svd_solver="randomized", random_state=0).fit(x)
        source_pc = basis.transform(source)
        donor_pc = basis.transform(donor)
        # Both populations can have a global location shift. Match their relative
        # positions within each population instead of measuring that common shift.
        source_pc -= source_pc.mean(0)
        donor_pc -= donor_pc.mean(0)
        # Partition along the axis of greatest combined spread. Each leaf is small
        # enough for exact assignment while every sampled donor is used once.
        pending = [(np.arange(size), np.arange(size))]
        while pending:
            left, right = pending.pop()
            source_part = source_pc[left]
            donor_part = donor_pc[right]
            if len(left) <= 96:
                i, j = linear_sum_assignment(cdist(source_part, donor_part, "sqeuclidean"))
                result[slots[left[i]]] = donor[right[j]]
                continue
            axis = (source_part.var(0) + donor_part.var(0)).argmax()
            left = left[np.argsort(source_part[:, axis], kind="stable")]
            right = right[np.argsort(donor_part[:, axis], kind="stable")]
            mid = len(left) // 2
            pending.extend(((left[:mid], right[:mid]), (left[mid:], right[mid:])))
    return dict(x=result, c=np.asarray(c), source_indices=np.arange(len(x)),
                weights=np.full(len(x), 1.0 / len(x)))
