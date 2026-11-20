"""Project measured MERFISH neighborhoods into cardiac niches learned from spatial RNA sections."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax


def fractions(x):
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1.)


def reorder_niches(model, genes):
    expected = model["genes"].tolist()
    if len(expected) != len(genes) or set(expected) != set(genes):
        raise ValueError("Spatial RNA niches require the same gene set as the MERFISH panel")
    idx = [expected.index(g) for g in genes]
    return {**model, **{k: model[k][..., idx] for k in
                        ("components", "center", "early_mean", "bulk", "local")}}


def merfish_posterior(x, c, ref):
    pooled = np.zeros(x.shape[1], dtype=float)
    for start in range(0, len(x), 2048):
        pooled += fractions(np.expm1(x[start:start + 2048])).sum(0)
    mapping = np.clip((ref["early_mean"] + 3e-5) / (pooled / len(x) + 3e-5), .05, 20)
    tree = cKDTree(c)
    result = []
    for start in range(0, len(c), 2048):
        positions = c[start:start + 2048]
        nn = tree.query(positions, k=min(16, len(c)))[1]
        means = np.zeros((len(positions), x.shape[1]), dtype=np.float32)
        for neighbours in nn.T:
            means += fractions(np.expm1(x[neighbours]))
        scaled = fractions(means * mapping)
        code = (np.log1p(500 * scaled) - ref["center"]) @ ref["components"].T
        distances = ((code[:, None] - ref["centroids"][None]) ** 2).sum(axis=2)
        result.append(softmax(-distances / 3, axis=1).astype(np.float32))
    return np.concatenate(result)
