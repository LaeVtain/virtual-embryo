"""Fit source-cell sampling weights to an externally projected heart pseudobulk."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.decomposition import PCA


def abundance(x):
    linear = np.expm1(x)
    return linear / np.maximum(linear.sum(axis=1, keepdims=True), 1)


def calibrate(x, delta, strength):
    rng = np.random.default_rng(371)
    ids = rng.choice(len(x), min(len(x), 8192), replace=False)
    template = abundance(x[ids])
    mean = template.mean(axis=0)
    scale = np.maximum(template.std(axis=0), .0002)
    standardized = (template - mean) / scale
    dimensions = min(24, len(template) - 1, template.shape[1])
    model = PCA(n_components=dimensions, whiten=True, svd_solver="randomized", random_state=371)
    coords = model.fit_transform(standardized)
    wanted = mean * np.exp(delta)
    wanted /= wanted.sum()
    target = model.transform(((wanted - mean) / scale)[None])[0] * strength

    def objective(theta):
        log_weights = coords @ theta
        value = logsumexp(log_weights) - target @ theta + .12 * (theta @ theta)
        gradient = softmax(log_weights) @ coords - target + .24 * theta
        return value, gradient

    optimum = minimize(objective, np.zeros(coords.shape[1]), jac=True, method="L-BFGS-B")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    scores = []
    for start in range(0, len(x), 2048):
        block = (abundance(x[start:start + 2048]) - mean) / scale
        code = model.transform(block)
        scores.append(np.clip(code @ optimum.x, -1.6, 1.6))
    probabilities = softmax(np.concatenate(scores))
    realized = np.zeros_like(mean, dtype=np.float64)
    for start in range(0, len(x), 2048):
        cells = abundance(x[start:start + 2048])
        realized += probabilities[start:start + len(cells)] @ cells
    residual = np.log((wanted + 1e-6) / (realized + 1e-6))
    return probabilities, residual
