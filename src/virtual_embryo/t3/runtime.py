"""Whole-population empirical ranks mapped to training KO quantiles."""
import numpy as np

GRID = np.linspace(0.0, 1.0, 65)
CONDITIONS = {"Mab21l2": "mab", "Gata4": "gata", "mab": "mab", "gata": "gata"}
PERTURBED_GENES = {"mab": "Mab21l2", "gata": "Gata4"}


def features(x, tilt, condition):
    keep = tilt[condition + "_keep"]
    a = (x[:, keep] - tilt[condition + "_center"]) / tilt[condition + "_scale"]
    basis = tilt[condition + "_components"]
    return (a @ basis.T - tilt[condition + "_pca_mean"] @ basis.T) / tilt[condition + "_pcscale"]


def transform(x, model, seed):
    """Rank one complete input population; equal values use seeded random ties."""
    n = len(x)
    q = (np.arange(n) + 0.5) / n
    rng = np.random.default_rng([20261111, seed])
    out = np.zeros_like(x, dtype=float)
    for g in range(x.shape[1]):
        rate = model["target_rate"][g]
        perm = rng.permutation(n)
        order = perm[np.argsort(x[perm, g], kind="stable")]
        active = q > 1 - rate
        if rate > 0:
            out[order[active], g] = np.interp(
                np.clip((q[active] - (1 - rate)) / rate, 0, 1), GRID, model["target_knots"][g]
            )
    np.expm1(out, out=out)
    total = out.sum(1)
    out *= (10000 / total)[:, None]
    np.log1p(out, out=out)
    return out


def predict(x, c, tilt, condition, model, seed=101):
    logits = features(x, tilt, condition) @ model["coefficients"]
    w = np.exp(logits - logits.max())
    w /= w.sum()
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    ix = np.searchsorted(cdf, np.random.default_rng(seed).random(len(x)), side="right")
    y = transform(x[ix], model, seed)
    return dict(x=y, c=c[ix], source_indices=ix, weights=w)
