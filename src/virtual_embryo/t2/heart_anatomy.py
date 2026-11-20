"""Match a cardiac spatial population to its later anatomical reference."""
import numpy as np
from scipy.spatial import cKDTree


def rms_radius(c):
    centered = c - c.mean(axis=0)
    return np.sqrt(np.mean((centered ** 2).sum(axis=1)))


def aligned_atlas(source, atlas):
    def canonical(x):
        x = x - x.mean(axis=0)
        radius = rms_radius(x)
        _, axes = np.linalg.eigh(np.cov(x.T))
        return x @ axes / radius, axes
    s, axes = canonical(source)
    a, _ = canonical(atlas)
    # Multiple mounts can be mirrored. Select anatomical orientation by symmetric surface proximity.
    s_fit = s[np.linspace(0, len(s) - 1, min(2048, len(s)), dtype=int)]
    a_fit = a[np.linspace(0, len(a) - 1, min(2048, len(a)), dtype=int)]
    tree_s, tree_a = cKDTree(s_fit), cKDTree(a_fit)
    flips = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    errors = [tree_s.query(a_fit * f)[0].mean() + tree_a.query(s_fit * f)[0].mean() for f in flips]
    return s, a * flips[np.argmin(errors)], axes


def shape_coordinates(source, atlas, fraction):
    s, a, axes = aligned_atlas(source, atlas)
    rng = np.random.default_rng(854)
    directions = rng.normal(size=(32, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    grid = np.linspace(.0005, .9995, 257)
    # Transport one-dimensional marginals and average the projected motions in 3D.
    directions = np.vstack((np.eye(3), directions))
    delta = np.zeros((len(s), len(directions)))
    for i, direction in enumerate(directions):
        current, future = s @ direction, a @ direction
        lo, hi = np.quantile(current, grid), np.quantile(future, grid)
        delta[:, i] = np.interp(current, lo, hi - lo)
    deformation = delta @ np.linalg.pinv(directions.T)
    warped = s + fraction * deformation
    return ((warped - warped.mean(axis=0)) @ axes.T) / rms_radius(warped)


def place_population(predicted, source_coords, ages, atlas):
    _, latest, target = ages
    if not latest < target <= 14.5:
        raise ValueError("The E14.5 anatomical reference must bracket the prediction with its source")
    fraction = (target - latest) / (14.5 - latest)
    atlas_coords, atlas_radius = atlas
    if atlas_coords.ndim != 2 or atlas_coords.shape[1] != 3 or not atlas_radius > 0:
        raise ValueError("Expected cardiac atlas coordinates and positive radius in micrometres")
    current = rms_radius(source_coords)
    projected = current * (atlas_radius / current) ** fraction
    previous = rms_radius(predicted)
    influence = 1 - np.exp(-((target - latest) / .75) ** 2)
    radius = np.exp((1 - influence) * np.log(previous) + influence * np.log(max(previous, projected)))
    return (shape_coordinates(predicted, atlas_coords, fraction) * radius).astype(np.float32)


def early_weights(x, c, genes, horizon):
    markers = [genes.index(g) for g in ("Myh7", "Myl7", "Myl2", "Tbx5", "Hand1", "Hand2", "Isl1") if g in genes]
    if not markers:
        raise ValueError("Early whole-embryo input must include cardiac lineage markers")
    positive = (x[:, markers] > 0).mean(axis=1)
    neighbors = cKDTree(c).query(c, k=min(40, len(c)))[1]
    local = positive[neighbors].mean(axis=1)
    spread = np.quantile(local, .85) - np.quantile(local, .15)
    presence = np.clip((local - np.median(local)) / max(spread, 1e-5), -1.2, 1.2)
    weights = np.exp((1 - np.exp(-horizon / .5)) * presence)
    return weights / weights.sum()


