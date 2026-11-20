"""Estimate reciprocal smooth deformations between the measured spatial endpoints."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.decomposition import PCA

from virtual_embryo.t2.temporal import _align_frames, _canonical_coordinates

GRID = 32
EXTENT = 3.6


def expression_codes(left, right):
    rng = np.random.default_rng(77111)
    arrays = []
    for pop in (left, right):
        abundance = np.expm1(pop["x"])
        x = np.sqrt(abundance / np.maximum(1., abundance.sum(1, keepdims=True)))
        x -= x.mean(0)
        arrays.append(x)
    fit = np.vstack([a[rng.choice(len(a), min(2048, len(a)), replace=False)] for a in arrays])
    pca = PCA(n_components=min(5, *fit.shape), svd_solver="randomized", random_state=712).fit(fit)
    scores = [pca.transform(a) for a in arrays]
    return [np.clip(s / np.maximum(s.std(0), 1e-6), -3, 3) for s in scores]


def voxel(c):
    return (c / (2 * EXTENT) + .5) * (GRID - 1)


def volumes(u, scores):
    points = np.clip(np.rint(voxel(u)).astype(int), 0, GRID - 1)
    idx = np.ravel_multi_index(points.T, (GRID,) * 3)
    def raster(weights=None):
        v = np.bincount(idx, weights=weights, minlength=GRID**3).reshape((GRID,) * 3).astype(float)
        return gaussian_filter(v, 1.3)
    density = raster()
    cutoff = .03 * np.quantile(density[density > np.max(density) * .01], .9)
    scaled = density / np.maximum(np.quantile(density[density > cutoff], .85), 1e-9)
    fields = [np.log1p(3 * scaled)]
    for vector in scores.T:
        fields.append(.60 * raster(vector) / (density + cutoff) * np.sqrt(np.minimum(1., scaled)))
    return np.stack(fields)


def move(fixed, moving):
    grid = np.indices((GRID,) * 3, dtype=float)
    gradients = np.stack([np.stack(np.gradient(channel)) for channel in moving])
    displacement = np.zeros_like(grid)
    for smooth, steps in ((2.4, 12), (1.4, 14), (.85, 10)):
        for _ in range(steps):
            positions = grid + displacement
            numerator = np.zeros_like(displacement)
            denom = np.full(displacement.shape[1:], .025)
            for target, channel, gradient in zip(fixed, moving, gradients):
                warped = map_coordinates(channel, positions, order=1, mode="nearest")
                grad = np.stack([map_coordinates(g, positions, order=1, mode="nearest") for g in gradient])
                diff = target - warped
                numerator += grad * diff
                denom += (grad**2).sum(0) + diff**2
            delta = np.stack([gaussian_filter(d, smooth) for d in numerator / denom])
            displacement += .6 * delta
            displacement = np.stack([gaussian_filter(d, .55) for d in displacement])
    return displacement


def sample_field(field, normalized):
    return np.column_stack([map_coordinates(d, voxel(normalized).T, order=1, mode="nearest")
                            for d in field]) * (2 * EXTENT / (GRID - 1))


def register(left, right, seed, scores, *, celltype_frame=False, density_comparison=True):
    z, s0 = _canonical_coordinates(left["c"].astype(float))
    w, s1 = _canonical_coordinates(right["c"].astype(float))
    w = _align_frames(z, w, np.random.default_rng(seed))
    if celltype_frame:
        from virtual_embryo.t2.embryo_interpolate import align_celltypes
        w = align_celltypes(z, w, np.asarray(left["labels"]).astype(str), np.asarray(right["labels"]).astype(str))
    a, b = volumes(z, scores[0]), volumes(w, scores[1])
    result = {}
    channels = (("density", 1), ("expression", len(a))) if density_comparison else (("expression", len(a)),)
    for name, num in channels:
        forward, backward = move(a[:num], b[:num]), move(b[:num], a[:num])
        # Enforce reciprocal motion at the displaced coordinates using both observed directions.
        f = sample_field(forward, z)
        g = sample_field(backward, w)
        f = .5 * (f - sample_field(backward, z + f))
        g = .5 * (g - sample_field(forward, w + g))
        result[name] = (f, g)
    return z, w, s0, s1, result


def interpolate(registration, pred, fraction, arm):
    z, w, s0, s1, fields = registration
    f, g = fields[arm]
    later, rows = pred["later"], pred["row"]
    out = np.zeros((len(rows), 3))
    out[~later] = (z + fraction * f)[rows[~later]]
    out[later] = (w + (1 - fraction) * g)[rows[later]]
    out -= out.mean(0)
    scale = s0**(1 - fraction) * s1**fraction
    return (out * (scale / np.sqrt(np.mean(out**2, axis=0)))).astype(np.float32)


def preserve_section_axis(coords, earlier, later, fraction):
    """Track section depth separately from the two in-plane widths when their ranks cross."""
    widths = []
    for specimen in (earlier, later):
        depth_levels = np.unique(np.round(specimen[:, 2], 2))
        if not 3 <= len(depth_levels) < len(specimen)/4:
            return coords
        centered = specimen - specimen.mean(axis=0, dtype=float)
        planar = centered[:, :2].T @ centered[:, :2] / len(centered)
        widths.append(np.r_[np.sqrt(np.linalg.eigvalsh(planar)[::-1]), centered[:, 2].std()])
    first, last = widths
    if np.array_equal(np.argsort(first), np.argsort(last)):
        return coords
    anatomical = np.sort(np.exp((1-fraction)*np.log(first)+fraction*np.log(last)))[::-1]
    previously_fitted = np.sort(first)[::-1]**(1-fraction) * np.sort(last)[::-1]**fraction
    centered = coords - coords.mean(axis=0, dtype=float)
    _, basis = np.linalg.eigh(centered.T @ centered)
    rotation = basis[:, ::-1]
    corrected = ((centered @ rotation) * (anatomical/previously_fitted)) @ rotation.T
    return (corrected + coords.mean(axis=0)).astype(np.float32)
