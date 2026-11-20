"""Interpolate joint spatial populations and forecast from the two latest stages."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def _align_population(population, genes):
    names = np.asarray(population["genes"]).astype(str).tolist()
    columns = {name: index for index, name in enumerate(names)}
    x = np.asarray(population["x"], dtype=np.float32)
    c = np.asarray(population["c"], dtype=np.float64)
    if len(columns) != len(names) or x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("Each expression column requires one unique gene name")
    if c.shape != (len(x), 3) or len(x) < 3:
        raise ValueError("Each population requires matching cells and 3D coordinates")
    if not np.isfinite(x).all() or (x < 0).any() or not np.isfinite(c).all():
        raise ValueError("Provide finite coordinates and finite nonnegative log expression")
    if set(columns) != set(genes):
        raise ValueError("Both stages must use the same gene panel")
    if names != genes:
        x = x[:, [columns[name] for name in genes]]
    return x, c


def _canonical_coordinates(c):
    centered = c - c.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    axes = axes[:, ::-1].copy()
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    rotated = centered @ axes
    scales = np.sqrt(np.mean(rotated ** 2, axis=0))
    if scales.min() <= 0:
        raise ValueError("Each stage requires nondegenerate three-dimensional coordinates")
    return rotated / scales, scales


def _align_frames(left, right, rng):
    """Choose a proper rotation using symmetric nearest-neighbour distances."""
    left_sample = left[rng.choice(len(left), min(4096, len(left)), replace=False)]
    right_sample = right[rng.choice(len(right), min(4096, len(right)), replace=False)]
    left_tree = cKDTree(left_sample)
    right_tree = cKDTree(right_sample)
    flips = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
    distances = []
    for flip in flips:
        forward = left_tree.query(right_sample * flip)[0]
        reverse = right_tree.query(left_sample * flip)[0]
        distances.append(forward.mean() + reverse.mean())
    return right * flips[np.argmin(distances)]


def _match_cells(left, right):
    """Balanced expression/position matching, solving small assignment blocks exactly."""
    partner = np.empty(len(left), dtype=np.intp)

    def partition(a, b):
        if len(a) <= 48:
            _, indices = linear_sum_assignment(cdist(left[a], right[b], "sqeuclidean"))
            partner[a] = b[indices]
            return
        axis = np.argmax(left[a].var(axis=0) + right[b].var(axis=0))
        midpoint = len(a) // 2
        a = a[np.argsort(left[a, axis], kind="stable")]
        b = b[np.argsort(right[b, axis], kind="stable")]
        partition(a[:midpoint], b[:midpoint])
        partition(a[midpoint:], b[midpoint:])

    partition(np.arange(len(left)), np.arange(len(right)))
    return partner


def predict_population(earlier, later, earlier_time, later_time, target_time, cells, seed=101):
    """Predict from two released populations. Input stages must exclude the target."""
    if not (np.isfinite([earlier_time, later_time, target_time]).all()
            and earlier_time < later_time and earlier_time < target_time and target_time != later_time):
        raise ValueError("Use distinct, ordered training times and a new target after the first time")
    if cells <= 0:
        raise ValueError("Specify a positive prediction size")
    genes = np.asarray(earlier["genes"]).astype(str).tolist()
    left_x, left_c = _align_population(earlier, genes)
    right_x, right_c = _align_population(later, genes)
    rng = np.random.default_rng(seed)
    fraction = (target_time - earlier_time) / (later_time - earlier_time)

    if fraction < 1:
        if cells > min(len(left_x), len(right_x)):
            raise ValueError("Interpolation requires enough distinct cells at each stage")
        left_z, left_scale = _canonical_coordinates(left_c)
        right_z, right_scale = _canonical_coordinates(right_c)
        right_z = _align_frames(left_z, right_z, rng)
        left_rows = rng.choice(len(left_x), cells, replace=False)
        right_rows = rng.choice(len(right_x), cells, replace=False)
        a, b = left_x[left_rows], right_x[right_rows]
        z, w = left_z[left_rows], right_z[right_rows]
        pooled = np.vstack([a, b])
        dimension = min(12, cells, a.shape[1])
        pca = PCA(n_components=dimension, svd_solver="randomized", random_state=seed)
        codes = pca.fit_transform(pooled)
        codes /= np.maximum(codes.std(axis=0), 1e-6) * np.sqrt(dimension / 3)
        partner = _match_cells(np.column_stack([z, codes[:cells]]),
                               np.column_stack([w, codes[cells:]]))
        scale = np.exp((1 - fraction) * np.log(left_scale) + fraction * np.log(right_scale))
        choose_later = np.zeros(cells, dtype=bool)
        choose_later[rng.permutation(cells)[:round(cells * fraction)]] = True
        parent = np.where(choose_later[:, None], w[partner], z)
        transport = (1 - fraction) * z + fraction * w[partner]
        # Partial displacement keeps within-stage neighbours near each other.
        coords = (parent + transport) / 2
        coords -= coords.mean(axis=0)
        coords *= scale / np.maximum(np.sqrt(np.mean(coords ** 2, axis=0)), 1e-8)
        expression = np.where(choose_later[:, None], b[partner], a)
    else:
        if cells > len(right_x):
            raise ValueError("Forecast requires enough distinct cells at the latest stage")
        rows = rng.choice(len(right_x), cells, replace=False)
        horizon = fraction - 1
        # Damping limits long-horizon sensitivity to a single measured interval.
        trend_weight = horizon / (1 + horizon)
        expression = right_x[rows].copy()
        library = np.expm1(expression).sum(axis=1)
        prior_detection = np.count_nonzero(left_x) / left_x.size
        recent_detection = np.count_nonzero(right_x) / right_x.size
        if 0 < recent_detection < prior_detection:
            keep = (recent_detection / prior_detection) ** trend_weight
            dropped = rng.random(expression.shape, dtype=np.float32) >= keep
            # Preserve at least the dominant measured gene in each nonempty cell.
            dropped[np.arange(cells), expression.argmax(axis=1)] = False
            expression[dropped] = 0
        # Estimate growth separately so the field cannot flatten the tissue.
        centered_right = right_c - right_c.mean(axis=0)
        radius_right = np.sqrt(np.mean(np.sum(centered_right ** 2, axis=1)))
        radius_left = np.sqrt(np.mean(np.sum((left_c - left_c.mean(axis=0)) ** 2, axis=1)))
        if min(radius_right, radius_left) <= 0:
            raise ValueError("Both stages require positive tissue radius")
        radius_gain = (radius_right / radius_left) ** trend_weight
        coords = centered_right[rows] * radius_gain
        abundance = np.expm1(expression)
        total = abundance.sum(axis=1)
        correction = np.divide(library, total, out=np.ones_like(library), where=total > 0)
        expression = np.log1p(abundance * correction[:, None])
    return {"x": expression, "c": coords.astype(np.float32), "genes": np.asarray(genes)}


def read_stage(path, panel=None):
    """Load official H5AD log expression or an extracted joint population NPZ."""
    if path.suffix.lower() == ".h5ad":
        import anndata as ad
        from scipy import sparse
        data = ad.read_h5ad(path, backed="r")
        try:
            names = data.var_names.astype(str)
            genes = list(panel) if panel is not None else names.tolist()
            if not names.is_unique:
                raise ValueError(f"Repeated gene names in {path}")
            indices = names.get_indexer(genes)
            if (indices < 0).any():
                raise ValueError(f"Missing panel genes in {path}")
            # Official released X contains log-normalised expression.
            stored = data.X
            matrix = stored.to_memory() if hasattr(stored, "to_memory") else stored[:]
            selected = matrix[:, indices]
            x = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            return {"x": x, "c": np.asarray(data.obsm["spatial_3D"]), "genes": np.asarray(genes)}
        finally:
            data.file.close()
    with np.load(path, allow_pickle=False) as archive:
        stage = {key: archive[key] for key in ("x", "c", "genes")}
    if panel is not None:
        names = stage["genes"].astype(str).tolist()
        if len(set(names)) != len(names) or any(gene not in names for gene in panel):
            raise ValueError(f"Invalid or incomplete gene panel in {path}")
        if names != list(panel):
            stage["x"] = stage["x"][:, [names.index(gene) for gene in panel]]
        stage["genes"] = np.asarray(panel)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier", required=True, type=Path)
    parser.add_argument("--later", required=True, type=Path)
    parser.add_argument("--earlier-time", required=True, type=float)
    parser.add_argument("--later-time", required=True, type=float)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--cells", required=True, type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--panel", type=Path, help="Official panel text file with one gene per line")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    genes = args.panel.read_text(encoding="utf-8").splitlines() if args.panel else None
    first, second = read_stage(args.earlier, genes), read_stage(args.later, genes)
    with threadpool_limits(limits=1):
        out = predict_population(first, second, args.earlier_time, args.later_time,
                                 args.target_time, args.cells, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)


if __name__ == "__main__":
    main()
