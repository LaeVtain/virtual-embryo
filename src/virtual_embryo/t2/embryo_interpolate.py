"""Whole-embryo interpolation using a molecular clock and cell-type landmarks."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from .temporal import _align_frames, _align_population, _canonical_coordinates, _match_cells, read_stage


def molecular_progress(times, means, target):
    """Estimate progress between the local endpoints using all supplied stage means."""
    times = np.asarray(times, dtype=np.float64)
    index = np.searchsorted(times, target) - 1
    start, end = times[index:index+2]
    linear = (target - start) / (end - start)
    direction = means[-1] - means[0]
    total_change = direction @ direction
    if len(times) < 3 or total_change < 1e-12:
        return linear
    times_unit = (times - times[0]) / (times[-1] - times[0])
    target_unit = (target - times[0]) / (times[-1] - times[0])
    observed = np.clip((means - means[0]) @ direction / total_change, 0, 1)

    def progress(t, center, slope):
        low = expit(-slope * center)
        high = expit(slope * (1 - center))
        return (expit(slope * (t - center)) - low) / (high - low)

    def residual(parameters):
        center, slope = parameters
        return np.append(progress(times_unit, center, slope) - observed, 0.1 * np.log(slope / 9))

    parameters = least_squares(residual, [0.5, 9], bounds=([0, 2], [1, 18])).x
    left, right = progress(times_unit[index:index+2], *parameters)
    return float(np.clip((progress(target_unit, *parameters) - left) / (right - left), 0, 1))


def spatial_mean(values, coordinates, neighbors):
    k = min(neighbors, len(coordinates))
    indices = cKDTree(coordinates).query(coordinates, k=k)[1].reshape(len(coordinates), k)
    weights = np.full(indices.size, 1 / k, dtype=np.float32)
    adjacency = sparse.csr_matrix((weights, indices.ravel(), np.arange(0, indices.size+1, k)),
                                  shape=(len(coordinates), len(coordinates)))
    return adjacency @ values


def align_celltypes(z, w, left_labels, right_labels):
    first, second, weights = [], [], []
    common = (set(left_labels) & set(right_labels)) - {"Unknown"}
    for label in sorted(common):
        left = left_labels == label
        right = right_labels == label
        if min(left.sum(), right.sum()) >= 20:
            first.append(z[left].mean(0))
            second.append(w[right].mean(0))
            weights.append(np.sqrt(left.mean() * right.mean()))
    if len(first) < 3:
        return w
    product = (np.asarray(second) * np.asarray(weights)[:, None]).T @ first
    u, _, vt = np.linalg.svd(product)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return w @ u @ vt


def _matrix_root(covariance):
    values, vectors = np.linalg.eigh(covariance + np.eye(3) * 0.025)
    return (vectors * np.sqrt(values)) @ vectors.T


def move_compartments(z, w, left_labels, right_labels, fraction, from_right):
    left_guess = z.copy()
    right_guess = w.copy()
    for label in set(left_labels) & set(right_labels):
        left = left_labels == label
        right = right_labels == label
        if min(left.sum(), right.sum()) < 8:
            continue
        a, b = z[left], w[right]
        center_a, center_b = a.mean(0), b.mean(0)
        root = _matrix_root(np.cov(a.T))
        inverse = np.linalg.inv(root)
        transport = inverse @ _matrix_root(root @ np.cov(b.T) @ root) @ inverse
        left_guess[left] = (1-fraction) * a + fraction * ((a-center_a) @ transport + center_b)
        right_guess[right] = fraction * b + (1-fraction) * ((b-center_b) @ np.linalg.inv(transport) + center_a)
    return np.where(from_right[:, None], right_guess, left_guess)


def interpolate(stages, target_time, cells, seed=101, *, include_donor=False, donor_policy=None, match_capture_depth=False,
                sample_endpoints=None):
    """Use stage dictionaries with x, c, genes and measured cell-type labels."""
    times = sorted(stages)
    if (len(times) < 2 or not np.isfinite([*times, target_time]).all()
            or target_time in stages or not times[0] < target_time < times[-1]):
        raise ValueError("Supply both flanking stages and exclude the target time")
    if cells < 3:
        raise ValueError("Specify at least three cells")
    source_time = max(t for t in times if t < target_time)
    end_time = min(t for t in times if t > target_time)
    fraction = (target_time - source_time) / (end_time - source_time)
    genes = np.asarray(stages[source_time]["genes"]).astype(str).tolist()
    populations = {}
    means = []
    for t in times:
        pop = stages[t]
        x, c = _align_population(pop, genes)
        if not match_capture_depth:
            means.append(x.mean(0, dtype=np.float64))
        if t in (source_time, end_time):
            labels = np.asarray(pop["labels"]).astype(str)
            if labels.shape != (len(x),) or cells > len(x):
                raise ValueError("Each bracketing stage needs labels and enough distinct cells")
            populations[t] = x, c, labels
    if match_capture_depth:
        from .embryo_capture import aligned_stage_means
        means = aligned_stage_means(stages, genes)
    later_share = molecular_progress(times, np.stack(means), target_time)
    left_x, left_c, left_labels = populations[source_time]
    right_x, right_c, right_labels = populations[end_time]
    rng = np.random.default_rng(seed)
    z, scale_a = _canonical_coordinates(left_c)
    w, scale_b = _canonical_coordinates(right_c)
    w = align_celltypes(z, _align_frames(z, w, rng), left_labels, right_labels)
    if sample_endpoints is None:
        a = rng.choice(len(z), cells, replace=False)
        b = rng.choice(len(w), cells, replace=False)
    else:
        a = sample_endpoints(stages[source_time], cells, rng)
        b = sample_endpoints(stages[end_time], cells, rng)

    left_features = spatial_mean(left_x, left_c, 24)[a]
    right_features = spatial_mean(right_x, right_c, 24)[b]
    features = np.vstack([left_features, right_features])
    dimension = min(12, cells, features.shape[1])
    codes = PCA(n_components=dimension, svd_solver="randomized", random_state=seed).fit_transform(features)
    codes /= np.maximum(codes.std(0), 1e-6) * np.sqrt(dimension / 3)
    partner = _match_cells(np.column_stack([z[a], codes[:cells]]),
                           np.column_stack([w[b], codes[cells:]]))
    b = b[partner]
    z, w = z[a], w[b]
    from_right = np.zeros(cells, dtype=bool)
    from_right[rng.permutation(cells)[:round(cells * later_share)]] = True
    if donor_policy is not None:
        from_right = donor_policy(a, b, from_right)

    left_field = spatial_mean(w-z, z, 32)
    right_field = spatial_mean(z-w, w, 32)
    smooth = np.where(from_right[:, None], w + (1-fraction)*right_field, z + fraction*left_field)
    tissue = move_compartments(z, w, left_labels[a], right_labels[b], fraction, from_right)
    coordinates = (smooth + tissue) / 2
    coordinates -= coordinates.mean(0)
    scale = scale_a ** (1-fraction) * scale_b ** fraction
    coordinates *= scale / np.maximum(np.sqrt(np.mean(coordinates * coordinates, axis=0)), 1e-8)
    prediction = {"x": np.where(from_right[:, None], right_x[b], left_x[a]),
                  "c": coordinates.astype("float32"), "genes": np.asarray(genes)}
    if include_donor:
        prediction["row"] = np.where(from_right, b, a)
        prediction["later"] = from_right
    return prediction


def read_annotated_stage(path, genes, *, include_depth=False):
    values = read_stage(path, genes)
    if path.suffix.lower() == ".h5ad":
        metadata = ad.read_h5ad(path, backed="r")
        try:
            values["labels"] = metadata.obs["celltype"].astype(str).to_numpy(dtype=str)
            if include_depth:
                columns = metadata.var_names.get_indexer(genes)
                counts = metadata.layers["counts"][:, columns]
                counts = counts.toarray() if sparse.issparse(counts) else counts
                values["depth"] = np.asarray(counts, dtype=np.float32).sum(axis=1)
        finally:
            metadata.file.close()
    else:
        with np.load(path, allow_pickle=False) as archive:
            values["labels"] = archive["labels" if "labels" in archive else "celltype"].astype(str)
            if include_depth:
                if "counts" not in archive:
                    raise ValueError("Depth matching requires a count matrix in each stage NPZ")
                columns = {str(g): i for i, g in enumerate(archive["genes"])}
                counts = archive["counts"][:, [columns[g] for g in genes]]
                values["depth"] = np.asarray(counts, dtype=np.float32).sum(axis=1)
    return values


def registered_prediction(stages, target, cells, seed=101, *, fit_compartments=False, track_sections=False,
                          match_capture_depth=False, include_donor=False, sample_endpoints=None):
    """Move each sampled donor along the reciprocal displacement between both measured tissues."""
    from . import tissue_registration
    if target in stages or not min(stages) < target < max(stages):
        raise ValueError("Supply flanking stages and exclude the target")
    earlier = max(time for time in stages if time < target)
    later = min(time for time in stages if time > target)
    donor_policy = None
    if fit_compartments:
        from .embryo_compartments import assign_donors, tissue_distribution, tissue_groups
        groups = tissue_groups(stages, earlier, later)
        desired = tissue_distribution(groups, target)
        def donor_policy(left_rows, right_rows, sampled):
            return assign_donors(groups[earlier][left_rows], groups[later][right_rows], sampled, desired)
    prediction = interpolate(stages, target, cells, seed, include_donor=True, donor_policy=donor_policy,
                             match_capture_depth=match_capture_depth, sample_endpoints=sample_endpoints)
    fraction = (target-earlier)/(later-earlier)
    genes = prediction["genes"].tolist()
    endpoints = []
    for time in (earlier, later):
        population = stages[time]
        expression, coordinates = _align_population(population, genes)
        endpoints.append(dict(x=expression, c=coordinates, labels=population["labels"]))
    left, right = endpoints
    codes = tissue_registration.expression_codes(left, right)
    field = tissue_registration.register(left, right, seed, codes, celltype_frame=True, density_comparison=False)
    moved = tissue_registration.interpolate(field, prediction, fraction, "expression")
    coords = .5*(moved+prediction["c"])
    if track_sections:
        coords = tissue_registration.preserve_section_axis(coords, left["c"], right["c"], fraction)
    result = dict(x=prediction["x"], c=coords, genes=prediction["genes"])
    if include_donor:
        result.update(row=prediction["row"], later=prediction["later"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=2, required=True, metavar=("TIME", "FILE"))
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--spatial-model", choices=("landmarks", "registered", "compartments", "sections"),
                        default="sections", help="sections fits tissue abundance and follows the measured slice axis")
    parser.add_argument("--match-capture-depth", action="store_true",
                        help="Estimate the clock from shared tissue and measured panel-depth strata")
    parser.add_argument("--section-volume", action="store_true",
                        help="Estimate the physical radius from tissue areas in the measured source sections")
    parser.add_argument("--growth-reference", type=Path,
                        help="Estimate the physical growth curve using measured external histology lengths")
    parser.add_argument("--sample-sections", action="store_true",
                        help="Reconstruct the measured section spacing and depth distribution")
    parser.add_argument("--continuous-section-depth", action="store_true",
                        help="Estimate the density between observed sections before resampling the new stack")
    parser.add_argument("--local-expression", action="store_true",
                        help="Match intact expression profiles to the nearby tissue on the finished spatial grid")
    parser.add_argument("--interpolate-gene-ranks", action="store_true",
                        help="Interpolate cell expression ranks within each flanking developmental stage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    genes = args.panel.read_text(encoding="utf-8").splitlines()
    stages = {}
    for age, path in args.stage:
        time = float(age)
        if time in stages:
            parser.error("Each stage time must be unique")
        stages[time] = read_annotated_stage(Path(path), genes, include_depth=args.match_capture_depth)
    need_donors = args.sample_sections or args.interpolate_gene_ranks
    with threadpool_limits(limits=1):
        if args.spatial_model == "landmarks":
            result = interpolate(stages, args.target_time, args.cells, args.seed,
                                 match_capture_depth=args.match_capture_depth, include_donor=need_donors)
        else:
            result = registered_prediction(stages, args.target_time, args.cells, args.seed,
                                           fit_compartments=args.spatial_model in ("compartments", "sections"),
                                           track_sections=args.spatial_model == "sections",
                                           match_capture_depth=args.match_capture_depth, include_donor=need_donors)
        if args.growth_reference is not None:
            from .embryo_growth import project_radius
            with np.load(args.growth_reference, allow_pickle=False) as reference:
                result["c"] = project_radius(stages, args.target_time, result["c"], reference)
        elif args.section_volume:
            from .section_volume import adjust_radius
            result["c"] = adjust_radius(stages, args.target_time, result["c"])
        if args.sample_sections:
            from .section_sampling import restore_sections
            result["c"] = restore_sections(stages, args.target_time, result["c"], result["row"], result["later"],
                                           continuous=args.continuous_section_depth)
        if args.interpolate_gene_ranks:
            from .embryo_expression_ranks import interpolate_ranks
            result["x"] = interpolate_ranks(result["x"], result["c"], result["later"], args.seed)
        if need_donors:
            del result["row"], result["later"]
        if args.local_expression:
            from .local_expression import place_local_profiles
            result["x"] = place_local_profiles(result["x"], result["c"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)


if __name__ == "__main__":
    main()
