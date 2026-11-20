"""Interpolate an embryo and calibrate its expression using an external stage atlas."""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits

from .temporal import predict_population, read_stage
from .embryo_interpolate import interpolate, read_annotated_stage


def _profile(values):
    return values / max(float(values.sum()), 1e-12)


def predict_with_atlas(stages, target, cells, seed, atlas, *, base_model="temporal"):
    """Use the enclosing spatial stages, plus shared observed ages for assay calibration.

    The atlas contains donor-balanced mean log expression by age. During stage
    validation its observed row at the target age is excluded before curve fitting.
    """
    if target in stages:
        raise ValueError("Omit the prediction target from spatial inputs")
    left = max(t for t in stages if t < target)
    right = min(t for t in stages if t > target)
    if base_model == "clock":
        base = interpolate(stages, target, cells, seed)
    elif base_model == "temporal":
        base = predict_population(stages[left], stages[right], left, right, target, cells, seed)
    else:
        raise ValueError(f"Unknown endpoint model {base_model}")
    return calibrate_prediction(stages, target, atlas, base)


def calibrate_prediction(stages, target, atlas, base):
    """Transfer replicated developmental curvature to an existing joint cell prediction."""
    if target in stages:
        raise ValueError("Omit the prediction target from spatial inputs")
    left = max(t for t in stages if t < target)
    right = min(t for t in stages if t > target)
    raw_times = np.asarray(atlas["times"])
    if ((raw_times > 7.25) & (raw_times < 8)).any():
        raise ValueError("External atlas includes a protected age")
    observed = raw_times != target
    times = raw_times[observed]
    means = np.asarray(atlas["stage_means"])[observed]
    gene_index = {gene: i for i, gene in enumerate(atlas["genes"].astype(str))}
    pairs = [(i, gene_index[g]) for i, g in enumerate(base["genes"].astype(str)) if g in gene_index]
    if not pairs:
        raise ValueError("No shared genes in the external and spatial panels")
    spatial_cols, atlas_cols = np.array(pairs).T
    shared = sorted(set(stages) & set(times))
    if len(shared) < 2 or left < min(times) or right > max(times):
        raise ValueError("The atlas must cover both endpoints and share at least two calibration ages")
    if any(list(pop["genes"]) != list(base["genes"]) for pop in stages.values()):
        raise ValueError("Spatial stages must use the same ordered gene panel")
    curve = PchipInterpolator(times, means, axis=0)

    def external_profile(time):
        indices = np.flatnonzero(times == time)
        mean = means[indices[0]] if len(indices) else np.maximum(0, curve(time))
        return _profile(mean[atlas_cols])

    external = np.stack([external_profile(t) for t in shared])
    spatial = np.stack([_profile(stages[t]["x"][:, spatial_cols].mean(0, dtype=np.float64)) for t in shared])
    ext_mean = external.mean(0)
    spatial_mean = spatial.mean(0)
    centered_ext = external - ext_mean
    centered_spatial = spatial - spatial_mean
    slope = np.sum(centered_ext * centered_spatial, axis=0) / (
        np.square(centered_ext).sum(0) + np.square(0.15 * ext_mean + .0001))
    slope = np.clip(slope, 0, 3 * (spatial_mean + .0001) / (ext_mean + .0001))
    fraction = (target - left) / (right - left)
    curvature = external_profile(target) - (
        (1-fraction) * external_profile(left) + fraction * external_profile(right))
    x = base["x"]
    mean = x[:, spatial_cols].mean(0, dtype=np.float64)
    adjusted_mean = _profile(np.maximum(0, _profile(mean) + slope * curvature)) * mean.sum()
    prevalence = np.count_nonzero(x[:, spatial_cols], axis=0) / len(x)
    log_weight = np.clip((adjusted_mean-mean) / (.01+prevalence), -np.log(2), np.log(2))
    abundance = np.expm1(x)
    library = abundance.sum(1)
    abundance[:, spatial_cols] *= np.exp(log_weight).astype(np.float32)
    total = abundance.sum(1)
    renorm = np.divide(library, total, out=np.ones_like(total), where=total>0)
    return dict(base, x=np.log1p(abundance * renorm[:, None]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=2, required=True, metavar=("TIME", "FILE"))
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--base-model", choices=("temporal", "clock"), default="temporal",
                        help="Use clock for the multi-stage, tissue-registered model")
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    genes = args.panel.read_text(encoding="utf-8").splitlines()
    stages = {}
    reader = read_annotated_stage if args.base_model == "clock" else read_stage
    for age, path in args.stage:
        time = float(age)
        if time in stages:
            parser.error("Each stage time must be unique")
        stages[time] = reader(Path(path), genes)
    with threadpool_limits(limits=1), np.load(args.atlas, allow_pickle=False) as atlas:
        result = predict_with_atlas(stages, args.target_time, args.cells, args.seed, atlas, base_model=args.base_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)


if __name__ == "__main__":
    main()
