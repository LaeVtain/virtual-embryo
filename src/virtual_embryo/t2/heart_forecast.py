"""Predict a future spatial heart with a ventricular development reference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from .heart_composition import calibrate
from .heart_sampling import donor_adjustment, molecule_counts, stratified_counts
from .heart_anatomy import early_weights, place_population, refine_shape, rms_radius
from .heart_spatial_rna import merfish_posterior, reorder_niches
from .temporal import _align_population, read_stage


def stage_profile(x):
    counts = np.zeros(x.shape[1], dtype=float)
    depths = []
    for start in range(0, len(x), 2048):
        block = molecule_counts(x[start:start + 2048])
        counts += block.sum(axis=0)
        depths.append(block.sum(axis=1))
    return (counts + 1) / (counts.sum() + len(counts)), np.median(np.concatenate(depths))


def developmental_change(trend, genes, recent, target):
    slopes = trend.reindex(genes).fillna(0).clip(-.5, .5)
    if recent < 9.5:
        rate = slopes["late"].to_numpy()
    else:
        early, late = slopes["early"].to_numpy(), slopes["late"].to_numpy()
        # Two age intervals estimate the local derivative at their respective midpoints.
        phase = np.clip(((target + recent) / 2 - 12) / 4.5, -.5, .5)
        rate = early + phase * np.clip(late - early, -.4, .4)
    return 2 * (target - recent) * rate


def predict(earlier, later, ages, cells, seed, trend, atlas=None, niches=None, stratified_molecules=False,
            *, state_reference=None, state_strength="evidence", refine_anatomy=False, local_expression=False,
            maturation_reference=None, donor_sampler=None, match_tissue_profiles=False):
    first, recent, target = ages
    if not first < recent < target:
        raise ValueError("Expected increasing ages for both observations and the forecast")
    if (refine_anatomy or maturation_reference is not None) and atlas is None:
        raise ValueError("An anatomical atlas is required for shape refinement")
    if match_tissue_profiles and (maturation_reference is None or "dissected_max_surrounding" not in maturation_reference):
        raise ValueError("Tissue profile matching requires the tissue-annotated cardiac maturation reference")
    genes = list(np.asarray(later["genes"]).astype(str))
    x0, c0 = _align_population(earlier, genes)
    x1, c1 = _align_population(later, genes)
    # The H5AD column slice and the NPZ cache have different strides. Compute gene sums in the
    # same order so a rounding difference cannot change weighted draws and their subsequent RNG stream.
    x0, x1 = np.ascontiguousarray(x0), np.ascontiguousarray(x1)
    if not 3 <= cells <= len(x1):
        raise ValueError("The output needs at least three distinct cells from the recent stage")
    if not trend.index.is_unique or not np.isfinite(trend[["early", "late"]].to_numpy()).all():
        raise ValueError("Each reference gene needs unique, finite early and late slopes")
    _, d0 = stage_profile(x0)
    p1, d1 = stage_profile(x1)
    if min(d0, d1, rms_radius(c0), rms_radius(c1)) <= 0:
        raise ValueError("Both heart stages require positive median depth and tissue radius")
    damping = (target - recent) / (target - first)
    early_dissection = atlas is not None and recent < 9.5
    if early_dissection:
        weights = early_weights(x1, c1, genes, target - recent)
    else:
        delta = developmental_change(trend, genes, recent, target)
        if niches is not None:
            if recent < 9.5:
                raise ValueError("Use spatial RNA niches only at or after their first observed age, E9.5")
            niches = reorder_niches(niches, genes)
            states = merfish_posterior(x1, c1, niches)
            delta = .5 * delta + (target - recent) / 5 * niches["bulk"]
        weights, residual = calibrate(x1, delta, strength=.8)
        if niches is not None:
            odds = np.exp(.8 * (target - recent) / 5 * (states @ niches["mass"]))
            weights = weights * odds
            weights /= weights.sum(dtype=np.float64)
    rng = np.random.default_rng(seed)
    if maturation_reference is not None and "surrounding_retention" in maturation_reference:
        from .heart_maturation import dissection_weights
        weights = dissection_weights(weights, later["labels"], cells, maturation_reference)
    rows = (rng.choice(len(x1), cells, replace=False, p=weights) if donor_sampler is None
            else donor_sampler(later, weights, cells, rng))
    x = x1[rows]
    observed_counts = molecule_counts(x)
    relative_depth = damping * np.clip(np.log(d1 / d0), -.8, .8)
    if early_dissection:
        ratio = np.full(x.shape[1], min(1., np.exp(1.6 * relative_depth)))
    else:
        if niches is not None:
            local = states[rows] @ niches["local"]
            residual = residual + 1.5 * (target - recent) / 5 * (local - local.mean(axis=0))
        relative = np.exp(residual)
        relative /= np.sum(relative * p1, axis=-1, keepdims=True)
        ratio = relative * np.exp(.5 * relative_depth)
    if stratified_molecules:
        adjustment = donor_adjustment(x1, weights, rows, observed_counts)
        ratio = np.clip(ratio * np.exp(adjustment), .1, 3)
        result = stratified_counts(observed_counts, ratio, c1[rows], seed)
    else:
        ratio = np.clip(ratio, .1, 3)
        result = rng.binomial(observed_counts, np.minimum(ratio, 1)).astype(np.float32)
        increasing = np.any(ratio > 1, axis=0) if ratio.ndim == 2 else ratio > 1
        if np.any(increasing):
            current = observed_counts[:, increasing].astype(np.float32)
            neighbors = cKDTree(c1[rows]).query(c1[rows], k=min(12, cells))[1]
            local = np.zeros_like(current)
            for neighbor in neighbors.T:
                local += current[neighbor]
            expected = .5 * current + .5 * local / neighbors.shape[1]
            result[:, increasing] += rng.poisson(expected * np.maximum(ratio[..., increasing] - 1, 0))
    library = np.expm1(x).sum(axis=1)
    missing = (result.sum(axis=1) == 0) & (library > 0)
    result[np.flatnonzero(missing), x[missing].argmax(axis=1)] = 1
    result = np.log1p(result * (library / np.maximum(result.sum(axis=1), 1))[:, None])
    coords = (c1[rows] - c1.mean(axis=0)) * (rms_radius(c1) / rms_radius(c0)) ** damping
    coords = coords.astype(np.float32)
    if atlas is not None:
        coords = place_population(coords, c1, ages, atlas)
    prediction = {"x": result.astype(np.float32), "c": coords, "genes": np.array(genes)}
    if state_reference is not None:
        from .heart_state_bridge import transfer
        prediction = transfer(prediction, x, later["labels"][rows], later["depth"][rows],
                              ages, seed, state_reference, state_strength)
    if maturation_reference is not None:
        from .heart_maturation import mature_heart
        prediction = mature_heart(prediction, later["c"], atlas[0], recent, target, maturation_reference, seed,
                            source_labels=later.get("labels"), atlas_radius=atlas[1])
        if match_tissue_profiles:
            from .heart_maturation import surrounding_tissues
            from .measured_neighborhoods import measured_neighborhoods
            if surrounding_tissues(later["labels"]).mean() < maturation_reference["dissected_max_surrounding"]:
                from .heart_capture import project_capture
                prediction["x"] = project_capture(prediction, later, rows, seed)
            prediction["x"] = measured_neighborhoods(prediction["x"], prediction["c"], [later], rows,
                                                      np.zeros(len(rows), dtype=int), .65)
        return prediction
    if refine_anatomy:
        prediction["c"] = refine_shape(prediction["c"], recent, target, atlas[0])
    if local_expression:
        from .local_expression import place_local_profiles
        prediction["x"] = place_local_profiles(prediction["x"], prediction["c"])
    return prediction


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--earlier", required=True, type=Path)
    cli.add_argument("--later", required=True, type=Path)
    cli.add_argument("--earlier-time", required=True, type=float)
    cli.add_argument("--later-time", required=True, type=float)
    cli.add_argument("--target-time", required=True, type=float)
    cli.add_argument("--panel", required=True, type=Path)
    cli.add_argument("--trend", required=True, type=Path)
    cli.add_argument("--anatomy", type=Path, help="E14.5 organ reference containing coordinates and RMS radius")
    cli.add_argument("--refine-anatomy", action="store_true", help="Refine spatial transport to the E14.5 reference")
    cli.add_argument("--local-expression", action="store_true",
                     help="Match complete expression profiles to nearby tissue on the forecast heart geometry")
    cli.add_argument("--maturation-reference", type=Path,
                     help="Nuclear forecast for joint spatial detection and developmental shape interpolation")
    cli.add_argument("--spatial-rna", type=Path, help="Replicated E9.5/E14.5 cardiac niche model")
    cli.add_argument("--state-reference", type=Path, help="Permitted-stage wild-type single-cell state transitions")
    cli.add_argument("--state-strength", choices=("fixed", "duration", "evidence", "animal-horizon"), default="evidence",
                     help="Use a fixed, duration-weighted, variance-weighted, or animal-calibrated state change")
    cli.add_argument("--stratified-molecules", action="store_true",
                     help="Balance per-gene detection and correct for stochastic donor sampling")
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--match-tissue-profiles", action="store_true",
                     help="Match native cardiac neighborhoods and calibrate gene detection in dissected heart tissue")
    cli.add_argument("--cells", type=int, default=8192)
    cli.add_argument("--seed", type=int, default=101)
    args = cli.parse_args()
    panel = args.panel.read_text(encoding="utf-8-sig").splitlines()
    first = read_stage(args.earlier, panel)
    maturation = None
    if args.maturation_reference is not None:
        with np.load(args.maturation_reference, allow_pickle=False) as data:
            maturation = dict(data)
    state_reference = None
    physical = maturation is not None and bool(maturation.get("physical_geometry", False))
    if args.state_reference is not None or physical:
        from .embryo_interpolate import read_annotated_stage
        second = read_annotated_stage(args.later, panel, include_depth=args.state_reference is not None)
    else:
        second = read_stage(args.later, panel)
    if args.state_reference is not None:
        with np.load(args.state_reference, allow_pickle=False) as data:
            state_reference = dict(data)
    trend = pd.read_csv(args.trend, index_col=0)
    atlas = None
    if args.anatomy is not None:
        with np.load(args.anatomy, allow_pickle=False) as data:
            atlas = data["c"], data["radius"].item()
    niches = None
    if args.spatial_rna is not None:
        with np.load(args.spatial_rna, allow_pickle=False) as data:
            niches = dict(data)
    with threadpool_limits(limits=1):
        prediction = predict(first, second, (args.earlier_time, args.later_time, args.target_time),
                             args.cells, args.seed, trend, atlas, niches, args.stratified_molecules,
                             state_reference=state_reference, state_strength=args.state_strength,
                             refine_anatomy=args.refine_anatomy, local_expression=args.local_expression,
                             maturation_reference=maturation, match_tissue_profiles=args.match_tissue_profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **prediction)


if __name__ == "__main__":
    main()
