"""Restore the depth distribution and spacing of physical sections in a registered embryo."""

import numpy as np
from scipy.stats import rankdata


def _section_axis(coordinates, measured_depth):
    centered = coordinates - coordinates.mean(0)
    depth = measured_depth - measured_depth.mean()
    gradient = np.linalg.lstsq(centered, depth, rcond=None)[0]
    axis = gradient / max(1e-9, np.linalg.norm(gradient))

    section = np.unique(measured_depth, return_inverse=True)[1].ravel()
    squared_error = 0.0
    for fold in range(3):
        train = section % 3 != fold
        held = ~train
        if held.any():
            design = np.column_stack([np.ones(train.sum()), coordinates[train]])
            coefficients = np.linalg.lstsq(design, measured_depth[train], rcond=None)[0]
            residual = measured_depth[held] - coefficients[0] - coordinates[held] @ coefficients[1:]
            squared_error += residual @ residual
    reliability = np.clip(1 - squared_error / max(depth @ depth, 1e-9), 0, 1)
    return axis, reliability


def _continuous_depth(depth, ranks):
    levels, counts = np.unique(np.round(depth, 2), return_counts=True)
    probabilities = (np.cumsum(counts)-.5*counts) / counts.sum()
    before = levels[0]-.5*(levels[1]-levels[0])
    after = levels[-1]+.5*(levels[-1]-levels[-2])
    return np.interp(ranks, np.r_[0., probabilities, 1.], np.r_[before, levels, after])


def restore_sections(stages, target, coordinates, donor_rows, from_later, *, continuous=False):
    """Estimate a target slice stack from measured source sections and the registered donor positions."""
    left = max(age for age in stages if age < target)
    right = min(age for age in stages if age > target)
    fraction = (target - left) / (right - left)
    coordinates = coordinates.astype(float)
    center = coordinates.mean(0)
    centered = coordinates - center
    depths, axes, confidence, spacing = [], [], [], []
    for age, cells in ((left, ~from_later), (right, from_later)):
        depth = stages[age]["c"][:, 2].astype(float)
        sampled = depth[donor_rows[cells]]
        if len(np.unique(sampled)) < 3:
            raise ValueError("Section reconstruction requires at least three donor sections in each source")
        axis, reliability = _section_axis(coordinates[cells], sampled)
        depth -= depth.mean()
        gaps = np.diff(np.unique(np.round(depth, 2)))
        # The lower half of measured gaps preserves the pitch in a stack with missing sections.
        spacing.append(np.median(gaps[gaps <= np.median(gaps)]))
        depths.append(depth)
        axes.append(axis)
        confidence.append(reliability)
    if axes[0] @ axes[1] < 0:
        axes[1] *= -1
        depths[1] *= -1
    weight = np.array([1 - fraction, fraction]) * np.maximum(confidence, 1e-6)
    normal = np.average(axes, axis=0, weights=weight)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        raise ValueError("A registered donor depth gradient is required to reconstruct the section stack")
    normal /= magnitude
    projection = centered @ normal
    rank = (rankdata(projection) - 0.5) / len(projection)
    if continuous:
        quantiles = [_continuous_depth(depth, rank) for depth in depths]
    else:
        quantiles = [np.interp(rank, np.linspace(0, 1, len(depth)), np.sort(depth)) for depth in depths]
    target_depth = (1 - fraction) * quantiles[0] + fraction * quantiles[1]
    target_depth -= target_depth.mean()
    target_depth *= projection.std() / max(1e-8, target_depth.std())
    pitch = np.exp((1 - fraction) * np.log(spacing[0]) + fraction * np.log(spacing[1]))
    planes = pitch * np.rint((target_depth - np.median(target_depth)) / pitch)
    planes -= planes.mean()
    return (center + centered + (planes - projection)[:, None] * normal).astype(np.float32)
