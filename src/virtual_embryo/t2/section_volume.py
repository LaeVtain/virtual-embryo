"""Calibrate whole-specimen radius from the tissue area on serial physical sections."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay, cKDTree


def tissue_volume(coords):
    levels, sections = np.unique(np.round(coords[:, 2], 2), return_inverse=True)
    if len(levels) < 3 or len(levels) > len(coords) / 4:
        raise ValueError("Section volume requires at least three physical sections with repeated z coordinates")
    areas = np.zeros(len(levels))
    for index in range(len(levels)):
        points = np.unique(coords[sections == index, :2].astype(float), axis=0)
        if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
            continue
        triangles = points[Delaunay(points).simplices]
        edges = triangles - np.roll(triangles, 1, axis=1)
        spacing = np.median(cKDTree(points).query(points, k=4)[0][:, -1])
        short = np.linalg.norm(edges, axis=2).max(axis=1) <= 2 * spacing
        cross = edges[:, 0, 0]*edges[:, 1, 1] - edges[:, 0, 1]*edges[:, 1, 0]
        areas[index] = np.abs(cross[short]).sum() / 2
    gaps = np.diff(levels)
    nominal = np.median(gaps)
    thickness = (np.r_[nominal, gaps] + np.r_[gaps, nominal]) / 2
    volume = float(areas @ thickness)
    if volume <= 0:
        raise ValueError("The physical sections contain no measurable tissue area")
    return volume


def adjust_radius(stages, target, coords):
    ages = sorted(stages)
    left = max(i for i, age in enumerate(ages) if age < target)
    right = left + 1
    fraction = (target - ages[left]) / (ages[right] - ages[left])
    log_volumes = np.log([tissue_volume(stages[age]["c"]) for age in ages])
    log_radii = np.log([np.linalg.norm(stages[age]["c"].std(0, dtype=float)) for age in ages])
    # The ratio R/V^(1/3) carries the observed change in elongation and tissue occupancy.
    log_shape = log_radii - log_volumes / 3
    target_shape = (1-fraction)*log_shape[left] + fraction*log_shape[right]
    radius = np.exp(PchipInterpolator(ages, log_volumes)(target)/3 + target_shape)
    center = coords.mean(axis=0, dtype=float)
    factor = radius / np.linalg.norm(coords.std(0, dtype=float))
    return ((coords-center)*factor + center).astype(np.float32)
