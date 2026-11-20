"""Compare molecular expression at stages sharing tissue identity and capture depth."""
import numpy as np

from .temporal import _align_population


def aligned_stage_means(stages, genes):
    """Average log expression in the same tissue and molecule-depth strata at every age."""
    genes = list(genes)
    aligned = []
    for age in sorted(stages):
        stage = stages[age]
        x, _ = _align_population(stage, genes)
        depth = np.asarray(stage["depth"]) if "depth" in stage else np.asarray(stage["counts"].sum(axis=1)).ravel()
        labels = np.asarray(stage["labels"]).astype(str)
        aligned.append((x, depth, labels))
    return matched_profile_means(aligned)


def matched_profile_means(aligned):
    """Average a common expression panel across equivalent tissue and molecule-depth groups."""
    profiles = []
    for x, depth, labels in aligned:
        if depth.shape != (len(x),) or labels.shape != (len(x),):
            raise ValueError("Each cell needs its panel molecule count and tissue label")
        bands = np.digitize(depth, [16, 32, 64, 128, 256])
        strata = {}
        for band in np.unique(bands):
            active = bands == band
            for label in np.unique(labels[active]):
                if label == "Unknown":
                    continue
                mask = active & (labels == label)
                n = int(mask.sum())
                if n >= 20:
                    strata[(band, label)] = n / len(x), x[mask].mean(0, dtype=np.float64)
        profiles.append(strata)
    shared = sorted(set.intersection(*(set(profile) for profile in profiles)))
    if not shared:
        raise ValueError("Depth matching needs a shared tissue and molecule-depth band with 20 cells in each stage")
    weights = np.min([[profile[group][0] for group in shared] for profile in profiles], axis=0)
    weights /= weights.sum()
    return np.stack([sum(weight * profile[group][1] for weight, group in zip(weights, shared))
                     for profile in profiles])
