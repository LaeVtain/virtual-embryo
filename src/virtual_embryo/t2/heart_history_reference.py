"""Transfer the observed change between two official RNA stages into the native cardiac dissection."""
import numpy as np
from scipy.special import expit, logit

from .heart_clean_reference import TISSUES, encode_tissue


def exposure_rates(expression, tissue_labels):
    labels = encode_tissue(tissue_labels)
    totals = np.zeros((len(TISSUES), expression.shape[1]))
    count = np.bincount(labels, minlength=len(TISSUES))
    for start in range(0, len(labels), 2048):
        block = expression[start:start + 2048]
        raw = np.expm1(block)
        depth = raw.sum(1)
        nonempty = depth > 0
        relative = raw[nonempty] / depth[nonempty, None]
        detection = -np.expm1(32 * np.log1p(-np.minimum(relative, .999999)))
        groups = labels[start:start + len(block)][nonempty]
        for group in np.unique(groups):
            totals[group] += detection[groups == group].sum(axis=0, dtype=float)
    global_rate = (totals.sum(0) + .5) / (count.sum() + 1)
    return (totals + 12 * global_rate + .25) / (count[:, None] + 12.5), count


def measured_history(previous_spatial, current_spatial, previous_rna, current_rna, elapsed_days):
    _, previous_native_n = previous_spatial
    current_native, current_native_n = current_spatial
    previous, previous_rna_n = previous_rna
    current, current_rna_n = current_rna
    available = np.minimum.reduce([previous_native_n, current_native_n, previous_rna_n, current_rna_n]) >= 25
    if not available.any() or elapsed_days <= 0:
        raise ValueError("Time-ordered source stages need shared, well-represented tissue labels")
    start = np.clip(previous[available], .0005, .9995)
    end = np.clip(current[available], .0005, .9995)
    shift = logit(end) - logit(start)
    shift -= np.median(shift, axis=1, keepdims=True)
    native = current_native[available]
    mass = current_native_n[available].astype(float)
    mass /= mass.sum()
    moving = expit(logit(native) + np.clip(shift, -2, 2))
    return (logit(mass @ moving) - logit(mass @ native)) / elapsed_days


def with_measured_history(reference, previous_spatial, current_spatial, previous_rna, current_rna, elapsed_days):
    rate = measured_history(previous_spatial, current_spatial, previous_rna, current_rna, elapsed_days)
    # The production spatial layer applies its stored shift at 0.35 times the forecast horizon.
    # Express the added 0.30 daily change on the same scale without touching the whole-field branch.
    return dict(reference, dissected_change=reference["dissected_change"] + (.30 / .35) * rate)
