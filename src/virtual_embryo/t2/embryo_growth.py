"""Estimate intact embryo radius from anatomical growth and measured spatial source stages."""
import numpy as np
from scipy.optimize import least_squares

from .section_volume import tissue_volume


def physical_lengths(coordinates):
    centered = coordinates - coordinates.mean(0, dtype=float)
    variances = np.linalg.eigvalsh(centered.T @ centered / len(centered))
    distances = np.linalg.norm(centered, axis=1)
    return np.r_[np.sqrt(variances.sum()), np.sqrt(variances), np.quantile(distances, [.5, .8, .95])]


def _gompertz(times, center, log_rate):
    scaled = np.clip(np.exp(log_rate) * (np.asarray(times)-center), -30., 30.)
    return np.exp(-np.exp(-scaled))


def _fit_clock(ages, measurements, prior, regularization):
    features = (measurements-measurements.mean(0)) / np.maximum(np.ptp(measurements, axis=0), 1e-8)
    penalty = np.sqrt(regularization * len(ages) * features.shape[1])
    def residual(parameters):
        maturity = _gompertz(ages, *parameters)
        centered = maturity - maturity.mean()
        fitted = centered[:, None] * (centered @ features / max(centered @ centered, 1e-10))
        return np.r_[(features-fitted).ravel(), penalty*(parameters[0]-prior[0]),
                     .4*penalty*(parameters[1]-prior[1])]
    return least_squares(residual, prior, bounds=([5.75, np.log(.6)], [9., np.log(14.)])).x


def project_radius(stages, target, coordinates, reference):
    ages = np.array(sorted(stages))
    if target in stages or not ages[0] < target < ages[-1]:
        raise ValueError("The growth model requires a masked day inside the observed time interval")
    ref_ages = np.asarray(reference["ages"])
    if ((ref_ages > 7.25) & (ref_ages < 8.)).any():
        raise ValueError("The anatomical reference contains a protected external embryo age")
    keep = ref_ages != target
    if keep.sum() < 3:
        raise ValueError("At least three independent reference ages are needed to fit the growth prior")
    prior = _fit_clock(ref_ages[keep], np.asarray(reference["lengths"])[keep], [7.5, np.log(4.)], .008)
    lengths = np.stack([np.r_[physical_lengths(stages[day]["c"]), tissue_volume(stages[day]["c"])**(1/3)]
                        for day in ages])
    parameters = _fit_clock(ages, lengths, prior, .012)
    i = np.searchsorted(ages, target) - 1
    lo, hi, at_target = _gompertz([ages[i], ages[i+1], target], *parameters)
    fraction = np.clip((at_target-lo) / max(1e-9, hi-lo), 0., 1.)
    radius = (1-fraction)*lengths[i, 0] + fraction*lengths[i+1, 0]
    centered = coordinates-coordinates.mean(0, dtype=float)
    return (coordinates + (radius/physical_lengths(coordinates)[0]-1)*centered).astype(np.float32)
