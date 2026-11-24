"""Convert raw metrics to board points using the supplied specimen's anchors."""
import math

from common import core_metrics as core


WEIGHTS = {
    "T1": {"de_score": .25, "de_direction": .25, "mmd_u": .30, "variogram": .20},
    "T2": {"de_score": .125, "de_direction": .125, "mmd_u": .15, "variogram": .10,
           "d2_shape": 1 / 12, "occupancy_dice": 1 / 12, "scale_log_ratio": 1 / 12,
           "neighborhood_mmd": .25},
    "T3": {"de_score": .30, "de_direction": .25, "severity_slope": .25,
           "mmd_u": .12, "variogram": .08},
}
HIGHER_IS_BETTER = {"de_score", "de_direction", "occupancy_dice"}
ABSOLUTE_DISTANCE = {"scale_log_ratio", "severity_slope"}
RAW_ALIASES = {"mmd_u": "mmd_rbf", "d2_shape": "d2_distance"}


def calibrated_score(task, metrics, anchors):
    skills = {}
    for name in WEIGHTS[task]:
        alias = RAW_ALIASES.get(name, name)
        anchor = anchors[name] if name in anchors else anchors[alias]
        floor, ceiling = float(anchor["floor"]), float(anchor["ceiling"])
        value = metrics[name] if name in metrics else metrics[alias]
        if name in ABSOLUTE_DISTANCE:
            floor, ceiling = abs(floor), abs(ceiling)
            value = abs(value) if value is not None else None
        if not (math.isfinite(floor) and math.isfinite(ceiling)) or abs(ceiling - floor) < 1e-12:
            raise ValueError(f"Expected distinct finite anchors for {name}")
        if value is None or not math.isfinite(value):
            skills[name] = 0.
        else:
            skills[name] = core.skill(value, floor, ceiling, lower_is_better=name not in HIGHER_IS_BETTER)
    points = {name: 100 * weight * skills[name] for name, weight in WEIGHTS[task].items()}
    if task == "T2":
        alignment = min(1., 2 * skills["neighborhood_mmd"])
        for name in ("d2_shape", "occupancy_dice", "scale_log_ratio"):
            points[name] *= alignment
    return {"total": sum(points.values()), "skills": skills, "credited_points": points}
