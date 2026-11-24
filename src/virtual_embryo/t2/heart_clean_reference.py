"""Build a cardiac maturation reference from measured source and permitted E14.5 nuclear RNA."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from .heart_maturation import surrounding_tissues


TISSUES = ("ventricle", "inflow", "endothelium", "surface", "mesoderm", "crest", "foregut", "hepatic",
           "ectoderm", "neural", "extraembryonic", "blood", "other")


def tissue_group(label):
    name = str(label).lower().strip()
    if name.endswith("-cm") or "cardiomyocyte" in name:
        return "inflow" if name.startswith(("a-", "sv-", "ift-", "avc-", "atrial")) else "ventricle"
    if ("endoth" in name or "endocard" in name or name in ("endo", "bec", "early-vec", "vic", "early vic")
            or name.endswith("-endo") or "valve interstitial" in name):
        return "endothelium"
    if name in ("peri", "proepi", "epi", "pericardium", "epicardium", "proepicardium"):
        return "surface"
    if ("phm" in name or "pam" in name or "shf" in name or "mesoderm" in name
            or name in ("jcf", "st", "dmp", "lpm", "fhf", "shf", "cp", "bw")):
        return "mesoderm"
    if "ncc" in name or name == "branch arch":
        return "crest"
    if "hepato" in name:
        return "hepatic"
    if "gut" in name or "fg" in name or name in ("bronchus", "endoderm"):
        return "foregut"
    if name.startswith("exe") or name in ("allantois", "p-exem"):
        return "extraembryonic"
    if "neural" in name or "brain" in name:
        return "neural"
    if "ectoderm" in name or name in ("se", "pse", "d-cse", "v-cse", "epithelium"):
        return "ectoderm"
    if name in ("blood", "macrophage", "erythrocytes", "erythroid"):
        return "blood"
    return "other"


def encode_tissue(labels):
    values, inverse = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    groups = np.array([TISSUES.index(tissue_group(label)) for label in values], dtype=int)
    return groups[inverse]


def expression_features(x):
    raw = np.expm1(x)
    return np.log1p(1000. * raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.))


def labeled_means(x, labels, detection=False):
    sizes = np.bincount(labels, minlength=len(TISSUES))
    background = (np.count_nonzero(x, axis=0) / len(x) if detection else x.mean(axis=0, dtype=float))
    means = np.empty((len(TISSUES), x.shape[1]), dtype=float)
    for group, count in enumerate(sizes):
        rows = x[labels == group]
        totals = np.count_nonzero(rows, axis=0) if detection else rows.sum(axis=0, dtype=float)
        means[group] = (totals + 24 * background) / (count + 24)
    return means, (sizes + 2) / (len(labels) + 2 * len(sizes))


def endpoint_means(source, source_labels, late):
    groups = encode_tissue(source_labels)
    model = LogisticRegression(C=.16, max_iter=450, tol=.0003, solver="lbfgs")
    model.fit(expression_features(source), groups)
    weights = np.zeros((len(late), len(TISSUES)))
    for start in range(0, len(late), 2048):
        part = late[start:start + 2048]
        weights[start:start + len(part), model.classes_] = model.predict_proba(expression_features(part))
    terminal = (weights.T @ late + 24 * late.mean(axis=0, dtype=float)) / (weights.sum(0)[:, None] + 24)
    initial, _ = labeled_means(source, groups)
    return initial, terminal


def clean_reference(genes, source, late, observed, source_time, target_time, native, native_labels,
                    conditional_means=None):
    """Use official source RNA/spatial labels and the single E14.5 heart specimen for both spatial branches."""
    if not source_time < target_time < 14.5 or source.shape[1] != late.shape[1] or source.shape[1] != len(genes):
        raise ValueError("The measured source, target time, and E14.5 panel must bracket the prediction")
    measured = np.asarray(observed, dtype=bool)
    fraction = (target_time - source_time) / (14.5 - source_time)
    initial = source.mean(axis=0, dtype=float)
    terminal = late.mean(axis=0, dtype=float)
    global_change = np.log1p(fraction * (terminal - initial) / (initial + .012))
    global_change = np.where(measured, np.clip(global_change, -2, 2), 0.)
    tissue_change = global_change
    if surrounding_tissues(native_labels).mean() < .05:
        if conditional_means is None:
            raise ValueError("The dissected heart needs measured, tissue-conditioned endpoints")
        young, old = conditional_means
        shift = np.clip(np.log1p(fraction * (old - young) / (young + .012)), -2., 2.)
        detection, mass = labeled_means(native, encode_tissue(native_labels), detection=True)
        detection = np.clip(detection, .0005, .9995)
        mature = expit(logit(detection) + shift)
        tissue_change = logit(np.clip(mass @ mature, .0005, .9995)) - logit(np.clip(mass @ detection, .0005, .9995))
        tissue_change = np.where(measured, np.clip(tissue_change, -2., 2.), 0.)
    return dict(genes=np.asarray(genes, dtype=str), change=global_change, dissected_change=tissue_change,
                source_time=source_time, target_time=target_time, physical_geometry=True,
                dissected_max_surrounding=.05, surrounding_retention=.1)
