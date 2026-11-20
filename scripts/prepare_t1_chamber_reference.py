"""Build gene-change and lineage-abundance priors from the two WT CD1 chamber samples."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from virtual_embryo.t1.lineage_response import marker_classes, mean


def summarize(counts, observations, genes):
    x = counts.tocsr().astype(np.float32)
    depth = np.asarray(x.sum(1)).ravel()
    x.data = np.log1p(x.data * np.repeat(10000 / depth, np.diff(x.indptr)))
    labels = marker_classes(x, genes)
    chamber = np.where(observations.zone.isin(["LA", "RA"]), "atrium", "ventricle")
    measurements, proportions = {}, {}
    for zone in ["atrium", "ventricle"]:
        area = chamber == zone
        for kind in np.unique(labels):
            selected = area & (labels == kind)
            n = selected.sum()
            proportions[zone, kind] = n / area.sum()
            if n < 12:
                continue
            cells = x[selected]
            mu = mean(cells)
            variance = np.maximum(0, mean(cells.power(2)) - mu**2)
            measurements[zone, kind] = (mu, variance / n)
    return measurements, proportions


def prepare(path, provenance, output):
    source = json.loads(provenance.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        if source["strain"] != "CD1" or hashlib.file_digest(stream, "sha256").hexdigest() != source["output_sha256"]:
            raise ValueError("Provide the CD1 count file and its matching source record.")
    atlas = ad.read_h5ad(path)
    stats, population, sample_sizes = {}, {}, {}
    for stage in ("E9.5", "E14.5"):
        rows = np.flatnonzero(atlas.obs.stage == stage)
        if not len(rows):
            raise ValueError(f"Missing cardiac chamber measurements at {stage}")
        stats[stage], population[stage] = summarize(atlas.X[rows], atlas.obs.iloc[rows], atlas.var_names)
        sample_sizes[stage] = len(rows)
    changes, abundance = {}, {}
    before, after = stats["E9.5"], stats["E14.5"]
    for zone, kind in sorted(before.keys() & after.keys()):
        old, v0 = before[zone, kind]
        new, v1 = after[zone, kind]
        effect = np.log((new + 0.05) / (old + 0.05))
        noise = v0 / (old + 0.05)**2 + v1 / (new + 0.05)**2
        changes.setdefault(kind, []).append((effect, noise))
    p0, p1 = population["E9.5"], population["E14.5"]
    for zone, kind in sorted(p0.keys() & p1.keys()):
        abundance.setdefault(kind, []).append(np.log((p1[zone, kind] + 0.02) / (p0[zone, kind] + 0.02)))
    labels = sorted(changes)
    effects, weights = [], []
    for kind in labels:
        change, noise = np.array(changes[kind]).transpose(1, 0, 2)
        effect = change.mean(axis=0)
        uncertainty = noise.mean(axis=0) + change.var(axis=0) + 0.025
        effects.append(effect)
        weights.append(effect**2 / (effect**2 + uncertainty))
    keys = sorted(abundance)
    output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(output, genes=atlas.var_names.to_numpy(dtype=str), times=[9.5, 14.5],
                        lineages=np.array(labels), log_change=effects, reliability=weights,
                        composition_lineages=np.array(keys), composition_log_change=[np.median(abundance[k]) for k in keys])
    citation = dict(series="GSE193346", study="https://doi.org/10.1038/s41467-022-35691-7", strain="CD1",
                    raw_data=source["raw_url"], compressed_prefix_sha256=source["compressed_prefix_sha256"],
                    compressed_prefix_end=source["compressed_prefix_end"], metadata_url=source["metadata_url"],
                    metadata_sha256=source["metadata_sha256"], stages=sample_sizes, genes=len(atlas.var_names),
                    raw_field="assays/RNA/counts", grouping="stage, chamber and per-cell marker expression",
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".sources.json").write_text(json.dumps(citation, indent=2), encoding="utf-8")
    print(json.dumps(citation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.counts, args.sources, args.output)
