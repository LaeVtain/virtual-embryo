"""Report all ranking metrics as raw values using the pinned challenge scorer."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import core_metrics as core, shape_metrics as shape
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .population import align_population, load_population


def neighborhood_mmd(pred_x, pred_c, true_x, true_c):
    def neighborhood(x, c):
        indices = NearestNeighbors(n_neighbors=15).fit(c).kneighbors(c)[1]
        return x[indices].mean(1)
    return core.mmd_unbiased(neighborhood(pred_x, pred_c), neighborhood(true_x, true_c))


def score(task, prediction, truth, reference):
    genes = truth["genes"].astype(str)
    pred = align_population(prediction, genes)
    true = align_population(truth, genes)
    ref = align_population(reference, genes)
    px, tx, rx = pred["x"], true["x"], ref["x"]
    de = core.de_score(px, tx, rx)
    metrics = {"de_score": de["score"], "de_direction": core.de_direction(px, tx, rx),
               "mmd_rbf": core.mmd_unbiased(px, tx), "variogram": core.variogram_score(px, tx)}
    diagnostics = {"de_genes_up": de["n_up"], "de_genes_down": de["n_dn"]}
    if task == "T2":
        pc, tc = pred["c"], true["c"]
        metrics.update(d2_distance=shape.d2_distance(pc, tc),
                       occupancy_dice=shape.occupancy_dice(pc, tc)[0],
                       scale_log_ratio=shape.scale_log_ratio(pc, tc),
                       neighborhood_mmd=neighborhood_mmd(px, pc, tx, tc))
    elif task == "T3":
        metrics["severity_slope"], diagnostics["severity_r2"] = core.severity_slope(px, tx, rx)
    elif task != "T1":
        raise ValueError(f"Unknown task: {task}")
    # Undefined DE metrics (no DE genes in the supplied truth) are represented as JSON null.
    clean = lambda values: {k: float(v) if np.isfinite(v) else None for k, v in values.items()}
    return {"task": task, "scorer": "veckit==0.1.1", "scale": "raw",
            "cells": {"prediction": len(px), "truth": len(tx), "reference": len(rx)},
            "genes": len(genes), "metrics": clean(metrics), "diagnostics": clean(diagnostics)}


def evaluation_populations(paths, layer, max_cells, seed):
    # A different prediction size must not change the truth or reference sample.
    # If the prediction is the reference itself, use the same sampled cells.
    prediction, truth, reference = (Path(path).resolve() for path in paths)
    loaded = {}
    seeds = np.random.SeedSequence(seed).spawn(3)
    for path, stream in zip((reference, truth, prediction), seeds):
        if path not in loaded:
            loaded[path] = load_population(path, layer, max_cells=max_cells, seed=stream)
    return [loaded[path] for path in (prediction, truth, reference)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("T1", "T2", "T3"), required=True)
    for name in ("prediction", "truth", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layer", help="H5AD expression layer; default X")
    parser.add_argument("--max-cells", type=int, help="Optional independent subsample limit for each population")
    parser.add_argument("--seed", type=int, default=0, help="Cell-subsampling seed")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    populations = evaluation_populations((args.prediction, args.truth, args.reference),
                                         args.layer, args.max_cells, args.seed)
    with threadpool_limits(limits=args.threads):
        result = score(args.task, *populations)
    result.update(max_cells=args.max_cells, subsample_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
