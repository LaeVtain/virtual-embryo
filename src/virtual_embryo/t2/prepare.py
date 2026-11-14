"""Prepare three explicitly supplied historical training populations."""
from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.decomposition import PCA
import torch
from threadpoolctl import threadpool_limits

from ..paths import task_assets
from . import losses

PAIRS = ((0, 1), (1, 2), (0, 2))


def prepare(setting, stage_paths, output, model_dir=None):
    """Each stage is an ordered list of NPZ blocks with x/c/rows/genes."""
    folder = Path(model_dir) if model_dir is not None else task_assets("t2")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))[setting]
    genes = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))[f"T2_{setting}"]["genes"]
    if len(stage_paths) != 3:
        raise ValueError("Provide the three historical stages in chronological order")
    stages, identities, rows_by_stage = [], [], []
    for paths in stage_paths:
        blocks = []
        for path in map(Path, paths):
            with np.load(path, allow_pickle=False) as archive:
                block = {key: archive[key] for key in ("x", "c", "rows", "genes")}
            if block["genes"].astype(str).tolist() != genes:
                raise ValueError(f"Training panel order differs: {path}")
            blocks.append(block)
            identities.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        x, c, rows = [np.concatenate([block[key] for block in blocks]) for key in ("x", "c", "rows")]
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Repeated source rows in one training stage")
        c = c.astype("float32")
        c = (c - c.mean(0)) / config["coordinate_unit"]
        stages.append((x.astype("float32"), c))
        rows_by_stage.append(rows.tolist())
    pca = PCA(n_components=30, random_state=41).fit(np.vstack([x for x, c in stages]))
    codes = [pca.transform(x) for x, c in stages]
    plans = {}
    for i, j in PAIRS:
        distance = cdist(codes[i], codes[j], "sqeuclidean")
        plans[i, j] = softmax(-distance / max(.1 * np.median(distance), 1e-6), axis=1).astype("float32")
    stats = {j: losses.fit_statistics(stages[j-1][0], stages[j][0],
              reference_coords=stages[j-1][1], target_coords=stages[j][1]) for j in (1, 2)}
    tau = float(np.quantile(np.concatenate([x[x > 0] for x, c in stages]), .01))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(stages=stages, plans=plans, stats=stats, tau=tau, genes=np.asarray(genes)), output)
    manifest = dict(inputs=identities, rows=rows_by_stage, unit=config["coordinate_unit"], tau=tau,
                    prepared_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("heart", "embryo"), required=True)
    parser.add_argument("--stage", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        prepare(args.setting, args.stage, args.output, args.model_dir)


if __name__ == "__main__":
    main()
