"""T3 known-condition prediction from a complete aligned WT gene panel."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..paths import task_assets
from . import runtime


def predict(x, c, genes, condition, seed=101, model_dir=None):
    folder = Path(model_dir) if model_dir is not None else task_assets("t3")
    co = runtime.CONDITIONS[condition]
    with np.load(folder / "tilt.npz") as z:
        tilt = dict(z)
    expected = tilt["genes"].astype(str)
    supplied = list(map(str, genes))
    order = [supplied.index(g) for g in expected]
    x = np.asarray(x, dtype=float)[:, order]
    c = np.asarray(c, dtype=float)
    with np.load(folder / f"{co}.npz") as z:
        model = dict(z)
    result = runtime.predict(x, c, tilt, co, model, seed)
    result["genes"] = expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=runtime.CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model-dir", type=Path, default=task_assets("t3"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    with np.load(args.input) as z:
        result = predict(z["x"], z["c"], z["genes"], args.condition, args.seed, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(json.dumps(dict(output=str(args.output), cells=len(result["x"]), seconds=time.perf_counter() - start)))


if __name__ == "__main__":
    main()
