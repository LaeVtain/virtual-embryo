"""Predict a T3 perturbation from an aligned WT spatial population."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..paths import task_assets
from ..population import align_population
from . import atlas_response, runtime
from .mixture import predict_mixture


TRANSFER_NAMES = ("bcatenin", "Ctnnb1", "bcatenin-v1", "bcatenin-v2")


def predict(x, c, genes, condition, seed=101, model_dir=None):
    folder = Path(model_dir) if model_dir is not None else task_assets("t3")
    if condition in TRANSFER_NAMES:
        use_v2 = condition == "bcatenin-v2" or (
            condition != "bcatenin-v1" and (folder / "bcatenin-v2.npz").is_file()
        )
        model_path = folder / ("bcatenin-v2.npz" if use_v2 else "bcatenin.npz")
        # PCA and the paired assignment must use the same reduction order as
        # the validated spatial files when several costs are nearly tied.
        with np.load(model_path, allow_pickle=False) as model, threadpool_limits(limits=1):
            expected = model["genes"].astype(str)
            aligned = align_population({"x": x, "genes": np.asarray(genes)}, expected)
            if use_v2:
                result = atlas_response.predict(aligned["x"], c, dict(model), seed)
            else:
                result = predict_mixture(aligned["x"], c, model["bank"], seed,
                                         signature=model["signature"], sensitivity=0.5)
        result["genes"] = expected
        return result
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
    parser.add_argument("--condition", choices=(*runtime.CONDITIONS, *TRANSFER_NAMES), required=True)
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
