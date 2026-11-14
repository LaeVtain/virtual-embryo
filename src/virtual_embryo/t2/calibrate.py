"""Fit the three frozen 257-quantile expression maps on training rows."""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..paths import task_assets
from .expression import latent
from .networks import ConditionalEndpoint
from .prepare import PAIRS


def fit(z, target):
    quantiles = np.linspace(0, 1, 257)
    return (np.quantile(z, quantiles, axis=0).astype("float32"),
            np.quantile(target, quantiles, axis=0).astype("float32"))


def calibrate(setting, prepared, output, model_dir=None):
    folder = Path(model_dir) if model_dir is not None else task_assets("t2")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))[setting]
    data = joblib.load(prepared)
    network = ConditionalEndpoint(len(data["genes"])).eval().requires_grad_(False)
    checkpoint = torch.load(folder / f"{setting}_control.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["expression"])
    maps = {}
    for i, j in PAIRS:
        # Prepared data from the original mature run used dictionaries per stage.
        source = data["stages"][i]
        target = data["stages"][j]
        x = source["x"] if isinstance(source, dict) else source[0]
        y = target["x"] if isinstance(target, dict) else target[0]
        z = np.concatenate([latent(network, x, config["times"][i], config["times"][j],
                                   config["times"], seed) for seed in (201, 202)])
        maps[i, j] = fit(z, y)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(maps, output)
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("heart", "embryo"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        calibrate(args.setting, args.prepared, args.output, args.model_dir)


if __name__ == "__main__":
    main()
