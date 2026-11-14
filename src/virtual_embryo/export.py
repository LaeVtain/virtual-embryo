"""Convert a T1 prediction directory or T2/T3 NPZ prediction into challenge H5AD."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .population import load_population


def export_h5ad(source, destination):
    pop = load_population(source)
    x = np.asarray(pop["x"], dtype=np.float32)
    genes = pop["genes"].astype(str)
    if x.ndim != 2 or len(genes) != x.shape[1] or len(set(genes)) != len(genes):
        raise ValueError("Prediction must have one unique gene name per expression column")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Prediction expression must be finite and nonnegative")
    result = ad.AnnData(sparse.csr_matrix(x),
                        obs=pd.DataFrame(index=[f"pred_{i}" for i in range(len(x))]),
                        var=pd.DataFrame(index=genes))
    if "c" in pop:
        coordinates = np.asarray(pop["c"])
        if coordinates.ndim != 2 or len(coordinates) != len(x) or not np.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates must be finite with one row per predicted cell")
        result.obsm["spatial_3D"] = coordinates.astype(np.float32)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression="gzip")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_h5ad(args.input, args.output))


if __name__ == "__main__":
    main()
