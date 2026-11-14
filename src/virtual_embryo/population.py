"""Population files shared by prediction export and local evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def load_population(path, layer=None):
    path = Path(path)
    if path.is_dir():
        genes = json.loads((path / "genes.json").read_text(encoding="utf-8"))
        return {"x": np.load(path / "candidate.npy", mmap_mode="r"), "genes": np.asarray(genes)}
    if path.suffix == ".h5ad":
        import anndata as ad
        data = ad.read_h5ad(path)
        x = data.X if layer is None else data.layers[layer]
        result = {"x": x.toarray() if sparse.issparse(x) else np.asarray(x),
                  "genes": data.var_names.to_numpy(dtype=str)}
        if "spatial_3D" in data.obsm:
            result["c"] = np.asarray(data.obsm["spatial_3D"])
        elif "spatial" in data.obsm:
            result["c"] = np.asarray(data.obsm["spatial"])
        return result
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("x", "genes", "c") if key in data}


def align_population(population, genes):
    supplied = population["genes"].astype(str).tolist()
    if len(supplied) != population["x"].shape[1] or len(set(supplied)) != len(supplied):
        raise ValueError("Each expression column must have one unique gene name")
    columns = {gene: i for i, gene in enumerate(supplied)}
    indices = [columns[str(gene)] for gene in genes]
    return {**population, "x": np.asarray(population["x"][:, indices]), "genes": np.asarray(genes)}
