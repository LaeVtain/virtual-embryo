"""Population files shared by prediction export and local evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def load_population(path, layer=None, *, max_cells=None, seed=None):
    path = Path(path)
    if path.is_dir():
        genes = json.loads((path / "genes.json").read_text(encoding="utf-8"))
        population = {"x": np.load(path / "candidate.npy", mmap_mode="r"), "genes": np.asarray(genes)}
    elif path.suffix == ".h5ad":
        import anndata as ad
        data = ad.read_h5ad(path)
        x = data.X if layer is None else data.layers[layer]
        population = {"x": x, "genes": data.var_names.to_numpy(dtype=str)}
        if "spatial_3D" in data.obsm:
            population["c"] = np.asarray(data.obsm["spatial_3D"])
        elif "spatial" in data.obsm:
            population["c"] = np.asarray(data.obsm["spatial"])
    else:
        with np.load(path, allow_pickle=False) as data:
            population = {key: data[key] for key in ("x", "genes", "c") if key in data}

    n_cells = population["x"].shape[0]
    if max_cells is not None:
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        if n_cells > max_cells:
            rows = np.random.default_rng(seed).choice(n_cells, max_cells, replace=False)
            population["x"] = population["x"][rows]
            if "c" in population:
                population["c"] = population["c"][rows]
    if sparse.issparse(population["x"]):
        population["x"] = population["x"].toarray()
    return population


def align_population(population, genes):
    supplied = population["genes"].astype(str).tolist()
    if len(supplied) != population["x"].shape[1] or len(set(supplied)) != len(supplied):
        raise ValueError("Each expression column must have one unique gene name")
    columns = {gene: i for i, gene in enumerate(supplied)}
    indices = [columns[str(gene)] for gene in genes]
    return {**population, "x": np.asarray(population["x"][:, indices]), "genes": np.asarray(genes)}
