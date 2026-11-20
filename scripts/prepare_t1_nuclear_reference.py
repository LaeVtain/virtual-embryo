"""Rebuild the E14.5 normal cardiac RNA reference from its single wild-type multiome library."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import requests
from scipy import sparse


ACCESSION = "GSM8394736"
BASE = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8394nnn/{ACCESSION}/suppl/"
MATRIX = f"{ACCESSION}_Wildtype_raw_feature_bc_matrix.h5"
METRICS = f"{ACCESSION}_Wildtype_per_barcode_metrics.csv.gz"


def checksum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def raw_asset(folder, filename, expected):
    path = folder / filename
    if path.exists():
        if checksum(path) != expected:
            raise ValueError(f"Raw reference checksum mismatch in {path}")
        return path
    temporary = None
    try:
        with requests.get(BASE + filename, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with NamedTemporaryFile(dir=folder, delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if checksum(temporary) != expected:
            raise ValueError(f"Downloaded reference checksum mismatch for {filename}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def prepare(counts_path, metrics_path, panel):
    fields = ["barcode", "is_cell", "gex_umis_count", "gex_genes_count", "gex_intronic_umis",
              "gex_exonic_umis", "atac_peak_region_fragments"]
    native = pd.read_csv(metrics_path, usecols=fields).set_index("barcode")
    with h5py.File(counts_path) as file:
        matrix = file["matrix"]
        names = pd.Index(matrix["features/name"].asstr()[:])
        rna = matrix["features/feature_type"].asstr()[:] == "Gene Expression"
        barcodes = matrix["barcodes"].asstr()[:]
        raw = sparse.csr_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
                                shape=matrix["shape"][:][::-1])
    matched = native.loc[barcodes]
    called = (matched.is_cell == 1) & (matched.gex_umis_count >= 300)
    observed = matched.loc[called].copy()
    expression = raw[called.to_numpy()][:, rna].tocsr()
    if not np.array_equal(np.asarray(expression.sum(axis=1)).ravel(), observed.gex_umis_count.to_numpy()):
        raise ValueError("Raw RNA counts differ from the library's own barcode UMI totals.")
    columns = panel.get_indexer(names[rna])
    shared = columns >= 0
    mapping = sparse.csr_matrix(
        (np.ones(shared.sum(), np.int32), (np.flatnonzero(shared), columns[shared])),
        shape=(len(columns), len(panel)),
    )
    aligned = (expression @ mapping).tocsr().astype(np.float32)
    aligned.sort_indices()
    depth = np.asarray(aligned.sum(axis=1)).ravel()
    if np.any(depth <= 0):
        raise ValueError("Panel coverage leaves an empty reference nucleus.")
    observed["sample"] = ACCESSION
    observed["intronic_fraction"] = observed.gex_intronic_umis / observed.gex_umis_count
    observed["panel_fraction"] = depth / observed.gex_umis_count
    aligned.data = np.log1p(aligned.data * np.repeat(10000 / depth, np.diff(aligned.indptr)))
    return ad.AnnData(aligned, obs=observed, var=pd.DataFrame(index=panel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Reuse the two downloaded GEO files from this directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = args.raw_dir or args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {MATRIX: "684f31c2f2fdbf3eb7207439f868a076d405462b65b34155219d0249a68f3a3a",
              METRICS: "2b6e2e5e4773067c5ced9bef0efb67c5dd72140c90bd11221334c88a852bf954"}
    raw = {name: raw_asset(folder, name, value) for name, value in hashes.items()}
    panel = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    data = prepare(raw[MATRIX], raw[METRICS], panel)
    output = args.output_dir / "E14.5_wt_nuclei.h5ad"
    data.write_h5ad(output, compression="gzip")
    manifest = dict(accession=ACCESSION, stage=14.5, specimen="wild-type whole heart",
                    cells=data.n_obs, genes=data.n_vars, files={k:dict(url=BASE+k, sha256=v)
                    for k,v in hashes.items()}, reference_sha256=checksum(output))
    (args.output_dir / "nuclear_reference_sources.json").write_text(json.dumps(manifest, indent=2))
    print(output)


if __name__ == "__main__":
    main()
