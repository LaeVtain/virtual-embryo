"""Prepare two public WT E14.5 heart samples in the order of a T1 gene panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import io, sparse
from threadpoolctl import threadpool_limits


def get_sample(accession, directory, stage="E14_5", tissue="heart"):
    response = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
                            params=dict(acc=accession, form="text", view="brief", targ="self"), timeout=60)
    response.raise_for_status()
    lines = response.text.splitlines()
    fields = [f"developmental stage: {stage}", "genotype: Wild-type C57Bl/6J", f"tissue: {tissue}"]
    if not all("!Sample_characteristics_ch1 = " + value in lines for value in fields):
        raise ValueError(f"Unexpected sample identity for {accession}")
    record = dict(accession=accession, metadata=response.url,
                  metadata_sha256=hashlib.sha256(response.content).hexdigest(), files=[])
    for line in lines:
        if not line.startswith("!Sample_supplementary_file"):
            continue
        url = line.split(" = ")[1].replace("ftp://", "https://")
        filename = url.rsplit("/", 1)[1]
        if not filename.startswith(f"{accession}_{stage}") or not filename.endswith(("features.tsv.gz", "matrix.mtx.gz")):
            continue
        path = directory / filename
        if not path.exists():
            with requests.get(url, stream=True, timeout=90) as download:
                download.raise_for_status()
                partial = path.with_suffix(".partial")
                with partial.open("wb") as handle:
                    for chunk in download.iter_content(chunk_size=1024*1024):
                        handle.write(chunk)
                partial.replace(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        record["files"].append(dict(name=filename, url=url, bytes=path.stat().st_size, sha256=digest))
    if len(record["files"]) != 2:
        raise ValueError(f"Missing matrix or features for {accession}")
    return record


def aligned_sample(directory, record, panel):
    files = [directory / item["name"] for item in record["files"]]
    features_path = next(path for path in files if path.name.endswith("features.tsv.gz"))
    matrix_path = next(path for path in files if path.name.endswith("matrix.mtx.gz"))
    features = pd.read_csv(features_path, sep="\t", header=None)[1].astype(str)
    with gzip.open(matrix_path, "rb") as stream:
        counts = io.mmread(stream).T.tocsr()
    depth = np.asarray(counts.sum(axis=1)).ravel()
    mitochondrial = features.str.lower().str.startswith("mt-").to_numpy()
    mito_counts = np.asarray(counts[:, mitochondrial].sum(axis=1)).ravel()
    keep = (np.diff(counts.indptr) >= 400) & (depth >= 1000) & (mito_counts <= 0.2*depth)
    destinations = panel.get_indexer(features)
    matched = destinations >= 0
    mapping = sparse.csr_matrix((np.ones(matched.sum(), np.float32),
                                 (np.flatnonzero(matched), destinations[matched])),
                                shape=(len(features), len(panel)))
    retained = (counts[keep].astype(np.float32) @ mapping).tocsr()
    totals = np.asarray(retained.sum(axis=1)).ravel()
    retained.data = np.log1p(retained.data * np.repeat(10000/totals, np.diff(retained.indptr)))
    retained.sort_indices()
    report = dict(accession=record["accession"], raw_barcodes=len(keep), retained=len(totals),
                  mapped_unique_genes=len(set(destinations[matched])), raw_genes=len(features),
                  detected_quantiles=np.quantile(np.diff(retained.indptr), [.1,.5,.9]).tolist())
    return retained, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Use downloaded files and recorded sources.json")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = args.output_dir / "sources.json"
    if args.offline:
        records = json.loads(metadata.read_text(encoding="utf-8"))
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(lambda name: get_sample(name,args.output_dir), ["GSM7226272","GSM7226273"]))
        metadata.write_text(json.dumps(records, indent=2), encoding="utf-8")
    panel = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    if not panel.is_unique:
        parser.error("Duplicate panel genes")
    matrices, labels, reports = [], [], []
    with threadpool_limits(limits=2):
        for record in records:
            matrix, report = aligned_sample(args.output_dir,record,panel)
            matrices.append(matrix)
            labels.extend([record["accession"]] * matrix.shape[0])
            reports.append(report)
    observation = pd.DataFrame(dict(sample=labels), index=[f"ref_{i}" for i in range(len(labels))])
    reference = ad.AnnData(sparse.vstack(matrices, format="csr"), obs=observation, var=pd.DataFrame(index=panel))
    reference.write_h5ad(args.output_dir / "E14.5_heart.h5ad", compression="gzip")
    (args.output_dir / "preprocessing.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
