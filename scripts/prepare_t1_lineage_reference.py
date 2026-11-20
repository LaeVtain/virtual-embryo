"""Build the lineage reference bundle from wild-type early cardiac and public 10x cohorts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from threadpoolctl import threadpool_limits

from prepare_t1_heart_reference import aligned_sample, get_sample
from virtual_embryo.t1.lineage_forecast import sample_libraries

WT = "https://cells.ucsc.edu/mouse-cardiac/wtAll/"
AGES = ("7.75", "8.25", "9.25")


def wild_type(cache, panel):
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in ("dataset.json", "meta.tsv", "exprMatrix.tsv.gz"):
        path = cache / name
        if not path.exists():
            with requests.get(WT + name, stream=True, timeout=180) as response:
                response.raise_for_status()
                partial = path.with_suffix(".partial")
                with partial.open("wb") as dest:
                    for chunk in response.iter_content(1024 * 1024):
                        dest.write(chunk)
                partial.replace(path)
        with path.open("rb") as handle:
            records[name] = dict(url=WT + name, sha256=hashlib.file_digest(handle, "sha256").hexdigest())
    description = json.loads((cache / "dataset.json").read_text())
    expected_bytes = description["fileVersions"]["outMatrix"]["size"]
    raw = cache / "exprMatrix.tsv.gz"
    if description["name"] != "mouse-cardiac/wtAll" or raw.stat().st_size != expected_bytes:
        raise ValueError("Expected the complete wild-type cardiac count matrix")
    meta = pd.read_csv(cache / "meta.tsv", sep="\t", index_col=0, dtype=str)
    positions = {gene: i for i, gene in enumerate(panel)}
    with gzip.open(raw, "rb") as stream:
        barcodes = stream.readline().decode().rstrip("\r\n").split("\t")[1:]
        if not meta.index.is_unique or set(barcodes) != set(meta.index):
            raise ValueError("The expression matrix and metadata have different cells")
        meta = meta.loc[barcodes]
        stages = meta["Time Point"].to_numpy()
        libraries = meta["gem.group"].to_numpy()
        if set(stages) != {"E" + age for age in AGES}:
            raise ValueError("Unexpected ages in the early wild-type source")
        chosen = []
        observations = []
        for age in AGES:
            pool = np.flatnonzero(stages == "E" + age)
            rows = pool[sample_libraries(libraries[pool], 700, 716)]
            chosen.extend(rows)
            observations.extend([("wt_" + age, libraries[i]) for i in rows])
        matrix_rows, columns = [], []
        for line in stream:
            symbol, numbers = line.rstrip(b"\r\n").split(b"\t", 1)
            column = positions.get(symbol.decode().split("|")[-1].strip('"'))
            if column is None:
                continue
            counts = np.fromstring(numbers, sep="\t", dtype=np.int32)
            if len(counts) != len(barcodes):
                raise ValueError(f"Invalid count row for {symbol!r}")
            matrix_rows.append(sparse.csr_matrix(counts[chosen].reshape(1, -1)))
            columns.append(column)
    counts = sparse.vstack(matrix_rows, format="csr").T.tocsr()
    index = sparse.csr_matrix((np.ones(len(columns), np.float32), (np.arange(len(columns)), columns)),
                              shape=(len(columns), len(panel)))
    expression = (counts.astype(np.float32) @ index).tocsr()
    totals = np.asarray(expression.sum(axis=1)).ravel()
    expression.data = np.log1p(expression.data * np.repeat(10000 / totals, np.diff(expression.indptr)))
    return expression, observations, records


def atlas(cache, panel, stage, samples, per_library, prepared):
    records = []
    if prepared is None:
        cache.mkdir(parents=True, exist_ok=True)
        matrices, names = [], []
        for sample in samples:
            record = get_sample(sample, cache, stage.replace(".", "_"), "embryo" if stage == "E8.5" else "heart")
            expression, report = aligned_sample(cache, record, panel)
            records.append(dict(source=record, preprocessing=report))
            matrices.append(expression)
            names.extend([sample] * expression.shape[0])
        matrix = sparse.vstack(matrices, format="csr")
        libraries = np.array(names)
    else:
        data = ad.read_h5ad(prepared)
        if not data.var_names.equals(panel):
            raise ValueError(f"Gene order differs in {prepared}")
        matrix = data.X
        libraries = data.obs["sample"].astype(str).to_numpy()
        with prepared.open("rb") as stream:
            records.append(dict(file=prepared.name, sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    if set(libraries) != set(samples):
        raise ValueError(f"Unexpected libraries at {stage}")
    selected = sample_libraries(libraries, per_library, 716)
    observations = [("10x_" + stage[1:], lib) for lib in libraries[selected]]
    return matrix[selected], observations, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--wt-cache", type=Path, help="Existing UCSC wtAll raw files")
    parser.add_argument("--prepared-8-5", type=Path)
    parser.add_argument("--prepared-16-5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    if not panel.is_unique or args.output.exists():
        parser.error("Provide a unique gene panel and a new output path")
    with threadpool_limits(limits=2):
        early = wild_type(args.wt_cache or args.cache / "wt", panel)
        young = atlas(args.cache / "E8.5", panel, "E8.5", ["GSM7226269"], 3000, args.prepared_8_5)
        late = atlas(args.cache / "E16.5", panel, "E16.5", ["GSM7226274", "GSM7226276"], 1200, args.prepared_16_5)
        cohorts = [early, young, late]
        matrix = sparse.vstack([x[0] for x in cohorts], format="csr")
        obs = pd.DataFrame([row for item in cohorts for row in item[1]], columns=["cohort", "sample"])
        obs.index = [f"reference_{i}" for i in range(len(obs))]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=panel)).write_h5ad(args.output, compression="gzip")
    args.output.with_suffix(".sources.json").write_text(
        json.dumps(dict(wild_type=early[2], E8_5=young[2], E16_5=late[2]), indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
