"""Assemble the November 24 inference assets from published parameters and local references."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import anndata as ad
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases" / "2026-11-24"
sys.path.insert(0, str(ROOT / "src"))
from virtual_embryo.t3.atlas_response import reference_banks


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_content_hash(path):
    """Hash the ordered cells, genes, grouping labels and expression used by T1."""
    digest = hashlib.sha256()
    reference = ad.read_h5ad(path, backed="r")
    try:
        labels = [reference.obs_names, reference.var_names]
        labels.extend(reference.obs[name] for name in ("sample", "cohort") if name in reference.obs)
        for values in labels:
            encoded = json.dumps(list(map(str, values)), ensure_ascii=False, separators=(",", ":"))
            digest.update(encoded.encode("utf-8") + b"\n")
        digest.update(np.asarray(reference.shape, dtype="<i8").tobytes())
        matrix = reference.X
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        for start in range(0, reference.n_obs, 256):
            rows = sparse.csr_matrix(matrix[start:start + 256, :], copy=True)
            rows.sum_duplicates()
            rows.eliminate_zeros()
            rows.sort_indices()
            for values, dtype in ((rows.indptr, "<i8"), (rows.indices, "<i8"), (rows.data, "<f8")):
                digest.update(np.asarray(values, dtype=dtype).tobytes())
    finally:
        reference.file.close()
    return digest.hexdigest()


def verify(path, expected, content_hash=None):
    if sha256(path) == expected:
        return
    if content_hash and reference_content_hash(path) == content_hash:
        return
    raise ValueError(f"Checksum mismatch for {path}. Expected SHA-256 {expected}")


def rebuild_t3(data, target, manifest):
    for name, expected in manifest["inputs_sha256"].items():
        verify(data / name, expected)
    coefficients = RELEASE / manifest["coefficients"]
    verify(coefficients, manifest["coefficients_sha256"])
    with np.load(coefficients, allow_pickle=False) as parameters:
        genes = parameters["genes"]
        ages = ("E8.25_late.h5ad", "E8.5.h5ad", "E10.5.h5ad")
        bank, atlas = reference_banks(data / "E9.5.h5ad", data / "E9.5_mab21l2_ko.h5ad",
                                      [data / name for name in ages], genes)
        np.savez_compressed(target, genes=genes, bank=bank, atlas=np.vstack(atlas),
                            atlas_means=parameters["atlas_means"], signature=parameters["signature"],
                            rna_response=parameters["rna_response"])


def prepare(data, references, output):
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    if output.exists():
        for name, record in manifest["assets"].items():
            verify(output / name, record["sha256"], record.get("content_sha256"))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "assets"
        for name, record in manifest["assets"].items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if "source" in record:
                shutil.copyfile(RELEASE / record["source"], target)
            elif "reference" in record:
                shutil.copyfile(references / record["reference"], target)
            else:
                rebuild_t3(data, target, manifest["t3"])
            verify(target, record["sha256"], record.get("content_sha256"))
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory of original challenge H5ADs")
    parser.add_argument("--references", type=Path, required=True, help="Directory of three prepared T1 H5ADs")
    parser.add_argument("--output", type=Path, required=True, help="New asset directory, or a previous complete build")
    args = parser.parse_args()
    prepare(args.data, args.references, args.output)
    print(f"Verified inference assets in {args.output}")


if __name__ == "__main__":
    main()
