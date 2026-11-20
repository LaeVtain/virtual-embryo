"""Read the WT 10x count slot from a GSE193346 archive and keep only the listed cardiac ages."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193346/suppl/GSE193346_"
AGES = ("E9.5", "E14.5")


class PrefixReader(io.RawIOBase):
    """Buffered chunks keep both layers of the gzip stream incremental."""
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.chunk = b""
        self.hash = hashlib.sha256()
        self.session = requests.Session()

    def readable(self):
        return True

    def readinto(self, out):
        if not self.chunk:
            start = self.position
            stop = start + (8 << 20) - 1
            for attempt in range(4):
                try:
                    response = self.session.get(self.url, headers={"Range": f"bytes={start}-{stop}"},
                                                timeout=(30, 120))
                    response.raise_for_status()
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{stop}/"):
                        raise ValueError(f"Unexpected HTTP range {response.headers.get('Content-Range')}")
                    self.chunk = memoryview(response.content)
                    self.hash.update(self.chunk)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        size = min(len(out), len(self.chunk))
        out[:size] = self.chunk[:size]
        self.chunk = self.chunk[size:]
        self.position += size
        return size


def read_word(stream):
    return struct.unpack(">I", stream.read(4))[0]


def expect_symbol(stream, key, kind):
    symbol = key.encode()
    expected = struct.pack(">IIII", 0x402, 1, 0x40009, len(symbol)) + symbol
    actual = stream.read(len(expected))
    if actual != expected or read_word(stream) != kind:
        raise ValueError(f"Unexpected R slot for {key}: {actual[:28]!r}")


def strings(stream):
    if read_word(stream) != 16:
        raise ValueError("Expected R character vector")
    size = read_word(stream)
    values = []
    for _ in range(size):
        tag = read_word(stream)
        if tag & 0xff != 9:
            raise ValueError(f"Unexpected R string tag {tag:#x}")
        values.append(stream.read(read_word(stream)).decode())
    return values


def selected_counts(stream, spans, total):
    """Decode molecular entries only for whitelisted barcodes."""
    output = np.empty(int(np.sum(spans[:, 1] - spans[:, 0])), dtype=np.int32)
    write = 0
    pointer = 0
    for offset in range(0, total, 1 << 20):
        end = min(total, offset + (1 << 20))
        buffer = stream.read((end - offset) * 8)
        while pointer < len(spans) and spans[pointer, 0] < end:
            first, last = spans[pointer]
            start_here, end_here = max(offset, first), min(end, last)
            count = end_here - start_here
            vals = np.frombuffer(buffer, dtype=">f8", offset=(start_here - offset) * 8, count=count)
            if not np.all((vals == np.rint(vals)) & (vals > 0)):
                raise ValueError("Expected positive integer molecule counts")
            output[write:write + count] = vals
            write += count
            if end_here != last:
                break
            pointer += 1
    assert write == len(output) and pointer == len(spans)
    return output


def extract(panel_file, out):
    strain = "CD1"
    out.mkdir(parents=True, exist_ok=True)
    url = BASE + strain + "_metadata.csv.gz"
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    metadata = pd.read_csv(io.BytesIO(response.content), compression="gzip",
                           usecols=["Unnamed: 0", "sample", "stage", "zone", "nCount_RNA", "nFeature_RNA"]).set_index("Unnamed: 0")
    meta_sha = hashlib.sha256(response.content).hexdigest()
    metadata = metadata.loc[metadata.stage.isin(AGES)]
    if not metadata.index.is_unique or metadata.empty:
        raise ValueError("Missing age-resolved sample barcodes")
    raw_url = BASE + strain + "_seurat_object.rds.gz"
    remote = PrefixReader(raw_url)
    outer = gzip.GzipFile(fileobj=io.BufferedReader(remote), mode="rb")
    stream = gzip.GzipFile(fileobj=outer, mode="rb")
    intro = stream.read(108)
    if (not intro.startswith(b"X\n") or b"assays" not in intro or b"counts" not in intro
            or not intro.endswith(b"i\x00\x00\x00\r")):
        raise ValueError(f"Unexpected RNA counts prefix: {intro!r}")
    total = read_word(stream)
    indices = np.frombuffer(stream.read(4 * total), dtype=">i4")
    expect_symbol(stream, "p", 13)
    pointers = np.frombuffer(stream.read(4 * read_word(stream)), dtype=">i4").astype(np.int64)
    expect_symbol(stream, "Dim", 13)
    assert read_word(stream) == 2
    n_genes, n_cells = read_word(stream), read_word(stream)
    assert pointers.shape == (n_cells + 1,) and pointers[-1] == total
    expect_symbol(stream, "Dimnames", 19)
    assert read_word(stream) == 2
    genes, barcodes = strings(stream), strings(stream)
    assert len(genes) == n_genes and len(barcodes) == n_cells
    selected = np.flatnonzero(pd.Index(barcodes).isin(metadata.index))
    obs = metadata.loc[np.asarray(barcodes)[selected]].copy()
    assert len(obs) == len(metadata)
    spans = np.column_stack([pointers[selected], pointers[selected + 1]])
    column_indices = np.concatenate([indices[lo:hi] for lo, hi in spans])
    del indices
    expect_symbol(stream, "x", 14)
    assert read_word(stream) == total
    counts = selected_counts(stream, spans, total)
    matrix = sparse.csr_matrix((counts, column_indices, np.r_[0, np.cumsum(np.diff(spans).ravel())]),
                               shape=(len(obs), n_genes))
    assert np.array_equal(np.asarray(matrix.sum(1)).ravel(), obs["nCount_RNA"])
    assert np.array_equal(np.diff(matrix.indptr), obs["nFeature_RNA"])
    panel = pd.Index(panel_file.read_text(encoding="utf-8-sig").splitlines())
    if not panel.is_unique:
        raise ValueError("Expected a unique challenge gene panel.")
    cols = panel.get_indexer(genes)
    valid = cols >= 0
    mapping = sparse.csr_matrix((np.ones(valid.sum(), dtype=np.int32), (np.where(valid)[0], cols[valid])),
                                shape=(len(cols), len(panel)))
    aligned = (matrix @ mapping).tocsr()
    aligned.sort_indices()
    data = ad.AnnData(aligned, obs=obs, var=pd.DataFrame(index=panel))
    path = out / f"{strain}.h5ad"
    data.write_h5ad(path, compression="gzip")
    receipt = dict(strain=strain, stages=obs.stage.value_counts().to_dict(), chambers=obs.zone.value_counts().to_dict(),
                   raw_url=raw_url, compressed_prefix_end=remote.position + len(remote.chunk),
                   compressed_prefix_sha256=remote.hash.hexdigest(), metadata_url=url, metadata_sha256=meta_sha,
                   total_genes=n_genes, mapped_genes=int(valid.sum()), total_assay_barcodes=n_cells,
                   kept_cells=len(obs), kept_full_molecules=int(counts.sum()), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / f"{strain}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    extract(args.panel, args.output_dir)
