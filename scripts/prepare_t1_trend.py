"""Build a mouse E9.5-to-E14.5 whole-embryo expression prior from GEO."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SAMPLES = {"E8.5": ["GSM701719", "GSM701720", "GSM701721"],
           "E9.5": ["GSM701722", "GSM701723", "GSM701724"],
           "E14.5": ["GSM701730", "GSM701731"]}
GPL_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz"


def absolute_signal(text, accession, stage):
    header, body = text.split("!sample_table_begin", 1)
    fields = set(header.splitlines())
    required = {
        f"^SAMPLE = {accession}",
        f"!Sample_characteristics_ch1 = developmental stage: {stage}",
        "!Sample_characteristics_ch1 = genotype: wild type",
        "!Sample_platform_id = GPL1261",
        "!Sample_data_processing = MAS5, default settings",
        "#VALUE = MAS5",
    }
    if missing := required - fields:
        raise ValueError(f"{accession} is missing expected fields: {sorted(missing)}")
    table = pd.read_csv(io.StringIO(body.split("!sample_table_end")[0].strip()), sep="\t")
    return np.log(np.maximum(table.set_index("ID_REF")["VALUE"], 1))


def fetch(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    samples = {}
    for stage, accessions in SAMPLES.items():
        for accession in accessions:
            response = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
                                    params={"acc": accession, "form": "text", "view": "full",
                                            "targ": "self"}, timeout=50)
            response.raise_for_status()
            samples[accession] = absolute_signal(response.text, accession, stage)
            provenance[accession] = {"stage": stage, "processing": "MAS5, default settings",
                                     "signal": "absolute", "url": response.url,
                                     "sha256": hashlib.sha256(response.content).hexdigest()}
    response = requests.get(GPL_URL, timeout=90)
    response.raise_for_status()
    annotation = gzip.decompress(response.content).decode("utf-8", errors="replace")
    table = annotation.split("!platform_table_begin")[1].split("!platform_table_end")[0].strip()
    annotations = pd.read_csv(io.StringIO(table), sep="\t", usecols=["ID", "Gene symbol"])
    annotations = annotations.dropna().loc[lambda d: ~d["Gene symbol"].str.contains("///")]
    expression = pd.DataFrame(samples)
    for start, filename in [("E9.5", "stage_effects.csv"), ("E8.5", "from_e85.csv")]:
        effect = expression[SAMPLES["E14.5"]].mean(axis=1) - expression[SAMPLES[start]].mean(axis=1)
        effects = annotations.join(effect.rename("log_ratio"), on="ID").groupby("Gene symbol").log_ratio.median()
        effects.dropna().to_csv(output_dir / filename)
    provenance["annotation"] = {"url": GPL_URL, "sha256": hashlib.sha256(response.content).hexdigest()}
    (output_dir / "sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("genes", len(effects), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    fetch(parser.parse_args().output_dir)
