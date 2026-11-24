"""Rebuild the four local predictions from the released challenge H5AD input files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
VERSIONS = ("2026-11-19", "2026-11-23", "2026-11-23-r2", "2026-11-24")
sys.path.insert(0, str(HERE / "src"))
from virtual_embryo.export import export_h5ad
from virtual_embryo.t2.temporal import read_stage
from virtual_embryo.t3.predict import predict


def module(name, *arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", name, *map(str, arguments)], env=env, check=True)


def run(data, output, assets_root=HERE, version=VERSIONS[-1]):
    output.mkdir(exist_ok=True, parents=True)
    asset = assets_root / "models"
    panels = assets_root / "panels"
    revised = version in VERSIONS[1:]
    round_two = version in ("2026-11-23-r2", "2026-11-24")
    november_24 = version == "2026-11-24"
    with tempfile.TemporaryDirectory(dir=output) as temp:
        work = Path(temp)
        t1_settings = [
            "--source", data / "E10.5_RNA.h5ad", "--calibration", data / "E8.5_RNA.h5ad",
            "--lineage-reference", asset / "t1/lineage_reference.h5ad",
            "--chamber-reference", asset / "t1/chamber_reference.npz", "--cross-gene-dispersion",
            "--panel", panels / "T1__test.genes.txt", "--trend", asset / "t1/stage_effects.csv",
            "--interval", 5, "--source-time", 10.5, "--target-time", 12.5, "--cells", 4096, "--seed", 101,
        ]
        module("virtual_embryo.t1.lineage_forecast", *t1_settings, "--reference",
               asset / "t1/E14.5_heart.h5ad", "--nuclear-reference", asset / "t1/E14.5_wt_nuclei.h5ad",
               *(["--nuclear-trajectory"] if revised else []),
               *(["--nuclear-population", "rank"] if round_two else []),
               *(["--empirical-fraction", .25] if november_24 else []),
               "--output", work / "T1__test.h5ad")
        embryo = "T2__embryo__test_interp"
        stages = []
        for age in (6.75, 7.25, 7.5, 8.0):
            stages.extend(["--stage", age, data / f"E{age}.h5ad"])
        embryo_size = ["--section-volume"]
        if round_two:
            embryo_size = ["--growth-reference", asset / "t2/embryo_physical_lengths.npz",
                           "--continuous-section-depth"]
        embryo_cells = 5000 if round_two else 4096
        module("virtual_embryo.t2.embryo_interpolate", *stages, "--target-time", 7.75, "--cells", embryo_cells,
               "--seed", 101, "--spatial-model", "sections", "--match-capture-depth", "--sample-sections", *embryo_size,
               *(["--local-expression"] if revised else []),
               *(["--interpolate-gene-ranks"] if round_two else []),
               "--panel", panels / (embryo + ".genes.txt"), "--output", work / (embryo + ".npz"))
        heart = "T2__heart__test_extrap"
        heart_model = []
        if round_two:
            heart_model = ["--maturation-reference", asset / "t2/heart_clean_history_105_125.npz"]
            if november_24:
                heart_model.append("--match-tissue-profiles")
        elif revised:
            heart_model = ["--refine-anatomy", "--local-expression"]
        module("virtual_embryo.t2.heart_forecast", "--earlier", data / "E9.5.h5ad", "--later", data / "E10.5.h5ad",
               "--earlier-time", 9.5, "--later-time", 10.5, "--target-time", 12.5, "--cells", 8192, "--seed", 101,
               "--trend", asset / "t2/heart_ventricular_trend.csv", "--anatomy", asset / "t2/heart_e145.npz",
               "--state-reference", asset / "t2/heart_state_095_145.npz", "--state-strength", "animal-horizon",
               *heart_model,
               "--spatial-rna", asset / "t2/heart_spatial_niches.npz", "--panel", panels / (heart + ".genes.txt"),
               "--output", work / (heart + ".npz"))
        for board in (embryo, heart):
            export_h5ad(work / (board + ".npz"), work / (board + ".h5ad"))
        board = "T3__bcatenin"
        genes = (panels / (board + ".genes.txt")).read_text(encoding="utf-8-sig").splitlines()
        control = read_stage(data / "E8.75.h5ad", genes)
        rng = np.random.default_rng(101)
        control_cells = len(control["x"])
        forecast_cells = min(7449,control_cells) if round_two else round(control_cells / 10)
        rows = np.sort(rng.choice(control_cells, forecast_cells, replace=False))
        condition = "bcatenin-v4" if round_two else "bcatenin-v3" if revised else "bcatenin-v2"
        values = predict(control["x"][rows], control["c"][rows], control["genes"], condition, 101, asset / "t3")
        columns = [list(values["genes"]).index(g) for g in genes]
        np.savez_compressed(work / (board + ".npz"), x=values["x"][:, columns], c=values["c"], genes=np.array(genes))
        export_h5ad(work / (board + ".npz"), work / (board + ".h5ad"))
        for saved in work.glob("*.h5ad"):
            saved.replace(output / saved.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Directory of the original challenge H5AD inputs")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--assets", type=Path, default=HERE, help="Directory containing the released models/ and panels/")
    p.add_argument("--version", choices=VERSIONS, default=VERSIONS[-1])
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.output, args.assets, args.version)
