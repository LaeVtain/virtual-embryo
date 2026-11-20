"""Import the known-condition parameters and the P3 transfer bank when available."""
import hashlib
from pathlib import Path
import shutil


def export_assets(workspace: Path, destination: Path) -> dict:
    source = Path(workspace) / "T3_batch_quantile_clean_20261112" / "model"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    paths = [source / name for name in ("tilt.npz", "mab.npz", "gata.npz")]
    transfer = Path(workspace) / "Optimization_20261119" / "t3" / "p3" / "bcatenin.npz"
    if transfer.is_file():
        paths.append(transfer)
    newest = Path(workspace) / "Optimization_round2" / "t3" / "p3" / "bcatenin-v2.npz"
    if newest.is_file():
        paths.append(newest)
    for path in paths:
        name = path.name
        target = destination / name
        shutil.copy2(path, target)
        files[name] = {"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return {"task": "t3", "files": files}
