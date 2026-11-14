"""Export the three parameter files used by the current T3 model."""
import hashlib
from pathlib import Path
import shutil


def export_assets(workspace: Path, destination: Path) -> dict:
    source = Path(workspace) / "T3_batch_quantile_clean_20261112" / "model"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("tilt.npz", "mab.npz", "gata.npz"):
        target = destination / name
        shutil.copy2(source / name, target)
        files[name] = {"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return {"task": "t3", "files": files}
