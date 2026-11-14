"""Export portable T2 model assets from the original workspace."""
from pathlib import Path
import hashlib
import json
import shutil

import torch


def export_assets(workspace: Path, destination: Path) -> dict:
    source = Path(workspace) / "T2_quantile_landmark_clean_20261112"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    sources = {}
    for setting in ("embryo", "heart"):
        original = source / f"model/checkpoints/T2_{setting}_control.pt"
        checkpoint = torch.load(original, map_location="cpu", weights_only=False)
        target = destination / f"{setting}_control.pt"
        torch.save({key: checkpoint[key] for key in ("expression", "spatial")}, target)
        records.append({"file": target.name, "original_bytes": original.stat().st_size})
        sources[target.name] = original
        shutil.copyfile(source / f"{setting}_maps.joblib", destination / f"{setting}_maps.joblib")
        sources[f"{setting}_maps.joblib"] = source / f"{setting}_maps.joblib"
    original = source / "model/checkpoints/T2_heart_spatial_continued.pt"
    checkpoint = torch.load(original, map_location="cpu", weights_only=False)
    target = destination / "heart_spatial_continued.pt"
    torch.save({"spatial": checkpoint["spatial"]}, target)
    records.append({"file": target.name, "original_bytes": original.stat().st_size})
    sources[target.name] = original
    config = json.loads((source / "CONFIG.json").read_text(encoding="utf-8"))
    for setting in config.values():
        setting.pop("pairs", None)
        setting.pop("expression", None)
        setting.pop("spatial_training", None)
        if isinstance(setting["spatial"], str):
            setting.pop("spatial")
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    sources["config.json"] = source / "CONFIG.json"
    metadata = json.loads((source / "model/metadata/runtime_metadata.json").read_text(encoding="utf-8"))
    metadata = {f"T2_{setting}": {"genes": metadata[f"T2_{setting}"]["genes"]} for setting in ("heart", "embryo")}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    sources["metadata.json"] = source / "model/metadata/runtime_metadata.json"
    files = {}
    for name, original in sources.items():
        target = destination / name
        files[name] = dict(bytes=target.stat().st_size,
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                           source=original.relative_to(Path(workspace)).as_posix(),
                           source_bytes=original.stat().st_size,
                           source_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
    for record in records:
        record.update(files[record["file"]])
    return {"task": "T2", "checkpoints": records, "files": files,
            "removed": ["optimizer", "RNG states", "unused continued expression weights"]}
