"""Location of model assets kept outside the source distribution."""
import os
from pathlib import Path


def task_assets(task: str) -> Path:
    root = Path(os.environ.get("VIRTUAL_EMBRYO_ASSETS", Path(__file__).resolve().parents[2] / "models"))
    return root / task
