"""Import compact frozen model assets from the original research workspace."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("models"))
    parser.add_argument("--tasks", nargs="+", choices=("t1", "t2", "t3"), default=("t1", "t2", "t3"))
    args = parser.parse_args()
    results = {}
    for task in args.tasks:
        module = importlib.import_module(f"virtual_embryo.{task}.assets")
        results[task] = module.export_assets(args.workspace, args.destination / task)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
