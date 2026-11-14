#!/usr/bin/env python3
"""Connect existing datasets and model assets without copying or modifying them."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original experiment root; supplies checkpoints, outputs and cache defaults")
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--legacy-outputs", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--cells-all", type=Path)
    parser.add_argument("--inference-data", type=Path)
    parser.add_argument("--raw-data", type=Path)
    args = parser.parse_args()
    source = args.source.resolve() if args.source else None
    choices = {
        "checkpoints": args.checkpoints or (source / "checkpoints" if source else None),
        "assets/legacy_outputs": args.legacy_outputs or (source / "outputs" if source else None),
        "hf_cache": args.cache or (source / "hf_cache" if source else None),
        "data/cells": args.cells,
        "data/cells_all": args.cells_all,
        "data/prediction_for_day_15": args.inference_data,
        "data/raw": args.raw_data,
    }
    resolved = {}
    # Validate all inputs before changing any link.
    for alias, target in choices.items():
        if target is None:
            continue
        target = target.resolve(strict=True)
        if not target.is_dir():
            parser.error(f"Expected a directory for {alias}")
        link = ROOT / alias
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target:
                parser.error(f"Refusing to replace existing path: {alias}")
        resolved[alias] = target
    if not resolved:
        parser.error("Provide --source or at least one asset path")
    for alias, target in resolved.items():
        link = ROOT / alias
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        print(f"Linked {alias}")
    # Store only portable aliases. Actual local paths remain in ignored symlinks.
    config_path = ROOT / "configs/local.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {"project_name": "orgacast", "linked_assets": {}}
    config["linked_assets"].pop("datasets", None)
    config["linked_assets"].update({alias: alias for alias in resolved})
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (ROOT / "outputs").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
