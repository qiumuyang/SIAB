from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copytree

if __name__ == "__main__":
    domains = ["Domain1", "Domain2", "Domain3", "Domain4"]

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # we only care about the ROIs
    # root/domain/train[test]/ROIs/image[mask]/*.png

    for domain in domains:
        domain_dir = args.root / domain
        if not domain_dir.exists():
            print("Expected domain not found:", domain_dir)
            exit(1)

        print("Copying domain:", domain)
        for stage in ["train", "test"]:
            stage_dir = domain_dir / stage
            output_dir = args.output / domain / stage

            image_dir = output_dir / "image"
            mask_dir = output_dir / "mask"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            copytree(stage_dir / "ROIs" / "image", image_dir)
            copytree(stage_dir / "ROIs" / "mask", mask_dir)
