from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from typing_extensions import Literal


def volume_slice_filter(
    path: Path,
    slice_type: Literal["image", "label"],
    volume_id: list[int] | None = None,
) -> str:
    """A filter template for volume slices.

    Use `functools.partial` to create a filter for a specific target.

    Args:
        path: the path of the file
        slice_type: required slice type.
        volume_id: required volume id, None for no restriction
    """
    if path.suffix != ".npy":
        return ""
    volume, slice, type = path.stem.rsplit("_")
    if slice_type == type:
        if volume_id is None or int(volume) in volume_id:
            return volume + "_" + slice
    return ""


def volume_slice_format(
    volume_id: int,
    slice_id: int,
    slice_type: Literal["image", "label"],
    max_digits: int = 4,
) -> str:
    v = str(volume_id).zfill(max_digits)
    s = str(slice_id).zfill(max_digits)
    return f"{v}_{s}_{slice_type}.npy"


def _prostate_image(path: Path) -> str | None:
    stem, suffix = path.name.split(".", 1)  # Case.nii.gz
    stem = stem.lower()
    if suffix == "nii.gz" and not stem.endswith("_segmentation"):
        return stem


def _prostate_label(path: Path) -> str | None:
    stem, suffix = path.name.split(".", 1)  # Case.nii.gz
    stem = stem.lower()
    if suffix == "nii.gz" and stem.endswith("_segmentation"):
        return stem.rsplit("_", 1)[0]  # remove "_segmentation"


def extract_slices(
    volume_dir: Path,
    slice_dir: Path,
    stage: Literal["train", "test"],
):
    slice_dir.mkdir(parents=True, exist_ok=True)

    image_volume_paths = [
        p for p in volume_dir.iterdir() if p.is_file() and _prostate_image(p)
    ]
    label_volume_paths = [
        p for p in volume_dir.iterdir() if p.is_file() and _prostate_label(p)
    ]
    image_volume_paths.sort()
    label_volume_paths.sort()

    vol_digits = len(str(len(image_volume_paths)))

    for volume_id, (image_path, label_path) in enumerate(
            zip(image_volume_paths, label_volume_paths)):
        if _prostate_image(image_path) != _prostate_label(label_path):
            raise RuntimeError(
                f"Image and label volume mismatch: {image_path} {label_path}")

        itk_image = sitk.ReadImage(str(image_path))
        itk_label = sitk.ReadImage(str(label_path))
        image: ndarray = sitk.GetArrayFromImage(itk_image).astype(np.float32)
        label: ndarray = sitk.GetArrayFromImage(itk_label).astype(np.uint8)

        # [-1, 1]
        min_ = np.min(image)
        max_ = np.max(image)
        if min_ != max_:
            image = 2 * (image - min_) / (max_ - min_) - 1
        # some domain labels are [0, 1, 2], convert to [0, 1]
        label[label > 0] = 1

        if stage == "train":
            # discard slices at both ends
            indices = label.sum(axis=(1, 2)).nonzero()[0]
            low, high = indices.min(), indices.max()
            # keep one slice before and after
            low = max(0, low - 1)
            high = min(image.shape[0] - 1, high + 1)
            image = image[low:high + 1]
            label = label[low:high + 1]
        else:
            # discard slices at any position
            indices = label.sum(axis=(1, 2)).nonzero()[0]
            image = image[indices]
            label = label[indices]

        # save to slices (3 consecutive slices as one sample)
        slice_digits = len(str(len(image)))
        n_digits = max(vol_digits, slice_digits)
        slice_id = 0
        for i in range(1, image.shape[0] - 1):
            image_path = slice_dir / volume_slice_format(
                volume_id, slice_id, "image", n_digits)
            label_path = slice_dir / volume_slice_format(
                volume_id, slice_id, "label", n_digits)
            image_slice = image[i - 1:i + 2]
            label_slice = label[i]
            np.save(image_path, image_slice)
            np.save(label_path, label_slice)
            slice_id += 1


if __name__ == "__main__":
    domains = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()

    volume_dir = args.root
    if args.output is None:
        slice_dir = volume_dir.parent / (volume_dir.name + "_slices")
    else:
        slice_dir = args.output

    for domain in domains:
        domain_dir = volume_dir / domain
        if not domain_dir.exists():
            print("Expected domain not found:", domain_dir)
            exit(1)

        print("Extracting slices for domain:", domain)
        for stage in ["train", "test"]:  # type: ignore
            stage: Literal["train", "test"]
            extract_slices(domain_dir, slice_dir / domain / stage, stage)
