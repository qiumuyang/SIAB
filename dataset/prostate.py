from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .base_dataset import BaseDataset, Stage


class Prostate(BaseDataset):
    """Prostate dataset. See https://liuquande.github.io/SAML/."""

    names = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]

    def __init__(self, root: str | Path, domain: int, stage: Stage):
        sub_dir = "train" if stage == "train" else "test"
        BaseDataset.__init__(
            self,
            data_dir=Path(root) / f"{self.names[domain]}" / sub_dir,
            domain=domain,
            stage=stage,
            image_filter=Prostate._prostate_image_filter,
            label_filter=Prostate._prostate_label_filter,
        )

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        image, label = np.load(str(image_path)), np.load(str(label_path))
        # [-1, 1] to [0, 1]
        image = (image + 1) / 2
        return image, label[np.newaxis, ...]

    @staticmethod
    def _prostate_image_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "image":
            return volume + "_" + slice

    @staticmethod
    def _prostate_label_filter(path: Path) -> str | None:
        volume, slice, type = path.stem.rsplit("_")
        if path.suffix == ".npy" and type == "label":
            return volume + "_" + slice

    def random_split(
        self: Prostate,
        *frac: float,
        rounding: Callable[[float], int] = round,
    ) -> tuple[list[Prostate], list[list[int]]]:
        def setup(dataset: Prostate, indices_: list[int]) -> None:
            """Setup the dataset and filter new paths with given indices."""
            dataset.image_paths = [self.image_paths[i] for i in indices_]
            dataset.label_paths = [self.label_paths[i] for i in indices_]

        n = len(self)  # for data loading

        # get all volume_id
        all_vols = set(int(p.stem.split("_")[0]) for p in self.image_paths)
        n_vols = len(all_vols)
        vol_indices = torch.randperm(n_vols).tolist()

        indices_split = []
        for f in frac:
            n_split = rounding(f * n_vols)
            vol_split = vol_indices[:n_split]
            vol_indices = vol_indices[n_split:]
            # get the indices that startswith any volume_id in vol_split
            indices = [
                i for i, p in enumerate(self.image_paths)
                if int(p.stem.split("_")[0]) in vol_split
            ]
            indices_split.append(indices)

        indices = [
            i for i, p in enumerate(self.image_paths)
            if int(p.stem.split("_")[0]) in vol_indices
        ]
        indices_split.append(indices)

        results = [deepcopy(self) for _ in range(len(indices_split))]

        [
            setup(dataset, indices_)
            for dataset, indices_ in zip(results, indices_split)
        ]
        return results, indices_split

    def random_split_k(
        self: Prostate,
        k: int,
    ) -> tuple[list[Prostate], list[int]]:
        raise NotImplementedError
