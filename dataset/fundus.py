from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset, Stage


class Fundus(BaseDataset):
    """Fundus dataset. See https://github.com/emma-sjwang/Dofe."""

    def __init__(self, root: str | Path, domain: int, stage: Stage):
        sub_dir = "train" if stage == "train" else "test"
        BaseDataset.__init__(
            self,
            data_dir=Path(root) / f"Domain{domain+1}" / sub_dir,
            domain=domain,
            stage=stage,
            image_filter=Fundus._fundus_image_filter,
            label_filter=Fundus._fundus_label_filter,
        )

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = Image.open(image_path)
        label = Image.open(label_path).convert("L")

        image = np.asarray(image)
        label = np.asarray(label)
        image = image.transpose([2, 0, 1])
        image = image / 255
        # 0: cup & disc, 128: disc, 255: background
        label = np.stack(
            [
                label == 0,  # index 0: cup
                label <= 128,  # index 1: disc
            ],
            axis=0,
        ).astype(np.uint8)
        return image, label

    @staticmethod
    def _fundus_image_filter(path: Path) -> str | None:
        if path.parent.name == "image":
            return path.name

    @staticmethod
    def _fundus_label_filter(path: Path) -> str | None:
        if path.parent.name == "mask":
            return path.name
