from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing_extensions import Literal, TypedDict

from .base_dataset import BaseDataset
from .fundus import Fundus
from .mnms import MNMS
from .mnms import get_all_data_folder as get_mnms_domains
from .prostate import Prostate
from .style_sampler import RandomStyleSampler, StyleSampler
from .transform import (amplitude_mixup, blur, hist_match, obtain_cutmix_box,
                        random_rot_flip, random_rotate)


class StrongAugConfig(TypedDict):
    color_jitter: float
    cutmix: float
    blur: float


dataset_zoo: dict[str, type[BaseDataset]] = {
    "fundus": Fundus,
    "mnms": MNMS,
    "prostate": Prostate,
}
default_style_mode: Literal["hist"] = "hist"
default_strong = StrongAugConfig(
    color_jitter=0.8,
    cutmix=0.5,
    blur=0.5,
)


class SSDGDataset(Dataset):
    """Dataset for Semi-Supervised Domain Generalization.

    Attributes:
        name: dataset name
        cls: dataset class
        root: dataset root
        target_domain: id of target domain
        mode: train/val
            if train, load all domains except target domain
            if val, load only target domain (images will not be augmented)
            path name of images will be returned if mode is val
        n_domains: number of domains
        image_size: resize image to image_size x image_size
        with_indices: return indices if True
        aug: augmentation type
            - weak augmentation is always applied
            - options are "strong" and "style", separated by "+"
            - extra configs for "strong" and "style" can be specified
              using StrongAugConfig and StyleSampler
    """

    def __init__(
        self,
        name: str,
        root: str | Path,
        target_domain: int,
        mode: Literal["train", "val"],
        n_domains: int,
        image_size: int,
        with_indices: bool = False,
    ):
        self.name = name.lower()
        self.cls = dataset_zoo[self.name]
        self.root = root
        self.target_domain = target_domain
        self.mode = mode
        self.image_size = image_size
        self.with_indices = with_indices
        self.strong_configs: list[StrongAugConfig] = []
        self.style_samplers: list[StyleSampler] = []

        # load data
        if self.cls is MNMS:
            # cannot get split dataset right now
            # get split dataset when calling split_ulb_lb
            self.datasets = get_split_dataset(
                root,
                target_domain,
                mode,
                1.0,
                get_mnms_domains,
                self.cls,
            )
        else:
            if mode == "val":
                self.datasets = [self.cls(root, target_domain, "val")]
            else:
                self.datasets = [
                    self.cls(root, domain, "train")
                    for domain in range(n_domains) if domain != target_domain
                ]

    def validation(self):
        dataset = deepcopy(self)
        dataset.mode = "val"
        return dataset

    def config_augmentation(
            self,
            aug: str,
            strong: StrongAugConfig | list[StrongAugConfig] = [],
            sampler: StyleSampler | list[StyleSampler] = [],
    ) -> SSDGDataset:
        if aug:
            augs = aug.lower().split("+")
        else:
            augs = []
        if any(a not in ["strong", "style"] for a in augs):
            raise ValueError(f"expected strong/style, got {aug}")

        n_style = augs.count("style")
        sampler = sampler if isinstance(sampler, list) else [sampler]
        if len(sampler) > n_style:
            raise ValueError(f"specified {len(sampler)} style samplers, "
                             f"got {len(sampler)}")
        while len(sampler) < n_style:
            # expect style but no sampler
            # use default sampler
            sampler.append(RandomStyleSampler(default_style_mode))
        # bind dataset with sampler
        [s.bind(self) for s in sampler if not s.bound]

        n_strong = augs.count("strong")
        strong = strong if isinstance(strong, list) else [strong]
        if len(strong) > n_strong:
            raise ValueError(f"specified {len(strong)} strong configs, "
                             f"got {len(strong)}")
        while len(strong) < n_strong:
            # expect strong but no config
            # use default config
            strong.append(default_strong.copy())

        self.style_samplers = sampler
        self.strong_configs = strong
        return self

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @property
    def lengths(self) -> list[int]:
        return [len(dataset) for dataset in self.datasets]

    def style_aug(self, img: torch.Tensor, domain: int, sampler: StyleSampler,
                  weak_fn, weak_args):
        """Apply style augmentation.

        Steps:
            1. sample a reference by sampler
            2. apply weak augmentation to the reference
            3. apply style augmentation to the image
        """
        fn = hist_match if sampler.mode == "hist" else amplitude_mixup

        ref, ref_domain, ref_id = sampler.sample(domain)
        ref = ref.transpose(1, 2, 0)

        if weak_fn:
            ref, *_ = weak_fn(ref, ref, *weak_args)
        ref = zoom(ref, (self.image_size / ref.shape[0],
                         self.image_size / ref.shape[1], 1),
                   order=0)
        ref = ref.transpose(2, 0, 1)
        ref = torch.from_numpy(ref).float()

        return fn(img, ref, **sampler.kwargs), ref_domain, ref_id

    def __getitem__(self, index):
        cnt = 0
        domain_id = -1
        for domain_id, dataset in enumerate(self.datasets):
            if index < cnt + len(dataset):
                break
            cnt += len(dataset)
        else:
            raise IndexError(f"out of range: {index}")

        img, mask, img_path = dataset[index - cnt]
        if self.mode == "val":
            if not self.with_indices:
                return (torch.from_numpy(img).float(),
                        torch.from_numpy(mask).float(), domain_id, img_path)
            return (index, torch.from_numpy(img).float(),
                    torch.from_numpy(mask).float(), domain_id, img_path)

        # c, h, w => h, w, c for weak augmentation
        img = img.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)

        weak_fn = None
        weak_args = tuple()
        if random.random() > 0.5:
            weak_fn = random_rot_flip
        elif random.random() > 0.5:
            weak_fn = random_rotate

        if weak_fn:
            img, mask, *weak_args = weak_fn(img, mask)

        img = zoom(img, (self.image_size / img.shape[0],
                         self.image_size / img.shape[1], 1),
                   order=0)
        mask = zoom(mask, (self.image_size / mask.shape[0],
                           self.image_size / mask.shape[1], 1),
                    order=0)

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        img_strong = []
        img_style = []
        cutmix_box = []
        domain_ids = []
        for config in self.strong_configs:
            img_s = deepcopy(img)
            if random.random() < config["color_jitter"]:
                img_s = T.ColorJitter(0.5, 0.5, 0.25)(img_s)  # type: ignore
            img_s = blur(img_s, p=config["blur"])
            img_strong.append(img_s)
            cutmix_box.append(
                obtain_cutmix_box(self.image_size, p=config["cutmix"]))

        for sampler in self.style_samplers:
            img_r, domain_r, _ = self.style_aug(img, domain_id, sampler,
                                                weak_fn, weak_args)
            img_style.append(img_r)
            domain_ids.append(domain_r)
            cutmix_box.append(
                obtain_cutmix_box(self.image_size, p=sampler.cutmix_prob))

        indices = [index] if self.with_indices else []
        return tuple(indices + [img] + img_strong + img_style + cutmix_box +
                     [domain_id] + domain_ids + [mask])

    def split_ulb_lb(self, fraction: float | int):
        lb_dataset = deepcopy(self)
        ulb_dataset = deepcopy(self)
        lb_dataset.mode = "train_l"
        ulb_dataset.mode = "train_u"

        if self.cls is MNMS:
            # for mnms, it is deterministic
            source = get_mnms_domains
            for dataset in [lb_dataset, ulb_dataset]:
                dataset.datasets = get_split_dataset(
                    self.root,
                    self.target_domain,
                    dataset.mode,
                    fraction,
                    source,
                    self.cls,
                )
            # not return indices for mnms
            return ulb_dataset, lb_dataset, []

        ulbs, lbs, indices = [], [], []
        for dataset in self.datasets:
            if type(fraction) is int:
                (lb, ulb), lb_indices = dataset.random_split_k(fraction)
                ulb_indices = lb_indices  # log any is fine
            elif type(fraction) is float:
                split, split_indices = dataset.random_split(fraction)
                lb, ulb = split
                _, ulb_indices = split_indices
            else:
                raise ValueError(f"expected float or int")
            ulbs.append(ulb)
            lbs.append(lb)
            indices.append(ulb_indices)

        ulb_dataset.datasets = ulbs
        lb_dataset.datasets = lbs
        return ulb_dataset, lb_dataset, indices

    def __repr__(self) -> str:
        return "\n".join([str(dataset) for dataset in self.datasets])


def get_split_dataset(
    root: str | Path,
    target_domain: int,
    mode: str,
    ratio: float | int,
    data_source_fn,
    cls,
):
    l1, l2, l3, u1, u2, u3, t = data_source_fn(
        root,
        target=target_domain,
        ratio=ratio,
    )
    if mode == "val":
        return [cls(t, domain_id=target_domain, stage="val")]
    elif mode == "train_l" or mode == "train":
        d = [i for i in range(4) if i != target_domain]
        return [cls(lb, i, stage="train") for lb, i in zip([l1, l2, l3], d)]
    elif mode == "train_u":
        d = [i for i in range(4) if i != target_domain]
        return [cls(ub, i, stage="train") for ub, i in zip([u1, u2, u3], d)]
    raise NotImplementedError
