# modified from https://github.com/xmed-lab/EPL_SemiDG/blob/master/mms_dataloader.py

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .base_dataset import BaseDataset

image_prefix = "mnms_split_2D_data/Labeled/"
mask_prefix = "mnms_split_2D_mask/Labeled/"
re_prefix = "mnms_split_2D_re/Labeled/"

domain_suffix = [
    ["vendorA"],
    ["vendorB/center2", "vendorB/center3"],
    ["vendorC"],
    ["vendorD"],
]
num_labels = [[95], [74, 51], [50], [50]]

resampling_rate = 1.2


def get_all_data_folder(
    dataset_dir: str | Path,
    target: int,
    ratio: float | int,
):
    random.seed(14)  # following prior work

    if target < 0 or target > 3:
        raise ValueError('Wrong test vendor!')
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    image_dir = [[(dataset_dir / (image_prefix + s)).as_posix()
                  for s in suffix] for suffix in domain_suffix]
    label_dir = [[(dataset_dir / (mask_prefix + s)).as_posix() for s in suffix]
                 for suffix in domain_suffix]
    re_dir = [[(dataset_dir / (re_prefix + s)).as_posix() for s in suffix]
              for suffix in domain_suffix]

    domain_shared = [image_dir, label_dir, re_dir, num_labels]
    domain_split = [[
        item for source, item in enumerate(shared) if source != target
    ] for shared in domain_shared]
    target_domain = [item[target] for item in domain_shared]

    img_dirs, label_dirs, re_dirs, num_samples = domain_split
    test_data_dirs, test_mask_dirs, test_re, _ = target_domain

    # order matters (affect random)
    # first labeled, then unlabeled, then test
    labeled_datasets = [
        MNMSDomain(img_dir,
                   label_dir,
                   re_dir,
                   domain_label=i,
                   num_label=num,
                   train=True,
                   labeled=True,
                   lb=ratio)
        for i, (
            img_dir, label_dir, re_dir,
            num) in enumerate(zip(img_dirs, label_dirs, re_dirs, num_samples))
    ]
    unlabeled_datasets = [
        MNMSDomain(img_dir,
                   label_dir,
                   re_dir,
                   domain_label=i,
                   train=True,
                   labeled=False,
                   lb=ratio)
        for i, (img_dir, label_dir,
                re_dir) in enumerate(zip(img_dirs, label_dirs, re_dirs))
    ]
    test_dataset = MNMSDomain(test_data_dirs,
                              test_mask_dirs,
                              test_re,
                              train=False,
                              labeled=True)
    return tuple(labeled_datasets + unlabeled_datasets + [test_dataset])


def make_dataset(dir) -> list[str]:
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images


class MNMSDomain(Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        mask_dirs: list[str],
        reso_dirs: list[str],
        train: bool = True,
        domain_label: int | None = None,
        num_label: list[int] | None = None,
        labeled=True,
        lb=1.0,
    ):
        temp_imgs = []
        temp_masks = []
        temp_re = []

        if not train:
            lb = 1.0

        for num_set in range(len(data_dirs)):
            reso_paths = sorted(make_dataset(reso_dirs[num_set]))
            data_paths = sorted(make_dataset(data_dirs[num_set]))
            mask_paths = sorted(make_dataset(mask_dirs[num_set]))
            if labeled:
                # for num_data in range(len(data_paths)):
                #     if train:
                #         n_label = str(math.ceil(num_label[num_set] * lb + 1))
                #         if '00' + n_label == data_paths[num_data][
                #                 -10:-7] or '0' + n_label == data_paths[
                #                     num_data][-10:-7]:
                #             break

                #     for num_mask in range(len(mask_paths)):
                #         if data_paths[num_data][-10:-4] == mask_paths[
                #                 num_mask][-10:-4]:
                #             temp_re.append(reso_paths[num_data])
                #             temp_imgs.append(data_paths[num_data])
                #             temp_masks.append(mask_paths[num_mask])
                #             domain_labels.append(domain_label)
                #             num_label_data += 1
                # the above nest loop is to find mask for each image
                # if not found, the image will not be added
                # the inner loop can be replaced by a dictionary:
                mask_dict = {
                    mask_path[-10:-4]: mask_path
                    for mask_path in mask_paths
                }
                # now the optimized version:
                for image_path, reso_path in zip(data_paths, reso_paths):
                    # do the lb ulb split at the end of the loop
                    # if train:
                    #     assert num_label is not None
                    #     if type(lb) is int:
                    #         n_label = str(lb)
                    #     else:
                    #         n_label = str(math.ceil(num_label[num_set] * lb + 1))
                    #     # use zfill instead of '00' + n_label
                    #     if n_label.zfill(3) == image_path[-10:-7]:
                    #         break

                    mask_path = mask_dict.get(image_path[-10:-4])
                    if mask_path is not None:
                        temp_re.append(reso_path)
                        temp_imgs.append(image_path)
                        temp_masks.append(mask_path)

                if train:
                    if type(lb) is int:
                        new_len = lb
                    else:
                        new_len = int(len(temp_imgs) * lb)

                    temp_re = temp_re[:new_len]
                    temp_imgs = temp_imgs[:new_len]
                    temp_masks = temp_masks[:new_len]
            else:
                temp_re.extend(reso_paths)
                temp_imgs.extend(data_paths)

        self.reso = temp_re
        self.imgs = temp_imgs
        self.masks = temp_masks

        if domain_label == 0:
            self.one_hot_domain_label = torch.tensor([[1], [0], [0]])
        elif domain_label == 1:
            self.one_hot_domain_label = torch.tensor([[0], [1], [0]])
        elif domain_label == 2:
            self.one_hot_domain_label = torch.tensor([[0], [0], [1]])
        else:
            self.one_hot_domain_label = torch.tensor([[0], [0], [0]])

        self.new_size = 288
        self.labeled = labeled
        self.train = train

    def __getitem__(self, index):
        path_re = self.reso[index]
        re = np.load(path_re)['arr_0'][0]
        path_img = self.imgs[index]
        img = np.load(path_img)['arr_0']

        # Intensity cropping:
        p5 = np.percentile(img.flatten(), 0.5)
        p95 = np.percentile(img.flatten(), 99.5)
        img = np.clip(img, p5, p95)

        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        crop_size = 300

        img_tensor = F.to_tensor(np.array(img))
        img_size = img_tensor.size()

        if self.labeled:
            path_mask = self.masks[index]
            mask = Image.open(path_mask)  # numpy, HxWx3
            # resize and center-crop to 280x280

            # Find the region of mask
            norm_mask = F.to_tensor(np.array(mask))
            region = norm_mask[0] + norm_mask[1] + norm_mask[2]
            non_zero_index = torch.nonzero(region == 1, as_tuple=False)
            if region.sum() > 0:
                len_m = len(non_zero_index[0])
                x_region = non_zero_index[len_m // 2][0]
                y_region = non_zero_index[len_m // 2][1]
                x_region = int(x_region.item())
                y_region = int(y_region.item())
            else:
                x_region = norm_mask.size(-2) // 2
                y_region = norm_mask.size(-1) // 2

            resize_order = re / resampling_rate
            resize_size_h = int(img_size[-2] * resize_order)
            resize_size_w = int(img_size[-1] * resize_order)

            left_size = 0
            top_size = 0
            right_size = 0
            bot_size = 0
            if resize_size_h < self.new_size:
                top_size = (self.new_size - resize_size_h) // 2
                bot_size = (self.new_size - resize_size_h) - top_size
            if resize_size_w < self.new_size:
                left_size = (self.new_size - resize_size_w) // 2
                right_size = (self.new_size - resize_size_w) - left_size

            transform_list = [
                transforms.Pad((left_size, top_size, right_size, bot_size))
            ]
            transform_list = [
                transforms.Resize((resize_size_h, resize_size_w))
            ] + transform_list
            transform_list = [transforms.ToPILImage()] + transform_list
            transform = transforms.Compose(transform_list)
            img = transform(img)
            img = F.to_tensor(np.array(img))

            # Define the crop index
            if top_size >= 0:
                top_crop = 0
            else:
                if x_region > self.new_size // 2:
                    if x_region - self.new_size // 2 + self.new_size <= norm_mask.size(
                            -2):
                        top_crop = x_region - self.new_size // 2
                    else:
                        top_crop = norm_mask.size(-2) - self.new_size
                else:
                    top_crop = 0

            if left_size >= 0:
                left_crop = 0
            else:
                if y_region > self.new_size // 2:
                    if y_region - self.new_size // 2 + self.new_size <= norm_mask.size(
                            -1):
                        left_crop = y_region - self.new_size // 2
                    else:
                        left_crop = norm_mask.size(-1) - self.new_size
                else:
                    left_crop = 0

            # crop to 224x224
            img = F.crop(img, top_crop, left_crop, self.new_size,
                         self.new_size)

            # resize and center-crop to 280x280
            # transform_mask_list = [transforms.CenterCrop((crop_size, crop_size))]
            transform_mask_list = [
                transforms.Pad((left_size, top_size, right_size, bot_size))
            ]
            transform_mask_list = [
                transforms.Resize((resize_size_h, resize_size_w),
                                  interpolation=InterpolationMode.NEAREST)
            ] + transform_mask_list
            transform_mask = transforms.Compose(transform_mask_list)

            mask = transform_mask(mask)  # C,H,W
            mask = F.crop(
                mask,  # type: ignore
                top_crop,
                left_crop,
                self.new_size,
                self.new_size)
            mask = F.to_tensor(np.array(mask))

            mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
            mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
            mask = torch.cat((mask, mask_bg), dim=0)

            return img, mask, path_img

        else:
            img = F.to_pil_image(img)

            # resize and center-crop to 280x280
            resize_order = re / resampling_rate
            resize_size_h = int(img_size[-2] * resize_order)
            resize_size_w = int(img_size[-1] * resize_order)

            left_size = 0
            top_size = 0
            right_size = 0
            bot_size = 0
            if resize_size_h < crop_size:
                top_size = (crop_size - resize_size_h) // 2
                bot_size = (crop_size - resize_size_h) - top_size
            if resize_size_w < crop_size:
                left_size = (crop_size - resize_size_w) // 2
                right_size = (crop_size - resize_size_w) - left_size

            transform_list = [transforms.CenterCrop((crop_size, crop_size))]
            transform_list = [
                transforms.Pad((left_size, top_size, right_size, bot_size))
            ] + transform_list
            transform_list = [
                transforms.Resize((resize_size_h, resize_size_w))
            ] + transform_list
            transform = transforms.Compose(transform_list)

            img = transform(img)

            # random crop to 224x224
            top_crop = random.randint(0, crop_size - self.new_size)
            left_crop = random.randint(0, crop_size - self.new_size)
            img = F.crop(img, top_crop, left_crop, self.new_size,
                         self.new_size)
            img = F.to_tensor(np.array(img))

            return img, self.one_hot_domain_label.squeeze(), path_img

    def __len__(self):
        return len(self.imgs)


class MNMS(BaseDataset):
    """Stub like"""
    def __init__(self, dataset: MNMSDomain, domain_id: int, stage: str):
        self.domain = domain_id
        self.dataset = dataset
        self.stage = stage

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        image, label, path_img = self.dataset[index]

        if label.ndim == 1:
            # domain label instead of segmentation label
            # replace with zeros
            label = torch.zeros(3, 288, 288)
        else:
            # mnms puts background at the last channel
            # just discard it
            label = label[:-1]
        return image.numpy(), label.numpy(), Path(path_img).name
