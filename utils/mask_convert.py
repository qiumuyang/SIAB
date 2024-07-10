"""Convert one-hot mask to multi-class mask.

[B, C, H, W] => [B, H, W] range in [0, C - 1]
"""

from __future__ import annotations
import numpy as np

import torch


def scgm_convert(mask: torch.Tensor):
    # mask: b, 1, h, w
    return mask.squeeze(1).long()


def prostate_convert(mask: torch.Tensor):
    # mask: b, 1, h, w
    return mask.squeeze(1).long()


def mnms_convert(mask: torch.Tensor):
    # mask: b, 3, h, w
    bg = mask.sum(dim=1, keepdim=True) == 0
    return torch.cat([bg, mask], dim=1).argmax(dim=1).long()


def fundus_convert(mask: torch.Tensor):
    # mask: b, 2, h, w
    # possible values: [0, 0], [0, 1], [1, 1]
    # since there is no [1, 0], we can use sum to get the class
    return mask.sum(dim=1).long()


def fundus_pred_mask(pred: torch.Tensor, mask: torch.Tensor, cls: int):
    # exists overlap between disc and cup
    # when evaluating disc, treat cup as part of disc
    # 0: bg, 1: disc - cup, 2: cup
    if cls != 1:
        p = pred == (cls + 1)
        m = mask == (cls + 1)
    else:
        cup_p = pred == 2
        cup_m = mask == 2
        disc_p = pred == 1
        disc_m = mask == 1
        p = cup_p | disc_p
        m = cup_m | disc_m
    return p, m


def shared_pred_mask(pred: torch.Tensor, mask: torch.Tensor, cls: int):
    return pred == (cls + 1), mask == (cls + 1)


def prostate_to_image(image: torch.Tensor) -> np.ndarray:
    assert image.ndim <= 3
    image = image[1, :, :]
    image = image.squeeze()
    return (image * 255).cpu().numpy().astype(np.uint8)


def shared_to_image(image: torch.Tensor) -> np.ndarray:
    assert image.ndim <= 3
    image = image.squeeze()
    if image.ndim == 3:
        image = image.permute(1, 2, 0)
    return (image * 255).cpu().numpy().astype(np.uint8)


def to_label(label: torch.Tensor, n_classes: int) -> np.ndarray:
    assert label.ndim == 2
    return (label / n_classes * 255).cpu().numpy().astype(np.uint8)


# the label fetched from dataloader is [c, h, w]
# where c is the number of different foreground classes
# use convert to make it [h, w] (class indice)
converter = {
    "scgm": scgm_convert,
    "mnms": mnms_convert,
    "fundus": fundus_convert,
    "prostate": prostate_convert,
}

pred_mask = {
    "fundus": fundus_pred_mask,
    "scgm": shared_pred_mask,
    "mnms": shared_pred_mask,
    "prostate": shared_pred_mask,
}

to_image = {
    "prostate": prostate_to_image,
    "fundus": shared_to_image,
    "scgm": shared_to_image,
    "mnms": shared_to_image,
}
