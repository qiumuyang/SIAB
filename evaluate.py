from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.ema import EMA
from model.siab import SIABWrapper
from utils.mask_convert import converter, pred_mask


def siab_enabled(model) -> bool:
    if isinstance(model, EMA):
        name = model.module.__class__.__name__
    else:
        name = model.__class__.__name__
    return name == SIABWrapper.__name__


def evaluate_slice(
    model,
    dataloader: DataLoader,
    cfg: dict,
    is_target_domain: bool = False,
) -> tuple[float, list[float], list[list[float]]]:
    model.eval()

    has_siab = siab_enabled(model)

    pred_fn = model
    return evaluate_with_pred_function(pred_fn, dataloader, cfg, has_siab,
                                       is_target_domain)


def evaluate(
        model,
        dataloader: DataLoader,
        cfg: dict,
        is_target_domain: bool = False,
        verbose: bool = False,
        eval: bool = True,
) -> tuple[float, list[float], list[list[float]]]:
    if eval:
        model.eval()

    has_siab = siab_enabled(model)
    pred_fn = model

    current_volume = None
    num_volumes = 0

    if cfg["dataset"] == "fundus":

        # fundus is 2D dataset, treat each image as a volume
        def is_next_volume(path):  # type: ignore
            return True

    elif cfg["dataset"] == "mnms":
        # must be careful with this dataset
        # vendorB contains 2 centers, whose patient ids are overlapped
        # however, the volume ids are ascending for each center
        # and we visit the dataset center by center
        # so the volume can be correctly identified by the first 3 digits
        def is_next_volume(path: str):
            nonlocal current_volume
            nonlocal num_volumes
            # {patient_id:03d}{slice_id:03d}
            if current_volume != path[:3]:
                current_volume = path[:3]
                num_volumes += 1
                return True
            return False

    elif cfg["dataset"] == "prostate":

        def is_next_volume(path: str):
            nonlocal current_volume
            nonlocal num_volumes
            # {volume}_{slice}_{type}
            vol = path.split("_")[0]
            if current_volume != vol:
                current_volume = vol
                num_volumes += 1
                return True
            return False
    else:
        raise ValueError(f"Unknown dataset {cfg['dataset']}")

    ret = evaluate_with_pred_function_volume(pred_fn, dataloader, cfg,
                                             is_next_volume, has_siab,
                                             is_target_domain)
    if verbose:
        print(num_volumes)
    return ret


@torch.no_grad()
def evaluate_with_pred_function(
    pred_fn,
    dataloader: DataLoader,
    cfg: dict,
    has_siab: bool = False,
    is_target_domain: bool = False,
) -> tuple[float, list[float], list[list[float]]]:
    # pred_fn: image tensor -> logits tensor

    convert = converter[cfg["dataset"]]

    n_domains = cfg["n_domains"]
    n_classes = cfg["n_classes"] - 1  # exclude background
    size = cfg["image_size"]

    dice_sum_class_domain = [[0] * n_classes for _ in range(n_domains)]
    count_domain = [0] * n_domains

    for img, mask, domain, _ in dataloader:
        img, mask = img.cuda(), mask.cuda()
        domain = domain.item()

        h, w = img.shape[-2:]
        # down
        if h != size or w != size:
            img = F.interpolate(img, (size, size),
                                mode="bilinear",
                                align_corners=False)
        # pred
        if has_siab:
            # target domain uses the statistics-aggregated branch (id=-1)
            kwargs = dict(domain_id=-1 if is_target_domain else domain)
        else:
            kwargs = {}
        pred = pred_fn(img, **kwargs)
        # up
        if h != size or w != size:
            pred = F.interpolate(pred, (h, w),
                                 mode="bilinear",
                                 align_corners=False)

        eps = 1e-4
        pred = pred.argmax(dim=1)
        mask = convert(mask)
        for cls in range(n_classes):
            p, m = pred_mask[cfg["dataset"]](pred, mask, cls)
            inter = (p * m).sum().item()
            union = (p.sum().item() + m.sum().item())
            dice = (2.0 * inter + eps) / (union + eps)
            dice_sum_class_domain[domain][cls] += dice
        count_domain[domain] += 1

    dice_class_domain = []
    dice_mean_domain = []
    mean_dice_sum = 0
    mean_dice_cnt = 0

    for dice_sum_class, cnt in zip(dice_sum_class_domain, count_domain):
        if cnt == 0:
            continue

        dc = [d * 100.0 / cnt for d in dice_sum_class]
        dm = sum(dc) / len(dc)

        dice_class_domain.append(dc)
        dice_mean_domain.append(dm)
        mean_dice_sum += sum(dc)
        mean_dice_cnt += len(dc)

    mean_dice = mean_dice_sum / mean_dice_cnt
    return mean_dice, dice_mean_domain, dice_class_domain


@torch.no_grad()
def evaluate_with_pred_function_volume(
    pred_fn,
    dataloader: DataLoader,
    cfg: dict,
    is_next_volume: Callable[[str], bool],
    has_siab: bool = False,
    is_target_domain: bool = False,
) -> tuple[float, list[float], list[list[float]]]:
    # pred_fn: image tensor -> logits tensor

    convert = converter[cfg["dataset"]]

    n_domains = cfg["n_domains"]
    n_classes = cfg["n_classes"] - 1  # exclude background
    size = cfg["image_size"]

    dice_sum_class_domain = [[0] * n_classes for _ in range(n_domains)]
    count_domain = [0] * n_domains

    current_volume_pred = []
    current_volume_mask = []
    domain = -1
    for img, mask, domain, path in dataloader:
        img, mask = img.cuda(), mask.cuda()
        domain = domain.item()

        h, w = img.shape[-2:]
        # down
        if h != size or w != size:
            img = F.interpolate(img, (size, size),
                                mode="bilinear",
                                align_corners=False)
        # pred
        if has_siab:
            # target domain uses the statistics-aggregated branch (id=-1)
            kwargs = dict(domain_id=-1 if is_target_domain else domain)
        else:
            kwargs = {}
        pred = pred_fn(img, **kwargs)
        # up
        if h != size or w != size:
            pred = F.interpolate(pred, (h, w),
                                 mode="bilinear",
                                 align_corners=False)

        if is_next_volume(path[0]) and len(current_volume_pred) > 0:
            # update current volume to dice_sum_class_domain
            vol_pred = torch.cat(current_volume_pred, dim=0)
            vol_mask = torch.cat(current_volume_mask, dim=0)
            eps = 1e-4
            for cls in range(n_classes):
                p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
                inter = (p * m).sum().item()
                union = (p.sum().item() + m.sum().item())
                dice = (2.0 * inter + eps) / (union + eps)
                dice_sum_class_domain[domain][cls] += dice
            count_domain[domain] += 1
            # clear current volume
            current_volume_pred = []
            current_volume_mask = []

        pred = pred.argmax(dim=1)
        mask = convert(mask)
        current_volume_pred.append(pred)
        current_volume_mask.append(mask)

    # update last volume
    if len(current_volume_pred) > 0:
        vol_pred = torch.cat(current_volume_pred, dim=0)
        vol_mask = torch.cat(current_volume_mask, dim=0)
        eps = 1e-4
        for cls in range(n_classes):
            p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
            inter = (p * m).sum().item()
            union = (p.sum().item() + m.sum().item())
            dice = (2.0 * inter + eps) / (union + eps)
            dice_sum_class_domain[domain][cls] += dice
        count_domain[domain] += 1

    dice_class_domain = []
    dice_mean_domain = []
    mean_dice_sum = 0
    mean_dice_cnt = 0

    for dice_sum_class, cnt in zip(dice_sum_class_domain, count_domain):
        if cnt == 0:
            continue

        dc = [d * 100.0 / cnt for d in dice_sum_class]
        dm = sum(dc) / len(dc)

        dice_class_domain.append(dc)
        dice_mean_domain.append(dm)
        mean_dice_sum += sum(dc)
        mean_dice_cnt += len(dc)

    mean_dice = mean_dice_sum / mean_dice_cnt
    return mean_dice, dice_mean_domain, dice_class_domain
