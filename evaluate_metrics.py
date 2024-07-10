from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import asd, dc, hd95
from torch.utils.data import DataLoader
from typing_extensions import TypeVar

from model.ema import EMA
from model.siab import SIABWrapper
from utils.mask_convert import converter, pred_mask, to_image, to_label

Collector = TypeVar("Collector", bound="ResultCollector")


class ResultCollector(ABC):

    def __init__(self, metric: str, n_classes: int, n_domains: int):
        self.metric = metric
        self.n_classes = n_classes
        self.n_domains = n_domains

    @abstractmethod
    def update(self, instance_metric_per_cls, domain_id):
        raise NotImplementedError

    @abstractmethod
    def get(self):
        raise NotImplementedError


class VolumewiseCollector(ResultCollector):

    def __init__(self, metric: str, n_classes: int, n_domains: int):
        super().__init__(metric, n_classes, n_domains)
        self.domain_instances: list[list[list[float]]] = [
            [] for _ in range(n_domains)
        ]  # (domain, instance, class)

    @property
    def averaged_instances(self) -> list[list[float]]:
        """Return the instance metric averaged over all classes."""
        return [[
            sum(inst) / len(inst) if len(inst) > 0 else 0 for inst in instances
        ] for instances in self.domain_instances]

    def update(self, instance_metric_per_cls, domain_id):
        k = 100 if self.metric.lower() in ["dice", "iou", "jaccard"] else 1
        self.domain_instances[domain_id].append(
            [v * k for v in instance_metric_per_cls])

    def get(self):
        sum_domain_class = [[
            sum(inst[c] for inst in instances) for c in range(self.n_classes)
        ] for instances in self.domain_instances]
        count_domain = [len(v) for v in self.domain_instances]

        metric_domain_classwise = []
        metric_domain_mean = []
        mean_metric_sum = 0
        mean_metric_cnt = 0

        for sum_class, cnt in zip(sum_domain_class, count_domain):
            if cnt == 0:
                continue

            classwise = [v / cnt for v in sum_class]
            mean = sum(classwise) / len(classwise)

            metric_domain_classwise.append(classwise)
            metric_domain_mean.append(mean)
            mean_metric_sum += sum(classwise)
            mean_metric_cnt += len(classwise)

        mean_dice = mean_metric_sum / mean_metric_cnt
        return mean_dice, metric_domain_mean, metric_domain_classwise


def siab_enabled(model) -> bool:
    if isinstance(model, EMA):
        name = model.module.__class__.__name__
    else:
        name = model.__class__.__name__
    return name == SIABWrapper.__name__


def evaluate_volume_metrics(
    model,
    dataloader: DataLoader,
    cfg: dict,
    is_target_domain: bool = False,
    verbose: bool = False,
    eval: bool = True,
    save_dir: Path | None = None,
    collector_cls: type[Collector] = VolumewiseCollector,
) -> list[Collector]:
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

    ret = evaluate_with_pred_function_volume(pred_fn,
                                             dataloader,
                                             cfg,
                                             is_next_volume,
                                             has_siab,
                                             is_target_domain,
                                             save_dir,
                                             collector_cls=collector_cls)
    if verbose:
        print(num_volumes)
    return ret


def evaluate_slice_metrics(
    model,
    dataloader: DataLoader,
    cfg: dict,
    is_target_domain: bool = False,
    eval: bool = True,
    save_dir: Path | None = None,
    penalty: int | None = None,
):
    if eval:
        model.eval()

    has_siab = siab_enabled(model)

    pred_fn = model

    # treat each slice as a volume, i.e., slice-wise evaluation
    def is_next_volume(path):
        return True

    return evaluate_with_pred_function_volume(
        pred_fn,
        dataloader,
        cfg,
        is_next_volume,
        has_siab,
        is_target_domain,
        save_dir,
        collector_cls=VolumewiseCollector,
        dist_penalty=penalty)


@torch.no_grad()
def evaluate_with_pred_function_volume(
    pred_fn,
    dataloader: DataLoader,
    cfg: dict,
    is_next_volume: Callable[[str], bool],
    has_siab: bool = False,
    is_target_domain: bool = False,
    save_dir: Path | None = None,
    dist_penalty: int | None = None,
    collector_cls: type[Collector] = VolumewiseCollector,
) -> list[Collector]:
    # pred_fn: image tensor -> logits tensor

    convert = converter[cfg["dataset"]]

    n_domains = cfg["n_domains"]
    n_classes = cfg["n_classes"] - 1  # exclude background
    size = cfg["image_size"]

    metrics = [("dice", dc, {}), ("asd", asd, {}), ("hd", hd95, {})]
    results = [
        collector_cls(metric, n_classes, n_domains) for metric, *_ in metrics
    ]
    dist_metric = ["asd", "assd", "hd"]
    dist_penalty = dist_penalty or 2

    current_volume_pred = []
    current_volume_mask = []
    domain = -1
    num_samples = 0
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
            result = np.zeros((len(metrics), n_classes))
            for cls in range(n_classes):
                p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
                p = p.cpu().numpy().astype(np.bool8)
                m = m.cpu().numpy().astype(np.bool8)
                for i, (metric, fn, kwargs) in enumerate(metrics):
                    if metric not in dist_metric:
                        result[i, cls] = fn(p, m, **kwargs)
                    else:
                        if p.sum() == 0 and m.sum() == 0:
                            result[i, cls] = 0
                        elif p.sum() == 0 or m.sum() == 0:
                            result[i, cls] = dist_penalty
                        else:
                            result[i, cls] = fn(p, m, **kwargs)

            for i, _ in enumerate(metrics):
                results[i].update(result[i], domain)

            # clear current volume
            current_volume_pred = []
            current_volume_mask = []

        pred = pred.argmax(dim=1)
        mask = convert(mask)
        current_volume_pred.append(pred)
        current_volume_mask.append(mask)
        if save_dir is not None:
            im = to_image[cfg["dataset"]](img[0])
            lb = to_label(mask[0], n_classes)
            pd = to_label(pred[0], n_classes)
            # dump as pdf
            if cfg["dataset"] == "prostate":
                im_kwargs = {"cmap": "gray"}
            else:
                im_kwargs = {}
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(im, **im_kwargs)
            axes[1].imshow(lb)
            axes[2].imshow(pd)
            fig.tight_layout()
            fig.savefig((save_dir / f"{num_samples:04d}.pdf").as_posix())
            plt.close(fig)

        num_samples += 1

    # update last volume
    if len(current_volume_pred) > 0:
        vol_pred = torch.cat(current_volume_pred, dim=0)
        vol_mask = torch.cat(current_volume_mask, dim=0)
        result = np.zeros((len(metrics), n_classes))
        for cls in range(n_classes):
            p, m = pred_mask[cfg["dataset"]](vol_pred, vol_mask, cls)
            p = p.cpu().numpy()
            m = m.cpu().numpy()
            for i, (metric, fn, kwargs) in enumerate(metrics):
                if metric not in dist_metric:
                    result[i, cls] = fn(p, m, **kwargs)
                else:
                    if p.sum() == 0 and m.sum() == 0:
                        result[i, cls] = 0
                    elif p.sum() == 0 or m.sum() == 0:
                        result[i, cls] = dist_penalty
                    else:
                        result[i, cls] = fn(p, m, **kwargs)

        for i, _ in enumerate(metrics):
            results[i].update(result[i], domain)

        # clear current volume
        current_volume_pred = []
        current_volume_mask = []

    # get the final result
    return results
