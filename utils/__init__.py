import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn


def get_train_config_from_log(log: Path) -> dict:
    lines = log.read_text().splitlines()
    # find multi-lines starts with cfg: and ends with empty line
    i = 0
    while i < len(lines):
        if "cfg:" in lines[i]:
            break
        i += 1
    else:
        return {}
    start = i
    i = 0
    while i < len(lines):
        if not lines[i]:
            break
        i += 1
    else:
        return {}
    end = i
    return eval("\n".join(lines[start + 1:end]))


def fix_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)  # for both CPU and CUDA
    random.seed(seed)  # for Python random module
    np.random.seed(seed)  # for NumPy


def sigmoid_rampup(current: int, rampup_length: int = 200):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class DiceLoss(nn.Module):

    def __init__(self, n_classes, p: int = 2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.p = p

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        if self.p == 2:
            y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
            z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        else:
            y_sum = torch.sum(target[ignore != 1])
            z_sum = torch.sum(score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax="", ignore=None, onehot=True):
        """
        Args:
            inputs: (N, C, H, W)
            target: (N, H, W) or (N, C, H, W)
            softmax: how to process the input
            ignore: (N, H, W) or (N, C, H, W)
            onehot: whether to convert the target (N, H, W) to onehot
        """
        if softmax == "softmax":
            inputs = inputs.softmax(dim=1)
        elif softmax == "sigmoid":
            inputs = inputs.sigmoid()
        elif softmax == "":
            pass
        else:
            raise ValueError("softmax should be softmax or sigmoid or empty")
        if onehot:
            if target.ndim == 3:
                target = target.unsqueeze(1)
            target = self._one_hot_encoder(target)

        if ignore is not None and ignore.ndim == 3:
            ignore = ignore.unsqueeze(1).expand(-1, self.n_classes, -1, -1)

        if inputs.size() != target.size():
            raise ValueError(
                f"predict & target shape do not match: {inputs.size()}, {target.size()}"
            )

        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(
                inputs[:, i],
                target[:, i],
                ignore[:, i] if ignore is not None else None,
            )
            loss += dice
        return loss / self.n_classes

    def instance_dice(self, inputs, target, argmax: bool = True):
        with torch.no_grad():
            if argmax:
                inputs = inputs.argmax(dim=1)
            if inputs.size() != target.size():
                raise ValueError(
                    f"predict & target shape do not match: {inputs.size()}, {target.size()}"
                )
            smooth = 1e-5
            dice = torch.zeros(inputs.size(0), device=inputs.device)
            for cls in range(1, self.n_classes):
                pred = inputs == cls
                gt = target == cls
                dice += ((pred * gt).sum((1, 2)) * 2.0 + smooth) / (pred.sum(
                    (1, 2)) + gt.sum((1, 2)) + smooth)
            return dice / (self.n_classes - 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO) -> logging.Logger:
    if (name, level) in logs:
        return  # type: ignore
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
