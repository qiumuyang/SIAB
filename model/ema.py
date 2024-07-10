from __future__ import annotations

import warnings
from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        copy_methods: list[str] | None = None,
    ):
        self.decay = decay
        self.module = deepcopy(model)
        self.module.requires_grad_(False)

        for method in copy_methods or []:
            if hasattr(model, method):
                setattr(self, method, getattr(model, method))
            else:
                warnings.warn("method {} not found in {}".format(
                    method, model.__class__.__name__))

    def update(self, model: nn.Module, step: int, update_buffer: bool = False):
        decay = min(self.decay, 1 - 1 / (step + 1))
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(),
                                        self.module.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1 - decay)

            if not update_buffer:
                return

            # named buffers (only running_mean and running_var)
            for (name, buffer), (ema_name, ema_buffer) in \
                    zip(model.named_buffers(), self.module.named_buffers()):
                if name == ema_name and "running" in name:
                    ema_buffer.mul_(decay).add_(buffer, alpha=1 - decay)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()
