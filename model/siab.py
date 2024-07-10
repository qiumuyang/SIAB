from __future__ import annotations

from contextlib import contextmanager
from random import choice
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing_extensions import Literal, TypedDict

M = TypeVar("M", bound=nn.Module)


class GroupBy(TypedDict):
    sort: list[int]
    count: list[int]
    unsort: list[int]


def group_by(ids: list[int]) -> GroupBy:
    """Group the tensor by domain id.

    Args:
        ids: the domain id of each sample in the tensor.

    Return:
        dict(sort, count, unsort)
        sort: how to sort the tensor to make the domain id
            consecutively ascending.
        unsort: how to restore the original order of the tensor.
        count: the number of samples in each domain.
    """
    sort_ids = []
    counts = []
    for domain in range(max(ids) + 1):
        idx = [i for i, x in enumerate(ids) if x == domain]
        sort_ids.extend(idx)
        counts.append(len(idx))
    unsort_ids = [0] * len(sort_ids)
    for i, j in enumerate(sort_ids):
        unsort_ids[j] = i
    return {
        "sort": sort_ids,
        "count": counts,
        "unsort": unsort_ids,
    }


def setattr_recursive(obj, name, value):
    if "." in name:
        setattr_recursive(getattr(obj,
                                  name.split(".")[0]),
                          ".".join(name.split(".")[1:]), value)
    else:
        setattr(obj, name, value)


def getattr_recursive(obj, name):
    if "." in name:
        return getattr_recursive(getattr(obj,
                                         name.split(".")[0]),
                                 ".".join(name.split(".")[1:]))
    else:
        return getattr(obj, name)


class SIAB(nn.Module):
    """Statistics-individual and aggregated branches."""

    alpha_attrname = "alpha"

    def __init__(
        self,
        num_domains: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        global_in: bool = True,
        layer_id: int = -1,
    ) -> None:
        nn.Module.__init__(self)
        self.args = (num_features, eps, momentum, affine, track_running_stats)
        self.ind_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine,
                           track_running_stats) for _ in range(num_domains)
        ])

        self.global_in = global_in
        self.layer_id = layer_id

        if global_in:
            self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)
            self.alpha = Parameter(torch.zeros(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        bn: nn.BatchNorm2d
        for bn in self.ind_bn:  # type: ignore
            bn.reset_running_stats()
        if self.global_in:
            self.instance_norm.reset_running_stats()

    def reset_parameters(self) -> None:
        bn: nn.BatchNorm2d
        for bn in self.ind_bn:  # type: ignore
            bn.reset_parameters()

        if self.global_in:
            self.instance_norm.reset_parameters()
            nn.init.constant_(self.alpha, 0)

    def forward(
        self,
        input: torch.Tensor,
        sort_index: list[int],
        unsort_index: list[int],
        num_per_domain: list[int],
        shuffle_prob: float = -1,
    ):
        """Forward input through the statistics-individual and
        aggregated branches.

        Args:
            input: the input batch (B, C, H, W) and is expected to be
                consecutive in domain id along the batch dimension.
        """
        # sort input by domain id and restore the original index after process
        input = input[sort_index]
        n_domains = len(self.ind_bn)
        eps = self.args[1]

        # forward input separately for each domain
        outputs = []
        i = 0
        for domain, num in enumerate(num_per_domain):
            if num == 0:
                continue

            x = input[i:i + num]
            if torch.rand(1) < shuffle_prob:
                # use another domain's individual BN (except the global one)
                domain = choice(
                    [x for x in range(n_domains - 1) if x != domain], )
                mean = x.mean(dim=(2, 3), keepdim=True)
                std = x.std(dim=(2, 3), keepdim=True)
                # affine with other domain's BN weight and bias
                weight = self.ind_bn[domain].weight.detach()  # type: ignore
                bias = self.ind_bn[domain].bias.detach()  # type: ignore
                x = (x - mean) / (std + eps)
                x = x * weight[None, :, None, None] + bias[None, :, None, None]
                outputs.append(x)
            elif domain == n_domains - 1 and self.global_in:
                bn_out = self.ind_bn[domain](x)
                in_out = self.instance_norm(x)
                # make sure alpha is in [0, 1]
                t = self.alpha.sigmoid()
                outputs.append(t * bn_out + (1 - t) * in_out)
            else:
                outputs.append(self.ind_bn[domain](x))
            i += num

        return torch.cat(outputs, dim=0)[unsort_index]

    @classmethod
    def from_BatchNorm2d(
        cls,
        bn: nn.BatchNorm2d,
        num_domains: int,
        global_in: bool = False,
        layer_id: int = 0,
    ) -> SIAB:
        args = (bn.num_features, bn.eps, bn.momentum, bn.affine,
                bn.track_running_stats)
        siab = cls(num_domains, *args, global_in=global_in, layer_id=layer_id)

        for domain_bn in siab.ind_bn:
            domain_bn.load_state_dict(bn.state_dict())

        siab.to(bn.weight.device)
        return siab

    @classmethod
    def convert_siab(
        cls,
        model: M,
        num_domains: int,
        num_global_in: int | Literal["all"] = 0,
        **kwargs,
    ) -> SIABWrapper[M]:
        """Convert all nn.BatchNorm2d in the model to SIAB."""
        if isinstance(model, SIABWrapper):
            raise ValueError(
                "The model has already been converted to SIAB model")

        # cannot modify the model.named_modules during iteration
        name_to_module = {}
        num_bn = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                global_in = num_global_in == "all" or num_bn < num_global_in
                name_to_module[name] = cls.from_BatchNorm2d(
                    module, num_domains, global_in, num_bn)
                num_bn += 1

        # replace the modules, name can be xx.yy.zz
        for name, module in name_to_module.items():
            setattr_recursive(model, name, module)

        return SIABWrapper(model, num_domains)

    @classmethod
    @contextmanager
    def stop_grad(cls, model: nn.Module):
        """Stop the BN weight and bias gradient of the model."""
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad_(False)
        yield
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad_(True)


class SIABWrapper(nn.Module, Generic[M]):

    def __init__(self, model: M, num_domains: int):
        nn.Module.__init__(self)
        self._model = model
        self._num_domains = num_domains

    def _check_input_domain(self, domain_id: int | list[int], dim: int):
        if domain_id == []:
            # use the last branch as default
            domain_id = -1
        if isinstance(domain_id, int):
            # broadcast to the batch dimension
            domain_id = [domain_id] * dim
        assert isinstance(domain_id, list)

        if len(domain_id) != dim:
            raise ValueError("domain_id and input must have the same length")
        # convert negative domain id to positive
        return [(x + self._num_domains) % self._num_domains for x in domain_id]

    def forward(
        self,
        input: torch.Tensor,
        domain_id: int | list[int] = [],
        random_layer: int | Literal["all"] = 0,
        p: float = 1.0,
        **kwargs,
    ):
        """Forward input through the SIAB model.

        Args:
            input: the input batch (B, C, H, W).
            domain_id: the domain id of each sample in the input.
                If it is a list, it must have the same length as the
                batch dimension of the input.
                If it is an int, it will be broadcasted to the batch dimension.
            random_layer: randomly selected domain in the first `random_layer`.
            p: the probability to use the randomly selected domain.
            **kwargs: other arguments for the model.
        """
        domain_id = self._check_input_domain(domain_id, dim=input.size(0))

        group = group_by(domain_id)

        def hook(module: SIAB, inputs):
            input = inputs[0]

            if module.layer_id == -1:
                raise ValueError("layer_id is not set")
            if random_layer == "all" or module.layer_id < random_layer:
                return input, group["sort"], group["unsort"], group["count"], p
            # disable random by setting p to 0
            return input, group["sort"], group["unsort"], group["count"], 0

        handles = []
        for module in self.modules():
            if isinstance(module, SIAB):
                handle = module.register_forward_pre_hook(hook)  # type: ignore
                handles.append(handle)
        output = self._model(input, **kwargs)
        [handle.remove() for handle in handles]
        return output

    @property
    def module(self) -> M:
        return self._model

    def init_bn_weight(self, coef: float = 0.5):
        # reverse sigmoid function
        alpha = np.log(coef / (1 - coef))
        for m in self.modules():
            if isinstance(m, SIAB) and m.global_in:
                assert isinstance(m.alpha, Parameter)
                nn.init.constant_(m.alpha, alpha)
        return self

    def separate_parameters(self) -> tuple[list[Parameter], list[Parameter]]:
        """Separate the normal network parameters and the statistics-aggregated
        branch mix weight parameters.
        """
        normal, alpha = [], []
        for param in self.parameters():
            if getattr(param, SIAB.alpha_attrname, False):
                alpha.append(param)
            else:
                normal.append(param)
        return normal, alpha
