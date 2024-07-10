from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .ssdg_dataset import SSDGDataset

Mode = Literal["fourier", "hist"]


class StyleSampler(ABC):
    """Sampler for style reference.

    Attributes:
        mode: style augmentation mode
        kwargs: extra arguments for style augmentation
        datasets: where to sample the style reference
    """

    def __init__(self, mode: Mode, **kwargs):
        self.mode: Mode = mode
        self.kwargs = kwargs
        self.datasets = []
        self.cutmix_prob = 0.5

    def set_cutmix_prob(self, cutmix_prob):
        self.cutmix_prob = cutmix_prob
        return self

    def bind(self, dataset: SSDGDataset):
        self.datasets = dataset.datasets
        return self

    @property
    def bound(self):
        return len(self.datasets) > 0

    @property
    def n_domains(self) -> int:
        return len(self.datasets)

    @property
    def n_samples(self) -> int:
        return sum(len(d) for d in self.datasets)

    @abstractmethod
    def _sample_index(self, domain: int, **kwargs) -> tuple[int, int]:
        """The inner implementation of ref sampling.

        Returns:
            (domain_id, sample_index relative to the current domain)
        """

    def sample(self, domain: int) -> tuple[np.ndarray, int, int]:
        """Sample a style reference from a domain.

        Returns:
            (image, domain_id, sample_index global)
        """
        domain_id, index = self._sample_index(domain)
        image = self.datasets[domain_id][index][0]
        return image, domain_id, index + sum(
            len(d) for d in self.datasets[:domain_id])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class RandomStyleSampler(StyleSampler):
    """Random sample a reference.

    Attributes:
        exclude_self: exclude the same domain, only for balanced=True
        balanced: whether fairly sample from each domain
    """

    def __init__(
        self,
        mode: Mode,
        exclude_self: bool = False,
        balanced: bool = True,
        **kwargs,
    ):
        StyleSampler.__init__(self, mode, **kwargs)

        self.exclude_self = exclude_self
        self.balanced = balanced

    def _sample_index(self, domain: int, **kwargs) -> tuple[int, int]:
        if not self.balanced:
            # sample from all domains
            # this will override exclude_self
            global_id = np.random.choice(self.n_samples)
            cum_sum = 0
            domain_id = -1
            ref_id = -1
            for domain_id, dataset in enumerate(self.datasets):
                cum_sum += len(dataset)
                if cum_sum > global_id:
                    ref_id = global_id - (cum_sum - len(dataset))
                    break
            return domain_id, ref_id

        other_domains = [
            i for i in range(self.n_domains)
            if i != domain or not self.exclude_self
        ]
        ref_domain = random.choice(other_domains)
        ref_dataset = self.datasets[ref_domain]
        return ref_domain, random.randint(0, len(ref_dataset) - 1)
