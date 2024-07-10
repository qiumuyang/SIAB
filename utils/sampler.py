from __future__ import annotations

from typing import Iterator

import numpy
from torch.utils.data import Sampler, SubsetRandomSampler


def shuffled(items, shuffled) -> Iterator:
    indices = numpy.arange(len(items))
    if shuffled:
        numpy.random.shuffle(indices)
    for i in indices:
        yield items[i]


class MultiDomainSampler(Sampler):
    """Multi-domain sampler for SSDG dataset.

    The sampler will create a subset sampler for each domain, and sample from
    them in a round-robin(balanced) or random(unbalanced) manner.

    Args:
        lengths: The number of samples in each domain.
        shuffle: Whether to shuffle the order of domains.
        balanced: Whether to sample in a round-robin manner.
    """

    def __init__(
        self,
        lengths: list[int],
        shuffle: bool = True,
        balanced: bool = True,
    ):
        self.lengths = lengths
        self.base_index = numpy.cumsum(lengths)
        self.sub_samplers = [
            iter(SubsetRandomSampler(range(length))) for length in lengths
        ]
        self.shuffle = shuffle
        self.balanced = balanced

        self.total_length = sum(lengths)
        self.global_sampler = iter(
            SubsetRandomSampler(range(self.total_length)))

    def __iter__(self) -> Iterator:
        if self.balanced:
            while True:
                for i, sampler in shuffled(list(enumerate(self.sub_samplers)),
                                           self.shuffle):
                    base = self.base_index[i - 1] if i > 0 else 0
                    try:
                        yield next(sampler) + base
                    except StopIteration:
                        self.sub_samplers[i] = iter(
                            SubsetRandomSampler(range(self.lengths[i])))
                        yield next(self.sub_samplers[i]) + base
        else:
            while True:
                try:
                    yield next(self.global_sampler)
                except StopIteration:
                    self.global_sampler = iter(
                        SubsetRandomSampler(range(self.total_length)))
                    yield next(self.global_sampler)


class SingleDomainSampler(Sampler):
    """Single-domain sampler for SSDG dataset.

    The sampler only samples from one domain.

    Args:
        lengths: The number of samples in each domain.
    """
    def __init__(self, lengths: list[int], domain: int):
        self.lengths = lengths
        self.base_index = numpy.cumsum(lengths)
        self.sub_samplers = [
            iter(SubsetRandomSampler(range(length))) for length in lengths
        ]
        self.current_domain = domain

    def __iter__(self) -> Iterator:
        i = self.current_domain
        while True:
            base = self.base_index[i - 1] if i > 0 else 0
            try:
                yield next(self.sub_samplers[i]) + base
            except StopIteration:
                self.sub_samplers[i] = iter(
                    SubsetRandomSampler(range(self.lengths[i])))
                yield next(self.sub_samplers[i]) + base


__all__ = ["MultiDomainSampler", "SingleDomainSampler"]
