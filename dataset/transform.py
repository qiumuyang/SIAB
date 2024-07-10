from __future__ import annotations

import random
from typing import Callable, TypeVar

import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy import ndimage
from skimage.exposure import match_histograms

Fn = TypeVar("Fn", bound=Callable)


def random_rot_flip(img, mask, k=None, axis=None):
    if k is None:
        k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    if axis is None:
        axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask, k, axis


def random_rotate(img, mask, angle=None):
    if angle is None:
        angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask, angle


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        radius = np.random.randint(2, 4) * 2 + 1
        return F.gaussian_blur(img, [radius] * 2, [sigma] * 2)
    return img


def obtain_cutmix_box(img_size,
                      p=0.5,
                      size_min=0.02,
                      size_max=0.3,
                      ratio_1=0.3,
                      ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask


def hist_match(
        source: torch.Tensor,
        reference: torch.Tensor,
) -> torch.Tensor:
    """Histogram matching.

    Args:
        source: (C, H, W)
        reference: (C, H, W)
    """
    # Note: reference can be empty (i.e. all zeros), skip in this case
    if (reference == 0).all():
        return source.clone()

    dev = source.device
    source = source.cpu().numpy()
    reference = reference.cpu().numpy()
    return torch.from_numpy(match_histograms(source, reference,
                                             channel_axis=0)).to(dev)


def amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0.0):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    # deep copy
    a_local_copy = a_local.copy()
    a_local[:, h1:h2, w1:w2] = (a_local[:, h1:h2, w1:w2] * (1 - ratio) +
                                a_trg[:, h1:h2, w1:w2] * ratio)
    a_trg[:, h1:h2, w1:w2] = (a_trg[:, h1:h2, w1:w2] * (1 - ratio) +
                              a_local_copy[:, h1:h2, w1:w2] * ratio)

    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    a_trg = np.fft.ifftshift(a_trg, axes=(-2, -1))
    return a_local, a_trg


def amplitude_mixup_np(
    local_img,
    trg_img,
    L=0.01,
    lo=1.0,
    hi=1.0,
):
    if lo != hi:
        ratio = np.random.uniform(lo, hi)
    else:
        ratio = lo

    local_img_np = local_img
    tar_img_np = trg_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(tar_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_, amp_trg_ = amp_spectrum_swap(amp_local,
                                             amp_target,
                                             L=L,
                                             ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    fft_trg_ = amp_trg_ * np.exp(1j * pha_trg)
    trg_in_local = np.fft.ifft2(fft_trg_, axes=(-2, -1))
    trg_in_local = np.real(trg_in_local)

    return local_in_trg, trg_in_local


def amplitude_mixup_torch(
        source: torch.Tensor,
        reference: torch.Tensor,
        L: float = 0.01,
        lo: float = 1.0,
        hi: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        source: (C, H, W)
        reference: (C, H, W)
    """
    if lo == hi:
        lam = lo
    else:
        lam = torch.empty(1).uniform_(lo, hi).item()

    src_fft = torch.fft.fft2(source, dim=(1, 2))
    mix_fft = torch.fft.fft2(reference, dim=(1, 2))
    src_amp, src_pha = torch.abs(src_fft), torch.angle(src_fft)
    mix_amp = torch.abs(mix_fft)

    src_amp = torch.fft.fftshift(src_amp, dim=(1, 2))
    mix_amp = torch.fft.fftshift(mix_amp, dim=(1, 2))

    h, w = source.shape[-2:]
    d = round(min(h, w) * L)
    c_h, c_w = h // 2, w // 2
    t = c_h - d
    b = c_h + d + 1
    l = c_w - d
    r = c_w + d + 1

    src_amp[:, t:b, l:r] = (mix_amp[:, t:b, l:r] * lam + src_amp[:, t:b, l:r] *
                            (1 - lam))

    src_amp = torch.fft.ifftshift(src_amp, dim=(1, 2))

    src_fft = src_amp * torch.exp(1j * src_pha)

    source = torch.fft.ifft2(src_fft, dim=(1, 2))
    source = source.real
    return torch.clamp(source, 0, 1)


def amplitude_mixup(
        source: torch.Tensor,
        reference: torch.Tensor,
        L: float = 0.01,
        lo: float = 1.0,
        hi: float = 1.0,
        engine: str = 'torch',
) -> torch.Tensor:
    # Note: reference can be empty (i.e. all zeros), skip in this case
    if (reference == 0).all():
        return source.clone()

    if engine == 'torch':
        return amplitude_mixup_torch(source, reference, L, lo, hi)
    elif engine == 'numpy':
        local_img = source.cpu().numpy()
        trg_img = reference.cpu().numpy()
        local_in_trg, _ = amplitude_mixup_np(local_img, trg_img, L, lo, hi)
        local_in_trg = local_in_trg.astype(np.float32)
        return torch.from_numpy(local_in_trg).to(source.device)
    else:
        raise ValueError(f"Unsupported engine {engine}")
