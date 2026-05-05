"""Saliency Sampler (Recasens et al., ECCV 2018), modernized for PyTorch >= 1.5.

Plug this in front of any task network. Given a high-resolution input it
produces a non-uniformly sampled view at the task network's input size,
magnifying the regions the saliency network deems important.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_2d(size: int, fwhm: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    sq_dist = (grid_x - center) ** 2 + (grid_y - center) ** 2
    return torch.exp(-4.0 * math.log(2.0) * sq_dist / fwhm ** 2)


class SaliencySampler(nn.Module):
    """Recasens-style saliency-based sampling layer.

    Args:
        task_fn: callable mapping a (B, 3, task_input_size, task_input_size)
            tensor to logits. Held as a plain attribute (not a submodule),
            so its parameters are not visible through ``self.parameters()``
            and thus not double-counted by the optimizer.
        saliency_network: backbone returning a (B, saliency_channels, h, w)
            feature map.
        saliency_channels: channel count of the backbone's output.
        saliency_scale: pre-softmax multiplier on saliency logits. Values >1
            sharpen the spatial softmax, producing a more aggressive zoom.
    """

    def __init__(
        self,
        task_fn: Callable[[torch.Tensor], torch.Tensor],
        saliency_network: nn.Module,
        saliency_channels: int,
        task_input_size: int = 224,
        saliency_input_size: int = 224,
        grid_size: int = 31,
        padding_size: int = 30,
        gaussian_fwhm: float = 13.0,
        saliency_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.task_fn = task_fn
        self.localization = saliency_network
        self.task_input_size = task_input_size
        self.saliency_input_size = saliency_input_size
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.global_size = grid_size + 2 * padding_size
        self.saliency_scale = saliency_scale

        self.conv_last = nn.Conv2d(saliency_channels, 1, kernel_size=1)

        # Gaussian kernel k(·) used to smooth the attractive mass in Eqs. (2)–(3).
        kernel_size = 2 * padding_size + 1
        kernel = _gaussian_2d(kernel_size, fwhm=gaussian_fwhm)
        self.register_buffer("gauss_kernel", kernel.view(1, 1, kernel_size, kernel_size))

        # Per-cell (x, y) coordinates used in the numerators of Eqs. (2)–(3).
        # Channel order MUST be (x, y) — F.grid_sample expects channel 0 to
        # index columns and channel 1 to index rows.
        coords = torch.arange(self.global_size, dtype=torch.float32)
        coord_y, coord_x = torch.meshgrid(coords, coords, indexing="ij")
        coord_x = (coord_x - padding_size) / (grid_size - 1.0)
        coord_y = (coord_y - padding_size) / (grid_size - 1.0)
        self.register_buffer("p_basis", torch.stack([coord_x, coord_y]))

    def _create_grid(self, sal_padded: torch.Tensor) -> torch.Tensor:
        # Compute Eqs. (2) and (3) of Recasens et al. with two F.conv2d calls.
        # The shared denominator and the (x, y) numerators are produced by
        # convolving (S) and (S * P_basis) with the Gaussian kernel respectively.
        B = sal_padded.size(0)
        P = self.p_basis.unsqueeze(0).expand(B, -1, -1, -1)

        denom = F.conv2d(sal_padded, self.gauss_kernel)
        weighted = (P * sal_padded).reshape(B * 2, 1, self.global_size, self.global_size)
        numer = F.conv2d(weighted, self.gauss_kernel).view(B, 2, self.grid_size, self.grid_size)

        u = numer[:, 0:1] / denom
        v = numer[:, 1:2] / denom
        # F.grid_sample expects coords in [-1, 1].
        grid = torch.cat([2.0 * u - 1.0, 2.0 * v - 1.0], dim=1).clamp(-1.0, 1.0)

        grid = F.interpolate(
            grid, size=self.task_input_size, mode="bilinear", align_corners=True
        )
        return grid.permute(0, 2, 3, 1)

    def forward(
        self, x: torch.Tensor, p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_low = F.adaptive_avg_pool2d(x, self.saliency_input_size)
        feat = F.relu(self.localization(x_low))
        sal = self.conv_last(feat)
        sal = F.interpolate(sal, size=self.grid_size, mode="bilinear", align_corners=True)
        sal = sal * self.saliency_scale
        sal = F.softmax(sal.flatten(1), dim=1).view_as(sal)
        sal_padded = F.pad(sal, [self.padding_size] * 4, mode="replicate")

        grid = self._create_grid(sal_padded)
        x_sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # Blur trick (paper §3.3): randomly downsample-then-upsample early in
        # training so the sampler is rewarded for zooming in. p=0 always
        # blurs, p=1 never does — flip with epoch count, not per-step.
        if random.random() > p:
            s = random.randint(64, self.task_input_size)
            x_sampled = F.adaptive_avg_pool2d(x_sampled, s)
            x_sampled = F.interpolate(
                x_sampled, size=self.task_input_size, mode="bilinear", align_corners=True
            )

        return self.task_fn(x_sampled), x_sampled, sal
