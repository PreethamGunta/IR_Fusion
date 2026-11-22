"""Small paired U-Net dehazer (fog -> clear) for the GATE-IR Stage-B ablation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDehazer(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.u4 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c4 = DoubleConv(base * 8, base * 4)
        self.u3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c3 = DoubleConv(base * 4, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c2 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))

        y = self.c4(torch.cat([self.u4(x4), x3], dim=1))
        y = self.c3(torch.cat([self.u3(y), x2], dim=1))
        y = self.c2(torch.cat([self.u2(y), x1], dim=1))
        return torch.sigmoid(self.out(y))


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-SSIM on [0,1] tensors (single-channel or RGB)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim.mean()
