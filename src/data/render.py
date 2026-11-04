"""Batched top-down table renderer on the GPU (Representation B).

Each ball is drawn as an anti-aliased filled circle in the channel of its role relative
to the shooter: [cue, eight, own, opponent, open-table]. This is equivalent to a colored
image where each role has its own color, but keeps roles linearly separable.
"""
from __future__ import annotations

import torch

from src.data.dataset import BALL_R, TABLE_L, TABLE_W


class TableRenderer(torch.nn.Module):
    """Render (B,16,2) ball positions + (B,16,5) roles into a (B,5,H,W) image."""

    def __init__(self, height: int = 256, width: int = 128):
        super().__init__()
        ys = (torch.arange(height) + 0.5) / height * TABLE_L
        xs = (torch.arange(width) + 0.5) / width * TABLE_W
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)
        self.px = TABLE_L / height

    def forward(self, pos: torch.Tensor, roles: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        dx = self.gx[None, None] - pos[..., 0, None, None]
        dy = self.gy[None, None] - pos[..., 1, None, None]
        d = torch.sqrt(dx * dx + dy * dy)
        disk = torch.sigmoid((BALL_R - d) / (0.5 * self.px)) * on[..., None, None]
        return torch.einsum("bnhw,bnc->bchw", disk, roles)
