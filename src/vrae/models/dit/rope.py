"""Three-dimensional rotary positional embeddings for VideoDiT."""

from __future__ import annotations

import torch
from torch import nn


def _precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim))
    phases = torch.outer(torch.arange(end), frequencies)
    return torch.polar(torch.ones_like(phases), phases)


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE dimension must be a positive even number")
    pair_count = dim // 2
    base, remainder = divmod(pair_count, 3)
    pair_dims = [base + (index < remainder) for index in range(3)]
    frame_dim, height_dim, width_dim = (2 * pairs for pairs in pair_dims)
    return (
        _precompute_freqs_cis(frame_dim, end, theta),
        _precompute_freqs_cis(height_dim, end, theta),
        _precompute_freqs_cis(width_dim, end, theta),
    )


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply complex RoPE to ``x`` shaped ``[B, S, H * D]``."""
    batch_size, seq_len, channels = x.shape
    head_dim = channels // num_heads
    x_heads = x.reshape(batch_size, seq_len, num_heads, head_dim)
    x_complex = torch.view_as_complex(
        x_heads.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
    )
    rotated = torch.view_as_real(x_complex * freqs)
    return rotated.flatten(2).to(dtype=x.dtype)


class RoPE3D(nn.Module):
    """Provide token-aligned complex frequencies for a 3-D token grid."""

    def __init__(self, dim: int, end: int = 1024, theta: float = 10_000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.end = int(end)
        self.theta = float(theta)
        freq_f, freq_h, freq_w = precompute_freqs_cis_3d(
            self.dim, end=self.end, theta=self.theta
        )
        self.register_buffer("freq_f", freq_f, persistent=False)
        self.register_buffer("freq_h", freq_h, persistent=False)
        self.register_buffer("freq_w", freq_w, persistent=False)

    def get_freqs(self, f: int, h: int, w: int) -> torch.Tensor:
        if min(int(f), int(h), int(w)) <= 0:
            raise ValueError("RoPE grid dimensions must be positive")
        if max(int(f), int(h), int(w)) > self.end:
            raise ValueError(f"RoPE grid dimensions must be <= {self.end}")
        freqs = torch.cat(
            [
                self.freq_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        return freqs.reshape(f * h * w, 1, -1)

    forward = get_freqs


__all__ = ["RoPE3D", "apply_rope", "precompute_freqs_cis_3d"]
