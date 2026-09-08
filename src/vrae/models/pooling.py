from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from vrae.registry import POOLERS


@POOLERS.decorator("temporal_attention")
class TemporalAttentionPool(nn.Module):
    """Same-spatial, non-overlapping temporal attention pooling."""

    def __init__(
        self,
        dim: int,
        group_size: int,
        num_heads: int = 16,
        use_time_bias: bool = True,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = int(config.get("dim", dim))
            group_size = int(config.get("group_size", group_size))
            num_heads = int(config.get("num_heads", num_heads))
            use_time_bias = bool(config.get("use_time_bias", use_time_bias))
            if config.get("output_norm_affine", False) is not False:
                raise ValueError("TemporalAttentionPool output norm must be non-affine")
        self.dim = int(dim)
        self.group_size = int(group_size)
        self.num_heads = int(num_heads)
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.num_heads <= 0 or self.dim % self.num_heads:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm_k = nn.LayerNorm(self.dim)
        self.query = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.norm_out = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.time_bias = (
            nn.Parameter(torch.zeros(self.num_heads, self.group_size)) if use_time_bias else None
        )
        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, dim: int) -> TemporalAttentionPool:
        return cls(
            dim=dim,
            group_size=int(config["group_size"]),
            num_heads=int(config.get("num_heads", 16)),
            use_time_bias=bool(config.get("use_time_bias", True)),
            config=config,
        )

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm_k.weight)
        nn.init.zeros_(self.norm_k.bias)
        nn.init.zeros_(self.query)
        for layer in (self.key, self.value, self.proj):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.time_bias is not None:
            nn.init.zeros_(self.time_bias)

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,Te,C,H,W], got {tuple(x.shape)}")
        batch, time, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}")
        if time % self.group_size:
            raise ValueError(
                f"Encoder time {time} must be divisible by group_size={self.group_size}"
            )

        output_time = time // self.group_size
        # Group consecutive temporal tokens:
        # [B, Te, C, H, W] -> [B, T_out, G, C, H, W]
        grouped = x.reshape(
            batch,
            output_time,
            self.group_size,
            channels,
            height,
            width,
        )
        # Flatten batch, output-time, and spatial locations. Each location
        # contains only the G tokens from one temporal group.
        grouped = (
            grouped.permute(
                0,
                1,
                4,
                5,
                2,
                3,
            )
            .reshape(
                batch * output_time * height * width,
                self.group_size,
                channels,
            )
        )

        locations = grouped.shape[0]

        # Project keys and values. Only the key input is normalized.
        key = (
            self.key(self.norm_k(grouped))
            .reshape(
                locations,
                self.group_size,
                self.num_heads,
                self.head_dim,
            )
        )

        value = (
            self.value(grouped)
            .reshape(
                locations,
                self.group_size,
                self.num_heads,
                self.head_dim,
            )
        )

        # Use the standard multi-head layout [L, heads, G, head_dim].
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # Expand the shared query across all locations without copying it.
        query = self.query.expand(
            locations,
            -1,
            -1,
            -1,
        )
        # Compute one query-to-key score for each token in the group:
        # [L, heads, 1, head_dim] x [L, heads, G, head_dim]
        # -> [L, heads, G]
        logits = (query * key).sum(-1) * self.scale

        # Add an optional bias for each relative position within the group.
        if self.time_bias is not None:
            time_bias = self.time_bias.to(
                dtype=logits.dtype,
                device=logits.device,
            )

            logits = logits + time_bias[None]

        # Normalize attention weights over the temporal group dimension.
        attention = logits.softmax(dim=-1)
        # Aggregate values and merge the attention heads back into C.
        pooled = (attention.unsqueeze(-1) * value).sum(dim=2)
        pooled = pooled.reshape(locations, channels)

        # Apply the output projection and non-affine LayerNorm.
        pooled = self.norm_out(self.proj(pooled))

        # Restore the output layout [B, T_out, C, H, W].
        pooled = pooled.reshape(
            batch,
            output_time,
            height,
            width,
            channels,
        )

        pooled = (
            pooled.permute(
                0,
                1,
                4,
                2,
                3,
            )
            .contiguous()
        )

        if not return_attention:
            return pooled

        # Return attention in [B, T_out, H, W, heads, G] layout when requested.
        attention = attention.reshape(
            batch,
            output_time,
            height,
            width,
            self.num_heads,
            self.group_size,
        )

        return pooled, attention
