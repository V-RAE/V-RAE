from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vrae.models.decoder import (
    ATTENTION_BACKENDS,
    _flash_attention,
    resolve_attention_backend,
)
from vrae.models.dit.rope import apply_rope


def as_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (int(value), int(value))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a length-2 sequence")
        pair = (int(value[0]), int(value[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {pair}")
    return pair


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1 + scale) + shift


def _fuse_legacy_linear_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    fused_name: str,
    legacy_names: tuple[str, ...],
) -> None:
    """Combine legacy Linear parameters along the output dimension."""
    for parameter_name in ("weight", "bias"):
        fused_key = f"{prefix}{fused_name}.{parameter_name}"
        legacy_keys = [
            f"{prefix}{legacy_name}.{parameter_name}"
            for legacy_name in legacy_names
        ]
        if fused_key in state_dict:
            continue
        if not all(key in state_dict for key in legacy_keys):
            continue
        state_dict[fused_key] = torch.cat(
            [state_dict[key] for key in legacy_keys],
            dim=0,
        )
        for key in legacy_keys:
            state_dict.pop(key)


class RMSNorm(nn.Module):
    """RMS normalization evaluated in fp32 and returned in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_float = value.float()
        normalized = value_float * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value.dtype)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.register_load_state_dict_pre_hook(self._load_legacy_w12)

    @staticmethod
    def _load_legacy_w12(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args: object,
    ) -> None:
        _fuse_legacy_linear_state(state_dict, prefix, "w12", ("w1", "w2"))
        _fuse_legacy_linear_state(
            state_dict,
            prefix,
            "w12",
            ("gate_proj", "up_proj"),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = self.w12(value).chunk(2, dim=-1)
        return self.w3(F.silu(first) * second)


# The shorter name matches the public DDT reference.
SwiGLUFFN = SwiGLUFeedForward


class GaussianFourierTimeEmbedding(nn.Module):
    """Gaussian Fourier features followed by the DDT timestep MLP."""

    def __init__(
        self,
        hidden_size: int,
        embedding_size: int = 256,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        self.frequencies = nn.Parameter(
            torch.randn(int(embedding_size), dtype=torch.float32) * float(scale),
            requires_grad=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * int(embedding_size), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed sample times, preserving ``[B]`` or ``[B,T]`` leading dims."""
        if time.ndim == 0:
            time = time[None]
        if time.ndim not in (1, 2):
            raise ValueError(f"time must have shape [B] or [B,T], got {tuple(time.shape)}")
        input_shape = time.shape
        flat_time = time.reshape(-1)
        angles = flat_time.float()[:, None] * self.frequencies.float()[None, :] * (2 * torch.pi)
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        parameter = self.mlp[0].weight
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        return self.mlp(features).reshape(*input_shape, -1)


class QKNormAttention(nn.Module):
    """Full self-attention with per-head Q/K RMSNorm followed by 3D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope_theta: float = 10_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden size {dim} must be divisible by {num_heads} heads")
        if not 0 <= float(attention_dropout) < 1:
            raise ValueError("attention_dropout must be in [0,1)")
        attention_backend = str(attention_backend).lower()
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Unknown attention backend: {attention_backend}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.rope_theta = float(rope_theta)
        self.attention_dropout = float(attention_dropout)
        self.attention_backend = attention_backend
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.register_load_state_dict_pre_hook(self._load_legacy_qkv)

    @staticmethod
    def _load_legacy_qkv(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args: object,
    ) -> None:
        _fuse_legacy_linear_state(state_dict, prefix, "qkv", ("q", "k", "v"))
        _fuse_legacy_linear_state(
            state_dict,
            prefix,
            "qkv",
            ("q_proj", "k_proj", "v_proj"),
        )

    def forward(
        self,
        value: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(f"attention input must be [B,L,D], got {tuple(value.shape)}")
        batch, length, _ = value.shape
        if freqs.shape[0] != length:
            raise ValueError(
                f"freqs must have shape [{length},1,R], got {tuple(freqs.shape)}"
            )
        qkv = self.qkv(value).reshape(
            batch,
            length,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
        k = apply_rope(k.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        dropout = self.attention_dropout if self.training else 0.0
        backend = resolve_attention_backend(
            self.attention_backend,
            q,
            training=self.training,
        )
        if backend != "sdpa" and (attention_mask is not None or dropout):
            if self.attention_backend != "auto":
                unsupported = "attention masks" if attention_mask is not None else "dropout"
                raise ValueError(f"{backend} does not support VideoDiT {unsupported}")
            backend = "sdpa"
        if backend == "sdpa":
            result = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=False,
            )
        else:
            result = _flash_attention(backend, q, k, v)
        result = result.transpose(1, 2).reshape(batch, length, self.dim)
        return self.proj(result)


# Backward-friendly name used by the public DDT implementation.
NormAttention = QKNormAttention


class DDTEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rope_theta: float = 10_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        mlp_hidden = max(1, int(2 / 3 * hidden_size * float(mlp_ratio)))
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = QKNormAttention(
            hidden_size,
            num_heads,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            attention_backend=attention_backend,
        )
        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden)

    def forward(
        self,
        value: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = value + self.attn(
            self.norm1(value),
            freqs,
            attention_mask,
        )
        return value + self.mlp(self.norm2(value))


class AdaLNZeroDDTBlock(DDTEncoderBlock):
    """A conditioned DDT block whose residual branches start at zero."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rope_theta: float = 10_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            mlp_ratio,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            attention_backend=attention_backend,
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition.ndim == 2:
            condition = condition[:, None]
        if condition.ndim != 3 or condition.shape[0] != value.shape[0]:
            raise ValueError("condition must be [B,D] or a broadcastable [B,L,D]")
        if condition.shape[1] not in {1, value.shape[1]}:
            raise ValueError("condition token count must be 1 or match the input sequence")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_modulation(condition).chunk(
            6, dim=-1
        )
        attended = self.attn(
            modulate(self.norm1(value), shift_a, scale_a),
            freqs,
            attention_mask,
        )
        value = value + gate_a * attended
        fed_forward = self.mlp(modulate(self.norm2(value), shift_m, scale_m))
        return value + gate_m * fed_forward


class DDTDecoderBlock(AdaLNZeroDDTBlock):
    pass


class DDTFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int | Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        patch_h, patch_w = as_pair(patch_size, "patch_size")
        self.patch_size = (patch_h, patch_w)
        self.norm = RMSNorm(hidden_size)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_h * patch_w * int(out_channels))
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 2:
            condition = condition[:, None]
        shift, scale = self.adaln_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(value), shift, scale))


class FramePatchEmbed(nn.Module):
    """Patch-embed latent frames while retaining rectangular spatial geometry."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int | Sequence[int] = 1,
    ) -> None:
        super().__init__()
        self.patch_size = as_pair(patch_size, "patch_size")
        self.proj = nn.Conv2d(
            int(in_channels),
            int(hidden_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.proj.weight.data = self.proj.weight.data.contiguous(memory_format=torch.channels_last)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"frame grid must be [B,C,H,W], got {tuple(value.shape)}")
        return self.proj(value).flatten(2).transpose(1, 2)


def video_tokens_to_grid(
    value: torch.Tensor,
    *,
    chunks: int,
    grid_size: tuple[int, int],
    channels: int,
    name: str = "video",
) -> tuple[torch.Tensor, int]:
    height, width = grid_size
    expected = (int(chunks), height * width, int(channels))
    if value.ndim != 4 or tuple(value.shape[1:]) != expected:
        raise ValueError(
            f"{name} must have shape [B,{expected[0]},{expected[1]},{expected[2]}], "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    batch = value.shape[0]
    grid = value.reshape(batch, chunks, height, width, channels)
    grid = grid.permute(0, 1, 4, 2, 3).reshape(batch * chunks, channels, height, width)
    return grid, batch


def embed_video_frames(
    embedder: FramePatchEmbed,
    grid: torch.Tensor,
    *,
    batch_size: int,
    chunks: int,
) -> torch.Tensor:
    embedded = embedder(grid)
    return embedded.reshape(batch_size, chunks * embedded.shape[1], embedded.shape[2])


def unpatchify_video_tokens(
    value: torch.Tensor,
    *,
    batch_size: int,
    chunks: int,
    patch_grid: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
) -> torch.Tensor:
    grid_h, grid_w = patch_grid
    patch_h, patch_w = patch_size
    expected_tokens = int(chunks) * grid_h * grid_w
    expected_dim = patch_h * patch_w * int(channels)
    if value.ndim != 3 or value.shape != (batch_size, expected_tokens, expected_dim):
        raise ValueError(
            f"patch output must be [{batch_size},{expected_tokens},{expected_dim}], "
            f"got {tuple(value.shape)}"
        )
    value = value.reshape(
        batch_size,
        chunks,
        grid_h,
        grid_w,
        patch_h,
        patch_w,
        channels,
    )
    value = value.permute(0, 1, 2, 4, 3, 5, 6)
    value = value.reshape(
        batch_size,
        chunks,
        grid_h * patch_h,
        grid_w * patch_w,
        channels,
    )
    return value.reshape(
        batch_size,
        chunks,
        grid_h * patch_h * grid_w * patch_w,
        channels,
    )


__all__ = [
    "AdaLNZeroDDTBlock",
    "DDTDecoderBlock",
    "DDTEncoderBlock",
    "DDTFinalLayer",
    "FramePatchEmbed",
    "GaussianFourierTimeEmbedding",
    "NormAttention",
    "QKNormAttention",
    "RMSNorm",
    "SwiGLUFFN",
    "SwiGLUFeedForward",
    "as_pair",
    "embed_video_frames",
    "modulate",
    "unpatchify_video_tokens",
    "video_tokens_to_grid",
]
