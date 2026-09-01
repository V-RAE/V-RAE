"""Action DiT expert for the V-RAE video/action model.

The action stream follows the same conventions as :mod:`video_dit`: token
attention, RMS normalization, SwiGLU feed-forward layers, and cosine/sine 3-D
RoPE all come from the existing ``dit.blocks``/``rope3d`` modules.  Action
tokens use a one-dimensional position grid ``(t, 0, 0)``.  This first version
keeps the expert intentionally small: it has timestep-conditioned self-attention
and an MLP, but no cross-attention.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.blocks import (
    AdaLNZeroDDTBlock,
    RMSNorm,
    modulate,
)
from vrae.models.rope3d import (
    build_3d_positions,
    build_video_dit_3d_rope_cache_from_positions,
)

LOGGER = logging.getLogger(__name__)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Build the cosine/sine timestep features used by Wan22."""

    if int(dim) <= 0 or int(dim) % 2:
        raise ValueError(f"embedding dimension must be a positive even number, got {dim}")
    if position.ndim == 0:
        position = position[None]
    if position.ndim != 1:
        raise ValueError(f"position must be scalar or [B], got {tuple(position.shape)}")
    half = int(dim) // 2
    frequency = torch.pow(
        torch.tensor(10_000.0, device=position.device, dtype=torch.float64),
        -torch.arange(half, device=position.device, dtype=torch.float64) / half,
    )
    angle = torch.outer(position.to(torch.float64), frequency)
    return torch.cat((angle.cos(), angle.sin()), dim=-1).to(dtype=position.dtype)


class ActionHead(nn.Module):
    """Timestep-modulated normalization and linear projection for action tokens."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(int(hidden_dim), eps=float(eps))
        self.proj = nn.Linear(int(hidden_dim), int(out_dim))
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * int(hidden_dim)),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            condition = torch.zeros((x.shape[0], x.shape[-1]), device=x.device, dtype=x.dtype)
        if condition.ndim == 3:
            condition = condition[:, 0]
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift[:, None], scale[:, None]))


class ActionDiTBlock(AdaLNZeroDDTBlock):
    """Existing conditioned DDT block with action-friendly metadata."""

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float,
        *,
        rope_theta: float = 10_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        if int(hidden_dim) != int(attn_head_dim) * int(num_heads):
            raise ValueError(
                "ActionDiT requires hidden_dim == num_heads * attn_head_dim for the "
                f"shared attention path, got {hidden_dim}, {num_heads}, {attn_head_dim}"
            )
        super().__init__(
            hidden_size=int(hidden_dim),
            num_heads=int(num_heads),
            # AdaLNZeroDDTBlock derives its SwiGLU width as
            # ``2/3 * hidden_size * mlp_ratio``; choose the ratio so the
            # existing block realizes the requested ``ffn_dim``.
            mlp_ratio=1.5 * float(ffn_dim) / float(hidden_dim),
            rope_theta=float(rope_theta),
            attention_dropout=float(attention_dropout),
            attention_backend=attention_backend,
        )
        self.attn_head_dim = int(attn_head_dim)
        self.ffn_dim = int(ffn_dim)


class ActionDiT(nn.Module):
    """Timestep-conditioned diffusion Transformer over action token sequences."""

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        rope_theta: float = 10_000.0,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("action_dim", action_dim),
            ("ffn_dim", ffn_dim),
            ("text_dim", text_dim),
            ("freq_dim", freq_dim),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("attn_head_dim", attn_head_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"`{name}` must be positive, got {value}")
        if int(freq_dim) % 2:
            raise ValueError(f"`freq_dim` must be even, got {freq_dim}")
        if int(attn_head_dim) % 2:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if int(hidden_dim) != int(num_heads) * int(attn_head_dim):
            raise ValueError("hidden_dim must equal num_heads * attn_head_dim")

        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.eps = float(eps)
        self.rope_theta = float(rope_theta)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    self.hidden_dim,
                    self.attn_head_dim,
                    self.num_heads,
                    self.ffn_dim,
                    self.eps,
                    rope_theta=self.rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = ActionHead(self.hidden_dim, self.action_dim, self.eps)

        # The action sequence is a 1-D video grid.  Buffers are non-persistent so
        # checkpoints keep only learned weights, matching VRAEVideoDiT caches.
        positions = build_3d_positions(1024, 1, 1)
        cosine, sine = build_video_dit_3d_rope_cache_from_positions(
            positions,
            self.attn_head_dim,
            theta=self.rope_theta,
            dtype=torch.float32,
        )
        self.register_buffer("_rope_cosine", cosine, persistent=False)
        self.register_buffer("_rope_sine", sine, persistent=False)
        self.register_buffer("_positions", positions, persistent=False)

    def _apply(self, fn: Any) -> "ActionDiT":
        return super()._apply(fn)

    def get_freqs(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if int(seq_len) <= 0 or int(seq_len) > self._rope_cosine.shape[0]:
            raise ValueError(f"action sequence length must be in [1, {self._rope_cosine.shape[0]}]")
        parameter = next(self.parameters())
        return (
            self._rope_cosine[:seq_len].to(device=parameter.device, dtype=parameter.dtype),
            self._rope_sine[:seq_len].to(device=parameter.device, dtype=parameter.dtype),
        )

    def _positions_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self._positions[:seq_len].to(device=device)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            str(key)
            for key in keys
            if not any(str(key).startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required")
        expert = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            return expert
        path = Path(action_dit_pretrained_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ActionDiT checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"ActionDiT checkpoint must be a mapping, got {type(payload)}")
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("ActionDiT checkpoint is missing mapping `meta`")
        expected = {
            "hidden_dim": int(action_dit_config["hidden_dim"]),
            "ffn_dim": int(action_dit_config["ffn_dim"]),
            "num_layers": int(action_dit_config["num_layers"]),
            "num_heads": int(action_dit_config["num_heads"]),
            "attn_head_dim": int(action_dit_config["attn_head_dim"]),
            "text_dim": int(action_dit_config["text_dim"]),
            "freq_dim": int(action_dit_config["freq_dim"]),
            "eps": float(action_dit_config["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"ActionDiT checkpoint is missing meta.{key}")
            if key == "eps":
                valid = abs(float(meta[key]) - expected[key]) <= 1e-12
            else:
                valid = int(meta[key]) == expected[key]
            if not valid:
                raise ValueError(f"ActionDiT meta.{key} mismatch: expected {expected[key]}, got {meta[key]}")
        backbone = payload.get("backbone_state_dict")
        if not isinstance(backbone, dict):
            raise ValueError("ActionDiT checkpoint is missing mapping `backbone_state_dict`")
        expected_keys = cls.backbone_key_set(expert.state_dict().keys())
        provided_keys = set(backbone)
        missing = sorted(expected_keys - provided_keys)
        unexpected = sorted(provided_keys - expected_keys)
        if missing or unexpected:
            raise ValueError(f"ActionDiT backbone key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
        merged = expert.state_dict()
        for key in expected_keys:
            value = backbone[key]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(merged[key].shape):
                raise ValueError(f"ActionDiT checkpoint tensor mismatch for {key}")
            merged[key] = value.to(device=merged[key].device, dtype=merged[key].dtype)
        expert.load_state_dict(merged, strict=True)
        LOGGER.info("Loaded ActionDiT backbone from %s", path)
        return expert

    def _prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(f"action_tokens must be [B,T,{self.action_dim}], got {tuple(action_tokens.shape)}")
        if context.ndim != 3 or context.shape[0] != action_tokens.shape[0]:
            raise ValueError(f"context must be [B,L,D] with matching batch, got {tuple(context.shape)}")
        if context.shape[-1] != self.text_dim:
            raise ValueError(f"context last dimension must be {self.text_dim}, got {context.shape[-1]}")
        batch_size, seq_len = action_tokens.shape[:2]
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"timestep must be [B] or [1], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training timestep length must match action batch size")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        elif context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(f"context_mask must match context [B,L], got {tuple(context_mask.shape)}")
        timestep_features = sinusoidal_embedding_1d(self.freq_dim, timestep)
        time_embedding = self.time_embedding(timestep_features)
        return {
            "tokens": self.action_encoder(action_tokens),
            "freqs": self.get_freqs(seq_len),
            "positions": self._positions_for(seq_len, action_tokens.device),
            "t": time_embedding,
            # MoT consumes the condition and lets each AdaLN block create its
            # own six-way modulation, exactly like VRAEVideoDiT.
            "t_mod": time_embedding[:, None],
            "context": context,
            "context_mask": context_mask[:, None, :].expand(-1, seq_len, -1),
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self._prepare(action_tokens, timestep, context, context_mask)

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t"])

    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        state = self._prepare(action_tokens, timestep, context, context_mask)
        return (
            state["tokens"],
            state["t"],
            state["t_mod"],
            state["context"],
            state["context_mask"],
            state["freqs"],
        )

    def post(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self._prepare(action_tokens, timestep, context, context_mask)
        value = state["tokens"]
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(
                    lambda current: block(
                        current,
                        state["t"],
                        state["positions"],
                        rope_cache=state["freqs"],
                    ),
                    value,
                    use_reentrant=False,
                )
            else:
                value = block(
                    value,
                    state["t"],
                    state["positions"],
                    rope_cache=state["freqs"],
                )
        return self.post_dit(value, state)


DiTBlock = ActionDiTBlock

__all__ = ["ActionDiT", "ActionDiTBlock", "ActionHead", "DiTBlock", "sinusoidal_embedding_1d"]
