"""Action DiT expert for the V-RAE video/action model.

The action stream follows the same conventions as :mod:`video_dit`: token
attention, RMS normalization, SwiGLU feed-forward layers, and complex-frequency
3-D RoPE come from the local ``dit.rope`` module.  Action
tokens use a one-dimensional position grid ``(t, 0, 0)``.  This first version
keeps the expert intentionally small: it has timestep-conditioned self-attention
and an MLP, but no cross-attention.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.blocks import (
    AdaLNZeroDDTBlock,
    GaussianFourierTimeEmbedding,
    RMSNorm,
    modulate,
)
from vrae.models.dit.rope import RoPE3D


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


class ActionDiT(nn.Module):
    """Timestep-conditioned diffusion Transformer over action token sequences."""

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
        self.time_embedder = GaussianFourierTimeEmbedding(
            self.hidden_dim,
            embedding_size=self.freq_dim,
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroDDTBlock(
                    hidden_size=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=1.5 * self.ffn_dim / self.hidden_dim,
                    rope_theta=self.rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = ActionHead(self.hidden_dim, self.action_dim, self.eps)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        self.rope = RoPE3D(self.attn_head_dim, end=1024, theta=self.rope_theta)

    def get_freqs(self, seq_len: int) -> torch.Tensor:
        return self.rope(seq_len, 1, 1).to(device=next(self.parameters()).device)

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
        time_embedding = self.time_embedder(timestep.to(action_tokens.device))
        return {
            "tokens": self.action_encoder(action_tokens),
            "freqs": self.get_freqs(seq_len),
            "t": time_embedding,
            "t_mod": time_embedding,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._prepare(action_tokens, timestep, context, context_mask)
        return (
            state["tokens"],
            state["t"],
            state["t_mod"],
            state["context"],
            state["context_mask"],
            state["freqs"],
        )

    def post(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(tokens, condition)

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
                        state["freqs"],
                    ),
                    value,
                    use_reentrant=False,
                )
            else:
                value = block(
                    value,
                    state["t"],
                    state["freqs"],
                )
        return self.post_dit(value, state)

__all__ = ["ActionDiT", "ActionHead"]
