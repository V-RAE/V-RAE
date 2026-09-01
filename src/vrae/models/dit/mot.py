"""Mixed self-attention for V-RAE video and action encoder blocks.

MoT deliberately has no cross-attention in this first implementation.  Each
expert keeps its own AdaLN, attention output projection, and MLP; only the
self-attention Q/K/V tensors are concatenated.  When a V-RAE video expert is
passed in, its ``encoder_blocks`` are selected and its decoder blocks remain
outside the mixed path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.blocks import modulate
from vrae.models.rope3d import apply_video_dit_3d_rope_cache as apply_rope


class MoT(nn.Module):
    """Layer-wise mixed self-attention over video and action token experts."""

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = False,
    ) -> None:
        super().__init__()
        if not mixtures or "video" not in mixtures or "action" not in mixtures:
            raise ValueError("mixtures must include both 'video' and 'action' experts")
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        if self.mot_checkpoint_mixed_attn:
            raise ValueError("mixed-attention checkpointing is not supported")
        self.compile_training_layers = False

        video_blocks = self._blocks_for("video")
        self.num_layers = len(video_blocks)
        if self.num_layers <= 0:
            raise ValueError("video expert must expose at least one encoder block")
        self.num_heads = self._num_heads(video_blocks[0])
        self.attn_head_dim = self._head_dim(video_blocks[0])
        for name in self.expert_order:
            blocks = self._blocks_for(name)
            if len(blocks) != self.num_layers:
                raise ValueError(
                    f"MoT mixes action with video encoder depth; {name!r} has {len(blocks)} blocks, "
                    f"expected {self.num_layers}"
                )
            if self._num_heads(blocks[0]) != self.num_heads or self._head_dim(blocks[0]) != self.attn_head_dim:
                raise ValueError(f"expert {name!r} attention heads do not match video expert")
            if bool(getattr(self.mixtures[name], "use_gradient_checkpointing", False)):
                raise ValueError("expert gradient checkpointing is incompatible with MoT")

    @staticmethod
    def _blocks_for_expert(expert: nn.Module, name: str):
        if name == "video" and hasattr(expert, "encoder_blocks"):
            return expert.encoder_blocks
        if hasattr(expert, "blocks"):
            return expert.blocks
        raise ValueError(f"expert {name!r} must expose blocks")

    def _blocks_for(self, name: str):
        return self._blocks_for_expert(self.mixtures[name], name)

    @staticmethod
    def _attention(block: nn.Module) -> nn.Module:
        attention = getattr(block, "attn", None)
        if attention is None:
            raise ValueError("MoT blocks must use the existing `attn` attention module")
        return attention

    @classmethod
    def _num_heads(cls, block: nn.Module) -> int:
        return int(getattr(cls._attention(block), "num_heads", 0))

    @classmethod
    def _head_dim(cls, block: nn.Module) -> int:
        return int(getattr(cls._attention(block), "head_dim", 0))

    @staticmethod
    def _split_modulation(
        block: nn.Module,
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Create six AdaLN tensors from an existing V-RAE block condition."""

        if t_mod.ndim == 2:
            t_mod = t_mod[:, None]
        if t_mod.ndim == 3 and t_mod.shape[-1] > 0:
            condition = t_mod
        else:
            raise ValueError(f"t_mod must be [B,D] or [B,S,D], got {tuple(t_mod.shape)}")
        if not hasattr(block, "adaln_modulation"):
            raise ValueError("MoT blocks must expose existing `adaln_modulation`")
        values = block.adaln_modulation(condition).chunk(6, dim=-1)
        return tuple(values)

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if q_cat.shape != k_cat.shape or k_cat.shape != v_cat.shape:
            raise ValueError("mixed Q/K/V tensors must have matching [B,S,D] shapes")
        total = q_cat.shape[1]
        if attention_mask.shape != (total, total):
            raise ValueError(f"attention_mask must be [{total},{total}]")
        q = q_cat.reshape(q_cat.shape[0], total, self.num_heads, self.attn_head_dim).transpose(1, 2)
        k = k_cat.reshape(k_cat.shape[0], total, self.num_heads, self.attn_head_dim).transpose(1, 2)
        v = v_cat.reshape(v_cat.shape[0], total, self.num_heads, self.attn_head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask.to(device=q.device),
            dropout_p=0.0,
            is_causal=False,
        )
        return result.transpose(1, 2).reshape(q_cat.shape[0], total, -1)

    @staticmethod
    def _build_expert_attention_io(
        expert: nn.Module,
        block: nn.Module,
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = MoT._split_modulation(block, t_mod)
        attention = MoT._attention(block)
        normalized = modulate(block.norm1(x), shift_a, scale_a)
        batch, length = normalized.shape[:2]
        q = attention.q(normalized).reshape(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = attention.k(normalized).reshape(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        v = attention.v(normalized)
        q = attention.q_norm(q)
        k = attention.k_norm(k)
        q = apply_rope(q, freqs[0], freqs[1]).transpose(1, 2).reshape(batch, length, -1)
        k = apply_rope(k, freqs[0], freqs[1]).transpose(1, 2).reshape(batch, length, -1)
        return q, k, v, x, gate_a, shift_m, scale_m, gate_m

    @staticmethod
    def _apply_expert_post_block(
        block: nn.Module,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict] = None,
    ) -> torch.Tensor:
        del context_payload
        attention = MoT._attention(block)
        x = residual_x + gate_msa * attention.proj(mixed_attn_out)
        fed_forward = block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp * fed_forward

    _apply_expert_post_block_tensor = _apply_expert_post_block

    def _apply_post_with_optional_checkpoint(
        self,
        block: nn.Module,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        if use_gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda mixed, residual, ga, sm, xm, gm: self._apply_expert_post_block(
                    block, residual, mixed, ga, sm, xm, gm, context_payload
                ),
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return self._apply_expert_post_block(
            block, residual_x, mixed_slice, gate_msa, shift_mlp, scale_mlp, gate_mlp, context_payload
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        cache_k, cache_v = self.prefill_video_cache_tensor(
            video_tokens,
            video_freqs,
            video_t_mod,
            torch.empty(0, device=video_tokens.device, dtype=video_tokens.dtype),
            torch.empty(0, device=video_tokens.device, dtype=torch.bool),
            video_attention_mask,
        )
        del video_context_payload
        return [{"k": key, "v": value} for key, value in zip(cache_k, cache_v)]

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: tuple[torch.Tensor, torch.Tensor],
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        del action_context_payload
        return self.forward_action_with_video_cache_tensor(
            action_tokens,
            action_freqs,
            action_t_mod,
            torch.empty(0, device=action_tokens.device, dtype=action_tokens.dtype),
            torch.empty(0, device=action_tokens.device, dtype=torch.bool),
            [item["k"] for item in video_kv_cache],
            [item["v"] for item in video_kv_cache],
            attention_mask[int(video_seq_len) :, :],
        )

    def prefill_video_cache_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        del video_context, video_context_mask
        expert = self.mixtures["video"]
        x = video_tokens
        keys, values = [], []
        for block in self._blocks_for("video"):
            q, k, v, residual, gate_a, shift, scale, gate_m = self._build_expert_attention_io(
                expert, block, x, video_freqs, video_t_mod
            )
            x = self._apply_expert_post_block_tensor(
                block, residual, self._mixed_attention(q, k, v, video_attention_mask), gate_a, shift, scale, gate_m
            )
            keys.append(k)
            values.append(v)
        return keys, values

    def forward_action_with_video_cache_tensor(
        self,
        action_tokens: torch.Tensor,
        action_freqs: tuple[torch.Tensor, torch.Tensor],
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del action_context, action_context_mask
        if len(video_cache_k) != self.num_layers or len(video_cache_v) != self.num_layers:
            raise ValueError("video cache length must match MoT depth")
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx, block in enumerate(self._blocks_for("action")):
            q, k, v, residual, gate_a, shift, scale, gate_m = self._build_expert_attention_io(
                expert, block, x, action_freqs, action_t_mod
            )
            mixed = self._mixed_attention(
                q,
                torch.cat((video_cache_k[layer_idx], k), dim=1),
                torch.cat((video_cache_v[layer_idx], v), dim=1),
                action_attention_mask,
            )
            x = self._apply_expert_post_block_tensor(
                block, residual, mixed, gate_a, shift, scale, gate_m
            )
        return x

    def _forward_joint_layer(
        self,
        video_block: nn.Module,
        action_block: nn.Module,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: tuple[torch.Tensor, torch.Tensor],
        action_freqs: tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del video_context, video_context_mask, action_context, action_context_mask
        video_io = self._build_expert_attention_io(self.mixtures["video"], video_block, video_tokens, video_freqs, video_t_mod)
        action_io = self._build_expert_attention_io(self.mixtures["action"], action_block, action_tokens, action_freqs, action_t_mod)
        mixed = self._mixed_attention(
            torch.cat((video_io[0], action_io[0]), dim=1),
            torch.cat((video_io[1], action_io[1]), dim=1),
            torch.cat((video_io[2], action_io[2]), dim=1),
            attention_mask,
        )
        split = video_tokens.shape[1]
        return (
            self._apply_expert_post_block_tensor(video_block, video_io[3], mixed[:, :split], video_io[4], video_io[5], video_io[6], video_io[7]),
            self._apply_expert_post_block_tensor(action_block, action_io[3], mixed[:, split:], action_io[4], action_io[5], action_io[6], action_io[7]),
        )

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: tuple[torch.Tensor, torch.Tensor],
        action_freqs: tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = video_tokens.shape[1] + action_tokens.shape[1]
        if attention_mask.shape != (expected, expected):
            raise ValueError(f"attention_mask must be [{expected},{expected}]")
        video_blocks = self._blocks_for("video")
        action_blocks = self._blocks_for("action")
        for index in range(self.num_layers):
            video_tokens, action_tokens = self._forward_joint_layer(
                video_blocks[index], action_blocks[index], video_tokens, action_tokens,
                video_freqs, action_freqs, video_t_mod, action_t_mod,
                video_context, video_context_mask, action_context, action_context_mask,
                attention_mask,
            )
        return video_tokens, action_tokens

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del context_all
        if any(name not in embeds_all for name in self.expert_order):
            raise ValueError("embeds_all must contain every configured expert")
        if any(name not in freqs_all for name in self.expert_order):
            raise ValueError("freqs_all must contain every configured expert")
        if any(name not in t_mod_all for name in self.expert_order):
            raise ValueError("t_mod_all must contain every configured expert")
        tokens = dict(embeds_all)
        for layer_idx in range(self.num_layers):
            states = []
            q_parts, k_parts, v_parts = [], [], []
            for name in self.expert_order:
                block = self._blocks_for(name)[layer_idx]
                state = self._build_expert_attention_io(
                    self.mixtures[name], block, tokens[name], freqs_all[name], t_mod_all[name]
                )
                states.append((name, block, state))
                q_parts.append(state[0]); k_parts.append(state[1]); v_parts.append(state[2])
            mixed = self._mixed_attention(torch.cat(q_parts, 1), torch.cat(k_parts, 1), torch.cat(v_parts, 1), attention_mask)
            offset = 0
            for name, block, state in states:
                length = state[3].shape[1]
                tokens[name] = self._apply_expert_post_block_tensor(
                    block, state[3], mixed[:, offset:offset + length], state[4], state[5], state[6], state[7]
                )
                offset += length
        return tokens


__all__ = ["MoT"]
