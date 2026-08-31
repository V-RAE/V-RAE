"""Template for a Mixture-of-Transformers (MoT) module.

This file documents the interface and computation contract from
``fastwam_mot.py`` without providing executable implementation. Multiple
structurally compatible Transformer experts share mixed self-attention while
keeping their residual, cross-attention, and MLP parameters separate.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MoT(nn.Module):
    """Wrapper for mixed attention across multiple Transformer experts.

    The current design requires experts named ``"video"`` and ``"action"``;
    the generic ``forward`` method can process additional experts in the order
    supplied by ``mixtures``. Every expert must have the same number of layers,
    attention heads, and per-head dimension, while retaining its own blocks,
    modulation, cross-attention, and MLP parameters.

    At each layer, Q/K/V are built independently for every expert, concatenated
    along the token dimension, and passed through one mixed self-attention call.
    The result is split back by expert sequence length, then each expert applies
    its residual path, optional cross-attention, and MLP. This template contains
    no parameter registration, validation, or computation code.
    """

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = False,
    ):
        """Declare the expert collection and shared attention metadata.

        Args:
            mixtures: Mapping from expert names to Transformer modules. It must
                include ``"video"`` and ``"action"``. Each expert provides
                ``blocks``, ``num_heads``, and ``attn_head_dim``; each block
                provides modulation, norm1/norm2/norm3, self_attn, cross_attn,
                ffn, and gate components.
            mot_checkpoint_mixed_attn: Whether mixed attention checkpointing is
                requested. The compiled tensor-core path does not support it;
                expert-level gradient checkpointing is also incompatible with
                that path.

        Construction logic:
            1. Store the experts in an ``nn.ModuleDict`` and preserve their order.
            2. Read ``num_layers``, ``num_heads``, and ``attn_head_dim`` from the
               first expert and require all other experts to match.
            3. Record training and compilation switches; initialization may also
               log expert names and parameter counts.

        Output structure:
            An MoT module containing all experts. The constructor returns nothing.
        """
        ...

    @staticmethod
    def _split_modulation(
        block: nn.Module,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Merge block and timestep modulation, then split it into six parts.

        Args:
            block: Current Transformer block, with a ``modulation`` parameter
                typically shaped ``[1, 6, D]``.
            t_mod: Timestep modulation, usually ``[B, 6, D]`` or token-specific
                ``[B, S, 6, D]``.

        Construction logic:
            Move the block modulation to the timestep tensor's dtype and device,
            add the two tensors, and chunk along the modulation axis. The six
            chunks are self-attention shift/scale/gate followed by MLP
            shift/scale/gate. For token-specific modulation, remove the singleton
            axis produced by chunking so the values broadcast over ``[B, S, D]``.

        Returns:
            ``(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
        """
        ...

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one mixed FlashAttention operation over all expert tokens.

        Args:
            q_cat: Concatenated queries, ``[B, S_total, H * Dh]``.
            k_cat: Concatenated keys, ``[B, S_total, H * Dh]``.
            v_cat: Concatenated values, ``[B, S_total, H * Dh]``.
            attention_mask: Joint mask, ``[S_total, S_total]``; rows represent
                query tokens and columns represent visible key/value tokens.

        Construction logic:
            Move the mask to the query device and call FlashAttention with the
            shared head count, preserving expert order and token segmentation.

        Returns:
            Mixed attention output ``[B, S_total, H * Dh]``.
        """
        ...

    @staticmethod
    def _apply_expert_post_block(
        block: nn.Module,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply expert post-attention processing, optional cross-attention, and MLP.

        Args:
            block: Current expert block, exposing the self-attention output
                projection, gates, norm2/norm3, optional cross-attention, and FFN.
            residual_x: Tokens before self-attention, ``[B, S, D]``.
            mixed_attn_out: This expert's slice of mixed attention,
                ``[B, S, H * Dh]``.
            gate_msa: Gate for the self-attention residual branch.
            shift_mlp: Shift for MLP input modulation.
            scale_mlp: Scale for MLP input modulation.
            gate_mlp: Gate for the MLP residual branch.
            context_payload: Optional dictionary with ``context [B, L, D]`` and
                ``mask [B, S, L]`` or ``[B, 1, S, L]``. Skip cross-attention when
                the payload is absent or has no context.

        Construction logic:
            Project the mixed attention output and gate-add it to ``residual_x``;
            if context exists, add cross-attention on ``norm3(x)``; modulate
            ``norm2(x)`` with MLP shift/scale, run the FFN, and gate-add its result.

        Returns:
            Updated expert tokens ``[B, S, D]``.
        """
        ...

    @staticmethod
    def _apply_expert_post_block_tensor(
        block: nn.Module,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply post-processing for the fixed-tensor-context path.

        Args:
            block: Current expert block.
            residual_x: Tokens before self-attention, ``[B, S, D]``.
            mixed_attn_out: This expert's mixed-attention slice,
                ``[B, S, H * Dh]``.
            gate_msa: Self-attention residual gate.
            shift_mlp: MLP shift.
            scale_mlp: MLP scale.
            gate_mlp: MLP residual gate.
            context: Required cross-attention context, ``[B, L, D]``.
            context_mask: Context mask, usually ``[B, S, L]``; expand it to the
                attention kernel's head dimension as required.

        Construction logic:
            Follow the same residual, cross-attention, and MLP sequence as
            ``_apply_expert_post_block``. Context and mask are mandatory, so
            cross-attention is always performed and no optional checkpoint branch
            is used.

        Returns:
            Updated expert tokens ``[B, S, D]``.
        """
        ...

    def _build_expert_attention_io(
        self,
        expert: nn.Module,
        block: nn.Module,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build Q/K/V and post-processing state for one expert block.

        Args:
            expert: Owning expert; used only to read
                ``use_gradient_checkpointing``.
            block: Current layer block, ``expert.blocks[layer_idx]``.
            x: Current expert tokens, ``[B, S, D]``.
            freqs: Token-aligned RoPE frequencies, ``[S, 1, rope_dim]``.
            t_mod: Timestep modulation for this expert and layer.

        Construction logic:
            Split modulation and apply the MSA shift/scale to ``norm1(x)``;
            project q/k/v, RMS-normalize q and k, apply RoPE to q and k, and keep
            v unchanged. Preserve the original x and MLP modulation tensors, then
            read the expert checkpoint flag.

        Returns:
            ``(q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
            use_gradient_checkpointing)``. q/k/v are ``[B, S, H * Dh]`` and
            ``residual_x`` is ``[B, S, D]``.
        """
        ...

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
        """Choose direct or checkpointed execution for expert post-processing.

        Args:
            block: Current expert block.
            residual_x: Tokens before attention, ``[B, S, D]``.
            gate_msa: Self-attention gate.
            shift_mlp: MLP shift.
            scale_mlp: MLP scale.
            gate_mlp: MLP gate.
            use_gradient_checkpointing: Enable checkpointing only when training.
            mixed_slice: This expert's attention output, ``[B, S, H * Dh]``.
            context_payload: Optional cross-attention context/mask dictionary.

        Construction logic:
            Wrap ``_apply_expert_post_block`` as a re-entrant-compatible callable.
            When the expert requests checkpointing and the module is training,
            run the non-reentrant checkpoint path; otherwise call it directly.

        Returns:
            Updated tokens ``[B, S, D]``.
        """
        ...

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the video expert once and cache per-layer K/V for the action branch.

        Args:
            video_tokens: Video tokens before layer 0, ``[B, Sv, D]``.
            video_freqs: Video RoPE frequencies, ``[Sv, 1, rope_dim]``.
            video_t_mod: Video timestep modulation.
            video_context_payload: Optional video cross-attention payload with
                ``context [B, L, D]`` and ``mask [B, Sv, L]`` or
                ``[B, 1, Sv, L]``.
            video_attention_mask: Video-only self-attention mask, ``[Sv, Sv]``.

        Construction logic:
            For each layer, build video Q/K/V, run attention using only the video
            mask, apply post-processing, and save that layer's K/V. Feed updated
            video tokens into the next layer so the video sequence is prefetched once.

        Returns:
            A list of length ``num_layers``. Entry ``i`` is
            ``{"k": [B, Sv, H * Dh], "v": [B, Sv, H * Dh]}``.
        """
        ...

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run the action expert using prefetched video K/V tensors.

        Args:
            action_tokens: Action tokens before layer 0, ``[B, Sa, D]``.
            action_freqs: Action RoPE frequencies, ``[Sa, 1, rope_dim]``.
            action_t_mod: Action timestep modulation.
            action_context_payload: Optional action cross-attention context/mask.
            video_kv_cache: Per-layer cache from ``prefill_video_cache``. It must
                contain ``num_layers`` entries, each with video ``k`` and ``v``.
            attention_mask: Joint video+action mask, ``[Sv + Sa, Sv + Sa]``.
            video_seq_len: Number of video tokens in the joint-sequence prefix, ``Sv``.

        Construction logic:
            Recompute action q/k/v at every layer, select the action query rows from
            the joint mask, concatenate cached video K/V with current action K/V,
            run attention, and apply action post-processing.

        Returns:
            Final action tokens ``[B, Sa, D]``.
        """
        ...

    def prefill_video_cache_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prefill video K/V using mandatory tensor context inputs.

        Args:
            video_tokens: Video input tokens, ``[B, Sv, D]``.
            video_freqs: Video RoPE frequencies, ``[Sv, 1, rope_dim]``.
            video_t_mod: Video timestep modulation.
            video_context: Required video cross-attention context, ``[B, L, D]``.
            video_context_mask: Video context mask, usually ``[B, Sv, L]``.
            video_attention_mask: Video self-attention mask, ``[Sv, Sv]``.

        Construction logic:
            For every layer, run video Q/K/V, FlashAttention, and fixed-context
            post-processing. Store each layer's K and V in separate lists; this
            path does not use optional payloads or optional checkpoint handling.

        Returns:
            ``(video_cache_k, video_cache_v)``. Both lists have ``num_layers``
            entries shaped ``[B, Sv, H * Dh]``.
        """
        ...

    def forward_action_with_video_cache_tensor(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the action cache path with mandatory tensor context inputs.

        Args:
            action_tokens: Action input tokens, ``[B, Sa, D]``.
            action_freqs: Action RoPE frequencies, ``[Sa, 1, rope_dim]``.
            action_t_mod: Action timestep modulation.
            action_context: Required action cross-attention context, ``[B, L, D]``.
            action_context_mask: Action context mask, usually ``[B, Sa, L]``.
            video_cache_k: Per-layer video K list from
                ``prefill_video_cache_tensor``.
            video_cache_v: Per-layer video V list from
                ``prefill_video_cache_tensor``.
            action_attention_mask: Mask for action queries over video+action keys,
                usually ``[Sa, Sv + Sa]``.

        Construction logic:
            Recompute action Q/K/V at each layer, concatenate the corresponding
            video and action K/V, run FlashAttention, and apply fixed-context action
            post-processing.

        Returns:
            Updated action tokens ``[B, Sa, D]``.
        """
        ...

    def _forward_joint_layer(
        self,
        video_block: nn.Module,
        action_block: nn.Module,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one joint video/action mixed-attention layer.

        Args:
            video_block: Current video expert block.
            action_block: Current action expert block.
            video_tokens: Current video tokens, ``[B, Sv, D]``.
            action_tokens: Current action tokens, ``[B, Sa, D]``.
            video_freqs: Video RoPE frequencies, ``[Sv, 1, rope_dim]``.
            action_freqs: Action RoPE frequencies, ``[Sa, 1, rope_dim]``.
            video_t_mod: Video timestep modulation.
            action_t_mod: Action timestep modulation.
            video_context: Video cross-attention context, ``[B, Lv, D]``.
            video_context_mask: Video context mask.
            action_context: Action cross-attention context, ``[B, La, D]``.
            action_context_mask: Action context mask.
            attention_mask: Joint token mask, ``[Sv + Sa, Sv + Sa]``.

        Construction logic:
            Build Q/K/V independently for both blocks, concatenate them into one
            sequence, run mixed attention, split the output at ``Sv``, and apply
            fixed-context post-processing to the two expert slices.

        Returns:
            ``(video_tokens_next, action_tokens_next)`` with shapes
            ``[B, Sv, D]`` and ``[B, Sa, D]``.
        """
        ...

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all joint video/action layers in sequence.

        Args:
            video_tokens: Initial video tokens, ``[B, Sv, D]``.
            action_tokens: Initial action tokens, ``[B, Sa, D]``.
            video_freqs: Video RoPE frequencies, ``[Sv, 1, rope_dim]``.
            action_freqs: Action RoPE frequencies, ``[Sa, 1, rope_dim]``.
            video_t_mod: Video timestep modulation.
            action_t_mod: Action timestep modulation.
            video_context: Video context, ``[B, Lv, D]``.
            video_context_mask: Video context mask.
            action_context: Action context, ``[B, La, D]``.
            action_context_mask: Action context mask.
            attention_mask: Joint self-attention mask, ``[Sv + Sa, Sv + Sa]``.

        Construction logic:
            Iterate over layer indices, retrieve the corresponding video and action
            blocks, and update both token streams through ``_forward_joint_layer``.
            During training, an enabled compilation switch may cache a full-graph
            ``torch.compile`` version of the single-layer function.

        Returns:
            Final ``(video_tokens, action_tokens)`` with shapes ``[B, Sv, D]`` and
            ``[B, Sa, D]``.
        """
        ...

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run generic layer-wise mixed attention for all configured experts.

        Args:
            embeds_all: Mapping from expert names to initial tokens, each
                ``[B, S_expert, D]``; it must cover every configured expert.
            attention_mask: Joint mask for experts concatenated in
                ``expert_order``, shaped ``[S_total, S_total]`` where
                ``S_total`` is the sum of expert sequence lengths.
            freqs_all: Mapping from expert names to token-aligned RoPE frequencies,
                usually ``[S_expert, 1, rope_dim]`` per expert.
            context_all: Mapping from expert names to optional cross-attention
                payloads. A payload may contain ``context [B, L, D]`` and
                ``mask [B, S_expert, L]`` or ``[B, 1, S_expert, L]``. Missing or
                ``None`` payloads skip cross-attention.
            t_mod_all: Mapping from expert names to timestep modulation tensors.

        Construction logic:
            At each layer, independently build q/k/v and post-processing state for
            every expert, concatenate all q/k/v in the fixed expert order, and run
            one joint attention call. Split the result by each expert's sequence
            length, apply its post-processing, and use the updated tokens as the
            next-layer inputs. A concrete implementation should reject missing
            dictionary keys, non-square masks, and total-length mismatches.

        Returns:
            A dictionary with the same expert keys as ``embeds_all``. Each value is
            the fully updated, un-concatenated token tensor ``[B, S_expert, D]``.
        """
        ...
