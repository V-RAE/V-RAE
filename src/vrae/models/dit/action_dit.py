"""Template for the action DiT expert.

This file records the interfaces and data contracts from ``fastwan_actiondit.py``
without executable implementation. It covers sinusoidal timestep features,
RoPE preparation, the action output head, checkpoint loading, and the regular
and compile-friendly ActionDiT paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Final projection head that maps hidden action tokens to action features."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        eps: float,
    ):
        """Declare normalization, projection, and two-way output modulation.

        Args:
            hidden_dim: Hidden token width ``D``.
            out_dim: Output action feature width.
            eps: Epsilon used by the affine-free LayerNorm.

        Construction logic:
            Create an affine-free LayerNorm over ``hidden_dim``, a linear
            ``hidden_dim -> out_dim`` projection, and a learnable modulation tensor
            shaped ``[1, 2, hidden_dim]`` for shift and scale.

        Output structure:
            An ActionHead module. The constructor returns nothing.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize, timestep-modulate, and project action tokens.

        Args:
            x: Hidden action tokens, shape ``[B, S, D]``.
            t: Per-batch timestep embedding, shape ``[B, D]``.

        Construction logic:
            Add the learned ``[shift, scale]`` modulation to ``t.unsqueeze(1)``;
            apply ``norm(x) * (1 + scale) + shift`` with broadcasting over tokens;
            project the result to the requested action width.

        Returns:
            Action predictions of shape ``[B, S, out_dim]``.
        """
        ...


class ActionDiT(nn.Module):
    """Diffusion Transformer expert for action-token sequences.

    The model embeds action inputs and text context, creates timestep modulation,
    applies a stack of DiT blocks with self/cross-attention and RoPE, then projects
    hidden tokens back to the action dimension. The same expert exposes a regular
    path and a tensor-core-friendly ``prepare``/core/``post`` decomposition.
    """

    ACTION_BACKBONE_SKIP_PREFIXES: tuple[str, ...]
    ACTION_BACKBONE_META_KEYS: tuple[str, ...]

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
    ):
        """Declare the action DiT architecture and its learned components.

        Args:
            hidden_dim: Internal token width ``D``.
            action_dim: Input and output action feature width.
            ffn_dim: Feed-forward hidden width used by each DiT block.
            text_dim: Width of incoming text/context features.
            freq_dim: Width of the sinusoidal timestep embedding input.
            eps: Normalization epsilon passed to DiT blocks.
            num_heads: Number of attention heads ``H``; must be positive.
            attn_head_dim: Per-head attention width ``Dh``; must be positive and
                even for RoPE.
            num_layers: Number of stacked DiT blocks.
            use_gradient_checkpointing: Whether block execution may use gradient
                checkpointing during training.

        Construction logic:
            1. Store architecture metadata and validate attention dimensions.
            2. Create the action input projection ``action_dim -> hidden_dim``.
            3. Create the text MLP ``text_dim -> hidden_dim`` and timestep MLP
               ``freq_dim -> hidden_dim``.
            4. Create a timestep projection to ``6 * hidden_dim`` for per-block
               modulation, ``num_layers`` DiT blocks, and a hidden-to-action head.
            5. Precompute a RoPE table for ``attn_head_dim`` and the supported
               sequence limit.

        Output structure:
            An initialized ActionDiT module. The constructor returns nothing.
        """
        ...

    def _apply(self, fn: Any) -> "ActionDiT":
        """Apply a module-wide conversion and move the RoPE table with the module.

        Args:
            fn: Transformation supplied by ``nn.Module._apply`` for device or
                dtype conversion.

        Construction logic:
            Delegate parameter/buffer conversion to the parent implementation,
            then move the precomputed ``freqs`` tensor to the resulting parameter
            device because it is stored as a plain tensor rather than a buffer.

        Returns:
            This ActionDiT module after conversion.
        """
        ...

    def get_freqs(self, seq_len: int) -> torch.Tensor:
        """Return the RoPE slice aligned with an action sequence.

        Args:
            seq_len: Number of action tokens ``S``; must not exceed the precomputed
                RoPE table length.

        Construction logic:
            Take the first ``seq_len`` rows of ``self.freqs`` and reshape them to
            insert a singleton broadcast dimension.

        Returns:
            Complex RoPE frequencies shaped ``[S, 1, attn_head_dim // 2]``.
        """
        ...

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        """Filter state-dict keys to the reusable action backbone subset.

        Args:
            keys: Iterable of state-dict key strings.

        Construction logic:
            Remove keys beginning with any prefix in
            ``ACTION_BACKBONE_SKIP_PREFIXES`` (the action input encoder and output
            head), retaining the shared Transformer backbone keys.

        Returns:
            A set containing the retained state-dict keys.
        """
        ...

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        """Construct ActionDiT and optionally load a preprocessed backbone payload.

        Args:
            action_dit_config: Keyword arguments for the ActionDiT constructor.
                Required even when loading is skipped.
            action_dit_pretrained_path: Optional checkpoint path. Relative paths
                are resolved using the source project's expected repository base.
            skip_dit_load_from_pretrain: If true, initialize random weights and
                leave checkpoint override to the caller.
            device: Target device for the created expert.
            torch_dtype: Target parameter dtype, defaulting to bfloat16.

        Construction logic:
            For the skip/no-path cases, create a random expert and move it to the
            requested device and dtype. Otherwise, resolve and load a dictionary
            payload containing optional ``policy``, required ``meta``, and required
            ``backbone_state_dict`` fields. Validate architecture metadata against
            the supplied config, compare filtered backbone key sets and tensor
            shapes, cast loaded tensors to target dtype/device, merge them with
            randomly initialized skipped prefixes, and load the merged state
            strictly before returning the expert.

        Returns:
            An ActionDiT instance on ``device`` with ``torch_dtype`` parameters.
        """
        ...

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare validated inputs and all conditioning for the regular path.

        Args:
            action_tokens: Raw action sequence, shape ``[B, T, action_dim]``.
            timestep: Timestep values, shape ``[B]`` or ``[1]``. A singleton is
                expanded for inference but must match batch size during training.
            context: Text/context features, shape ``[B, L, text_dim]``.
            context_mask: Optional validity mask, shape ``[B, L]``; an all-true
                mask is created when omitted.

        Construction logic:
            Validate ranks and batch/sequence compatibility; create sinusoidal
            timestep embeddings and ``[B, 6, hidden_dim]`` modulation; project
            action tokens and context; expand the context mask to one row per
            action token; and retrieve token-aligned RoPE frequencies.

        Returns:
            A dictionary with:
            - ``tokens``: embedded action tokens, ``[B, T, hidden_dim]``;
            - ``freqs``: RoPE frequencies, ``[T, 1, attn_head_dim // 2]``;
            - ``t``: timestep embedding, ``[B, hidden_dim]``;
            - ``t_mod``: block modulation, ``[B, 6, hidden_dim]``;
            - ``context``: embedded context, ``[B, L, hidden_dim]``;
            - ``context_mask``: expanded mask, ``[B, T, L]``;
            - ``meta``: ``{"batch_size": B, "seq_len": T}``.
        """
        ...

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Dict[str, Any],
    ) -> torch.Tensor:
        """Project regular-path hidden tokens back to action features.

        Args:
            tokens: Hidden action tokens produced by the DiT blocks,
                shape ``[B, T, hidden_dim]``.
            pre_state: Preparation dictionary retained for caller bookkeeping;
                the source implementation does not use it in the projection.

        Returns:
            Action output tensor of shape ``[B, T, action_dim]``.
        """
        ...

    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare compile-friendly tensor inputs for an external DiT/MoT core.

        Args:
            action_tokens: Raw action sequence, shape ``[B, T, action_dim]``.
            timestep: Timestep values, normally shape ``[B]``.
            context: Text/context features, shape ``[B, L, text_dim]``.
            context_mask: Required context mask, shape ``[B, L]``.

        Construction logic:
            Compute timestep and six-way modulation, project action and context
            features, expand the context mask from ``[B, L]`` to ``[B, T, L]``,
            and obtain the first ``T`` RoPE frequencies.

        Returns:
            ``(tokens, t, t_mod, context_emb, context_attn_mask, freqs)`` where
            shapes are ``[B, T, hidden_dim]``, ``[B, hidden_dim]``,
            ``[B, 6, hidden_dim]``, ``[B, L, hidden_dim]``, ``[B, T, L]``, and
            ``[T, 1, attn_head_dim // 2]`` respectively.
        """
        ...

    def post(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project tensor-core hidden tokens to the action output dimension.

        Args:
            tokens: Hidden action tokens, shape ``[B, T, hidden_dim]``.

        Returns:
            Action predictions shaped ``[B, T, action_dim]``.
        """
        ...

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the complete regular ActionDiT path.

        Args:
            action_tokens: Raw action sequence, ``[B, T, action_dim]``.
            timestep: Timestep values, ``[B]`` or ``[1]`` according to the regular
                preparation contract.
            context: Text/context features, ``[B, L, text_dim]``.
            context_mask: Optional mask ``[B, L]``; defaults to all valid entries.

        Construction logic:
            Prepare embeddings, timestep modulation, context, mask, and RoPE;
            pass the embedded action tokens through every DiT block with the same
            conditioning; then apply the output projection.

        Returns:
            Final action predictions of shape ``[B, T, action_dim]``.
        """
        ...
