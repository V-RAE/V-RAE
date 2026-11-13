from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.blocks import (
    AdaLNZeroDDTBlock,
    DDTDecoderBlock,
    DDTFinalLayer,
    FramePatchEmbed,
    GaussianFourierTimeEmbedding,
    VideoDiTRoPECache,
    as_pair,
    embed_video_frames,
    unpatchify_video_tokens,
    video_tokens_to_grid,
)
from vrae.models.dit.conditioning import LabelConditionAdapter
from vrae.models.rope3d import (
    build_3d_positions,
    build_video_dit_3d_rope_cache_from_positions,
)
from vrae.registry import DIT_MODELS


def _architecture_pair(
    value: int | Sequence[int],
    *,
    name: str,
    scalar_second: int | None = None,
) -> tuple[int, int]:
    if isinstance(value, int):
        result = (int(value), int(value) if scalar_second is None else int(scalar_second))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a length-2 sequence")
        result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _resolve_grid_size(
    grid_size: int | Sequence[int],
    *,
    input_size: int | Sequence[int] | None,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    result = as_pair(grid_size, "grid_size")
    if input_size is not None:
        alias = as_pair(input_size, "input_size")
        if result != (16, 16) and result != alias:
            raise ValueError("grid_size and input_size disagree")
        result = alias
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be specified together")
        explicit = (int(height), int(width))
        if result != (16, 16) and result != explicit:
            raise ValueError("grid_size and height/width disagree")
        result = as_pair(explicit, "height/width")
    return result


class VRAEVideoDiT(nn.Module):
    """Shared class-conditional video DDT for UCF101 and Kinetics-600.

    Both input and output use the generation contract ``[B,T,H*W,C]``. Dataset
    identity is intentionally absent: UCF101 and K600 differ only in
    ``num_classes`` and configuration values.
    """

    def __init__(
        self,
        *,
        num_chunks: int = 4,
        grid_size: int | Sequence[int] = (16, 16),
        in_channels: int = 1024,
        patch_size: int | Sequence[int] = 1,
        hidden_size: int | Sequence[int] = (1536, 2048),
        depth: int | Sequence[int] = (28, 2),
        num_heads: int | Sequence[int] = (24, 16),
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        num_classes: int = 101,
        class_dropout: float = 0.1,
        rope_theta: float = 10_000.0,
        base_model_depth: int = 8,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
        gradient_checkpointing: bool = False,
        input_dim: int | None = None,
        num_frames: int | None = None,
        input_size: int | Sequence[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        encoder_depth: int | None = None,
        condition_dropout: float | None = None,
        use_gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            if in_channels != 1024 and int(input_dim) != int(in_channels):
                raise ValueError("input_dim and in_channels disagree")
            in_channels = int(input_dim)
        if num_frames is not None:
            if num_chunks != 4 and int(num_frames) != int(num_chunks):
                raise ValueError("num_frames and num_chunks disagree")
            num_chunks = int(num_frames)
        if encoder_depth is not None:
            if base_model_depth != 8 and int(encoder_depth) != int(base_model_depth):
                raise ValueError("encoder_depth and base_model_depth disagree")
            base_model_depth = int(encoder_depth)
        if condition_dropout is not None:
            if class_dropout != 0.1 and float(condition_dropout) != float(class_dropout):
                raise ValueError("condition_dropout and class_dropout disagree")
            class_dropout = float(condition_dropout)
        if use_gradient_checkpointing is not None:
            gradient_checkpointing = bool(use_gradient_checkpointing)

        self.num_chunks = int(num_chunks)
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        self.grid_size = _resolve_grid_size(
            grid_size,
            input_size=input_size,
            height=height,
            width=width,
        )
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.patch_size = as_pair(patch_size, "patch_size")
        if any(size % patch for size, patch in zip(self.grid_size, self.patch_size, strict=True)):
            raise ValueError("grid_size must be divisible by patch_size")
        self.patch_grid = tuple(
            size // patch for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        )
        self.tokens_per_chunk = self.grid_size[0] * self.grid_size[1]
        self.patches_per_chunk = self.patch_grid[0] * self.patch_grid[1]

        enc_dim, dec_dim = _architecture_pair(hidden_size, name="hidden_size")
        enc_depth, dec_depth = _architecture_pair(depth, name="depth", scalar_second=2)
        enc_heads, dec_heads = _architecture_pair(num_heads, name="num_heads")
        if enc_dim % enc_heads or dec_dim % dec_heads:
            raise ValueError("each hidden size must be divisible by its number of heads")
        self.enc_hidden_size = enc_dim
        self.dec_hidden_size = dec_dim
        self.num_enc_blocks = enc_depth
        self.num_dec_blocks = dec_depth
        self.base_model_depth = int(base_model_depth)
        if not 1 <= self.base_model_depth <= self.num_enc_blocks:
            raise ValueError(
                f"base_model_depth must be in [1,{self.num_enc_blocks}], "
                f"got {self.base_model_depth}"
            )
        self.num_classes = int(num_classes)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.encoder_embed = FramePatchEmbed(self.in_channels, enc_dim, self.patch_size)
        self.decoder_embed = FramePatchEmbed(self.in_channels, dec_dim, self.patch_size)
        self.encoder_blocks = nn.ModuleList(
            [
                AdaLNZeroDDTBlock(
                    enc_dim,
                    enc_heads,
                    mlp_ratio,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(enc_depth)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    dec_dim,
                    dec_heads,
                    mlp_ratio,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(dec_depth)
            ]
        )
        self.time_embedder = GaussianFourierTimeEmbedding(
            enc_dim,
            embedding_size=int(time_embed_dim),
        )
        self.condition_adapter = LabelConditionAdapter(
            enc_dim,
            self.num_classes,
            dropout_prob=class_dropout,
        )
        self.encoder_to_decoder: nn.Module = (
            nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        )
        self.final_layer = DDTFinalLayer(dec_dim, self.patch_size, self.in_channels)
        self.base_final_layer = DDTFinalLayer(enc_dim, self.patch_size, self.in_channels)

        positions = build_3d_positions(
            self.num_chunks,
            self.patch_grid[0],
            self.patch_grid[1],
        )
        self.register_buffer("_encoder_positions", positions, persistent=False)
        self.register_buffer("_decoder_positions", positions.clone(), persistent=False)
        self._encoder_rope_cache: tuple[object, ...] | None = None
        self._decoder_rope_cache: tuple[object, ...] | None = None
        self._initialize_weights()

    @property
    def num_visual_tokens(self) -> int:
        return self.num_chunks * self.patches_per_chunk

    @property
    def s_embedder(self) -> FramePatchEmbed:
        return self.encoder_embed

    @property
    def x_embedder(self) -> FramePatchEmbed:
        return self.decoder_embed

    @property
    def t_embedder(self) -> GaussianFourierTimeEmbedding:
        return self.time_embedder

    @property
    def ctx_embedder(self) -> LabelConditionAdapter:
        return self.condition_adapter

    @property
    def s_projector(self) -> nn.Module:
        return self.encoder_to_decoder

    @property
    def blocks(self) -> tuple[AdaLNZeroDDTBlock, ...]:
        return tuple(self.encoder_blocks) + tuple(self.decoder_blocks)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _initialize_weights(self) -> None:
        for embedder in (self.encoder_embed, self.decoder_embed):
            weight = embedder.proj.weight.data
            nn.init.xavier_uniform_(weight.reshape(weight.shape[0], -1))
            if embedder.proj.bias is not None:
                nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.condition_adapter.embedding.weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

    def _run_conditioned_block(
        self,
        block: AdaLNZeroDDTBlock,
        value: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        rope_cache: VideoDiTRoPECache,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda current, cond: block(
                    current,
                    cond,
                    positions,
                    rope_cache=rope_cache,
                ),
                value,
                condition,
                use_reentrant=False,
            )
        return block(value, condition, positions, rope_cache=rope_cache)

    def _cached_rope(
        self,
        *,
        scope: str,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        block: AdaLNZeroDDTBlock,
    ) -> VideoDiTRoPECache:
        if positions.device != hidden.device:
            raise ValueError("VideoDiT positions and hidden states must be on the same device")
        attribute = f"_{scope}_rope_cache"
        cached = getattr(self, attribute)
        head_dim = block.attn.head_dim
        theta = block.attn.rope_theta
        if (
            cached is None
            or cached[0] is not positions
            or cached[1] != hidden.dtype
            or cached[2] != head_dim
            or cached[3] != theta
        ):
            cosine, sine = build_video_dit_3d_rope_cache_from_positions(
                positions,
                head_dim,
                theta=theta,
                dtype=hidden.dtype,
            )
            cached = (positions, hidden.dtype, head_dim, theta, cosine, sine)
            setattr(self, attribute, cached)
        return cached[4], cached[5]

    def _unpatchify(self, value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        return unpatchify_video_tokens(
            value,
            batch_size=batch_size,
            chunks=self.num_chunks,
            patch_grid=self.patch_grid,
            patch_size=self.patch_size,
            channels=self.in_channels,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        class_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        if labels is None:
            labels = context
        elif context is not None:
            raise ValueError("pass class labels as either labels or context, not both")
        if labels is None:
            raise ValueError("class labels are required")
        if condition_drop_mask is not None and class_drop_mask is not None:
            raise ValueError("pass only one of condition_drop_mask or class_drop_mask")
        drop_mask = class_drop_mask if class_drop_mask is not None else condition_drop_mask

        grid, batch = video_tokens_to_grid(
            noisy,
            chunks=self.num_chunks,
            grid_size=self.grid_size,
            channels=self.in_channels,
            name="noisy",
        )
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must have shape [{batch}], got {tuple(time.shape)}")
        if labels.shape != (batch,):
            raise ValueError(f"labels must have shape [{batch}], got {tuple(labels.shape)}")
        time_embedding = self.time_embedder(time.to(noisy.device))
        label_embedding, _, _ = self.condition_adapter.prepare(
            labels,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        encoder_condition = time_embedding + label_embedding.to(time_embedding.dtype)
        return self._denoise(
            grid,
            batch=batch,
            encoder_condition=encoder_condition,
            fusion_condition=time_embedding,
            return_base=return_base,
            return_intermediate=return_intermediate,
        )

    def _denoise(
        self,
        grid: torch.Tensor,
        *,
        batch: int,
        encoder_condition: torch.Tensor,
        fusion_condition: torch.Tensor,
        return_base: bool,
        return_intermediate: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        """Shared DDT body: AdaLN encoder, fused decoder condition, full and base heads.

        ``encoder_condition`` modulates every encoder block. ``fusion_condition``
        is added to the final and base encoder states before the decoder/full head
        and base head conditions. Both are ``[B,1,D]`` (broadcast) or per-token
        ``[B,num_chunks*P,D]`` in chunk-major order.
        """

        sequence = embed_video_frames(
            self.encoder_embed,
            grid,
            batch_size=batch,
            chunks=self.num_chunks,
        )
        encoder_rope_cache = self._cached_rope(
            scope="encoder",
            positions=self._encoder_positions,
            hidden=sequence,
            block=self.encoder_blocks[0],
        )
        base_sequence: torch.Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            sequence = self._run_conditioned_block(
                block,
                sequence,
                encoder_condition,
                self._encoder_positions,
                encoder_rope_cache,
            )
            if index == self.base_model_depth:
                base_sequence = sequence
        if base_sequence is None:
            raise RuntimeError("base encoder activation was not captured")

        decoder_condition = self.encoder_to_decoder(F.silu(fusion_condition + sequence))
        decoder_sequence = embed_video_frames(
            self.decoder_embed,
            grid,
            batch_size=batch,
            chunks=self.num_chunks,
        )
        decoder_rope_cache = self._cached_rope(
            scope="decoder",
            positions=self._decoder_positions,
            hidden=decoder_sequence,
            block=self.decoder_blocks[0],
        )
        for block in self.decoder_blocks:
            decoder_sequence = self._run_conditioned_block(
                block,
                decoder_sequence,
                decoder_condition,
                self._decoder_positions,
                decoder_rope_cache,
            )
        full = self._unpatchify(
            self.final_layer(decoder_sequence, decoder_condition),
            batch_size=batch,
        )
        base_condition = F.silu(fusion_condition + base_sequence)
        base = self._unpatchify(
            self.base_final_layer(base_condition, base_condition),
            batch_size=batch,
        )
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        result = (full, base) if return_base else full
        if return_intermediate:
            return result, base_sequence
        return result


def _factory_parameters(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(config)
    values.pop("name", None)
    nested = values.pop("parameters", values.pop("params", {}))
    if nested:
        if not isinstance(nested, Mapping):
            raise TypeError("DiT parameters must be a mapping")
        parameters = dict(nested)
        parameters.update(values)
    else:
        parameters = values
    parameters.update(overrides)
    return parameters


@DIT_MODELS.decorator("vrae_video_dit")
def build_vrae_video_dit(
    *,
    config: Mapping[str, Any],
    **overrides: Any,
) -> VRAEVideoDiT:
    return VRAEVideoDiT(**_factory_parameters(config, overrides))


__all__ = ["VRAEVideoDiT", "build_vrae_video_dit"]
