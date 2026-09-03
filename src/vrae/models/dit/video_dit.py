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
    embed_video_frames,
    unpatchify_video_tokens,
)
from vrae.models.dit.conditioning import LabelConditionAdapter
from vrae.models.dit.rope import RoPE3D
from vrae.registry import DIT_MODELS


class VRAEVideoDiT(nn.Module):
    """Class-conditional VideoDiT over frozen V-RAE latent tokens.

    Single-view inputs and outputs use ``[B,T,H*W,C]``. Multiview inputs and
    outputs use ``[B,T,V,H*W,C]``.
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
        num_classes: int = 40,
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
        multiview_enabled: bool = False,
        num_views: int = 1,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            in_channels = int(input_dim)
        if num_frames is not None:
            num_chunks = int(num_frames)
        if encoder_depth is not None:
            base_model_depth = int(encoder_depth)
        if condition_dropout is not None:
            class_dropout = float(condition_dropout)
        if use_gradient_checkpointing is not None:
            gradient_checkpointing = bool(use_gradient_checkpointing)

        self.num_chunks = int(num_chunks)
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        if input_size is not None:
            grid_size = input_size
        if height is not None:
            if width is None:
                raise ValueError("width is required when height is provided")
            grid_size = (height, width)
        if isinstance(grid_size, int):
            self.grid_size = (int(grid_size), int(grid_size))
        else:
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.in_channels = int(in_channels)
        if isinstance(patch_size, int):
            self.patch_size = (int(patch_size), int(patch_size))
        else:
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        if any(size <= 0 for size in self.grid_size):
            raise ValueError("grid_size values must be positive")
        if any(patch <= 0 for patch in self.patch_size):
            raise ValueError("patch_size values must be positive")
        if any(
            size % patch != 0
            for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        ):
            raise ValueError("grid_size values must be divisible by patch_size")
        self.patch_grid = tuple(
            size // patch
            for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        )
        self.tokens_per_chunk = self.grid_size[0] * self.grid_size[1]
        self.patches_per_chunk = self.patch_grid[0] * self.patch_grid[1]

        if isinstance(hidden_size, int):
            enc_dim, dec_dim = int(hidden_size), int(hidden_size)
        else:
            enc_dim, dec_dim = int(hidden_size[0]), int(hidden_size[1])
        if isinstance(depth, int):
            enc_depth, dec_depth = int(depth), 2
        else:
            enc_depth, dec_depth = int(depth[0]), int(depth[1])
        if isinstance(num_heads, int):
            enc_heads, dec_heads = int(num_heads), int(num_heads)
        else:
            enc_heads, dec_heads = int(num_heads[0]), int(num_heads[1])
        self.enc_hidden_size = enc_dim
        self.dec_hidden_size = dec_dim
        self.num_enc_blocks = enc_depth
        self.num_dec_blocks = dec_depth
        self.base_model_depth = int(base_model_depth)
        self.num_classes = int(num_classes)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.multiview_enabled = bool(multiview_enabled)
        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")
        self.num_streams = self.num_views
        if not 1 <= self.base_model_depth <= enc_depth:
            raise ValueError("base_model_depth must be within encoder depth")

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
        if enc_dim != dec_dim:
            self.encoder_to_decoder: nn.Module = nn.Linear(enc_dim, dec_dim)
        else:
            self.encoder_to_decoder = nn.Identity()
        self.final_layer = DDTFinalLayer(
            dec_dim,
            self.patch_size,
            self.in_channels,
        )
        self.base_final_layer = DDTFinalLayer(
            enc_dim,
            self.patch_size,
            self.in_channels,
        )
        self.encoder_view_embedding = nn.Embedding(self.num_views, enc_dim)
        self.decoder_view_embedding = nn.Embedding(self.num_views, dec_dim)
        nn.init.zeros_(self.encoder_view_embedding.weight)
        nn.init.zeros_(self.decoder_view_embedding.weight)

        self.encoder_rope = RoPE3D(
            enc_dim // enc_heads,
            end=max(self.num_chunks, *self.patch_grid),
            theta=rope_theta,
        )
        self.decoder_rope = RoPE3D(
            dec_dim // dec_heads,
            end=max(self.num_chunks, *self.patch_grid),
            theta=rope_theta,
        )
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
        return (
            tuple(self.encoder_blocks)
            + tuple(self.decoder_blocks)
        )

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
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                lambda current, cond: block(
                    current,
                    cond,
                    freqs,
                ),
                value,
                condition,
                use_reentrant=False,
            )
        return block(
            value,
            condition,
            freqs,
        )

    def _unpatchify(self, value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        return unpatchify_video_tokens(
            value,
            batch_size=batch_size,
            chunks=self.num_chunks,
            patch_grid=self.patch_grid,
            patch_size=self.patch_size,
            channels=self.in_channels,
        )

    def _prepare_input_grid(
        self,
        noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Convert token latents to the shared ``[B*V,T,C,H,W]`` path.

        Returns:
            grid: [B*V*T,C,H,W] tensor
            batch_size: batch size B
            views: number of views V
        """
        views = self.num_views if self.multiview_enabled else 1
        if not self.multiview_enabled:
            noisy = noisy.unsqueeze(2)
        batch = noisy.shape[0]
        tokens = noisy.permute(0, 2, 1, 3, 4).contiguous()
        tokens = tokens.reshape(
            batch * views,
            self.num_chunks,
            *self.grid_size,
            self.in_channels,
        )
        grid = tokens.permute(0, 1, 4, 2, 3).reshape(
            batch * views * self.num_chunks,
            self.in_channels,
            *self.grid_size,
        )
        return grid, batch, views

    def _validate_forward_inputs(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor | None,
        context: torch.Tensor | None,
        condition_drop_mask: torch.Tensor | None,
        class_drop_mask: torch.Tensor | None,
        stream_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Validate and normalize forward inputs without preparing the latent grid."""
        if not self.multiview_enabled:
            if noisy.ndim != 4:
                raise ValueError(
                    f"noisy must have shape [B,{self.num_chunks},N,C]"
                )
        else:
            expected_shape = (self.num_chunks, self.num_views)
            if noisy.ndim != 5 or tuple(noisy.shape[1:3]) != expected_shape:
                raise ValueError(
                    f"noisy must have shape [B,{self.num_chunks},{self.num_views},N,C]"
                )
        if tuple(noisy.shape[-2:]) != (
            self.tokens_per_chunk,
            self.in_channels,
        ):
            raise ValueError("noisy token grid has incorrect patch shape")
        if not noisy.is_floating_point():
            raise TypeError("noisy must be floating point")

        if labels is None:
            labels = context
        elif context is not None:
            raise ValueError("pass class labels as either labels or context, not both")
        if labels is None:
            raise ValueError("class labels are required")
        if condition_drop_mask is not None and class_drop_mask is not None:
            raise ValueError("pass only one of condition_drop_mask or class_drop_mask")
        batch_size = noisy.shape[0]
        if time.ndim == 0:
            time = time.expand(batch_size)
        if time.ndim != 1 or time.shape[0] != batch_size:
            raise ValueError(f"time must have shape [{batch_size}], got {tuple(time.shape)}")
        if labels.shape != (batch_size,):
            raise ValueError(f"labels must have shape [{batch_size}], got {tuple(labels.shape)}")
        if self.multiview_enabled:
            if stream_ids is None:
                stream_ids = torch.arange(self.num_views, device=noisy.device).expand(
                    batch_size, self.num_views
                )
            if stream_ids.shape != (batch_size, self.num_views):
                raise ValueError(f"stream_ids must have shape [{batch_size},{self.num_views}]")
            if stream_ids.min() < 0 or stream_ids.max() >= self.num_streams:
                raise ValueError("stream_ids are outside num_streams")
        drop_mask = (
            class_drop_mask
            if class_drop_mask is not None
            else condition_drop_mask
        )
        return noisy, time, labels, drop_mask, stream_ids

    @staticmethod
    def _expand_view_freqs(freqs: torch.Tensor, *, chunks: int, patches: int, views: int) -> torch.Tensor:
        if views == 1:
            return freqs
        return freqs.reshape(chunks, 1, patches, -1).expand(-1, views, -1, -1).reshape(
            chunks * patches * views, 1, -1
        )

    def _embed_sequence(
        self,
        embedder: FramePatchEmbed,
        view_embedding: nn.Embedding,
        grid: torch.Tensor,
        batch_size: int,
        views: int,
        stream_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed a prepared latent grid and add view-specific embeddings.
        """
        sequence = embed_video_frames(
            embedder,
            grid,
            batch_size=batch_size * views,
            chunks=self.num_chunks,
        )

        patches = sequence.shape[1] // self.num_chunks

        if views > 1:
            sequence = sequence.reshape(
                batch_size,
                views,
                self.num_chunks,
                patches,
                sequence.shape[-1],
            )

            sequence = sequence.permute(0, 2, 1, 3, 4)

            if stream_ids is None:
                raise ValueError(
                    "stream_ids are required for multiview embedding"
                )

            view_bias = view_embedding(stream_ids).to(sequence.dtype)
            sequence = sequence + view_bias[:, None, :, None]

            sequence = sequence.reshape(
                batch_size,
                -1,
                sequence.shape[-1],
            )

        return sequence


    def _restore_output(self, tokens: torch.Tensor, *, batch: int, views: int) -> torch.Tensor:
        if views == 1:
            return self._unpatchify(tokens, batch_size=batch)
        tokens = tokens.reshape(batch, self.num_chunks, views, self.patches_per_chunk, -1)
        tokens = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch * views, self.num_chunks * self.patches_per_chunk, -1
        )
        restored = self._unpatchify(tokens, batch_size=batch * views)
        restored = restored.reshape(batch, views, self.num_chunks, self.patches_per_chunk, -1)
        return restored.permute(0, 2, 1, 3, 4).contiguous()

    def prepare(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        class_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        stream_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare validated inputs for the VideoDiT tensor core."""
        noisy, time, labels, drop_mask, stream_ids = self._validate_forward_inputs(
            noisy,
            time,
            labels,
            context,
            condition_drop_mask,
            class_drop_mask,
            stream_ids,
        )
        grid, batch_size, views = self._prepare_input_grid(noisy)
        time_embedding = self.time_embedder(time.to(noisy.device))
        label_embedding, _, _ = self.condition_adapter.prepare(
            labels,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        encoder_condition = time_embedding + label_embedding.to(
            time_embedding.dtype
        )
        encoder_freqs = self._expand_view_freqs(
            self.encoder_rope(self.num_chunks, self.patch_grid[0], self.patch_grid[1]),
            chunks=self.num_chunks,
            patches=self.patches_per_chunk,
            views=views,
        ).to(device=noisy.device)
        decoder_freqs = self._expand_view_freqs(
            self.decoder_rope(self.num_chunks, self.patch_grid[0], self.patch_grid[1]),
            chunks=self.num_chunks,
            patches=self.patches_per_chunk,
            views=views,
        ).to(device=noisy.device)
        return (
            grid,
            batch_size,
            views,
            time_embedding,
            encoder_condition,
            stream_ids,
            encoder_freqs,
            decoder_freqs,
        )

    def _forward_core(
        self,
        grid: torch.Tensor,
        batch_size: int,
        views: int,
        time_embedding: torch.Tensor,
        encoder_condition: torch.Tensor,
        stream_ids: torch.Tensor | None,
        encoder_freqs: torch.Tensor,
        decoder_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder and decoder blocks on prepared tensor inputs."""
        sequence = self._embed_sequence(
            self.encoder_embed,
            self.encoder_view_embedding,
            grid,
            batch_size,
            views,
            stream_ids,
        )
        base_sequence: torch.Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            sequence = self._run_conditioned_block(
                block,
                sequence,
                encoder_condition,
                encoder_freqs,
            )
            if index == self.base_model_depth:
                base_sequence = sequence
        if base_sequence is None:
            raise RuntimeError("base encoder activation was not captured")

        decoder_condition = self.encoder_to_decoder(
            F.silu(time_embedding + sequence)
        )
        decoder_sequence = self._embed_sequence(
            self.decoder_embed,
            self.decoder_view_embedding,
            grid,
            batch_size,
            views,
            stream_ids,
        )
        for block in self.decoder_blocks:
            decoder_sequence = self._run_conditioned_block(
                block,
                decoder_sequence,
                decoder_condition,
                decoder_freqs,
            )

        full_tokens = self.final_layer(decoder_sequence, decoder_condition)
        base_condition = F.silu(time_embedding + base_sequence)
        base_tokens = self.base_final_layer(base_condition, base_condition)
        return full_tokens, base_tokens, base_sequence

    def post(
        self,
        full_tokens: torch.Tensor,
        base_tokens: torch.Tensor,
        base_sequence: torch.Tensor,
        *,
        batch_size: int,
        views: int,
        return_base: bool,
        return_intermediate: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        """Restore latent grids and assemble the requested outputs."""
        full = self._restore_output(full_tokens, batch=batch_size, views=views)
        base = self._restore_output(base_tokens, batch=batch_size, views=views)
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if return_base:
            result = (full, base)
        else:
            result = full
        if return_intermediate:
            return result, base_sequence
        return result

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
        stream_ids: torch.Tensor | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        prepared = self.prepare(
            noisy,
            time,
            labels,
            context=context,
            condition_drop_mask=condition_drop_mask,
            class_drop_mask=class_drop_mask,
            condition_generator=condition_generator,
            stream_ids=stream_ids,
        )
        full_tokens, base_tokens, base_sequence = self._forward_core(*prepared)
        return self.post(
            full_tokens,
            base_tokens,
            base_sequence,
            batch_size=prepared[1],
            views=prepared[2],
            return_base=return_base,
            return_intermediate=return_intermediate,
        )


def _factory_parameters(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
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
    parameters = _factory_parameters(config, overrides)
    return VRAEVideoDiT(**parameters)


__all__ = ["VRAEVideoDiT", "build_vrae_video_dit"]
