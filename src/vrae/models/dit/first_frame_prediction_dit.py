from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vrae.models.dit.blocks import video_tokens_to_grid
from vrae.models.dit.conditioning import FirstFrameConditionAdapter
from vrae.models.dit.video_dit import VRAEVideoDiT, _factory_parameters
from vrae.registry import DIT_MODELS


def first_frame_adaln_condition(
    time_condition: torch.Tensor,
    frame_condition: torch.Tensor,
    *,
    chunks: int,
) -> torch.Tensor:
    """``[B,1,D] + [B,P,D]`` -> chunk-major ``[B,chunks*P,D]``.

    Every future chunk receives the same per-patch condition, so DiT patch ``p``
    of any chunk is modulated by the first-frame patch at the same position.
    """

    if time_condition.ndim != 3 or time_condition.shape[1] != 1:
        raise ValueError(f"time condition must be [B,1,D], got {tuple(time_condition.shape)}")
    if frame_condition.ndim != 3 or frame_condition.shape[0] != time_condition.shape[0]:
        raise ValueError(f"frame condition must be [B,P,D], got {tuple(frame_condition.shape)}")
    batch, patches, width = frame_condition.shape
    grid = (time_condition + frame_condition.to(time_condition.dtype))[:, None]
    return grid.expand(-1, int(chunks), -1, -1).reshape(batch, int(chunks) * patches, width)


class VRAEFirstFramePredictionDiT(VRAEVideoDiT):
    """Future-only DDT conditioned on one observed frame through spatial AdaLN.

    Only the noisy future chunks ``[B,F,h*w,C]`` enter self-attention, with RoPE
    time coordinates ``0..F-1``. The first frame arrives as frozen pre-pool
    encoder features ``[B,h*w,C_f]`` and is projected inside the model to one
    condition per DiT patch, added to the time embedding, and repeated over the
    future chunks as the per-token AdaLN condition of the encoder blocks and of
    the DDT decoder/base-head fusion. Outputs keep the transport contract:
    ``full`` (and ``base`` with ``return_base=True``) of the noisy input's shape.
    """

    def __init__(
        self,
        *,
        future_chunks: int = 4,
        grid_size: int | Sequence[int] = (16, 16),
        in_channels: int = 1024,
        frame_channels: int | None = None,
        frame_dropout: float = 0.1,
        patch_size: int | Sequence[int] = 1,
        hidden_size: int | Sequence[int] = (1536, 2048),
        depth: int | Sequence[int] = (28, 2),
        num_heads: int | Sequence[int] = (24, 16),
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        rope_theta: float = 10_000.0,
        base_model_depth: int = 8,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
        gradient_checkpointing: bool = False,
        input_dim: int | None = None,
        input_size: int | Sequence[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        encoder_depth: int | None = None,
        num_future_chunks: int | None = None,
        condition_dropout: float | None = None,
        use_gradient_checkpointing: bool | None = None,
    ) -> None:
        if num_future_chunks is not None:
            if future_chunks != 4 and int(num_future_chunks) != int(future_chunks):
                raise ValueError("num_future_chunks and future_chunks disagree")
            future_chunks = int(num_future_chunks)
        if condition_dropout is not None:
            if frame_dropout != 0.1 and float(condition_dropout) != float(frame_dropout):
                raise ValueError("condition_dropout and frame_dropout disagree")
            frame_dropout = float(condition_dropout)
        super().__init__(
            num_chunks=int(future_chunks),
            grid_size=grid_size,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            time_embed_dim=time_embed_dim,
            num_classes=1,
            class_dropout=0.0,
            rope_theta=rope_theta,
            base_model_depth=base_model_depth,
            attention_dropout=attention_dropout,
            attention_backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
            input_dim=input_dim,
            input_size=input_size,
            height=height,
            width=width,
            encoder_depth=encoder_depth,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        # Replace the class-label adapter of the shared DDT with the first-frame one.
        del self.num_classes
        self.frame_channels = int(self.in_channels if frame_channels is None else frame_channels)
        self.condition_adapter = FirstFrameConditionAdapter(
            self.frame_channels,
            self.enc_hidden_size,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            dropout_prob=frame_dropout,
        )

    @property
    def future_chunks(self) -> int:
        return self.num_chunks

    @property
    def frame_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def future_latent_shape(self) -> tuple[int, int, int]:
        """Per-sample generation shape ``[chunks, h*w, C]``."""

        return (self.num_chunks, self.tokens_per_chunk, self.in_channels)

    @property
    def frame_projection(self) -> FirstFrameConditionAdapter:
        return self.condition_adapter

    def adaln_condition(
        self,
        time: torch.Tensor,
        frame_features: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        time_condition = self.time_embedder(time)
        frame_condition, _ = self.condition_adapter.prepare(
            frame_features,
            drop_mask=drop_mask,
            generator=generator,
        )
        return first_frame_adaln_condition(
            time_condition,
            frame_condition,
            chunks=self.num_chunks,
        )

    def forward(
        self,
        noisy_future: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        frame_features: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        context_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        if frame_features is None:
            frame_features = context
        elif context is not None:
            raise ValueError("pass first-frame features as either context or frame_features")
        if frame_features is None:
            raise ValueError("first-frame features are required")
        if condition_drop_mask is not None and context_drop_mask is not None:
            raise ValueError("pass only one of condition_drop_mask or context_drop_mask")
        drop_mask = condition_drop_mask if condition_drop_mask is not None else context_drop_mask

        grid, batch = video_tokens_to_grid(
            noisy_future,
            chunks=self.num_chunks,
            grid_size=self.grid_size,
            channels=self.in_channels,
            name="noisy_future",
        )
        if frame_features.shape[0] != batch:
            raise ValueError("frame_features and noisy_future batch sizes must match")
        if frame_features.device != noisy_future.device:
            raise ValueError("frame_features and noisy_future must be on the same device")
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must have shape [{batch}], got {tuple(time.shape)}")
        condition = self.adaln_condition(
            time.to(noisy_future.device),
            frame_features,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        return self._denoise(
            grid,
            batch=batch,
            encoder_condition=condition,
            fusion_condition=condition,
            return_base=return_base,
            return_intermediate=return_intermediate,
        )


@DIT_MODELS.decorator("vrae_first_frame_prediction_dit")
def build_vrae_first_frame_prediction_dit(
    *,
    config: Mapping[str, Any],
    **overrides: Any,
) -> VRAEFirstFramePredictionDiT:
    return VRAEFirstFramePredictionDiT(**_factory_parameters(config, overrides))


__all__ = [
    "VRAEFirstFramePredictionDiT",
    "build_vrae_first_frame_prediction_dit",
    "first_frame_adaln_condition",
]
