"""Frozen V-RAE encodings for first-frame conditioned future prediction.

The observed frame and the sixteen future frames are encoded by separate calls:

* condition: ``encode_image_grid`` -> pre-pool fused patch features of frame 0,
  ``[B,h*w,C]`` (one image per sample, no temporal pooling);
* target: ``encode_grid`` on the future frames only -> pooled latents,
  ``[B,4,h*w,C]`` (four 4-frame chunks).

Neither path sees the other's pixels, so replacing the future cannot change the
condition and replacing the first frame cannot change the clean target.
"""

from __future__ import annotations

from typing import Any

import torch

from vrae.training.nanovvm_video_pred.data import FUTURE_FRAMES


def _check_rgb(value: torch.Tensor, *, name: str, ndim: int) -> None:
    if value.ndim != ndim or value.shape[-3] != 3:
        layout = "[B,3,H,W]" if ndim == 4 else f"[B,{FUTURE_FRAMES},3,H,W]"
        raise ValueError(f"{name} must be RGB {layout}, got {tuple(value.shape)}")


@torch.no_grad()
def encode_first_frame(stage1: Any, condition_rgb: torch.Tensor) -> torch.Tensor:
    """``[B,3,H,W]`` -> pre-pool patch features ``[B,h*w,C]`` (row-major grid order)."""

    _check_rgb(condition_rgb, name="condition_rgb", ndim=4)
    grid = stage1.encode_image_grid(condition_rgb)
    return stage1.image_grid_to_tokens(grid)


@torch.no_grad()
def encode_future(stage1: Any, future_rgb: torch.Tensor) -> torch.Tensor:
    """``[B,16,3,H,W]`` -> pooled future latents ``[B,4,h*w,C]``."""

    _check_rgb(future_rgb, name="future_rgb", ndim=5)
    if future_rgb.shape[1] != FUTURE_FRAMES:
        raise ValueError(
            f"future_rgb must contain {FUTURE_FRAMES} frames, got {future_rgb.shape[1]}"
        )
    return stage1.grid_to_tokens(stage1.encode_grid(future_rgb))


@torch.no_grad()
def encode_condition_and_future(
    stage1: Any,
    condition_rgb: torch.Tensor,
    future_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition_rgb.shape[0] != future_rgb.shape[0]:
        raise ValueError("condition and future batch sizes must match")
    frame_features = encode_first_frame(stage1, condition_rgb)
    future_tokens = encode_future(stage1, future_rgb)
    if future_tokens.shape[2:] != frame_features.shape[1:]:
        raise ValueError(
            "first-frame features and future latents must share the patch grid and width: "
            f"{tuple(frame_features.shape)} vs {tuple(future_tokens.shape)}"
        )
    return frame_features, future_tokens


@torch.no_grad()
def decode_future_tokens(
    stage1: Any,
    future_tokens: torch.Tensor,
    *,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """``[B,4,h*w,C]`` -> RGB ``[B,16,3,H,W]`` through the frozen V-RAE decoder."""

    grid = stage1.tokens_to_grid(future_tokens, height=int(grid_size[0]), width=int(grid_size[1]))
    return stage1.decode_grid(grid)


__all__ = [
    "decode_future_tokens",
    "encode_condition_and_future",
    "encode_first_frame",
    "encode_future",
]
