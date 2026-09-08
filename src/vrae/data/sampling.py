"""Deterministic, backend-independent temporal frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

ClipSamplingMode: TypeAlias = Literal["start", "center", "random"]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def clip_span(clip_length: int, frame_interval: int = 1) -> int:
    """Number of source frames covered by a strided clip."""

    length = _positive_int(clip_length, "clip_length")
    interval = _positive_int(frame_interval, "frame_interval")
    return 1 + (length - 1) * interval


def is_clip_long_enough(num_frames: int, clip_length: int, frame_interval: int = 1) -> bool:
    return num_frames >= clip_span(clip_length, frame_interval)


def sample_clip_indices(
    num_frames: int,
    clip_length: int,
    *,
    frame_interval: int = 1,
    mode: ClipSamplingMode = "random",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one strict temporal clip without implicit padding or truncation."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    length = _positive_int(clip_length, "clip_length")
    interval = _positive_int(frame_interval, "frame_interval")
    if mode not in {"start", "center", "random"}:
        raise ValueError("mode must be start, center, or random")

    required = clip_span(length, interval)
    if num_frames < required:
        raise ValueError(
            f"video has {num_frames} frames, but {required} are required for "
            f"clip_length={length}, frame_interval={interval}; implicit padding is disabled"
        )
    max_start = num_frames - required
    if mode == "start":
        start = 0
    elif mode == "center":
        start = max_start // 2
    else:
        start = int(torch.randint(max_start + 1, (), generator=generator).item())
    return start + torch.arange(length, dtype=torch.long) * interval


def uniform_frame_indices(num_frames: int, num_samples: int) -> Tensor:
    """Select one centered index from each of ``num_samples`` equal temporal bins."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    samples = _positive_int(num_samples, "num_samples")
    if num_frames < samples:
        raise ValueError(
            f"cannot choose {samples} distinct uniform frames from a {num_frames}-frame video"
        )
    offsets = (torch.arange(samples, dtype=torch.float64) + 0.5) * num_frames / samples
    return offsets.floor().to(torch.long).clamp_max(num_frames - 1)


def random_segment_indices(
    num_frames: int,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly select one frame from each equal temporal segment."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    samples = _positive_int(num_samples, "num_samples")
    if num_frames < samples:
        raise ValueError(
            f"cannot choose {samples} distinct segments from a {num_frames}-frame video"
        )

    boundaries = torch.linspace(0, num_frames, samples + 1, dtype=torch.float64).floor().long()
    chosen: list[int] = []
    for left, right in zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True):
        if right <= left:
            raise RuntimeError("empty temporal segment produced by invalid sampling boundaries")
        offset = int(torch.randint(right - left, (), generator=generator).item())
        chosen.append(left + offset)
    return torch.tensor(chosen, dtype=torch.long)


@dataclass(frozen=True)
class ClipSampler:
    """Serializable clip sampling policy used by the LeRobot dataset."""

    clip_length: int
    frame_interval: int = 1
    mode: ClipSamplingMode = "random"

    def __post_init__(self) -> None:
        _positive_int(self.clip_length, "clip_length")
        _positive_int(self.frame_interval, "frame_interval")
        if self.mode not in {"start", "center", "random"}:
            raise ValueError("mode must be start, center, or random")

    @property
    def required_frames(self) -> int:
        return clip_span(self.clip_length, self.frame_interval)

    def __call__(
        self,
        num_frames: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return sample_clip_indices(
            num_frames,
            self.clip_length,
            frame_interval=self.frame_interval,
            mode=self.mode,
            generator=generator,
        )


__all__ = [
    "ClipSampler",
    "ClipSamplingMode",
    "clip_span",
    "is_clip_long_enough",
    "random_segment_indices",
    "sample_clip_indices",
    "uniform_frame_indices",
]
