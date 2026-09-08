from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class LatentNormalizer(nn.Module):
    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean).flatten().float()
        std = torch.as_tensor(std).flatten().float()
        if mean.shape != std.shape or mean.numel() == 0:
            raise ValueError("Latent mean/std must be non-empty vectors with equal shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("Latent statistics must be finite and std must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.metadata = dict(metadata or {})

    def _view(self, value: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            shape = (1, 1, 1, 1, -1) if latents.shape[-1] == self.mean.numel() else (1, 1, -1, 1, 1)
        elif latents.ndim == 4:
            shape = (1, 1, 1, -1)
        else:
            raise ValueError(f"Expected grid [B,T,C,H,W] or tokens [B,T,N,C], got {latents.shape}")
        return value.to(device=latents.device, dtype=latents.dtype).view(*shape)

    def normalize(self, clean_latents: torch.Tensor) -> torch.Tensor:
        return (clean_latents - self._view(self.mean, clean_latents)) / self._view(
            self.std, clean_latents
        )

    def denormalize(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return normalized_latents * self._view(self.std, normalized_latents) + self._view(
            self.mean, normalized_latents
        )

    def forward(self, clean_latents: torch.Tensor) -> torch.Tensor:
        return self.normalize(clean_latents)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        torch.save(
            {"mean": self.mean.cpu(), "std": self.std.cpu(), "metadata": self.metadata}, temporary
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> LatentNormalizer:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(payload["mean"], payload["std"], metadata=payload.get("metadata", {}))


class DistributedLatentStats:
    def __init__(self, channels: int, *, device: torch.device | str = "cpu") -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sum_squares = torch.zeros_like(self.sum)
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self._reduced = False

    @torch.no_grad()
    def update(self, clean_latents: torch.Tensor) -> None:
        if self._reduced:
            raise RuntimeError("Cannot update statistics after distributed reduction")
        if clean_latents.ndim == 6:
            values = clean_latents.permute(0, 1, 2, 4, 5, 3).reshape(-1, self.channels)
        elif clean_latents.ndim == 5:
            if clean_latents.shape[-1] == self.channels:
                values = clean_latents.reshape(-1, self.channels)
            else:
                values = clean_latents.permute(0, 1, 3, 4, 2).reshape(-1, self.channels)
        elif clean_latents.ndim == 4:
            values = clean_latents.reshape(-1, self.channels)
        else:
            raise ValueError("Expected latent grid or token tensor")
        values = values.to(dtype=torch.float64, device=self.sum.device)
        self.sum += values.sum(0)
        self.sum_squares += values.square().sum(0)
        self.count += values.shape[0]

    @torch.no_grad()
    def reduce(self) -> None:
        if self._reduced:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.sum)
            dist.all_reduce(self.sum_squares)
            dist.all_reduce(self.count)
        self._reduced = True

    def finalize(
        self, *, metadata: Mapping[str, Any] | None = None, eps: float = 1.0e-6
    ) -> LatentNormalizer:
        self.reduce()
        if self.count.item() <= 0:
            raise ValueError("Cannot finalize empty latent statistics")
        mean = self.sum / self.count
        variance = (self.sum_squares / self.count - mean.square()).clamp_min(eps**2)
        return LatentNormalizer(mean.float(), variance.sqrt().float(), metadata=metadata)


def validate_normalizer_compatibility(
    normalizer: LatentNormalizer,
    *,
    stage1_metadata: Mapping[str, Any],
    stage1_checkpoint: str,
    dataset: str | None = None,
    split: str | None = None,
    scope: str | None = None,
) -> None:
    from vrae.training.common.contracts import STAGE1_STRUCTURE_FIELDS

    recorded = normalizer.metadata.get("stage1", normalizer.metadata)
    if not isinstance(recorded, Mapping):
        raise ValueError("Latent normalizer stage1 metadata must be a mapping")
    fields = STAGE1_STRUCTURE_FIELDS
    mismatches = [field for field in fields if recorded.get(field) != stage1_metadata.get(field)]
    if mismatches:
        raise ValueError(f"Latent normalizer V-RAE metadata mismatch: {mismatches}")
    if normalizer.metadata.get("stage1_checkpoint") != str(stage1_checkpoint):
        raise ValueError("Latent normalizer was computed from a different V-RAE checkpoint")
    expected_semantics = {"dataset": dataset, "split": split, "scope": scope}
    semantic_mismatches = [
        f"{name}: expected={expected!r}, actual={normalizer.metadata.get(name)!r}"
        for name, expected in expected_semantics.items()
        if expected is not None and normalizer.metadata.get(name) != expected
    ]
    if semantic_mismatches:
        raise ValueError(
            "Latent normalizer dataset semantics mismatch:\n" + "\n".join(semantic_mismatches)
        )


def normalizer_identity(normalizer: LatentNormalizer) -> dict[str, Any]:
    fields = (
        "dataset",
        "split",
        "scope",
        "clean_latent",
        "normalized",
        "stage1_checkpoint",
        "future_relative_frames",
    )
    return {
        **{field: normalizer.metadata.get(field) for field in fields},
        "channels": int(normalizer.mean.numel()),
    }
