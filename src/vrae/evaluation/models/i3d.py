from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vrae.evaluation.models.checkpoint_path import local_checkpoint_entry


class I3DFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        checkpoint_root: str | Path,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        checkpoint = local_checkpoint_entry(checkpoint, checkpoint_root=checkpoint_root)
        self.model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> I3DFeatureExtractor:
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("I3D expects uncompressed [B,T,3,H,W]")
        if video.is_floating_point():
            if video.min() < 0 or video.max() > 1:
                raise ValueError("floating I3D input must be in RGB range [0,1]")
            # Multiplication is followed by uint8 truncation, not rounding.
            video = video.clamp(0, 1).mul(255).to(torch.uint8)
        elif video.dtype != torch.uint8:
            raise TypeError("I3D input must be uint8 or floating RGB in [0,1]")
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        outputs: list[torch.Tensor] = []
        batch_size = int(getattr(self, "batch_size", 32))
        for start in range(0, video.shape[0], batch_size):
            batch = video[start : start + batch_size]
            output = self.model(batch, rescale=True, resize=True, return_features=True)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, dict):
                output = output.get("features", next(iter(output.values())))
            output = output.flatten(1).float()
            if output.shape[1] != 400:
                raise ValueError(f"Expected Kinetics-400 I3D dimension 400, got {output.shape[1]}")
            outputs.append(output)
        return torch.cat(outputs, dim=0)
