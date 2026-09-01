from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vrae.evaluation.models.checkpoint_path import (
    local_checkpoint_directory,
    local_checkpoint_entry,
)
from vrae.training.recon_training.losses import PerceptualGramLoss


class LPIPSMetric(nn.Module):
    """Locally checkpointed VGG-LPIPS evaluator used by reconstruction protocols."""

    def __init__(self, checkpoint: str | Path, *, checkpoint_root: str | Path) -> None:
        super().__init__()
        raw_checkpoint = Path(checkpoint)
        if raw_checkpoint.is_dir():
            checkpoint_dir = local_checkpoint_directory(
                raw_checkpoint,
                checkpoint_root=checkpoint_root,
            )
            self.model = PerceptualGramLoss(
                checkpoint_dir / "vgg16.pt",
                checkpoint_dir / "lpips_vgg.pt",
            ).eval()
            self._uses_state_dict = True
        else:
            checkpoint_file = local_checkpoint_entry(
                raw_checkpoint,
                checkpoint_root=checkpoint_root,
            )
            if checkpoint_file.name == "lpips_vgg.pt":
                self.model = PerceptualGramLoss(
                    checkpoint_file.with_name("vgg16.pt"),
                    checkpoint_file,
                ).eval()
                self._uses_state_dict = True
            else:
                self.model = torch.jit.load(str(checkpoint_file), map_location="cpu").eval()
                self._uses_state_dict = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> LPIPSMetric:
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        if real.shape != reconstructed.shape or real.ndim != 4 or real.shape[1] != 3:
            raise ValueError("LPIPS expects equal RGB frame batches [N,3,H,W]")
        if self._uses_state_dict:
            output = self.model.per_sample_lpips(real.float(), reconstructed.float())
        else:
            output = self.model(real.float(), reconstructed.float())
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("lpips", next(iter(output.values())))
        output = output.reshape(real.shape[0], -1).mean(1)
        if not torch.isfinite(output).all():
            raise ValueError("LPIPS evaluator returned non-finite values")
        return output


__all__ = ["LPIPSMetric"]
