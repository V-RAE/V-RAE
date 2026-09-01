from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, channels: int, value: torch.Tensor) -> torch.Tensor:
    coordinate = torch.arange(size, device=value.device, dtype=value.dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinate.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).expand(channels, 1, size, size).contiguous()
    return window


def ssim_per_frame(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    if real.shape != reconstructed.shape or real.ndim != 4:
        raise ValueError("SSIM expects equal [N,C,H,W] tensors")
    channels = real.shape[1]
    window = _gaussian_window(11, 1.5, channels, real)
    mu_real = F.conv2d(real, window, padding=5, groups=channels)
    mu_reconstructed = F.conv2d(reconstructed, window, padding=5, groups=channels)
    variance_real = F.conv2d(real.square(), window, padding=5, groups=channels) - mu_real.square()
    variance_reconstructed = (
        F.conv2d(reconstructed.square(), window, padding=5, groups=channels)
        - mu_reconstructed.square()
    )
    covariance = (
        F.conv2d(real * reconstructed, window, padding=5, groups=channels)
        - mu_real * mu_reconstructed
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_real * mu_reconstructed + c1) * (2 * covariance + c2)) / (
        (mu_real.square() + mu_reconstructed.square() + c1)
        * (variance_real + variance_reconstructed + c2)
    )
    return score.flatten(1).mean(1)


class PairedMetricAccumulator:
    """Accumulate frame metrics in bounded float32 microbatches.

    Videos stay in ``[B,T,C,H,W]`` form at the call site, frames are flattened
    and processed in bounded chunks, and global sums use float64 so uneven
    final batches and distributed shards remain exact.
    """

    _SQUARED_ERROR = 0
    _PIXELS = 1
    _FRAME_PSNR = 2
    _FRAMES = 3
    _SSIM = 4
    _LPIPS = 5
    _PSNR_MSE_FLOOR = 1.0e-10

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        batch_size: int = 32,
        lpips_model: Any | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lpips_model = lpips_model
        self._totals = torch.zeros(6, dtype=torch.float64, device=self.device)

    @torch.inference_mode()
    def update(
        self,
        real: torch.Tensor,
        reconstructed: torch.Tensor,
        *,
        lpips_model: Any | None = None,
    ) -> None:
        if real.shape != reconstructed.shape or real.ndim != 5:
            raise ValueError("Paired metrics expect equal [B,T,C,H,W] tensors")
        if real.shape[2] != 3:
            raise ValueError("Paired metrics require RGB videos")

        frame_count = int(real.shape[0] * real.shape[1])
        real_frames = real.reshape(frame_count, *real.shape[2:])
        reconstructed_frames = reconstructed.reshape(frame_count, *reconstructed.shape[2:])
        pixels_per_frame = int(real_frames[0].numel())
        metric = self.lpips_model if lpips_model is None else lpips_model

        for start in range(0, frame_count, self.batch_size):
            stop = min(start + self.batch_size, frame_count)
            target = real_frames[start:stop].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            prediction = reconstructed_frames[start:stop].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
                raise ValueError("Paired metrics received non-finite values")
            count = int(stop - start)
            squared_error = (prediction - target).square()
            frame_squared_error = squared_error.flatten(1).sum(dim=1, dtype=torch.float64)
            frame_mse = frame_squared_error / float(pixels_per_frame)
            frame_psnr = -10.0 * torch.log10(frame_mse.clamp_min(self._PSNR_MSE_FLOOR))
            ssim = ssim_per_frame(target, prediction)

            self._totals[self._SQUARED_ERROR].add_(frame_squared_error.sum())
            self._totals[self._PIXELS].add_(float(count * pixels_per_frame))
            self._totals[self._FRAME_PSNR].add_(frame_psnr.sum())
            self._totals[self._FRAMES].add_(float(count))
            self._totals[self._SSIM].add_(ssim.double().sum())
            if metric is not None:
                lpips = metric(target * 2.0 - 1.0, prediction * 2.0 - 1.0)
                self._totals[self._LPIPS].add_(lpips.reshape(count, -1).mean(1).double().sum())

    def compute(self, *, synchronize: bool = False) -> dict[str, float]:
        totals = self._totals.clone()
        if synchronize and dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        values = totals.cpu().tolist()
        frame_count = int(round(values[self._FRAMES]))
        pixel_count = int(round(values[self._PIXELS]))
        if frame_count == 0 or pixel_count == 0:
            raise ValueError("Paired metric accumulator is empty")
        mse = values[self._SQUARED_ERROR] / pixel_count
        return {
            "LPIPS": values[self._LPIPS] / frame_count,
            "PSNR": values[self._FRAME_PSNR] / frame_count,
            "SSIM": values[self._SSIM] / frame_count,
            "PSNR_global": float("inf") if mse <= 0.0 else -10.0 * math.log10(mse),
        }
