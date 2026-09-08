from __future__ import annotations

import torch


def add_reconstruction_noise(
    clean_latents: torch.Tensor,
    tau: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the supported V-RAE reconstruction latent augmentation."""
    if clean_latents.ndim not in {5, 6}:
        raise ValueError(f"Expected clean latents [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(clean_latents.shape)}")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    shape = (*clean_latents.shape[:2], *((1,) * (clean_latents.ndim - 2)))
    sigma = torch.rand(
        shape, device=clean_latents.device, dtype=clean_latents.dtype, generator=generator
    )
    sigma = sigma * float(tau)
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    return clean_latents + sigma * noise, sigma
