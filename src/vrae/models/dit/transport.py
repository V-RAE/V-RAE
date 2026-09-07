from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from vrae.models.dit.guidance import (
    GuidanceInterval,
    guided_model_forward,
    split_full_base,
)


def expand_time(time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if time.ndim == 0:
        time = time.expand(value.shape[0])
    if time.ndim != 1 or time.shape[0] != value.shape[0]:
        raise ValueError(f"time must have shape [{value.shape[0]}], got {tuple(time.shape)}")
    return time.reshape(time.shape[0], *([1] * (value.ndim - 1)))


class FlowMatchingTransport:
    """Shifted logit-normal flow matching on x_t=(1-t)x_1+t*x_0."""

    def __init__(
        self,
        prediction: str = "velocity",
        time_dist_type: str = "logit-normal_0_1",
        time_dist_shift: float = 1.0,
        t_eps: float = 0.05,
        base_model_coeff: float = 1.0,
    ) -> None:
        if prediction not in {"velocity", "x"}:
            raise ValueError("prediction must be 'velocity' or 'x'")
        parts = str(time_dist_type).split("_")
        if len(parts) != 3 or parts[0] != "logit-normal":
            raise ValueError("time_dist_type must have form 'logit-normal_MU_SIGMA'")
        self.time_mu = float(parts[1])
        self.time_sigma = float(parts[2])
        if self.time_sigma <= 0:
            raise ValueError("logit-normal sigma must be positive")
        if float(time_dist_shift) <= 0:
            raise ValueError("time_dist_shift must be positive")
        if not 0 < float(t_eps) <= 1:
            raise ValueError("t_eps must be in (0,1]")
        if float(base_model_coeff) < 0:
            raise ValueError("base_model_coeff must be non-negative")
        self.prediction = prediction
        self.time_dist_type = str(time_dist_type)
        self.time_dist_shift = float(time_dist_shift)
        self.t_eps = float(t_eps)
        self.base_model_coeff = float(base_model_coeff)

    def shift_time(self, time: torch.Tensor) -> torch.Tensor:
        shift = self.time_dist_shift
        return shift * time / (1 + (shift - 1) * time)

    def sample_time(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        logits = torch.randn(
            int(batch_size),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        time = (self.time_mu + self.time_sigma * logits).sigmoid()
        return self.shift_time(time)

    def sample(
        self,
        clean: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not clean.is_floating_point():
            raise TypeError("clean data must be floating point")
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        time = self.sample_time(clean.shape[0], device=clean.device, generator=generator)
        return time, noise, clean

    def interpolate(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean and noise tensors must have matching shapes")
        expanded = expand_time(time.to(device=clean.device, dtype=clean.dtype), clean)
        return (1 - expanded) * clean + expanded * noise

    def target_velocity(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if clean.shape != noisy.shape:
            raise ValueError("clean and noisy tensors must have matching shapes")
        expanded = expand_time(time.to(device=noisy.device, dtype=noisy.dtype), noisy)
        return (noisy - clean) / expanded.clamp_min(self.t_eps)

    def convert_model_prediction(
        self,
        output: torch.Tensor,
        noisy: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if output.shape != noisy.shape:
            raise ValueError("model output and noisy state must have matching shapes")
        if self.prediction == "velocity":
            return output
        expanded = expand_time(time.to(device=noisy.device, dtype=noisy.dtype), noisy)
        return (noisy - output) / expanded.clamp_min(self.t_eps)

    # The public reference uses this shorter spelling.
    convert_model_pred = convert_model_prediction

    def training_losses(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        generator: torch.Generator | None = None,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        clean_mask: torch.Tensor | None = None,
        reduction: str = "mean",
        request_base_output: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if not clean.is_floating_point() or clean.ndim < 2:
            raise ValueError("clean must be a batched floating-point tensor")
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        batch_size, chunks = clean.shape[:2]
        has_clean_mask = clean_mask is not None
        if clean_mask is None:
            clean_mask = torch.zeros(
                batch_size,
                chunks,
                dtype=torch.bool,
                device=clean.device,
            )
        else:
            clean_mask = torch.as_tensor(
                clean_mask,
                device=clean.device,
                dtype=torch.bool,
            )
            if clean_mask.shape != (batch_size, chunks):
                raise ValueError(
                    f"clean_mask must have shape [{batch_size},{chunks}], "
                    f"got {tuple(clean_mask.shape)}"
                )
        target_mask = (~clean_mask).reshape(
            batch_size,
            chunks,
            *([1] * (clean.ndim - 2)),
        )
        if not bool(target_mask.flatten(1).any(dim=1).all()):
            raise ValueError("clean_mask must leave at least one target chunk per sample")
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        elif noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        if time is None:
            time = self.sample_time(clean.shape[0], device=clean.device, generator=generator)
        else:
            if time.ndim == 0:
                time = time.expand(clean.shape[0])
            if time.ndim != 1 or time.shape[0] != clean.shape[0]:
                raise ValueError(f"time must have shape [{clean.shape[0]}]")
            time = time.to(device=clean.device, dtype=torch.float32)

        effective_noise = torch.where(target_mask, noise, torch.zeros_like(noise))
        noisy = self.interpolate(clean, effective_noise, time)
        noisy = torch.where(target_mask, noisy, clean)
        target = self.target_velocity(clean, noisy, time)
        target = torch.where(target_mask, target, torch.zeros_like(target))
        arguments = dict(model_kwargs or {})
        if has_clean_mask:
            if "clean_mask" in arguments:
                raise ValueError(
                    "clean_mask must be passed to training_losses, not model_kwargs"
                )
            arguments["clean_mask"] = clean_mask
        if context is not None:
            if "context" in arguments or "context_latents" in arguments:
                raise ValueError("context was supplied both positionally and in model_kwargs")
            arguments["context"] = context
        if request_base_output is None:
            request_base_output = self.base_model_coeff != 0
        if request_base_output:
            arguments.setdefault("return_base", True)
        output = model(noisy, time, **arguments)
        if isinstance(output, torch.Tensor):
            full_output = output
            base_output = None
        else:
            full_output, base_output = split_full_base(output)
        full_velocity = self.convert_model_prediction(full_output, noisy, time)
        error_full = (full_velocity.float() - target.float()).square()
        valid_full = target_mask.expand_as(error_full)
        if reduction == "mean":
            loss_full = error_full.masked_select(valid_full).mean()
        else:
            loss_full = error_full.masked_fill(~valid_full, 0).flatten(1).sum(dim=1)
            loss_full = loss_full / valid_full.flatten(1).sum(dim=1)
        if base_output is None:
            loss_base = torch.zeros_like(loss_full)
        else:
            base_velocity = self.convert_model_prediction(base_output, noisy, time)
            error_base = (base_velocity.float() - target.float()).square()
            if reduction == "mean":
                loss_base = error_base.masked_select(valid_full).mean()
            else:
                loss_base = error_base.masked_fill(~valid_full, 0).flatten(1).sum(dim=1)
                loss_base = loss_base / valid_full.flatten(1).sum(dim=1)
        loss_total = loss_full + self.base_model_coeff * loss_base
        return {
            "loss": loss_total,
            "loss_total": loss_total,
            "loss_full": loss_full,
            "loss_base": loss_base,
            "t": time,
            "noise": noise,
            "noisy": noisy,
            "target_velocity": target,
        }

    @torch.no_grad()
    def euler_sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        unconditional_kwargs: Mapping[str, Any] | None = None,
        num_steps: int = 100,
        steps: int | None = None,
        cfg_scale: float | torch.Tensor = 1.0,
        ig_scale: float | torch.Tensor = 1.0,
        ig_intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
        return_trajectory: bool = False,
        clean_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps is not None:
            num_steps = int(steps)
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if not noise.is_floating_point() or noise.ndim < 2:
            raise ValueError("noise must be a batched floating-point tensor")
        conditional_arguments = dict(model_kwargs or {})
        unconditional_arguments = dict(unconditional_kwargs or {})
        condition_chunks = 0
        if clean_condition is not None:
            clean_condition = clean_condition.to(device=noise.device)
            if clean_condition.ndim != noise.ndim:
                raise ValueError(
                    "clean_condition must have the same rank as noise, got "
                    f"{clean_condition.ndim} and {noise.ndim}"
                )
            if clean_condition.shape[0] != noise.shape[0]:
                raise ValueError("clean_condition and noise must have the same batch size")
            if clean_condition.shape[2:] != noise.shape[2:]:
                raise ValueError(
                    "clean_condition and noise must match in all dimensions except time"
                )
            condition_chunks = int(clean_condition.shape[1])
            clean_mask = torch.ones(
                noise.shape[0],
                condition_chunks + noise.shape[1],
                dtype=torch.bool,
                device=noise.device,
            )
            clean_mask[:, condition_chunks:] = False
            for arguments in (conditional_arguments, unconditional_arguments):
                if "clean_mask" in arguments:
                    raise ValueError(
                        "clean_mask is derived from clean_condition during sampling"
                    )
                arguments["clean_mask"] = clean_mask
        value = noise
        time_grid = torch.linspace(
            1,
            0,
            int(num_steps) + 1,
            device=value.device,
            dtype=torch.float32,
        )
        time_grid = self.shift_time(time_grid)
        trajectory = [value.clone()] if return_trajectory else None
        for index in range(int(num_steps)):
            current = time_grid[index]
            following = time_grid[index + 1]
            time = current.expand(value.shape[0])
            model_value = value
            if clean_condition is not None:
                model_value = torch.cat((clean_condition, value), dim=1)
            prediction = guided_model_forward(
                model,
                model_value,
                time,
                condition_kwargs=conditional_arguments,
                unconditional_kwargs=unconditional_arguments,
                cfg_scale=cfg_scale,
                ig_scale=ig_scale,
                ig_intervals=ig_intervals,
            )
            if clean_condition is not None:
                prediction = prediction[:, condition_chunks:]
            velocity = self.convert_model_prediction(prediction, value, time)
            value = value - (current - following).to(value.dtype) * velocity
            if trajectory is not None:
                trajectory.append(value.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return value


FutureFlowMatchingTransport = FlowMatchingTransport


__all__ = ["FlowMatchingTransport", "FutureFlowMatchingTransport", "expand_time"]
