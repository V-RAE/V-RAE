"""FastWAM video/action world-model wrapper.

The wrapper is deliberately loader-agnostic.  A Wan-compatible video expert,
VAE, and (optionally) text stack are injected by the caller; this repository has
no FastWAM-specific checkpoint loader.  The denoising core is nevertheless the
same one used by Wan22: prepare both streams, mix only the video encoder layers
through :class:`MoT`, and project each stream back to its data space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vrae.models.dit.action_dit import ActionDiT
from vrae.models.dit.mot import MoT


class WanContinuousFlowMatchScheduler:
    """Continuous flow-matching scheduler with Wan22's shifted time grid."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        if int(num_train_timesteps) <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if float(shift) <= 0:
            raise ValueError("shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        grid = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1, dtype=torch.float64)[:-1]
        values = torch.exp(-2.0 * ((self._phi(grid, self.shift) - 0.5) ** 2))
        self._weight_min = float(values.min())
        self._weight_norm = float((values - self._weight_min).mean())

    @staticmethod
    def _phi(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1 + (shift - 1) * value)

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        uniform = torch.rand(int(batch_size), device=device)
        return (self._phi(uniform, self.shift) * self.num_train_timesteps).to(dtype=dtype)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep.to(device=original_samples.device, dtype=original_samples.dtype)
        sigma = sigma / self.num_train_timesteps
        if sigma.ndim:
            sigma = sigma.reshape(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        normalized = timestep.to(dtype=torch.float32) / self.num_train_timesteps
        values = torch.exp(-2.0 * (normalized - 0.5).square())
        return (values - self._weight_min) / max(self._weight_norm, 1e-10)

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device | str,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be positive")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError("shift must be positive")
        grid = torch.linspace(1, 0, int(num_inference_steps) + 1, device=device)
        sigma = self._phi(grid, shift)
        return (
            (sigma[:-1] * self.num_train_timesteps).to(dtype=dtype),
            (sigma[1:] - sigma[:-1]).to(dtype=dtype),
        )

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        if delta.ndim:
            delta = delta.reshape(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


class FastWAM(nn.Module):
    """World model containing video and action experts joined by MoT."""

    def __init__(
        self,
        video_expert: nn.Module,
        action_expert: ActionDiT,
        mot: MoT,
        vae: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ) -> None:
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if text_encoder is None or not hasattr(text_encoder, "dim"):
                raise ValueError("text_dim is required when no text encoder with `.dim` is provided")
            text_dim = int(text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.text_dim).to(dtype=torch_dtype)
            if self.proprio_dim is not None
            else None
        )
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(video_num_train_timesteps, video_train_shift)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(video_num_train_timesteps, video_infer_shift)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(action_num_train_timesteps, action_train_shift)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(action_num_train_timesteps, action_infer_shift)
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.compile_training_denoise = bool(compile_training_denoise)
        self.mot.compile_training_layers = self.compile_training_denoise
        self.to(device=self.device, dtype=torch_dtype)

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        video_expert: nn.Module | None = None,
        vae: nn.Module | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        text_encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        mot_checkpoint_mixed_attn: bool = False,
        **kwargs: Any,
    ) -> "FastWAM":
        """Build FastWAM from injected components and an action configuration.

        Wan22's heavyweight component downloader is intentionally outside this
        repository.  Callers must provide ``video_expert`` (and usually ``vae``)
        explicitly; ``video_dit_config`` is retained for metadata validation.
        """

        if video_expert is None:
            raise ValueError("video_expert must be provided; V-RAE does not ship a Wan22 loader")
        if action_dit_config is None:
            raise ValueError("action_dit_config is required")
        action_expert = ActionDiT.from_pretrained(
            action_dit_config,
            action_dit_pretrained_path,
            skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            {"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        if text_dim is None and video_dit_config is not None:
            text_dim = video_dit_config.get("text_dim")
        return cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def encode_prompt(self, prompt: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Prompt encoding requires both tokenizer and text_encoder")
        encoded = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        if isinstance(encoded, Mapping):
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
        else:
            ids, mask = encoded
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        return context.to(device=self.device, dtype=self.torch_dtype), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B,{self.proprio_dim}]")
        token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype))[:, None]
        valid = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)
        return torch.cat((context, token), dim=1), torch.cat((context_mask, valid), dim=1)

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("A VAE is required to encode pixel videos")
        if hasattr(self.vae, "encode"):
            return self.vae.encode(video.to(device=self.device, dtype=self.torch_dtype))
        if hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
            return self.vae.model.encode(video.to(self.device))
        raise TypeError("vae must expose encode(video) or vae.model.encode(video)")

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents when the injected VAE exposes a decode method."""

        if self.vae is None or not hasattr(self.vae, "decode"):
            raise ValueError("A VAE exposing decode(latents) is required")
        return self.vae.decode(latents)

    def build_inputs(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a training sample and encode pixels only when needed."""

        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping")
        action = sample.get("action")
        if action is None or action.ndim != 3:
            raise ValueError("sample['action'] must be [B,S,action_dim]")
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None or context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("sample requires context/context_mask or prompt")
            context, context_mask = self.encode_prompt(prompt)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError("context/context_mask must have shapes [B,L,D]/[B,L]")
        proprio = sample.get("proprio")
        if proprio is not None:
            if proprio.ndim == 3:
                proprio = proprio[:, 0]
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        input_latents = sample.get("input_latents", sample.get("video_latents"))
        if input_latents is None:
            if "video" not in sample:
                raise ValueError("sample requires input_latents/video_latents or video")
            input_latents = self._encode_video_latents(sample["video"])
        return {
            "input_latents": input_latents.to(device=self.device, dtype=self.torch_dtype),
            "action": action.to(device=self.device, dtype=self.torch_dtype),
            "context": context.to(device=self.device, dtype=self.torch_dtype),
            "context_mask": context_mask.to(device=self.device, dtype=torch.bool),
            "action_is_pad": sample.get("action_is_pad"),
            "image_is_pad": sample.get("image_is_pad"),
            "fuse_vae_embedding_in_latents": bool(sample.get("fuse_vae_embedding_in_latents", False)),
        }

    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0 or action_seq_len <= 0 or video_tokens_per_frame <= 0:
            raise ValueError("video/action token lengths and tokens_per_frame must be positive")
        mask = torch.zeros(
            (video_seq_len + action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=device,
        )
        if hasattr(self.video_expert, "build_video_to_video_mask"):
            video_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len, video_tokens_per_frame, device
            )
        else:
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = video_mask
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, : min(video_tokens_per_frame, video_seq_len)] = True
        return mask

    def _video_token_layout(self, latents: torch.Tensor) -> tuple[int, int]:
        if latents.ndim == 4:
            return int(latents.shape[1] * latents.shape[2]), int(latents.shape[2])
        if latents.ndim != 5:
            return latents.shape[1], 1
        patch = getattr(self.video_expert, "patch_size", (1, 1, 1))
        if len(patch) == 2:
            patch = (1, *patch)
        f = latents.shape[2] // int(patch[0])
        h = latents.shape[3] // int(patch[1])
        w = latents.shape[4] // int(patch[2])
        return f * h * w, h * w

    def _joint_denoise_core(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool = False,
        action_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_prepared = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if not isinstance(video_prepared, tuple) or len(video_prepared) < 10:
            raise TypeError("video_expert.prepare must return the Wan22 tensor tuple")
        (
            video_tokens,
            video_t,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = video_prepared[:10]
        (
            action_tokens,
            _action_t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(latents_action, timestep_action, context, context_mask)
        video_tokens, action_tokens = self.mot.forward_joint_core(
            video_tokens,
            action_tokens,
            video_freqs,
            action_freqs,
            video_t_mod,
            action_t_mod,
            video_context,
            video_context_mask,
            action_context,
            action_context_mask,
            attention_mask,
        )
        video_output = self.video_expert.post(video_tokens, video_t, f_video, h_video, w_video)
        return video_output, self.action_expert.post(action_tokens)

    @torch.no_grad()
    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise actions with a video K/V cache prepared once per image."""

        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(latents_action, timestep_action, context, context_mask)
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens,
            action_freqs,
            action_t_mod,
            action_context,
            action_context_mask,
            video_cache_k,
            video_cache_v,
            action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    def training_loss(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted video/action flow-matching losses from a sample."""

        inputs = self.build_inputs(sample)
        video = inputs["input_latents"]
        action = inputs["action"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        batch = video.shape[0]
        t_video = self.train_video_scheduler.sample_training_t(batch, self.device, video.dtype)
        t_action = self.train_action_scheduler.sample_training_t(batch, self.device, action.dtype)
        noise_video = torch.randn_like(video)
        noise_action = torch.randn_like(action)
        noisy_video = self.train_video_scheduler.add_noise(video, noise_video, t_video)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, t_action)
        video_seq_len, tokens_per_frame = self._video_token_layout(noisy_video)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len, noisy_action.shape[1], tokens_per_frame, noisy_video.device
        )
        pred_video, pred_action = self._joint_denoise_core(
            noisy_video,
            noisy_action,
            t_video,
            t_action,
            context,
            context_mask,
            attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )
        target_video = self.train_video_scheduler.training_target(video, noise_video, t_video)
        target_action = self.train_action_scheduler.training_target(action, noise_action, t_action)
        loss_video = F.mse_loss(pred_video.float(), target_video.float())
        loss_action = F.mse_loss(pred_action.float(), target_action.float())
        loss_video = loss_video * self.train_video_scheduler.training_weight(t_video).mean().to(loss_video)
        loss_action = loss_action * self.train_action_scheduler.training_weight(t_action).mean().to(loss_action)
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return total, {"loss_video": float(loss_video.detach()), "loss_action": float(loss_action.detach())}

    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("FastWAM checkpoint must contain a mapping")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError("FastWAM checkpoint is missing `mot` or legacy `dit`")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return dict(payload)

    def forward(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample)


__all__ = ["FastWAM", "WanContinuousFlowMatchScheduler"]
