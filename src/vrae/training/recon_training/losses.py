from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def temporal_difference_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if reconstructed.shape != target.shape:
        raise ValueError("Reconstruction and target shapes differ")
    if reconstructed.shape[1] < 2:
        return reconstructed.new_zeros(())
    return F.l1_loss(reconstructed[:, 1:] - reconstructed[:, :-1], target[:, 1:] - target[:, :-1])


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([[-0.030, -0.088, -0.188]])[:, :, None, None])
        self.register_buffer("scale", torch.tensor([[0.458, 0.448, 0.450]])[:, :, None, None])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.shift) / self.scale


class LinearCalibration(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.model = nn.Conv2d(channels, 1, kernel_size=1, bias=False)


def _unwrap_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("Perceptual checkpoint must contain a state mapping")
    for key in ("state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return payload


def normalize_lpips_vgg_calibration(
    state: Mapping[str, torch.Tensor],
    channels: tuple[int, ...] = (64, 128, 256, 512, 512),
) -> dict[str, torch.Tensor]:
    expected = {f"{index}.model.weight" for index in range(len(channels))}
    if set(state) == expected:
        normalized = dict(state)
    else:
        upstream = {f"lin{index}.model.1.weight" for index in range(len(channels))}
        if set(state) != upstream:
            raise ValueError(
                "LPIPS VGG calibration must use the standard lin0..lin4 model.1 weights"
            )
        normalized = {
            f"{index}.model.weight": state[f"lin{index}.model.1.weight"]
            for index in range(len(channels))
        }
    for index, channel_count in enumerate(channels):
        key = f"{index}.model.weight"
        expected_shape = (1, channel_count, 1, 1)
        if tuple(normalized[key].shape) != expected_shape:
            raise ValueError(
                f"LPIPS calibration {key} has shape {tuple(normalized[key].shape)}, "
                f"expected {expected_shape}"
            )
    return normalized


class PerceptualGramLoss(nn.Module):
    taps = (3, 8, 15, 22, 29)
    channels = (64, 128, 256, 512, 512)

    def __init__(
        self,
        backbone_checkpoint: str | Path,
        calibration_checkpoint: str | Path,
        *,
        channels_last: bool = False,
        weight_dtype: str = "fp32",
    ) -> None:
        super().__init__()
        backbone_path = Path(backbone_checkpoint)
        calibration_path = Path(calibration_checkpoint)
        if not backbone_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError(
                f"Local perceptual weights are required: {backbone_path}, {calibration_path}"
            )
        vgg = models.vgg16(weights=None)
        state = _unwrap_state(torch.load(backbone_path, map_location="cpu", weights_only=True))
        try:
            vgg.load_state_dict(state, strict=True)
        except RuntimeError:
            vgg.features.load_state_dict(state, strict=True)
        self.features = vgg.features
        self.channels_last = bool(channels_last)
        if getattr(self, "channels_last", False):
            self.features.to(memory_format=torch.channels_last)
        selected_weight_dtype = str(weight_dtype).lower()
        if selected_weight_dtype not in {"fp32", "bf16"}:
            raise ValueError("Perceptual weight dtype must be fp32 or bf16")
        self.weight_dtype = selected_weight_dtype
        if self.weight_dtype == "bf16":
            self.features.to(dtype=torch.bfloat16)
        self.scaling = ScalingLayer()
        self.calibration = nn.ModuleList(LinearCalibration(channels) for channels in self.channels)
        calibration = _unwrap_state(
            torch.load(calibration_path, map_location="cpu", weights_only=True)
        )
        self.calibration.load_state_dict(
            normalize_lpips_vgg_calibration(calibration, self.channels),
            strict=True,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> PerceptualGramLoss:
        super().train(False)
        return self

    @staticmethod
    def _normalized(value: torch.Tensor) -> torch.Tensor:
        return value / value.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-10)

    @staticmethod
    def _gram(value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        flattened = value.reshape(batch, channels, height * width)
        return flattened @ flattened.transpose(1, 2) / (channels * height * width)

    def _stream_losses(
        self,
        target: torch.Tensor,
        reconstructed: torch.Tensor,
        *,
        calculate_lpips: bool,
        calculate_gram: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate VGG losses without retaining a second Python list of every feature."""

        if getattr(self, "channels_last", False):
            target = target.contiguous(memory_format=torch.channels_last)
            reconstructed = reconstructed.contiguous(memory_format=torch.channels_last)
        target_feature = self.scaling(target)
        reconstructed_feature = self.scaling(reconstructed)
        selected = (
            {
                layer_index: calibration
                for layer_index, calibration in zip(self.taps, self.calibration, strict=True)
            }
            if calculate_lpips
            else {}
        )
        lpips = target.new_zeros(target.shape[0])
        gram = target.new_zeros(())
        layers = list(self.features)
        deferred_gram = False
        for layer_index, layer in enumerate(layers):
            target_feature = layer(target_feature)
            reconstructed_feature = layer(reconstructed_feature)
            calibration = selected.get(layer_index)
            if calibration is not None:
                difference = (
                    self._normalized(target_feature) - self._normalized(reconstructed_feature)
                ).square()
                lpips = lpips + calibration.model(difference).flatten(1).mean(1)
            if calculate_gram:
                next_is_inplace = layer_index + 1 < len(layers) and bool(
                    getattr(layers[layer_index + 1], "inplace", False)
                )
                if next_is_inplace:
                    deferred_gram = True
                    continue
                gram_term = F.mse_loss(
                    self._gram(reconstructed_feature),
                    self._gram(target_feature),
                )
                # torchvision VGG uses in-place ReLUs. The legacy feature-list
                # path appended the preceding convolution output and then let
                # the ReLU mutate that same tensor, so both list entries were
                # evaluated after the ReLU. Defer a feature whose next layer is
                # in-place, then account for both aliases after the mutation.
                if deferred_gram:
                    gram = gram + gram_term
                    deferred_gram = False
                gram = gram + gram_term
        if deferred_gram:
            raise RuntimeError("Perceptual Gram stream ended before an in-place layer")
        return lpips.mean(), gram

    def per_sample_lpips(
        self,
        target: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        if target.shape != reconstructed.shape or target.ndim != 4 or target.shape[1] != 3:
            raise ValueError("LPIPS expects equal RGB frame batches [N,3,H,W]")
        # Evaluation only needs the five tapped VGG activations. Advance the
        # real/fake streams together and consume each tap immediately instead
        # of retaining all 30 layer outputs for both sides. This is equivalent
        # to the reference LPIPS calculation while bounding memory by the
        # current layer's activation pair.
        if getattr(self, "channels_last", False):
            target = target.contiguous(memory_format=torch.channels_last)
            reconstructed = reconstructed.contiguous(memory_format=torch.channels_last)
        target_feature = self.scaling(target)
        reconstructed_feature = self.scaling(reconstructed)
        selected = {
            layer_index: calibration
            for layer_index, calibration in zip(self.taps, self.calibration, strict=True)
        }
        lpips = target.new_zeros(target.shape[0], dtype=torch.float32)
        for layer_index, layer in enumerate(self.features):
            target_feature = layer(target_feature)
            reconstructed_feature = layer(reconstructed_feature)
            calibration = selected.get(layer_index)
            if calibration is None:
                continue
            target_normalized = target_feature / (
                target_feature.square().sum(dim=1, keepdim=True).sqrt() + 1.0e-10
            )
            reconstructed_normalized = reconstructed_feature / (
                reconstructed_feature.square().sum(dim=1, keepdim=True).sqrt() + 1.0e-10
            )
            difference = (target_normalized - reconstructed_normalized).square()
            lpips.add_(calibration.model(difference).mean(dim=(1, 2, 3)).float())
        return lpips

    def forward(
        self, target: torch.Tensor, reconstructed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target.shape != reconstructed.shape or target.ndim != 4 or target.shape[1] != 3:
            raise ValueError("LPIPS+Gram expects equal RGB frame batches [N,3,H,W]")
        native_is_gram_resolution = tuple(target.shape[-2:]) == (224, 224)
        lpips, gram = self._stream_losses(
            target,
            reconstructed,
            calculate_lpips=True,
            calculate_gram=native_is_gram_resolution,
        )
        # The public reference evaluates LPIPS at native resolution, but evaluates
        # the VGG Gram branch at 224x224. Reuse the native stream only when
        # they already have that exact spatial contract.
        if not native_is_gram_resolution:
            target_gram = F.interpolate(
                target, size=(224, 224), mode="bilinear", align_corners=False
            )
            reconstructed_gram = F.interpolate(
                reconstructed, size=(224, 224), mode="bilinear", align_corners=False
            )
            _, gram = self._stream_losses(
                target_gram,
                reconstructed_gram,
                calculate_lpips=False,
                calculate_gram=True,
            )
        return lpips, gram


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        perceptual: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "l1": float(config.get("l1", 1.0)),
            "lpips": float(config.get("lpips", 0.0)),
            "gram": float(config.get("gram", 0.0)),
            "temporal_difference": float(config.get("temporal_difference", 0.0)),
        }
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Loss weights must be non-negative")
        perceptual_frames = config.get("perceptual_frames")
        self.perceptual_frames = None if perceptual_frames is None else int(perceptual_frames)
        if self.perceptual_frames is not None and self.perceptual_frames <= 0:
            raise ValueError("loss.perceptual_frames must be positive")
        frames_per_chunk = config.get("perceptual_frames_per_chunk")
        self.perceptual_frames_per_chunk = (
            None if frames_per_chunk is None else int(frames_per_chunk)
        )
        self.perceptual_chunk_size = int(config.get("perceptual_chunk_size", 4))
        if self.perceptual_frames is not None and self.perceptual_frames_per_chunk is not None:
            raise ValueError(
                "loss.perceptual_frames and loss.perceptual_frames_per_chunk are mutually exclusive"
            )
        if self.perceptual_chunk_size <= 0:
            raise ValueError("loss.perceptual_chunk_size must be positive")
        if (
            self.perceptual_frames_per_chunk is not None
            and not 0 < self.perceptual_frames_per_chunk <= self.perceptual_chunk_size
        ):
            raise ValueError(
                "loss.perceptual_frames_per_chunk must be between 1 and perceptual_chunk_size"
            )
        if (self.weights["lpips"] or self.weights["gram"]) and perceptual is None:
            perceptual = PerceptualGramLoss(
                config["backbone_checkpoint"],
                config["calibration_checkpoint"],
                channels_last=bool(config.get("perceptual_channels_last", False)),
                weight_dtype=str(config.get("perceptual_weight_dtype", "fp32")),
            )
        self.perceptual = perceptual
        self._compiled_forward: (
            Callable[
                [torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, dict[str, torch.Tensor]],
            ]
            | None
        ) = None

    def enable_compile(self, **kwargs: Any) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        self._compiled_forward = torch.compile(self._forward_impl, **kwargs)

    def _select_perceptual_frames(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reconstructed.ndim != 5:
            raise ValueError("Perceptual video inputs must be [B,T,C,H,W]")
        batch, time = reconstructed.shape[:2]
        frames_per_chunk = self.perceptual_frames_per_chunk
        if frames_per_chunk is not None:
            chunk_size = self.perceptual_chunk_size
            if time % chunk_size:
                raise ValueError(
                    f"Perceptual video time={time} must be divisible by chunk size={chunk_size}"
                )
            if frames_per_chunk == chunk_size:
                return reconstructed.flatten(0, 1), target.flatten(0, 1)

            chunks = time // chunk_size
            # Sample without replacement inside every decoder chunk. Sorting
            # restores temporal order after the random top-k selection.
            local_indices = (
                torch.rand((batch, chunks, chunk_size), device=reconstructed.device)
                .topk(frames_per_chunk, dim=-1, largest=False)
                .indices
            )
            local_indices = local_indices.sort(dim=-1).values
            chunk_offsets = (
                torch.arange(chunks, device=reconstructed.device) * chunk_size
            ).reshape(1, chunks, 1)
            indices = (local_indices + chunk_offsets).reshape(batch, -1)
            batch_indices = torch.arange(batch, device=reconstructed.device).unsqueeze(1)
            return (
                reconstructed[batch_indices, indices].flatten(0, 1),
                target[batch_indices, indices].flatten(0, 1),
            )

        count = self.perceptual_frames
        if count is None or count >= time:
            return reconstructed.flatten(0, 1), target.flatten(0, 1)

        # Draw one frame from each temporal stratum for every video.  This keeps
        # coverage spread across the clip while preserving independent samples
        # across videos and steps.  Target and reconstruction share the indices.
        boundaries = torch.arange(count + 1, device=reconstructed.device)
        boundaries = torch.div(boundaries * time, count, rounding_mode="floor")
        starts = boundaries[:-1]
        widths = boundaries[1:] - starts
        offsets = torch.floor(torch.rand((batch, count), device=reconstructed.device) * widths).to(
            dtype=torch.long
        )
        indices = starts.unsqueeze(0) + offsets
        batch_indices = torch.arange(batch, device=reconstructed.device).unsqueeze(1)
        return (
            reconstructed[batch_indices, indices].flatten(0, 1),
            target[batch_indices, indices].flatten(0, 1),
        )

    def _forward_impl(
        self, reconstructed: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if reconstructed.shape != target.shape:
            raise ValueError("Reconstruction and target shapes differ")
        if reconstructed.ndim == 6:
            batch, time, views, channels, height, width = reconstructed.shape
            reconstructed = reconstructed.permute(0, 2, 1, 3, 4, 5).reshape(batch * views, time, channels, height, width)
            target = target.permute(0, 2, 1, 3, 4, 5).reshape(batch * views, time, channels, height, width)
        elif reconstructed.ndim != 5:
            raise ValueError("Reconstruction inputs must be [B,T,3,H,W] or [B,T,V,3,H,W]")
        zero = reconstructed.new_zeros(())
        l1 = F.l1_loss(reconstructed, target) if self.weights["l1"] else zero
        temporal = (
            temporal_difference_loss(reconstructed, target)
            if self.weights["temporal_difference"]
            else zero
        )
        lpips = gram = zero
        if self.perceptual is not None:
            reconstructed_frames, target_frames = self._select_perceptual_frames(
                reconstructed, target
            )
            reconstructed_frames = reconstructed_frames * 2.0 - 1.0
            target_frames = target_frames * 2.0 - 1.0
            lpips, gram = self.perceptual(target_frames, reconstructed_frames)
        total = (
            self.weights["l1"] * l1
            + self.weights["lpips"] * lpips
            + self.weights["gram"] * gram
            + self.weights["temporal_difference"] * temporal
        )
        return total, {
            "l1": l1.detach(),
            "lpips": lpips.detach(),
            "gram": gram.detach(),
            "temporal_difference": temporal.detach(),
        }

    def forward(
        self,
        reconstructed: torch.Tensor,          # 重建结果；需与 target 形状兼容
        target: torch.Tensor,                 # 目标张量
        stream_ids: torch.Tensor | None = None,  # 可选的 stream 编号
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 阶段 1/3：选择核心计算路径
        # 如果存在预编译版本，就使用它；否则调用普通实现。
        # 两个函数应返回相同结构：
        #   (一个 Tensor, 一个 dict[str, Tensor])
        result = (
            self._compiled_forward(reconstructed, target)
            if self._compiled_forward is not None
            else self._forward_impl(reconstructed, target)
        )

        # 阶段 2/3：判断是否需要统计每个 stream 的误差
        # 只有当：
        #   1. stream_ids 被传入；
        #   2. reconstructed 恰好是 6 维；
        #   3. stream_ids 是 2 维；
        # 才执行下面的 per-view 统计。
        if stream_ids is not None and reconstructed.ndim == 6 and stream_ids.ndim == 2:

            # 阶段 3/3：计算每个 view 的平均绝对误差（L1）
            #
            # 假设 reconstructed 形状为 [B, T, V, H, W, C]：
            #   dim=(0, 1, 3, 4, 5) 会平均掉 B、T、H、W、C，
            #   保留 V，因此 per_view 形状为 [V]。
            #
            # .detach() 表示这个统计量不参与反向传播，只作为日志指标。
            per_view = (
                (reconstructed - target)
                .abs()
                .mean(dim=(0, 1, 3, 4, 5))
                .detach()
            )

            # stream_ids[0] 取第一个 batch 样本的 stream 编号。
            # view 是 view 下标，stream 是对应的实际 stream ID。
            for view, stream in enumerate(stream_ids[0].tolist()):
                # 例如 stream=7 时，新增：
                # result[1]["l1_stream_7"] = per_view[view]
                result[1][f"l1_stream_{int(stream)}"] = per_view[view]

        # 返回主结果和指标字典
        return result
