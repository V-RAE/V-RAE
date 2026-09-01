from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from vrae.data.sampling import uniform_frame_indices

_LEGACY_VIDEOMAE_IMAGE_MEAN = (0.485, 0.456, 0.406)
_LEGACY_VIDEOMAE_IMAGE_STD = (0.229, 0.224, 0.225)


def _load_legacy_videomae_state(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Legacy VideoMAE checkpoint must contain a mapping: {checkpoint}")
    for key in ("model", "module"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not isinstance(payload, Mapping):
        raise TypeError(f"Legacy VideoMAE checkpoint has no state mapping: {checkpoint}")
    state = {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    if len(state) != len(payload):
        non_tensor = sorted(
            str(key) for key, value in payload.items() if not isinstance(value, torch.Tensor)
        )
        raise TypeError(f"Legacy VideoMAE state contains non-tensors: {non_tensor}")
    return state


def _legacy_source_key(target_key: str) -> str | None:
    patch_prefix = "embeddings.patch_embeddings.projection."
    if target_key.startswith(patch_prefix):
        return "encoder.patch_embed.proj." + target_key.removeprefix(patch_prefix)
    prefix = "encoder.layer."
    if not target_key.startswith(prefix):
        return None
    layer_text, separator, suffix = target_key.removeprefix(prefix).partition(".")
    if not separator or not layer_text.isdigit():
        return None
    legacy_prefix = f"encoder.blocks.{int(layer_text)}."
    suffix_map = {
        "attention.output.dense.weight": "attn.proj.weight",
        "attention.output.dense.bias": "attn.proj.bias",
        "intermediate.dense.weight": "mlp.fc1.weight",
        "intermediate.dense.bias": "mlp.fc1.bias",
        "output.dense.weight": "mlp.fc2.weight",
        "output.dense.bias": "mlp.fc2.bias",
        "layernorm_before.weight": "norm1.weight",
        "layernorm_before.bias": "norm1.bias",
        "layernorm_after.weight": "norm2.weight",
        "layernorm_after.bias": "norm2.bias",
    }
    mapped = suffix_map.get(suffix)
    return legacy_prefix + mapped if mapped is not None else None


def _legacy_videomae_to_transformers(
    legacy_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the InternVideo/EVATok VideoMAE-B encoder into Transformers keys."""

    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def source_tensor(source_key: str) -> torch.Tensor:
        if source_key not in legacy_state:
            raise ValueError(f"Legacy VideoMAE checkpoint is missing key: {source_key}")
        value = legacy_state[source_key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Legacy VideoMAE value is not a tensor: {source_key}")
        consumed.add(source_key)
        return value

    qkv_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def split_qkv(layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer in qkv_cache:
            return qkv_cache[layer]
        source_key = f"encoder.blocks.{layer}.attn.qkv.weight"
        qkv = source_tensor(source_key)
        if qkv.ndim != 2 or qkv.shape[0] % 3:
            raise ValueError(
                f"Legacy VideoMAE QKV weight must split evenly on dim 0: "
                f"{source_key} shape={tuple(qkv.shape)}"
            )
        query, key, value = qkv.chunk(3, dim=0)
        qkv_cache[layer] = (query, key, value)
        return qkv_cache[layer]

    attention_prefix = "encoder.layer."
    attention_suffix = ".attention.attention."
    for target_key, target_value in target_state.items():
        value: torch.Tensor | None = None
        if target_key.startswith(attention_prefix) and attention_suffix in target_key:
            layer_text, _, remainder = target_key.removeprefix(attention_prefix).partition(
                attention_suffix
            )
            if layer_text.isdigit():
                layer = int(layer_text)
                parameter, separator, kind = remainder.partition(".")
                if separator and parameter in {"query", "key", "value"}:
                    if kind == "weight":
                        index = {"query": 0, "key": 1, "value": 2}[parameter]
                        value = split_qkv(layer)[index]
                    elif kind == "bias":
                        if parameter == "key":
                            value = torch.zeros_like(target_value)
                        else:
                            source_name = "q_bias" if parameter == "query" else "v_bias"
                            value = source_tensor(f"encoder.blocks.{layer}.attn.{source_name}")
                elif remainder in {"q_bias", "v_bias"}:
                    source_name = remainder
                    value = source_tensor(f"encoder.blocks.{layer}.attn.{source_name}")
        if value is None:
            source_key = _legacy_source_key(target_key)
            if source_key is None:
                raise ValueError(f"Unsupported Transformers VideoMAE target key: {target_key}")
            value = source_tensor(source_key)
        if tuple(value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"Legacy VideoMAE shape mismatch for {target_key}: "
                f"source={tuple(value.shape)} target={tuple(target_value.shape)}"
            )
        converted[target_key] = value

    missing_targets = sorted(set(target_state) - set(converted))
    unexpected_targets = sorted(set(converted) - set(target_state))
    if missing_targets or unexpected_targets:
        raise ValueError(
            "Legacy VideoMAE target coverage mismatch: "
            f"missing={missing_targets} unexpected={unexpected_targets}"
        )
    allowed_unused = {"encoder.norm.weight", "encoder.norm.bias"}
    unused_encoder = sorted(
        key
        for key in legacy_state
        if key.startswith("encoder.") and key not in consumed and key not in allowed_unused
    )
    if unused_encoder:
        raise ValueError(f"Legacy VideoMAE contains unmapped encoder keys: {unused_encoder}")
    return converted


def _load_legacy_videomae(checkpoint: Path) -> nn.Module:
    from transformers import VideoMAEConfig, VideoMAEModel

    config = VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1.0e-6,
        qkv_bias=True,
        use_mean_pooling=True,
    )
    backbone = VideoMAEModel(config)
    converted = _legacy_videomae_to_transformers(
        _load_legacy_videomae_state(checkpoint), backbone.state_dict()
    )
    result = backbone.load_state_dict(converted, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Legacy VideoMAE load mismatch: "
            f"missing={list(result.missing_keys)} unexpected={list(result.unexpected_keys)}"
        )
    return backbone


class VideoDiffAugment:
    """Temporally consistent translation, color, and cutout for video GAN inputs."""

    def __init__(self, *, probability: float, cutout_ratio: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Video DiffAug probability must be in [0,1]")
        if not 0.0 <= cutout_ratio <= 1.0:
            raise ValueError("Video DiffAug cutout ratio must be in [0,1]")
        self.probability = float(probability)
        self.cutout_ratio = float(cutout_ratio)
        self._video_grids: dict[tuple[int, int, int, int, str], tuple[torch.Tensor, ...]] = {}
        self._image_grids: dict[tuple[int, int, int, str], tuple[torch.Tensor, ...]] = {}

    def _video_grid(
        self, batch: int, time: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        key = (batch, time, height, width, str(device))
        if key not in self._video_grids:
            self._video_grids[key] = torch.meshgrid(
                torch.arange(batch, dtype=torch.long, device=device),
                torch.arange(time, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self._video_grids[key]

    def _image_grid(
        self, batch: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        key = (batch, height, width, str(device))
        if key not in self._image_grids:
            self._image_grids[key] = torch.meshgrid(
                torch.arange(batch, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self._image_grids[key]

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0:
            return video
        video = video.float()
        batch, _, time, height, width = video.shape
        device = video.device
        translate, color, cutout = tuple(
            bool(value) for value in (torch.rand(3, device=device) <= self.probability).tolist()
        )
        if not (translate or color or cutout):
            return video
        random = torch.rand(7, batch, 1, 1, 1, device=device)

        if translate:
            delta_height = round(height * 0.125)
            delta_width = round(width * 0.125)
            shift_height = random[0].mul(2 * delta_height + 1).floor().long() - delta_height
            shift_width = random[1].mul(2 * delta_width + 1).floor().long() - delta_width
            grid_batch, grid_time, grid_height, grid_width = self._video_grid(
                batch, time, height, width, device
            )
            grid_height = (grid_height + shift_height).add(1).clamp(0, height + 1)
            grid_width = (grid_width + shift_width).add(1).clamp(0, width + 1)
            padded = F.pad(video, (1, 1, 1, 1, 0, 0))
            video = padded.permute(0, 2, 3, 4, 1).contiguous()[
                grid_batch, grid_time, grid_height, grid_width
            ]
            video = video.permute(0, 4, 1, 2, 3).contiguous()

        if color:
            video = video + random[2].unsqueeze(-1) - 0.5
            channel_mean = video.mean(dim=1, keepdim=True)
            video = (video - channel_mean) * (random[3].unsqueeze(-1) * 2) + channel_mean
            global_mean = video.mean(dim=(1, 2, 3, 4), keepdim=True)
            video = (video - global_mean) * (random[4].unsqueeze(-1) + 0.5) + global_mean

        if cutout and self.cutout_ratio > 0.0:
            cutout_height = round(height * self.cutout_ratio)
            cutout_width = round(width * self.cutout_ratio)
            if cutout_height > 0 and cutout_width > 0:
                offset_height = (
                    random[5, :, 0, 0, 0]
                    .mul(height + (1 - cutout_height % 2))
                    .floor()
                    .long()
                    .view(batch, 1, 1)
                )
                offset_width = (
                    random[6, :, 0, 0, 0]
                    .mul(width + (1 - cutout_width % 2))
                    .floor()
                    .long()
                    .view(batch, 1, 1)
                )
                grid_batch, grid_height, grid_width = self._image_grid(
                    batch, cutout_height, cutout_width, device
                )
                grid_height = (
                    (grid_height + offset_height).sub(cutout_height // 2).clamp(0, height - 1)
                )
                grid_width = (grid_width + offset_width).sub(cutout_width // 2).clamp(0, width - 1)
                mask = torch.ones(batch, height, width, dtype=video.dtype, device=device)
                mask[grid_batch, grid_height, grid_width] = 0
                video = video * mask[:, None, None]
        return video


class LocalBatchNorm1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        affine: bool = True,
        virtual_batch_size: int = 8,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.virtual_batch_size = int(virtual_batch_size)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.virtual_batch_size <= 0:
            raise ValueError("virtual_batch_size must be positive")
        if self.affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        value = value.float()
        groups = math.ceil(value.shape[0] / self.virtual_batch_size)
        if value.shape[0] % groups:
            raise ValueError(
                f"Batch {value.shape[0]} cannot be split into {groups} local normalization groups"
            )
        grouped = value.reshape(groups, -1, value.shape[-2], value.shape[-1])
        mean = grouped.mean(dim=(1, 3), keepdim=True)
        variance = grouped.var(dim=(1, 3), keepdim=True, unbiased=False)
        grouped = (grouped - mean) / torch.sqrt(variance + self.eps)
        if self.affine:
            grouped = grouped * self.weight[None, :, None] + self.bias[None, :, None]
        return grouped.reshape(shape)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        spectral_norm(self, name="weight", n_power_iterations=1, dim=0, eps=1.0e-12)


class ResidualHeadBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (self.block(value) + value) * (1.0 / math.sqrt(2.0))


def _video_head_block(
    channels: int,
    *,
    kernel_size: int,
    norm_type: str,
    norm_eps: float,
    use_spectral_norm: bool,
) -> nn.Sequential:
    normalized = str(norm_type).strip().lower()
    if normalized == "bn":
        norm: nn.Module = LocalBatchNorm1d(channels, eps=norm_eps)
    elif normalized == "sbn":
        norm = nn.SyncBatchNorm(channels, eps=norm_eps)
    elif normalized == "gn":
        if channels % 32:
            raise ValueError("Group-normalized VideoMAE heads require channels divisible by 32")
        norm = nn.GroupNorm(32, channels, eps=norm_eps)
    else:
        raise ValueError(f"Unsupported VideoMAE head normalization: {norm_type}")
    convolution = SpectralConv1d if use_spectral_norm else nn.Conv1d
    return nn.Sequential(
        convolution(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def _video_discriminator_head(
    channels: int,
    *,
    kernel_size: int,
    norm_type: str,
    norm_eps: float,
    use_spectral_norm: bool,
) -> nn.Sequential:
    convolution = SpectralConv1d if use_spectral_norm else nn.Conv1d
    return nn.Sequential(
        _video_head_block(
            channels,
            kernel_size=1,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_spectral_norm=use_spectral_norm,
        ),
        ResidualHeadBlock(
            _video_head_block(
                channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                norm_eps=norm_eps,
                use_spectral_norm=use_spectral_norm,
            )
        ),
        convolution(channels, 1, kernel_size=1),
    )


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def non_saturating_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(fake_logits).mean()


def smooth_non_saturating_discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    real_target = (1.0 - torch.randn_like(real_logits).abs() * 0.15).clamp_min(0.7)
    fake_target = (torch.randn_like(fake_logits).abs() * 0.15).clamp_max(0.3)
    return F.binary_cross_entropy_with_logits(
        real_logits, real_target
    ) + F.binary_cross_entropy_with_logits(fake_logits, fake_target)


def _generator_loss(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized in {"hinge", "hinge_g"}:
        return hinge_generator_loss(logits)
    if normalized == "ns_g_loss":
        return non_saturating_generator_loss(logits)
    raise ValueError(f"Unsupported GAN generator loss: {loss_type}")


def _discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor, loss_type: str
) -> torch.Tensor:
    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized == "hinge":
        return hinge_discriminator_loss(real_logits, fake_logits)
    if normalized == "ns_smooth":
        return smooth_non_saturating_discriminator_loss(real_logits, fake_logits)
    raise ValueError(f"Unsupported GAN discriminator loss: {loss_type}")


def _distributed_logit_means(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> tuple[float, float]:
    statistics = torch.stack(
        (
            real_logits.detach().double().sum(),
            fake_logits.detach().double().sum(),
            real_logits.new_tensor(real_logits.numel(), dtype=torch.float64),
            fake_logits.new_tensor(fake_logits.numel(), dtype=torch.float64),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    if float(statistics[2]) <= 0.0 or float(statistics[3]) <= 0.0:
        raise ValueError("LeCam EMA cannot update from empty logits")
    return (
        float((statistics[0] / statistics[2]).item()),
        float((statistics[1] / statistics[3]).item()),
    )


class LeCamLogitEMA:
    def __init__(self, *, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("LeCam EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.real = 0.0
        self.fake = 0.0
        self.updates = 0

    def update(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> None:
        real, fake = _distributed_logit_means(real_logits, fake_logits)
        decay = self.decay if self.updates > 0 else 0.0
        self.real = self.real * decay + real * (1.0 - decay)
        self.fake = self.fake * decay + fake * (1.0 - decay)
        self.real = max(-10.0, min(10.0, self.real))
        self.fake = max(-10.0, min(10.0, self.fake))
        self.updates += 1

    def regularization(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        value = F.relu(real_logits - self.fake).square().mean()
        value = value + F.relu(self.real - fake_logits).square().mean()
        return value if torch.isfinite(value) else real_logits.new_zeros(())

    def state_dict(self) -> dict[str, float | int]:
        return {
            "real": self.real,
            "fake": self.fake,
            "decay": self.decay,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"real", "fake", "decay", "updates"}
        if set(state) != required:
            raise ValueError(
                f"LeCam EMA state fields mismatch: expected={sorted(required)} "
                f"actual={sorted(state)}"
            )
        if float(state["decay"]) != self.decay:
            raise ValueError("LeCam EMA decay changed across resume")
        updates = int(state["updates"])
        real = float(state["real"])
        fake = float(state["fake"])
        if updates < 0 or not math.isfinite(real) or not math.isfinite(fake):
            raise ValueError("LeCam EMA state is invalid")
        self.real = real
        self.fake = fake
        self.updates = updates


class VideoMAEDiscriminator(nn.Module):
    """Frozen local VideoMAE feature tower with trainable multi-level heads."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        backbone: nn.Module | None = None,
        hidden_size: int | None = None,
        taps: Sequence[int] = (2, 5, 8, 11),
        preprocess_size: int | Sequence[int] | None = None,
        image_mean: Sequence[float] | None = None,
        image_std: Sequence[float] | None = None,
        input_clamp: bool = True,
        head_kernel_size: int = 9,
        head_norm_type: str = "bn",
        spectral_norm_heads: bool = True,
        norm_eps: float = 1.0e-6,
        diff_aug_prob: float = 0.0,
        diff_aug_cutout: float = 0.2,
        temporal_sampling: str = "strict",
    ) -> None:
        super().__init__()
        legacy_checkpoint = False
        if backbone is None:
            checkpoint_path = Path(checkpoint) if checkpoint is not None else None
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError(f"Local VideoMAE checkpoint is required: {checkpoint}")
            if checkpoint_path.is_dir():
                processor_path = checkpoint_path / "preprocessor_config.json"
                if not processor_path.is_file():
                    raise FileNotFoundError(
                        f"VideoMAE model directory is missing {processor_path.name}"
                    )
                processor = json.loads(processor_path.read_text(encoding="utf-8"))
                if not isinstance(processor, Mapping):
                    raise TypeError("VideoMAE preprocessor_config.json must contain an object")
                from transformers import VideoMAEModel

                backbone = VideoMAEModel.from_pretrained(
                    checkpoint_path, local_files_only=True, output_hidden_states=True
                )
                hidden_size = int(backbone.config.hidden_size)
                preprocess_size = (
                    preprocess_size or processor.get("crop_size") or processor.get("size")
                )
                image_mean = image_mean or processor.get("image_mean")
                image_std = image_std or processor.get("image_std")
            elif checkpoint_path.is_file():
                legacy_checkpoint = True
                backbone = _load_legacy_videomae(checkpoint_path)
                hidden_size = int(backbone.config.hidden_size)
                preprocess_size = preprocess_size or 224
                image_mean = image_mean or _LEGACY_VIDEOMAE_IMAGE_MEAN
                image_std = image_std or _LEGACY_VIDEOMAE_IMAGE_STD
            else:
                raise ValueError(
                    "VideoMAE checkpoint must be a local Transformers directory "
                    "or legacy InternVideo .pth file"
                )
        if hidden_size is None:
            config = getattr(backbone, "config", None)
            hidden_size = int(getattr(config, "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError("VideoMAE hidden_size must be provided")
        if preprocess_size is None:
            preprocess_size = getattr(getattr(backbone, "config", None), "image_size", 224)
        self.preprocess_size = self._parse_size(preprocess_size)
        mean = tuple(float(value) for value in (image_mean or (0.5, 0.5, 0.5)))
        std = tuple(float(value) for value in (image_std or (0.5, 0.5, 0.5)))
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("VideoMAE image_mean/image_std must be three-channel values")
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 1, 3, 1, 1))
        self.input_clamp = bool(input_clamp)
        self.temporal_sampling = str(temporal_sampling).strip().lower()
        if self.temporal_sampling not in {"strict", "uniform"}:
            raise ValueError("VideoMAE temporal_sampling must be strict or uniform")
        self.legacy_preprocessing = legacy_checkpoint
        self.backbone = backbone.requires_grad_(False).eval()
        self.taps = tuple(int(tap) for tap in taps)
        if not self.taps or any(tap < 0 for tap in self.taps):
            raise ValueError("VideoMAE taps must be non-empty non-negative block indices")
        if tuple(sorted(set(self.taps))) != self.taps:
            raise ValueError("VideoMAE taps must be unique and strictly increasing")
        if int(head_kernel_size) <= 0 or int(head_kernel_size) % 2 == 0:
            raise ValueError("VideoMAE head_kernel_size must be a positive odd integer")
        self.diff_augment = VideoDiffAugment(
            probability=float(diff_aug_prob), cutout_ratio=float(diff_aug_cutout)
        )
        self.heads = nn.ModuleList(
            _video_discriminator_head(
                hidden_size,
                kernel_size=int(head_kernel_size),
                norm_type=head_norm_type,
                norm_eps=float(norm_eps),
                use_spectral_norm=bool(spectral_norm_heads),
            )
            for _ in range(len(self.taps) + 1)
        )

    @staticmethod
    def _parse_size(value: object) -> tuple[int, int]:
        if isinstance(value, Mapping):
            if "height" in value and "width" in value:
                result = (int(value["height"]), int(value["width"]))
            elif "shortest_edge" in value:
                edge = int(value["shortest_edge"])
                result = (edge, edge)
            else:
                raise ValueError("Unsupported VideoMAE processor size mapping")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("VideoMAE preprocess_size must have two values")
            result = tuple(int(item) for item in value)
        else:
            edge = int(value)
            result = (edge, edge)
        if any(item <= 0 for item in result):
            raise ValueError("VideoMAE preprocess dimensions must be positive")
        return result

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        if video.dtype == torch.uint8:
            video = video.float().div(255.0)
        elif not torch.is_floating_point(video):
            raise TypeError("VideoMAE input must be uint8 or floating point")
        if not torch.isfinite(video).all():
            raise ValueError("VideoMAE input contains non-finite values")
        if self.input_clamp:
            video = video.clamp(0.0, 1.0)
        else:
            detached = video.detach()
            if float(detached.amin()) < 0.0 or float(detached.amax()) > 1.0:
                raise ValueError("Floating-point VideoMAE input must be in [0,1]")
        if self.training and self.diff_augment.probability > 0.0:
            centered = video.permute(0, 2, 1, 3, 4).contiguous() - 0.5
            video = self.diff_augment(centered).permute(0, 2, 1, 3, 4).contiguous() + 0.5
        batch, time, channels, height, width = video.shape
        if (height, width) != self.preprocess_size:
            frames = video.reshape(batch * time, channels, height, width)
            if self.legacy_preprocessing:
                frames = F.interpolate(
                    frames,
                    size=self.preprocess_size,
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                frames = F.interpolate(
                    frames,
                    size=self.preprocess_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            video = frames.reshape(batch, time, channels, *self.preprocess_size)
        mean = self.image_mean.to(device=video.device, dtype=video.dtype)
        std = self.image_std.to(device=video.device, dtype=video.dtype)
        return (video - mean) / std

    def train(self, mode: bool = True) -> VideoMAEDiscriminator:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("VideoMAE discriminator expects [B,T,3,H,W]")
        expected_frames = getattr(getattr(self.backbone, "config", None), "num_frames", None)
        if expected_frames is not None and video.shape[1] != int(expected_frames):
            expected_frames = int(expected_frames)
            if self.temporal_sampling != "uniform" or video.shape[1] < expected_frames:
                raise ValueError(
                    f"VideoMAE discriminator expects {expected_frames} frames, "
                    f"got {video.shape[1]} with temporal_sampling={self.temporal_sampling}"
                )
            indices = uniform_frame_indices(int(video.shape[1]), expected_frames).to(video.device)
            video = video.index_select(1, indices)
        pixel_values = self._preprocess(video)
        output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("VideoMAE backbone did not return hidden states")
        if not hidden_states:
            raise RuntimeError("VideoMAE backbone returned no embedding features")
        activations = [hidden_states[0]]
        for tap in self.taps:
            if tap + 1 >= len(hidden_states):
                raise ValueError(f"VideoMAE tap {tap} exceeds available blocks")
            activations.append(hidden_states[tap + 1])
        batch = video.shape[0]
        logits = [
            head(activation.transpose(1, 2)).reshape(batch, -1)
            for head, activation in zip(self.heads, activations, strict=True)
        ]
        return torch.cat(logits, dim=1)


class GANController:
    """Lazily constructs all adversarial state at the configured 0-based epoch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        factory: Callable[[], nn.Module],
        *,
        device: torch.device,
    ) -> None:
        self.config = dict(config)
        self.factory = factory
        self.device = device
        self.discriminator: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.constructed_epoch: int | None = None
        self.generator_loss_type = (
            str(self.config.get("gen_loss_type", "hinge")).strip().lower().replace("-", "_")
        )
        self.discriminator_loss_type = (
            str(self.config.get("disc_loss_type", "hinge")).strip().lower().replace("-", "_")
        )
        if self.generator_loss_type not in {"hinge", "hinge_g", "ns_g_loss"}:
            raise ValueError(f"Unsupported GAN generator loss: {self.generator_loss_type}")
        if self.discriminator_loss_type not in {"hinge", "ns_smooth"}:
            raise ValueError(f"Unsupported GAN discriminator loss: {self.discriminator_loss_type}")
        self.discriminator_weight = float(
            self.config.get("discriminator_weight", self.config.get("generator_weight", 1.0))
        )
        if self.discriminator_weight < 0.0:
            raise ValueError("GAN discriminator weight must be non-negative")
        self.lecam_weight = float(self.config.get("lecam_weight", 0.0))
        if self.lecam_weight < 0.0:
            raise ValueError("LeCam weight must be non-negative")
        lecam_enabled = bool(self.config.get("lecam", False))
        self.lecam = (
            LeCamLogitEMA(decay=float(self.config.get("lecam_decay", 0.999)))
            if lecam_enabled
            else None
        )
        self.last_discriminator_loss: torch.Tensor | None = None
        self.last_discriminator_base_loss: torch.Tensor | None = None
        self.last_lecam_loss: torch.Tensor | None = None
        self.last_real_logits_mean: torch.Tensor | None = None
        self.last_real_logits_std: torch.Tensor | None = None
        self.last_fake_logits_mean: torch.Tensor | None = None
        self.last_fake_logits_std: torch.Tensor | None = None
        self.last_logit_gap: torch.Tensor | None = None

    def active(self, epoch: int) -> bool:
        return bool(self.config.get("enabled", False)) and epoch >= min(
            int(self.config.get("generator_start_epoch", 40)),
            int(self.config.get("discriminator_start_epoch", 40)),
        )

    def ensure_initialized(self, epoch: int) -> bool:
        if not self.active(epoch):
            return False
        if self.discriminator is not None:
            return True
        from vrae.training.common.optim import build_optimizer, build_scheduler

        self.discriminator = self.factory().to(self.device)
        self.optimizer = build_optimizer(self.discriminator.parameters(), self.config["optimizer"])
        scheduler_config = self.config.get("scheduler", {"name": "constant", "warmup_steps": 0})
        self.scheduler = build_scheduler(self.optimizer, scheduler_config)
        self.constructed_epoch = int(epoch)
        return True

    def generator_loss(self, reconstructed: torch.Tensor, epoch: int) -> torch.Tensor | None:
        if epoch < int(self.config.get("generator_start_epoch", 40)):
            return None
        self.ensure_initialized(epoch)
        discriminator = getattr(self.discriminator, "module", self.discriminator)
        parameters = list(discriminator.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            return _generator_loss(
                discriminator(reconstructed),
                self.generator_loss_type,
            )
        finally:
            for parameter, required in zip(parameters, requires_grad, strict=True):
                parameter.requires_grad_(required)

    def discriminator_step(
        self,
        real: torch.Tensor,
        reconstructed: torch.Tensor,
        *,
        epoch: int,
        step: int,
        microstep: int = 0,
        accumulation_steps: int = 1,
    ) -> torch.Tensor | None:
        if epoch < int(self.config.get("discriminator_start_epoch", 40)):
            return None
        if step % int(self.config.get("update_interval", 1)):
            return None
        if not 0 <= int(microstep) < int(accumulation_steps):
            raise ValueError("Invalid discriminator accumulation microstep")
        self.ensure_initialized(epoch)
        if int(microstep) == 0:
            self.optimizer.zero_grad(set_to_none=True)
        parameter = next(self.discriminator.parameters())
        discriminator_dtype = parameter.dtype
        final_microstep = int(microstep) + 1 == int(accumulation_steps)
        no_sync = getattr(self.discriminator, "no_sync", None)
        sync_context = no_sync() if not final_microstep and callable(no_sync) else nullcontext()
        with sync_context:
            real_logits = self.discriminator(real.detach().to(dtype=discriminator_dtype))
            fake_logits = self.discriminator(reconstructed.detach().to(dtype=discriminator_dtype))
            base_loss = _discriminator_loss(
                real_logits,
                fake_logits,
                self.discriminator_loss_type,
            )
            lecam_loss = real_logits.new_zeros(())
            if self.lecam is not None:
                self.lecam.update(real_logits, fake_logits)
                if self.lecam.updates > 1:
                    lecam_loss = self.lecam_weight * self.lecam.regularization(
                        real_logits, fake_logits
                    )
            loss = self.discriminator_weight * base_loss + lecam_loss
            real_logits_float = real_logits.detach().float()
            fake_logits_float = fake_logits.detach().float()
            self.last_discriminator_loss = loss.detach()
            self.last_discriminator_base_loss = base_loss.detach()
            self.last_lecam_loss = lecam_loss.detach()
            self.last_real_logits_mean = real_logits_float.mean()
            self.last_real_logits_std = real_logits_float.std(unbiased=False)
            self.last_fake_logits_mean = fake_logits_float.mean()
            self.last_fake_logits_std = fake_logits_float.std(unbiased=False)
            self.last_logit_gap = self.last_real_logits_mean - self.last_fake_logits_mean
            (loss / int(accumulation_steps)).backward()
        if not final_microstep:
            return None
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()

    def monitoring_metrics(self) -> dict[str, torch.Tensor]:
        """Return the most recent discriminator update for periodic logging."""

        values = {
            "discriminator": self.last_discriminator_loss,
            "discriminator_base": self.last_discriminator_base_loss,
            "discriminator_lecam": self.last_lecam_loss,
            "discriminator_real_logits_mean": self.last_real_logits_mean,
            "discriminator_real_logits_std": self.last_real_logits_std,
            "discriminator_fake_logits_mean": self.last_fake_logits_mean,
            "discriminator_fake_logits_std": self.last_fake_logits_std,
            "discriminator_logit_gap": self.last_logit_gap,
        }
        return {name: value for name, value in values.items() if value is not None}

    def loss_state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "generator_loss_type": self.generator_loss_type,
            "discriminator_loss_type": self.discriminator_loss_type,
            "discriminator_weight": self.discriminator_weight,
            "lecam_weight": self.lecam_weight,
            "lecam": self.lecam.state_dict() if self.lecam is not None else None,
        }

    def load_loss_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_version",
            "generator_loss_type",
            "discriminator_loss_type",
            "discriminator_weight",
            "lecam_weight",
            "lecam",
        }
        if set(state) != required:
            raise ValueError(
                f"GAN loss state fields mismatch: expected={sorted(required)} "
                f"actual={sorted(state)}"
            )
        if int(state["format_version"]) != 1:
            raise ValueError("Unsupported GAN loss state format")
        static_fields = {
            "generator_loss_type": self.generator_loss_type,
            "discriminator_loss_type": self.discriminator_loss_type,
            "discriminator_weight": self.discriminator_weight,
            "lecam_weight": self.lecam_weight,
        }
        mismatches = {
            key: (state[key], expected)
            for key, expected in static_fields.items()
            if state[key] != expected
        }
        if mismatches:
            raise ValueError(f"GAN loss configuration changed across resume: {mismatches}")
        lecam_state = state["lecam"]
        if self.lecam is None:
            if lecam_state is not None:
                raise ValueError("Checkpoint has LeCam state but current config disables LeCam")
        else:
            if not isinstance(lecam_state, Mapping):
                raise ValueError("Checkpoint is missing enabled LeCam EMA state")
            self.lecam.load_state_dict(lecam_state)
