from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    variant: str
    layers: tuple[int, ...]
    fusion: str
    hidden_size: int
    num_blocks: int
    patch_size: int
    encoder_tubelet_size: int
    pixel_normalization: str
    image_size: tuple[int, int] = (256, 256)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "layers": list(self.layers),
            "fusion": self.fusion,
            "hidden_size": self.hidden_size,
            "num_blocks": self.num_blocks,
            "patch_size": self.patch_size,
            "encoder_tubelet_size": self.encoder_tubelet_size,
            "pixel_normalization": self.pixel_normalization,
            "image_size": list(self.image_size),
        }


def validate_encoder_config(config: Mapping[str, Any], spec: EncoderSpec) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("encoder config must be a mapping")
    expected = spec.metadata()
    actual: dict[str, Any] = {}
    for key in expected:
        value = config.get(key)
        if (
            key in {"layers", "image_size"}
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            value = list(value)
        actual[key] = value
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"Invalid formal {spec.name} encoder config: {mismatches}")
    invalid_height = spec.image_size[0] % spec.patch_size
    invalid_width = spec.image_size[1] % spec.patch_size
    if invalid_height or invalid_width:
        raise ValueError("encoder image_size must be divisible by patch_size")
    checkpoint = config.get("checkpoint")
    if checkpoint is None or not str(checkpoint).strip():
        raise ValueError("encoder.checkpoint must be a non-empty local path")


def resolve_checkpoint_path(
    config: Mapping[str, Any],
    *,
    paths: Any | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    configured = checkpoint_path if checkpoint_path is not None else config.get("checkpoint")
    if configured is None or not str(configured).strip():
        raise ValueError("encoder.checkpoint must name an explicit local checkpoint")
    if checkpoint_path is None and paths is not None:
        return Path(paths.checkpoint(configured, require_exists=True))
    path = Path(configured).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_source_path(
    name: str,
    *,
    paths: Any | None = None,
    source_dir: str | Path | None = None,
) -> Path:
    if source_dir is not None:
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        return source
    if paths is None:
        raise ValueError(
            f"{name} requires an explicit local source_dir or ProjectPaths.third_party entry"
        )
    return Path(paths.source(name, require_exists=True)).resolve()


def load_torch_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint did not contain a mapping: {path}")
    return value


def normalize_state_dict(
    state: Mapping[str, Any], *, prefixes: Sequence[str] = ("module.", "backbone.")
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value
    return normalized


def inferred_patch_size(backbone: nn.Module) -> int | None:
    raw = getattr(backbone, "patch_size", None)
    if raw is not None:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(int(value) for value in raw)
            if len(values) >= 2 and values[-1] == values[-2]:
                return values[-1]
        else:
            return int(raw)

    patch_embed = getattr(backbone, "patch_embed", None)
    raw = getattr(patch_embed, "patch_size", None)
    if raw is not None:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(int(value) for value in raw)
            if len(values) >= 2 and values[-1] == values[-2]:
                return values[-1]
        else:
            return int(raw)
    projection = getattr(patch_embed, "proj", None)
    kernel = getattr(projection, "kernel_size", None)
    if kernel is not None:
        values = (int(kernel), int(kernel)) if isinstance(kernel, int) else tuple(kernel)
        if len(values) == 2 and values[0] == values[1]:
            return int(values[0])
    return None


class EncoderAdapter(nn.Module, ABC):
    SPEC: EncoderSpec
    # True only for backbones whose ``_encode_preprocessed`` encodes every frame
    # independently (image ViTs with tubelet 1). Those can run one observed
    # frame through ``encode_images`` without padding it to a four-frame clip.
    PER_FRAME_ENCODING: bool = False

    def __init__(
        self,
        config: Mapping[str, Any],
        backbone: nn.Module,
        *,
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        runtime_size = config.get("runtime_image_size", self.SPEC.image_size)
        if isinstance(runtime_size, int):
            runtime_size = (runtime_size, runtime_size)
        if (
            not isinstance(runtime_size, Sequence)
            or isinstance(runtime_size, (str, bytes))
            or len(runtime_size) != 2
        ):
            raise ValueError("runtime_image_size must be an integer or [height,width]")
        # Preserve the task-declared geometry in checkpoint metadata. Forward
        # geometry is derived independently from every input tensor.
        self.runtime_image_size = tuple(int(value) for value in runtime_size)
        if any(value <= 0 or value % self.SPEC.patch_size for value in self.runtime_image_size):
            raise ValueError("runtime image dimensions must be positive multiples of patch_size")
        self._validate_config(config)
        self.backbone = backbone
        mean = self._normalization_values(normalization_mean, name="mean")
        std = self._normalization_values(normalization_std, name="std")
        if any(value <= 0.0 for value in std):
            raise ValueError("normalization std values must be positive")
        self.register_buffer(
            "_normalization_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_normalization_std",
            torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._configure_backbone()
        self._validate_backbone()
        self._compiled_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.requires_grad_(False)
        self.train(False)

    @staticmethod
    def _normalization_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)) or len(values) != 3:
            raise ValueError(f"normalization {name} must contain three values")
        parsed = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"normalization {name} values must be finite")
        return parsed

    @property
    def hidden_size(self) -> int:
        return self.SPEC.hidden_size

    @property
    def patch_size(self) -> int:
        return self.SPEC.patch_size

    @property
    def encoder_tubelet_size(self) -> int:
        return self.SPEC.encoder_tubelet_size

    @property
    def grid_size(self) -> tuple[int, int]:
        height, width = self.runtime_image_size
        return height // self.patch_size, width // self.patch_size

    @property
    def num_patches(self) -> int:
        height, width = self.grid_size
        return height * width

    def metadata(self) -> dict[str, Any]:
        return {
            **self.SPEC.metadata(),
            "runtime_image_size": list(self.runtime_image_size),
            "runtime_grid_size": list(self.grid_size),
        }

    def train(self, mode: bool = True) -> EncoderAdapter:
        super().train(False)
        self.backbone.eval()
        return self

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        validate_encoder_config(config, self.SPEC)

    def _configure_backbone(self) -> None:
        pass

    def _validate_backbone(self) -> None:
        hidden = getattr(self.backbone, "embed_dim", None)
        if hidden is None:
            backbone_config = getattr(self.backbone, "config", None)
            hidden = getattr(backbone_config, "hidden_size", None)
        if hidden is not None and int(hidden) != self.SPEC.hidden_size:
            raise ValueError(
                f"{self.SPEC.name} backbone hidden_size={hidden}, expected {self.SPEC.hidden_size}"
            )

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None and len(blocks) != self.SPEC.num_blocks:
            raise ValueError(
                f"{self.SPEC.name} backbone has {len(blocks)} blocks, "
                f"expected {self.SPEC.num_blocks}"
            )

        patch_size = inferred_patch_size(self.backbone)
        if patch_size is not None and patch_size != self.SPEC.patch_size:
            raise ValueError(
                f"{self.SPEC.name} backbone patch_size={patch_size}, "
                f"expected {self.SPEC.patch_size}"
            )

    def _validate_video(self, video: torch.Tensor) -> torch.Tensor:
        return self._validate_pixels(video, require_formal_time=True)

    def _validate_pixels(self, video: torch.Tensor, *, require_formal_time: bool) -> torch.Tensor:
        if not torch.is_tensor(video):
            raise TypeError("video must be a torch.Tensor")
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {tuple(video.shape)}")
        batch, time, channels, height, width = (int(value) for value in video.shape)
        if batch <= 0 or time <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"video dimensions must be positive, got {tuple(video.shape)}")
        if channels != 3:
            raise ValueError(f"Expected three RGB channels, got {channels}")
        if require_formal_time and time % 4:
            raise ValueError(f"Formal V-RAE input time={time} must be divisible by 4")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "height and width must both be divisible by "
                f"patch_size={self.patch_size}, got {height}x{width}"
            )

        if video.dtype == torch.uint8:
            return video.to(dtype=torch.float32).div(255.0)
        if not torch.is_floating_point(video):
            raise TypeError("video must be uint8 or floating-point RGB in [0,1]")

        result = video.to(dtype=torch.float32)
        # CUDA inputs have already passed unit-range validation in the CPU data
        # pipeline.  Calling .item() here would introduce several device-to-host
        # synchronizations in every training step.
        if result.device.type != "cpu":
            return result

        minimum_tensor, maximum_tensor = torch.aminmax(result)
        minimum = float(minimum_tensor.item())
        maximum = float(maximum_tensor.item())
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("video contains non-finite RGB values")
        tolerance = 1.0e-6
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            raise ValueError(
                f"Floating-point RGB must be in [0,1], got min={minimum} max={maximum}"
            )
        if minimum < 0.0 or maximum > 1.0:
            result = result.clamp(0.0, 1.0)
        return result

    def _normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self._normalization_mean) / self._normalization_std

    def _tokens_to_frame_grid(
        self,
        tokens: torch.Tensor,
        *,
        batch: int,
        time: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        grid_h, grid_w = grid_size
        expected = (batch * time, grid_h * grid_w, self.hidden_size)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"Unexpected {self.SPEC.name} patch token shape: got={tuple(tokens.shape)} "
                f"expected={expected}"
            )
        return (
            tokens.reshape(batch, time, grid_h, grid_w, self.hidden_size)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

    def _validate_output(
        self,
        output: torch.Tensor,
        *,
        batch: int,
        time: int,
        grid_size: tuple[int, int],
    ) -> None:
        grid_h, grid_w = grid_size
        expected = (
            batch,
            time // self.encoder_tubelet_size,
            self.hidden_size,
            grid_h,
            grid_w,
        )
        if not torch.is_tensor(output) or tuple(output.shape) != expected:
            actual = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
            raise RuntimeError(
                f"Unexpected {self.SPEC.name} adapter output: got={actual} expected={expected}"
            )
        if not torch.is_floating_point(output):
            raise RuntimeError(f"{self.SPEC.name} adapter output must be floating-point")
        # Keep strict validation for CPU/test execution without synchronizing
        # CUDA back to the host on every encoder forward.
        if output.device.type == "cpu" and not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"{self.SPEC.name} adapter output contains non-finite values")

    @abstractmethod
    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _encode_preprocessed(
        self, video: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        raise NotImplementedError

    def enable_compile(self, **kwargs: Any) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        self._compiled_forward = torch.compile(self._forward_impl, **kwargs)

    def _forward_impl(self, video: torch.Tensor) -> torch.Tensor:
        video = self._validate_video(video)
        batch, time, _, height, width = (int(value) for value in video.shape)
        grid_size = (height // self.patch_size, width // self.patch_size)
        self.backbone.eval()
        output = self._encode_preprocessed(self._preprocess(video), grid_size=grid_size)
        self._validate_output(output, batch=batch, time=time, grid_size=grid_size)
        return output

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return (
            self._compiled_forward(video)
            if self._compiled_forward is not None
            else self._forward_impl(video)
        )

    @property
    def supports_single_image(self) -> bool:
        return bool(self.PER_FRAME_ENCODING) and self.encoder_tubelet_size == 1

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode independent RGB images ``[B,3,H,W]`` to pre-pool features ``[B,C,h,w]``.

        This is the video path with ``T=1``: the same pixel validation and
        normalization, the same fused multi-layer features, and the same
        ``[C,h,w]`` grid order as one frame of ``forward``. Only the formal
        four-frame requirement is skipped, so it is limited to per-frame backbones.
        """

        if not self.supports_single_image:
            raise NotImplementedError(
                f"{self.SPEC.name} (encoder_tubelet_size={self.encoder_tubelet_size}) does not "
                "encode frames independently and has no single-image feature path"
            )
        if not torch.is_tensor(images) or images.ndim != 4:
            shape = tuple(images.shape) if torch.is_tensor(images) else type(images).__name__
            raise ValueError(f"Expected RGB images [B,3,H,W], got {shape}")
        video = self._validate_pixels(images[:, None], require_formal_time=False)
        batch, time, _, height, width = (int(value) for value in video.shape)
        grid_size = (height // self.patch_size, width // self.patch_size)
        self.backbone.eval()
        output = self._encode_preprocessed(self._preprocess(video), grid_size=grid_size)
        self._validate_output(output, batch=batch, time=time, grid_size=grid_size)
        return output[:, 0]
