from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vrae.models.encoders.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderAdapter,
    EncoderSpec,
    load_torch_mapping,
    normalize_state_dict,
    resolve_checkpoint_path,
    resolve_source_path,
    validate_encoder_config,
)
from vrae.registry import ENCODERS

DINOV3_SPEC = EncoderSpec(
    name="dinov3",
    variant="vit_l16",
    layers=(11, 13, 15, 17, 19, 21, 23),
    fusion="mean_plus_final_spatial_mean",
    hidden_size=1024,
    num_blocks=24,
    patch_size=16,
    encoder_tubelet_size=1,
    pixel_normalization="imagenet",
)


@contextmanager
def _local_source_import_path(source_dir: Path) -> Iterator[None]:
    source_text = str(source_dir)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(source_text)


def _import_local_backbones(source_dir: Path) -> Any:
    module_path = source_dir / "dinov3" / "hub" / "backbones.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"DINOv3 local source is missing {module_path}")
    module_name = "dinov3.hub.backbones"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        try:
            loaded_path.relative_to(source_dir)
        except ValueError as error:
            raise RuntimeError(
                "DINOv3 backbones are already imported from a different checkout: "
                f"loaded={loaded_path} requested={source_dir}"
            ) from error
        return loaded
    with _local_source_import_path(source_dir):
        module = importlib.import_module(module_name)
    loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        loaded_path.relative_to(source_dir)
    except ValueError as error:
        raise RuntimeError(
            f"Imported DINOv3 backbones from {loaded_path}, expected source under {source_dir}"
        ) from error
    return module


def load_local_dinov3(*, source_dir: Path, checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint is not a file: {checkpoint_path}")
    backbones = _import_local_backbones(source_dir)
    backbone = backbones.dinov3_vitl16(pretrained=False)
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                "Loading DINOv3 safetensors requires the optional encoders dependency"
            ) from error
        raw_state: Mapping[str, Any] = load_file(str(checkpoint_path), device="cpu")
    else:
        raw_state = load_torch_mapping(checkpoint_path)
    for key in ("state_dict", "model"):
        nested = raw_state.get(key)
        if isinstance(nested, Mapping):
            raw_state = nested
            break
    state = normalize_state_dict(raw_state)
    if not state:
        raise TypeError(f"DINOv3 checkpoint contains no tensor state dict: {checkpoint_path}")
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Strict DINOv3 load returned incompatibilities: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    return backbone


@ENCODERS.decorator("dinov3")
class DINOv3Adapter(EncoderAdapter):
    SPEC = DINOV3_SPEC
    PER_FRAME_ENCODING = True

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        paths: Any | None = None,
        source_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        if backbone is None:
            validate_encoder_config(config, self.SPEC)
            checkpoint = resolve_checkpoint_path(
                config, paths=paths, checkpoint_path=checkpoint_path
            )
            source = resolve_source_path("dinov3", paths=paths, source_dir=source_dir)
            backbone = load_local_dinov3(source_dir=source, checkpoint_path=checkpoint)
        super().__init__(
            config=config,
            backbone=backbone,
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

    def _configure_backbone(self) -> None:
        if not hasattr(self.backbone, "norm"):
            raise TypeError("DINOv3 backbone must expose its patch-token output norm")
        self.backbone.norm = nn.LayerNorm(
            self.hidden_size,
            eps=float(getattr(self.backbone.norm, "eps", 1e-5)),
            elementwise_affine=False,
        )

    def _validate_backbone(self) -> None:
        super()._validate_backbone()
        if not callable(getattr(self.backbone, "get_intermediate_layers", None)):
            raise TypeError("DINOv3 backbone must implement get_intermediate_layers")
        if len(getattr(self.backbone, "blocks", ())) != self.SPEC.num_blocks:
            raise ValueError("DINOv3 ViT-L backbone must contain exactly 24 blocks")

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        return self._normalize_video(video)

    def _encode_preprocessed(
        self, video: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        batch, time, channels, height, width = video.shape
        frames = video.reshape(batch * time, channels, height, width)
        outputs = self.backbone.get_intermediate_layers(
            frames,
            n=list(self.SPEC.layers),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.SPEC.layers):
            count = len(outputs) if isinstance(outputs, (tuple, list)) else 0
            raise RuntimeError(f"Collected {count}/{len(self.SPEC.layers)} DINOv3 K7 outputs")
        patch_outputs: list[torch.Tensor] = []
        for index, output in zip(self.SPEC.layers, outputs, strict=True):
            if not torch.is_tensor(output) or output.ndim != 3:
                actual = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
                raise RuntimeError(
                    f"DINOv3 block {index} did not return [BT,N,C] patch tokens: {actual}"
                )
            patch_outputs.append(output)
        fused = torch.stack(patch_outputs, dim=0).mean(dim=0)
        fused = fused + patch_outputs[-1].mean(dim=1, keepdim=True)
        return self._tokens_to_frame_grid(
            fused,
            batch=int(batch),
            time=int(time),
            grid_size=grid_size,
        )
