from __future__ import annotations

from collections.abc import Mapping
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
    resolve_checkpoint_path,
    resolve_source_path,
    validate_encoder_config,
)
from vrae.registry import ENCODERS

EUPE_SPEC = EncoderSpec(
    name="eupe",
    variant="vit_b16",
    layers=(5, 6, 7, 8, 9, 10, 11),
    fusion="sum",
    hidden_size=768,
    num_blocks=12,
    patch_size=16,
    encoder_tubelet_size=1,
    pixel_normalization="imagenet",
)


def load_local_eupe(*, source_dir: Path, checkpoint_path: Path) -> nn.Module:
    hubconf = source_dir / "hubconf.py"
    if not hubconf.is_file():
        raise FileNotFoundError(f"EUPE local source is missing {hubconf}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"EUPE checkpoint is not a file: {checkpoint_path}")
    backbone = torch.hub.load(
        str(source_dir),
        "eupe_vitb16",
        source="local",
        trust_repo=True,
        skip_validation=True,
        pretrained=False,
    )
    state: Mapping[str, Any] = load_torch_mapping(checkpoint_path)
    nested = state.get("state_dict")
    if isinstance(nested, Mapping):
        state = nested
    tensor_state = {str(key): value for key, value in state.items() if torch.is_tensor(value)}
    if not tensor_state:
        raise TypeError(f"EUPE checkpoint contains no tensor state dict: {checkpoint_path}")
    result = backbone.load_state_dict(tensor_state, strict=False)
    disallowed = sorted(key for key in result.unexpected_keys if not key.startswith("projectors."))
    if result.missing_keys or disallowed:
        raise RuntimeError(
            f"EUPE checkpoint mismatch: missing={list(result.missing_keys)} unexpected={disallowed}"
        )
    return backbone


@ENCODERS.decorator("eupe")
class EUPEAdapter(EncoderAdapter):
    SPEC = EUPE_SPEC
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
            source = resolve_source_path("eupe", paths=paths, source_dir=source_dir)
            backbone = load_local_eupe(source_dir=source, checkpoint_path=checkpoint)
        super().__init__(
            config=config,
            backbone=backbone,
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

    def _configure_backbone(self) -> None:
        if not hasattr(self.backbone, "norm"):
            raise TypeError("EUPE backbone must expose its final patch-token norm")
        self.backbone.norm = nn.LayerNorm(
            self.hidden_size,
            eps=float(getattr(self.backbone.norm, "eps", 1e-5)),
            elementwise_affine=False,
        )

    def _validate_backbone(self) -> None:
        super()._validate_backbone()
        if not callable(getattr(self.backbone, "get_intermediate_layers", None)):
            raise TypeError("EUPE backbone must implement get_intermediate_layers")
        if len(getattr(self.backbone, "blocks", ())) != self.SPEC.num_blocks:
            raise ValueError("EUPE ViT-B backbone must contain exactly 12 blocks")

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
            raise RuntimeError(f"Collected {count}/{len(self.SPEC.layers)} EUPE K7 outputs")
        patch_outputs: list[torch.Tensor] = []
        for index, output in zip(self.SPEC.layers, outputs, strict=True):
            if not torch.is_tensor(output) or output.ndim != 3:
                actual = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
                raise RuntimeError(
                    f"EUPE block {index} did not return [BT,N,C] patch tokens: {actual}"
                )
            patch_outputs.append(output)
        fused = torch.stack(patch_outputs, dim=0).sum(dim=0)
        return self._tokens_to_frame_grid(
            fused,
            batch=int(batch),
            time=int(time),
            grid_size=grid_size,
        )
