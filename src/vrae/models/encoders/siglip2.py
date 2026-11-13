from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vrae.models.encoders.base import (
    EncoderAdapter,
    EncoderSpec,
    resolve_checkpoint_path,
    validate_encoder_config,
)
from vrae.registry import ENCODERS

SIGLIP2_SPEC = EncoderSpec(
    name="siglip2",
    variant="vit_l16_256",
    layers=(11, 13, 15, 17, 19, 21, 23),
    fusion="mean_plus_final_spatial_mean",
    hidden_size=1024,
    num_blocks=24,
    patch_size=16,
    encoder_tubelet_size=1,
    pixel_normalization="siglip2_native_0p5",
)


_CANONICAL_INJECTED_PROCESSOR_CONFIG = {
    "do_resize": True,
    "do_rescale": True,
    "rescale_factor": 1.0 / 255.0,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "size": {"height": 256, "width": 256},
    "resample": 2,
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _validate_processor_config(
    processor: Mapping[str, Any], *, origin: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if processor.get("do_resize") is not True:
        raise RuntimeError(f"SigLIP2 processor must enable native resizing: {origin}")
    if processor.get("do_rescale") is not True:
        raise RuntimeError(f"SigLIP2 processor must enable rescaling: {origin}")
    factor = float(processor.get("rescale_factor", float("nan")))
    if not math.isclose(factor, 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"SigLIP2 processor rescale_factor must be 1/255: {origin}")
    if processor.get("do_normalize") is not True:
        raise RuntimeError(f"SigLIP2 processor must enable normalization: {origin}")

    def rgb_values(key: str) -> tuple[float, float, float]:
        values = processor.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
            raise RuntimeError(f"SigLIP2 processor {key} must contain three values: {origin}")
        parsed = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in parsed):
            raise RuntimeError(f"SigLIP2 processor {key} must be finite: {origin}")
        return parsed

    image_mean = rgb_values("image_mean")
    image_std = rgb_values("image_std")
    expected = (0.5, 0.5, 0.5)
    if image_mean != expected or image_std != expected:
        raise RuntimeError(f"SigLIP2 processor mean/std must both be {expected}: {origin}")
    size = processor.get("size")
    if (
        not isinstance(size, Mapping)
        or (
            int(size.get("height", -1)),
            int(size.get("width", -1)),
        )
        != SIGLIP2_SPEC.image_size
    ):
        raise RuntimeError(f"SigLIP2 processor size must be 256x256: {origin}")
    if int(processor.get("resample", -1)) != 2:
        raise RuntimeError(f"SigLIP2 processor must use native PIL bilinear resample=2: {origin}")
    return image_mean, image_std


def _validate_local_model(model_dir: Path) -> Mapping[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SigLIP2 checkpoint directory not found: {model_dir}")
    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete local SigLIP2 checkpoint at {model_dir}; missing={missing}"
        )
    config = _read_json(model_dir / "config.json")
    vision = config.get("vision_config")
    if not isinstance(vision, Mapping):
        raise RuntimeError(f"SigLIP2 config lacks vision_config: {model_dir / 'config.json'}")
    actual = (
        int(vision.get("hidden_size", -1)),
        int(vision.get("num_hidden_layers", -1)),
        int(vision.get("patch_size", SIGLIP2_SPEC.patch_size)),
        int(vision.get("image_size", -1)),
    )
    expected = (
        SIGLIP2_SPEC.hidden_size,
        SIGLIP2_SPEC.num_blocks,
        SIGLIP2_SPEC.patch_size,
        SIGLIP2_SPEC.image_size[0],
    )
    if actual != expected:
        raise RuntimeError(
            f"SigLIP2 checkpoint architecture mismatch: got={actual} expected={expected}"
        )
    return _read_json(model_dir / "preprocessor_config.json")


def load_local_siglip2(model_dir: Path) -> nn.Module:
    _validate_local_model(model_dir)
    try:
        from transformers import SiglipVisionModel
        from transformers.utils import logging as transformers_logging
    except ImportError as error:
        raise ImportError(
            "SigLIP2 requires the optional encoders dependency: transformers==5.13.0"
        ) from error
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        loaded = SiglipVisionModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], Mapping):
        raise RuntimeError("SigLIP2 loader did not return strict loading diagnostics")
    wrapper, loading_info = loaded
    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    errors = list(loading_info.get("error_msgs", ()))
    unexpected = [
        key
        for key in loading_info.get("unexpected_keys", ())
        if not (str(key).startswith("text_model.") or str(key) in {"logit_bias", "logit_scale"})
    ]
    if missing or mismatched or errors or unexpected:
        raise RuntimeError(
            "Strict SigLIP2 vision load failed: "
            f"missing={missing} mismatched={mismatched} "
            f"unexpected={unexpected} errors={errors}"
        )
    return getattr(wrapper, "vision_model", wrapper)


@ENCODERS.decorator("siglip2")
class SigLIP2Adapter(EncoderAdapter):
    SPEC = SIGLIP2_SPEC
    PER_FRAME_ENCODING = True

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        paths: Any | None = None,
        checkpoint_path: str | Path | None = None,
        backbone: nn.Module | None = None,
        processor_config: Mapping[str, Any] | None = None,
    ) -> None:
        if backbone is None and processor_config is not None:
            raise ValueError("processor_config injection is only valid with an injected backbone")

        if backbone is None:
            validate_encoder_config(config, self.SPEC)
            model_dir = resolve_checkpoint_path(
                config, paths=paths, checkpoint_path=checkpoint_path
            )
            processor = _validate_local_model(model_dir)
            backbone = load_local_siglip2(model_dir)
            origin = str(model_dir / "preprocessor_config.json")
        else:
            processor = processor_config or _CANONICAL_INJECTED_PROCESSOR_CONFIG
            origin = "injected test processor config"
        self.image_mean, self.image_std = _validate_processor_config(processor, origin=origin)
        super().__init__(
            config=config,
            backbone=backbone,
            normalization_mean=self.image_mean,
            normalization_std=self.image_std,
        )

    def _validate_backbone(self) -> None:
        super()._validate_backbone()
        backbone_config = getattr(self.backbone, "config", None)
        num_layers = getattr(backbone_config, "num_hidden_layers", None)
        patch_size = getattr(backbone_config, "patch_size", None)
        image_size = getattr(backbone_config, "image_size", None)
        if num_layers is not None and int(num_layers) != self.SPEC.num_blocks:
            raise ValueError("SigLIP2-L backbone must contain exactly 24 blocks")
        if patch_size is not None and int(patch_size) != self.SPEC.patch_size:
            raise ValueError("SigLIP2-L backbone must use patch size 16")
        if image_size is not None and int(image_size) != self.SPEC.image_size[0]:
            raise ValueError("SigLIP2-L backbone must use its native 256px geometry")
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is not None and len(layers) != self.SPEC.num_blocks:
            raise ValueError("SigLIP2-L backbone must contain exactly 24 encoder layers")
        if not callable(getattr(self.backbone, "forward", None)):
            raise TypeError("SigLIP2 backbone must implement forward")
        if not isinstance(getattr(self.backbone, "post_layernorm", None), nn.Module):
            raise TypeError("SigLIP2 backbone must expose post_layernorm")

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        return self._normalize_video(video)

    def _encode_preprocessed(
        self, video: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        batch, time, channels, height, width = video.shape
        frames = video.reshape(batch * time, channels, height, width)
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)
        embeddings = getattr(self.backbone, "embeddings", None)
        if isinstance(embeddings, nn.Module) and isinstance(layers, nn.ModuleList):
            hidden = embeddings(
                frames,
                interpolate_pos_encoding=(int(height), int(width)) != self.SPEC.image_size,
            )
            selected: list[torch.Tensor] = []
            selected_indices = set(self.SPEC.layers)
            final_block = self.SPEC.num_blocks - 1
            for block_index, layer in enumerate(layers):
                hidden = layer(hidden, attention_mask=None)
                if block_index in selected_indices:
                    selected.append(
                        hidden
                        if block_index == final_block
                        else self.backbone.post_layernorm(hidden)
                    )
            if len(selected) != len(self.SPEC.layers):
                raise RuntimeError(
                    f"SigLIP2 selected {len(selected)} hidden states; "
                    f"expected {len(self.SPEC.layers)}"
                )
            return self._fuse_selected(
                selected,
                batch=int(batch),
                time=int(time),
                grid_size=grid_size,
            )

        outputs = self.backbone(
            pixel_values=frames,
            interpolate_pos_encoding=(int(height), int(width)) != self.SPEC.image_size,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)):
            raise RuntimeError("SigLIP2 forward did not return hidden_states")
        if len(hidden_states) != self.SPEC.num_blocks + 1:
            raise RuntimeError(f"SigLIP2 returned {len(hidden_states)} hidden states; expected 25")

        selected: list[torch.Tensor] = []
        final_block = self.SPEC.num_blocks - 1
        for block_index in self.SPEC.layers:
            raw = hidden_states[block_index + 1]
            if not torch.is_tensor(raw) or raw.ndim != 3:
                actual = tuple(raw.shape) if torch.is_tensor(raw) else type(raw).__name__
                raise RuntimeError(
                    f"SigLIP2 hidden_states[{block_index + 1}] is not [BT,N,C]: {actual}"
                )
            normalized = raw if block_index == final_block else self.backbone.post_layernorm(raw)
            selected.append(normalized)

        return self._fuse_selected(
            selected,
            batch=int(batch),
            time=int(time),
            grid_size=grid_size,
        )

    def _fuse_selected(
        self,
        selected: list[torch.Tensor],
        *,
        batch: int,
        time: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        fused = torch.stack(selected, dim=0).mean(dim=0)
        fused = fused + selected[-1].mean(dim=1, keepdim=True)
        return self._tokens_to_frame_grid(
            fused,
            batch=batch,
            time=time,
            grid_size=grid_size,
        )
