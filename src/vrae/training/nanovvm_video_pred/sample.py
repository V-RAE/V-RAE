"""Generate sixteen future frames from one RGB frame.

Inputs, one of:
  --first-frame IMAGE           an image file (PNG/JPEG)
  --video PATH --start-frame S  frame S of a video, decoded exactly
  --split val [--count N]       the condition frames of the configured val manifests

The first frame gets the training spatial transform (short side 256, bilinear,
no antialias, center crop). Its frozen pre-pool features are extracted once,
outside the Euler loop, and both CFG branches reuse them. Each sample starts
from independent Gaussian noise ``[4,h*w,C]`` drawn from (base_seed, seed).

Each output ``.pt`` holds ``predicted_future`` ``[16,3,H,W]`` (decoded V-RAE
output), ``condition_rgb`` ``[3,H,W]``, and ``display17`` ``[17,3,H,W]``, which
is only a visualization (first frame followed by the 16 predicted frames).
Val samples also store ``real_future_rgb`` = RGB clip[1:17] for metrics.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from vrae.checkpoint import compare_metadata, load_checkpoint
from vrae.config import load_config
from vrae.evaluation.common.population import sample_seed
from vrae.models.dit.transport import FlowMatchingTransport
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import build_flow_transport, load_frozen_stage1
from vrae.training.common.latent_norm import LatentNormalizer
from vrae.training.nanovvm_video_pred.data import (
    IMAGE_SIZE,
    PREPROCESSING,
    build_nanovvm_dataset,
    collate_nanovvm,
    decode_clip_frames,
    spatial_preprocess,
)
from vrae.training.nanovvm_video_pred.latents import decode_future_tokens, encode_first_frame


def deterministic_future_noise(
    sample_indices: Sequence[int],
    *,
    base_seed: int,
    latent_shape: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Independent ``[chunks,tokens,channels]`` Gaussian noise per sample index."""

    shape = tuple(int(value) for value in latent_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("latent_shape must be [chunks,tokens,channels]")
    samples = []
    for sample_index in sample_indices:
        generator = torch.Generator(device=device).manual_seed(
            sample_seed(int(base_seed), int(sample_index))
        )
        samples.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    if not samples:
        return torch.empty((0, *shape), device=device, dtype=dtype)
    return torch.stack(samples)


@torch.no_grad()
def predict_future_tokens(
    dit: nn.Module,
    frame_features: torch.Tensor,
    normalizer: LatentNormalizer,
    *,
    sample_indices: Sequence[int],
    base_seed: int,
    steps: int,
    transport: FlowMatchingTransport,
    cfg_scale: float = 1.0,
    internal_guidance_scale: float = 1.0,
) -> torch.Tensor:
    """First-frame features ``[B,h*w,C_f]`` -> raw future latents ``[B,4,h*w,C]``."""

    expected = (dit.frame_tokens, dit.frame_channels)
    if frame_features.ndim != 3 or tuple(frame_features.shape[1:]) != expected:
        raise ValueError(
            f"frame_features must be [B,{expected[0]},{expected[1]}], got {tuple(frame_features.shape)}"
        )
    if len(sample_indices) != frame_features.shape[0]:
        raise ValueError("sample_indices length must match the frame batch")
    device = frame_features.device
    noise = deterministic_future_noise(
        sample_indices,
        base_seed=base_seed,
        latent_shape=dit.future_latent_shape,
        device=device,
    )
    keep = torch.zeros(frame_features.shape[0], dtype=torch.bool, device=device)
    model_kwargs = {"frame_features": frame_features, "condition_drop_mask": keep}
    unconditional_kwargs = None
    if float(cfg_scale) != 1:
        unconditional_kwargs = {
            "frame_features": frame_features,
            "condition_drop_mask": torch.ones_like(keep),
        }
    normalized_future = transport.euler_sample(
        dit,
        noise,
        model_kwargs=model_kwargs,
        unconditional_kwargs=unconditional_kwargs,
        num_steps=int(steps),
        cfg_scale=float(cfg_scale),
        ig_scale=float(internal_guidance_scale),
    )
    return normalizer.denormalize(normalized_future)


@torch.no_grad()
def sample_from_first_frame(
    stage1: Any,
    dit: nn.Module,
    normalizer: LatentNormalizer,
    condition_rgb: torch.Tensor,
    *,
    sample_indices: Sequence[int],
    base_seed: int,
    steps: int,
    transport: FlowMatchingTransport,
    cfg_scale: float = 1.0,
    internal_guidance_scale: float = 1.0,
) -> torch.Tensor:
    """RGB first frames ``[B,3,H,W]`` -> decoded future RGB ``[B,16,3,H,W]``."""

    frame_features = encode_first_frame(stage1, condition_rgb)
    tokens = predict_future_tokens(
        dit,
        frame_features,
        normalizer,
        sample_indices=sample_indices,
        base_seed=base_seed,
        steps=steps,
        transport=transport,
        cfg_scale=cfg_scale,
        internal_guidance_scale=internal_guidance_scale,
    )
    return decode_future_tokens(stage1, tokens, grid_size=tuple(dit.grid_size))


def load_sampling_stack(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    checkpoint: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> tuple[Any, nn.Module, LatentNormalizer]:
    from vrae.training.nanovvm_video_pred.train import (
        build_nanovvm_dit,
        load_task_normalizer,
        nanovvm_model_metadata,
    )

    stage1, _ = load_frozen_stage1(config, paths, device)
    dit = build_nanovvm_dit(config, stage1.metadata()).to(device)
    normalizer = load_task_normalizer(
        config, paths, stage1_metadata=stage1.metadata(), device=device
    )
    payload = load_checkpoint(paths.checkpoint(checkpoint))
    expected_metadata = nanovvm_model_metadata(config, stage1.metadata(), dit, normalizer)
    compare_metadata(expected_metadata, payload["model_metadata"], tuple(expected_metadata))
    dit.load_state_dict(payload["model"], strict=True)
    if use_ema and payload.get("ema") is not None:
        average = ExponentialMovingAverage(dit, decay=float(payload["ema"]["decay"]))
        average.load_state_dict(payload["ema"])
        average.copy_to(dit)
    return stage1, dit.eval(), normalizer


def load_first_frame_image(path: str | Path, *, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        rgb = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
        width, height = image.size
    frame = rgb.reshape(height, width, 3).permute(2, 0, 1)[None].contiguous()
    return spatial_preprocess(frame, image_size, preprocessing=PREPROCESSING)[0]


def load_first_frame_from_video(
    path: str | Path,
    start_frame: int,
    *,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    frames, _ = decode_clip_frames(
        Path(path),
        int(start_frame),
        where=f"frame {start_frame} of {path}",
        rgb_seq_len=1,
        frame_interval=1,
    )
    return spatial_preprocess(frames, image_size, preprocessing=PREPROCESSING)[0]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", value).strip("._") or "sample"


def _save(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save({key: value.detach().cpu() for key, value in tensors.items()}, temporary)
    temporary.replace(path)


def _sampling_settings(config: Mapping[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    sampling = config.get("sampling", {})
    return {
        "base_seed": int(sampling.get("base_seed", 3407)),
        "steps": int(
            arguments.steps if arguments.steps is not None else sampling.get("steps", 100)
        ),
        "cfg_scale": float(
            arguments.cfg_scale
            if arguments.cfg_scale is not None
            else sampling.get("cfg_scale", 1.0)
        ),
        "internal_guidance_scale": float(sampling.get("internal_guidance_scale", 1.0)),
    }


@torch.no_grad()
def run_sampling(arguments: argparse.Namespace) -> list[Path]:
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1, dit, normalizer = load_sampling_stack(
        config, paths, arguments.checkpoint, device, use_ema=not arguments.no_ema
    )
    transport = build_flow_transport(config)
    settings = _sampling_settings(config, arguments)
    output = Path(arguments.output)
    written: list[Path] = []

    if arguments.split is None:
        if arguments.first_frame:
            frame = load_first_frame_image(arguments.first_frame)
            name = Path(arguments.first_frame).stem
        else:
            frame = load_first_frame_from_video(arguments.video, arguments.start_frame)
            name = f"{Path(arguments.video).stem}@{arguments.start_frame}"
        condition = frame[None].to(device)
        predicted = sample_from_first_frame(
            stage1,
            dit,
            normalizer,
            condition,
            sample_indices=[int(arguments.seed)],
            transport=transport,
            **settings,
        )[0]
        unit = condition[0].float() / 255.0
        path = output / f"{_safe_name(name)}_seed{int(arguments.seed)}.pt"
        _save(
            path,
            {
                "condition_rgb": unit,
                "predicted_future": predicted,
                "display17": torch.cat((unit[None], predicted.float().clamp(0, 1)), dim=0),
            },
        )
        return [path]

    dataset = build_nanovvm_dataset(config, paths, split=arguments.split)
    count = len(dataset) if arguments.count is None else min(int(arguments.count), len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(count))),
        batch_size=int(arguments.batch_size),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_nanovvm,
    )
    for batch in loader:
        condition = batch["condition"].to(device)
        predicted = sample_from_first_frame(
            stage1,
            dit,
            normalizer,
            condition,
            sample_indices=[int(value) for value in batch["index"]],
            transport=transport,
            **settings,
        )
        for item, sample_id, real, first, prediction in zip(
            batch["index"].tolist(),
            batch["sample_id"],
            batch["future"],
            batch["condition"],
            predicted.cpu(),
            strict=True,
        ):
            unit = first.float() / 255.0
            path = output / f"{int(item):08d}_{_safe_name(sample_id)}.pt"
            _save(
                path,
                {
                    "condition_rgb": unit,
                    "real_future_rgb": real.float() / 255.0,
                    "predicted_future": prediction,
                    "display17": torch.cat((unit[None], prediction.float().clamp(0, 1)), dim=0),
                },
            )
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample NanoVVM future frames from one RGB frame")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--first-frame", default=None, help="image file used as the observed frame")
    source.add_argument(
        "--video", default=None, help="video whose --start-frame is the observed frame"
    )
    source.add_argument("--split", choices=("train", "val"), default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="noise seed for a single first frame")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--no-ema", action="store_true")
    arguments = parser.parse_args()
    for path in run_sampling(arguments):
        print(path, flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "decode_future_tokens",
    "deterministic_future_noise",
    "load_first_frame_from_video",
    "load_first_frame_image",
    "load_sampling_stack",
    "predict_future_tokens",
    "sample_from_first_frame",
]
