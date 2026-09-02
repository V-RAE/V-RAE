#!/usr/bin/env python3
"""Sample a trained class-conditional VideoDiT checkpoint.

Typical usage after training finishes::

    python sampling.py

Use ``--config`` when the checkpoint was trained with another configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/generation/vjepa2_1_libero_videodit_high_util_fixed.yaml"


def parse_labels(
    value: str | None,
    *,
    num_samples: int,
    num_classes: int,
    default_labels: list[int] | None = None,
) -> list[int]:
    if value is None or not value.strip():
        labels = default_labels or list(range(num_classes))
        return [labels[index % len(labels)] for index in range(num_samples)]
    labels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one integer")
    if any(label < 0 or label >= num_classes for label in labels):
        raise ValueError(f"--labels must be in [0, {num_classes})")
    return [labels[index % len(labels)] for index in range(num_samples)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample generated videos from a trained VideoDiT checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"training config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint path; defaults to the config run's checkpoints/latest.pt",
    )
    parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="optional project path mapping, matching the training entry point",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to the checkpoint run's samples directory",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated class ids, repeated cyclically to num-samples",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--internal-guidance-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    return parser.parse_args()


def resolve_path(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_checkpoint(
    checkpoint: Path | None,
    *,
    config: dict[str, Any],
    paths: Any,
) -> Path:
    if checkpoint is not None:
        return paths.checkpoint(checkpoint)
    run = paths.training_run(str(config["task"]), str(config["run_name"]), create=False)
    return run / "checkpoints" / "latest.pt"


def save_outputs(
    sampled_video: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_dir: Path,
    fps: int,
    checkpoint: Path,
    config_path: Path,
) -> None:
    from vrae.training.common.visualization import save_sample_tensors, save_video_mp4

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "samples.pt"
    save_sample_tensors(
        {
            "video": sampled_video,
            "labels": labels,
            "sample_ids": torch.arange(sampled_video.shape[0]),
        },
        tensor_path,
    )

    if sampled_video.ndim == 6:
        # [B,T,V,C,H,W] -> one [T,C,H,W] video per sample/view.
        for sample_index in range(sampled_video.shape[0]):
            for view_index in range(sampled_video.shape[2]):
                save_video_mp4(
                    sampled_video[sample_index, :, view_index].detach().cpu(),
                    output_dir / f"sample-{sample_index:04d}-view-{view_index:02d}.mp4",
                    fps=fps,
                )
    elif sampled_video.ndim == 5:
        # [B,T,C,H,W] -> one video per sample.
        for sample_index in range(sampled_video.shape[0]):
            save_video_mp4(
                sampled_video[sample_index].detach().cpu(),
                output_dir / f"sample-{sample_index:04d}.mp4",
                fps=fps,
            )
    else:
        raise ValueError(
            "generated video must have shape [B,T,C,H,W] or [B,T,V,C,H,W], "
            f"got {tuple(sampled_video.shape)}"
        )

    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "shape": list(sampled_video.shape),
        "labels": labels.tolist(),
        "fps": int(fps),
    }
    (output_dir / "sampling_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"[sampling] saved tensors: {tensor_path}", flush=True)
    print(f"[sampling] saved videos/metadata: {output_dir}", flush=True)


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default sampling device")

    from vrae.config import load_config
    from vrae.paths import find_project_root, load_project_paths
    from vrae.training.common.engine import (
        build_flow_transport,
        generate_class_conditional,
        load_class_conditional_sampling_stack,
    )

    config_path = resolve_path(args.config, root=PROJECT_ROOT)
    config = load_config(config_path)
    project_root = find_project_root(config_path)
    paths = load_project_paths(
        config,
        override=args.paths,
        project_root=project_root,
    )
    checkpoint = resolve_checkpoint(args.checkpoint, config=config, paths=paths)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}; pass --checkpoint explicitly "
            "or train the configured run first"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[sampling] config={config_path}", flush=True)
    print(f"[sampling] checkpoint={checkpoint}", flush=True)
    print(f"[sampling] device={device}", flush=True)

    stage1, dit, normalizer = load_class_conditional_sampling_stack(
        config,
        paths,
        checkpoint,
        device,
    )
    num_samples = int(args.num_samples)
    sampling_config = config.get("sampling", {})
    labels = torch.tensor(
        parse_labels(
            args.labels,
            num_samples=num_samples,
            num_classes=int(dit.num_classes),
            default_labels=[int(value) for value in sampling_config.get("class_ids", [])],
        ),
        dtype=torch.long,
    )
    seed = int(
        args.seed
        if args.seed is not None
        else sampling_config.get("base_seed", 3407)
    )
    steps = int(args.steps if args.steps is not None else sampling_config.get("steps", 100))
    cfg_scale = float(
        args.cfg_scale
        if args.cfg_scale is not None
        else sampling_config.get("cfg_scale", 1.0)
    )
    internal_guidance_scale = float(
        args.internal_guidance_scale
        if args.internal_guidance_scale is not None
        else sampling_config.get("internal_guidance_scale", 1.0)
    )
    num_chunks = int(config["data"]["num_frames"]) // 4
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        sampled_video = generate_class_conditional(
            dit,
            stage1,
            normalizer,
            transport=build_flow_transport(config),
            sample_ids=list(range(num_samples)),
            labels=labels,
            base_seed=seed,
            num_chunks=num_chunks,
            grid_size=tuple(dit.grid_size),
            channels=int(dit.in_channels),
            steps=steps,
            cfg_scale=cfg_scale,
            internal_guidance_scale=internal_guidance_scale,
            internal_guidance_t_min=float(
                sampling_config.get("internal_guidance_t_min", 0.0)
            ),
            internal_guidance_t_max=float(
                sampling_config.get("internal_guidance_t_max", 1.0)
            ),
            device=device,
        )

    run_dir = checkpoint.parent.parent
    output_dir = (
        resolve_path(args.output_dir, root=PROJECT_ROOT)
        if args.output_dir is not None
        else run_dir / "samples"
    )
    save_outputs(
        sampled_video,
        labels,
        output_dir=output_dir,
        fps=int(config["data"].get("fps", 10)),
        checkpoint=checkpoint,
        config_path=config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
