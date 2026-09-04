"""Generate stitched multi-view LIBERO videos from a VideoDiT checkpoint.

Example:
    PYTHONPATH=src python scripts/sample_libero_multiview.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vrae.config import load_config
from vrae.paths import ProjectPaths
from vrae.training.common.engine import (
    build_flow_transport,
    generate_class_conditional,
    load_class_conditional_sampling_stack,
)
from vrae.training.common.visualization import save_video_mp4


def stitch_views(video: torch.Tensor) -> torch.Tensor:
    """Concatenate [B,T,V,C,H,W] views along width -> [B,T,C,H,V*W]."""

    if video.ndim != 6:
        raise ValueError(f"Expected [B,T,V,C,H,W], got {tuple(video.shape)}")
    batch, frames, views, channels, height, width = video.shape
    return (
        video.permute(0, 1, 3, 4, 2, 5)
        .reshape(batch, frames, channels, height, views * width)
        .contiguous()
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/training/generation/libero_videodit_reconstruction_multiview.yaml",
    )
    parser.add_argument(
        "--dit-checkpoint",
        default="ckpts/libero_videogen/vjepa2_1_libero_videodit_reconstruction_multiview/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="test/vis/libero_multiview_sampling",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 10, 25, 50, 100])
    parser.add_argument("--class-ids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    paths = ProjectPaths(project_root=root)
    config = load_config(args.config)
    configured_classes = config.get("sampling", {}).get("class_ids", [0])
    class_ids = [int(value) for value in (args.class_ids or configured_classes)]
    if not class_ids:
        raise ValueError("At least one class id is required")
    if any(value < 0 or value >= int(config["dit"]["num_classes"]) for value in class_ids):
        raise ValueError(f"class ids must be in [0, {int(config['dit']['num_classes'])})")
    if any(int(value) <= 0 for value in args.steps):
        raise ValueError("sampling steps must be positive")
    if int(args.warmup_steps) <= 0:
        raise ValueError("warmup steps must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1, dit, normalizer = load_class_conditional_sampling_stack(
        config,
        paths,
        args.dit_checkpoint,
        device,
    )
    transport = build_flow_transport(config)
    labels = torch.tensor(class_ids, dtype=torch.long, device=device)
    sample_ids = [int(config["sampling"].get("base_seed", 3407)) + index for index in range(len(class_ids))]
    sampling = config.get("sampling", {})
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, float | int]] = []

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        synchronize(device)
        warmup_started = time.perf_counter()
        generate_class_conditional(
            dit,
            stage1,
            normalizer,
            transport=transport,
            sample_ids=sample_ids,
            labels=labels,
            base_seed=int(sampling.get("base_seed", 3407)),
            num_chunks=dit.num_chunks,
            grid_size=tuple(dit.grid_size),
            channels=dit.in_channels,
            steps=int(args.warmup_steps),
            cfg_scale=float(sampling.get("cfg_scale", 1.0)),
            internal_guidance_scale=float(
                sampling.get("internal_guidance_scale", 1.0)
            ),
            internal_guidance_t_min=float(
                sampling.get("internal_guidance_t_min", 0.0)
            ),
            internal_guidance_t_max=float(
                sampling.get("internal_guidance_t_max", 1.0)
            ),
            device=device,
        )
        synchronize(device)
        print(
            f"warmup_steps={int(args.warmup_steps)} "
            f"seconds={time.perf_counter() - warmup_started:.3f}",
            flush=True,
        )

        for steps in args.steps:
            synchronize(device)
            started = time.perf_counter()
            video = generate_class_conditional(
                dit,
                stage1,
                normalizer,
                transport=transport,
                sample_ids=sample_ids,
                labels=labels,
                base_seed=int(sampling.get("base_seed", 3407)),
                num_chunks=dit.num_chunks,
                grid_size=tuple(dit.grid_size),
                channels=dit.in_channels,
                steps=int(steps),
                cfg_scale=float(sampling.get("cfg_scale", 1.0)),
                internal_guidance_scale=float(
                    sampling.get("internal_guidance_scale", 1.0)
                ),
                internal_guidance_t_min=float(
                    sampling.get("internal_guidance_t_min", 0.0)
                ),
                internal_guidance_t_max=float(
                    sampling.get("internal_guidance_t_max", 1.0)
                ),
                device=device,
            )
            synchronize(device)
            generation_seconds = time.perf_counter() - started

            video = video.float().clamp(0.0, 1.0).cpu()
            stitched = stitch_views(video)
            step_dir = output_dir / f"steps-{int(steps):03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video_multiview": video,
                    "video_stitched": stitched,
                    "labels": labels.cpu(),
                    "sample_ids": sample_ids,
                    "steps": int(steps),
                },
                step_dir / "samples.pt",
            )
            for index, class_id in enumerate(class_ids):
                save_video_mp4(
                    stitched[index],
                    step_dir / f"class-{class_id:03d}.mp4",
                    fps=int(config["data"].get("fps", 8)),
                )
            total_seconds = time.perf_counter() - started
            timings.append(
                {
                    "steps": int(steps),
                    "num_samples": len(class_ids),
                    "generation_seconds": generation_seconds,
                    "total_seconds": total_seconds,
                }
            )
            print(
                f"steps={int(steps)} samples={len(class_ids)} "
                f"generation_seconds={generation_seconds:.3f} "
                f"total_seconds={total_seconds:.3f} output={step_dir}",
                flush=True,
            )

    summary = {
        "config": args.config,
        "dit_checkpoint": args.dit_checkpoint,
        "class_ids": class_ids,
        "steps": [int(value) for value in args.steps],
        "device": str(device),
        "video_shape_multiview": list(video.shape),
        "video_shape_stitched": list(stitched.shape),
        "timings": timings,
    }
    with (output_dir / "timing.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(output_dir / "timing.json")


if __name__ == "__main__":
    main()
