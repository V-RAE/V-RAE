"""Export reconstruction samples saved by recon_training as MP4 videos.

Example:
    python test/save_reconstruction_video.py \
        ckpts/recon_training/vjepa2_1_libero_reconstruction_single_gpu_bs4_50k_detached/samples/step-00036000.pt

The default ``comparison`` mode writes one side-by-side (real | reconstruction)
video per item in the sample batch.  Use ``--mode real`` or ``--mode recon`` to
export only one side.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from vrae.training.common.visualization import (
    comparison_video,
    save_video_mp4,
    video_to_uint8,
)


def _as_batch(video: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize a clip to [B,T,C,H,W] (or [B,T,V,C,H,W])."""

    if not torch.is_tensor(video):
        raise TypeError(f"Sample field {name!r} must be a torch.Tensor")
    if video.ndim == 4:
        return video.unsqueeze(0)
    if video.ndim in (5, 6):
        return video
    raise ValueError(
        f"Sample field {name!r} must have shape [T,C,H,W], [B,T,C,H,W], "
        f"or [B,T,V,C,H,W]; got {tuple(video.shape)}"
    )


def save_sample_videos(
    sample_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    fps: int = 8,
    mode: str = "comparison",
) -> list[Path]:
    """Load one reconstruction sample and save its clips as MP4 files."""

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if mode not in {"comparison", "real", "recon"}:
        raise ValueError(f"Unsupported mode {mode!r}")

    sample_path = Path(sample_path)
    payload = torch.load(sample_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {sample_path}, got {type(payload).__name__}")

    if mode == "comparison":
        try:
            real = _as_batch(payload["real"], name="real")
            recon = _as_batch(payload["recon"], name="recon")
        except KeyError as error:
            raise KeyError("Comparison mode requires sample fields 'real' and 'recon'") from error
        if real.shape != recon.shape:
            raise ValueError(
                "Real and reconstructed videos must have equal shape; "
                f"got {tuple(real.shape)} and {tuple(recon.shape)}"
            )
        video = comparison_video(real, recon)
    else:
        video = _as_batch(payload[mode], name=mode)

    # video_to_uint8 also flattens a possible view dimension, yielding [N,T,C,H,W].
    clips = video_to_uint8(video)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "vis"
    )
    destination.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for index, clip in enumerate(clips):
        output_path = destination / f"{sample_path.stem}-{mode}-{index:04d}.mp4"
        saved.append(save_video_mp4(clip, output_path, fps=fps))
    return saved


def save_sample_frames(
    sample_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    mode: str = "comparison",
) -> list[Path]:
    """Save each clip frame as a PNG, optionally with real/reconstruction side by side."""

    if mode not in {"comparison", "real", "recon"}:
        raise ValueError(f"Unsupported mode {mode!r}")

    sample_path = Path(sample_path)
    payload = torch.load(sample_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {sample_path}, got {type(payload).__name__}")

    if mode == "comparison":
        try:
            real = _as_batch(payload["real"], name="real")
            recon = _as_batch(payload["recon"], name="recon")
        except KeyError as error:
            raise KeyError("Comparison mode requires sample fields 'real' and 'recon'") from error
        if real.shape != recon.shape:
            raise ValueError(
                "Real and reconstructed videos must have equal shape; "
                f"got {tuple(real.shape)} and {tuple(recon.shape)}"
            )
        video = comparison_video(real, recon)
    else:
        video = _as_batch(payload[mode], name=mode)

    clips = video_to_uint8(video)
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "vis"
    saved: list[Path] = []
    for video_index, clip in enumerate(clips):
        clip_dir = destination / f"{sample_path.stem}-{mode}-{video_index:04d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(clip):
            frame_array = frame.permute(1, 2, 0).contiguous().numpy()
            if frame_array.shape[-1] == 1:
                frame_array = frame_array[..., 0]
            elif frame_array.shape[-1] == 4:
                frame_array = frame_array[..., :3]
            output_path = clip_dir / f"frame-{frame_index:02d}.png"
            Image.fromarray(frame_array).save(output_path)
            saved.append(output_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Path to a step-XXXXXXXX.pt sample")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MP4 files (default: test/vis)",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output frame rate (default: 8)")
    parser.add_argument(
        "--mode",
        choices=("comparison", "real", "recon"),
        default="comparison",
        help="Export side-by-side comparison, real, or reconstructed clips",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save every frame as a PNG (one directory per video)",
    )
    args = parser.parse_args()

    for output_path in save_sample_videos(
        args.sample,
        args.output_dir,
        fps=args.fps,
        mode=args.mode,
    ):
        print(output_path)
    if args.save_frames:
        for output_path in save_sample_frames(
            args.sample,
            args.output_dir,
            mode=args.mode,
        ):
            print(output_path)


if __name__ == "__main__":
    main()
