"""Visualize temporal changes in a saved V-RAE reconstruction sample.

The reconstruction sample written by ``recon_training`` contains decoded
``real`` and ``recon`` videos, not the intermediate latent tensor.  This tool
therefore analyzes ``recon`` as the decoded latent output.

Example:
    python test/analyze_reconstruction_temporal.py \
        ckpts/recon_training/vjepa2_1_libero_reconstruction_random_interval_bs8/samples/step-00008000.pt \
        --clip-index 0 \
        --output-dir test/vis/step-00008000-temporal

Outputs:
    recon_temporal_overview.png  All frames and adjacent-frame differences.
    recon_four_frame_windows.png Four-frame windows with local differences.
    recon_boundary_frame04_05.png The 4 -> 5 frame boundary in detail.
    temporal_metrics.json         Per-pair and per-window mean absolute changes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps


THUMBNAIL_SIZE = 160
LABEL_HEIGHT = 24
PADDING = 8
BACKGROUND = (245, 245, 245)
TEXT = (20, 20, 20)


def _as_batch(video: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(video):
        raise TypeError(f"{name!r} must be a tensor")
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"{name!r} must have shape [B,T,C,H,W], got {tuple(video.shape)}")
    if video.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name!r} must have 1, 3, or 4 channels, got {video.shape[2]}")
    return video.detach().float().clamp(0.0, 1.0)


def _frame_image(frame: torch.Tensor, size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Convert one [C,H,W] frame to a labeled-friendly RGB thumbnail."""
    frame = frame.clamp(0.0, 1.0)
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    elif frame.shape[0] == 4:
        frame = frame[:3]
    array = frame.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)


def _difference_image(difference: torch.Tensor, scale: float, size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Render a per-pixel absolute difference using a fixed clip-wide scale."""
    difference = difference.mean(dim=0).clamp(0.0, scale) / max(scale, 1.0e-8)
    array = difference.mul(255.0).round().to(torch.uint8).numpy()
    image = Image.fromarray(array, mode="L").resize((size, size), Image.Resampling.BILINEAR)
    # A colored map makes small motion easier to see than a grayscale map.
    return ImageOps.colorize(image, black=(8, 8, 8), mid=(255, 220, 50), white=(220, 20, 20))


def _caption(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + LABEL_HEIGHT), BACKGROUND)
    canvas.paste(image, (0, LABEL_HEIGHT))
    ImageDraw.Draw(canvas).text((4, 4), label, fill=TEXT)
    return canvas


def _add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 30), BACKGROUND)
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((PADDING, 7), title, fill=TEXT)
    return canvas


def _save_overview(frames: torch.Tensor, output: Path) -> None:
    differences = (frames[1:] - frames[:-1]).abs()
    scale = float(torch.quantile(differences, 0.99).item())
    images: list[Image.Image] = []
    for index, frame in enumerate(frames):
        images.append(_caption(_frame_image(frame), f"frame {index + 1}"))
    for index, difference in enumerate(differences):
        images.append(_caption(_difference_image(difference, scale), f"diff {index + 1}->{index + 2}"))
    grid = _grid_to_image(images, columns=4)
    _add_title(grid, "Decoded reconstruction: frames (top block) and adjacent differences (second block onward)").save(output)


def _grid_to_image(images: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (columns * cell_width + (columns + 1) * PADDING, rows * cell_height + (rows + 1) * PADDING),
        BACKGROUND,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(image, (PADDING + column * cell_width, PADDING + row * cell_height))
    return canvas


def _save_four_frame_windows(frames: torch.Tensor, output: Path) -> None:
    differences = (frames[1:] - frames[:-1]).abs()
    scale = float(torch.quantile(differences, 0.99).item())
    windows: list[Image.Image] = []
    for start in range(0, frames.shape[0], 4):
        end = min(start + 4, frames.shape[0])
        if end - start < 2:
            continue
        images = [_caption(_frame_image(frames[index]), f"frame {index + 1}") for index in range(start, end)]
        images.extend(
            _caption(_difference_image(differences[index], scale), f"diff {index + 1}->{index + 2}")
            for index in range(start, end - 1)
        )
        window = _add_title(
            _grid_to_image(images, columns=len(images)),
            f"4-frame window: frames {start + 1}-{end}",
        )
        windows.append(window)
    width = max(image.width for image in windows)
    height = sum(image.height for image in windows) + (len(windows) + 1) * PADDING
    canvas = Image.new("RGB", (width + 2 * PADDING, height), BACKGROUND)
    y = PADDING
    for image in windows:
        canvas.paste(image, (PADDING, y))
        y += image.height + PADDING
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _save_boundary(frames: torch.Tensor, output: Path, left_index: int = 3) -> None:
    if left_index < 0 or left_index + 1 >= frames.shape[0]:
        raise ValueError(f"Boundary index {left_index} is outside a {frames.shape[0]}-frame clip")
    difference = (frames[left_index + 1] - frames[left_index]).abs()
    scale = float(torch.quantile((frames[1:] - frames[:-1]).abs(), 0.99).item())
    images = [
        _caption(_frame_image(frames[left_index]), f"frame {left_index + 1}"),
        _caption(_frame_image(frames[left_index + 1]), f"frame {left_index + 2}"),
        _caption(_difference_image(difference, scale), f"absolute diff {left_index + 1}->{left_index + 2}"),
    ]
    image = _add_title(_grid_to_image(images, columns=3), "Critical interval boundary: frame 4 -> frame 5")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _metrics(frames: torch.Tensor) -> dict[str, object]:
    differences = (frames[1:] - frames[:-1]).abs()
    pair_mae = differences.mean(dim=(1, 2, 3))
    pair_p95 = torch.quantile(differences.flatten(1), 0.95, dim=1)
    window_metrics = []
    for start in range(0, frames.shape[0], 4):
        end = min(start + 4, frames.shape[0])
        if end - start >= 2:
            values = pair_mae[start : end - 1]
            window_metrics.append(
                {
                    "frames_1_based": [start + 1, end],
                    "mean_adjacent_mae": float(values.mean()),
                    "max_adjacent_mae": float(values.max()),
                }
            )
    return {
        "num_frames": int(frames.shape[0]),
        "shape_t_c_h_w": list(frames.shape),
        "adjacent_pair_mae": [float(value) for value in pair_mae],
        "adjacent_pair_p95": [float(value) for value in pair_p95],
        "frame_4_to_frame_5_mae": float(pair_mae[3]) if pair_mae.numel() > 3 else None,
        "four_frame_windows": window_metrics,
    }


def analyze(sample: Path, output_dir: Path, clip_index: int) -> None:
    payload = torch.load(sample, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "recon" not in payload:
        raise KeyError(f"Expected a mapping with a 'recon' tensor in {sample}")
    recon = _as_batch(payload["recon"], "recon")
    if not 0 <= clip_index < recon.shape[0]:
        raise IndexError(f"clip-index must be in [0, {recon.shape[0] - 1}], got {clip_index}")
    frames = recon[clip_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_overview(frames, output_dir / "recon_temporal_overview.png")
    _save_four_frame_windows(frames, output_dir / "recon_four_frame_windows.png")
    _save_boundary(frames, output_dir / "recon_boundary_frame04_05.png")
    metrics = _metrics(frames)
    metrics["sample"] = str(sample)
    metrics["clip_index"] = clip_index
    (output_dir / "temporal_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"sample={sample}")
    print(f"clip_index={clip_index}, recon_shape={tuple(recon.shape)}")
    print(f"frame_4_to_frame_5_mae={metrics['frame_4_to_frame_5_mae']}")
    print(f"outputs={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Saved recon_training sample .pt file")
    parser.add_argument("--clip-index", type=int, default=0, help="Batch item to inspect (default: 0)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test/vis/temporal_analysis"),
        help="Output directory for PNGs and JSON metrics",
    )
    args = parser.parse_args()
    analyze(args.sample, args.output_dir, args.clip_index)


if __name__ == "__main__":
    main()
