"""Reconstruct random Kinetics-600 clips with the released V-RAE checkpoint.

The manifest and directory layout follow the ``Upload data manifest`` README:
the CSV contains paths such as ``vol_28/<clip>.mp4`` and ``--data-root`` points
at the ``Kinetics600/videos`` directory containing those ``vol_*`` folders.

Example::

    python test/reconstruct_k600.py --data-root /path/to/Kinetics600/videos \
        --num-clips 4

Outputs are side-by-side ``original | reconstruction`` videos in
``test/vis/k600`` and a JSON manifest describing the sampled clips.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrae.checkpoint import load_checkpoint
from vrae.models.autoencoder import VRAE
from vrae.paths import ProjectPaths
from vrae.training.common.visualization import comparison_video, save_video_mp4


DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpts/pretrained/vrae/vrae_vjepa2.1.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/metadata/k600_val.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/Kinetics600/videos"
DEFAULT_OUTPUT = PROJECT_ROOT / "test/vis/k600"
IMAGE_SIZE = (256, 256)


def _ema_load(model: VRAE, payload: Mapping[str, object]) -> None:
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V-RAE checkpoint does not contain EMA weights")
    shadow = ema["shadow"]
    state = torch.nn.ModuleDict(model.trainable_groups()).state_dict()
    expected = {key for key, value in state.items() if torch.is_floating_point(value)}
    if set(shadow) != expected:
        missing = sorted(expected.difference(shadow))
        unexpected = sorted(set(shadow).difference(expected))
        raise ValueError(
            "EMA state does not match the V-RAE decoder: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    for key, value in shadow.items():
        state[key].copy_(torch.as_tensor(value).to(state[key]))


def _model_config(payload: Mapping[str, object]) -> dict[str, object]:
    raw = payload.get("resolved_config")
    if not isinstance(raw, Mapping):
        raise ValueError("V-RAE checkpoint is missing resolved_config")
    config = copy.deepcopy(dict(raw))
    model_config = dict(config.get("model", config))
    decoder = dict(model_config.get("decoder", {}))
    decoder["init"] = "scratch"
    decoder.pop("checkpoint", None)
    parameters = decoder.get("parameters")
    if isinstance(parameters, Mapping):
        decoder["parameters"] = {**dict(parameters), "attention_backend": "sdpa"}
    else:
        decoder["attention_backend"] = "sdpa"
    model_config["decoder"] = decoder
    # The released config points to the same project-relative encoder path.
    encoder = dict(model_config.get("encoder", {}))
    encoder["checkpoint"] = "ckpts/pretrained/encoders/vjepa2_1/model.pt"
    model_config["encoder"] = encoder
    if "model" in config:
        config["model"] = model_config
    else:
        config.update(model_config)
    return config


def _load_model(checkpoint: Path, device: torch.device) -> VRAE:
    payload = load_checkpoint(checkpoint, map_location="cpu", mmap=True)
    model = VRAE.from_config(
        _model_config(payload), project_paths=ProjectPaths(project_root=PROJECT_ROOT)
    )
    model.load_state_dict(payload["model"], strict=True)
    _ema_load(model, payload)
    model.requires_grad_(False).eval().to(device)
    return model


def _read_rows(manifest: Path, root: Path) -> list[dict[str, object]]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        relative = Path(str(row["path"]))
        path = relative if relative.is_absolute() else root / relative
        if not path.is_file():
            # Official CVD tar extraction uses ``<split>/<action>/<file>.mp4``
            # rather than the manifest's volume buckets.  Match by filename
            # so the same script works with either layout.
            candidates = tuple(root.rglob(relative.name))
            if len(candidates) == 1:
                path = candidates[0]
        if path.is_file():
            result.append({**row, "path": path, "relative_path": str(relative)})
    if not result:
        raise FileNotFoundError(
            f"No manifest videos were found below {root}. "
            "Download Kinetics-600 from the OpenDataLab link in README.md and "
            "point --data-root at its Kinetics600/videos directory."
        )
    return result


def _decode_clip(path: Path, num_frames: int) -> tuple[torch.Tensor, float, list[int]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames = [frame.to_rgb().to_ndarray() for frame in container.decode(stream)]
    if len(frames) < num_frames:
        raise ValueError(f"{path} has {len(frames)} frames; need {num_frames}")
    # Uniformly sample one contiguous clip from the already trimmed K600 video.
    start = 0
    indices = torch.linspace(start, len(frames) - 1, num_frames).round().long().tolist()
    video = torch.from_numpy(np.stack([frames[i] for i in indices]))
    video = video.permute(0, 3, 1, 2).float().div_(255.0)
    height, width = video.shape[-2:]
    scale = max(IMAGE_SIZE[0] / height, IMAGE_SIZE[1] / width)
    resized = F.interpolate(
        video,
        size=(round(height * scale), round(width * scale)),
        mode="bilinear",
        align_corners=False,
    )
    top = (resized.shape[-2] - IMAGE_SIZE[0]) // 2
    left = (resized.shape[-1] - IMAGE_SIZE[1]) // 2
    return resized[..., top : top + IMAGE_SIZE[0], left : left + IMAGE_SIZE[1]].contiguous(), fps, indices


def reconstruct_k600(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    manifest: str | Path = DEFAULT_MANIFEST,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    num_clips: int = 4,
    num_frames: int = 20,
    seed: int = 3407,
    device: str | None = None,
) -> list[Path]:
    if num_clips <= 0 or num_frames <= 0 or num_frames % 4:
        raise ValueError("num-clips and num-frames must be positive; num-frames divisible by 4")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    root_path = Path(data_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    rows = _read_rows(manifest_path, root_path)
    if num_clips > len(rows):
        raise ValueError(f"Requested {num_clips} clips, but only {len(rows)} are available")
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(rows), generator=generator)[:num_clips].tolist()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(checkpoint_path, selected_device)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    records: list[dict[str, object]] = []
    for clip_number, row_index in enumerate(selected):
        row = rows[row_index]
        real, fps, frame_indices = _decode_clip(Path(row["path"]), num_frames)
        model_input = real.unsqueeze(0).to(selected_device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if selected_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            reconstructed = model(model_input)["recon"][0].float().clamp_(0.0, 1.0).cpu()
        if reconstructed.shape != real.shape:
            raise RuntimeError(f"Unexpected reconstruction shape {tuple(reconstructed.shape)}")
        destination = output_path / f"clip-{clip_number:04d}-comparison.mp4"
        comparison = comparison_video(
            real.unsqueeze(0), reconstructed.unsqueeze(0)
        )[0]
        save_video_mp4(comparison, destination, fps=max(1, round(fps)))
        outputs.append(destination)
        records.append(
            {
                "clip": clip_number,
                "dataset_index": row_index,
                "path": row["relative_path"],
                "action": row.get("action"),
                "label": int(row["label"]),
                "frame_indices": frame_indices,
                "video": destination.name,
            }
        )
    manifest_output = output_path / "manifest.json"
    manifest_output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "source_manifest": str(manifest_path),
                "seed": seed,
                "num_clips": num_clips,
                "videos": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(manifest_output)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    for output in reconstruct_k600(**vars(args)):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
