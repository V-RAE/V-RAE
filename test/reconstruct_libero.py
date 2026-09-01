"""Sample LIBERO clips, run the V-RAE reconstruction model, and export videos.

Example::

    python test/reconstruct_libero.py --num-clips 4

The script uses the EMA weights from the requested Stage 1 checkpoint by
default. For each sampled clip and camera view it writes one side-by-side
comparison (real | reconstruction) below ``test/vis``.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from vrae.checkpoint import load_checkpoint
from vrae.models.autoencoder import VRAE
from vrae.paths import ProjectPaths, find_project_root
from vrae.training.common.visualization import comparison_video, save_video_mp4
from vrae.training.recon_training.data import (
    build_reconstruction_dataset,
    collate_reconstruction,
    flatten_multiview_video,
)


DEFAULT_CHECKPOINT = (
    "ckpts/recon_training/"
    "vjepa2_1_libero_reconstruction_single_gpu_bs4_50k_detached/"
    "checkpoints/latest.pt"
)


def _config_with_sdpa(config: Mapping[str, object]) -> dict[str, object]:
    """Use the portable SDPA decoder backend during standalone evaluation."""

    result = copy.deepcopy(dict(config))
    model = dict(result.get("model", result))
    decoder = dict(model.get("decoder", {}))
    parameters = decoder.get("parameters")
    if isinstance(parameters, Mapping):
        decoder["parameters"] = {**parameters, "attention_backend": "sdpa"}
    else:
        decoder["attention_backend"] = "sdpa"
    model["decoder"] = decoder
    if "model" in result:
        result["model"] = model
    else:
        result.update(model)
    return result


def _load_ema_weights(model: VRAE, payload: Mapping[str, object]) -> None:
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("Checkpoint does not contain EMA weights")

    shadow = ema["shadow"]
    model_state = model.state_dict()
    trainable_state = torch.nn.ModuleDict(model.trainable_groups()).state_dict()
    expected = {
        name for name, value in trainable_state.items() if torch.is_floating_point(value)
    }
    if set(shadow) != expected:
        raise ValueError("Checkpoint EMA keys do not match the V-RAE trainable state")
    for name, value in shadow.items():
        model_state[name].copy_(torch.as_tensor(value).to(model_state[name]))


def _resolve_checkpoint(paths: ProjectPaths, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return paths.checkpoint(path, require_exists=True)


def _save_comparison_video(
    real: torch.Tensor,
    reconstructed: torch.Tensor,
    destination: Path,
    *,
    fps: int,
    clip_number: int,
    view_name: str,
) -> Path:
    """Save one side-by-side comparison video for one camera view."""

    comparison = comparison_video(real.unsqueeze(0), reconstructed.unsqueeze(0))[0]
    output = destination / f"clip-{clip_number:04d}-{view_name}-comparison.mp4"
    return save_video_mp4(comparison, output, fps=fps)


def reconstruct_libero(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    num_clips: int = 4,
    batch_size: int = 1,
    seed: int = 3407,
    device: str | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path = PROJECT_ROOT / "test" / "vis",
    weights: str = "ema",
    fps: int | None = None,
) -> list[Path]:
    """Reconstruct randomly selected LIBERO clips and save MP4 visualizations."""

    if num_clips <= 0:
        raise ValueError("num_clips must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if weights not in {"ema", "model"}:
        raise ValueError("weights must be 'ema' or 'model'")

    project_root = find_project_root(__file__)
    paths = ProjectPaths(project_root=project_root)
    checkpoint_path = _resolve_checkpoint(paths, checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if str(payload.get("task")) != "recon_training":
        raise ValueError(f"Expected a recon_training checkpoint, got {payload.get('task')!r}")

    model_config = _config_with_sdpa(payload["resolved_config"])
    if data_root is not None:
        model_config["data"] = {
            **dict(model_config.get("data", {})),
            "root": str(Path(data_root).expanduser()),
        }
    model = VRAE.from_config(model_config, project_paths=paths)
    model.load_state_dict(payload["model"], strict=True)
    if weights == "ema":
        _load_ema_weights(model, payload)
    del payload

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device).eval()
    dataset = build_reconstruction_dataset(model_config, paths)
    if num_clips > len(dataset):
        raise ValueError(f"num_clips={num_clips} exceeds dataset size {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(dataset), generator=generator)[:num_clips].tolist()
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_reconstruction,
        pin_memory=selected_device.type == "cuda",
    )

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    manifest: list[dict[str, object]] = []
    output_fps = int(fps if fps is not None else model_config.get("data", {}).get("fps", 10))
    if output_fps <= 0:
        raise ValueError("fps must be positive")

    with torch.inference_mode():
        for batch in loader:
            real = batch["video"]
            if real.ndim != 6:
                raise ValueError(f"Expected multiview LIBERO video [B,T,V,C,H,W], got {tuple(real.shape)}")
            batch_count, frames, views, channels, height, width = real.shape
            model_input = flatten_multiview_video(real).to(selected_device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if selected_device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                reconstructed = model(model_input)["recon"]
            reconstructed = reconstructed.float().clamp(0.0, 1.0)
            expected_shape = (batch_count * views, frames, channels, height, width)
            if tuple(reconstructed.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected reconstruction shape {tuple(reconstructed.shape)}; "
                    f"expected {expected_shape}"
                )
            reconstructed = (
                reconstructed.reshape(batch_count, views, frames, channels, height, width)
                .permute(0, 2, 1, 3, 4, 5)
                .cpu()
            )

            for item_index in range(batch_count):
                clip_number = len(manifest)
                item_manifest = {
                    "clip": clip_number,
                    "dataset_index": selected_indices[clip_number],
                    "sample_id": batch["sample_id"][item_index],
                    "frame_indices": batch["frame_indices"][item_index].tolist(),
                    "task": batch.get("task", [None] * batch_count)[item_index],
                }
                item_manifest["videos"] = {}
                for view_index, camera in enumerate(dataset.camera_keys):
                    view_name = str(camera["name"]).replace("/", "_")
                    output_path = _save_comparison_video(
                        real[item_index, :, view_index],
                        reconstructed[item_index, :, view_index],
                        destination,
                        fps=output_fps,
                        clip_number=clip_number,
                        view_name=view_name,
                    )
                    output_paths.append(output_path)
                    item_manifest["videos"][view_name] = str(
                        output_path.relative_to(destination)
                    )
                manifest.append(item_manifest)

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "weights": weights,
                "seed": seed,
                "num_clips": num_clips,
                "videos": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths.append(manifest_path)
    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", help="Torch device (default: cuda when available, otherwise cpu)")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test" / "vis")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--fps", type=int)
    args = parser.parse_args(argv)
    for output in reconstruct_libero(
        checkpoint=args.checkpoint,
        num_clips=args.num_clips,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        output_dir=args.output_dir,
        weights=args.weights,
        fps=args.fps,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
