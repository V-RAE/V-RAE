from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from vrae.checkpoint import load_checkpoint
from vrae.config import load_config
from vrae.evaluation.common.frechet import frechet_from_features
from vrae.evaluation.common.paired import PairedMetricAccumulator
from vrae.evaluation.models.i3d import I3DFeatureExtractor
from vrae.evaluation.models.lpips import LPIPSMetric
from vrae.paths import ProjectPaths, find_project_root
from vrae.training.recon_training.data import build_reconstruction_dataset, collate_reconstruction
from vrae.models.autoencoder import VRAE


def _ema_to_model(model: torch.nn.Module, payload: Mapping[str, object]) -> None:
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("EMA evaluation requested but checkpoint has no EMA state")
    shadow = ema["shadow"]
    state = model.state_dict()
    trainable_state = torch.nn.ModuleDict(model.trainable_groups()).state_dict()
    expected = {name for name, value in trainable_state.items() if torch.is_floating_point(value)}
    if set(shadow) != expected:
        raise ValueError("checkpoint EMA keys do not match the evaluated model")
    for name, value in shadow.items():
        state[name].copy_(torch.as_tensor(value).to(state[name]))


def _config_with_eval_backend(config: Mapping[str, object]) -> dict[str, object]:
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


def evaluate(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, float | int | str]:
    config_file = Path(config_path).expanduser().resolve()
    project_root = find_project_root(config_file)
    config = load_config(config_file)
    paths = ProjectPaths(project_root=project_root)
    checkpoint_value = checkpoint or config["checkpoints"]["stage1"]
    checkpoint_path = paths.checkpoint(checkpoint_value, require_exists=True)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = _config_with_eval_backend(payload["resolved_config"])
    model = VRAE.from_config(model_config, project_paths=paths)
    model.load_state_dict(payload["model"], strict=True)
    if str(config.get("evaluation", {}).get("weights", "ema")) == "ema":
        _ema_to_model(model, payload)

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device).eval()
    dataset = build_reconstruction_dataset(payload["resolved_config"], paths)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    local_batch_size = int(batch_size or evaluation.get("batch_size", 1))
    if local_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=int(evaluation.get("num_workers", 0)),
        collate_fn=collate_reconstruction,
        pin_memory=selected_device.type == "cuda",
    )

    extractors = config.get("extractors", {})
    if not isinstance(extractors, Mapping):
        raise ValueError("extractors must be a mapping")
    eval_root = project_root / "ckpts" / "eval_models"
    i3d_path = paths.checkpoint(extractors["i3d"], require_exists=True)
    lpips_path = paths.checkpoint(extractors["lpips"], require_exists=True)
    i3d = I3DFeatureExtractor(i3d_path, checkpoint_root=eval_root).to(selected_device).eval()
    lpips = LPIPSMetric(
        lpips_path,
        checkpoint_root=project_root / "ckpts",
    ).to(selected_device).eval()
    paired = PairedMetricAccumulator(
        device=selected_device,
        batch_size=int(evaluation.get("metric_batch_size", 8)),
        lpips_model=lpips,
    )
    real_features: list[torch.Tensor] = []
    reconstructed_features: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(selected_device, non_blocking=True)
            if video.ndim == 6:
                batch_count, frames, views, channels, height, width = video.shape
                video = video.permute(0, 2, 1, 3, 4, 5).reshape(
                    batch_count * views, frames, channels, height, width
                )
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda",
            ):
                reconstructed = model(video)["recon"]
            reconstructed = reconstructed.float().clamp(0, 1)
            if reconstructed.shape != video.shape:
                raise ValueError("reconstruction shape differs from input")
            real_features.append(i3d(video))
            reconstructed_features.append(i3d(reconstructed))
            paired.update(video.float(), reconstructed)

    real = torch.cat(real_features, dim=0).cpu()
    reconstructed = torch.cat(reconstructed_features, dim=0).cpu()
    metrics = {
        "rFVD": frechet_from_features(real, reconstructed, covariance="sample"),
        **paired.compute(),
        "samples": int(real.shape[0]),
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LIBERO V-RAE video reconstruction")
    parser.add_argument("--config", default="configs/evaluation/libero_rfvd/config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    metrics = evaluate(
        args.config,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
