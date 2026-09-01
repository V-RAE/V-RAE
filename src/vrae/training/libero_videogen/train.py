from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.training.common.engine import (
    compute_class_conditional_latent_stats,
    flow_transport_metadata,
    parse_config_argument,
    prepare_run,
    train_class_conditional,
)


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    validate_wandb_config(config["wandb"])
    if config.get("task") != "libero_videogen":
        raise ValueError("LIBERO generation requires task=libero_videogen")
    dataset_name = str(config["data"].get("dataset", "lerobot"))
    if dataset_name not in {"lerobot", "reconstruction"}:
        raise ValueError(
            "LIBERO generation requires data.dataset=lerobot or reconstruction"
        )

    num_classes = int(config["dit"]["num_classes"])
    if num_classes <= 0:
        raise ValueError("dit.num_classes must be positive")
    sample_class_ids = [int(value) for value in config["sampling"].get("class_ids", [])]
    if any(value < 0 or value >= num_classes for value in sample_class_ids):
        raise ValueError("sampling.class_ids must reference valid LIBERO classes")

    num_frames = int(config["data"]["num_frames"])
    if num_frames <= 0 or num_frames % 4:
        raise ValueError("data.num_frames must be a positive multiple of four")
    cameras = config["data"]["camera_keys"]
    multiview = config["model"].get("multiview", {})
    num_views = int(multiview.get("num_views", 1)) if multiview.get("enabled") else 1
    if num_views != len(cameras):
        raise ValueError("model.multiview.num_views must match data.camera_keys")

    transport = flow_transport_metadata(config)
    num_chunks = num_frames // 4
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"]["encoder"]["patch_size"])
    hidden_size = int(config["model"]["encoder"]["hidden_size"])
    spatial_tokens = (image_size // patch_size) ** 2
    expected_shift = math.sqrt(
        num_chunks * num_views * spatial_tokens * hidden_size / 4096
    )
    if transport["prediction"] != "x":
        raise ValueError("LIBERO VideoDiT uses x-prediction")
    if abs(transport["time_dist_shift"] - expected_shift) > 1.0e-9:
        raise ValueError(
            "transport.time_dist_shift must match the multiview latent dimension: "
            f"expected {expected_shift}"
        )
    pipeline = config.get("runtime", {}).get("data_pipeline", {})
    if pipeline.get("kind") != "lerobot_bounded":
        raise ValueError("LIBERO generation requires runtime.data_pipeline.kind=lerobot_bounded")
    return {
        "task": config["task"],
        "dataset": dataset_name,
        "num_classes": num_classes,
        "num_chunks": num_chunks,
        "num_views": num_views,
        "transport": transport,
    }


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    from vrae.training.common.distributed import broadcast_object, initialize_distributed

    context = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing: bool | None = None
    if context.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if context.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output

    if context.is_main:
        print(f"[latent-stats] computing LIBERO statistics: {output}", flush=True)
    compute_class_conditional_latent_stats(
        config,
        paths,
        dataset_name=str(config["data"].get("dataset", "lerobot")),
    )

    failure: str | None = None
    if context.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"latent statistics were not created: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    return output


def main() -> None:
    arguments = parse_config_argument("Train LIBERO class-conditional V-RAE VideoDiT")
    initial_config = load_config(arguments.config)
    validation = validate_build(initial_config)
    if arguments.build_only:
        print(json.dumps(validation, indent=2))
        return

    initial_paths = load_project_paths(
        initial_config,
        override=arguments.paths,
        project_root=find_project_root(arguments.config),
    )
    ensure_latent_stats(initial_config, initial_paths)
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    dataset_name = str(config["data"].get("dataset", "lerobot"))
    train_class_conditional(
        config,
        paths,
        run,
        dataset_name=dataset_name,
        max_steps=arguments.max_steps,
    )


if __name__ == "__main__":
    main()
