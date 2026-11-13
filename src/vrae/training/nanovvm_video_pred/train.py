"""First-frame conditioned 16-frame future prediction on NanoVVM clip manifests.

One observed RGB frame conditions the DiT through spatial AdaLN; the DiT
generates the four V-RAE latent chunks of the next sixteen frames. Noise,
supervision, and the full/base flow-matching losses act on the future chunks only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.registry import DIT_MODELS, register_builtin_models
from vrae.training.common.accumulation import GradientAccumulator, optimizer_step
from vrae.training.common.checkpoint import (
    build_training_checkpoint,
    load_model_init,
    resume_training,
    save_training_checkpoint,
)
from vrae.training.common.contracts import (
    load_run_metadata,
    resolve_batch_contract,
    update_decoder_execution_metadata,
    update_run_metadata,
)
from vrae.training.common.distributed import gather_rng_states, initialize_distributed, reduce_mean
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import (
    build_flow_transport,
    dit_architecture_metadata,
    flow_transport_metadata,
    load_frozen_stage1,
    parse_config_argument,
    prepare_run,
    seed_everything,
)
from vrae.training.common.latent_norm import LatentNormalizer
from vrae.training.common.optim import build_optimizer, build_scheduler
from vrae.training.common.precision import PrecisionPolicy
from vrae.training.common.visualization import comparison_video, save_sample_tensors
from vrae.training.common.wandb import WandbLogger
from vrae.training.nanovvm_video_pred.data import (
    CONDITION_FRAMES,
    DATASET_KEY,
    FUTURE_FRAMES,
    IMAGE_SIZE,
    PREPROCESSING,
    RGB_SEQ_LEN,
    VIDEO_DECODERS,
    build_nanovvm_dataset,
    build_nanovvm_loader,
    frame_protocol_metadata,
    split_source_entries,
    validate_frame_protocol,
    validate_preprocessing,
)
from vrae.training.nanovvm_video_pred.latents import encode_condition_and_future

TASK = "nanovvm_video_pred"
DIT_NAME = "vrae_first_frame_prediction_dit"
TEMPORAL_COMPRESSION = 4
SINGLE_IMAGE_ENCODERS = ("eupe", "dinov3", "siglip2")


def _image_size(data: Mapping[str, Any]) -> int:
    value = data.get("image_size", IMAGE_SIZE)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise ValueError("data.image_size must be square for the 256x256 center crop")
        value = value[0]
    return int(value)


def task_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Shapes derived from the encoder, image size, and future frame count."""

    encoder = config["model"]["encoder"]
    image_size = _image_size(config["data"])
    patch_size = int(encoder["patch_size"])
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by the encoder patch size")
    tubelet = int(encoder.get("encoder_tubelet_size", 1))
    group = int(config["model"]["pooling"]["group_size"])
    if tubelet * group != TEMPORAL_COMPRESSION:
        raise ValueError("V-RAE must compress time by four")
    if FUTURE_FRAMES % TEMPORAL_COMPRESSION:
        raise ValueError("future frames must fill whole V-RAE chunks")
    grid = (image_size // patch_size, image_size // patch_size)
    channels = int(encoder["hidden_size"])
    chunks = FUTURE_FRAMES // TEMPORAL_COMPRESSION
    tokens = grid[0] * grid[1]
    dit_patch = config["dit"].get("patch_size", 1)
    dit_patch = (
        (int(dit_patch), int(dit_patch))
        if isinstance(dit_patch, int)
        else tuple(int(value) for value in dit_patch)
    )
    if grid[0] % dit_patch[0] or grid[1] % dit_patch[1]:
        raise ValueError("the latent grid must be divisible by dit.patch_size")
    patches = (grid[0] // dit_patch[0]) * (grid[1] // dit_patch[1])
    hidden = config["dit"].get("hidden_size", [1536, 2048])
    encoder_width = int(hidden if isinstance(hidden, int) else hidden[0])
    return {
        "image_size": image_size,
        "latent_grid": list(grid),
        "latent_channels": channels,
        "future_chunks": chunks,
        "tokens_per_chunk": tokens,
        "dit_patch_size": list(dit_patch),
        "dit_patches_per_chunk": patches,
        "dit_encoder_width": encoder_width,
        "time_dist_shift": math.sqrt(chunks * tokens * channels / 4096),
    }


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    """Static checks and the shape contract; reads no files (see check_data for paths)."""

    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    if str(config.get("task")) != TASK:
        raise ValueError(f"NanoVVM entry requires task={TASK}")
    data = config["data"]
    if str(data.get("dataset")) != DATASET_KEY:
        raise ValueError(f"data.dataset must be {DATASET_KEY}")
    validate_frame_protocol(data)
    preprocessing = data.get("preprocessing", PREPROCESSING)
    validate_preprocessing(preprocessing)
    if dict(preprocessing) != PREPROCESSING:
        raise ValueError(
            "data.preprocessing must be the NanoVVM protocol for aligned condition/future frames: "
            f"{PREPROCESSING}"
        )
    if _image_size(data) != IMAGE_SIZE:
        raise ValueError(f"data.image_size must be {IMAGE_SIZE}")
    if str(data.get("video_decoder", "decord")) not in VIDEO_DECODERS:
        raise ValueError(f"data.video_decoder must be one of {VIDEO_DECODERS}")
    train_sources = split_source_entries(config, "train")
    val_sources = split_source_entries(config, "val") if data.get("val") is not None else []
    encoder = config["model"]["encoder"]
    if str(encoder.get("name")) not in SINGLE_IMAGE_ENCODERS:
        raise ValueError(
            f"first-frame conditioning needs a per-frame image encoder {SINGLE_IMAGE_ENCODERS}, "
            f"got {encoder.get('name')!r}"
        )
    dit = config["dit"]
    if str(dit.get("name")) != DIT_NAME:
        raise ValueError(f"NanoVVM future prediction must use {DIT_NAME}")
    geometry = task_geometry(config)
    channels = geometry["latent_channels"]
    for key in ("input_dim", "in_channels", "frame_channels"):
        if key in dit and int(dit[key]) != channels:
            raise ValueError(f"dit.{key} must match the V-RAE hidden size {channels}")
    for key in ("future_chunks", "num_future_chunks", "num_chunks", "num_frames"):
        if key in dit and int(dit[key]) != geometry["future_chunks"]:
            raise ValueError(f"dit.{key} must be {geometry['future_chunks']} future chunks")
    for key in ("grid_size", "input_size", "height", "width"):
        if key in dit:
            raise ValueError(f"dit.{key} is derived from the encoder grid; remove it")
    transport = flow_transport_metadata(config)
    if transport["prediction"] != "x":
        raise ValueError("NanoVVM future prediction uses x-prediction")
    if abs(transport["time_dist_shift"] - geometry["time_dist_shift"]) > 1e-9:
        raise ValueError(
            "transport.time_dist_shift must be sqrt(future_chunks*h*w*C/4096) = "
            f"{geometry['time_dist_shift']!r} for the future latents"
        )
    batch = "B"
    grid = geometry["latent_grid"]
    patches = geometry["dit_patches_per_chunk"]
    width = geometry["dit_encoder_width"]
    return {
        "task": TASK,
        "dit": DIT_NAME,
        "frame_protocol": frame_protocol_metadata(),
        "image_size": [geometry["image_size"], geometry["image_size"]],
        "latent_grid": grid,
        "future_chunks": geometry["future_chunks"],
        "transport": transport,
        "expected_time_dist_shift": geometry["time_dist_shift"],
        "shapes": {
            "clip_rgb": [batch, RGB_SEQ_LEN, 3, geometry["image_size"], geometry["image_size"]],
            "condition_rgb": [batch, 3, geometry["image_size"], geometry["image_size"]],
            "future_rgb": [batch, FUTURE_FRAMES, 3, geometry["image_size"], geometry["image_size"]],
            "frame_features": [batch, geometry["tokens_per_chunk"], channels],
            "z_future": [batch, geometry["future_chunks"], geometry["tokens_per_chunk"], channels],
            "frame_condition": [batch, patches, width],
            "time_condition": [batch, 1, width],
            "adaln_condition": [batch, geometry["future_chunks"] * patches, width],
            "dit_sequence": [batch, geometry["future_chunks"] * patches, width],
        },
        "condition_frames": CONDITION_FRAMES,
        "train_sources": [entry["name"] for entry in train_sources],
        "val_sources": [entry["name"] for entry in val_sources],
    }


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    """Fit the train-future normalizer when absent; an existing file is identity-checked later."""

    from vrae.training.common.distributed import barrier, broadcast_object
    from vrae.training.nanovvm_video_pred.latent_stats import compute_nanovvm_latent_stats

    distributed = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing = None
    if distributed.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if distributed.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output
    if distributed.is_main:
        print(
            f"[latent-stats] file missing; fitting NanoVVM train-future statistics: {output}",
            flush=True,
        )
    compute_nanovvm_latent_stats(config, paths)
    barrier()
    failure = None
    if distributed.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"Latent statistics were not created successfully: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    return output


def build_nanovvm_dit(config: Mapping[str, Any], stage1_metadata: Mapping[str, Any]) -> nn.Module:
    register_builtin_models()
    image_size = _image_size(config["data"])
    patch_size = int(stage1_metadata["patch_size"])
    if image_size % patch_size:
        raise ValueError("image size must be divisible by the V-RAE patch size")
    compression = int(stage1_metadata["temporal_compression_ratio"])
    if FUTURE_FRAMES % compression:
        raise ValueError("future frames must fill whole V-RAE chunks")
    grid_size = (image_size // patch_size, image_size // patch_size)
    channels = int(stage1_metadata["hidden_size"])
    return DIT_MODELS.build(
        config["dit"],
        grid_size=grid_size,
        in_channels=channels,
        frame_channels=channels,
        future_chunks=FUTURE_FRAMES // compression,
    )


def nanovvm_model_metadata(
    config: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any],
    dit: nn.Module,
    normalizer: LatentNormalizer | None = None,
) -> dict[str, Any]:
    from vrae.training.common.contracts import structural_stage1_metadata
    from vrae.training.common.latent_norm import normalizer_identity

    return {
        "stage1": structural_stage1_metadata(stage1_metadata),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "latent_normalizer_source": str(config["latent_normalizer"]["path"]),
        "dit_name": DIT_NAME,
        "dit_input_dim": int(dit.in_channels),
        "dit_frame_channels": int(dit.frame_channels),
        "dit_grid_size": list(dit.grid_size),
        "future_chunks": int(dit.future_chunks),
        "frame_protocol": frame_protocol_metadata(),
        "image_size": [_image_size(config["data"])] * 2,
        "first_frame_condition": "pre_pool_patch_features_spatial_adaln",
        **dit_architecture_metadata(dit),
        "transport": flow_transport_metadata(config),
        "latent_normalizer_identity": (
            {
                **normalizer_identity(normalizer),
                "train_sources_sha256": normalizer.metadata.get("train_sources_sha256"),
            }
            if normalizer is not None
            else None
        ),
    }


def load_task_normalizer(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    stage1_metadata: Mapping[str, Any],
    device: torch.device,
) -> LatentNormalizer:
    from vrae.training.nanovvm_video_pred.latent_stats import validate_nanovvm_normalizer

    normalizer = LatentNormalizer.load(paths.checkpoint(config["latent_normalizer"]["path"]))
    validate_nanovvm_normalizer(normalizer, config, paths, stage1_metadata=stage1_metadata)
    return normalizer.to(device)


@torch.no_grad()
def prepare_nanovvm_latents(
    stage1: Any,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB batch -> (first-frame features ``[B,h*w,C]``, raw future latents ``[B,4,h*w,C]``)."""

    condition = batch["condition"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    return encode_condition_and_future(stage1, condition, future)


def scalar_loss_metrics(losses: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in losses.items()
        if name.startswith("loss") and value.numel() == 1
    }


def rgb_unit(video: torch.Tensor) -> torch.Tensor:
    return video.float().div(255.0) if video.dtype == torch.uint8 else video.float()


def _checkpoint_payload(
    *,
    sampler: Any,
    step: int,
    dit: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    accumulation: GradientAccumulator,
    config: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    return build_training_checkpoint(
        task=TASK,
        epoch=sampler.epoch,
        step=step,
        model=dit,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_state=sampler.state_dict(gradient_accumulation_microstep=accumulation.microstep),
        resolved_config=config,
        model_metadata=model_metadata,
        rng_state=rng_state,
    )


def train(
    config: dict[str, Any],
    paths: ProjectPaths,
    run: Path,
    *,
    max_steps: int | None = None,
) -> None:
    distributed = initialize_distributed()
    seed_everything(int(config["data"].get("seed", 3407)), rank=distributed.rank)
    stage1, _ = load_frozen_stage1(config, paths, distributed.device)
    stage1_metadata = stage1.metadata()
    if not stage1.supports_single_image:
        raise ValueError("the frozen V-RAE encoder has no single-image feature path")
    dit = build_nanovvm_dit(config, stage1_metadata).to(distributed.device)
    dit.requires_grad_(True)
    model: nn.Module = dit
    if distributed.world_size > 1:
        model = DistributedDataParallel(
            dit,
            device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    normalizer = load_task_normalizer(
        config, paths, stage1_metadata=stage1_metadata, device=distributed.device
    )
    model_metadata = nanovvm_model_metadata(config, stage1_metadata, dit, normalizer)
    optimizer = build_optimizer(dit.parameters(), config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"]["scheduler"])
    ema = ExponentialMovingAverage(dit, decay=float(config["training"]["ema"]["decay"]))
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")), distributed.device.type
    )
    scaler = precision.make_scaler()
    transport = build_flow_transport(config)
    dataset = build_nanovvm_dataset(config, paths, split="train")
    if distributed.is_main:
        print(f"[data] train: {len(dataset)} clips from {len(dataset.sources)} sources", flush=True)
        for line in dataset.summary():
            print(f"[data]   {line}", flush=True)
    loader, sampler = build_nanovvm_loader(
        dataset,
        config,
        rank=distributed.rank,
        world_size=distributed.world_size,
        shuffle=True,
        drop_last=True,
    )
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    batch_contract = resolve_batch_contract(config["training"], world_size=distributed.world_size)
    if distributed.is_main:
        update_run_metadata(
            run,
            {
                "world_size": distributed.world_size,
                "local_micro_batch_size": batch_contract.local_micro_batch_size,
                "global_micro_batch_size": batch_contract.global_micro_batch_size,
                "train_sources": dataset.identity(),
            },
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    logger = WandbLogger(
        config["wandb"],
        enabled=distributed.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(config["training"].get("resume")),
    )

    start_epoch = 0
    step = 0
    metadata_fields = tuple(model_metadata)
    init_path = config["training"].get("init_from")
    resume_path = config["training"].get("resume")
    if init_path:
        load_model_init(
            paths.checkpoint(init_path),
            dit,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
        )
    if resume_path:
        payload = resume_training(
            paths.checkpoint(resume_path),
            model=dit,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch = sampler.epoch
        step = int(payload["step"])
    if max_steps is not None and step >= int(max_steps):
        logger.finish()
        return

    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    log_interval = int(config["wandb"]["log_interval"])
    sample_interval = int(config["wandb"]["sample_interval"])
    last_saved_step = step
    stop = False
    for epoch in range(start_epoch, int(config["training"].get("epochs", 1))):
        for batch in loader:
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                frame_features, raw_future = prepare_nanovvm_latents(
                    stage1, batch, device=distributed.device
                )
                future = normalizer.normalize(raw_future)
            with accumulation.sync_context(model):
                with precision.autocast():
                    losses = transport.training_losses(model, future, frame_features)
                    loss = losses["loss"]
                accumulation.backward(loss, scaler)
            sampler.commit_batch()
            if not accumulation.advance():
                continue
            if not optimizer_step(optimizer, scaler=scaler):
                continue
            scheduler.step()
            ema.update(dit)
            step += 1
            reduced_losses = None
            if step % log_interval == 0:
                reduced_losses = {
                    name: reduce_mean(value.detach())
                    for name, value in scalar_loss_metrics(losses).items()
                }
            if distributed.is_main and reduced_losses is not None:
                values = {name: float(value.item()) for name, value in reduced_losses.items()}
                print(
                    json.dumps({"task": TASK, "epoch": epoch, "step": step, **values}), flush=True
                )
                logger.log(
                    {
                        **{f"train/{name}": value for name, value in values.items()},
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_batch_size": batch_contract.global_batch_size,
                    },
                    step=step,
                )
            if distributed.is_main and step % sample_interval == 0:
                from vrae.training.nanovvm_video_pred.sample import (
                    decode_future_tokens,
                    predict_future_tokens,
                )

                count = min(int(config["wandb"].get("sample_count", 2)), frame_features.shape[0])
                indices = [int(value) for value in batch["index"][:count].tolist()]
                was_training = dit.training
                dit.eval()
                with ema.average_parameters(dit), precision.autocast():
                    predicted_tokens = predict_future_tokens(
                        dit,
                        frame_features[:count],
                        normalizer,
                        sample_indices=indices,
                        base_seed=int(config["sampling"]["base_seed"]),
                        steps=int(config["sampling"]["steps"]),
                        cfg_scale=float(config["sampling"]["cfg_scale"]),
                        internal_guidance_scale=float(
                            config["sampling"]["internal_guidance_scale"]
                        ),
                        transport=transport,
                    )
                    predicted = decode_future_tokens(
                        stage1, predicted_tokens, grid_size=tuple(dit.grid_size)
                    )
                dit.train(was_training)
                # Ground truth is the RGB clip[:, 1:17], not a V-RAE reconstruction.
                real = rgb_unit(batch["future"][:count]).to(predicted.device)
                condition = rgb_unit(batch["condition"][:count]).to(predicted.device)
                save_sample_tensors(
                    {
                        "condition_rgb": condition,
                        "real_future_rgb": real,
                        "predicted_future": predicted,
                    },
                    run / "samples" / f"step-{step:08d}.pt",
                )
                logger.log_video(
                    "samples/future16_real_vs_predicted",
                    comparison_video(real, predicted),
                    step=step,
                    fps=int(config["data"].get("fps", 10)),
                )
                logger.log_video(
                    "samples/display17_first_frame_plus_future",
                    comparison_video(
                        torch.cat((condition[:, None], real), dim=1),
                        torch.cat((condition[:, None], predicted.float().clamp(0, 1)), dim=1),
                    ),
                    step=step,
                    fps=int(config["data"].get("fps", 10)),
                )
                update_decoder_execution_metadata(
                    run, stage1.decoder_execution_metadata(), training_dtype=precision.name
                )
            if step % checkpoint_interval == 0:
                rng_state = gather_rng_states()
                if distributed.is_main:
                    save_training_checkpoint(
                        _checkpoint_payload(
                            sampler=sampler,
                            step=step,
                            dit=dit,
                            ema=ema,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            accumulation=accumulation,
                            config=config,
                            model_metadata=model_metadata,
                            rng_state=rng_state,
                        ),
                        run / "checkpoints",
                    )
                last_saved_step = step
            if max_steps is not None and step >= int(max_steps):
                stop = True
                break
        if stop:
            break
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        rng_state = gather_rng_states()
        if distributed.is_main:
            save_training_checkpoint(
                _checkpoint_payload(
                    sampler=sampler,
                    step=step,
                    dit=dit,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    accumulation=accumulation,
                    config=config,
                    model_metadata=model_metadata,
                    rng_state=rng_state,
                ),
                run / "checkpoints",
            )
    logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train first-frame conditioned NanoVVM future prediction")
    initial_config = load_config(arguments.config)
    if arguments.build_only:
        print(json.dumps(validate_build(initial_config), indent=2))
        return
    validate_build(initial_config)
    initial_paths = load_project_paths(
        initial_config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    ensure_latent_stats(initial_config, initial_paths)
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps)


if __name__ == "__main__":
    main()


__all__ = [
    "build_nanovvm_dit",
    "ensure_latent_stats",
    "load_task_normalizer",
    "nanovvm_model_metadata",
    "prepare_nanovvm_latents",
    "task_geometry",
    "train",
    "validate_build",
]
