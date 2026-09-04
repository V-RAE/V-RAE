from __future__ import annotations

import argparse
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.config import load_config, save_resolved_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


@dataclass
class TrainingState:
    epoch: int = 0
    step: int = 0
    gradient_accumulation_microstep: int = 0


def seed_everything(seed: int, *, rank: int = 0) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual % (2**32))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)


def distributed_host_memory_metrics(device: torch.device) -> dict[str, float]:
    """Collect one host-memory snapshot from every training rank."""

    from vrae.training.common.memory import host_memory_metrics

    local_metrics = host_memory_metrics()
    local_rss = float(local_metrics.get("memory/host_process_rss_gb", -1.0))
    local_available = float(local_metrics.get("memory/host_available_gb", -1.0))
    values = [(local_rss, local_available)]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = torch.tensor(
            [local_rss, local_available],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.empty_like(payload) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, payload)
        values = [(float(item[0].item()), float(item[1].item())) for item in gathered]

    rss_values = [rss for rss, _ in values if rss >= 0.0]
    available_values = [available for _, available in values if available >= 0.0]
    metrics = dict(local_metrics)
    if rss_values:
        metrics["memory/host_process_rss_world_sum_gb"] = sum(rss_values)
        metrics["memory/host_process_rss_rank_max_gb"] = max(rss_values)
    if available_values:
        metrics["memory/host_available_min_gb"] = min(available_values)
    return metrics


def resolve_runtime_gradient_checkpointing(configured: bool) -> bool:
    value = os.environ.get("VRAE_GRADIENT_CHECKPOINTING")
    if value is None:
        return bool(configured)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "VRAE_GRADIENT_CHECKPOINTING must be one of 1/0, true/false, yes/no, or on/off"
    )


def build_flow_transport(config: Mapping[str, Any]):
    from vrae.models.dit.transport import FlowMatchingTransport

    value = config.get("transport", {})
    if not isinstance(value, Mapping):
        raise TypeError("transport configuration must be a mapping")
    return FlowMatchingTransport(**dict(value))


def flow_transport_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    transport = build_flow_transport(config)
    return {
        "prediction": transport.prediction,
        "time_dist_type": transport.time_dist_type,
        "time_dist_shift": transport.time_dist_shift,
        "t_eps": transport.t_eps,
        "base_model_coeff": transport.base_model_coeff,
    }


def flatten_multiview_video(video: torch.Tensor) -> torch.Tensor:
    """Keep the training model single-view by folding [B,T,V] into [B*V,T]."""

    if video.ndim == 5:
        return video
    if video.ndim != 6:
        raise ValueError(f"Expected [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(video.shape)}")
    batch, time, views, channels, height, width = video.shape
    return (
        video.permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * views, time, channels, height, width)
        .contiguous()
    )


def _training_progress_line(
    *,
    task: str,
    epoch: int,
    step: int,
    loss: float,
    loss_full: float,
    loss_base: float,
    ms_per_step: float,
) -> str:
    return (
        f"task={task} epoch={epoch} step={step} "
        f"loss={loss:.6f} loss_full={loss_full:.6f} loss_base={loss_base:.6f} "
        f"ms_per_step={ms_per_step:.2f}"
    )


def dit_architecture_metadata(dit: nn.Module) -> dict[str, Any]:
    encoder_attention = dit.encoder_blocks[0].attn
    decoder_attention = dit.decoder_blocks[0].attn
    return {
        "dit_hidden_size": [int(dit.enc_hidden_size), int(dit.dec_hidden_size)],
        "dit_depth": [int(dit.num_enc_blocks), int(dit.num_dec_blocks)],
        "dit_num_heads": [
            int(encoder_attention.num_heads),
            int(decoder_attention.num_heads),
        ],
        "dit_patch_size": list(dit.patch_size),
        "dit_base_model_depth": int(dit.base_model_depth),
        "dit_rope_theta": [
            float(encoder_attention.rope_theta),
            float(decoder_attention.rope_theta),
        ],
        "dit_time_embedding_size": int(dit.time_embedder.frequencies.numel()),
        "dit_fourier_frequency_kind": "parameter",
        "dit_fourier_frequency_trainable_during_training": True,
    }


def parse_config_argument(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def prepare_run(
    config_path: str | Path, paths_path: str | Path | None = None
) -> tuple[dict[str, Any], ProjectPaths, Path]:
    from vrae.training.common.distributed import broadcast_object, initialize_distributed

    distributed = initialize_distributed()
    config = load_config(config_path)
    from vrae.training.common.wandb import bind_wandb_run_name, validate_wandb_config

    config["wandb"] = bind_wandb_run_name(config.get("wandb", {}), str(config["run_name"]))
    validate_wandb_config(config.get("wandb", {}))
    paths = load_project_paths(
        config,
        override=paths_path,
        project_root=find_project_root(config_path),
    )
    run = paths.training_run(str(config["task"]), str(config["run_name"]), create=False)
    root_error: Exception | None = None
    failure: dict[str, str] | None = None
    if distributed.is_main:
        try:
            from vrae.training.common.contracts import (
                compare_resolved_configs,
                create_run_metadata,
                load_run_metadata,
                run_identity,
                update_run_metadata,
                validate_checkpoint_identity,
            )
            from vrae.checkpoint import CheckpointError, load_checkpoint

            resume_value = config.get("training", {}).get("resume")
            if resume_value:
                if not run.is_dir():
                    raise CheckpointError(f"Exact-resume run directory does not exist: {run}")
                checkpoint_path = paths.checkpoint(resume_value)
                payload = load_checkpoint(checkpoint_path)
                validate_checkpoint_identity(
                    payload,
                    config,
                    checkpoint_path=checkpoint_path,
                    run_directory=run,
                )
                resolved_path = run / "resolved_config.yaml"
                if not resolved_path.is_file():
                    raise CheckpointError("Exact-resume run is missing resolved_config.yaml")
                previous_config = load_config(resolved_path)
                compare_resolved_configs(previous_config, payload["resolved_config"])
                compare_resolved_configs(previous_config, config)
                metadata = load_run_metadata(run)
                if metadata.get("run_identity") != run_identity(config):
                    raise CheckpointError("run_metadata.yaml has a different run identity")
                update_run_metadata(
                    run,
                    {
                        "last_resume_pid": os.getpid(),
                        "resume_count": int(metadata.get("resume_count", 0)) + 1,
                    },
                )
            else:
                if run.exists() and any(run.iterdir()):
                    raise FileExistsError(
                        f"Fresh run would overwrite existing artifacts: {run}. "
                        "Choose a new run_name or use training.resume."
                    )
                run.mkdir(parents=True, exist_ok=True)
                save_resolved_config(config, run / "resolved_config.yaml")
                metadata = create_run_metadata(config)
                metadata["torch_version"] = str(torch.__version__)
                save_resolved_config(metadata, run / "run_metadata.yaml")
            for directory in ("checkpoints", "samples", "wandb"):
                (run / directory).mkdir(exist_ok=True)
        except Exception as error:  # synchronize rank-zero startup failures
            root_error = error
            failure = {"type": type(error).__name__, "message": str(error)}
    failure = broadcast_object(failure)
    if failure is not None:
        if root_error is not None:
            raise root_error
        raise RuntimeError(
            f"Rank 0 failed while preparing the run: {failure['type']}: {failure['message']}"
        )
    return config, paths, run


class ClassConditionalVideoDataset:
    """Build the LeRobot dataset used by class-conditional LIBERO training."""

    @staticmethod
    def build(config: Mapping[str, Any], paths: ProjectPaths, dataset_name: str):
        data = config["data"]
        root_value = data.get("root")
        root = Path(str(root_value)).expanduser() if root_value else paths.dataset("lerobot")
        if dataset_name == "reconstruction":
            from vrae.data.reconstruction import ReconstructionDataset

            manifest = data.get("manifest")
            if not manifest:
                raise ValueError("Reconstruction generation requires data.manifest")
            manifest_path = Path(str(manifest))
            if not manifest_path.is_absolute():
                manifest_path = paths.project_root / manifest_path
            return ReconstructionDataset(
                root,
                manifest_path,
                repo_id=str(data.get("repo_id", "libero")),
                num_frames=int(data["num_frames"]),
                view_key=(str(data["view_key"]) if data.get("view_key") else None),
                camera_keys=data.get("camera_keys"),
                frame_interval=int(data.get("frame_interval", 3)),
                random_frame_interval=bool(data.get("random_frame_interval", False)),
                frame_interval_min=int(data.get("frame_interval_min", 1)),
                frame_interval_max=int(data.get("frame_interval_max", 4)),
                image_size=int(data.get("image_size", 256)),
                random_flip=bool(data.get("random_flip", True)),
                seed=int(data.get("seed", 3407)),
            )
        if dataset_name != "lerobot":
            raise ValueError(
                "class-conditional training only supports dataset_name='lerobot' or 'reconstruction'"
            )

        from vrae.data.lerobot import LeRobotVideoDataset

        return LeRobotVideoDataset(
            root,
            repo_id=str(data.get("repo_id", "libero")),
            frame_num=int(data["num_frames"]),
            frame_interval=int(data.get("frame_interval", 1)),
            camera_keys=data.get("camera_keys"),
        )


def _build_bounded_thread_loader(
    config: Mapping[str, Any],
    *,
    dataset: Any,
    batch_sampler: Any,
    rank: int,
):
    from vrae.training.common.bounded_loader import (
        AsyncPrefetchLoader,
        BoundedThreadBatchLoader,
    )

    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime configuration must be a mapping")
    pipeline = runtime.get("data_pipeline", {})
    if not isinstance(pipeline, Mapping):
        raise TypeError("runtime.data_pipeline configuration must be a mapping")
    core = BoundedThreadBatchLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        rank=rank,
        decode_threads=int(pipeline.get("worker_threads", 8)),
        max_inflight=int(pipeline.get("max_inflight", 32)),
        max_buffered_batches=int(pipeline.get("max_buffered_batches", 4)),
        max_decode_attempts_per_batch=int(
            pipeline.get("max_decode_attempts_per_batch", 2048)
        ),
        pin_memory=bool(pipeline.get("pin_memory", True)),
        glibc_arena_max=int(pipeline.get("glibc_arena_max", 2)),
        glibc_trim_threshold_bytes=int(
            pipeline.get("glibc_trim_threshold_bytes", 128 * 1024**2)
        ),
        trim_heap_each_epoch=bool(pipeline.get("trim_heap_each_epoch", True)),
        collect_python_each_epoch=bool(
            pipeline.get("collect_python_each_epoch", False)
        ),
    )
    return AsyncPrefetchLoader(
        core,
        prefetch_batches=int(pipeline.get("async_prefetch_batches", 2)),
    )


def build_class_conditional_loader(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    dataset_name: str,
    rank: int,
    world_size: int,
) -> tuple[Any, Any]:
    from vrae.training.common.contracts import resolve_batch_contract
    from vrae.training.common.sampler import StatefulDistributedBatchSampler

    dataset = ClassConditionalVideoDataset.build(config, paths, dataset_name)
    batch = resolve_batch_contract(config["training"], world_size=world_size)
    sampler = StatefulDistributedBatchSampler(
        len(dataset),
        batch.local_micro_batch_size,
        rank=rank,
        world_size=world_size,
        seed=int(config["data"].get("seed", 3407)),
        shuffle=True,
        drop_last=True,
        gradient_accumulation_steps=batch.gradient_accumulation_steps,
    )

    loader = _build_bounded_thread_loader(
        config,
        dataset=dataset,
        batch_sampler=sampler,
        rank=rank,
    )
    return loader, sampler


def load_frozen_stage1(config: Mapping[str, Any], paths: ProjectPaths, device: torch.device):
    from vrae.training.common.contracts import STAGE1_WEIGHT_COMPATIBILITY_FIELDS
    from vrae.training.common.ema import ExponentialMovingAverage
    from vrae.checkpoint import compare_metadata, load_checkpoint
    from vrae.models.adapter import VRAELatentAdapter
    from vrae.models.autoencoder import VRAE

    stage1 = VRAE.from_config(config, project_paths=paths).to(device)
    payload = load_checkpoint(paths.checkpoint(config["stage1"]["checkpoint"]))
    expected = stage1.metadata()
    fields = tuple(field for field in STAGE1_WEIGHT_COMPATIBILITY_FIELDS if field in expected)
    compare_metadata(expected, payload["model_metadata"], fields)
    stage1.load_state_dict(payload["model"], strict=True)
    weight_source = str(config["stage1"].get("weights", "ema"))
    if weight_source not in {"model", "ema"}:
        raise ValueError("stage1.weights must be model or ema")
    if weight_source == "ema":
        trainable = nn.ModuleDict(stage1.trainable_groups())
        average = ExponentialMovingAverage(trainable, decay=float(payload["ema"]["decay"]))
        average.load_state_dict(payload["ema"])
        average.copy_to(trainable)
    stage1.requires_grad_(False).eval()
    adapter_metadata = stage1.metadata()
    adapter_metadata["checkpoint_weight_source"] = weight_source
    stage1_precision = str(config["stage1"].get("precision", "fp32"))
    return VRAELatentAdapter(
        stage1,
        adapter_metadata,
        precision=stage1_precision,
    ), payload


def episode_start_indices(dataset: Any) -> list[int]:
    """Select the first valid clip index from each LeRobot episode."""

    return [int(episode["dataset_from_index"]) for episode in dataset.meta.episodes]


def compute_class_conditional_latent_stats(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    dataset_name: str,
) -> None:
    from tqdm.auto import tqdm

    from vrae.training.common.distributed import barrier, initialize_distributed
    from vrae.training.common.latent_norm import DistributedLatentStats

    context = initialize_distributed()
    stage1, stage1_checkpoint_payload = load_frozen_stage1(config, paths, context.device)
    # The model has copied all required weights. Do not retain a second full
    # checkpoint state for the lifetime of the training process.
    del stage1_checkpoint_payload
    dataset = ClassConditionalVideoDataset.build(config, paths, dataset_name)
    if dataset_name == "reconstruction":
        # ReconstructionDataset indexes one sampled clip per valid episode.
        clip_indices = list(range(len(dataset)))
    else:
        clip_indices = episode_start_indices(dataset)
    local_indices = clip_indices[context.rank :: context.world_size]
    global_batch = int(config["training"]["global_batch_size"])
    if global_batch <= 0:
        raise ValueError("training.global_batch_size must be positive")
    if global_batch % context.world_size != 0:
        raise ValueError(
            "Latent-statistics global batch must be divisible by the distributed world size: "
            f"global_batch_size={global_batch}, world_size={context.world_size}"
        )
    local_batch = global_batch // context.world_size

    class SequentialBatchSampler:
        def __len__(self) -> int:
            return (len(local_indices) + local_batch - 1) // local_batch

        def __iter__(self):
            for start in range(0, len(local_indices), local_batch):
                yield local_indices[start : start + local_batch]

    loader = _build_bounded_thread_loader(
        config,
        dataset=dataset,
        batch_sampler=SequentialBatchSampler(),
        rank=context.rank,
    )
    stats = DistributedLatentStats(int(stage1.metadata()["hidden_size"]), device=context.device)
    progress = tqdm(
        total=len(loader),
        desc=f"latent stats: {dataset_name} (global_batch={global_batch})",
        unit="batch",
        dynamic_ncols=True,
        disable=not context.is_main,
    )
    try:
        with torch.no_grad():
            for batch in loader:
                video = batch["video"].to(context.device, non_blocking=True)
                if video.dtype == torch.uint8:
                    video = video.float().div_(255.0)
                video = flatten_multiview_video(video)
                stats.update(stage1.encode_grid(video))
                progress.update(1)
    finally:
        progress.close()
        loader.close()
    from vrae.training.common.contracts import structural_stage1_metadata

    metadata = {
        "stage1": structural_stage1_metadata(stage1.metadata()),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "dataset": dataset_name,
        "split": str(config["data"].get("split", "train")),
        "scope": "one_clip_per_episode",
        "clean_latent": True,
        "normalized": False,
        "clean_latents": True,
    }
    normalizer = stats.finalize(metadata=metadata)
    if context.is_main:
        output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
        normalizer.save(output)
    barrier()


def load_class_conditional_sampling_stack(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    checkpoint: str | Path,
    device: torch.device,
    *,
    stage1_checkpoint_override: str | Path | None = None,
    latent_normalizer_override: str | Path | None = None,
):
    from vrae.training.common.ema import ExponentialMovingAverage
    from vrae.training.common.latent_norm import (
        LatentNormalizer,
        validate_normalizer_compatibility,
    )
    from vrae.checkpoint import compare_metadata, load_checkpoint

    stage1_config = dict(config)
    stage1_config["stage1"] = dict(config["stage1"])
    if stage1_checkpoint_override is not None:
        stage1_config["stage1"]["checkpoint"] = str(stage1_checkpoint_override)
    stage1, _ = load_frozen_stage1(stage1_config, paths, device)
    chunks = int(config["data"]["num_frames"]) // 4
    dit = build_class_conditional_dit(config, stage1.metadata(), num_chunks=chunks).to(device)
    normalizer_path = paths.checkpoint(
        latent_normalizer_override
        if latent_normalizer_override is not None
        else config["latent_normalizer"]["path"]
    )
    normalizer = LatentNormalizer.load(normalizer_path).to(device)
    validate_normalizer_compatibility(
        normalizer,
        stage1_metadata=stage1.metadata(),
        stage1_checkpoint=str(config["stage1"]["checkpoint"]),
        dataset=str(config["data"]["dataset"]),
        split=str(config["data"].get("split", "train")),
        scope="one_clip_per_episode",
    )
    payload = load_checkpoint(paths.checkpoint(checkpoint))
    expected_metadata = class_conditional_metadata(config, stage1.metadata(), dit, normalizer)
    compare_metadata(
        expected_metadata,
        payload["model_metadata"],
        tuple(expected_metadata),
    )
    dit.load_state_dict(payload["model"], strict=True)
    if payload.get("ema") is not None:
        average = ExponentialMovingAverage(dit, decay=float(payload["ema"]["decay"]))
        average.load_state_dict(payload["ema"])
        average.copy_to(dit)
    return stage1, dit.eval(), normalizer


def build_class_conditional_dit(
    config: Mapping[str, Any], stage1_metadata: Mapping[str, Any], *, num_chunks: int
) -> nn.Module:
    from vrae.registry import DIT_MODELS, register_builtin_models

    register_builtin_models()
    image_size = config["data"].get("image_size", 256)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    patch = int(stage1_metadata["patch_size"])
    grid = (int(image_size[0]) // patch, int(image_size[1]) // patch)
    dit_config = dict(config["dit"])
    multiview = config.get("model", {}).get("multiview", {})
    if isinstance(multiview, Mapping):
        dit_config.setdefault("multiview_enabled", bool(multiview.get("enabled", False)))
        for key in ("num_views",):
            if key in multiview:
                dit_config.setdefault(key, multiview[key])
    dit_config.pop("input_dim", None)
    return DIT_MODELS.build(
        dit_config,
        num_chunks=num_chunks,
        grid_size=grid,
        in_channels=int(stage1_metadata["hidden_size"]),
        num_classes=int(config["dit"]["num_classes"]),
    )


def class_conditional_metadata(
    config: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any],
    dit: nn.Module,
    normalizer: Any | None = None,
) -> dict[str, Any]:
    from vrae.training.common.contracts import structural_stage1_metadata
    from vrae.training.common.latent_norm import normalizer_identity

    return {
        "stage1": structural_stage1_metadata(stage1_metadata),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "latent_normalizer_source": str(config["latent_normalizer"]["path"]),
        "dit_name": "vrae_video_dit",
        "dit_input_dim": int(dit.in_channels),
        "dit_num_chunks": int(dit.num_chunks),
        "dit_grid_size": list(dit.grid_size),
        "dit_num_classes": int(dit.num_classes),
        "dit_multiview_enabled": bool(getattr(dit, "multiview_enabled", False)),
        "dit_num_views": int(getattr(dit, "num_views", 1)),
        "dit_num_streams": int(getattr(dit, "num_streams", 1)),
        **dit_architecture_metadata(dit),
        "transport": flow_transport_metadata(config),
        "latent_normalizer_identity": (
            normalizer_identity(normalizer) if normalizer is not None else None
        ),
    }


def train_class_conditional(
    config: dict[str, Any],
    paths: ProjectPaths,
    run: Path,
    *,
    dataset_name: str,
    max_steps: int | None = None,
) -> None:
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
        resolve_checkpoint_interval_steps,
        update_decoder_execution_metadata,
        update_run_metadata,
    )
    from vrae.training.common.distributed import (
        barrier,
        gather_rng_states,
        initialize_distributed,
        reduce_mean,
    )
    from vrae.training.common.ema import ExponentialMovingAverage
    from vrae.training.common.latent_norm import (
        LatentNormalizer,
        validate_normalizer_compatibility,
    )
    from vrae.training.common.optim import build_optimizer, build_scheduler
    from vrae.training.common.precision import PrecisionPolicy
    from vrae.training.common.profiler import TrainingProfiler, save_profile_json
    from vrae.training.common.visualization import save_sample_tensors, video_to_uint8
    from vrae.training.common.wandb import WandbLogger

    context = initialize_distributed()
    seed_everything(int(config["data"].get("seed", 3407)), rank=context.rank)
    stage1, stage1_checkpoint_payload = load_frozen_stage1(config, paths, context.device)
    del stage1_checkpoint_payload
    loader, sampler = build_class_conditional_loader(
        config,
        paths,
        dataset_name=dataset_name,
        rank=context.rank,
        world_size=context.world_size,
    )
    class_map_path = run / "class_map.json"
    if context.is_main and not class_map_path.is_file():
        loader.dataset.save_class_map(class_map_path)
    barrier()
    class_map = loader.dataset.load_class_map(class_map_path)
    map_num_classes = int(class_map.get("num_classes", len(class_map["class_names"])))
    if map_num_classes != len(class_map["class_names"]):
        raise ValueError("class_map.json has an inconsistent num_classes field")
    if map_num_classes != loader.dataset.num_classes:
        raise ValueError("class_map.json does not match the dataset metadata")
    if map_num_classes != int(config["dit"]["num_classes"]):
        raise ValueError("dit.num_classes must match the dataset class map")
    num_chunks = int(config["data"]["num_frames"]) // 4
    dit = build_class_conditional_dit(config, stage1.metadata(), num_chunks=num_chunks).to(
        context.device
    )
    for embedder in (dit.encoder_embed, dit.decoder_embed):
        embedder.proj.to(memory_format=torch.channels_last)
    gradient_checkpointing_configured = bool(dit.gradient_checkpointing)
    gradient_checkpointing_resolved = resolve_runtime_gradient_checkpointing(
        gradient_checkpointing_configured
    )
    dit.set_gradient_checkpointing(gradient_checkpointing_resolved)
    dit.requires_grad_(True)
    model: nn.Module = dit
    if context.world_size > 1:
        model = DistributedDataParallel(
            dit,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    normalizer = LatentNormalizer.load(paths.checkpoint(config["latent_normalizer"]["path"])).to(
        context.device
    )
    validate_normalizer_compatibility(
        normalizer,
        stage1_metadata=stage1.metadata(),
        stage1_checkpoint=str(config["stage1"]["checkpoint"]),
        dataset=dataset_name,
        split=str(config["data"].get("split", "train")),
        scope="one_clip_per_episode",
    )
    metadata = class_conditional_metadata(config, stage1.metadata(), dit, normalizer)
    optimizer = build_optimizer(dit.parameters(), config["training"]["optimizer"])
    ema = ExponentialMovingAverage(dit, decay=float(config["training"]["ema"]["decay"]))
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")), context.device.type
    )
    scaler = precision.make_scaler()
    transport = build_flow_transport(config)
    metadata["dit_class_names"] = list(class_map["class_names"])
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    steps_per_epoch = len(loader) // accumulation.steps
    checkpoint_interval = resolve_checkpoint_interval_steps(
        config["training"], steps_per_epoch=steps_per_epoch
    )
    scheduler = build_scheduler(
        optimizer,
        config["training"]["scheduler"],
        steps_per_epoch=steps_per_epoch,
    )
    batch_contract = resolve_batch_contract(config["training"], world_size=context.world_size)
    if context.is_main:
        update_run_metadata(
            run,
            {
                "world_size": context.world_size,
                "local_micro_batch_size": batch_contract.local_micro_batch_size,
                "global_micro_batch_size": batch_contract.global_micro_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "checkpoint_interval_steps": checkpoint_interval,
                "gradient_checkpointing_configured": gradient_checkpointing_configured,
                "gradient_checkpointing_resolved": gradient_checkpointing_resolved,
                "optimizer_fused": bool(optimizer.defaults.get("fused", False)),
                "ddp_gradient_as_bucket_view": context.world_size > 1,
                "ddp_static_graph": context.world_size > 1,
                "distributed_timeout_seconds": context.timeout_seconds,
                "dataloader_timeout_seconds": float(loader.timeout),
                "dataloader_prefetch_factor": loader.prefetch_factor,
                "dataloader_persistent_workers": bool(loader.persistent_workers),
                "data_pipeline_kind": str(
                    config.get("runtime", {})
                    .get("data_pipeline", {})
                    .get("kind", "lerobot_bounded")
                ),
                "data_decode_threads": int(
                    getattr(getattr(loader, "loader", None), "decode_threads", 0)
                ),
                "data_max_inflight": int(
                    getattr(getattr(loader, "loader", None), "max_inflight", 0)
                ),
                "data_max_buffered_batches": int(
                    getattr(getattr(loader, "loader", None), "max_buffered_batches", 0)
                ),
                "data_backend": "lerobot",
            },
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    start_epoch = 0
    step = 0
    init_path = config["training"].get("init_from")
    resume_path = config["training"].get("resume")
    metadata_fields = tuple(metadata)
    if init_path:
        load_model_init(
            paths.checkpoint(init_path),
            dit,
            expected_metadata=metadata,
            metadata_fields=metadata_fields,
        )
    if resume_path:
        resume_checkpoint = paths.checkpoint(resume_path)
        allow_world_size_change = os.environ.get(
            "VRAE_ALLOW_WORLD_SIZE_CHANGE_ON_RESUME", ""
        ).lower() in {"1", "true", "yes", "on"}
        if context.is_main and allow_world_size_change:
            print(
                "[resume] allowing an epoch-boundary world-size change; "
                "rank-local RNG streams will be re-seeded",
                flush=True,
            )
        payload = resume_training(
            resume_checkpoint,
            model=dit,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
            allow_world_size_change=allow_world_size_change,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch, step = sampler.epoch, int(payload["step"])
        # Optimizer/model/sampler have copied the resumed state; keeping the
        # 20+ GiB payload mapped or resident would only consume host memory.
        del payload

    logger = WandbLogger(
        config["wandb"],
        enabled=context.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(resume_path),
        resume_from_step=step if resume_path else None,
        run_config=config,
    )

    if max_steps is not None and step >= int(max_steps):
        close_loader = getattr(loader, "close", None)
        if callable(close_loader):
            close_loader()
        logger.finish()
        return

    runtime = config.get("runtime", {})
    checkpoint_runtime = runtime.get("checkpoint", {})
    host_memory_runtime = runtime.get("host_memory", {})
    profile_runtime = runtime.get("profile", {})
    checkpoint_drop_page_cache = bool(checkpoint_runtime.get("drop_page_cache", False))
    log_host_memory = bool(host_memory_runtime.get("log", False))
    min_host_available_gb = float(host_memory_runtime.get("min_available_gb", 0.0))
    if min_host_available_gb < 0.0:
        raise ValueError("runtime.host_memory.min_available_gb must be non-negative")

    profile_warmup_steps = int(profile_runtime.get("warmup_steps", 20))
    profile_interval_steps = int(
        profile_runtime.get("interval_steps", config["wandb"]["log_interval"])
    )
    if profile_warmup_steps < 0:
        raise ValueError("runtime.profile.warmup_steps must be non-negative")
    if profile_interval_steps <= 0:
        raise ValueError("runtime.profile.interval_steps must be positive")
    profile_output_path = Path(str(profile_runtime.get("output_file", "profile.json")))
    if not profile_output_path.is_absolute():
        profile_output_path = run / profile_output_path

    epochs = int(config["training"].get("epochs", 1))
    last_saved_step = step
    stop = False
    memory_guard_message: str | None = None
    profiler = TrainingProfiler(
        context.device,
        gpu_util_enabled=profile_runtime.get("gpu_util") if isinstance(profile_runtime, Mapping) else None,
        gpu_util_interval_seconds=(
            float(profile_runtime.get("gpu_util_interval_seconds", 0.2))
            if isinstance(profile_runtime, Mapping)
            else 0.2
        ),
        synchronize_phases=(
            bool(profile_runtime.get("synchronize_phases", False))
            if isinstance(profile_runtime, Mapping)
            else False
        ),
    )

    # Profile only the four large phases. Small module wrappers add overhead
    # and make the timing report harder to interpret.
    profiler.register_phase("data.loading", gpu_util=True)

    @profiler("stage1", gpu_util=True)
    def run_stage1(flat_video: torch.Tensor, *, multiview: bool, views: int):
        with torch.no_grad():
            flat_grid = stage1.encode_grid(flat_video)
            flat_tokens = stage1.grid_to_tokens(flat_grid)
            if multiview:
                batch_size, _, tokens, channels = flat_tokens.shape
                clean = flat_tokens.reshape(
                    batch_size // views,
                    views,
                    flat_tokens.shape[1],
                    tokens,
                    channels,
                ).permute(0, 2, 1, 3, 4).contiguous()
            else:
                clean = flat_tokens
            return normalizer.normalize(clean)

    @profiler("dit.forward", gpu_util=True)
    def run_dit_forward(
        clean: torch.Tensor,
        labels: torch.Tensor,
        stream_ids: torch.Tensor | None,
    ):
        with precision.autocast():
            losses = transport.training_losses(
                model,
                clean,
                model_kwargs={
                    "labels": labels,
                    **(
                        {"stream_ids": stream_ids}
                        if stream_ids is not None
                        else {}
                    ),
                },
            )
            loss = losses["loss"]
        return losses, loss

    @profiler("dit.backward", gpu_util=True)
    def run_dit_backward(loss: torch.Tensor):
        accumulation.backward(loss, scaler)

    profile_average_metrics: dict[str, float] = {}
    profile_metric_counts: dict[str, int] = {}

    def nest_profile_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Expand ``namespace/path.with.dots`` names into nested JSON objects."""

        nested: dict[str, Any] = {}
        for name, value in metrics.items():
            parts = str(name).split("/", 1)
            path = parts[0].split(".")
            if len(parts) == 2:
                path.extend(parts[1].split("."))
            target = nested
            for part in path[:-1]:
                child = target.get(part)
                if not isinstance(child, dict):
                    child = {}
                    target[part] = child
                target = child
            target[path[-1]] = value
        return nested

    def write_profile_report() -> None:
        if not context.is_main:
            return
        save_profile_json(
            {
                "version": 2,
                "run_name": str(config["run_name"]),
                "warmup_steps": profile_warmup_steps,
                "interval_steps": profile_interval_steps,
                "average_metrics": nest_profile_metrics(profile_average_metrics),
            },
            profile_output_path,
        )

    def update_profile_average(metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                count = profile_metric_counts.get(name, 0) + 1
                previous = profile_average_metrics.get(name, 0.0)
                profile_average_metrics[name] = previous + (
                    float(value) - previous
                ) / count
                profile_metric_counts[name] = count

    def profiled_batches():
        iterator = iter(loader)
        while True:
            profiler.begin_phase("data.loading")
            try:
                batch = next(iterator)
            except StopIteration:
                profiler.end_phase("data.loading")
                return
            except BaseException:
                profiler.end_phase("data.loading")
                raise
            profiler.end_phase("data.loading")
            yield batch

    reset_data_profile = getattr(loader, "profile_snapshot", None)
    if callable(reset_data_profile):
        reset_data_profile(reset=True)
    profiler.start()
    write_profile_report()
    training_started_at = time.perf_counter()
    train_window_started_at = time.perf_counter()
    train_window_updates = 0
    profile_window_started_at = time.perf_counter()
    profile_window_updates = 0
    for epoch in range(start_epoch, epochs):
        epoch_started_at = time.perf_counter()
        if context.is_main:
            print(
                f"[epoch-start] epoch={epoch} step={step} steps_per_epoch={steps_per_epoch}",
                flush=True,
            )
        set_epoch = getattr(loader.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        batch_requested_at = time.perf_counter()
        for batch in profiled_batches():
            batch_received_at = time.perf_counter()
            data_wait = batch_received_at - batch_requested_at
            profiler.record_data_wait(data_wait)
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            profiler.begin_batch()
            video = batch["video"].to(context.device, non_blocking=True)
            if video.dtype == torch.uint8:
                video = video.float().div_(255.0)
            input_has_views = video.ndim == 6
            multiview = input_has_views and bool(getattr(dit, "multiview_enabled", False))
            views = int(video.shape[2]) if input_has_views else 1
            flat_video = flatten_multiview_video(video)
            labels = batch["label"].to(context.device, non_blocking=True)
            if not multiview and views > 1:
                labels = labels.repeat_interleave(views)
            stream_ids = batch.get("stream_ids")
            if multiview and stream_ids is not None:
                stream_ids = stream_ids.to(context.device, non_blocking=True)
            else:
                stream_ids = None
            profiler.finish_h2d()
            clean = run_stage1(flat_video, multiview=multiview, views=views)
            profiler.finish_stage1()
            with accumulation.sync_context(model):
                losses, loss = run_dit_forward(clean, labels, stream_ids)
                run_dit_backward(loss)
            profiler.finish_backward()
            sampler.commit_batch()
            if not accumulation.advance():
                batch_requested_at = time.perf_counter()
                continue
            if not optimizer_step(
                optimizer,
                scaler=scaler,
                max_grad_norm=config["training"].get("clip_grad"),
                parameters=dit.parameters(),
            ):
                batch_requested_at = time.perf_counter()
                continue
            scheduler.step()
            ema.update(dit)
            profiler.finish_update()
            step += 1
            train_window_updates += 1
            if step > profile_warmup_steps:
                profile_window_updates += 1
            log_interval = int(config["wandb"]["log_interval"])
            loss_log_due = step % log_interval == 0
            profile_due = (
                step > profile_warmup_steps
                and step % profile_interval_steps == 0
            )
            if loss_log_due:
                train_window_seconds = time.perf_counter() - train_window_started_at
                reduced_losses = reduce_mean(
                    torch.stack(
                        (
                            losses["loss"],
                            losses["loss_full"],
                            losses["loss_base"],
                        )
                    ).detach()
                )
                reduced_loss, reduced_loss_full, reduced_loss_base = reduced_losses.unbind()
                train_updates_per_second = train_window_updates / max(
                    train_window_seconds, 1.0e-9
                )
                train_ms_per_step = 1000.0 / train_updates_per_second
            else:
                reduced_loss = None
                reduced_loss_full = None
                reduced_loss_base = None
                train_updates_per_second = None
                train_ms_per_step = None
            if profile_due:
                profile_window_seconds = time.perf_counter() - profile_window_started_at
                profile_metrics = profiler.summarize(
                    window_seconds=profile_window_seconds,
                    updates=profile_window_updates,
                    global_batch_size=batch_contract.global_batch_size,
                )
                data_profile = getattr(loader, "profile_snapshot", None)
                if callable(data_profile):
                    profile_metrics.update(data_profile(reset=True))
                host_metrics = (
                    distributed_host_memory_metrics(context.device)
                    if log_host_memory or min_host_available_gb > 0.0
                    else {}
                )
                profile_metrics = {**profile_metrics, **host_metrics}
                update_profile_average(profile_metrics)
                write_profile_report()
                min_available_gb = host_metrics.get("memory/host_available_min_gb")
                memory_guard_triggered = (
                    min_host_available_gb > 0.0
                    and min_available_gb is not None
                    and min_available_gb < min_host_available_gb
                )
                reset_profile_window = True
            else:
                profile_metrics = {}
                host_metrics = {}
                min_available_gb = None
                memory_guard_triggered = False
                reset_profile_window = False
            if context.is_main and reduced_loss is not None:
                print(
                    _training_progress_line(
                        task=str(config["task"]),
                        epoch=epoch,
                        step=step,
                        loss=float(reduced_loss.item()),
                        loss_full=float(reduced_loss_full.item()),
                        loss_base=float(reduced_loss_base.item()),
                        ms_per_step=train_ms_per_step,
                    ),
                    flush=True,
                )
                if profile_due:
                    print(
                        "[profile] "
                        f"window={profile_metrics['profile/window_elapsed_sec']:.2f}s "
                        f"updates={int(profile_metrics['profile/window_updates'])} "
                        f"data_wait={profile_metrics['profile/data_wait_ms_mean']:.1f}ms "
                        f"h2d={profile_metrics['profile/h2d_ms']:.1f}ms "
                        f"stage1={profile_metrics['profile/stage1_ms']:.1f}ms "
                        f"dit={profile_metrics['profile/dit_ms']:.1f}ms "
                        f"update={profile_metrics['profile/update_ms']:.1f}ms "
                        f"allocated={profile_metrics['profile/allocated_gb']:.1f}GiB "
                        f"reserved={profile_metrics['profile/reserved_gb']:.1f}GiB "
                        f"peak={profile_metrics['profile/max_allocated_gb']:.1f}GiB",
                        flush=True,
                    )
                train_metrics = {
                    "train/loss": float(reduced_loss.item()),
                    "train/loss_total": float(reduced_loss.item()),
                    "train/loss_full": float(reduced_loss_full.item()),
                    "train/loss_base": float(reduced_loss_base.item()),
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/global_batch_size": batch_contract.global_batch_size,
                }
                if train_updates_per_second is not None:
                    train_metrics.update(
                        {
                            "train/step_time_sec": 1.0 / train_updates_per_second,
                            "train/steps_per_sec": train_updates_per_second,
                        }
                    )
                logger.log(train_metrics, step=step)
            if memory_guard_triggered:
                accumulation.require_boundary()
                rng_state = gather_rng_states()
                if context.is_main:
                    memory_guard_message = (
                        "Host-memory guard stopped training before node exhaustion: "
                        f"available={float(min_available_gb):.2f} GiB, "
                        f"required={min_host_available_gb:.2f} GiB"
                    )
                    print(f"[memory-guard] {memory_guard_message}", flush=True)
                    payload = build_training_checkpoint(
                        task=str(config["task"]),
                        epoch=sampler.epoch,
                        step=step,
                        model=dit,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        data_state=sampler.state_dict(
                            gradient_accumulation_microstep=accumulation.microstep
                        ),
                        resolved_config=config,
                        model_metadata=metadata,
                        rng_state=rng_state,
                    )
                    save_training_checkpoint(
                        payload,
                        run / "checkpoints",
                        drop_page_cache=checkpoint_drop_page_cache,
                    )
                    del payload
                barrier()
                if memory_guard_message is None:
                    memory_guard_message = (
                        "Host-memory guard stopped training before node exhaustion"
                    )
                last_saved_step = step
                stop = True
                break
            sample_interval = int(config["wandb"]["sample_interval"])
            if step % sample_interval == 0:
                if context.is_main:
                    sample_started_at = time.perf_counter()
                    sample_count = int(config["wandb"].get("sample_count", 4))
                    sample_ids = [step * sample_count + index for index in range(sample_count)]
                    configured_class_ids = config["sampling"].get("class_ids")
                    if configured_class_ids:
                        class_ids = [int(value) for value in configured_class_ids]
                        sample_labels = torch.tensor(
                            [class_ids[index % len(class_ids)] for index in range(sample_count)]
                        )
                    else:
                        sample_labels = torch.arange(sample_count) % int(dit.num_classes)
                    was_training = dit.training
                    dit.eval()
                    with ema.average_parameters(dit), precision.autocast():
                        sampled_video = generate_class_conditional(
                            dit,
                            stage1,
                            normalizer,
                            transport=transport,
                            sample_ids=sample_ids,
                            labels=sample_labels,
                            base_seed=int(config["sampling"]["base_seed"]),
                            num_chunks=dit.num_chunks,
                            grid_size=tuple(dit.grid_size),
                            channels=dit.in_channels,
                            steps=int(config["sampling"]["steps"]),
                            cfg_scale=float(config["sampling"]["cfg_scale"]),
                            internal_guidance_scale=float(
                                config["sampling"]["internal_guidance_scale"]
                            ),
                            internal_guidance_t_min=float(
                                config["sampling"].get("internal_guidance_t_min", 0.0)
                            ),
                            internal_guidance_t_max=float(
                                config["sampling"].get("internal_guidance_t_max", 1.0)
                            ),
                            device=context.device,
                        )
                    dit.train(was_training)
                    save_sample_tensors(
                        {"video": sampled_video, "labels": sample_labels},
                        run / "samples" / f"step-{step:08d}.pt",
                    )
                    logger.log_video(
                        "samples",
                        video_to_uint8(sampled_video),
                        step=step,
                        fps=int(config["data"].get("fps", 8)),
                        caption=(
                            f"step={step}, cfg={float(config['sampling']['cfg_scale']):g}, "
                            f"ig={float(config['sampling']['internal_guidance_scale']):g}, "
                            f"sampling_steps={int(config['sampling']['steps'])}"
                        ),
                    )
                    update_decoder_execution_metadata(
                        run,
                        stage1.decoder_execution_metadata(),
                        training_dtype=precision.name,
                    )
                    print(
                        f"[sample] step={step} elapsed={time.perf_counter() - sample_started_at:.2f}s",
                        flush=True,
                    )
                barrier()
            if step % checkpoint_interval == 0:
                checkpoint_started_at = time.perf_counter()
                rng_state = gather_rng_states()
                if context.is_main:
                    payload = build_training_checkpoint(
                        task=str(config["task"]),
                        epoch=sampler.epoch,
                        step=step,
                        model=dit,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        data_state=sampler.state_dict(
                            gradient_accumulation_microstep=accumulation.microstep
                        ),
                        resolved_config=config,
                        model_metadata=metadata,
                        rng_state=rng_state,
                    )
                    save_training_checkpoint(
                        payload,
                        run / "checkpoints",
                        drop_page_cache=checkpoint_drop_page_cache,
                    )
                    del payload
                last_saved_step = step
                if context.is_main:
                    print(
                        f"[checkpoint] step={step} elapsed={time.perf_counter() - checkpoint_started_at:.2f}s",
                        flush=True,
                    )
            if max_steps is not None and step >= max_steps:
                stop = True
                break
            if step == profile_warmup_steps:
                profile_window_started_at = time.perf_counter()
                profile_window_updates = 0
                profiler.reset()
                if callable(reset_data_profile):
                    reset_data_profile(reset=True)
            elif reset_profile_window:
                profile_window_started_at = time.perf_counter()
                profile_window_updates = 0
                profiler.reset()
                if callable(reset_data_profile):
                    reset_data_profile(reset=True)
            if loss_log_due:
                train_window_started_at = time.perf_counter()
                train_window_updates = 0
            batch_requested_at = time.perf_counter()
        if stop:
            break
        release_epoch_memory = getattr(loader, "release_epoch_memory", None)
        if callable(release_epoch_memory):
            release_epoch_memory()
        if context.is_main:
            print(
                f"[epoch-end] epoch={epoch} elapsed={time.perf_counter() - epoch_started_at:.2f}s step={step}",
                flush=True,
            )
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        checkpoint_started_at = time.perf_counter()
        rng_state = gather_rng_states()
        if context.is_main:
            payload = build_training_checkpoint(
                task=str(config["task"]),
                epoch=sampler.epoch,
                step=step,
                model=dit,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_state=sampler.state_dict(
                    gradient_accumulation_microstep=accumulation.microstep
                ),
                resolved_config=config,
                model_metadata=metadata,
                rng_state=rng_state,
            )
            save_training_checkpoint(
                payload,
                run / "checkpoints",
                drop_page_cache=checkpoint_drop_page_cache,
            )
            del payload
        if context.is_main:
            print(
                f"[checkpoint] step={step} elapsed={time.perf_counter() - checkpoint_started_at:.2f}s",
                flush=True,
            )
    close_loader = getattr(loader, "close", None)
    if callable(close_loader):
        close_loader()
    profiler.stop()
    logger.finish()
    if context.is_main:
        print(
            f"[training-end] step={step} elapsed={time.perf_counter() - training_started_at:.2f}s",
            flush=True,
        )
    if memory_guard_message is not None:
        raise RuntimeError(memory_guard_message)


@torch.no_grad()
def generate_class_conditional(
    dit: nn.Module,
    stage1,
    normalizer,
    *,
    transport,
    sample_ids: list[int],
    labels: torch.Tensor,
    base_seed: int,
    num_chunks: int,
    grid_size: tuple[int, int],
    channels: int,
    steps: int,
    cfg_scale: float,
    internal_guidance_scale: float,
    device: torch.device,
    internal_guidance_t_min: float = 0.0,
    internal_guidance_t_max: float = 1.0,
) -> torch.Tensor:
    def sample_seed(sample_id: int) -> int:
        value = ((int(base_seed) << 32) ^ int(sample_id)) & 0xFFFFFFFFFFFFFFFF
        value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (value ^ (value >> 31)) & 0x7FFFFFFFFFFFFFFF

    samples = []
    multiview = bool(getattr(dit, "multiview_enabled", False))
    num_views = int(getattr(dit, "num_views", 1))
    sample_shape = (
        (num_chunks, num_views, grid_size[0] * grid_size[1], channels)
        if multiview
        else (num_chunks, grid_size[0] * grid_size[1], channels)
    )
    for sample_id in sample_ids:
        generator = torch.Generator(device=device).manual_seed(sample_seed(sample_id))
        samples.append(
            torch.randn(
                sample_shape,
                generator=generator,
                device=device,
            )
        )
    noise = torch.stack(samples)
    labels = labels.to(device)
    stream_ids = (
        torch.arange(num_views, device=device).expand(labels.shape[0], num_views)
        if multiview
        else None
    )
    conditional_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
    unconditional_mask = torch.ones_like(conditional_mask)
    stream_kwargs = {"stream_ids": stream_ids} if stream_ids is not None else {}
    normalized = transport.euler_sample(
        dit,
        noise,
        model_kwargs={
            "labels": labels,
            "condition_drop_mask": conditional_mask,
            **stream_kwargs,
        },
        unconditional_kwargs={
            "labels": labels,
            "condition_drop_mask": unconditional_mask,
            **stream_kwargs,
        },
        num_steps=steps,
        cfg_scale=cfg_scale,
        ig_scale=internal_guidance_scale,
        ig_intervals=((internal_guidance_t_min, internal_guidance_t_max),),
    )
    if multiview:
        # DiT keeps views explicit: [B,T,V,N,C]. The single-view V-RAE
        # adapter expects [B*V,T,N,C], so fold views before token-to-grid.
        batch, time, views, tokens, channels = normalized.shape
        flat_tokens = (
            normalized.permute(0, 2, 1, 3, 4)
            .reshape(batch * views, time, tokens, channels)
            .contiguous()
        )
        flat_tokens = normalizer.denormalize(flat_tokens)
        flat_grid = stage1.tokens_to_grid(
            flat_tokens,
            height=grid_size[0],
            width=grid_size[1],
        )
        decoded = stage1.decode_grid(flat_grid)
        return decoded.reshape(
            batch,
            views,
            decoded.shape[1],
            decoded.shape[2],
            decoded.shape[3],
            decoded.shape[4],
        ).permute(0, 2, 1, 3, 4, 5).contiguous()
    clean_tokens = normalizer.denormalize(normalized)
    clean_grid = stage1.tokens_to_grid(
        clean_tokens,
        height=grid_size[0],
        width=grid_size[1],
    )
    return stage1.decode_grid(clean_grid)
