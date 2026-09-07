from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from vrae.config import load_config, save_resolved_config
from vrae.data import LeRobotVideoDataset
from vrae.paths import find_project_root, load_project_paths
from vrae.training.common.distributed import barrier, initialize_distributed, reduce_mean
from vrae.training.common.engine import parse_config_argument
from vrae.training.common.wandb import WandbLogger, bind_wandb_run_name
from vrae.training.fastwam.profiler import StepProfiler, instrument_fastwam
from vrae.training.fastwam.sampling import fixed_sample_indices, run_sampling


def _import_factory(specification: str):
    module_name, separator, attribute = str(specification).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("model.factory must use the form 'package.module:callable'")
    return getattr(importlib.import_module(module_name), attribute)


def _torch_dtype(value: object) -> torch.dtype:
    name = str(value or "bf16").lower()
    values = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if name not in values:
        raise ValueError("training.precision must be fp32, bf16, or fp16")
    return values[name]


def validate_build(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("task") != "ci2va" or config.get("trainer") != "fastwam":
        raise ValueError("FastWAM training requires task=ci2va and trainer=fastwam")
    data = config["data"]
    if data.get("dataset") != "lerobot":
        raise ValueError("FastWAM training requires data.dataset=lerobot")
    frame_num = int(data["num_frames"])
    if frame_num <= 1 or (frame_num - 1) % 4:
        raise ValueError("data.num_frames must satisfy (num_frames - 1) % 4 == 0")
    action_horizon = int(data.get("action_horizon", frame_num - 1))
    if action_horizon != frame_num - 1:
        raise ValueError("data.action_horizon must equal data.num_frames - 1")
    if data.get("output_format", "fastwam") != "fastwam":
        raise ValueError("data.output_format must be fastwam")
    model = config.get("model", {})
    factory = model.get("factory")
    if not factory:
        raise ValueError("model.factory is required")
    factory_kwargs = model.get("factory_kwargs", {})
    if not isinstance(factory_kwargs, dict):
        raise ValueError("model.factory_kwargs must be a mapping")
    video_dit_config = factory_kwargs.get("video_dit_config", {})
    loss_config = factory_kwargs.get("loss", {})
    if not isinstance(video_dit_config, dict) or not isinstance(loss_config, dict):
        raise ValueError("model.factory_kwargs.video_dit_config and loss must be mappings")
    base_model_depth = video_dit_config.get("base_model_depth")
    base_model_coeff = float(loss_config.get("base_model_coeff", 0.0))
    if base_model_coeff < 0:
        raise ValueError("model.factory_kwargs.loss.base_model_coeff must be non-negative")
    if base_model_depth is not None:
        num_layers = int(video_dit_config.get("num_layers", 0))
        if not 1 <= int(base_model_depth) <= num_layers:
            raise ValueError("video_dit_config.base_model_depth must be within num_layers")
    if base_model_coeff > 0 and base_model_depth is None:
        raise ValueError(
            "base_model_coeff > 0 requires video_dit_config.base_model_depth"
        )
    training = config.get("training", {})
    if training.get("resume") and training.get("init_from"):
        raise ValueError("training.resume and training.init_from are mutually exclusive")
    cameras = data.get("camera_keys")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("data.camera_keys must be a non-empty list")
    sampling = config.get("sampling", {})
    if bool(sampling.get("enabled", False)):
        if int(sampling.get("interval_steps", 0)) <= 0:
            raise ValueError("sampling.interval_steps must be positive")
        if int(sampling.get("num_samples", 0)) <= 0:
            raise ValueError("sampling.num_samples must be positive")
        if int(sampling.get("num_video_frames", 0)) != frame_num:
            raise ValueError("sampling.num_video_frames must equal data.num_frames")
        if int(sampling.get("action_horizon", 0)) != action_horizon:
            raise ValueError("sampling.action_horizon must equal data.action_horizon")
        layout = sampling.get("action_layout", {})
        required_layout = {
            "mode",
            "position_indices",
            "orientation_indices",
            "gripper_index",
            "state_position_indices",
        }
        if not isinstance(layout, dict) or not required_layout.issubset(layout):
            raise ValueError("sampling.action_layout is incomplete")
    return {
        "task": "ci2va",
        "trainer": "fastwam",
        "frame_num": frame_num,
        "action_horizon": action_horizon,
        "num_cameras": len(cameras),
        "model_factory": str(factory),
        "base_model_depth": base_model_depth,
        "base_model_coeff": base_model_coeff,
        "checkpoint_mode": (
            "resume"
            if training.get("resume")
            else "init_from"
            if training.get("init_from")
            else "fresh"
        ),
        "wandb_enabled": bool(config.get("wandb", {}).get("enabled", True)),
    }


def _build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    model_config = config["model"]
    factory = _import_factory(str(model_config["factory"]))
    kwargs = dict(model_config.get("factory_kwargs", {}))
    kwargs.setdefault("device", str(device))
    kwargs.setdefault("model_dtype", _torch_dtype(config["training"].get("precision", "bf16")))
    model = factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("model.factory must return torch.nn.Module")
    return model.to(device)


def _build_dataset(config: dict[str, Any], paths) -> LeRobotVideoDataset:
    data = config["data"]
    root = Path(str(data["root"])) if data.get("root") else paths.dataset("lerobot")
    if not root.is_absolute():
        root = paths.project_root / root
    cache_dir = None
    if data.get("text_embedding_cache_dir"):
        cache_dir = Path(str(data["text_embedding_cache_dir"]))
        if not cache_dir.is_absolute():
            cache_dir = paths.project_root / cache_dir
    return LeRobotVideoDataset(
        root,
        repo_id=str(data.get("repo_id", "libero")),
        frame_num=int(data["num_frames"]),
        frame_interval=int(data.get("frame_interval", 1)),
        camera_keys=data["camera_keys"],
        action_horizon=int(data.get("action_horizon", int(data["num_frames"]) - 1)),
        output_format="fastwam",
        text_embedding_cache_dir=cache_dir,
        context_len=int(data.get("context_len", 128)),
    )


def _target_steps(config: dict[str, Any], max_steps: int | None, num_batches: int) -> int:
    if max_steps is not None:
        return int(max_steps)
    configured_max_steps = config["training"].get("max_steps")
    if configured_max_steps is not None:
        return int(configured_max_steps)
    return int(config["training"].get("epochs", 1)) * max(num_batches, 1)


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run: Path,
    step: int,
    training_state: dict[str, Any] | None = None,
) -> Path:
    run.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    path = run / "checkpoints" / f"step-{step:08d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    if not hasattr(unwrapped, "save_checkpoint"):
        raise TypeError("FastWAM model must provide save_checkpoint(path, optimizer, step)")
    unwrapped.save_checkpoint(
        str(temporary),
        optimizer=optimizer,
        step=step,
        training_state=training_state,
    )
    temporary.replace(path)
    latest = run / "checkpoints" / "latest.json"
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"path": path.name, "step": int(step)}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(latest)
    return path


def _resolve_resume_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if path.suffix == ".json":
        pointer_path = path
        pointer = json.loads(path.read_text(encoding="utf-8"))
        path = Path(str(pointer["path"]))
        if not path.is_absolute():
            path = pointer_path.parent / path
    return path


def _load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> int:
    if not hasattr(model, "load_checkpoint"):
        raise TypeError("FastWAM model must provide load_checkpoint(path, optimizer)")
    payload = model.load_checkpoint(str(path), optimizer=optimizer)
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError("FastWAM resume checkpoint must contain `mot`")
    if "optimizer" not in payload:
        raise ValueError("FastWAM resume checkpoint must contain `optimizer`")
    if payload.get("step") is None:
        raise ValueError("FastWAM resume checkpoint must contain `step`")
    return int(payload["step"])


def _load_weights(model: nn.Module, path: Path) -> int | None:
    if not hasattr(model, "load_checkpoint"):
        raise TypeError("FastWAM model must provide load_checkpoint(path, optimizer=None)")
    payload = model.load_checkpoint(str(path), optimizer=None)
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError("FastWAM init_from checkpoint must contain `mot`")
    step = payload.get("step")
    return None if step is None else int(step)


def _prepare_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    trainable = list(model.dit.parameters())
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)
        trainable.extend(proprio_encoder.parameters())
    return trainable


def _loss_values_for_logging(
    loss: torch.Tensor,
    metrics: dict[str, Any],
) -> torch.Tensor:
    def scalar(name: str, fallback: object) -> torch.Tensor:
        value = metrics.get(name, fallback)
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=loss.device, dtype=torch.float32).reshape(())
        return torch.as_tensor(value, device=loss.device, dtype=torch.float32).reshape(())

    return torch.stack(
        (
            loss.detach().float(),
            scalar("loss_video_full", metrics.get("loss_video", 0.0)),
            scalar("loss_video_base", 0.0),
            scalar("loss_action", 0.0),
        )
    )


def _format_timings(values: dict[str, float]) -> str:
    order = (
        "data_wait_ms",
        "data_wait_p95_ms",
        "vae_encode_ms",
        "denoise_forward_ms",
        "forward_other_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
        "data_wait_pct",
        "checkpoint_ms",
    )
    return " ".join(f"{name}={values[name]:.1f}" for name in order if name in values)


def train(config: dict[str, Any], paths, run: Path, max_steps: int | None) -> None:
    context = initialize_distributed()
    dataset = _build_dataset(config, paths)
    sampler = DistributedSampler(dataset, shuffle=True) if context.world_size > 1 else None
    num_workers = int(config["training"].get("num_workers", 4))
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = int(
            config["training"].get("prefetch_factor", 2)
        )
        loader_options["persistent_workers"] = bool(
            config["training"].get("persistent_workers", False)
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=context.device.type == "cuda",
        **loader_options,
    )
    model = _build_model(config, context.device)
    trainable = _prepare_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["optimizer"].get("lr", 1.0e-4)),
        weight_decay=float(config["training"]["optimizer"].get("weight_decay", 0.0)),
        betas=tuple(config["training"]["optimizer"].get("betas", [0.9, 0.95])),
    )
    step = 0
    initialization: dict[str, Any] | None = None
    resume = config["training"].get("resume")
    init_from = config["training"].get("init_from")
    if resume and init_from:
        raise ValueError("training.resume and training.init_from are mutually exclusive")
    if resume:
        resume_path = _resolve_resume_path(str(resume), paths.project_root)
        step = _load_checkpoint(model, optimizer, resume_path)
        initialization = {
            "mode": "resume",
            "checkpoint": str(resume_path),
            "source_step": step,
        }
        if context.is_main:
            print(
                f"Resumed FastWAM checkpoint path={resume_path} step={step}; "
                "starting a new shuffled data epoch.",
                flush=True,
            )
    elif init_from:
        init_path = _resolve_resume_path(str(init_from), paths.project_root)
        source_step = _load_weights(model, init_path)
        initialization = {
            "mode": "init_from",
            "checkpoint": str(init_path),
            "source_step": source_step,
        }
        if context.is_main:
            print(
                f"Initialized FastWAM weights path={init_path} "
                f"source_step={source_step}; optimizer and training step start fresh.",
                flush=True,
            )
    profiler = StepProfiler(
        enabled=bool(config["training"].get("profile", True)),
        device=context.device,
        window_steps=int(config["training"].get("profile_window_steps", 200)),
    )
    instrument_fastwam(model, profiler)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
    precision = str(config["training"].get("precision", "bf16"))
    autocast_enabled = context.device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = _torch_dtype(precision)
    target_steps = _target_steps(config, max_steps, len(loader))
    run.mkdir(parents=True, exist_ok=True)
    run_config = dict(config)
    if initialization is not None:
        run_config["initialization"] = initialization
    if context.is_main:
        save_resolved_config(run_config, run / "resolved_config.yaml")
        run.joinpath("checkpoints").mkdir(exist_ok=True)
        build_report = validate_build(config)
        if initialization is not None:
            build_report["initialization"] = initialization
        (run / "build.json").write_text(
            json.dumps(build_report, indent=2) + "\n",
            encoding="utf-8",
        )
    sampling_config = dict(config.get("sampling", {}))
    sampling_enabled = bool(sampling_config.get("enabled", False))
    sample_interval = int(sampling_config.get("interval_steps", 5000))
    sample_indices = (
        fixed_sample_indices(
            len(dataset),
            int(sampling_config.get("num_samples", 4)),
            int(sampling_config.get("seed", 3407)),
        )
        if context.is_main and sampling_enabled
        else []
    )
    sample_dir = Path(
        str(sampling_config.get("local_dir", run / "samples")).replace(
            "${run_name}",
            str(config["run_name"]),
        )
    )
    if not sample_dir.is_absolute():
        sample_dir = paths.project_root / sample_dir
    wandb_config = bind_wandb_run_name(config.get("wandb", {}), str(config["run_name"]))
    wandb_logger = WandbLogger(
        wandb_config,
        enabled=context.is_main and bool(wandb_config.get("enabled", True)),
        run_dir=str(run),
        run_config=run_config,
    )
    batches_per_epoch = max(len(loader), 1)
    start_epoch = math.ceil(step / batches_per_epoch)
    remaining_steps = max(target_steps - step, 0)
    stop_epoch = start_epoch + math.ceil(remaining_steps / batches_per_epoch)
    log_interval = int(config["training"].get("log_interval", 20))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 5000))
    last_checkpoint_step: int | None = None
    initial_step = step
    training_state = {"epoch": start_epoch, "batch_in_epoch": 0}
    previous_step_finished_at = time.perf_counter()
    for epoch in range(start_epoch, stop_epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            step_started_at = previous_step_finished_at
            batch_ready_at = time.perf_counter()
            profiler.record_wall(
                "data_wait_ms",
                (batch_ready_at - previous_step_finished_at) * 1000.0,
            )
            with profiler.phase("forward_ms"):
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    loss, metrics = model(batch)
            with profiler.phase("backward_ms"):
                loss.backward()
            clip_grad = float(config["training"].get("clip_grad", 1.0))
            with profiler.phase("optimizer_ms"):
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            training_state = {
                "epoch": epoch,
                "batch_in_epoch": batch_index + 1,
            }
            previous_step_finished_at = time.perf_counter()
            profiler.record_wall(
                "step_ms",
                (previous_step_finished_at - step_started_at) * 1000.0,
            )
            if step % log_interval == 0:
                reduced = reduce_mean(_loss_values_for_logging(loss, metrics))
                (
                    reduced_loss,
                    reduced_video_full,
                    reduced_video_base,
                    reduced_action,
                ) = reduced.tolist()
                timings = profiler.averages()
                if context.is_main:
                    if "forward_ms" in timings:
                        timings["forward_other_ms"] = max(
                            0.0,
                            timings.pop("forward_ms")
                            - timings.get("vae_encode_ms", 0.0)
                            - timings.get("denoise_forward_ms", 0.0),
                        )
                    if timings.get("step_ms", 0.0) > 0:
                        timings["data_wait_pct"] = (
                            100.0
                            * timings.get("data_wait_ms", 0.0)
                            / timings["step_ms"]
                        )
                    print(
                        f"task=ci2va step={step} loss={reduced_loss:.6f} "
                        f"loss_video_full={reduced_video_full:.6f} "
                        f"loss_video_base={reduced_video_base:.6f} "
                        f"loss_action={reduced_action:.6f} "
                        f"{_format_timings(timings)}",
                        flush=True,
                    )
                    wandb_logger.log(
                        {
                            "train/loss": reduced_loss,
                            "train/loss_video_full": reduced_video_full,
                            "train/loss_video_base": reduced_video_base,
                            "train/loss_action": reduced_action,
                            **{f"timing/{name}": value for name, value in timings.items()},
                        },
                        step=step,
                    )
            if step % checkpoint_interval == 0:
                barrier()
                checkpoint_started_at = time.perf_counter()
                if context.is_main:
                    _save_checkpoint(
                        model,
                        optimizer,
                        run,
                        step,
                        training_state=training_state,
                    )
                barrier()
                checkpoint_ms = (time.perf_counter() - checkpoint_started_at) * 1000.0
                profiler.record_wall("checkpoint_ms", checkpoint_ms)
                if context.is_main:
                    print(
                        f"task=ci2va step={step} checkpoint_ms={checkpoint_ms:.1f}",
                        flush=True,
                    )
                    wandb_logger.log(
                        {"timing/checkpoint_ms": checkpoint_ms},
                        step=step,
                    )
                last_checkpoint_step = step
                previous_step_finished_at = time.perf_counter()
            if sampling_enabled and step % sample_interval == 0:
                barrier()
                if context.is_main:
                    try:
                        sample_ms, upload_ms = run_sampling(
                            model.module if isinstance(model, DistributedDataParallel) else model,
                            dataset,
                            sample_indices,
                            sampling_config,
                            sample_dir,
                            step,
                            wandb_logger,
                        )
                        profiler.record_wall("sample_ms", sample_ms)
                        profiler.record_wall("wandb_upload_ms", upload_ms)
                        print(
                            f"task=ci2va step={step} sample_ms={sample_ms:.1f} "
                            f"wandb_upload_ms={upload_ms:.1f}",
                            flush=True,
                        )
                    except (ValueError, KeyError, TypeError):
                        raise
                    except Exception as error:
                        print(f"FastWAM sample failed at step={step}: {error}", flush=True)
                barrier()
                previous_step_finished_at = time.perf_counter()
            if step >= target_steps:
                break
        if step >= target_steps:
            break
    if step > initial_step and step != last_checkpoint_step:
        barrier()
        if context.is_main:
            _save_checkpoint(
                model,
                optimizer,
                run,
                step,
                training_state=training_state,
            )
        barrier()
    wandb_logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train FastWAM MoT on LeRobot ci2va")
    config = load_config(arguments.config)
    validation = validate_build(config)
    if arguments.build_only:
        print(json.dumps(validation, indent=2))
        return
    paths = load_project_paths(
        config,
        override=arguments.paths,
        project_root=find_project_root(arguments.config),
    )
    run = paths.training_run("fastwam", str(config["run_name"]), create=True)
    train(config, paths, run, arguments.max_steps)


if __name__ == "__main__":
    main()
