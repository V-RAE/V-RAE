from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.training.common.accumulation import GradientAccumulator, optimizer_step
from vrae.training.common.checkpoint import (
    build_training_checkpoint,
    load_model_init,
    resume_training,
    save_training_checkpoint,
)
from vrae.training.common.contracts import (
    STAGE1_STRUCTURE_FIELDS,
    STAGE1_WEIGHT_COMPATIBILITY_FIELDS,
    load_run_metadata,
    resolve_batch_contract,
    update_decoder_execution_metadata,
    update_run_metadata,
    validate_checkpoint_identity,
)
from vrae.training.common.distributed import (
    barrier,
    configure_ddp_gradient_compression,
    gather_rng_states,
    initialize_distributed,
    reduce_mean,
    resolve_ddp_gradient_compression,
)
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import parse_config_argument, prepare_run, seed_everything
from vrae.training.common.optim import build_optimizer, build_scheduler
from vrae.training.common.precision import PrecisionPolicy
from vrae.training.common.visualization import comparison_video
from vrae.training.common.wandb import WandbLogger
from vrae.training.recon_training.data import (
    CudaVideoPrefetchIterator,
    build_reconstruction_loader,
    flatten_multiview_video,
)
from vrae.training.recon_training.losses import ReconstructionLoss
from vrae.training.recon_training.noise import add_reconstruction_noise
from vrae.training.recon_training.sample import save_reconstruction_sample
from vrae.checkpoint import CheckpointError, compare_metadata, load_checkpoint
from vrae.config import load_config
from vrae.models.autoencoder import VRAE


class ReconstructionGraph(nn.Module):
    def __init__(self, vrae: VRAE, noise_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.vrae = vrae
        self.noise_config = dict(noise_config)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        clean = self.vrae.encode(video)
        train_latents = clean
        sigma = clean.new_zeros((*clean.shape[:2], *((1,) * (clean.ndim - 2))))
        if self.training and bool(self.noise_config.get("enabled", False)):
            train_latents, sigma = add_reconstruction_noise(
                clean, float(self.noise_config.get("tau", 0.0))
            )
        return {"recon": self.vrae.decode(train_latents), "latents": clean, "sigma": sigma}


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    training = config["training"]
    if str(config.get("task")) != "recon_training":
        raise ValueError("V-JEPA 2.1 entry requires task=recon_training")
    if str(config["model"]["encoder"]["name"]) != "vjepa2_1":
        raise ValueError("This training entry only supports the V-JEPA 2.1 encoder")
    if str(config["data"].get("dataset")) != "lerobot":
        raise ValueError("V-JEPA 2.1 reconstruction requires data.dataset=lerobot")
    if str(training["scheduler"]["name"]) != "constant":
        raise ValueError("The formal reconstruction scheduler must be constant")
    if float(training["optimizer"]["lr"]) <= 0:
        raise ValueError("The reconstruction learning rate must be positive")
    gradient_compression = resolve_ddp_gradient_compression(
        training.get("ddp_gradient_compression", "none")
    )
    init_weights = str(training.get("init_weights", "model"))
    if init_weights not in {"model", "ema"}:
        raise ValueError("training.init_weights must be model or ema")
    if init_weights == "ema" and not training.get("init_from") and not training.get("resume"):
        raise ValueError("training.init_weights=ema requires init_from or resume")
    return {
        "task": config["task"],
        "encoder": config["model"]["encoder"]["name"],
        "pool_group": config["model"]["pooling"]["group_size"],
        "attention_mode": config["model"]["decoder"]["attention_mode"],
        "dataset": "lerobot",
        "ddp_gradient_compression": gradient_compression,
    }


def _resolve_loss_config(config: Mapping[str, Any], paths) -> dict[str, Any]:
    value = dict(config["loss"])
    for key in ("backbone_checkpoint", "calibration_checkpoint"):
        if key in value:
            value[key] = str(paths.checkpoint(value[key]))
    return value


@torch.no_grad()
def _copy_checkpoint_ema_to_trainable(
    trainable: nn.Module,
    ema_state: object,
) -> None:
    if not isinstance(ema_state, Mapping):
        raise CheckpointError("Initial checkpoint does not contain EMA state")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, Mapping):
        raise CheckpointError("Initial checkpoint EMA state does not contain a shadow mapping")
    target = trainable.state_dict()
    expected = {name for name, value in target.items() if torch.is_floating_point(value)}
    if set(shadow) != expected:
        missing = sorted(expected.difference(shadow))
        unexpected = sorted(set(shadow).difference(expected))
        raise CheckpointError(
            "Initial checkpoint EMA keys do not match trainable V-RAE modules: "
            f"missing={missing} unexpected={unexpected}"
        )
    for name, average in shadow.items():
        if not torch.is_tensor(average) or average.shape != target[name].shape:
            raise CheckpointError(f"Invalid initial checkpoint EMA tensor: {name}")
        target[name].copy_(average.to(device=target[name].device, dtype=target[name].dtype))


def _initialize_model(
    config: Mapping[str, Any],
    paths,
    model: VRAE,
    trainable: nn.Module,
    *,
    model_metadata: Mapping[str, Any],
    weight_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    init_value = config["training"].get("init_from")
    if not init_value:
        return None
    init_path = paths.checkpoint(init_value)
    payload = load_model_init(
        init_path,
        model,
        expected_metadata=model_metadata,
        metadata_fields=weight_fields,
    )
    if str(payload.get("task")) != "recon_training":
        raise CheckpointError(
            f"V-RAE initialization requires task='recon_training', got {payload.get('task')!r}"
        )
    weight_source = str(config["training"].get("init_weights", "model"))
    if weight_source not in {"model", "ema"}:
        raise ValueError("training.init_weights must be model or ema")
    if weight_source == "ema":
        _copy_checkpoint_ema_to_trainable(trainable, payload.get("ema"))
    summary = {
        "checkpoint": str(init_path.resolve()),
        "weights": weight_source,
        "source_epoch": int(payload["epoch"]),
        "source_step": int(payload["step"]),
    }
    del payload
    gc.collect()
    return summary


def train(
    config: dict[str, Any],
    paths,
    run: Path,
    *,
    max_steps: int | None = None,
    loader_builder=build_reconstruction_loader,
    batch_preparer: Callable[[Mapping[str, Any], torch.device, Mapping[str, Any]], torch.Tensor]
    | None = None,
) -> None:
    # 阶段 1/3：初始化分布式环境、随机种子和计算设备
    context = initialize_distributed()
    # context 至少包含：
    #   rank        当前进程编号
    #   world_size  进程总数
    #   local_rank  当前进程在本机上的编号
    #   device      例如 cuda:0 或 cpu
    #   is_main     是否为主进程
    seed_everything(int(config["data"].get("seed", 3407)), rank=context.rank)
    if context.device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", False))
     # 阶段 2/3：创建 VRAE 模型并移动到当前设备
    model = VRAE.from_config(config, project_paths=paths).to(context.device)
    # 可选：编译 encoder
    if bool(config["training"].get("compile_encoder", False)):
        model.encoder.enable_compile(
            mode=str(config["training"].get("compile_encoder_mode", "default")),
            fullgraph=bool(config["training"].get("compile_encoder_fullgraph", False)),
            dynamic=False,
        )
    # 可选：编译 decoder
    if bool(config["training"].get("compile_decoder", False)):
        compile_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[str(config["training"].get("precision", "bf16")).lower()]
        model.decoder.prime_rope_cache(
            num_frames=int(config["data"]["num_frames"]),
            device=context.device,
            dtype=compile_dtype,
        )
        model.decoder.enable_compile(
            mode=str(config["training"].get("compile_decoder_mode", "default")),
            fullgraph=bool(config["training"].get("compile_decoder_fullgraph", False)),
            dynamic=False,
        )
    model_metadata = model.metadata()
    metadata_fields = tuple(field for field in STAGE1_STRUCTURE_FIELDS if field in model_metadata)
    weight_fields = tuple(
        field for field in STAGE1_WEIGHT_COMPATIBILITY_FIELDS if field in model_metadata
    )
    # 只把模型中允许训练的模块注册进 ModuleDict
    trainable = nn.ModuleDict(model.trainable_groups())
    # 可能执行预训练权重加载、模块初始化或冻结参数等操作(check)
    initialization = _initialize_model(
        config,
        paths,
        model,
        trainable,
        model_metadata=model_metadata,
        weight_fields=weight_fields,
    )
    graph: nn.Module = ReconstructionGraph(model, config["training"]["latent_noise"]).to(
        context.device
    )
    # 多卡训练：DDP 包装的是 graph
    if context.world_size > 1:
        graph = DistributedDataParallel(
            graph,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None, 
            broadcast_buffers=False,
            find_unused_parameters=False,
            bucket_cap_mb=config["training"].get("ddp_bucket_cap_mb"),
            bucket_cap_mb_list=config["training"].get("ddp_bucket_cap_mb_list"),
            gradient_as_bucket_view=bool(
                config["training"].get("ddp_gradient_as_bucket_view", False)
            ),
            static_graph=bool(config["training"].get("ddp_static_graph", False)),
            batched_grad_copy=bool(config["training"].get("ddp_batched_grad_copy", False)),
        )
    ddp_gradient_compression = configure_ddp_gradient_compression(
        graph,
        config["training"].get("ddp_gradient_compression", "none"),
    )
    # 优化器和学习率调度器只接收 trainable 中的参数
    optimizer = build_optimizer(trainable.parameters(), config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"]["scheduler"])
    # EMA 保存一份参数的指数移动平均版本，用于更稳定的采样和评估
    ema = ExponentialMovingAverage(trainable, decay=float(config["training"]["ema"]["decay"]))
    # 根据配置创建混合精度策略
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")), context.device.type
    )
    scaler = precision.make_scaler()
    # (check)
    loss_module = ReconstructionLoss(_resolve_loss_config(config, paths)).to(context.device)
    if bool(config["training"].get("compile_reconstruction_loss", False)):
        loss_module.enable_compile(
            mode=str(config["training"].get("compile_reconstruction_loss_mode", "default")),
            fullgraph=bool(config["training"].get("compile_reconstruction_loss_fullgraph", False)),
            dynamic=False,
        )
    # 数据加载、类别校验与断点恢复
    # 阶段 1/3：构建 DataLoader 和分布式 sampler
    loader, sampler = loader_builder(
        config, paths, rank=context.rank, world_size=context.world_size
    )
    class_map_path = run / "class_map.json"
    if context.is_main and not class_map_path.is_file():
        loader.dataset.save_class_map(class_map_path)
    # 等待主进程写完文件，所有进程再读取
    barrier()
    class_map = loader.dataset.load_class_map(class_map_path)
    map_num_classes = int(class_map.get("num_classes", len(class_map["class_names"])))
    if map_num_classes != len(class_map["class_names"]):
        raise ValueError("class_map.json has an inconsistent num_classes field")
    if map_num_classes != loader.dataset.num_classes:
        raise ValueError("class_map.json does not match the dataset metadata")
    prepare_batch = batch_preparer or (
        lambda batch, device, _config: batch["video"].to(device, non_blocking=True)
    )
    # # 梯度累积，例如 accumulation_steps=4：
    # 4 个 micro-batch 才执行一次 optimizer.step() (check)
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    batch_contract = resolve_batch_contract(config["training"], world_size=context.world_size)
    if context.is_main:
        run_updates: dict[str, Any] = {
            "world_size": context.world_size,
            "local_micro_batch_size": batch_contract.local_micro_batch_size,
            "global_micro_batch_size": batch_contract.global_micro_batch_size,
            "ddp_static_graph": bool(config["training"].get("ddp_static_graph", False)),
            "ddp_gradient_as_bucket_view": bool(
                config["training"].get("ddp_gradient_as_bucket_view", False)
            ),
            "ddp_gradient_compression": ddp_gradient_compression,
            "ddp_batched_grad_copy": bool(config["training"].get("ddp_batched_grad_copy", False)),
            "ddp_bucket_cap_mb": config["training"].get("ddp_bucket_cap_mb"),
            "ddp_bucket_cap_mb_list": config["training"].get("ddp_bucket_cap_mb_list"),
            "optimizer_fused": bool(optimizer.defaults.get("fused", False)),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "distributed_timeout_seconds": context.timeout_seconds,
            "dataloader_num_workers": int(config["training"].get("num_workers", 4)),
            "dataloader_prefetch_factor": loader.prefetch_factor,
            "dataloader_persistent_workers": bool(loader.persistent_workers),
        }
        if initialization is not None:
            run_updates["initialization"] = initialization
        update_run_metadata(
            run,
            run_updates,
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    logger = WandbLogger(
        config["wandb"],
        enabled=context.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(config["training"].get("resume")),
    )
    # 阶段 2/3：初始化训练位置
    start_epoch = 0
    step = 0
    resume_path = config["training"].get("resume")
    if resume_path:
        resume_checkpoint = paths.checkpoint(resume_path)
        preview = load_checkpoint(resume_checkpoint)
        validate_checkpoint_identity(
            preview,
            config,
            checkpoint_path=resume_checkpoint,
            run_directory=run,
        )
        compare_metadata(
            model_metadata,
            preview["model_metadata"],
            metadata_fields,
        )
        optimizer_objects: dict[str, Any] = {"generator": optimizer}
        scheduler_objects: dict[str, Any] = {"generator": scheduler}
        # 恢复模型、EMA、优化器、scheduler、scaler、sampler 等状态(check)
        payload = resume_training(
            resume_checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer_objects,
            scheduler=scheduler_objects,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch, step = sampler.epoch, int(payload["step"])

    if max_steps is not None and step >= int(max_steps):
        logger.finish()
        return
    # 阈值配置
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    log_interval = int(config["wandb"]["log_interval"])
    sample_interval = int(config["wandb"]["sample_interval"])
    # 固定一批样本，用于不同训练阶段的重建结果对比
    fixed_input_path = run / "samples" / "fixed_input.pt"
    fixed_video: torch.Tensor | None = None
    if context.is_main and fixed_input_path.is_file():
        fixed_payload = torch.load(fixed_input_path, map_location="cpu", weights_only=True)
        fixed_video = fixed_payload["video"]
    last_saved_step = step
    throughput_started = time.perf_counter()
    throughput_start_step = step
    execution_metadata_recorded = False
    stop = False
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        batches = iter(loader)
        # 仅在默认 batch 预处理、CUDA 和显式开启时启用预取
        if (
            batch_preparer is None
            and context.device.type == "cuda"
            and bool(config["training"].get("prefetch_to_device", False))
        ):
            batches = CudaVideoPrefetchIterator(batches, context.device)
        for batch in batches:
            # 一个梯度累积周期的第一个 micro-step 才清空梯度
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            # 输入处理：(check)
            # 默认是 batch["video"] -> 当前 device
            video = prepare_batch(batch, context.device, config)
            # 可能的形状变化：(check)
            #   [B, V, ...] -> [B*V, ...]
            # 其中 V 是多视角数量。
            # 如果输入本来没有多视角维度，通常保持原形状。
            video = flatten_multiview_video(video)
            # 第一次拿到真实 batch 时，保存固定输入样本
            if context.is_main and fixed_video is None:
                sample_count = min(int(config["wandb"].get("sample_count", 4)), video.shape[0])
                fixed_video = video[:sample_count].detach().cpu()
                save_reconstruction_sample({"video": fixed_video}, fixed_input_path)
            if any(
                bool(config["training"].get(enabled, False))
                and str(config["training"].get(mode, "default")) == "reduce-overhead"
                for enabled, mode in (
                    ("compile_encoder", "compile_encoder_mode"),
                    ("compile_decoder", "compile_decoder_mode"),
                    ("compile_reconstruction_loss", "compile_reconstruction_loss_mode"),
                )
            ):
                torch.compiler.cudagraph_mark_step_begin()
            # 阶段 2/3：前向、损失和反向传播
            with accumulation.sync_context(graph):
                # 非最终 micro-step 可能使用 DDP no_sync，
                # 避免每个 micro-step 都进行梯度通信
                with precision.autocast():
                    result = graph(video)
                    # result 至少包含 result["recon"]
                    total, metrics = loss_module(result["recon"], video)
                    # total：用于反向传播的标量总损失
                    # metrics：各项损失或监控指标的字典
                accumulation.backward(total, scaler)
            sampler.commit_batch()
            # 尚未积累够梯度时，继续读取下一个 micro-batch
            if not accumulation.advance():
                continue
            if not optimizer_step(optimizer, scaler=scaler):
                continue
            scheduler.step()
            ema.update(trainable)
            # step 是 optimizer update 次数，不是 micro-batch 数
            step += 1
            # 第一次成功更新后记录 decoder 执行信息
            if context.is_main and not execution_metadata_recorded:
                update_decoder_execution_metadata(
                    run,
                    model.decoder.execution_metadata(),
                    training_dtype=precision.name,
                )
                execution_metadata_recorded = True
            # 阶段 3/3：定期聚合和记录指标
            if step % log_interval == 0:
                metric_tensors = {**metrics, "total": total.detach()}
                reduced_metrics = {
                    name: reduce_mean(value.detach()) for name, value in metric_tensors.items()
                }
                throughput_finished = time.perf_counter()
                measured_steps = step - throughput_start_step
                samples_per_second = (
                    measured_steps
                    * batch_contract.global_batch_size
                    / max(throughput_finished - throughput_started, 1.0e-9)
                )
                throughput_started = throughput_finished
                throughput_start_step = step
            else:
                reduced_metrics = None
                samples_per_second = None
            if context.is_main and reduced_metrics is not None:
                values = {name: float(value.item()) for name, value in reduced_metrics.items()}
                print(json.dumps({"epoch": epoch, "step": step, **values}), flush=True)
                logger.log(
                    {
                        **{f"train/{name}": value for name, value in values.items()},
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_batch_size": batch_contract.global_batch_size,
                        "train/samples_per_second": samples_per_second,
                    },
                    step=step,
                )
            if context.is_main and step % sample_interval == 0:
                if fixed_video is None:
                    raise RuntimeError("Fixed reconstruction sample was not initialized")
                was_training = model.training
                model.eval()
                with ema.average_parameters(trainable), torch.no_grad(), precision.autocast():
                    fixed_on_device = fixed_video.to(context.device)
                    if any(
                        bool(config["training"].get(enabled, False))
                        and str(config["training"].get(mode, "default")) == "reduce-overhead"
                        for enabled, mode in (
                            ("compile_encoder", "compile_encoder_mode"),
                            ("compile_decoder", "compile_decoder_mode"),
                        )
                    ):
                        torch.compiler.cudagraph_mark_step_begin()
                    reconstructed = model(fixed_on_device)["recon"]
                model.train(was_training)
                sample = {
                    "real": fixed_video,
                    "recon": reconstructed.detach().cpu(),
                }
                save_reconstruction_sample(
                    sample,
                    run / "samples" / f"step-{step:08d}.pt",
                )
                logger.log_video(
                    "samples/reconstruction",
                    comparison_video(sample["real"], sample["recon"]),
                    step=step,
                    fps=int(config["data"].get("fps", 8)),
                )
                update_decoder_execution_metadata(
                    run,
                    model.decoder.execution_metadata(),
                    training_dtype=precision.name,
                )
            if step % checkpoint_interval == 0:
                rng_state = gather_rng_states()
                if context.is_main:
                    optimizer_objects: dict[str, Any] = {"generator": optimizer}
                    scheduler_objects: dict[str, Any] = {"generator": scheduler}
                    payload = build_training_checkpoint(
                        task="recon_training",
                        epoch=sampler.epoch,
                        step=step,
                        model=model,
                        ema=ema,
                        optimizer=optimizer_objects,
                        scheduler=scheduler_objects,
                        scaler=scaler,
                        data_state=sampler.state_dict(
                            gradient_accumulation_microstep=accumulation.microstep
                        ),
                        resolved_config=config,
                        model_metadata=model.metadata(),
                        rng_state=rng_state,
                    )
                    save_training_checkpoint(payload, run / "checkpoints")
                last_saved_step = step
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        if stop:
            break
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        rng_state = gather_rng_states()
        if context.is_main:
            optimizer_objects = {"generator": optimizer}
            scheduler_objects = {"generator": scheduler}
            payload = build_training_checkpoint(
                task="recon_training",
                epoch=sampler.epoch,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer_objects,
                scheduler=scheduler_objects,
                scaler=scaler,
                data_state=sampler.state_dict(
                    gradient_accumulation_microstep=accumulation.microstep
                ),
                resolved_config=config,
                model_metadata=model.metadata(),
                rng_state=rng_state,
            )
            save_training_checkpoint(payload, run / "checkpoints")
    if context.is_main:
        update_decoder_execution_metadata(
            run,
            model.decoder.execution_metadata(),
            training_dtype=precision.name,
        )
    logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train V-RAE reconstruction")
    if arguments.build_only:
        config = load_config(arguments.config)
        print(json.dumps(validate_build(config), indent=2))
        return
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps)


if __name__ == "__main__":
    main()
