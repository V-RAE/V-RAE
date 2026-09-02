 # VideoDiT smoke 执行记录

## 第 0 阶段：代码与基线盘点

- 入口：`python -m vrae.training.libero_videogen.train`，支持 `--max-steps`；本目录所有训练验证固定为 10 步。
- 训练循环：`src/vrae/training/common/engine.py::train_class_conditional`。
- 数据路径：`LeRobotVideoDataset` → `BoundedThreadBatchLoader` → `AsyncPrefetchLoader`，只在内存中保留有限的 batch，不写入 RAE latent。
- RAE：`load_frozen_stage1` 加载 EMA 权重；训练循环中 `torch.no_grad()` 在线编码，随后立即交给 DiT。
- DiT：`transport.training_losses` 完成前向和反向；`GradientAccumulator` 在累积边界执行 optimizer/scheduler/EMA。
- 分布式：`StatefulDistributedBatchSampler` 按 rank 分片；多卡时使用 DDP、`no_sync` 累积梯度。
- 固定变量：`data.num_frames=16`、`image_size=256`、双视角、`training.global_batch_size=32`、bf16、gradient checkpointing 开启。
- smoke 输出：配置写入 `ckpts/libero_videogen/smoke/libero_videogen/<run_name>/`。

### 基线配置

`vjepa2_1_libero_videodit_smoke_baseline.yaml` 将在线解码并发降到单线程/单预取，作为数据等待对照；`global_batch_size` 保持 32。

验证命令：

```bash
python -m vrae.training.libero_videogen.train \
  --config configs/training/generation/smoke/vjepa2_1_libero_videodit_smoke_baseline.yaml \
  --build-only
```

## 第 1 阶段：粗粒度 profiler

现有 `TrainingProfiler` 已接入训练循环，记录 `data_wait`、H2D、stage1/RAE、DiT、optimizer/update 的 CUDA Event 时间，以及各阶段显存 allocated/reserved/peak。每次 `wandb.log_interval`（smoke 为 10）输出一组窗口指标；profiler 不建立长期 latent 缓存。

## 第 2 阶段：细粒度追踪

当前 smoke 只运行 10 步，不开启长时间 `torch.profiler` trace；训练阶段名称和边界已由粗粒度 profiler 固定，后续需要定位 kernel 空洞时再按 PLAN 的 1/1/3 schedule 单独采集。

## 第 3 阶段：在线数据流水线优化

`vjepa2_1_libero_videodit_smoke_optimized.yaml` 使用 8 个解码线程、32 个 in-flight 样本、4 个 batch 缓冲和 2 个异步预取 batch，仍保持 global batch 32。优化仅作用于短生命周期内存队列、pinned host batch 和 non-blocking H2D，不保存 RAE latent 到磁盘。

### 10 步实测（2026-09-02，单张 H100 80GB）

| 配置 | step 平均 | data wait | H2D | RAE/stage1 | DiT | peak 显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline（1 解码线程、预取 1） | 8.10s | 720.0ms | 19.4ms | 1.327s | 5.761s | 45.8GiB |
| optimized（8 解码线程、32/4/2 队列） | 8.49s | 816.5ms | 20.1ms | 1.330s | 6.051s | 45.8GiB |

两次运行 loss 均为 `2.572266`（step 10），无 OOM、NaN 或解码错误。10 步样本不足以证明吞吐提升；本机结果显示增加解码并发反而略慢，因此默认保留 bounded loader 的有限队列实现，但不继续扩大线程/队列。两次运行均生成了 step-00000010 checkpoint，路径在 `ckpts/libero_videogen/smoke/libero_videogen/` 下。

## 第 2 阶段：细粒度 profiler API

新增 `run_fine_profile`，支持 `wait/warmup/active/repeat`、rank 过滤、CUDA/CPU activity、trace 输出和开销较高选项（shape、memory、stack、modules）。默认 `enabled=False`；smoke 训练不打开 trace，避免把 profiler 时间混入吞吐比较。调用方只需传入单步 callback，trace 输出到指定目录的 `rank<N>` 子目录。
