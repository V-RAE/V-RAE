# VideoDiT 200-step 运行时 Profile

## 实验数据

| 项目 | 值 |
|---|---|
| 配置 | `configs/training/generation/libero_videodit_reconstruction_head.yaml` |
| run | `vjepa2_1_libero_videodit_reconstruction_head_200steps` |
| GPU | H100 80GB |
| 视角 / batch | head 单视角 / global 42，microbatch 42（accumulation=1） |
| 精度 | BF16 |
| steps | 200 |
| W&B | [run 53151d](https://wandb.ai/Aircraft/V-RAE/runs/53151d467f6e42d393d465d7a3b4a40f) |
| checkpoint | `ckpts/libero_videogen/vjepa2_1_libero_videodit_reconstruction_head_200steps/checkpoints/step-00000200.pt` |

## 最后 profile 窗口（step 181–200）

| 指标 | 值 |
|---|---:|
| 窗口时间 | 161.06 s |
| updates/s | 0.1242 |
| samples/s | 5.216 |
| data wait | 0.58 ms |
| H2D | 13.46 ms |
| Stage1 | 1730.28 ms |
| DIT 总体 | 6140.71 ms |
| DIT CUDA Event | 6262.61 ms |
| optimizer update | 27.67 ms |
| allocated / reserved | 29.71 / 45.97 GiB |
| 全局 peak allocated | 41.40 GiB |
| loss（step 200） | 1.9633 |

## DIT 主阶段

| 阶段 | 调用次数 | 平均时间/ms | allocated 均值/最大 GiB | reserved 均值 GiB | SM 利用率 | 显存控制器利用率 | 样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dit.forward` | 20 | 3424.53 | 36.23 / 36.23 | 45.97 | 38.00% | 12.00% | 1 |
| `dit.prepare` | 20 | 31.36 | 25.23 / 25.23 | 45.97 | 71.28% | 35.31% | 32 |
| `dit.forward_core` | 20 | 3385.99 | 36.23 / 36.23 | 45.97 | 31.71% | 15.10% | 21 |
| `dit.encoder` | 560 | 111.88 | 28.87 / 32.01 | 45.97 | 15.53% | 9.72% | 419 |
| `dit.decoder` | 40 | 104.82 | 33.36 / 33.54 | 45.97 | 19.26% | 10.29% | 35 |
| `dit.post` | 20 | 4.17 | 36.23 / 36.23 | 45.97 | 45.25% | 15.00% | 4 |
| `dit.backward` | 20 | 2690.78 | 30.33 / 30.33 | 45.97 | 65.34% | 40.64% | 1273 |

## Encoder 子模块

| 子模块 | 调用次数 | 平均时间/ms | allocated 均值/最大 GiB | SM 利用率 | 显存控制器利用率 | 样本数 |
|---|---:|---:|---:|---:|---:|---:|
| `adaln_modulation` | 1120 | 2.63 | 29.66 / 31.81 | 30.93% | 18.08% | 188 |
| `norm1` | 1120 | 5.62 | 30.04 / 31.96 | 21.80% | 13.35% | 283 |
| `attn` | 1120 | 45.37 | 31.14 / 33.36 | 29.82% | 18.13% | 2313 |
| `norm2` | 1120 | 5.48 | 31.60 / 34.13 | 28.63% | 17.06% | 251 |
| `mlp` | 1120 | 7.29 | 32.69 / 36.12 | 26.58% | 16.57% | 324 |

## Decoder 子模块

| 子模块 | 调用次数 | 平均时间/ms | allocated 均值/最大 GiB | SM 利用率 | 显存控制器利用率 | 样本数 |
|---|---:|---:|---:|---:|---:|---:|
| `adaln_modulation` | 80 | 3.65 | 33.56 / 34.47 | 33.89% | 16.11% | 18 |
| `norm1` | 80 | 8.21 | 34.07 / 34.67 | 28.79% | 13.61% | 28 |
| `attn` | 80 | 59.95 | 35.65 / 36.60 | 35.82% | 16.23% | 228 |
| `norm2` | 80 | 8.08 | 36.27 / 37.63 | 38.14% | 19.89% | 28 |
| `mlp` | 80 | 10.42 | 37.83 / 40.49 | 41.40% | 19.51% | 43 |

## 数据分析

### 时间瓶颈

- `forward_core` 占完整 forward 的 `3385.99 / 3424.53 = 98.9%`。
- backward 平均 `2690.78 ms`，约为 forward 的 78.6%；其中包含 checkpoint 重算。
- encoder attention 是 encoder 子模块最大耗时项：`45.37 ms/调用`，约占 encoder 单次聚合时间的主要部分。
- decoder attention 为 `59.95 ms/调用`，高于 encoder attention；decoder MLP 也高于 encoder MLP，但 decoder 只有 2 层。
- `prepare`、`post` 的绝对时间很小，不是优化重点。

### 显存

- 窗口全局 peak allocated 为 `41.40 GiB`，没有容量压力。
- forward/core 的 allocated 约 `36.23 GiB`，是 DIT 主体的高点。
- decoder MLP 退出时 allocated 最大 `40.49 GiB`，接近全局峰值，说明 decoder 后段临时激活/输出张量占用最高。
- backward 结束后 allocated 回落到 `30.33 GiB`，reserved 仍约 `45.97 GiB`，符合 caching allocator 行为。

### GPU 利用率

- backward SM `65.34%`、显存控制器 `40.64%`，是最密集的稳定阶段。
- encoder aggregate SM `15.53%`，但其 attention 子模块 SM `29.82%`；encoder 的低均值主要来自 block 间隙、norm/调制等短阶段和 NVML 设备级平均。
- decoder aggregate SM `19.26%`，decoder attention/MLP 分别为 `35.82%/41.40%`，单次计算更密集但持续时间短。
- `dit.forward`、`prepare`、`post` 样本数不足（1/32/4），这些均值只能作方向性参考；encoder、decoder、backward 样本量较充分。
- DIT 总体 SM 均值不应使用本窗口的单/少量样本值；应以内部 aggregate 和子模块样本为主。

## 结论

200-step 实测表明：DIT 时间主要集中在 `forward_core`，其中 encoder/decoder 的 attention 是核心耗时子模块；backward 的 GPU 利用率最高。显存峰值约 41.4 GiB，容量不是瓶颈。encoder aggregate 的低 SM 均值与 attention 子模块较高 SM 均值并存，说明优化方向应是减少 block 间隙、提高 attention/MLP 的连续执行效率，并评估 checkpoint 重算代价，而不是优化 `prepare/post`。

## Attention 计时一致性复核

针对“attention 用时长但 GPU 利用率低”的矛盾，使用相同的 `B=14、L=1280、D=1536、24 heads、BF16`，对单个 `AdaLNZeroDDTBlock` 做 1 次 warmup + 5 次 CUDA Event 对照测试：

| 次数 | Python wall/ms | CUDA Event/ms |
|---:|---:|---:|
| 1 | 5.20 | 6.90 |
| 2 | 3.67 | 6.60 |
| 3 | 3.51 | 6.55 |
| 4 | 3.48 | 6.56 |
| 5 | 3.58 | 6.59 |

CUDA Event 稳定约 `6.6 ms`，而 Python wall 只有约 `3.6 ms`，证明 CUDA 异步执行会使当前 decorator 的 wall-clock 计时失真。200-step 报告中的 `encoder.attn=45.37 ms` 和 `decoder.attn=59.95 ms` 是 Python decorator 时间，不是严格的 GPU 执行时间；同时 NVML 采样也是设备级周期均值，两者不能直接比较。

因此“attention 用时长、SM 很低”的表象主要来自两个统计口径不一致：

1. 时间按未同步的 Python 函数边界统计，可能包含前序 CUDA 队列等待，也可能漏掉函数返回后的异步 kernel。
2. GPU 利用率按 NVML 采样点归属，短 attention kernel 可能只被少量采样点覆盖。

当前 attention GPU 利用率和子模块时间只能用于趋势判断，不能用于精确排序。要得到可信的 attention GPU 时间，phase decorator 需要在入口/出口记录 CUDA Event，并在窗口汇总时读取 Event elapsed；NVML 均值应同时报告样本数和覆盖时长。

## 同步边界实验（50 steps）

配置新增 `runtime.profile.synchronize_phases: true`。DIT 主 phase 在入口和出口执行 CUDA synchronize；高频子模块保持非同步包装，避免每个 norm/attention/MLP 调用都触发全局同步。

W&B：[sync50 run](https://wandb.ai/Aircraft/V-RAE/runs/dfdfdaad2f384e61b5e78fd919e229bc)

| 指标（最后 10 steps） | 异步 200-step窗口 | 同步 50-step窗口 |
|---|---:|---:|
| step time | 8.05 s | 12.16 s |
| Stage1 | 1723.24 ms | 2117.38 ms |
| DIT CUDA time | 6262.61 ms | 8968.81 ms |
| DIT forward | 3424.53 ms | 4428.99 ms |
| DIT backward | 2690.78 ms | 4502.09 ms |
| peak allocated | 41.40 GiB | 41.40 GiB |

同步后阶段时间显著增加，说明此前 Python wall time 确实掩盖了 CUDA 队列尾部执行和等待；显存峰值不变，证明同步没有改变模型内存需求。同步实验中 DIT forward/backward 的时间边界更可信，但 GPU 利用率仍受 NVML 设备采样影响。

同步窗口的 GPU 指标：

| 阶段 | SM | 显存控制器 | 样本数 |
|---|---:|---:|---:|
| DIT forward_core | 22.43% | 10.38% | 21 |
| encoder aggregate | 13.08% | 8.32% | 248 |
| decoder aggregate | 21.00% | 10.02% | 46 |
| backward | 43.83% | 27.86% | 924 |
| encoder attention | 21.80% | 13.84% | 1672 |
| decoder attention | 33.04% | 12.36% | 135 |

同步实验仍显示 attention 的 GPU 利用率高于 encoder aggregate，但不再把未同步 wall time 当作 GPU 执行时间。若要得到严格的 attention GPU elapsed，下一步应把 CUDA Event elapsed 作为独立指标上传，而不是使用 phase 的 Python duration。
