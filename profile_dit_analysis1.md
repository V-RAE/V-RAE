# VideoDiT：关闭 Gradient Checkpointing、Global Batch 16 的 Profile

## 实验配置

| 项目 | 值 |
|---|---|
| 配置 | `libero_videodit_reconstruction_head_no_ckpt_bs16_profile_acc2.yaml` |
| checkpoint | step200 训练 checkpoint |
| GPU / 精度 | H100 80GB / BF16 |
| global batch / accumulation / microbatch | 16 / 2 / 8 |
| gradient checkpointing | false |
| steps | 50 |
| W&B | [run 408b2d](https://wandb.ai/Aircraft/V-RAE/runs/408b2dcb856c4595babaef67e1093b6b) |

## 最后 10-step 窗口数据

| 指标 | 值 |
|---|---:|
| window time | 39.66 s |
| step time | 3.97 s |
| updates/s | 0.2522 |
| samples/s | 4.034 |
| Stage1 | 1397.85 ms |
| DIT 总体 | 483.24 ms |
| DIT CUDA Event | 489.37 ms |
| DIT forward | 215.97 ms |
| DIT backward | 262.94 ms |
| H2D / data wait / update | 4.47 / 0.42 / 36.63 ms |
| allocated / reserved | 28.60 / 68.29 GiB |
| peak allocated | 63.26 GiB |
| loss（step50） | 2.1757 |

## DIT 主阶段

| 阶段 | 平均时间/ms | allocated 均值/最大 GiB | SM | 显存控制器 | 样本数 |
|---|---:|---:|---:|---:|---:|
| forward | 215.97 | 60.45 / 63.03 | 23.00%* | 12.00%* | 2 |
| prepare | 1.91 | 26.15 / 28.72 | 11.00%* | 5.00%* | 2 |
| forward_core | 212.65 | 60.48 / 63.06 | 30.91% | 15.27% | 11 |
| encoder | 6.13 | 41.75 / 58.76 | 23.67% | 10.90% | 61 |
| decoder | 13.16 | 59.08 / 62.56 | 41.24% | 20.76% | 17 |
| post | 0.10 | 60.48 / 63.06 | 50.00%* | 27.00%* | 1 |
| backward | 262.94 | 28.72 / 28.72 | 67.83% | 34.80% | 260 |

`*` 样本数过少，仅作方向性参考。

## Encoder / Decoder 子模块

| 模块 | 时间/ms | SM | 显存控制器 | allocated 均值/最大 GiB | 样本数 |
|---|---:|---:|---:|---:|---:|
| encoder AdaLN | 0.174 | 19.44% | 8.67% | 40.68 / 57.69 | 18 |
| encoder norm1 | 0.237 | 22.83% | 11.33% | 40.80 / 57.81 | 6 |
| encoder attention | 2.605 | 22.43% | 10.29% | 41.20 / 58.21 | 65 |
| encoder norm2 | 0.210 | 23.40% | 11.40% | 41.34 / 58.36 | 5 |
| encoder MLP | 0.382 | 17.71% | 7.82% | 41.75 / 58.76 | 17 |
| decoder AdaLN | 0.173 | 29.00% | 13.00% | 57.55 / 61.03 | 1 |
| decoder norm1 | 0.204 | 67.00% | 36.00% | 57.71 / 61.19 | 1 |
| decoder attention | 2.110 | 14.00% | 6.00% | 58.29 / 61.77 | 1 |
| decoder norm2 | 0.191 | 63.00% | 32.00% | 58.48 / 61.96 | 1 |
| decoder MLP | 0.352 | 47.00% | 26.00% | 59.08 / 62.56 | 2 |

子模块 GPU 样本极少，不能据此精确比较单个 norm/attention/MLP 的利用率；encoder aggregate 和 backward 的样本量更可信。

## 与启用 checkpointing 的 50-step 对比

| 指标 | checkpointing 开启、global42 | checkpointing 关闭、global16 |
|---|---:|---:|
| step time | 12.16 s | 3.97 s |
| DIT forward | 4428.99 ms | 215.97 ms |
| DIT backward | 4502.09 ms | 262.94 ms |
| peak allocated | 41.40 GiB | 63.26 GiB |
| samples/s | 3.453 | 4.034 |

该对比同时改变了 global batch 和 microbatch，不能视为单变量 benchmark；但可以确认关闭 checkpointing 显著降低单 microbatch 计算时间，同时增加约 21.9 GiB peak allocated 显存。

## 数据结论

1. 关闭 gradient checkpointing 后，DIT forward/backward 的同步阶段时间分别为 `215.97/262.94 ms`，明显低于之前 checkpointing 开启的 `4428.99/4502.09 ms`。
2. 显存从 `41.40 GiB` 上升到 `63.26 GiB`，仍低于 H100 80GB，但安全余量缩小到约 16GB。
3. backward SM `67.83%`，高于 encoder aggregate `23.67%`；backward 是 GPU 利用率最高的稳定阶段。
4. encoder attention 的子模块时间 `2.605 ms`，但其 NVML 样本只有 65；不能把该均值直接解释为 attention 的精确 GPU elapsed。
5. Stage1 仍约 `1398 ms`，高于 DIT `483 ms`，在该 batch 配置下 Stage1 成为主要时间占比。
6. 该实验有效证明 checkpointing 的主要代价是显存换计算时间；是否采用它取决于显存余量和吞吐目标。

## 单 Encoder Block 独立测试

测试文件：`test/test_dit_block_profile.py`。使用 `B=8、L=1280、D=1536、24 heads、BF16`，warmup 后连续执行 100 次；时间为 CUDA Event 每次平均值，GPU 利用率为 NVML 1ms 采样均值。

| 子模块 | CUDA 时间/ms | Python wall/ms | SM | 显存控制器 | 样本数 |
|---|---:|---:|---:|---:|---:|
| AdaLN modulation | 1.711 | 1.808 | 62.58% | 32.75% | 12 |
| norm1 | 1.366 | 1.284 | 45.04% | 23.44% | 27 |
| attention | 4.044 | 4.590 | 47.33% | 24.68% | 242 |
| norm2 | 0.275 | 0.211 | 51.11% | 26.67% | 18 |
| MLP | 0.872 | 0.409 | 38.30% | 19.90% | 10 |

该独立测试表明 attention 是单 block 中耗时最长的子模块（约 4.04 ms），SM 均值约 47.3%，并非“长时间但 GPU 极低利用率”。此前训练中的低均值主要来自短调用的 NVML 覆盖和 phase 边界，而不是 attention kernel 本身没有使用 GPU。
