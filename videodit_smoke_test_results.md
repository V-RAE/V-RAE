# VideoDiT Smoke 测试结果与正式方案

当前有效结果见 `configs/training/generation/smoke/RECORD.md`：所有 smoke 测试均为单卡、10 steps、global batch 32，并复用已有 latent statistics；`wandb.sample_interval` 保持 `1000`。本文件中的历史 batch36/40/48 结果不再作为当前方案。

## 结果

| 方案 | 关键改动 | 窗口/step | data wait | stage1 | DiT | peak 显存 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | batch32，checkpointing，线程/队列 8/32/4，预取 2 | 161.03 s / 8.05 s | 388 ms | 1.38 s | 5.99 s | 45.8 GiB | 吞吐最好但显存偏低 |
| prefetch | 线程/队列改为 16/64/8，预取 4 | 295.43 s / 14.77 s | 598 ms | 5.36 s | 8.71 s | 45.8 GiB | CPU/解码争用，排除 |
| no-checkpoint | 关闭 DiT gradient checkpointing | OOM | — | — | — | 约 78.8 GiB | 第一个 batch 前向即 OOM，排除 |
| batch36 | batch36，其余 baseline | 168.88 s / 8.44 s | 1331 ms | 1.23 s | 5.59 s | 48.5 GiB | 稳定折中 |
| batch40 | batch40，其余 baseline | 201.00 s / 10.05 s | 2121 ms | 1.42 s | 6.20 s | 51.3 GiB | 显存增加但吞吐下降 |
| batch40 + prefetch | batch40 并叠加高预取 | 344.68 s / 17.23 s | 678 ms | 6.89 s | 9.44 s | 51.3 GiB | 争用最严重，排除 |
| batch48 | batch48，其余 baseline | 230.45 s / 11.52 s | 3417 ms | 1.36 s | 6.45 s | 56.9 GiB | 无 OOM，显存最高；选为正式方案 |

### GPU 利用率观察

在高预取 smoke 的 1 秒 `nvidia-smi dmon` 采样中，SM 利用率表现为数据/同步间隙时接近 0%，进入 DiT kernel 后多次达到 100%；这与训练日志中 GPU 阶段占主导、`data_wait` 偶发升高的现象一致。显存则在训练窗口内稳定，没有持续增长，说明主要问题是工作阶段之间的空隙而非显存泄漏。

## 选择理由

正式方案采用 **batch48 + gradient checkpointing=true + 原始 bounded loader（8 解码线程、32 inflight、4 buffered、异步预取 2）**。它把 peak 显存从 baseline 的 45.8 GiB 提升到 56.9 GiB，同时样本吞吐约 `48 / 11.52 = 4.17 samples/s`，略高于 baseline 的 `32 / 8.05 = 3.98 samples/s`。关闭 checkpointing 虽然能把显存推到上限，但已验证不可训练；提高预取线程/队列反而增加 CPU 解码争用，因此不采用。

## 正式训练记录

- 配置：`configs/training/generation/vjepa2_1_libero_videodit_high_util_fixed.yaml`
- 初始化：`ckpts/pretrained/vrae/vrae_vjepa2.1.pt`，stage1 使用 EMA 权重
- `global_batch_size: 48`，`dit.gradient_checkpointing: true`
- `sample_interval: 1000`（保持不变）
- 启动后应继续观察新增的 `[epoch-start]`、`[profile] window=...`、`[checkpoint]`、`[training-end]` 日志，以及 W&B 中的 `train/window_elapsed_sec` 和 `train/profiled_batches`。

正式训练已启动并确认进入 epoch 0、step 20：当前窗口为 217.86 s（10.89 s/step），peak 显存 56.9 GiB，`gradient_checkpointing_resolved=true`，数据管线参数为 8/32/4/2。启动后的 1 秒 `nvidia-smi dmon` 采样（358 个有效样本）SM 平均 34.7%，最高 100%，其中 21.2% 的采样达到至少 80%；这表明计算 kernel 会达到满载，但仍有数据等待和 kernel 间隙，后续可直接依据新增 profile 字段继续定位。
