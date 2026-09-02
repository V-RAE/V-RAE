# VideoDiT GPU 利用率 Smoke 测试计划

## 目标

在不改变 `wandb.sample_interval`（保持现有配置的 `1000`）的前提下，找出造成 GPU 利用率波动和显存利用偏低的主要瓶颈，并选择一组能提高显存占用、吞吐和计算连续性的 VideoDiT 配置，随后用该配置启动正式训练。

## 当前基线

详细执行计划与 10 步 smoke 记录统一放在 `configs/training/generation/smoke/PLAN.md` 和 `RECORD.md`。当前基线为 batch 32、DiT gradient checkpointing 开启；旧的 batch36/40/48 探索不再作为当前方案。

## 实验设计

每个方案使用独立 `run_name`，复用已经生成的 latent statistics，使用 `--max-steps 10`，不触发 sample（因此不会改变或绕过 `sample_interval`）。每次记录：

- step wall time、吞吐、data wait、H2D、stage1、DiT、optimizer/update 阶段时长；
- allocated/reserved/peak 显存和各阶段显存；
- 训练进程是否正常完成、是否 OOM/数据解码失败；
- smoke 期间的 `nvidia-smi dmon` SM 利用率和显存占用（若工具可用）。

测试顺序：

1. **baseline**：保持现有配置，用于验证新日志与旧基线一致。
2. **prefetch**：只提高 `worker_threads=16`、`max_inflight=64`、`max_buffered_batches=8`、`async_prefetch_batches=4`，验证数据等待是否是波动来源。
3. **no-checkpoint**：只关闭 `dit.gradient_checkpointing`，验证重计算是否造成 kernel 间隙，并观察显存是否明显上升。
4. **batch40**：保持 checkpointing，改 `global_batch_size=40`，在 81.6 GiB 卡上测试扩大工作集的收益和 OOM 风险。
5. **combined**：采用前面安全且有效的 prefetch/batch/checkpoint 组合；若 no-checkpoint 或 batch40 单独已经 OOM，则不叠加该项。

## 选择标准

优先选择：无 OOM、数据等待低且稳定、`steps_per_sec` 高、DiT 阶段连续、显存 peak 更接近卡容量但保留至少约 10% 余量的组合。若吞吐和利用率冲突，先保证训练稳定，再选择吞吐更高者。所有结论以 smoke 日志和运行元数据为准。

## 正式训练

选择方案后生成/固定正式配置，保留 `sample_interval: 1000` 不变，启动不带 `--max-steps` 的 VideoDiT 训练，并记录启动命令、run 目录和首个稳定日志窗口。
