# VideoDiT smoke 入口

执行计划与阶段记录已统一迁移到：

- `configs/training/generation/smoke/PLAN.md`
- `configs/training/generation/smoke/RECORD.md`

当前 smoke 约束为单卡、`--max-steps 10`、`training.global_batch_size: 32`。请使用该目录中的 baseline/optimized 配置；旧的 batch36/40/48 实验仅保留在 Git 历史中。
