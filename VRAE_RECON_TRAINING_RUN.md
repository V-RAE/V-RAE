# V-JEPA 2.1 + LIBERO 重建训练

## 当前训练

当前使用单张 NVIDIA H100 持续训练，配置为：

- 模型：V-JEPA 2.1 encoder + V-RAE decoder
- 数据：`/zsh/cache/data/Lerobot/libero`
- 正式 loss：L1 + LPIPS + Gram
- 精度：bf16
- 单卡 batch size：4
- 数据加载：8 workers、pinned memory、prefetch factor 4、persistent workers
- 设备预取：开启
- 最大训练步数：10000
- 每 1000 steps 保存 checkpoint
- WandB：关闭

配置文件：

```text
configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml
```

启动命令：

```bash
cd /zsh/code/V-RAE
env VJEPA_PYTHON=/zsh/miniconda3/envs/lerobotv3/bin/python \\
    VJEPA_TORCHRUN=/zsh/miniconda3/envs/lerobotv3/bin/torchrun \\
    scripts/train/vjepa2_1.sh 1 1 0 \\
    --config configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml \\
    --max-steps 10000
```

当前运行目录：

```text
ckpts/recon_training/vjepa2_1_libero_reconstruction_single_gpu_bs4/
```

初始化权重保存在当前运行目录的 `initialization/step-00001000.pt`。训练 checkpoint 会写入同一目录下的 `checkpoints/`。

## 当前资源状态

batch size 从 1 调整到 4，并开启数据/设备预取后，实测显存约 46.6GB、GPU 利用率约 84–100%、功耗约 560–580W；训练日志已持续输出并正常下降。低利用率主要只会出现在 checkpoint 写盘等 I/O 阶段。

## 运行环境

```text
Python: /zsh/miniconda3/envs/lerobotv3/bin/python (3.12)
LeRobot: /zsh/code/lerobot-main
V-JEPA checkpoint: ckpts/pretrained/encoders/vjepa2_1/model.pt
VGG16 weights: ckpts/pretrained/perceptual/vgg16.pt
LPIPS calibration: ckpts/pretrained/perceptual/lpips_vgg.pt
```
