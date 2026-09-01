# V-RAE — V-JEPA 2.1 + LIBERO 训练

本分支提供两阶段训练：先在 LIBERO LeRobot 数据上训练 V-RAE 重建模型，再冻结 V-RAE、以 40 个 `suite/task` 为类别训练 class-conditional VideoDiT。默认同时使用 head 和 wrist 两个同步相机。

## 安装

需要 Linux、NVIDIA GPU、Python 3.10+ 和 CUDA。LeRobot 必须安装在当前 Python 环境中；本机源码可直接安装：

```bash
conda create -n vrae python=3.10 -y
conda activate vrae
conda install -c conda-forge ffmpeg -y
pip install uv
uv pip install -e .
uv pip install -e /zsh/code/lerobot-main
```

下载官方 V-JEPA 2.1 ViT-L/16 checkpoint：

```bash
mkdir -p ckpts/pretrained/encoders/vjepa2_1
curl -fL https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o ckpts/pretrained/encoders/vjepa2_1/model.pt
```

重建 loss 使用 VGG/LPIPS 权重，下载方式见 [third_party/README.md](third_party/README.md)。

## 数据与类别

共享数据配置为 `configs/data/libero.yaml`，默认读取：

```text
/zsh/cache/data/Lerobot/libero
```

40 个类别按 metadata 中的 `task_index` 显式映射：

- `0..9`：`libero_10`
- `10..19`：`libero_goal`
- `20..29`：`libero_object`
- `30..39`：`libero_spatial`

数据集启动时会验证 40 个真实 task 是否被覆盖且没有重复。每个样本的 label 是对应 `suite/task` 的连续 class ID。

## Stage 1：LIBERO 重建训练

配置：`configs/training/vjepa2_1_lerobot.yaml`。V-JEPA 2.1 始终冻结，只训练 temporal pool 和 V-RAE decoder。

单机八卡启动：

```bash
scripts/train/vjepa2_1.sh 1 8 0
```

仅检查配置和启动命令：

```bash
VJEPA_DRY_RUN=1 scripts/train/vjepa2_1.sh 1 1 0
```

默认 Stage 1 checkpoint 最终位于：

```text
ckpts/recon_training/vjepa2_1_libero_reconstruction/checkpoints/latest.pt
```

### Stage 1 视频重建评估

先下载评估模型（I3D、Inception 和 LPIPS/VGG 权重）：

```bash
scripts/download_eval_weights.sh
```

对 LIBERO LeRobot 重建 checkpoint 计算 rFVD、LPIPS、PSNR 和 SSIM：

```bash
scripts/eval/libero_rfvd.sh --max-samples 8 --output outputs/libero_rfvd/eval.json
```

`--max-samples` 可用于快速 smoke 评估；省略时会评估配置中数据集的全部样本。默认使用
`ckpts/recon_training/vjepa2_1_libero_reconstruction_single_gpu_bs4_50k_detached/checkpoints/step-00050000.pt`
的 EMA 权重，也可通过 `--checkpoint` 指定其他 Stage 1 checkpoint。

## Stage 2：LIBERO VideoDiT 生成训练

配置：`configs/training/libero_videodit.yaml`。它会读取 Stage 1 的 EMA 权重；latent mean/std 不存在时，会先从每个 episode 采样一个 clip，分布式计算通道统计量，然后开始 VideoDiT 训练。

单机八卡启动：

```bash
scripts/train/libero_videodit.sh 1 8 0
```

只验证构建配置：

```bash
scripts/train/libero_videodit.sh 1 1 0 --build-only
```

多机时，在所有节点使用相同的 `MASTER_ADDR`/`MASTER_PORT`，并分别传入 node rank。

## Tensor 契约

`LeRobotVideoDataset` 从每个 episode 采样同步 clip，默认输出 `[T,2,3,256,256]` 的 float32 `[0,1]` 视频。V-RAE encoder/decoder 保持原始单视角接口；进入 V-RAE 前将 `[B,T,V,C,H,W]` 重排为 `[B*V,T,C,H,W]`。16 帧经过 V-JEPA tubelet=2 和 temporal pool group=2 后，V-RAE 看到 `[B*V,4,1024,16,16]` latent grid；VideoDiT 仍可使用其原有的 `[B,4,2,256,1024]` 多视角 token 和 `stream_ids` 逻辑。

VideoDiT 只更新自身参数；V-JEPA、temporal pool 和 V-RAE decoder 全部冻结。定期采样会从噪声生成双视角 latent，再经冻结的 V-RAE decoder 还原视频。

## 目录

```text
src/vrae/models/dit/                    class-conditional VideoDiT 与 flow transport
src/vrae/data/                          LeRobot episode/clip 数据集
src/vrae/training/recon_training/       Stage 1 重建训练
src/vrae/training/libero_videogen/      Stage 2 LIBERO 生成入口
configs/data/libero.yaml                数据、相机和 40 类映射
configs/training/vjepa2_1_lerobot.yaml  重建配置
configs/training/libero_videodit.yaml   生成配置
```
