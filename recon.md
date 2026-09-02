# V-JEPA 2.1 + LIBERO 重建训练：完整文件调用链

本文只描述当前正在使用的单卡配置：
configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml。

## 1. 启动脚本：scripts/train/vjepa2_1.sh

~~~bash
# 阶段 1/3：解析单机/单卡参数
nnodes="$1"                 # 当前为 1
gpus_per_node="$2"          # 当前为 1
node_rank="$3"              # 当前为 0
shift 3

# 阶段 2/3：设置源码路径并选择解释器
project_root="/zsh/code/V-RAE"
export PYTHONPATH="$project_root/src:$PYTHONPATH"
python_bin="/zsh/miniconda3/envs/lerobotv3/bin/python"
torchrun_bin="/zsh/miniconda3/envs/lerobotv3/bin/torchrun"

# 阶段 3/3：启动 Python 训练模块
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m vrae.training.recon_training.train \
  --config configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml \
  --max-steps 10000
~~~

脚本不做训练计算，只负责校验 rank/GPU 参数、设置 PYTHONPATH、选择 Python 3.12 lerobotv3 环境，并通过 torchrun 启动 vrae.training.recon_training.train。当前是单节点单进程，因此 WORLD_SIZE=1，不建立多卡 DDP 通信。

## 2. 配置：configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml

~~~yaml
# 阶段 1/3：继承正式 V-JEPA 2.1 LIBERO 重建配置
include:
  - vjepa2_1_lerobot.yaml

# 阶段 2/3：覆盖当前单卡随机 interval 与预训练初始化设置
run_name: vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained
data:
  frame_interval: 4
  random_frame_interval: true
  frame_interval_min: 1
  frame_interval_max: 4
training:
  init_from: ckpts/pretrained/vrae/vrae_vjepa2.1.pt
  init_weights: ema
  resume: null
  global_batch_size: 8
  num_workers: 8
  pin_memory: true
  prefetch_factor: 4
  persistent_workers: true
  prefetch_to_device: true
  checkpoint_interval: 1000

# 阶段 3/3：在线记录并保留本地采样逻辑
wandb:
  mode: online
  sample_interval: 2000
~~~

当前配置从 V-JEPA 2.1 预训练 EMA 权重开始，单卡每个 micro-batch 包含 8 个样本；视频 clip 使用随机 frame interval，8 个 worker、pinned memory 和 CUDA 预取用于减少视频解码与 Host-to-Device 拷贝造成的 GPU 空档。基础模型、数据和 loss 仍由 `vjepa2_1_lerobot.yaml` 及其 include 提供，当前文件中的值覆盖被包含文件中的同名字段。

## 3. 配置加载与路径：src/vrae/config.py、src/vrae/paths.py

~~~python
# src/vrae/config.py
def load_config(path, *, overrides=None):
    config = _load_yaml_file(Path(path), ())  # 递归读取 include
    if overrides:
        config = deep_merge(config, overrides)
    validate_config(config)                    # 检查 encoder、数据、checkpoint 等契约
    return config

def _load_yaml_file(path, stack):
    payload = yaml.safe_load(path.open()) or {}
    includes = payload.pop("include", [])
    merged = {}
    for include in includes:
        merged = deep_merge(merged, _load_yaml_file(path.parent / include, ...))
    return deep_merge(merged, payload)
~~~

config.py 将 YAML 展开为最终 mapping，并确认 encoder 必须是 vjepa2_1、数据集必须是 lerobot、V-JEPA tubelet 必须为 2、pool group 必须为 2，同时限制 checkpoint 位于项目 ckpts/ 下。paths.py 再把数据、第三方源码、感知权重和训练输出解析为绝对路径。

## 4. 训练运行初始化：src/vrae/training/common/engine.py

~~~python
def prepare_run(config_path, paths_path=None):
    # 阶段 1/3：初始化分布式上下文并读取最终配置
    distributed = initialize_distributed()
    config = load_config(config_path)
    config["wandb"] = bind_wandb_run_name(config["wandb"], config["run_name"])
    paths = load_project_paths(config, override=paths_path, ...)

    # 阶段 2/3：创建 run 目录并保存 resolved_config.yaml
    run = paths.training_run(config["task"], config["run_name"], create=False)
    if not config["training"].get("resume"):
        run.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, run / "resolved_config.yaml")
        save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")

    # 阶段 3/3：准备 checkpoints/samples/wandb 子目录
    for directory in ("checkpoints", "samples", "wandb"):
        (run / directory).mkdir(exist_ok=True)
    return config, paths, run
~~~

prepare_run 在建模前完成配置、路径、run identity 和输出目录准备；resume 模式还会校验 checkpoint identity 与 resolved config。当前训练使用 init_from 而不是 resume：只加载已有模型参数，优化器、采样器和 step 计数从新 run 初始化。

## 5. 模型注册与 V-RAE 构建：src/vrae/registry.py、src/vrae/models/autoencoder.py

~~~python
# src/vrae/registry.py
def register_builtin_models():
    # 延迟导入各模型模块，并执行 registry decorator
    for module in (
        "vrae.models.autoencoder",
        "vrae.models.decoder",
        "vrae.models.pooling",
        "vrae.models.dit.video_dit",
        "vrae.models.encoders.vjepa2_1",
    ):
        importlib.import_module(module)

# src/vrae/models/autoencoder.py
@MODELS.decorator("vrae")
class VRAE(nn.Module):
    @classmethod
    def from_config(cls, config, **kwargs):
        register_builtin_models()
        encoder = ENCODERS.build(config["model"]["encoder"], ...)
        pool = POOLERS.build(
            {**config["model"]["pooling"], "dim": encoder.hidden_size},
            dim=encoder.hidden_size,
            group_size=config["model"]["pooling"]["group_size"],
        )
        decoder = DECODERS.build(
            {"name": "vrae_decoder", **config["model"]["decoder"]},
            input_dim=encoder.hidden_size,
        )
        return cls(encoder, pool, decoder)

    def __init__(self, encoder, temporal_pool, decoder):
        self.encoder = encoder
        self.temporal_pool = temporal_pool
        self.decoder = decoder
        self.encoder.requires_grad_(False)       # V-JEPA 冻结
        self.encoder.eval()
        self.temporal_pool.requires_grad_(True)  # 训练 temporal pool
        self.decoder.requires_grad_(True)        # 训练 decoder
~~~

registry 把字符串名称映射到模型工厂；VRAE.from_config 组合冻结的 V-JEPA encoder、可训练的 temporal attention pool 和可训练的 V-RAE decoder。虽然重建训练不使用 VideoDiT，统一注册时仍会导入 vrae.models.dit.video_dit，因此该模块必须可导入。

## 6. 配置文件的模型与数据来源

~~~yaml
# configs/training/common/vjepa2_1_lerobot.yaml
include:
  - ../../models/vrae_vjepa2_1_l_k7.yaml
  - ../../data/libero.yaml

# configs/models/vrae_vjepa2_1_l_k7.yaml
model:
  encoder:
    name: vjepa2_1
    variant: vit_l16
    layers: [11, 13, 15, 17, 19, 21, 23]
    hidden_size: 1024
    encoder_tubelet_size: 2
    checkpoint: ckpts/pretrained/encoders/vjepa2_1/model.pt
  pooling:
    name: temporal_attention
    group_size: 2

# configs/models/decoder_vit_xl.yaml（继续被模型配置 include）
model:
  decoder:
    hidden_size: 1152
    depth: 28
    num_heads: 16
    patch_size: 16
    tubelet_size: 4
~~~

这组配置定义 V-JEPA ViT-L/16、7 个中间层融合、1024 维 latent、tubelet=2，以及 group=2 的 temporal pool；二者共同产生 4 倍时间压缩。decoder 的 tubelet=4 将 latent 的 4 个时间块还原为 16 帧；训练配置把 decoder attention mode 覆盖为 full。

~~~yaml
# configs/data/libero.yaml
data:
  dataset: lerobot
  root: /zsh/cache/data/Lerobot/libero
  num_frames: 16
  image_size: 256
  camera_keys:
    - {key: observation.images.image,  name: head,  stream_id: 0}
    - {key: observation.images.image2, name: wrist, stream_id: 1}
  sampling: random
~~~

LIBERO 配置选择两个同步相机、16 帧、256×256 RGB 输入，并附带 40 个 task 的 class 映射；重建阶段保留 label/task metadata，但 loss 只比较视频像素与感知特征。

## 7. 数据读取：src/vrae/training/recon_training/data.py、src/vrae/data/lerobot.py

~~~python
# data.py
def build_reconstruction_loader(config, paths, *, rank, world_size):
    dataset = build_reconstruction_dataset(config, paths)
    batch = resolve_batch_contract(config["training"], world_size=world_size)
    sampler = StatefulDistributedBatchSampler(
        len(dataset), batch.local_micro_batch_size,
        rank=rank, world_size=world_size, seed=config["data"]["seed"],
        shuffle=True, drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate_reconstruction,
    )
    return loader, sampler

def flatten_multiview_video(video):
    # 当前 B=4,T=16,V=2,C=3,H=W=256
    # [B,T,V,C,H,W] -> [B*V,T,C,H,W]
    return video.permute(0, 2, 1, 3, 4, 5).reshape(
        video.shape[0] * video.shape[2], video.shape[1],
        video.shape[3], video.shape[4], video.shape[5],
    ).contiguous()
~~~

~~~python
# src/vrae/data/lerobot.py
def __getitem__(self, index):
    episode, start, stop, class_entry, task = self._episodes[index]
    indices = self.sampler(stop - start, generator=self._generator(index)) + start
    rows = [self.get_frame(row_index) for row_index in indices.tolist()]
    views = []
    for camera in self.camera_keys:
        frames = [row[camera["key"]].permute(2, 0, 1).float() / 255.0 for row in rows]
        views.append(torch.stack(frames))              # [T,3,H,W]
    video = torch.stack(views, dim=1)                  # [T,V,3,H,W]
    return {"video": video, "label": class_entry.class_id, ...}
~~~

LeRobotVideoDataset 继承仓库内置的 `src/vrae/data/lerobot3/lerobot_dataset.py`，由本地
reader 从 parquet metadata 和 MP4 视频解码连续帧；V-RAE 子类负责组合 head/wrist 两个
view、应用图像后处理并归一化到 [0,1]。当前 batch 在 collate 后是
[4,16,2,3,256,256]，随后展平为 [8,16,3,256,256]；CudaVideoPrefetchIterator 在配置
开启时把下一个 pinned batch 提前传到 GPU。

## 8. 帧读取、类别映射与 prompt：src/vrae/data/lerobot.py

`LeRobotVideoDataset.__init__` 直接读取 `meta.tasks`，按 `task_index` 排序后生成
`task_index_to_class_id`、`class_id_to_task_index` 和 `class_names`。每个运行目录在训练
开始时保存一份 `class_map.json`，训练进程随后从该文件读取并校验类别数量；不再依赖
单独的类别映射类。

数据集用 `delta_indices = range(frame_num)` 请求连续帧，经过 resize-smallest-side 和
center-crop 到 256x256，输出 `[T,V,3,H,W]`（单视角时为 `[T,3,H,W]`）的 float RGB
`[0,1]` 张量。每个 item 同时返回 `task` 和 FastWAM 风格的字符串 `prompt`；collate
后 `prompt` 是长度为 batch 的字符串列表，可自然支持多 prompt batch。

## 9. 主训练入口：src/vrae/training/recon_training/train.py

~~~python
def main():
    # 阶段 1/3：解析 --config/--paths/--max-steps
    arguments = parse_config_argument("Train V-RAE reconstruction")
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)

    # 阶段 2/3：构建模型、loss、优化器、loader
    train(config, paths, run, max_steps=arguments.max_steps)

class ReconstructionGraph(nn.Module):
    def forward(self, video):
        clean = self.vrae.encode(video)             # 冻结 V-JEPA + 可训 pool
        train_latents = clean
        if self.training and self.noise_config.get("enabled", False):
            train_latents, sigma = add_reconstruction_noise(clean, tau=0.8)
        return {
            "recon": self.vrae.decode(train_latents),
            "latents": clean,
            "sigma": sigma,
        }
~~~

main 是 Python 入口；train 负责把所有组件接起来。每次迭代的核心数据流是 [B*V,16,3,256,256] -> latent -> noisy latent -> reconstruction，其中 V-JEPA encoder 在 no_grad 下冻结，latent noise 只在训练态开启。

## 10. V-JEPA 编码：src/vrae/models/encoders/base.py、src/vrae/models/encoders/vjepa2_1.py

~~~python
# base.py
def _forward_impl(self, video):
    video = self._validate_video(video)             # [N,16,3,256,256]
    grid_size = (video.shape[-2] // 16, video.shape[-1] // 16)  # (16,16)
    output = self._encode_preprocessed(
        self._preprocess(video), grid_size=grid_size
    )
    self._validate_output(output, batch=video.shape[0], time=video.shape[1],
                          grid_size=grid_size)
    return output

# vjepa2_1.py
def _preprocess(self, video):
    video = self._normalize_video(video)             # ImageNet mean/std
    return video.permute(0, 2, 1, 3, 4).contiguous()

def _encode_preprocessed(self, video, *, grid_size):
    # [N,3,16,256,256] -> [N, 8*16*16, 1024]
    hidden = self.backbone.patch_embed(video)
    selected = []
    for index, block in enumerate(self.backbone.blocks):
        hidden = block(hidden, T=8, H_patches=16, W_patches=16, ...)
        if index in {11, 13, 15, 17, 19, 21, 23}:
            selected.append(self.backbone.norms_block[-1](hidden))
    fused = torch.stack(selected, dim=0).mean(dim=0)
    return fused.reshape(N, 8, 1024, 16, 16)
~~~

base.py 统一处理 RGB 合法性、输入归一化前的契约、patch grid 和输出 shape 校验；vjepa2_1.py 动态导入 third_party/vjepa2/ 的官方 ViT-L/16，实现 tubelet=2、24 个 block、7 个层的 K7 融合。当前输入 16 帧经 tubelet=2 后得到 8 个时间 token，输出为 [8,8,1024,16,16]。

## 11. Temporal pool：src/vrae/models/pooling.py

~~~python
def forward(self, x):
    # x: [N,Te,C,H,W] = [8,8,1024,16,16]
    output_time = Te // self.group_size       # 8 // 2 = 4
    grouped = x.reshape(N, output_time, 2, C, H, W)
    grouped = grouped.permute(0, 1, 4, 5, 2, 3).reshape(
        N * output_time * H * W, 2, C
    )
    key = self.key(self.norm_k(grouped))
    value = self.value(grouped)
    attention = (query * key).sum(-1).softmax(dim=-1)
    pooled = (attention.unsqueeze(-1) * value).sum(dim=2)
    return pooled.reshape(N, output_time, C, H, W)
~~~

TemporalAttentionPool 在每个空间位置上把相邻 2 个 V-JEPA 时间 token 做注意力聚合，[8,8,1024,16,16] -> [8,4,1024,16,16]。它是 Stage 1 的第一个可训练模块，输出 latent 的时间压缩比例与 decoder tubelet=4 对齐。

## 12. V-RAE 解码：src/vrae/models/decoder.py

~~~python
def _forward_impl(self, latents):
    # 阶段 1/3：latent grid -> token sequence
    # [N,4,1024,16,16] -> [N,4*16*16,1024]
    hidden = latents.permute(0, 1, 3, 4, 2).reshape(
        N, chunks * height * width, channels
    )
    hidden = self.input_projection(hidden)
    hidden = (hidden.reshape(N, chunks, height * width, -1)
              + self._position(height, width, hidden)[:, None]).reshape(...)

    # 阶段 2/3：28 个 decoder block
    rope_cosine, rope_sine = self._rope_cache(chunks, height, width, hidden)
    for block in self.blocks:
        hidden = block(hidden, chunks, height, width, rope_cosine, rope_sine)

    # 阶段 3/3：预测 RGB tubelet 并反 patchify
    predicted = self.prediction(self.norm(hidden))
    return self.depatchify(predicted, chunks=chunks, height=height, width=width)
~~~

decoder 把 [8,4,1024,16,16] 展平为空间 token，经过 28 层 Transformer、3D RoPE 和 full attention，再用 tubelet_size=4 的 prediction/depatchify 还原为 [8,16,3,256,256]。decoder 及其位置参数参与梯度更新；V-JEPA 不参与梯度。

## 13. Loss：src/vrae/training/recon_training/losses.py

~~~python
def _forward_impl(self, reconstructed, target):
    # 当前输入是 [N,16,3,256,256]，N=B*V=8
    l1 = F.l1_loss(reconstructed, target)
    reconstructed_frames, target_frames = self._select_perceptual_frames(
        reconstructed, target
    )
    # 每个 4 帧 chunk 取 4 帧；16 帧因此全部进入感知分支
    reconstructed_frames = reconstructed_frames * 2.0 - 1.0
    target_frames = target_frames * 2.0 - 1.0
    lpips, gram = self.perceptual(target_frames, reconstructed_frames)
    total = 1.0 * l1 + 1.0 * lpips + 100.0 * gram
    return total, {"l1": l1.detach(), "lpips": lpips.detach(),
                   "gram": gram.detach()}
~~~

ReconstructionLoss 同时计算像素 L1、VGG16/LPIPS 和 Gram loss；当前正式权重位于 ckpts/pretrained/perceptual/。感知分支将 [N,T,3,H,W] 展平成帧批次，再由 PerceptualGramLoss 使用 VGG16 特征计算 LPIPS/Gram，最终按 1、1、100 加权。

## 14. 反向更新：accumulation.py、optim.py、precision.py、ema.py

~~~python
# train.py 主循环
with accumulation.sync_context(graph):
    with precision.autocast():              # 当前为 CUDA bf16
        result = graph(video)
        total, metrics = loss_module(result["recon"], video)
    accumulation.backward(total, scaler)

sampler.commit_batch()
if accumulation.advance():                  # 当前 accumulation=1，立即更新
    optimizer_step(optimizer, scaler=scaler)
    scheduler.step()
    ema.update(trainable)
    step += 1
~~~

当前 gradient_accumulation_steps=1，每个 batch 都执行一次 AdamW 更新；PrecisionPolicy 用 CUDA bf16 autocast，fp16 时才创建 GradScaler。ExponentialMovingAverage 只跟踪 temporal pool 和 decoder 的浮点参数，后续采样使用 EMA 权重而不是瞬时训练权重。

## 15. 日志、采样与 checkpoint：wandb.py、sample.py、checkpoint.py

~~~python
# train.py
if step % sample_interval == 0:             # 当前继承值为 1000
    with ema.average_parameters(trainable), torch.no_grad():
        reconstructed = model(fixed_video.to(device))["recon"]
    sample = {"real": fixed_video, "recon": reconstructed.cpu()}
    save_reconstruction_sample(
        sample, run / "samples" / f"step-{step:08d}.pt"
    )
    logger.log_video(
        "samples/reconstruction", comparison_video(...), step=step, fps=10
    )

if step % checkpoint_interval == 0:          # 当前为 1000
    payload = build_training_checkpoint(
        task="recon_training", epoch=..., step=step, model=model,
        ema=ema, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        data_state=sampler.state_dict(), resolved_config=config,
        model_metadata=model.metadata(), rng_state=gather_rng_states(),
    )
    save_training_checkpoint(payload, run / "checkpoints")
~~~

当前 wandb.mode=disabled，因此 WandbLogger 不登录、不上传 scalar 或 video；但本地采样仍每 1000 steps 写入 samples/step-XXXXXXXX.pt。checkpoint.py 将模型、EMA、优化器、scheduler、RNG、sampler state 和 resolved config 原子保存，并更新 checkpoints/latest.pt。

## 16. 当前训练涉及的文件清单

~~~text
scripts/train/vjepa2_1.sh
  单卡 torchrun 启动、环境变量和 rank 参数。

configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml
  当前 batch=8、随机 interval、8 workers、预取、初始化 checkpoint 和 WandB 开关。
configs/training/common/vjepa2_1_lerobot.yaml
  Stage 1 重建任务、优化器、loss、epoch 和采样策略。
configs/models/vrae_vjepa2_1_l_k7.yaml
  V-JEPA 2.1 encoder、K7 层融合和 temporal pool。
configs/models/decoder_vit_xl.yaml
  V-RAE decoder 的 28 层 Transformer 结构。
configs/data/libero.yaml
  LIBERO 根目录、head/wrist 相机、16 帧和 40 类 task 映射。

src/vrae/config.py
  YAML include 深度合并与配置契约校验。
src/vrae/paths.py
  数据、第三方源码和 checkpoint 路径解析。
src/vrae/training/common/engine.py
  run 初始化、resolved config、metadata 和通用运行准备。
src/vrae/training/recon_training/train.py
  重建训练主循环、模型/loss/优化器装配、step 控制。
src/vrae/training/recon_training/data.py
  DataLoader、batch sampler、多视角展平和 CUDA 预取。
src/vrae/data/lerobot.py
  LIBERO episode clip 采样、视频帧读取和双相机拼接。
src/vrae/data/sampling.py
  严格的 16 帧时间采样规则。
src/vrae/libero.py
  40 个 LIBERO task 到 class ID 的稳定映射。

src/vrae/models/autoencoder.py
  组装 VRAE，冻结 encoder，暴露 encode/decode。
src/vrae/models/encoders/base.py
  encoder 输入校验、ImageNet 归一化和统一输出契约。
src/vrae/models/encoders/vjepa2_1.py
  加载本地官方 V-JEPA 2.1、提取 K7 hidden states 并融合。
src/vrae/models/pooling.py
  temporal attention pooling，时间 8 -> 4。
src/vrae/models/decoder.py
  latent token 化、Transformer/3D RoPE 和视频反 patchify。

src/vrae/training/recon_training/losses.py
  L1、VGG/LPIPS、Gram 和 temporal difference loss。
src/vrae/training/common/optim.py
  AdamW 与 constant scheduler。
src/vrae/training/common/precision.py
  bf16 autocast 和 fp16 scaler。
src/vrae/training/common/accumulation.py
  梯度累积边界与 optimizer step。
src/vrae/training/common/ema.py
  训练参数的 EMA 维护与采样时临时替换。
src/vrae/training/common/checkpoint.py
  checkpoint 组装、原子保存、latest 指针和恢复。
src/vrae/training/common/wandb.py
  WandB 开关、scalar/video logging；当前为 disabled。
src/vrae/training/recon_training/sample.py
  本地 .pt 重建样本的原子保存。

third_party/vjepa2/
  V-JEPA 2.1 官方 ViT runtime source。
/zsh/code/lerobot-main/src/lerobot/datasets/lerobot_dataset.py
  LeRobot parquet/MP4 数据集父类与底层视频解码。
~~~

## 17. 一条样本的 shape 流

~~~text
LIBERO episode frames
  -> [T,V,C,H,W] = [16,2,3,256,256]
DataLoader batch
  -> [B,T,V,C,H,W] = [4,16,2,3,256,256]
flatten_multiview_video
  -> [B*V,T,C,H,W] = [8,16,3,256,256]
V-JEPA tubelet=2 + ViT-L/16
  -> [8,8,1024,16,16]
TemporalAttentionPool group=2
  -> [8,4,1024,16,16]
V-RAE decoder tubelet=4
  -> [8,16,3,256,256]
L1 + LPIPS + Gram
  -> scalar total loss -> backward -> AdamW -> EMA/checkpoint
~~~

这条路径解释了当前配置中的关键约束：16 帧必须能被 encoder tubelet、pool group 和 decoder tubelet 共同整除；双视角在进入 V-RAE 前合并到 batch 维，重建后仍以相同的单视角布局计算 loss。
