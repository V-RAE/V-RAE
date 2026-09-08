# VideoDiT 多任务与 CI2V 设计计划

## 1. 目标

在现有类别到视频（c2v）训练基础上增加类别图像到视频（ci2v）：

```text
[类别, 多视角首帧] -> 首帧之后的 20 帧视频
```

任务配置预留：

- `c2v`：类别到视频，当前已有任务；
- `ci2v`：类别图像到视频，本次实现；
- `ti2v`：文本图像到视频，暂时只保留配置；
- `t2v`：文本到视频，暂时只保留配置。

当前代码使用 `libero_videogen` 作为训练入口。为避免和现有入口的 `task` 含义冲突，新增任务选择与训练器分开：

```yaml
task: ci2v
trainer: libero_videogen
```

旧配置中的 `task: libero_videogen` 继续按 c2v 兼容处理。

## 2. 固定帧语义与 shape

CI2V 的帧定义固定如下：

- 数据加载总数：21 帧；
- 第 0 帧：额外的 clean image condition，不计入目标帧数；
- 第 1～20 帧：模型生成的 20 个 target frames；
- 多视角始终保留 `V` 维，不融合 camera。

```text
raw video       [B,21,V,3,H,W]
image condition [B, 1,V,3,H,W]
target video    [B,20,V,3,H,W]
```

V-RAE 的时间压缩链路为：

```text
target: 20 frames -> tubelet=2 -> 10 tokens -> pool group=2 -> 5 latents
image:  1 frame   -> repeat 4  -> tubelet=2 ->  2 tokens -> pool group=2 -> 1 latent
```

因此：

```text
image latent [B,1,V,N,C]
target latent [B,5,V,N,C]
DiT sequence [B,6,V,N,C]
decoder input [B,5,V,N,C] -> decoded target [B,20,V,3,H,W]
```

最终输出只在采样/可视化层拼接原始首帧：

```text
decoded target  [B,20,V,3,H,W]
raw first frame [B, 1,V,3,H,W]
display video   [B,21,V,3,H,W]
```

首帧不使用 decoder 重建，`V` 维在训练、采样、解码和可视化中保持不变。

## 3. 顶端配置

建议配置结构：

```yaml
task: ci2v
trainer: libero_videogen

task_config:
  c2v:
    status: active
    condition: [class]
  ci2v:
    status: active
    condition: [class, image]
    image_condition:
      source: first_frame
      multiview: true
      pooling: same_as_target
      temporal_lift: repeat_to_four_frames
    video_target:
      frames: 20
      frame_interval: 1
    sampling:
      decode_frames: 20
      prepend_condition_frame: true
  ti2v: {status: reserved, condition: [text, image]}
  t2v: {status: reserved, condition: [text]}

data:
  num_frames: 21        # CI2V 总读取帧数：1 个首帧 condition + 20 个 target
  frame_interval: 1

dit:
  name: vrae_video_dit
  image_condition: true
  image_condition_injection: temporal_concat
  class_dropout: 0.10

latent_normalizer:
  path: ckpts/libero_videogen/latent_stats/ci2v_shared_20f_multiview.pt
```

校验规则：CI2V 使用 `data.num_frames=21` 作为总读取帧数，拆分为 1 个首帧 condition 和 20 个 target；`latent_frames=5`、`decode_frames=20`，并确保 target 帧数能被 `encoder_tubelet * pool_group = 4` 整除。

## 4. 数据加载

CI2V 直接复用 `src/vrae/data/reconstruction.py` 的 `LiberoVideoDataset`，复用原有配置和采样逻辑，加载同一 episode、同一时间起点的 21 帧。只在原数据集样本返回值上增加 CI2V 所需字段，不新增 dataset 专用配置：

1. 继续使用现有 `data.num_frames`；CI2V 要求其为 21，`LiberoVideoDataset` 读取完整 21 帧；
2. 所有 camera 使用同一组 source-frame indices；
3. `video[0:1]` 对应 condition，`video[1:]` 对应 target；数据集只新增对应的 indices 和 `clean_mask` 字段，不复制视频张量；
4. 增加 `condition_frame_indices`、`target_frame_indices`，保留原有 `label` 和 `stream_ids`；
5. episode 不足 21 帧时拒绝该样本，不能跨 episode 补帧；
6. 复用现有 `camera_keys`、resize、center crop 和 class-id 映射。

数据 collator/bounded loader 只搬运原始 `video` 与新增 indices，训练侧根据首帧/目标帧切片，避免复制两份视频张量；pinned-memory 和 H2D 行为保持一致。

## 5. V-JEPA 与 shared latent normalizer

首帧也走完整的 V-RAE 编码链路，以确保和 target 使用相同 latent 定义：

```text
image_condition:
  [B*V,1,3,H,W]
  -> repeat temporal dimension to [B*V,4,3,H,W]
  -> V-JEPA tubelet=2
  -> TemporalAttentionPool(group=2)
  -> [B*V,1,C,h,w]

video_target:
  [B*V,20,3,H,W]
  -> V-JEPA tubelet=2
  -> TemporalAttentionPool(group=2)
  -> [B*V,5,C,h,w]
```

不能把 21 帧整体送入当前 V-RAE，因为 21 不能满足 4 倍时间压缩；必须先拆成首帧和 20 帧目标。

shared normalizer 使用新的独立文件，不覆盖旧 c2v 统计。统计集合包含：

- 每个样本的 5 个 target pooled latents；
- 每个样本的 1 个 repeated-first-frame pooled latent。

建议按 latent token 数自然加权（image:target = 1:5），并在 metadata 记录：

```text
scope = target_and_repeated_first_frame
condition_temporal_lift = repeat_4
target_frames = 20
latent_frames = 5
```

两条路径都使用：

```text
normalize(stage1.encode(...), shared_normalizer)
```

但只有 target latent 进入 flow matching；image latent 始终是 clean condition。

## 6. DiT：`VRAEVideoDiT` 原生支持 c2v/ci2v

不新增 CI2V 专用 DiT 类，直接扩展现有 `src/vrae/models/dit/video_dit.py` 中的 `VRAEVideoDiT`。c2v 和 ci2v 共用同一套 latent embedding、attention/MLP blocks、AdaLN、RoPE、view embedding、输出层和权重初始化。

调用方负责把 clean condition 和 noisy target 沿时间维拼接好，`forward` 只接收统一序列和一个显式的 clean/noisy mask：

```python
def forward(
    self,
    sequence,
    time,
    labels=None,
    *,
    clean_mask=None,
    ...,
):
    ...
```

其中：

- c2v：`sequence=[B,4,V,N,C]`，`clean_mask` 缺省或为 `[B,4]` 全 `False`；
- ci2v：调用方先拼接 `image_clean=[B,1,V,N,C]` 和 `target_noisy=[B,5,V,N,C]`，得到 `sequence=[B,6,V,N,C]`；
- ci2v 的 `clean_mask=[B,6]` 为 `[True,False,False,False,False,False]`；
- `clean_mask=True` 表示该 chunk 是 clean，`False` 表示该 chunk 是 noisy/target。

不增加以下结构：

- `condition_frame_embedding`；
- `target_frame_embedding`；
- cross-attention；
- image adapter；
- 额外 gate。

使用现有 `FramePatchEmbed`、attention、MLP、AdaLN 和 view/spatial 处理。统一的 `VRAEVideoDiT` 内部只负责验证 mask、编码统一序列并执行 DiT blocks：

```text
sequence [B,T,V,N,C]
    -> FramePatchEmbed
    -> view embedding
    -> self-attention / AdaLN / MLP
    -> output [B,T,V,N,C]
```

`clean_mask` 是区分 clean/noisy 的显式语义输入；时间值不再承担唯一的角色识别职责。当只传 transport 生成的 sample-level `time` 时，CI2V 的默认时间构造为：

```text
time = [0, t_video, t_video, t_video, t_video, t_video]
```

第 0 个 chunk 的 temporal position 为 0，target chunks 为 1～5，继续使用现有 3D RoPE。clean/noisy 的最终判断以 `clean_mask` 为准，即使未来 clean chunk 的时间值不是 0，也不会被误判为 noisy。

### 时间条件适配

统一的时间输入支持 scalar、`[B]` 和 `[B,T]`，并在进入 time embedder 前保持 batch/chunk 维度：

```text
time scalar       -> expand to [B]
time [B]          -> broadcast to [B,T]
time [B,T]        -> use directly
```

如果只提供 transport 的 `[B]` `time`，模型根据 `clean_mask` 构造逐 chunk 时间：clean 位置置为 0，其他位置使用 sample-level `time`。显式提供 `[B,T]` 的 `time` 时直接使用它，并校验其和序列 chunk 数一致。

时间 embedding 直接支持 `[B]` 和 `[B,T]` 输入，分别返回 `[B,D]` 和 `[B,T,D]`，再展开到对应的 view/spatial tokens。

### 类别条件适配

类别标签也在统一的 `VRAEVideoDiT` 中做形状适配：

```text
labels [B]       -> embed once -> broadcast to all T chunks
```

默认 c2v 和 ci2v 都使用 `[B]` 的 sample-level class label，并广播到完整序列。`drop_mask` 同样只接受 `[B]`，在类别 embedding 生成后再扩展 chunk 维度。

推荐模型边界：

```text
caller input:     sequence [B,6,V,N,C]
caller mask:      clean_mask [B,6]
model input:      sequence + clean_mask
model output:     [B,6,V,N,C] -> caller/engine slice target [B,5,V,N,C]
```

DiT 的 sequence geometry 根据输入模式动态确定：

```text
c2v:  condition_chunks = 0, target_chunks = 4, sequence_chunks = 4
ci2v: condition_chunks = 1, target_chunks = 5, sequence_chunks = 6
```

`full` 和 `base` 输出都恢复为对应的完整序列。c2v 直接使用全部 4 个 chunks；ci2v 由训练/采样调用方切出后 5 个 target chunks，保持 flow matching 的输入输出 shape。

构建器按任务选择模型：

```text
task=c2v  -> vrae_video_dit -> VRAEVideoDiT(sequence_chunks=4)
task=ci2v -> vrae_video_dit -> VRAEVideoDiT(sequence_chunks=6)
```

两种任务共用 `dit.name: vrae_video_dit`，由调用方根据拼接后序列的长度和 `clean_mask` 选择 c2v/ci2v 几何。c2v 在未传 `clean_mask` 时默认所有 chunk 都是 noisy，保持旧调用路径和 checkpoint 兼容。

## 7. 时间条件、dropout 与 loss

训练和采样都复用现有 transport。`training_losses` 接收完整的 clean sequence 和 `[B,T]` 的 `clean_mask`：mask 为 `True` 的 chunk 保持 clean，只有 `False` 的 target chunk 参与加噪和 loss。这样 CI2V 不需要额外的 `ConditionalModel`：

```python
def run_dit_forward(clean, labels, stream_ids, clean_mask):
    losses = transport.training_losses(
        model,
        clean,
        clean_mask=clean_mask,
        model_kwargs={
            "labels": labels,
            "stream_ids": stream_ids,
        },
    )
    return losses, losses["loss"]
```

CI2V 在进入 `run_dit_forward` 前先拼接 `clean = torch.cat((image_clean, target_clean), dim=1)`；transport 内部只对 `clean_mask=False` 的 target chunks 生成 noisy state，因此送入 DiT 的 sequence 等价于 `torch.cat((image_clean, noisy_target), dim=1)`。

clean image condition：

```text
t_image = 0
noise_image = none
image_condition remains clean
```

CI2V 不增加 image dropout，只保留已有的 class dropout 和一个序列语义 mask：

```text
class_drop_mask [B]   # 类别条件 dropout
clean_mask      [B,6]  # [1,0,0,0,0,0]
```

image condition 始终以 clean latent 参与拼接，不置零、不采样 image dropout，也不新增 null image 参数。CFG 的 unconditional branch 只移除类别条件，仍保留 image condition；因此 CFG 主要提供 class guidance，而不是 image guidance。

训练时 c2v 和 ci2v 都调用同一个 `run_dit_forward(clean, labels, stream_ids, clean_mask)`。transport 只对 `clean_mask=False` 的 target 生成 noisy state，并用同一 mask 过滤 loss；DiT 始终接收完整 sequence 并返回完整输出，因此不需要额外的 `ConditionalModel`。

统一的 `VRAEVideoDiT` 返回与输入 sequence 对齐的完整序列：c2v 返回 4 个 chunks，ci2v 返回 6 个 chunks。loss 直接由 transport 根据 `clean_mask` 过滤，首帧位置永远不参与 denoising loss：

```text
loss_mask = [0,1,1,1,1,1]
```

首帧永远不计算 denoising loss。

## 8. Sampling 与解码

1. 从 loader 取得固定的 21 帧和类别；
2. 保存原始首帧 `[B,1,V,3,H,W]`；
3. 对首帧 repeat 4 后编码、pool 和 normalize；
4. 初始化 5 个 target latent chunks 的 noise；
5. 每一步将 clean image latent 与当前 noisy target 沿时间维拼接；
6. 调用方将二者拼接为 `sequence`，并使用 `clean_mask=[1,0,0,0,0,0]` 和 `time=[0,t_video,...,t_video]` 调用 DiT；
7. 只更新后 5 个 target chunks；
8. denormalize target latent，decoder 只解码 20 帧；
9. 沿时间维把原始首帧拼到 decoder 输出前面。

采样增强逻辑完全复用现有实现，不修改 `guidance.py` 和 `transport.py` 的 CFG/IG 算法：

```text
transport.euler_sample
  -> guided_model_forward
  -> conditional model call
  -> unconditional model call（cfg_scale != 1 时）
  -> internal guidance：full/base 先融合
  -> CFG：unconditional + cfg_scale * (conditional - unconditional)
  -> convert_model_prediction
  -> Euler update target state
```

CI2V 只替换 conditional/unconditional model call 的输入准备：两个分支都使用同一个 clean image latent，分别把它和当前 target state 拼接后送入 DiT；两个分支只沿用现有 `class_drop_mask` 的差异。Euler state 始终是 `[B,5,V,N,C]`，不会把 clean image chunk 纳入更新。

```text
decoder output: [B,20,V,3,H,W]
final output:   [B,21,V,3,H,W]
```

最终输出 metadata 记录 `condition_frame_indices`、`target_frame_indices`、`decode_frames=20` 和 `num_views`。

## 9. 代码修改范围

| 文件/模块 | 修改内容 |
|---|---|
| `src/vrae/config.py` | task/trainer 和现有 `data.num_frames` 的 CI2V 兼容校验 |
| `src/vrae/data/reconstruction.py` | 修改 `LiberoVideoDataset`；内部读取额外首帧，并在样本返回值增加首帧/目标字段和 indices |
| `src/vrae/training/common/bounded_loader.py` | 搬运新 batch 字段 |
| `src/vrae/training/common/engine.py` | 双路 stage1、shared statistics、CI2V 的输入拼接适配和首帧 prepend |
| `src/vrae/models/dit/video_dit.py` | 扩展 `VRAEVideoDiT` 原生支持 c2v/ci2v，负责统一 sequence、`[B,T]` clean mask、逐 chunk time/label 适配和完整序列输出 |
| `src/vrae/models/dit/conditioning.py` | 保持已有 class dropout，不增加 image dropout |
| `configs/training/generation/` | 新增 CI2V 配置，保留 c2v 配置 |
| `test/` | shape、mask、时间条件、解码长度和兼容性测试 |

不修改 V-JEPA 预训练权重、tubelet=2、pool group=2、decoder tubelet=4 或多视角排列。

## 10. 实施顺序与验收

1. **配置与数据**：build-only 通过，固定输出 `21 -> 10 -> 5 -> 20`；验证多视角同步和 episode 边界。
2. **Stage1 与 normalizer**：首帧和 target 都经过同一个 stage1/shared normalizer；验证 latent shape 和统计 metadata。
3. **原生 VideoDiT**：验证同一个 `VRAEVideoDiT` 支持 c2v 和 ci2v；调用方传入已拼接的 sequence 与 `[B,T]` clean mask，CI2V 输入为 `[B,6,V,N,C]`、时间为 `[0,t_video,...]`，模型输出完整 `[B,6,V,N,C]`，再由 engine 切出 target `[B,5,V,N,C]`；c2v 回归不变。
4. **训练 loss**：验证首帧不加噪、不计 loss，target 正常 flow matching；不产生 image dropout，CFG 只移除 class 条件并保留 image condition。
5. **Sampling**：验证沿用现有 `euler_sample`、internal guidance 和 CFG；仅在每次送入 DiT 前拼接 clean image，decoder 只收到 5 个 latent chunks、输出 20 帧，最后 prepend 原始首帧得到 `[B,21,V,3,H,W]`。
6. **兼容性**：旧 c2v 配置和 checkpoint 不回归；CI2V checkpoint 的 resume 必须匹配 task、geometry、normalizer 和 view 数。
7. **效果 smoke test**：固定类别和首帧做小规模 overfit，替换首帧后生成结果应发生可观测变化。

## 11. 暂不实现

`ti2v`、`t2v` 的文本编码和训练分支；随机目标长度；latent 混合序列；首帧与未来帧共同构成 condition clip；decoder 解码首帧。
