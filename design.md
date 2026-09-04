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
      dropout: 0.10
    video_target:
      frames: 20
      frame_interval: 1
    sampling:
      decode_frames: 20
      prepend_condition_frame: true
  ti2v: {status: reserved, condition: [text, image]}
  t2v: {status: reserved, condition: [text]}

data:
  condition_frames: 1
  target_frames: 20
  read_frames: 21       # 由 condition_frames + target_frames 校验
  frame_interval: 1

dit:
  image_condition: true
  image_condition_injection: temporal_concat
  class_dropout: 0.10
  image_dropout: 0.10

latent_normalizer:
  path: ckpts/libero_videogen/latent_stats/ci2v_shared_20f_multiview.pt
```

校验规则：`read_frames=21`、`target_frames=20`、`latent_frames=5`、`decode_frames=20`，并确保 `target_frames` 能被 `encoder_tubelet * pool_group = 4` 整除。

## 4. 数据加载

新增 CI2V 数据模式，加载同一 episode、同一时间起点的 21 帧：

1. 所有 camera 使用同一组 source-frame indices；
2. `video[0:1]` 保存为 `image_condition`；
3. `video[1:21]` 保存为 `video_target`；
4. 返回 `condition_frame_indices`、`target_frame_indices`、`label` 和 `stream_ids`；
5. episode 不足 21 帧时拒绝该样本，不能跨 episode 补帧；
6. 复用现有 `camera_keys`、resize、center crop 和 class-id 映射。

数据 collator/bounded loader 需要同时搬运 `image_condition` 和原始首帧，保证 pinned-memory 和 H2D 行为一致。

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

## 6. DiT：最小时间拼接改造

不增加以下结构：

- `condition_frame_embedding`；
- `target_frame_embedding`；
- cross-attention；
- image adapter；
- 额外 gate。

使用现有 `FramePatchEmbed`、attention、MLP、AdaLN 和 view/spatial 处理。DiT 内部执行：

```text
image_clean  [B,1,V,N,C]
target_noisy [B,5,V,N,C]
             concat(dim=1)
             -> [B,6,V,N,C]
```

clean/noisy 只通过逐 chunk 的时间条件区分：

```text
frame_time = [0, t_video, t_video, t_video, t_video, t_video]
```

这是唯一新增的 clean/noisy 语义信号，不使用额外 role embedding。第 0 个 chunk 的 temporal position 为 0，target chunks 为 1～5，继续使用现有 3D RoPE。

当前 DiT 的 `time_embedder` 只接受 `[B]`，需要最小扩展为支持 `[B,6]`，并将每个 chunk 的时间 embedding 展开到对应的 view/spatial tokens。label embedding 继续广播到全部序列。

推荐模型边界：

```text
transport input:  noisy_target [B,5,V,N,C]
model condition:  image_clean [B,1,V,N,C]
model internal:   concat -> [B,6,V,N,C]
model output:     slice target chunks [B,5,V,N,C]
```

DiT 的 sequence geometry 需要明确拆分：

```text
condition_chunks = 1
target_chunks = 5
sequence_chunks = 6
```

`full` 和 `base` 输出都只取后 5 个 target chunks，保持现有 flow transport 的输入输出 shape。

## 7. 时间条件、dropout 与 loss

训练时 transport 只对 target latent 加噪：

```text
t_video ~ configured time distribution
x_target = (1 - t_video) * clean_target + t_video * noise
```

image condition：

```text
t_image = 0
noise_image = none
image_condition remains clean
```

CI2V 使用两个独立 mask：

```text
class_drop_mask [B]
image_drop_mask [B]
condition_mask   [B,6] = [1,0,0,0,0,0]
```

image dropout 时，将归一化后的 image slot 替换为全零 clean slot，保持序列长度不变；不新增 null image 参数。CFG 的 unconditional branch 同时移除类别和图像条件。

首版推荐 DiT 只返回 5 个 target chunks，因此 transport 不需要改成完整 6-chunk loss。若后续让 DiT 返回完整序列，再使用：

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
6. 使用 `frame_time=[0,t_video,...,t_video]` 调用 DiT；
7. 只更新后 5 个 target chunks；
8. denormalize target latent，decoder 只解码 20 帧；
9. 沿时间维把原始首帧拼到 decoder 输出前面。

```text
decoder output: [B,20,V,3,H,W]
final output:   [B,21,V,3,H,W]
```

最终输出 metadata 记录 `condition_frame_indices`、`target_frame_indices`、`decode_frames=20` 和 `num_views`。

## 9. 代码修改范围

| 文件/模块 | 修改内容 |
|---|---|
| `src/vrae/config.py` | task/trainer、帧数和 CI2V 配置校验 |
| `src/vrae/data/lerobot.py` | 21 帧读取、首帧/目标拆分、indices |
| `src/vrae/training/common/bounded_loader.py` | 搬运新 batch 字段 |
| `src/vrae/training/common/engine.py` | 双路 stage1、shared statistics、CI2V sampling |
| `src/vrae/models/dit/video_dit.py` | 6-chunk 拼接、逐 chunk time、target 输出切片 |
| `src/vrae/models/dit/conditioning.py` | 独立 class/image dropout mask |
| `configs/training/generation/` | 新增 CI2V 配置，保留 c2v 配置 |
| `test/` | shape、mask、时间条件、解码长度和兼容性测试 |

不修改 V-JEPA 预训练权重、tubelet=2、pool group=2、decoder tubelet=4 或多视角排列。

## 10. 实施顺序与验收

1. **配置与数据**：build-only 通过，固定输出 `21 -> 10 -> 5 -> 20`；验证多视角同步和 episode 边界。
2. **Stage1 与 normalizer**：首帧和 target 都经过同一个 stage1/shared normalizer；验证 latent shape 和统计 metadata。
3. **DiT 拼接**：验证输入 `[B,6,V,N,C]`、时间 `[0,t_video...]`、无 role embedding，输出 target `[B,5,V,N,C]`。
4. **训练 loss**：验证首帧不加噪、不计 loss，target 正常 flow matching，CFG 同时移除 class/image 条件。
5. **Sampling**：验证 decoder 只收到 5 个 latent chunks、输出 20 帧，最终拼接后为 `[B,21,V,3,H,W]`。
6. **兼容性**：旧 c2v 配置和 checkpoint 不回归；CI2V checkpoint 的 resume 必须匹配 task、geometry、normalizer 和 view 数。
7. **效果 smoke test**：固定类别和首帧做小规模 overfit，替换首帧后生成结果应发生可观测变化。

## 11. 暂不实现

`ti2v`、`t2v` 的文本编码和训练分支；随机目标长度；latent 混合序列；首帧与未来帧共同构成 condition clip；decoder 解码首帧。
