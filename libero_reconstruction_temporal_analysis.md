# Libero 视频重建时序问题分析

## 结论

`temporal_difference` 一直为 0 是当前配置明确造成的，不是日志计算错误；视频卡顿主要由三件事叠加导致：

1. `temporal_difference` 权重为 `0.0`，训练没有相邻帧一致性约束。
2. Libero 数据加载在重构后从“每个 episode 随机采样完整 clip”变成“每个原始帧都作为起点”，episode 尾部越界帧被重复为最后一帧。
3. decoder 以 4 帧 tubelet 为单位生成视频，现有样本中每个第 4 帧边界的运动误差明显更高，表现为块状跳动。

`Upload data manifest` 对应历史提交 `a7783e8`，不是独立分支。本轮已按该逻辑补齐 Libero 重建数据集和训练入口；下文保留问题定位，并补充实施结果。

## 1. `temporal_difference` 为什么恒为 0

### 损失计算

```python
def temporal_difference_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if reconstructed.shape != target.shape:
        raise ValueError("Reconstruction and target shapes differ")
    if reconstructed.shape[1] < 2:
        return reconstructed.new_zeros(())

    return F.l1_loss(
        reconstructed[:, 1:] - reconstructed[:, :-1],
        target[:, 1:] - target[:, :-1],
    )
```

```python
temporal = (
    temporal_difference_loss(reconstructed, target)
    if self.weights["temporal_difference"]
    else zero
)

total = (
    self.weights["l1"] * l1
    + self.weights["lpips"] * lpips
    + self.weights["gram"] * gram
    + self.weights["temporal_difference"] * temporal
)
```

代码位置：

- [`src/vrae/training/recon_training/losses.py:13`](/zsh/code/V-RAE/src/vrae/training/recon_training/losses.py:13)
- [`src/vrae/training/recon_training/losses.py:393`](/zsh/code/V-RAE/src/vrae/training/recon_training/losses.py:393)

当前配置是：

```yaml
loss:
  l1: 1.0
  lpips: 1.0
  gram: 100.0
  temporal_difference: 0.0
```

配置位置：[`configs/training/vjepa2_1_lerobot.yaml:34`](/zsh/code/V-RAE/configs/training/vjepa2_1_lerobot.yaml:34)

当权重为 0 时，代码直接使用 `zero`，不会调用 `temporal_difference_loss`。因此日志恒为 0 是预期行为。

对保存的 `step-50000.pt` 重新计算后，重建视频的未加权时序差分误差约为 `0.00862`，说明重建结果并不是时序差分天然为 0，而是该项在训练中被关闭了。

## 2. Libero 数据加载的时序采样回归

### 当前 Libero 的采样方式

```python
# 训练时将 frame_interval 转换为：
temporal_indices = [0, interval, 2 * interval, ..., 15 * interval]

self.delta_indices = {
    camera["key"]: temporal_indices
    for camera in self.camera_keys
}
```

```python
query_indices = {
    key: [
        max(episode_start, min(episode_end - 1, idx + delta))
        for delta in deltas
    ]
    for key, deltas in self.delta_indices.items()
}
```

代码位置：

- [`src/vrae/data/lerobot.py:22`](/zsh/code/V-RAE/src/vrae/data/lerobot.py:22)
- [`src/vrae/data/lerobot3/lerobot_dataset.py:206`](/zsh/code/V-RAE/src/vrae/data/lerobot3/lerobot_dataset.py:206)

关键问题：

- 当前数据集长度是所有原始帧数量，而不是有效 clip 数量；
- 每个原始帧都可以作为 clip 起点；
- 超出 episode 末尾的帧会被 `clamp` 到最后一帧；
- `*_is_pad` 标记虽然被生成，但上层没有过滤。

因此 episode 末尾会出现：

```text
真实帧索引: 100, 101, 102, 103, 104, ...
实际取样:   100, 101, 102, 103, 104, ..., 150, 150, 150, 150
```

### 本地 Libero 元数据统计

Libero 数据集共有 1693 个 episode、273465 帧，episode 长度范围为 75 到 505 帧。

| `frame_interval` | 含重复尾帧的 clip | 重复帧槽位占比 |
|---:|---:|---:|
| 1 | 约 9.3% | 约 4.6% |
| 3 | 约 27.9% | 约 13.9% |
| 4 | 约 37.1% | 约 18.6% |

因此 `frame_interval=4` 会把问题进一步放大：超过三分之一的 clip 含尾部重复帧。

另外，当前 Libero 每个 episode 的所有帧都会作为起点，平均一个 episode 产生约 161 个高度重叠的 clip；原始 K600/UCF101 是一个视频记录对应一个随机完整 clip，训练时序多样性更高。

### 原始 K600/UCF101 的采样器

`a7783e8` 提交中的原始采样逻辑是：

```python
required = 1 + (clip_length - 1) * frame_interval

if num_frames < required:
    raise ValueError("implicit padding is disabled")

max_start = num_frames - required
start = random_start(max_start)
indices = start + torch.arange(clip_length) * frame_interval
```

原始采样器明确禁止隐式 padding，只从完整窗口中采样。原始 `VideoDataset` 还会在每个 epoch 使用新的随机种子重新选择 clip；当前 Libero 实现没有相同的 epoch-level clip 随机采样。

## 3. decoder 的 4 帧 tubelet 边界产生跳动

### decoder 输出方式

```python
# latent: [B, T_latent, C, H, W]
predicted = self.prediction(self.norm(hidden))

# 每个 latent token 直接预测 4 帧
tokens = tokens.reshape(
    batch,
    chunks,
    height,
    width,
    tubelet,
    patch,
    patch,
    channels,
)

return tokens.reshape(
    batch,
    chunks * tubelet,
    channels,
    height * patch,
    width * patch,
)
```

代码位置：[`src/vrae/models/decoder.py:415`](/zsh/code/V-RAE/src/vrae/models/decoder.py:415)、[`src/vrae/models/decoder.py:459`](/zsh/code/V-RAE/src/vrae/models/decoder.py:459)

当前时序路径是：

```text
16 RGB 帧
 -> V-JEPA tubelet=2
 -> temporal pool group=2
 -> 4 个 latent 时间块
 -> decoder tubelet=4
 -> 16 RGB 帧
```

这意味着 decoder 的每 4 帧属于同一个输出块，没有重叠帧约束。虽然 attention 是 `full`，但最终仍由 4 个 latent token 分别负责 4 个 tubelet。

### 已保存样本的逐帧统计

对 `step-50000.pt` 的 `[4,16,3,256,256]` 重建样本计算结果：

- 普通块内帧间误差约为 `0.00774`；
- `3→4`、`7→8`、`11→12` 边界误差约为 `0.01214`；
- 边界误差约为块内误差的 `1.57` 倍。

重建视频的运动幅度也呈现周期性：

```text
块内帧间运动: 约 0.002 ~ 0.003
tubelet 边界: 约 0.010 ~ 0.014
```

这与肉眼看到的“每隔几帧跳一下”吻合。开启 temporal difference loss 后，正好可以直接惩罚这种边界不连续。

## 4. `Upload data manifest` 中 K600/UCF101 与 Libero 的差异

| 项目 | Kinetics-600 / UCF101 | 修复前 Libero |
|---|---|---|
| 数据组织 | manifest 中每行一个视频 | LeRobot parquet + episode |
| 每个 dataset item | 一个视频随机采样一个 clip | 一个原始帧作为一个 item |
| clip 长度 | 16 帧重建；20 帧 VideoDiT | 16 帧 |
| 默认间隔 | 重建配置为 `3` | 主 run 为 `1`；interval-4 run 为 `4` |
| 越界处理 | 不允许 padding，短视频直接失败 | 越界 clamp 到 episode 最后一帧 |
| temporal sampling | 每个 epoch 随机变化 | 当前实现基本固定 |
| 视角 | 单视角 `[B,T,C,H,W]` | 双视角 `[B,T,V,C,H,W]` 后展平 |
| 空间增强 | resize + random crop + random flip | resize + center crop，无 random flip |
| batch size | 原始重建配置 `128` | 当前实际 run `4` |
| GAN | epoch 30 起启用 VideoMAE GAN | Libero 重建入口没有 GAN |
| loss | L1 + LPIPS + Gram，temporal 仍为 0 | 同样是 L1 + LPIPS + Gram，temporal 仍为 0 |

原始 K600/UCF101 重建配置位于 `a7783e8` 提交中的 `configs/training/recon_training/vjepa2_1_vrae_training.yaml`：

```yaml
data:
  num_frames: 16
  frame_interval: 3
  sampling: random
  random_flip: true

loss:
  temporal_difference: 0.0

gan:
  enabled: true
  generator_start_epoch: 30
  generator_weight: 0.3
```

当前 Libero 实际 run 的配置：

- [`ckpts/recon_training/vjepa2_1_libero_reconstruction_single_gpu_bs4_50k_detached/resolved_config.yaml`](/zsh/code/V-RAE/ckpts/recon_training/vjepa2_1_libero_reconstruction_single_gpu_bs4_50k_detached/resolved_config.yaml)
- 当前运行目录：`ckpts/recon_training/vjepa2_1_libero_reconstruction_current/`

## 5. 本轮实施内容

- 新增 [`src/vrae/data/reconstruction.py`](/zsh/code/V-RAE/src/vrae/data/reconstruction.py) 中的 `ReconstructionDataset`。它继承 `LeRobotDataset`，复用 LeRobot v3 的 metadata、parquet 和视频解码能力；manifest 只负责 episode、视角、长度和可选 task 映射。
- 重建 manifest 使用 CSV。脚本 [`src/vrae/data/create_reconstruction_manifest.py`](/zsh/code/V-RAE/src/vrae/data/create_reconstruction_manifest.py) 可从任意本地 LeRobot v3 根目录生成 manifest；支持 `view`/`camera_key`、`frame_count`/`length` 等字段别名，缺失长度时使用 episode metadata。
- 每个 episode 每次只随机选择一个视角和一个 clip。固定模式默认 `interval=3`，完整窗口不足时过滤 episode，不做 padding；随机模式可在 `1..4` 中采样，越界时向更小 interval 回退。`set_epoch(epoch)` 会改变 temporal clip 和空间增强的随机序列。
- 输出固定为单视角 `[B,T,C,H,W]`，空间处理为 resize short side、random crop、random flip。生成训练仍使用 [`src/vrae/data/lerobot.py`](/zsh/code/V-RAE/src/vrae/data/lerobot.py)，没有复用重建采样器。
- `configs/training/vjepa2_1_lerobot.yaml` 已切换到 `lerobot_reconstruction` manifest 数据集，保留 `L1 + LPIPS + Gram`，`temporal_difference=0`，并在 epoch 30 启用 VideoMAE GAN。GAN 的 discriminator、optimizer、scheduler 和 loss state 一并写入 checkpoint，支持 resume。

本地验证结果：重建 build-only、生成 build-only、`test/test_lerobot_dataset.py`（2 个测试）和一个真实 Libero manifest 采样 smoke test 均通过。生成的 `data/` 内容保持在 `.gitignore` 中，不提交数据文件。

## 6. 影响因素优先级（修复前诊断）

### 高优先级：时序 loss 被关闭

当前训练目标只有：

```text
像素误差 + LPIPS + Gram
```

这些损失可以让单帧看起来清晰，但不会惩罚第 4 帧和第 5 帧之间突然跳变。

### 高优先级：Libero 尾部 clamp

尤其是 `frame_interval=4`：

- clip 覆盖源视频 61 个源帧；
- 许多起点无法提供完整窗口；
- 尾部大量重复最后一帧。

这会把“冻结画面”直接作为训练目标。

### 高优先级：decoder tubelet 边界没有连续性约束

保存样本已显示第 4 帧边界误差明显更高。该证据与 decoder 的 tubelet=4 输出结构相互印证。

### 中高优先级：过度重叠的 Libero clip

原始 K600/UCF101 每个视频每次随机采一段；当前 Libero 一个 episode 的每一帧都作为起点，训练样本高度重复，容易过拟合静态背景，降低有效运动多样性。

### 中优先级：batch size 从 128 降到 4，但学习率仍为 `2e-4`

当前实际 run：

```yaml
training:
  global_batch_size: 4
  optimizer:
    lr: 0.0002
```

这会显著增加梯度噪声。它不解释 `temporal_difference=0`，但可能让 decoder 更容易学习平均化、静态化的重建结果。

### 中优先级：Libero 没有原始 VideoMAE GAN（修复前）

原始 K600/UCF101 重建在 epoch 30 后启用 VideoMAE 对抗损失；修复前 Libero 代码没有 GAN 分支。本轮已恢复该分支。它主要影响视频感知质量、纹理和动态真实性，不应被当作 temporal loss 的替代品。

### 低优先级：空间增强差异（修复前）

Libero 修复前使用固定的 resize + center crop，没有 random crop 和 random flip；本轮已切换为 resize + random crop + random flip。这会减少空间增强，但通常不是“卡顿”的主因。

## 7. 建议的最小修复顺序（已落实项）

1. `temporal_difference` 保持 0（按你的要求），因此它仍会在日志中恒为 0；如要用时序连续性约束，需要另行设为非零。
2. 已修复 Libero clip 索引，只保留满足以下条件的完整窗口：

   ```text
   episode_length >= 1 + (num_frames - 1) * frame_interval
   ```

   禁止用 episode 最后一帧进行 clamp padding。
3. 当前默认使用 `frame_interval=3`；固定模式下越界 episode 直接失效。需要更密时间变化时可设置 `random_frame_interval=true`，在 1..4 采样并向下回退。
4. 已恢复“每个 episode 一个随机完整 clip”的采样策略，避免每个原始帧都成为训练样本。
5. 添加 tubelet 边界诊断指标：

   ```text
   mean(|Δrecon - Δtarget| at 3→4, 7→8, 11→12)
   /
   mean(|Δrecon - Δtarget| elsewhere)
   ```

   当前保存样本该比例约为 `1.57`，修复后应明显下降。
6. VideoMAE GAN 已设为 epoch 30 启用；仍需在真实训练中观察小 batch 下是否需要调整学习率。

## 8. 已排除的因素

修复前的生成训练没有发现多视角展平顺序错误：

```text
[B,T,V,C,H,W] -> [B,V,T,C,H,W] -> [B*V,T,C,H,W]
```

对应旧实现见历史提交；当前重建路径已直接输出单视角 `[B,T,C,H,W]`，生成路径仍由 [`src/vrae/data/lerobot.py`](/zsh/code/V-RAE/src/vrae/data/lerobot.py) 保持原有多视角契约。

另外，视频保存函数会把输出裁剪到 `[0,1]` 后再编码，见 [`src/vrae/training/common/visualization.py:10`](/zsh/code/V-RAE/src/vrae/training/common/visualization.py:10)。因此 decoder 输出范围略超出 `[0,1]` 会造成高光/暗部裁切，但不是当前卡顿的主要来源。
