# DIT 训练数据加载与 RAE 编码瓶颈排查及优化计划

> 目标：定位训练过程中数据读取、视频解码、主机到 GPU 拷贝、RAE 编码、DiT 前向/反向、梯度同步等阶段的耗时和 GPU 空闲区间，并在**不采用“预计算 RAE/VAE latent 持久化到磁盘”**的前提下，提高 DiT 的有效 GPU 利用率和训练吞吐。
>
> 约束：RAE 输出维度 `D` 较大，长期保存 latent 会产生不可接受的存储和 IO 成本。因此本计划不把离线 RAE/VAE latent cache 作为候选方案；只允许短生命周期的内存/显存缓冲、原始视频或解码帧缓存，以及在线 RAE 编码流水线。

## 1. 当前判断与总体策略

如果训练循环每个 step 都执行以下串行路径：

```text
读取原视频 → 解码/resize/crop → H2D → RAE encode → DiT forward/backward → 梯度同步
```

那么 DiT GPU 利用率低是合理现象。视频生成项目中常见的解决方向包括：

1. **先测量再优化**：先用低开销的粗粒度 profiler 判断到底是 `next(dataloader)`、H2D、RAE、通信还是 DiT 本身造成空闲。
2. **细粒度追踪**：只在短时间窗口、少量 rank 上启用 `torch.profiler`，定位 CPU 线程、CUDA kernel、数据拷贝和 NCCL 的时间线。
3. **在线流水线化**：用 DataLoader worker、pinned memory、prefetch、CUDA stream 和有限大小队列把数据准备与 DiT 计算重叠。
4. **降低 RAE 在线成本**：RAE 使用 inference mode、混合精度、micro-batch、chunk/tiling 或独立编码 worker；不把完整 latent 数据集写入磁盘。
5. **控制动态形状和通信**：按 `(帧数, 高, 宽, token/D 成本)` 做 bucket，每个 rank 读取独立分片，避免跨 GPU 交换原始视频或大 latent。

当前 `/Users/zhengshuhang/Desktop/paper` 工作区未发现 `TrainingProfiler` 源文件，因此实施的第 0 步必须在实际训练代码仓库中确认其路径、训练入口和配置系统。

## 2. 参考实现和设计依据

- [CosmosRL Profiler 文档](https://nvidia-cosmos.github.io/cosmos-rl/profiler/overview.html)：使用 `wait_steps`、`warmup_steps`、`active_steps`、`rank_filter` 控制短窗口 profiling，并输出压缩的 Chrome trace，可在 Perfetto 中查看。
- [CosmosRL 配置文档](https://nvidia-cosmos.github.io/cosmos-rl/quickstart/configuration.html)：将 profiler 作为可选配置项，区分常规 profiler 和 Nsys，并提供 `record_shape`、`profile_memory`、`with_stack`、`with_modules` 等开关。
- [Open-Sora 加速文档镜像](https://huggingface.co/spaces/kadirnar/Open-Sora/blob/main/docs/acceleration.md)：指出 T5/VAE 编码会拉低视频训练吞吐；在线编码时采用更快 encoder 和较小 VAE micro-batch。
- [HunyuanVideo-I2V DataLoader](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/main/train_image2video_lora.py)：示例中使用 `DistributedSampler`、`pin_memory`、`prefetch_factor` 和 `persistent_workers`。
- [VeOmni Wan2.1 训练示例](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/examples/wan2.1_I2V_1.3B.md)：明确区分 online 和 offline embedding；本计划只借鉴其数据流水线和分片思路，不采用持久化 latent 方案。
- [LTX-2 视频预处理脚本](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/scripts/process_videos.py)：参考其多进程分片、有限 worker、跳过已完成项目和原子写入思想；本计划只用于原始视频/解码帧预处理，不保存全量 RAE latent。

## 3. 第 0 阶段：代码和实验基线盘点

### 3.1 目标

建立一份可复现的 baseline，明确当前训练循环中每个阶段的位置、输入输出 shape、设备和同步点。

### 3.2 实施步骤

1. 在实际训练仓库中定位：
   - `TrainingProfiler` 定义和调用点；
   - 训练入口、配置文件、DataLoader/IterableDataset；
   - RAE encoder 的加载、dtype、设备和 `forward/encode` 调用；
   - DDP/FSDP/ZeRO/sequence parallel 初始化和通信点。
2. 固定实验变量：GPU 型号、GPU 数量、CPU 核数、存储设备、CUDA/PyTorch 版本、batch size、帧数、分辨率、随机种子。
3. 选择一个代表性数据子集，至少运行：
   - 200～500 steps warm-up；
   - 800～1,000 steps measurement；
   - 相同配置重复 3 次，报告均值、p50、p95 和标准差。
4. 暂时关闭 validation、checkpoint、日志上传和 sample generation，避免把非训练阶段混入 baseline。
5. 保存 baseline 指标：总 step time、samples/s、tokens/s、GPU utilization、显存、CPU 利用率、磁盘吞吐、NCCL 时间和 loss。

### 3.3 交付物

- `baseline_config.*`：固定训练配置；
- `baseline_metrics.json/csv`：粗粒度统计；
- 一张训练阶段时间占比图；
- 一份问题假设排序：`data_wait`、`h2d`、`rae`、`dit`、`allreduce`。

## 4. 第一阶段：通用化 TrainingProfiler，完成粗粒度调研

### 4.1 设计目标

将 profiler 从“只服务某个训练循环的计时器”改成通用的阶段计时和资源采样组件：

- 任意训练代码都可以通过 `with profiler.section("name")` 添加阶段；
- profiler 关闭时接近零开销；
- 支持 CPU wall-clock、可选 CUDA 同步计时、step 统计和 p50/p95；
- 支持 rank 过滤；
- 支持 GPU utilization/显存采样，但不强依赖 NVML；
- 所有功能都通过配置开启，不需要修改训练逻辑；
- 输出 JSON/CSV，便于后续比较不同优化方案。

### 4.2 建议的配置

配置字段应与项目现有 YAML/TOML 系统一致。建议结构如下：

```yaml
profiler:
  enabled: false
  mode: coarse                 # off | coarse | fine | both
  output_dir: outputs/profile
  log_interval_steps: 20
  flush_interval_steps: 100
  rank_filter: [0]
  sample_gpu_util: true
  gpu_util_interval_ms: 200
  sync_cuda_for_sections: false
  collect_memory: true
  collect_disk_io: false
  fine:
    enabled: false
    wait_steps: 1
    warmup_steps: 1
    active_steps: 3
    repeat: 1
    record_shapes: false
    profile_memory: false
    with_stack: false
    with_modules: false
    export_chrome_trace: true
    output_dir: outputs/profile/fine
  nsys:
    enabled: false
    start_step: 100
    end_step: 110
```

默认 `enabled: false`，避免正常训练引入额外开销。粗粒度统计默认不强制 CUDA 全局同步；只有需要精确测量 GPU 阶段时，才在局部 section 开启同步。

### 4.3 建议的通用 API

```python
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import time
import torch


@dataclass
class ProfilerConfig:
    enabled: bool = False
    rank_filter: tuple[int, ...] = (0,)
    log_interval_steps: int = 20
    flush_interval_steps: int = 100
    sync_cuda_for_sections: bool = False
    output_dir: str = "outputs/profile"


class TrainingProfiler:
    """低开销、与训练框架无关的阶段计时器。"""

    def __init__(self, cfg: ProfilerConfig, rank: int = 0):
        self.cfg = cfg
        self.rank = rank
        self.enabled = cfg.enabled and rank in cfg.rank_filter
        self.step = 0
        self.samples = defaultdict(list)

    def start_step(self, step: int):
        self.step = step

    @contextmanager
    def section(self, name: str, sync_cuda: bool | None = None):
        if not self.enabled:
            yield
            return

        sync = self.cfg.sync_cuda_for_sections if sync_cuda is None else sync_cuda
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.samples[name].append((time.perf_counter() - t0) * 1000.0)

    def end_step(self):
        if not self.enabled:
            return
        if self.step % self.cfg.log_interval_steps == 0:
            self.log_summary()
        if self.step % self.cfg.flush_interval_steps == 0:
            self.flush()

    def log_summary(self):
        # 实际实现中增加 p50/p95、占比和 GPU 采样结果。
        summary = {
            name: {
                "count": len(values),
                "mean_ms": sum(values) / max(len(values), 1),
                "p95_ms": sorted(values)[max(int(len(values) * .95) - 1, 0)],
            }
            for name, values in self.samples.items()
            if values
        }
        print(f"[profiler][rank={self.rank}][step={self.step}] {summary}")

    def flush(self):
        if not self.enabled:
            return
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = f"{self.cfg.output_dir}/rank{self.rank}_coarse.json"
        with open(path, "w") as f:
            json.dump({k: list(v) for k, v in self.samples.items()}, f)
```

训练循环中的最小接入方式：

```python
profiler.start_step(step)

with profiler.section("data.next_batch"):
    batch = next(data_iter)

with profiler.section("host_to_device"):
    batch = move_to_cuda(batch, non_blocking=True)

with profiler.section("rae.encode", sync_cuda=True):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = rae.encode(batch["video"])

with profiler.section("dit.forward_backward", sync_cuda=True):
    loss = train_dit_step(latent, batch["condition"])

with profiler.section("optimizer_and_allreduce", sync_cuda=True):
    optimizer_step()

profiler.end_step()
```

### 4.4 粗粒度指标定义

必须统一指标定义，避免优化前后口径不同：

| 指标 | 定义 |
|---|---|
| `step_ms` | 从开始读取当前 batch 到 optimizer step 完成 |
| `data_wait_ms` | `next(data_iter)` 的主线程等待时间 |
| `decode_ms` | 视频解码、抽帧、resize、crop 时间 |
| `h2d_ms` | CPU tensor 到 GPU tensor 的拷贝时间 |
| `rae_ms` | RAE encoder 前向时间，不含 DiT |
| `dit_fwd_bwd_ms` | DiT 前向和反向时间 |
| `allreduce_ms` | 梯度同步或其他 NCCL collective 时间 |
| `optimizer_ms` | optimizer、梯度裁剪和 scaler 时间 |
| `gpu_util_mean/p50/p95` | 采样窗口内 GPU utilization |
| `gpu_mem_peak_mb` | step 或窗口内显存峰值 |
| `cpu_util` | DataLoader/主进程 CPU 利用率 |
| `disk_read_mb_s` | 数据所在存储的读取吞吐 |

另外记录 `frames`、`height`、`width`、`batch_size`、`latent_shape`、`D` 和 `rank`，用于识别动态 shape 或负载不均衡。

### 4.5 第一阶段测试矩阵

先测 baseline，再逐项打开 profiler 配置：

| 实验 | `num_workers` | `pin_memory` | `prefetch_factor` | persistent workers | 目的 |
|---|---:|---|---:|---|---|
| B0 | 0 | false | - | false | 同步基线 |
| B1 | 2 | true | 2 | true | 低 worker 基线 |
| B2 | 4 | true | 2 | true | 常用配置 |
| B3 | 8 | true | 2 | true | 增加解码并行度 |
| B4 | 8 | true | 4 | true | 增加预取深度 |
| B5 | 12 | true | 2 | true | 检查 CPU/IO 是否饱和 |

每个配置只改变一个变量，至少运行 800 个有效 steps。第一阶段不引入 CUDA stream 或 RAE worker，以免无法判断收益来自哪里。

### 4.6 第一阶段判定规则

以下阈值是初始工程判据，不是理论定律，应结合 baseline 调整：

- `data_wait_ms / step_ms > 20%`：优先优化 DataLoader、视频解码、bucket 和预取；
- `h2d_ms / step_ms > 10%`：优先 pinned memory、non-blocking H2D 和双缓冲；
- `rae_ms / step_ms > 20%` 且 RAE 阶段 GPU 利用率高：优先 RAE micro-batch、混合精度或独立编码 worker；
- `allreduce_ms / step_ms > 15%`：检查 rank 负载、bucket、梯度累积和通信重叠；
- 某些 rank 的 `step_ms` p95 明显高于其他 rank：优先检查动态视频长度、数据分片和 bucket；
- GPU utilization 低但上述阶段时间都不高：检查 CPU 调度、同步 barrier、日志/保存、频率限制和 GPU context 切换。

## 5. 第二阶段：CosmosRL 风格的细粒度追踪

### 5.1 目标

粗粒度定位后，只对 baseline 和候选优化方案各采集一个短 trace，回答以下问题：

- CPU 是否在等待视频解码或 DataLoader worker；
- H2D copy 是否与 DiT kernel 重叠；
- RAE kernel 是否阻塞 DiT；
- NCCL 是否与反向计算重叠；
- 是否存在大量小 kernel、同步点或异常的显存分配。

### 5.2 配置方式

参考 CosmosRL 的 `wait/warmup/active/rank_filter`：

```toml
[profiler]
enable_profiler = true
enable_nsys = false

[profiler.sub_profiler_config]
wait_steps = 1
warmup_steps = 1
active_steps = 3
rank_filter = [0]
record_shape = false
profile_memory = false
with_stack = false
with_modules = false
```

建议默认只 profile rank 0；如果怀疑负载不均衡，再单独 profile 最慢 rank。`with_stack`、`profile_memory` 和 `with_modules` 会显著增加开销，只在需要时打开。

### 5.3 简化实现

```python
import os
import torch
from torch.profiler import profile, schedule, ProfilerActivity, record_function


def run_fine_profile(train_one_step, steps, cfg, rank):
    if not cfg.enabled or rank not in cfg.rank_filter:
        for step in range(steps):
            train_one_step(step, profiler=None)
        return

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(cfg.output_dir, exist_ok=True)
    trace_dir = os.path.join(cfg.output_dir, f"rank{rank}")

    with profile(
        activities=activities,
        schedule=schedule(
            wait=cfg.wait_steps,
            warmup=cfg.warmup_steps,
            active=cfg.active_steps,
            repeat=cfg.repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_modules=cfg.with_modules,
    ) as prof:
        for step in range(steps):
            with record_function("train.step"):
                train_one_step(step, profiler=prof)
            prof.step()
```

训练代码中必须使用稳定的 `record_function` 名称：

```python
with record_function("data.next_batch"):
    batch = next(data_iter)

with record_function("video.decode_and_transform"):
    video = decode_and_transform(batch)

with record_function("host_to_device"):
    video = video.to(device, non_blocking=True)

with record_function("rae.encode"):
    latent = encode_with_rae(video)

with record_function("dit.forward"):
    pred = dit(latent, condition)

with record_function("dit.backward"):
    loss.backward()

with record_function("distributed.allreduce"):
    optimizer.step()
```

trace 使用 Perfetto 或 `chrome://tracing` 打开。重点观察：

1. `data.next_batch` 是否覆盖了 DiT 计算区间；
2. H2D memcpy 是否出现在 DiT kernel 之间形成空洞；
3. `rae.encode` 是否和 DiT 完全串行；
4. NCCL 是否只在反向结束后集中出现；
5. CPU worker 是否有长时间空闲或单个 worker 明显落后。

### 5.4 细粒度阶段的执行规范

- 每个候选方案最多 profile 3～5 个 active steps；
- 同一配置先跑 10～20 个 warm-up steps；
- 一次只 profile 1 个 rank，避免 trace 文件过大；
- trace 只用于定位，不用 trace 期间的吞吐评价最终优化效果；
- 需要跨 rank 分析时，分别采集少量 rank，再对齐时间戳；
- 若仍无法解释 GPU 空洞，再使用 Nsys 采集 10～20 steps，且单独执行，不与常规 benchmark 混合。

## 6. 第三阶段：不保存 RAE latent 的优化方案

下面的方案按照“侵入性从低到高”排列。第二阶段根据 profiler 结果选择，不要求全部实施。

### 方案 A：DataLoader 并行、pinned memory 和预取

#### 适用条件

`data_wait_ms` 较高，CPU 解码或文件读取没有跟上 DiT。

#### 实施步骤

1. 将原始视频按短 clip 组织，避免每个样本都随机扫描超长视频；
2. 使用 `DistributedSampler` 或 rank-aware `IterableDataset`，保证每个 rank 读取独立分片；
3. 开启 `pin_memory=True`、`persistent_workers=True` 和 `prefetch_factor`；
4. 逐步测试 `num_workers=2/4/8/12`，以 `data_wait_ms` p95 和整体 step time 作为依据；
5. 检查 `/dev/shm`、worker 内存和文件句柄，避免 worker 过多导致反效果。

#### 简化代码

```python
loader = DataLoader(
    dataset,
    batch_size=local_batch_size,
    sampler=DistributedSampler(dataset, shuffle=True),
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=cfg.num_workers > 0,
    prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    drop_last=True,
)

for batch in loader:
    batch = move_to_device(batch, device, non_blocking=True)
    train_step(batch)
```

#### 验收指标

- `data_wait_ms / step_ms` 降到 10～15% 以下；
- GPU 空闲区间明显减少；
- 不出现 worker OOM、数据重复、样本顺序错误或随机性异常。

### 方案 B：H2D 双缓冲和 CUDA copy stream

#### 适用条件

`h2d_ms` 较高，且 CPU batch 已经可以提前准备好。

#### 实施步骤

1. 保证 DataLoader 输出来自 pinned memory；
2. 用单独 CUDA stream 把下一个 batch 拷贝到 GPU；
3. 当前 batch 计算前调用 `wait_stream`；
4. 对 batch 中的每个 tensor 使用 `non_blocking=True`；
5. 检查 RAE 和 DiT 是否因为显存不足发生反而更多的同步。

#### 简化代码

```python
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = move_to_device(batch, self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


prefetcher = CUDAPrefetcher(loader, device)
while True:
    batch = prefetcher.next()
    if batch is None:
        break
    train_step(batch)
```

#### 风险

CUDA stream 不能消除 RAE 和 DiT 在同一 GPU 上的计算竞争。如果 trace 显示 RAE 与 DiT 仍完全串行，应该转向方案 C/D，而不是继续增加 stream 数量。

### 方案 C：在线 RAE 编码 worker + 有限大小内存队列

#### 适用条件

RAE 编码占据较大比例，且不能接受全量 latent 落盘；有空闲 GPU，或可以将 RAE 编码与 DiT 放到不同 GPU/进程。

#### 实施步骤

1. CPU DataLoader 只负责读取和轻量预处理；
2. RAE worker 批量消费视频并在线编码；
3. 编码结果写入 `maxsize=2～4` 的 bounded queue；
4. DiT trainer 消费 queue 中的 latent；
5. 只保留少量 in-flight batch，不建立全数据集缓存；
6. 若使用独立 GPU，测量 GPU 间传输或 pinned host staging 的代价；
7. 若只有一张 GPU，则先尝试 CUDA stream；只有 trace 显示有可重叠空间时再拆进程。

#### 简化代码

```python
# 概念代码：实际项目中建议使用 multiprocessing + pinned host buffer。
from queue import Queue
from threading import Thread


latent_queue = Queue(maxsize=3)  # 只保留少量 batch，不保存整个数据集


def rae_worker():
    for video, condition in cpu_loader:
        video = video.pin_memory().to(rae_device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = rae.encode(video)
        # 如果 DiT 在另一 GPU，先放入 pinned CPU buffer，再由 trainer 异步上传。
        latent_queue.put((latent.cpu().pin_memory(), condition))
    latent_queue.put(None)


Thread(target=rae_worker, daemon=True).start()
while True:
    item = latent_queue.get()
    if item is None:
        break
    latent_cpu, condition = item
    latent = latent_cpu.to(dit_device, non_blocking=True)
    train_dit_step(latent, condition)
```

#### 注意

- 该方案不把 latent 写入磁盘，只使用有限的内存队列；
- 如果 RAE 和 DiT 共用同一张 GPU，进程拆分本身不会产生计算重叠；
- 队列过大可能导致 CPU RAM 或 pinned memory 爆炸；
- 队列过小会让 DiT 等待 RAE；建议通过 profiler 测试 `maxsize=2/3/4/8`。

### 方案 D：RAE 在线编码 micro-batch、混合精度和 chunk

#### 适用条件

RAE 编码是主要耗时或显存峰值，但暂时没有独立 GPU。

#### 实施步骤

1. 确认 RAE 是否冻结，设置 `eval()` 和 `inference_mode()`；
2. 使用与 checkpoint 兼容的 `bf16/fp16` autocast；
3. 将 DiT batch 拆成较小的 RAE micro-batch，编码后再拼接；
4. 长视频沿时间维度 chunk，必要时使用空间 tile；
5. 评估 `torch.compile` 只对 RAE 编码模块的收益，排除首次编译时间；
6. 保存 latent 时只在当前 step 内保留，不写入长期 cache；
7. 对比 `mode()` 和 `sample()` 的结果，确保与原训练逻辑一致。

#### 简化代码

```python
rae.eval()
rae.requires_grad_(False)


def encode_online(video, micro_batch=1):
    outputs = []
    for video_mb in video.split(micro_batch, dim=0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = rae.encode(video_mb)
        outputs.append(z)
    return torch.cat(outputs, dim=0)
```

#### 验收指标

- RAE 峰值显存下降；
- `rae_ms` p95 不因过度 chunk 明显增加；
- DiT 训练 loss 与 baseline 在相同随机种子下保持一致；
- 不出现 latent scaling、dtype、时间维度或随机采样方式变化。

### 方案 E：按动态 token/D 成本做 bucket 和负载均衡

#### 适用条件

不同视频的帧数、分辨率或 RAE 输出大小差异大，导致某些 rank 经常等待其他 rank。

#### 实施步骤

1. 预先读取每个视频的 `frames/height/width` 元数据，不运行 RAE；
2. 按 `(frames, height, width)` 建立基本 bucket；
3. 如果 RAE 的 `D` 或输出 token 数影响显存，使用成本：

   ```text
   cost = batch_size × latent_frames × latent_height × latent_width × D
   ```

4. 为每个 bucket 设置独立 batch size；
5. 每个 rank 使用相同的 bucket 顺序和近似相同的 cost；
6. 不足一个 batch 的样本采用 `drop_last` 或受控补齐，避免随机复制导致数据分布异常。

#### 简化代码

```python
def bucket_key(meta):
    return (meta["frames"], meta["height"], meta["width"])


buckets = defaultdict(list)
for sample in metadata:
    buckets[bucket_key(sample)].append(sample)

batch_sampler = []
for key, samples in buckets.items():
    bs = batch_size_for_bucket(key)
    batch_sampler.extend(
        [samples[i:i + bs] for i in range(0, len(samples) - bs + 1, bs)]
    )
random.shuffle(batch_sampler)
```

#### 验收指标

- 各 rank 的 step time p95 差异减小；
- `allreduce_ms` 和 rank idle gap 降低；
- padding token 或无效 RAE 计算减少。

### 方案 F：减少不必要的跨 GPU 通信并重叠梯度同步

#### 适用条件

细粒度 trace 显示 NCCL collective 或 rank 间等待占比较高。

#### 实施步骤

1. 使用 `DistributedSampler`，每个 rank 只读取自己的数据；
2. 禁止在训练循环中 all-gather 原始视频或完整 RAE latent；
3. 只同步 DiT 需要的梯度和必要 metadata；
4. 使用 gradient accumulation，并在非最后一个 micro-step 使用 DDP `no_sync()`；
5. 检查 FSDP/ZeRO bucket size，使通信尽量与反向计算重叠；
6. 只有在数据 pipeline 已稳定后，再启用 sequence parallel，避免把模型通信问题和数据问题混在一起。

#### 简化代码

```python
from contextlib import nullcontext

for micro_step in range(grad_accum_steps):
    sync_ctx = (
        model.no_sync()
        if hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
        else nullcontext()
    )
    with sync_ctx:
        loss = train_micro_step(next(data_iter))
        (loss / grad_accum_steps).backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

该方案需要验证优化器语义、梯度缩放和全局 batch size，不能只看 GPU utilization。

## 7. 方案选择决策树

```text
粗粒度 profiler
        │
        ├─ data_wait > 20% ──► 方案 A：worker/prefetch/pinned memory
        │                         + 方案 E：bucket/负载均衡
        │
        ├─ h2d > 10% ───────► 方案 B：copy stream/双缓冲
        │
        ├─ RAE > 20% ───────► 方案 D：micro-batch/chunk/AMP
        │                         │
        │                         └─ 有空闲 GPU ─► 方案 C：独立 RAE worker
        │
        ├─ allreduce > 15% ─► 方案 F：rank 分片/no_sync/通信重叠
        │
        └─ 多个阶段都低但 GPU 仍空闲
             └─ 检查 barrier、日志、checkpoint、CPU 调度、频率限制和存储抖动
```

推荐的实施优先级：

1. 方案 A + 方案 E：风险最低，通常最先验证；
2. 方案 B：只有 H2D 占比高时加入；
3. 方案 D：RAE 显著占用时间/显存时加入；
4. 方案 C：有空闲 GPU 或可以独立部署编码进程时采用；
5. 方案 F：trace 证明通信是瓶颈后再做。

## 8. 实验记录和验收标准

每次实验必须记录：

```text
git commit / 代码版本
配置文件 hash
GPU/CPU/存储环境
数据集版本和样本数
frames、resolution、batch size、D
num_workers、prefetch_factor、pin_memory
RAE dtype、micro-batch、chunk 配置
mean/p50/p95 step time
data_wait/h2d/rae/dit/allreduce 占比
GPU utilization mean/p50/p95
显存峰值、CPU/RAM、磁盘吞吐
loss、梯度 norm、是否出现 NaN/OOM
```

建议的阶段性通过标准：

- 粗粒度 profiler 关闭时训练吞吐与 baseline 差异小于 3%；
- 粗粒度 profiler 开启时额外开销小于 5%；
- 细粒度 profiler 只用于定位，不将其运行速度作为最终 benchmark；
- 选定方案后，median step time 至少改善 20%，或 GPU busy ratio 提高 15 个百分点以上；
- `data_wait_ms / step_ms` 目标低于 10～15%；
- 不增加长期 RAE latent 存储；
- loss 曲线、随机性、latent scaling、checkpoint 恢复和多 GPU 结果保持正确。

## 9. 建议的执行顺序和里程碑

### M0：半天，代码盘点

- 定位实际 `TrainingProfiler` 和训练入口；
- 画出现有训练循环；
- 确认配置解析方式；
- 添加 section 名称但不改变训练行为。

### M1：1 天，粗粒度 profiler

- 完成通用 `TrainingProfiler` API；
- 加入 `enabled/mode/rank_filter/output_dir` 配置；
- 输出阶段耗时 JSON/CSV；
- 加入可选 GPU utilization 和显存采样；
- 完成 baseline 和 DataLoader 测试矩阵。

### M2：半天，第一阶段分析

- 运行 3 次重复实验；
- 计算各阶段 p50/p95 和占比；
- 判断主要瓶颈是数据、H2D、RAE、DiT 还是通信；
- 选出最多两个优化候选，不同时改动多个变量。

### M3：1 天，CosmosRL 风格细粒度 profiler

- 加入 `wait/warmup/active/repeat/rank_filter`；
- 接入 `torch.profiler.schedule` 和 Chrome trace；
- 添加 `record_function` 阶段标记；
- 对 baseline 和候选方案各采集短 trace；
- 用 Perfetto 验证粗粒度结论。

### M4：2～3 天，实施优化方案

按照决策树实施方案 A～F 中被选中的方案。每次只改动一个核心变量，避免把 worker、stream、RAE micro-batch 和通信同时修改。

### M5：半天，回归和最终报告

- 关闭 fine profiler，使用 coarse profiler 或完全关闭 profiler 做正式 benchmark；
- 检查吞吐、GPU 利用率、显存、loss 和多 GPU 一致性；
- 保存 trace、配置、指标和结论；
- 将最终配置设为默认关闭 profiler、按命令行或配置显式开启。

## 10. 需要避免的错误

1. 只看 `nvidia-smi` 的瞬时 GPU utilization，不看 step 时间和 trace；
2. 盲目把 `num_workers` 增大，导致 CPU、RAM、共享内存或磁盘争用；
3. 在没有 pinned memory 时使用 `non_blocking=True`，误以为已经发生异步 H2D；
4. 用 VAE slicing/tiling 解决显存问题后，直接宣称吞吐提升；它们可能降低显存但增加编码时间；
5. 把全量 RAE latent 写到磁盘；本项目约束明确排除该方案；
6. 在 DataLoader worker 内初始化 CUDA 模型，导致多进程 context、显存复制或死锁；
7. 细粒度 profiler 长时间开启，导致 trace 文件过大并污染吞吐数据；
8. 缓存或异步队列改变了 RAE posterior 的 `sample/mode`、scaling factor、dtype 或随机数语义；
9. 只优化平均 step time，不检查 p95 和多 rank 最慢样本；
10. 在数据 pipeline 尚未稳定前启用复杂 sequence parallel，把通信和 IO 问题混为一谈。

## 11. 预期最终架构

最终优先实现以下在线流水线，而不是 RAE latent 离线存储：

```text
CPU DataLoader workers
    ├─ 原始视频读取
    ├─ 解码 / resize / crop
    └─ pinned host batch
             │  prefetch + non_blocking H2D
             ▼
RAE encoder（micro-batch / AMP / chunk；可选独立 GPU worker）
             │  bounded in-memory queue
             ▼
DiT forward / backward
             │
             └─ 梯度同步与 optimizer
```

核心原则是：让 CPU 预处理、H2D、RAE 和 DiT 尽可能形成有限深度的生产者-消费者流水线，同时严格限制 in-flight 数据量，不以牺牲巨大磁盘空间为代价换取 GPU 利用率。
