# VideoDiT 单 Encoder Block Profile

## 测试信息

| 项目 | 值 |
|---|---|
| 测试文件 | `test/test_dit_block_profile.py` |
| block 类型 | `AdaLNZeroDDTBlock` |
| batch / sequence length | `B=8 / L=1280` |
| hidden size / heads | `D=1536 / 24` |
| 精度 | BF16 |
| 执行方式 | 1 次 warmup + 连续 100 次 |
| GPU 时间 | CUDA Event，每次调用平均值 |
| GPU 利用率 | NVML 1ms 采样均值 |

## 测试结果

| 子模块 | CUDA 用时/ms | Python wall/ms | SM 利用率 | 显存控制器利用率 | NVML 样本数 |
|---|---:|---:|---:|---:|---:|
| AdaLN modulation | 1.711 | 1.808 | 62.58% | 32.75% | 12 |
| norm1 | 1.366 | 1.284 | 45.04% | 23.44% | 27 |
| attention | 4.044 | 4.590 | 47.33% | 24.68% | 242 |
| norm2 | 0.275 | 0.211 | 51.11% | 26.67% | 18 |
| MLP | 0.872 | 0.409 | 38.30% | 19.90% | 10 |

## 数据分析

- attention 是单个 encoder block 中 CUDA 用时最长的子模块，平均 `4.044 ms`，约占上述子模块 CUDA 时间总和的 43.8%。
- AdaLN modulation 的 SM 利用率最高，为 `62.58%`，但用时短，因此不是主要耗时来源。
- attention 的 SM 利用率为 `47.33%`，显存控制器利用率为 `24.68%`，不属于“耗时长但 GPU 利用率极低”。
- MLP 的 CUDA 用时 `0.872 ms`，低于 attention；norm2 用时最低，仅 `0.275 ms`。
- Python wall 与 CUDA Event 接近，说明在连续 100 次 workload 下异步误差已明显小于单次调用测试。
- NVML 样本数因各子模块持续时间不同而不同；attention 样本最多（242），其利用率均值可信度最高。

## 结论

单 block 测试表明，attention 确实是主要耗时子模块，但其 GPU 利用率约 47.3%，不是 GPU 空转。之前训练 profile 中 attention 低利用率的现象主要由短阶段 NVML 覆盖不足、Python 异步计时和阶段边界归属造成；评估真实 GPU 时间应优先使用 CUDA Event。

## Block 优化对比（实际 CUDA 测试）

测试文件 `test/test_dit_block_optimizations.py` 使用相同随机输入、相同初始权重和 BF16 autocast，CUDA Event 统计 20 次平均 forward 时间；各变体通过 forward 数值校验，组合变体额外通过输入梯度校验。

| 版本 | CUDA forward/ms | 相对基线 | 最大输出误差 | 最大输入梯度误差 |
|---|---:|---:|---:|---:|
| 独立 Q/K/V + 独立 MLP（SDPA 基线） | 3.743 | 1.000x | — | — |
| QKV 合并为一个 Linear | 3.011 | 1.243x | 0 | — |
| MLP 两个输入 Linear 合并 | 2.854 | 1.311x | 0 | — |
| QKV + MLP 同时合并 | 2.744 | 1.364x | 0 | 0 |
| SDPA attention | 4.356 | 1.000x | — | — |
| flash-attn v2 attention | 3.395 | 1.283x | 0 | 0 |

### 分析

- 在完整 block 上，单独合并 QKV、单独合并 MLP 分别得到 24.3% 和 31.1% 的 speedup；两者同时使用得到 36.4% 加速，收益不是简单相加。
- flash-attn v2 在本次 attention-only 对比中比 PyTorch SDPA 快约 28.3%。输出和输入梯度的最大绝对误差均为 0，说明测试中两者实际产生了逐元素一致的结果。
- 两组测试的 block/输入配置和 warmup 次数不同，绝对毫秒数不直接横向比较；应使用各自测试内的相对 speedup。
- 结果是单卡 H100、BF16、无 dropout/attention mask 的测量。flash-attn 编译时仅包含 `sm90`，该 wheel 不适用于其他计算能力的 GPU。
