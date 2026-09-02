"""Standalone semantic GPU profile for one VideoDiT encoder block.

Run with: ``pytest -s test/test_dit_block_profile.py``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import pytest
import torch


class _NvmlSampler:
    def __init__(self, interval_seconds: float = 0.001) -> None:
        try:
            import pynvml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("pynvml is required for this GPU profile test") from error
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval_seconds = interval_seconds
        self.phase: str | None = None
        self.running = False
        self.lock = threading.Lock()
        self.values: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.thread.join()
        self._nvml.nvmlShutdown()

    def set_phase(self, phase: str | None) -> None:
        with self.lock:
            self.phase = phase

    def _loop(self) -> None:
        while self.running:
            with self.lock:
                phase = self.phase
            if phase is not None:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                value = self.values[phase]
                value[0] += float(utilization.gpu)
                value[1] += float(utilization.memory)
                value[2] += 1.0
            time.sleep(self.interval_seconds)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_encoder_block_gpu_profile() -> None:
    from vrae.models.dit.blocks import AdaLNZeroDDTBlock

    device = torch.device("cuda")
    block = AdaLNZeroDDTBlock(
        hidden_size=1536,
        num_heads=24,
        mlp_ratio=4.0,
        attention_backend="auto",
    ).to(device).eval()
    value = torch.randn(8, 1280, 1536, device=device, dtype=torch.bfloat16)
    condition = torch.randn(8, 1536, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(1280, 3, device=device, dtype=torch.long)

    sampler = _NvmlSampler()
    timings: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    wall_times: dict[str, float] = defaultdict(float)
    modules = {
        "adaln_modulation": block.adaln_modulation,
        "norm1": block.norm1,
        "attention": block.attn,
        "norm2": block.norm2,
        "mlp": block.mlp,
    }
    hooks = []
    for name, module in modules.items():
        def pre(_module, _inputs, phase=name):
            sampler.set_phase(phase)
            _module._profile_wall_start = time.perf_counter()
            _module._profile_start = torch.cuda.Event(enable_timing=True)
            _module._profile_start.record()

        def post(_module, _inputs, _output, phase=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            wall_times[phase] += time.perf_counter() - _module._profile_wall_start
            timings[phase].append((_module._profile_start, end))
            sampler.set_phase(None)

        hooks.extend(
            [
                module.register_forward_pre_hook(pre),
                module.register_forward_hook(post),
            ]
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        block(value, condition, positions)
    torch.cuda.synchronize(device)
    sampler.start()
    repetitions = 100
    for _ in range(repetitions):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            block(value, condition, positions)
    torch.cuda.synchronize(device)
    sampler.stop()
    for hook in hooks:
        hook.remove()

    print("phase\tcuda_ms\twall_ms\tsm_mean\tmemory_mean\tnvml_samples")
    for name in modules:
        cuda_ms = sum(start.elapsed_time(end) for start, end in timings[name]) / repetitions
        wall_ms = wall_times[name] * 1000.0 / repetitions
        sm_sum, memory_sum, samples = sampler.values[name]
        sm_mean = sm_sum / samples if samples else float("nan")
        memory_mean = memory_sum / samples if samples else float("nan")
        print(
            f"{name}\t{cuda_ms:.3f}\t{wall_ms:.3f}\t{sm_mean:.2f}\t"
            f"{memory_mean:.2f}\t{int(samples)}"
        )
