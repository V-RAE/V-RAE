from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class StepProfiler:
    """Collect low-overhead wall/CUDA timings for FastWAM training phases."""

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        window_steps: int = 200,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.window_steps = int(window_steps)
        self._values: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_steps)
        )
        self._pending_cuda: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending_cuda[name].append((start, end))
            return

        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._values[name].append((time.perf_counter() - started_at) * 1000.0)

    def record_wall(self, name: str, milliseconds: float) -> None:
        if self.enabled:
            self._values[name].append(float(milliseconds))

    def averages(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self.device.type == "cuda" and self._pending_cuda:
            torch.cuda.synchronize(self.device)
        for name, events in self._pending_cuda.items():
            self._values[name].extend(start.elapsed_time(end) for start, end in events)
        self._pending_cuda.clear()
        result = {
            name: sum(values) / len(values)
            for name, values in self._values.items()
            if values
        }
        data_wait = sorted(self._values.get("data_wait_ms", ()))
        if data_wait:
            p95_index = max(0, math.ceil(0.95 * len(data_wait)) - 1)
            result["data_wait_p95_ms"] = data_wait[p95_index]
        return result


def instrument_fastwam(model: nn.Module, profiler: StepProfiler) -> None:
    """Time official FastWAM VAE encode and joint denoise calls in place."""

    if not profiler.enabled:
        return
    for method_name, phase_name in (
        ("_encode_video_latents", "vae_encode_ms"),
        ("_joint_denoise_core", "denoise_forward_ms"),
    ):
        original = getattr(model, method_name, None)
        if original is None:
            continue

        def timed(*args, _original=original, _phase_name=phase_name, **kwargs):
            with profiler.phase(_phase_name):
                return _original(*args, **kwargs)

        setattr(model, method_name, timed)
