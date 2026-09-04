from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class TrainingProfiler:
    """Collect training-stage latency and CUDA-memory measurements per log window."""

    def __init__(
        self,
        device: torch.device,
        *,
        gpu_util_enabled: bool | None = None,
        gpu_util_interval_seconds: float = 0.2,
        synchronize_phases: bool = False,
    ) -> None:
        self.device = device
        self.cuda_enabled = device.type == "cuda"
        self.gpu_util_enabled = (
            os.environ.get("VRAE_GPU_UTIL_PROFILE", "0").lower()
            in {"1", "true", "yes", "on"}
            if gpu_util_enabled is None
            else bool(gpu_util_enabled)
        )
        if gpu_util_interval_seconds <= 0:
            raise ValueError("gpu_util_interval_seconds must be positive")
        self.gpu_util_interval_seconds = float(gpu_util_interval_seconds)
        self.synchronize_phases = bool(synchronize_phases)
        self._phase_specs: dict[str, bool] = {}
        self._phase_lock = threading.Lock()
        self._active_phase: str | None = None
        self._active_gpu_phase: str | None = None
        self._manual_phase_started_at: dict[str, float] = {}
        self._manual_phase_previous: tuple[str | None, str | None] | None = None
        self._gpu_sampler_thread: threading.Thread | None = None
        self._gpu_sampler_running = False
        self._nvml = None
        self._nvml_handle = None
        self.reset()

    def __call__(
        self,
        phase: str,
        *,
        gpu_util: bool = False,
        synchronize: bool | None = None,
    ):
        """Register a phase and return a decorator that measures its duration."""

        phase_name = str(phase).strip()
        if not phase_name:
            raise ValueError("Profiler phase name must be non-empty")
        collect_gpu_util = bool(gpu_util)
        synchronize_phase = (
            self.synchronize_phases if synchronize is None else bool(synchronize)
        )
        registered = self._phase_specs.get(phase_name)
        if registered is not None and registered != collect_gpu_util:
            raise ValueError(
                f"Profiler phase {phase_name!r} was registered with a different gpu_util setting"
            )
        self._phase_specs[phase_name] = collect_gpu_util
        self._phase_duration_seconds.setdefault(phase_name, 0.0)
        self._phase_duration_count.setdefault(phase_name, 0)
        self._phase_gpu_sm_sum.setdefault(phase_name, 0.0)
        self._phase_gpu_memory_sum.setdefault(phase_name, 0.0)
        self._phase_gpu_sample_count.setdefault(phase_name, 0)
        self._phase_allocated_gb_sum.setdefault(phase_name, 0.0)
        self._phase_allocated_gb_max.setdefault(phase_name, 0.0)
        self._phase_reserved_gb_sum.setdefault(phase_name, 0.0)
        self._phase_reserved_gb_max.setdefault(phase_name, 0.0)

        def decorate(function):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                if synchronize_phase and self.cuda_enabled:
                    torch.cuda.synchronize(self.device)
                started_at = time.perf_counter()
                with self._phase_lock:
                    previous_phase = self._active_phase
                    previous_gpu_phase = self._active_gpu_phase
                    self._active_phase = phase_name
                    if collect_gpu_util:
                        self._active_gpu_phase = phase_name
                try:
                    return function(*args, **kwargs)
                finally:
                    if synchronize_phase and self.cuda_enabled:
                        torch.cuda.synchronize(self.device)
                    elapsed = time.perf_counter() - started_at
                    with self._phase_lock:
                        self._phase_duration_seconds[phase_name] += elapsed
                        self._phase_duration_count[phase_name] += 1
                        if self.cuda_enabled and collect_gpu_util:
                            allocated = torch.cuda.memory_allocated(self.device) / float(1024**3)
                            reserved = torch.cuda.memory_reserved(self.device) / float(1024**3)
                            self._phase_allocated_gb_sum[phase_name] += allocated
                            self._phase_allocated_gb_max[phase_name] = max(
                                self._phase_allocated_gb_max[phase_name],
                                allocated,
                            )
                            self._phase_reserved_gb_sum[phase_name] += reserved
                            self._phase_reserved_gb_max[phase_name] = max(
                                self._phase_reserved_gb_max[phase_name],
                                reserved,
                            )
                        self._active_phase = previous_phase
                        self._active_gpu_phase = previous_gpu_phase

            return wrapped

        return decorate

    def start(self) -> None:
        """Start optional NVML sampling for phases registered with gpu_util=True."""

        if not self.cuda_enabled or not self.gpu_util_enabled:
            return
        if self._gpu_sampler_running:
            return
        gpu_phases = [name for name, enabled in self._phase_specs.items() if enabled]
        if not gpu_phases:
            return
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError(
                "GPU utilization profiling requires the pynvml package"
            ) from error
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.device.index or 0
        )
        self._gpu_sampler_running = True
        self._gpu_sampler_thread = threading.Thread(
            target=self._sample_gpu_utilization,
            name="vrae-gpu-util-profiler",
            daemon=True,
        )
        self._gpu_sampler_thread.start()

    def stop(self) -> None:
        """Stop optional NVML sampling and release its handle."""

        if not self._gpu_sampler_running:
            return
        self._gpu_sampler_running = False
        if self._gpu_sampler_thread is not None:
            self._gpu_sampler_thread.join(timeout=1.0)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
        self._gpu_sampler_thread = None
        self._nvml = None
        self._nvml_handle = None

    def _sample_gpu_utilization(self) -> None:
        while self._gpu_sampler_running:
            with self._phase_lock:
                phase = self._active_gpu_phase
                collect_gpu_util = bool(self._phase_specs.get(phase, False))
            if phase is not None and collect_gpu_util:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(
                    self._nvml_handle
                )
                with self._phase_lock:
                    self._phase_gpu_sm_sum[phase] += float(utilization.gpu)
                    self._phase_gpu_memory_sum[phase] += float(utilization.memory)
                    self._phase_gpu_sample_count[phase] += 1
            time.sleep(self.gpu_util_interval_seconds)

    def reset(self) -> None:
        self.data_wait_seconds = 0.0
        self.data_wait_max_seconds = 0.0
        self.data_wait_batches = 0
        self._batch_events: list[tuple[torch.cuda.Event, ...]] = []
        self._update_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._current_events: tuple[torch.cuda.Event, ...] | None = None
        self._last_backward_event: torch.cuda.Event | None = None
        self._phase_duration_seconds: dict[str, float] = {}
        self._phase_duration_count: dict[str, int] = {}
        self._phase_gpu_sm_sum: dict[str, float] = {}
        self._phase_gpu_memory_sum: dict[str, float] = {}
        self._phase_gpu_sample_count: dict[str, int] = {}
        self._phase_allocated_gb_sum: dict[str, float] = {}
        self._phase_allocated_gb_max: dict[str, float] = {}
        self._phase_reserved_gb_sum: dict[str, float] = {}
        self._phase_reserved_gb_max: dict[str, float] = {}
        self._manual_phase_started_at = {}
        self._manual_phase_previous = None
        with self._phase_lock:
            for phase in self._phase_specs:
                self._phase_duration_seconds[phase] = 0.0
                self._phase_duration_count[phase] = 0
                self._phase_gpu_sm_sum[phase] = 0.0
                self._phase_gpu_memory_sum[phase] = 0.0
                self._phase_gpu_sample_count[phase] = 0
                self._phase_allocated_gb_sum[phase] = 0.0
                self._phase_allocated_gb_max[phase] = 0.0
                self._phase_reserved_gb_sum[phase] = 0.0
                self._phase_reserved_gb_max[phase] = 0.0

    def record_data_wait(self, seconds: float) -> None:
        self.data_wait_seconds += float(seconds)
        self.data_wait_max_seconds = max(self.data_wait_max_seconds, float(seconds))
        self.data_wait_batches += 1

    def register_phase(self, phase: str, *, gpu_util: bool = False) -> None:
        """Register a manually measured phase without wrapping a function."""

        phase_name = str(phase).strip()
        if not phase_name:
            raise ValueError("Profiler phase name must be non-empty")
        collect_gpu_util = bool(gpu_util)
        registered = self._phase_specs.get(phase_name)
        if registered is not None and registered != collect_gpu_util:
            raise ValueError(
                f"Profiler phase {phase_name!r} was registered with a different gpu_util setting"
            )
        self._phase_specs[phase_name] = collect_gpu_util
        self._phase_duration_seconds.setdefault(phase_name, 0.0)
        self._phase_duration_count.setdefault(phase_name, 0)
        self._phase_gpu_sm_sum.setdefault(phase_name, 0.0)
        self._phase_gpu_memory_sum.setdefault(phase_name, 0.0)
        self._phase_gpu_sample_count.setdefault(phase_name, 0)
        self._phase_allocated_gb_sum.setdefault(phase_name, 0.0)
        self._phase_allocated_gb_max.setdefault(phase_name, 0.0)
        self._phase_reserved_gb_sum.setdefault(phase_name, 0.0)
        self._phase_reserved_gb_max.setdefault(phase_name, 0.0)

    def begin_phase(self, phase: str) -> None:
        """Begin a manually measured phase, such as waiting for a batch."""

        phase_name = str(phase).strip()
        if phase_name not in self._phase_specs:
            raise ValueError(f"Profiler phase {phase_name!r} was not registered")
        with self._phase_lock:
            if self._active_phase is not None:
                raise RuntimeError(
                    f"Cannot begin phase {phase_name!r} while {self._active_phase!r} is active"
                )
            self._manual_phase_previous = (
                self._active_phase,
                self._active_gpu_phase,
            )
            self._active_phase = phase_name
            if self._phase_specs[phase_name]:
                self._active_gpu_phase = phase_name
            self._manual_phase_started_at[phase_name] = time.perf_counter()

    def end_phase(self, phase: str) -> None:
        """Finish a manually measured phase."""

        phase_name = str(phase).strip()
        with self._phase_lock:
            if self._active_phase != phase_name:
                raise RuntimeError(f"Profiler phase {phase_name!r} is not active")
            started_at = self._manual_phase_started_at.pop(phase_name)
            elapsed = time.perf_counter() - started_at
            self._phase_duration_seconds[phase_name] += elapsed
            self._phase_duration_count[phase_name] += 1
            if self.cuda_enabled and self._phase_specs[phase_name]:
                allocated = torch.cuda.memory_allocated(self.device) / float(1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / float(1024**3)
                self._phase_allocated_gb_sum[phase_name] += allocated
                self._phase_allocated_gb_max[phase_name] = max(
                    self._phase_allocated_gb_max[phase_name],
                    allocated,
                )
                self._phase_reserved_gb_sum[phase_name] += reserved
                self._phase_reserved_gb_max[phase_name] = max(
                    self._phase_reserved_gb_max[phase_name],
                    reserved,
                )
            previous_phase, previous_gpu_phase = self._manual_phase_previous or (None, None)
            self._active_phase = previous_phase
            self._active_gpu_phase = previous_gpu_phase
            self._manual_phase_previous = None

    def begin_batch(self) -> None:
        if self.cuda_enabled:
            self._current_events = tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(4)
            )
            self._current_events[0].record()

    def finish_h2d(self) -> None:
        if self._current_events is not None:
            self._current_events[1].record()

    def finish_stage1(self) -> None:
        if self._current_events is not None:
            self._current_events[2].record()

    def finish_backward(self) -> None:
        if self._current_events is not None:
            self._current_events[3].record()
            self._batch_events.append(self._current_events)
            self._last_backward_event = self._current_events[3]
            self._current_events = None

    def finish_update(self) -> None:
        if self._last_backward_event is not None:
            finished = torch.cuda.Event(enable_timing=True)
            finished.record()
            self._update_events.append((self._last_backward_event, finished))
            self._last_backward_event = None

    def summarize(
        self,
        *,
        window_seconds: float,
        updates: int,
        global_batch_size: int,
    ) -> dict[str, float]:
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)

        phase_ms = [0.0, 0.0, 0.0, 0.0]
        if self._batch_events:
            phase_ms[0] = sum(start.elapsed_time(h2d) for start, h2d, _, _ in self._batch_events)
            phase_ms[1] = sum(h2d.elapsed_time(stage1) for _, h2d, stage1, _ in self._batch_events)
            phase_ms[2] = sum(
                stage1.elapsed_time(backward)
                for _, _, stage1, backward in self._batch_events
            )
        if self._update_events:
            phase_ms[3] = sum(start.elapsed_time(end) for start, end in self._update_events)

        sums = [
            self.data_wait_seconds,
            float(self.data_wait_batches),
            *phase_ms,
            float(len(self._batch_events)),
            float(len(self._update_events)),
        ]
        maxima = [self.data_wait_max_seconds, float(window_seconds)]
        if self.cuda_enabled:
            maxima.extend(
                (
                    torch.cuda.memory_allocated(self.device) / float(1024**3),
                    torch.cuda.memory_reserved(self.device) / float(1024**3),
                    torch.cuda.max_memory_allocated(self.device) / float(1024**3),
                )
            )
        else:
            maxima.extend((0.0, 0.0, 0.0))

        sum_tensor = torch.tensor(sums, dtype=torch.float64, device=self.device)
        max_tensor = torch.tensor(maxima, dtype=torch.float64, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_tensor)
            torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)

        wait_seconds, wait_batches = sum_tensor[:2]
        h2d_sum, stage1_sum, dit_sum, update_sum = sum_tensor[2:6]
        batch_count = max(float(sum_tensor[6].item()), 1.0)
        update_count = max(float(sum_tensor[7].item()), 1.0)
        elapsed = max(float(max_tensor[1].item()), 1.0e-9)
        updates_per_second = int(updates) / elapsed
        metrics = {
            "profile/window_elapsed_sec": elapsed,
            "profile/window_updates": float(updates),
            "profile/profiled_batches": float(sum_tensor[6].item()),
            "profile/data_wait_ms_mean": float(
                wait_seconds.item()
                / wait_batches.clamp_min(1).item()
                * 1000.0
            ),
            "profile/data_wait_ms_max": float(max_tensor[0].item() * 1000.0),
            "profile/h2d_ms": float(h2d_sum.item() / batch_count),
            "profile/stage1_ms": float(stage1_sum.item() / batch_count),
            "profile/dit_ms": float(dit_sum.item() / batch_count),
            "profile/update_ms": float(update_sum.item() / update_count),
            "profile/updates_per_second": updates_per_second,
            "profile/samples_per_second": updates_per_second * int(global_batch_size),
            "profile/allocated_gb": float(max_tensor[-3].item()),
            "profile/reserved_gb": float(max_tensor[-2].item()),
            "profile/max_allocated_gb": float(max_tensor[-1].item()),
        }
        with self._phase_lock:
            for phase, collect_gpu_util in self._phase_specs.items():
                duration_count = self._phase_duration_count[phase]
                if duration_count:
                    metrics[f"profile/{phase}.duration_ms"] = (
                        self._phase_duration_seconds[phase]
                        / duration_count
                        * 1000.0
                    )
                    metrics[f"profile/{phase}.duration_count"] = float(duration_count)
                    if collect_gpu_util:
                        metrics[f"profile_memory/{phase}.allocated_gb.mean"] = (
                            self._phase_allocated_gb_sum[phase] / duration_count
                        )
                        metrics[f"profile_memory/{phase}.allocated_gb.max"] = (
                            self._phase_allocated_gb_max[phase]
                        )
                        metrics[f"profile_memory/{phase}.reserved_gb.mean"] = (
                            self._phase_reserved_gb_sum[phase] / duration_count
                        )
                        metrics[f"profile_memory/{phase}.reserved_gb.max"] = (
                            self._phase_reserved_gb_max[phase]
                        )
                if collect_gpu_util:
                    sample_count = self._phase_gpu_sample_count[phase]
                    if sample_count:
                        metrics[f"profile_gpu/{phase}.sm_util.mean"] = (
                            self._phase_gpu_sm_sum[phase] / sample_count
                        )
                        metrics[f"profile_gpu/{phase}.memory_util.mean"] = (
                            self._phase_gpu_memory_sum[phase] / sample_count
                        )
                        metrics[f"profile_gpu/{phase}.sample_count"] = float(sample_count)
        return metrics


def save_profile_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically overwrite the single JSON profile artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def run_fine_profile(
    train_one_step: Callable[[int], Any],
    *,
    steps: int,
    enabled: bool = False,
    rank: int = 0,
    rank_filter: Sequence[int] = (0,),
    output_dir: str | Path = "outputs/profile/fine",
    wait_steps: int = 1,
    warmup_steps: int = 1,
    active_steps: int = 3,
    repeat: int = 1,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    with_modules: bool = False,
) -> None:
    """Run a short torch.profiler window without changing normal training.

    The callback owns one training step. Profiling is rank-filtered and disabled
    by default so callers can keep this helper out of the steady-state path.
    """

    if not enabled or int(rank) not in {int(value) for value in rank_filter}:
        for step in range(int(steps)):
            train_one_step(step)
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_dir = Path(output_dir) / f"rank{int(rank)}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(wait_steps),
            warmup=int(warmup_steps),
            active=int(active_steps),
            repeat=int(repeat),
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(record_shapes),
        profile_memory=bool(profile_memory),
        with_stack=bool(with_stack),
        with_modules=bool(with_modules),
    ) as profiler:
        for step in range(int(steps)):
            with torch.profiler.record_function("train.step"):
                train_one_step(step)
            profiler.step()
