from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class TrainingProfiler:
    """Collect training-stage latency and CUDA-memory measurements per log window."""

    _MEMORY_PHASES = (
        "batch_start",
        "after_h2d",
        "after_stage1",
        "after_backward",
        "after_update",
    )

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda_enabled = device.type == "cuda"
        self.reset()

    def reset(self) -> None:
        self.data_wait_seconds = 0.0
        self.data_wait_max_seconds = 0.0
        self.data_wait_batches = 0
        self._batch_events: list[tuple[torch.cuda.Event, ...]] = []
        self._update_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._current_events: tuple[torch.cuda.Event, ...] | None = None
        self._last_backward_event: torch.cuda.Event | None = None
        self._allocated_gb: dict[str, list[float]] = defaultdict(list)
        self._reserved_gb: dict[str, list[float]] = defaultdict(list)

    def record_data_wait(self, seconds: float) -> None:
        self.data_wait_seconds += float(seconds)
        self.data_wait_max_seconds = max(self.data_wait_max_seconds, float(seconds))
        self.data_wait_batches += 1

    def _record_memory(self, phase: str) -> None:
        if not self.cuda_enabled:
            return
        gib = float(1024**3)
        self._allocated_gb[phase].append(
            torch.cuda.memory_allocated(self.device) / gib
        )
        self._reserved_gb[phase].append(
            torch.cuda.memory_reserved(self.device) / gib
        )

    def begin_batch(self) -> None:
        if self.cuda_enabled:
            self._current_events = tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(4)
            )
            self._current_events[0].record()
        self._record_memory("batch_start")

    def finish_h2d(self) -> None:
        if self._current_events is not None:
            self._current_events[1].record()
        self._record_memory("after_h2d")

    def finish_stage1(self) -> None:
        if self._current_events is not None:
            self._current_events[2].record()
        self._record_memory("after_stage1")

    def finish_backward(self) -> None:
        if self._current_events is not None:
            self._current_events[3].record()
            self._batch_events.append(self._current_events)
            self._last_backward_event = self._current_events[3]
            self._current_events = None
        self._record_memory("after_backward")

    def finish_update(self) -> None:
        if self._last_backward_event is not None:
            finished = torch.cuda.Event(enable_timing=True)
            finished.record()
            self._update_events.append((self._last_backward_event, finished))
            self._last_backward_event = None
        self._record_memory("after_update")

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
        for phase in self._MEMORY_PHASES:
            sums.extend(
                (
                    sum(self._allocated_gb[phase]),
                    sum(self._reserved_gb[phase]),
                    float(len(self._allocated_gb[phase])),
                )
            )
        maxima = [self.data_wait_max_seconds, float(window_seconds)]
        for phase in self._MEMORY_PHASES:
            maxima.extend(
                (
                    max(self._allocated_gb[phase], default=0.0),
                    max(self._reserved_gb[phase], default=0.0),
                )
            )
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
            "train/window_elapsed_sec": elapsed,
            "train/window_updates": float(updates),
            "train/profiled_batches": float(sum_tensor[6].item()),
            "train/data_wait_ms_mean": float(
                wait_seconds.item()
                / wait_batches.clamp_min(1).item()
                * 1000.0
            ),
            "train/data_wait_ms_max": float(max_tensor[0].item() * 1000.0),
            "train/h2d_ms": float(h2d_sum.item() / batch_count),
            "train/stage1_ms": float(stage1_sum.item() / batch_count),
            "train/dit_ms": float(dit_sum.item() / batch_count),
            "train/update_ms": float(update_sum.item() / update_count),
            "train/updates_per_second": updates_per_second,
            "train/samples_per_second": updates_per_second * int(global_batch_size),
            "memory/allocated_gb": float(max_tensor[-3].item()),
            "memory/reserved_gb": float(max_tensor[-2].item()),
            "memory/max_allocated_gb": float(max_tensor[-1].item()),
        }
        sum_offset = 8
        max_offset = 2
        for phase in self._MEMORY_PHASES:
            allocated_sum, reserved_sum, count = sum_tensor[sum_offset : sum_offset + 3]
            divisor = max(float(count.item()), 1.0)
            metrics[f"memory/{phase}_allocated_gb_mean"] = float(
                allocated_sum.item() / divisor
            )
            metrics[f"memory/{phase}_reserved_gb_mean"] = float(
                reserved_sum.item() / divisor
            )
            metrics[f"memory/{phase}_allocated_gb_max"] = float(
                max_tensor[max_offset].item()
            )
            metrics[f"memory/{phase}_reserved_gb_max"] = float(
                max_tensor[max_offset + 1].item()
            )
            sum_offset += 3
            max_offset += 2
        return metrics


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
