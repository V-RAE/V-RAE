from __future__ import annotations

import queue
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import torch

from vrae.training.common.memory import (
    configure_glibc_allocator,
    host_memory_metrics,
    trim_process_heap,
)


class _ExclusiveIterator:
    def __init__(self, iterator: Iterator[Any], release: Any) -> None:
        self._iterator = iterator
        self._release = release
        self._closed = False

    def __iter__(self) -> _ExclusiveIterator:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        finally:
            self._release()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class AsyncPrefetchLoader:
    """One bounded producer thread, matching the proven uni-vug CPU path."""

    _STOP = object()
    _PRODUCER_JOIN_TIMEOUT_SECONDS = 30.0

    def __init__(self, loader: Any, *, prefetch_batches: int = 2) -> None:
        self.loader = loader
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.timeout = 0.0
        self.prefetch_factor = self.prefetch_batches
        self.persistent_workers = False
        self.num_workers = 0

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    @property
    def sampler(self) -> Any:
        return self.loader.sampler

    def release_epoch_memory(self) -> dict[str, float]:
        return self.loader.release_epoch_memory()

    def profile_snapshot(self, *, reset: bool = False) -> dict[str, float]:
        return self.loader.profile_snapshot(reset=reset)

    def close(self) -> None:
        self.loader.close()

    def __iter__(self) -> Iterator[Any]:
        return self._iterate(iter(self.loader))

    def _iterate(self, source: Iterator[Any]) -> Iterator[Any]:
        items: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_batches)
        stopped = threading.Event()
        producer_errors: list[BaseException] = []
        producer_errors_lock = threading.Lock()

        def put(item: Any) -> bool:
            while not stopped.is_set():
                try:
                    items.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def report_error(error: BaseException) -> None:
            with producer_errors_lock:
                producer_errors.append(error)
            put(error)

        def produce() -> None:
            try:
                while not stopped.is_set():
                    try:
                        item = next(source)
                    except StopIteration:
                        break
                    if not put(item):
                        break
            except BaseException as error:
                report_error(error)
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        report_error(error)
                if not stopped.is_set():
                    put(self._STOP)

        producer = threading.Thread(
            target=produce,
            name="vrae-batch-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = items.get()
                if item is self._STOP:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            active_error = sys.exc_info()[1]
            stopped.set()
            while True:
                try:
                    items.get_nowait()
                except queue.Empty:
                    break
            producer.join(timeout=self._PRODUCER_JOIN_TIMEOUT_SECONDS)
            if producer.is_alive():
                raise RuntimeError("Data prefetch producer did not stop within 30 seconds")
            with producer_errors_lock:
                teardown_error = producer_errors[0] if producer_errors else None
            if teardown_error is not None and (
                active_error is None or isinstance(active_error, GeneratorExit)
            ):
                raise teardown_error


@dataclass
class _BatchState:
    index: int
    indices: tuple[int, ...]
    results: list[dict[str, Any] | None]
    next_submit: int = 0
    completed: int = 0


class BoundedThreadBatchLoader:
    """Decode with persistent rank-local threads and bounded in-flight work.

    There are no subprocesses and therefore no CUDA-after-fork state, shared
    memory queues, resource sharers, or per-worker copies of the training
    process. Samples stay CPU uint8 until the pinned batch is transferred.
    """

    def __init__(
        self,
        *,
        dataset: Any,
        batch_sampler: Any,
        rank: int,
        decode_threads: int = 8,
        max_inflight: int = 32,
        max_buffered_batches: int = 4,
        max_decode_attempts_per_batch: int = 2048,
        pin_memory: bool = True,
        glibc_arena_max: int = 2,
        glibc_trim_threshold_bytes: int = 128 * 1024**2,
        trim_heap_each_epoch: bool = True,
        collect_python_each_epoch: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sampler = batch_sampler
        self.rank = int(rank)
        self.decode_threads = max(1, int(decode_threads))
        self.max_inflight = max(self.decode_threads, int(max_inflight))
        self.max_buffered_batches = max(2, int(max_buffered_batches))
        self.max_decode_attempts_per_batch = max(1, int(max_decode_attempts_per_batch))
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.trim_heap_each_epoch = bool(trim_heap_each_epoch)
        self.collect_python_each_epoch = bool(collect_python_each_epoch)
        self.glibc_allocator_configured = configure_glibc_allocator(
            arena_max=int(glibc_arena_max),
            trim_threshold_bytes=int(glibc_trim_threshold_bytes),
        )
        self._iterator_lock = threading.Lock()
        self._closed = False
        self._profile_lock = threading.Lock()
        self._profile_collate_seconds = 0.0
        self._profile_collate_batches = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self.decode_threads,
            thread_name_prefix=f"lerobot-cpu-rank{self.rank}",
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> _ExclusiveIterator:
        if self._closed:
            raise RuntimeError("BoundedThreadBatchLoader is closed")
        if not self._iterator_lock.acquire(blocking=False):
            raise RuntimeError("BoundedThreadBatchLoader does not allow overlapping iterators")
        return _ExclusiveIterator(self._iterate(), self._iterator_lock.release)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self) -> None:
        try:
            if not self._closed:
                self._executor.shutdown(wait=False, cancel_futures=True)
        except BaseException:
            pass

    def release_epoch_memory(self) -> dict[str, float]:
        before = host_memory_metrics()
        trimmed = False
        if self.trim_heap_each_epoch:
            trimmed = trim_process_heap(
                collect_python=self.collect_python_each_epoch,
            )
        after = host_memory_metrics()
        metrics = dict(after)
        metrics["memory/host_allocator_trimmed"] = float(trimmed)
        before_rss = before.get("memory/host_process_rss_gb")
        after_rss = after.get("memory/host_process_rss_gb")
        if before_rss is not None:
            metrics["memory/host_process_rss_before_trim_gb"] = before_rss
        if before_rss is not None and after_rss is not None:
            metrics["memory/host_process_rss_reclaimed_gb"] = max(0.0, before_rss - after_rss)
        return metrics

    def profile_snapshot(self, *, reset: bool = False) -> dict[str, float]:
        result = {}
        dataset_snapshot = getattr(self.dataset, "profile_snapshot", None)
        if callable(dataset_snapshot):
            result.update(dataset_snapshot(reset=reset))
        with self._profile_lock:
            collate_seconds = self._profile_collate_seconds
            collate_batches = self._profile_collate_batches
            if reset:
                self._profile_collate_seconds = 0.0
                self._profile_collate_batches = 0
        result["profile/data_detail.batch_count"] = float(collate_batches)
        result["profile/data_detail.batch.collate_ms"] = (
            collate_seconds / max(collate_batches, 1) * 1000.0
        )
        return result

    def _collate(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        started_at = time.perf_counter()
        if not items:
            raise RuntimeError("Cannot collate an empty video batch")
        expected_shape = tuple(items[0]["video"].shape)
        video_dtype = items[0]["video"].dtype
        if video_dtype not in {torch.uint8, torch.float32}:
            raise RuntimeError(f"Unsupported CPU clip dtype: {video_dtype}")
        video = torch.empty(
            (len(items), *expected_shape),
            dtype=video_dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        for index, item in enumerate(items):
            sample = item["video"]
            if sample.dtype != video_dtype or sample.device.type != "cpu":
                raise RuntimeError("Bounded CPU loader requires uniform CPU clip dtype")
            if tuple(sample.shape) != expected_shape:
                raise RuntimeError(
                    f"Inconsistent decoded clip shape: {tuple(sample.shape)} != {expected_shape}"
                )
            video[index].copy_(sample)
        decode_attempts = [int(item.get("decode_attempts", 1)) for item in items]
        if sum(decode_attempts) > self.max_decode_attempts_per_batch:
            raise RuntimeError(
                "Decoded batch exceeded the configured attempt limit: "
                f"{sum(decode_attempts)} > {self.max_decode_attempts_per_batch}"
            )
        batch = {
            "video": video,
            "label": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
            "sample_id": [str(item["sample_id"]) for item in items],
            "decode_attempts": torch.tensor(
                decode_attempts,
                dtype=torch.long,
            ),
            "_video_normalized": torch.is_floating_point(video),
            "_video_pinned": bool(video.is_pinned()),
        }
        if "stream_ids" in items[0]:
            batch["stream_ids"] = torch.stack([item["stream_ids"] for item in items])
        for key in ("state", "action"):
            if key in items[0]:
                batch[key] = torch.stack([item[key] for item in items])
        if "task" in items[0]:
            batch["task"] = [item["task"] for item in items]
        if "prompt" in items[0]:
            batch["prompt"] = [item["prompt"] for item in items]
        with self._profile_lock:
            self._profile_collate_seconds += time.perf_counter() - started_at
            self._profile_collate_batches += 1
        return batch

    def _iterate(self) -> Iterator[dict[str, Any]]:
        batch_indices = iter(self.sampler)
        states: OrderedDict[int, _BatchState] = OrderedDict()
        pending: dict[Future[Any], tuple[_BatchState, int]] = {}
        exhausted = False
        next_state_index = 0

        def fill_states() -> None:
            nonlocal exhausted, next_state_index
            while not exhausted and len(states) < self.max_buffered_batches:
                try:
                    indices = tuple(int(value) for value in next(batch_indices))
                except StopIteration:
                    exhausted = True
                    break
                if not indices:
                    raise RuntimeError("Batch sampler returned an empty batch")
                states[next_state_index] = _BatchState(
                    index=next_state_index,
                    indices=indices,
                    results=[None] * len(indices),
                )
                next_state_index += 1

        def fill_futures() -> None:
            while len(pending) < self.max_inflight:
                state = next(
                    (
                        candidate
                        for candidate in states.values()
                        if candidate.next_submit < len(candidate.indices)
                    ),
                    None,
                )
                if state is None:
                    break
                position = state.next_submit
                state.next_submit += 1
                future = self._executor.submit(self.dataset.__getitem__, state.indices[position])
                pending[future] = (state, position)

        try:
            fill_states()
            fill_futures()
            while states:
                state = next(iter(states.values()))
                while state.completed < len(state.indices):
                    if not pending:
                        raise RuntimeError("Decode pipeline drained before a batch completed")
                    completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in completed:
                        target, position = pending.pop(future)
                        target.results[position] = future.result()
                        target.completed += 1
                    fill_futures()
                items = [item for item in state.results if item is not None]
                if len(items) != len(state.results):
                    raise RuntimeError("Decode pipeline produced an incomplete batch")
                states.pop(state.index)
                fill_states()
                fill_futures()
                yield self._collate(items)
        finally:
            for future in pending:
                future.cancel()
            pending.clear()
