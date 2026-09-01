from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from vrae.training.common.contracts import resolve_batch_contract
from vrae.training.common.sampler import StatefulDistributedBatchSampler
from vrae.data.lerobot import LeRobotVideoDataset
from vrae.paths import ProjectPaths


def flatten_multiview_video(video: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,V,C,H,W] clips to the single-view [B*V,T,C,H,W] layout."""

    if video.ndim == 5:
        return video
    if video.ndim != 6:
        raise ValueError(f"Expected [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(video.shape)}")
    batch, time, views, channels, height, width = video.shape
    return (
        video.permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * views, time, channels, height, width)
        .contiguous()
    )


class CudaVideoPrefetchIterator:
    """Overlap the next pinned-memory video transfer with the current training step."""

    def __init__(self, iterator: Iterator[dict[str, Any]], device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA video prefetch requires a CUDA device")
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._next_batch: dict[str, Any] | None = None
        self._preload()

    def __iter__(self) -> CudaVideoPrefetchIterator:
        return self

    def _preload(self) -> None:
        try:
            batch = dict(next(self.iterator))
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            batch["video"] = batch["video"].to(self.device, non_blocking=True)
        self._next_batch = batch

    def __next__(self) -> dict[str, Any]:
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch["video"].record_stream(current_stream)
        self._preload()
        return batch


def build_reconstruction_dataset(config: Mapping[str, Any], paths: ProjectPaths) -> LeRobotVideoDataset:
    data = config["data"]
    root_value = data.get("root")
    root = paths.dataset("lerobot") if not root_value else str(root_value)
    return LeRobotVideoDataset(
        root,
        repo_id=str(data.get("repo_id", "libero")),
        frame_num=int(data.get("num_frames", 16)),
        frame_interval=int(data.get("frame_interval", 1)),
        camera_keys=data.get("camera_keys"),
    )


def collate_reconstruction(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "video": torch.stack([item["video"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "sample_id": [item["sample_id"] for item in items],
        "frame_indices": torch.stack([item["frame_indices"] for item in items]),
        "decode_attempts": torch.tensor(
            [int(item.get("decode_attempts", 1)) for item in items], dtype=torch.long
        ),
    }
    for key in ("state", "action"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    if "task" in items[0]:
        batch["task"] = [item["task"] for item in items]
    if "prompt" in items[0]:
        batch["prompt"] = [item["prompt"] for item in items]
    return batch


def build_reconstruction_loader(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, StatefulDistributedBatchSampler]:
    dataset = build_reconstruction_dataset(config, paths)
    batch = resolve_batch_contract(config["training"], world_size=world_size)
    sampler = StatefulDistributedBatchSampler(
        len(dataset),
        batch.local_micro_batch_size,
        rank=rank,
        world_size=world_size,
        seed=int(config["data"].get("seed", 3407)),
        shuffle=True,
        drop_last=True,
        gradient_accumulation_steps=batch.gradient_accumulation_steps,
    )
    num_workers = int(config["training"].get("num_workers", 4))
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
        loader_options["persistent_workers"] = bool(
            config["training"].get("persistent_workers", False)
        )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_reconstruction,
        generator=sampler.loader_generator,
        **loader_options,
    )
    return loader, sampler
