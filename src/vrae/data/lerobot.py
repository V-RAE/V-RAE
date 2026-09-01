"""LeRobot v3 dataset adapter for V-JEPA 2.1 video inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

from .dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving

from .lerobot3.lerobot_dataset import LeRobotDataset

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def build_temporal_indices(frame_num: int, frame_interval: int) -> list[int]:
    """Return source-frame offsets for a fixed-length sampled clip."""

    frame_num = int(frame_num)
    frame_interval = int(frame_interval)
    if frame_num <= 0 or frame_num % 4:
        raise ValueError("frame_num must be a positive multiple of 4")
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    return [frame_interval * index for index in range(frame_num)]


class LeRobotVideoDataset(LeRobotDataset):
    """Expose LeRobot's native frame loader as V-JEPA 2.1 video samples.

    ``video`` is returned as ``[T,C,H,W]`` for one view or ``[T,V,C,H,W]``
    for multiple views, with float32 RGB values in ``[0, 1]``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str = "libero",
        frame_num: int = 16,
        frame_interval: int = 1,
        camera_keys: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(repo_id, root=Path(root))
        # Match FastWAM's resize-then-center-crop policy while keeping
        # V-RAE's expected float RGB range [0, 1].
        self.image_transforms = transforms.Compose(
            [
                ResizeSmallestSideAspectPreserving(
                    args={"img_w": 256, "img_h": 256},
                ),
                CenterCrop(args={"img_w": 256, "img_h": 256}),
            ]
        )
        self.frame_num = int(frame_num)
        self.frame_interval = int(frame_interval)
        self.camera_keys = self._resolve_camera_keys(camera_keys)
        temporal_indices = build_temporal_indices(self.frame_num, self.frame_interval)
        self.delta_indices = {
            camera["key"]: temporal_indices for camera in self.camera_keys
        }
        for key in ("observation.state", "action"):
            if key in self.meta.features:
                self.delta_indices[key] = temporal_indices
        self.multiview_enabled = len(self.camera_keys) > 1
        self.stream_ids = torch.tensor(
            [item["stream_id"] for item in self.camera_keys], dtype=torch.long
        )
        self.num_views = len(self.camera_keys)
        self.num_streams = max(item["stream_id"] for item in self.camera_keys) + 1
        # Build stable class IDs directly from the dataset's task metadata.
        task_rows = sorted(
            (
                int(row.task_index),
                str(task),
            )
            for task, row in self.meta.tasks.iterrows()
        )
        self.task_index_to_class_id = {
            task_index: class_id
            for class_id, (task_index, _task) in enumerate(task_rows)
        }
        self.class_id_to_task_index = {
            class_id: task_index
            for task_index, class_id in self.task_index_to_class_id.items()
        }
        self.class_names = tuple(task for _task_index, task in task_rows)
        self.num_classes = len(self.class_names)

    def save_class_map(self, path: str | Path) -> Path:
        """Persist the metadata-derived task/class mapping for a training run."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "task_index_to_class_id": {
                str(task_index): class_id
                for task_index, class_id in self.task_index_to_class_id.items()
            },
            "class_id_to_task_index": {
                str(class_id): task_index
                for class_id, task_index in self.class_id_to_task_index.items()
            },
            "class_names": list(self.class_names),
        }
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(output)
        return output

    @staticmethod
    def load_class_map(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_camera_keys(
        self, camera_keys: Sequence[Mapping[str, Any]] | None
    ) -> tuple[dict[str, Any], ...]:
        values = camera_keys or (
            {"key": "observation.images.image", "name": "image", "stream_id": 0},
        )
        return tuple(
            {
                "key": str(camera["key"]),
                "name": str(camera.get("name", camera["key"])),
                "stream_id": int(camera.get("stream_id", index)),
            }
            for index, camera in enumerate(values)
        )

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(int(index))
        episode = int(item["episode_index"])
        task_index = int(item["task_index"])
        try:
            class_id = self.task_index_to_class_id[task_index]
        except KeyError as error:
            raise ValueError(f"unmapped task_index from dataset metadata: {task_index}") from error
        task = self.class_names[class_id]
        episode_meta = self.meta.episodes[episode]
        episode_start = int(episode_meta["dataset_from_index"])
        episode_stop = int(episode_meta["dataset_to_index"])
        current_index = int(item["index"])
        global_indices = torch.tensor(self.delta_indices[self.camera_keys[0]["key"]])
        global_indices = (current_index + global_indices).clamp(
            min=episode_start,
            max=episode_stop - 1,
        )
        views: list[torch.Tensor] = []
        for camera in self.camera_keys:
            view = item[camera["key"]].float()
            if float(view.max()) > 1.0:
                view = view / 255.0
            views.append(view)
        video = torch.stack(views, dim=1)
        result: dict[str, Any] = {
            "video": video,
            "label": class_id,
            "sample_id": f"episode-{episode:06d}",
            "frame_indices": global_indices,
            "task": task,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "state": item["observation.state"],
            "action": item["action"],
            "stream_ids": self.stream_ids.clone()
        }
        return result


__all__ = ["LeRobotVideoDataset"]
