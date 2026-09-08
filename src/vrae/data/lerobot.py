"""LeRobot v3 dataset adapter for V-JEPA 2.1 video inputs."""

from __future__ import annotations

import hashlib
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
    """Return source-frame offsets for a fixed-length sampled clip.

    In addition to ordinary four-frame groups, allow ``1 + 4k`` frames for
    image-conditioned video tasks such as FastWAM ``ci2va``.
    """

    frame_num = int(frame_num)
    frame_interval = int(frame_interval)
    if frame_num <= 0 or (frame_num % 4 and (frame_num <= 1 or (frame_num - 1) % 4)):
        raise ValueError("frame_num must be a positive multiple of 4 or one plus a multiple of 4")
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
        action_horizon: int | None = None,
        output_format: str = "vrae",
        text_embedding_cache_dir: str | Path | None = None,
        context_len: int = 128,
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
        self.action_horizon = self.frame_num if action_horizon is None else int(action_horizon)
        if self.action_horizon <= 0 or self.action_horizon > self.frame_num:
            raise ValueError("action_horizon must be in [1, frame_num]")
        self.output_format = str(output_format)
        if self.output_format not in {"vrae", "fastwam"}:
            raise ValueError("output_format must be 'vrae' or 'fastwam'")
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        if self.context_len <= 0:
            raise ValueError("context_len must be positive")
        self._context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.normalize_action_state = self.output_format == "fastwam"
        self._action_min, self._action_max = self._feature_minmax("action")
        self._state_min, self._state_max = self._feature_minmax("observation.state")
        self.camera_keys = self._resolve_camera_keys(camera_keys)
        temporal_indices = build_temporal_indices(self.frame_num, self.frame_interval)
        action_indices = build_temporal_indices(self.action_horizon, self.frame_interval)
        self.delta_indices = {
            camera["key"]: temporal_indices for camera in self.camera_keys
        }
        for key in ("observation.state", "action"):
            if key in self.meta.features:
                self.delta_indices[key] = action_indices if key == "action" else temporal_indices
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

    def _load_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not configured")
        cached = self._context_cache.get(prompt)
        if cached is not None:
            return cached[0].clone(), cached[1].clone()
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_embedding_cache_dir / (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing FastWAM text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "context" not in payload or "mask" not in payload:
            raise ValueError(f"Invalid FastWAM text embedding cache: {path}")
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != self.context_len:
            raise ValueError(f"Invalid context shape in FastWAM text embedding cache: {path}")
        self._context_cache[prompt] = (context, mask)
        return context.clone(), mask.clone()

    def _feature_minmax(self, key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            not self.normalize_action_state
            or not self.meta.stats
            or key not in self.meta.stats
        ):
            return None, None
        values = self.meta.stats[key]
        return torch.as_tensor(values["min"], dtype=torch.float32), torch.as_tensor(
            values["max"], dtype=torch.float32
        )

    @staticmethod
    def _normalize_minmax(
        value: torch.Tensor,
        minimum: torch.Tensor | None,
        maximum: torch.Tensor | None,
    ) -> torch.Tensor:
        if minimum is None or maximum is None:
            return value
        scale = (maximum - minimum).clamp_min(1.0e-6).to(device=value.device)
        return (value - minimum.to(device=value.device)) / scale * 2.0 - 1.0

    @staticmethod
    def _denormalize_minmax(
        value: torch.Tensor,
        minimum: torch.Tensor | None,
        maximum: torch.Tensor | None,
    ) -> torch.Tensor:
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if minimum is None or maximum is None:
            return value
        return (value + 1.0) * 0.5 * (maximum - minimum) + minimum

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._denormalize_minmax(action, self._action_min, self._action_max)

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._denormalize_minmax(state, self._state_min, self._state_max)

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
        global_indices = current_index + global_indices
        image_is_pad = (global_indices < episode_start) | (global_indices >= episode_stop)
        global_indices = global_indices.clamp(
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
        action = item["action"].float()
        state = item["observation.state"].float()
        if self.normalize_action_state:
            action = self._normalize_minmax(action, self._action_min, self._action_max)
            state = self._normalize_minmax(state, self._state_min, self._state_max)
        action_is_pad = item.get("action_is_pad")
        if action_is_pad is None:
            action_is_pad = torch.zeros(action.shape[0], dtype=torch.bool)
        proprio_is_pad = item.get("observation.state_is_pad")
        if proprio_is_pad is None:
            proprio_is_pad = torch.zeros(state.shape[0], dtype=torch.bool)
        if self.output_format == "fastwam":
            # FastWAM consumes [C,T,H,W] in [-1, 1] and expects synchronized
            # multi-camera frames as one wide image.
            video = torch.cat(tuple(view.permute(1, 0, 2, 3) for view in views), dim=-1)
            video = video.mul(2.0).sub(1.0)
            state_for_action = state[: action.shape[0]]
        else:
            state_for_action = state
        result: dict[str, Any] = {
            "video": video,
            "label": class_id,
            "sample_id": f"episode-{episode:06d}",
            "frame_indices": global_indices,
            "task": task,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "state": state,
            "action": action,
            "stream_ids": self.stream_ids.clone(),
        }
        if self.output_format == "fastwam":
            result.update({
                "proprio": state_for_action,
                "image_is_pad": image_is_pad,
                "action_is_pad": action_is_pad,
                "proprio_is_pad": proprio_is_pad,
            })
            if self.text_embedding_cache_dir is not None:
                context, context_mask = self._load_text_context(result["prompt"])
                result["context"] = context
                result["context_mask"] = context_mask
        return result


__all__ = ["LeRobotVideoDataset"]
