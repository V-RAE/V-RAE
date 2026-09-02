"""Manifest-backed, single-view LeRobot dataset for reconstruction training."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing import Value
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from vrae.data.lerobot3.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class LiberoManifestView:
    episode_index: int
    view_key: str
    frame_count: int | None
    task_index: int | None
    path: str


def load_libero_manifest(path: str | Path) -> dict[int, tuple[LiberoManifestView, ...]]:
    """Load one or more camera rows per episode from a CSV manifest."""

    records: dict[int, list[LiberoManifestView]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"episode_index"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Libero manifest is missing columns: {sorted(missing)}")
        fields = set(reader.fieldnames or ())
        if "view" in fields:
            view_field = "view"
        elif "camera_key" in fields:
            view_field = "camera_key"
        else:
            view_field = None
        if view_field is None:
            raise ValueError("Libero manifest must contain view or camera_key")
        for line_number, row in enumerate(reader, start=2):
            try:
                task_value = row.get("task_index") or row.get("task")
                frame_value = row.get("frame_count") or row.get("length")
                record = LiberoManifestView(
                    episode_index=int(row["episode_index"]),
                    view_key=str(row[view_field]),
                    frame_count=int(frame_value) if frame_value not in (None, "") else None,
                    task_index=int(task_value) if task_value not in (None, "") else None,
                    path=str(row.get("path") or row.get("data_path") or ""),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid Libero manifest row {line_number}") from error
            if record.frame_count is not None and record.frame_count <= 0:
                raise ValueError(f"Libero manifest row {line_number} has invalid frame_count")
            records[record.episode_index].append(record)
    if not records:
        raise ValueError(f"Libero manifest is empty: {path}")
    return {episode: tuple(views) for episode, views in records.items()}


def _resize_short_side(video: torch.Tensor, image_size: int) -> torch.Tensor:
    _, _, height, width = video.shape
    scale = max(image_size / height, image_size / width)
    target = (max(image_size, round(height * scale)), max(image_size, round(width * scale)))
    return F.interpolate(video, size=target, mode="bilinear", align_corners=False, antialias=True)


def _random_crop(video: torch.Tensor, image_size: int, generator: torch.Generator) -> torch.Tensor:
    _, _, height, width = video.shape
    top = int(torch.randint(height - image_size + 1, (), generator=generator).item())
    left = int(torch.randint(width - image_size + 1, (), generator=generator).item())
    return video[:, :, top : top + image_size, left : left + image_size]


class ReconstructionDataset(LeRobotDataset):
    """Sample strict, random, single-view clips from a LeRobot manifest.

    The LeRobot reader owns metadata/parquet loading.  This class only adds
    manifest filtering, temporal sampling, and reconstruction augmentations.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        repo_id: str = "libero",
        num_frames: int = 16,
        view_key: str | None = None,
        frame_interval: int = 3,
        random_frame_interval: bool = False,
        frame_interval_min: int = 1,
        frame_interval_max: int = 4,
        image_size: int = 256,
        random_flip: bool = True,
        seed: int = 3407,
    ) -> None:
        if num_frames <= 0 or num_frames % 4:
            raise ValueError("num_frames must be a positive multiple of four")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if frame_interval_min <= 0 or frame_interval_max < frame_interval_min:
            raise ValueError("frame interval bounds are invalid")
        super().__init__(repo_id, root=Path(root))
        self._manifest_episodes = load_libero_manifest(manifest)
        if view_key is not None:
            requested_view = str(view_key)
            self._manifest_episodes = {
                episode: tuple(view for view in views if view.view_key == requested_view)
                for episode, views in self._manifest_episodes.items()
            }
            self._manifest_episodes = {
                episode: views for episode, views in self._manifest_episodes.items() if views
            }
        self.reconstruction_num_frames = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.random_frame_interval = bool(random_frame_interval)
        self.frame_interval_min = int(frame_interval_min)
        self.frame_interval_max = int(frame_interval_max)
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.seed = int(seed)
        self._epoch = Value("q", 0)
        known_episodes = {int(row["episode_index"]): row for row in self.meta.episodes}
        unknown_manifest_episodes = sorted(
            set(self._manifest_episodes).difference(known_episodes)
        )
        if unknown_manifest_episodes:
            raise ValueError(
                "Manifest references unknown LeRobot episodes: "
                f"{unknown_manifest_episodes[:5]}"
            )
        manifest_views = {
            episode: tuple(
                replace(
                    view,
                    frame_count=view.frame_count or int(known_episodes[episode]["length"]),
                )
                for view in views
            )
            for episode, views in self._manifest_episodes.items()
        }
        self._manifest_episodes = manifest_views
        min_required = 1 + (self.reconstruction_num_frames - 1) * self.frame_interval_min
        unsupported_views = sorted(
            {
                view.view_key
                for views in manifest_views.values()
                for view in views
                if view.view_key not in self.meta.camera_keys
            }
        )
        if unsupported_views:
            raise ValueError(
                "Manifest camera keys are not present in LeRobot metadata: "
                f"{unsupported_views}"
            )
        if self.random_frame_interval:
            valid = {
                episode: views
                for episode, views in manifest_views.items()
                if any(view.frame_count >= min_required for view in views)
            }
        else:
            required = 1 + (self.reconstruction_num_frames - 1) * self.frame_interval
            valid = {
                episode: views
                for episode, views in manifest_views.items()
                if any(view.frame_count >= required for view in views)
            }
        self._episode_rows = known_episodes
        self._episodes = tuple(sorted(valid))
        if not self._episodes:
            raise ValueError("Libero manifest contains no episode with a valid clip")
        task_rows = sorted(
            (int(row.task_index), str(task))
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

    def __len__(self) -> int:
        return len(self._episodes)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def save_class_map(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
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
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @staticmethod
    def load_class_map(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _generator(self, index: int) -> torch.Generator:
        epoch = int(self._epoch.value)
        return torch.Generator(device="cpu").manual_seed(
            (self.seed + epoch * 1_000_003 + int(index) * 10_007) % (2**63 - 1)
        )

    def _choose_view_and_interval(
        self, views: Sequence[LiberoManifestView], generator: torch.Generator
    ) -> tuple[LiberoManifestView, int] | None:
        valid_views = list(views)
        if self.random_frame_interval:
            candidate = int(
                torch.randint(
                    self.frame_interval_min,
                    self.frame_interval_max + 1,
                    (),
                    generator=generator,
                ).item()
            )
            intervals = range(candidate, self.frame_interval_min - 1, -1)
        else:
            intervals = (self.frame_interval,)
        for interval in intervals:
            required = 1 + (self.reconstruction_num_frames - 1) * interval
            valid = [view for view in valid_views if view.frame_count >= required]
            if valid:
                view = valid[int(torch.randint(len(valid), (), generator=generator).item())]
                return view, interval
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self._episodes[int(index)]
        generator = self._generator(index)
        selected = self._choose_view_and_interval(self._manifest_episodes[episode], generator)
        if selected is None:
            raise IndexError(f"episode {episode} has no valid clip")
        view, interval = selected
        required = 1 + (self.reconstruction_num_frames - 1) * interval
        max_start = view.frame_count - required
        start = int(torch.randint(max_start + 1, (), generator=generator).item())
        offsets = torch.arange(self.reconstruction_num_frames, dtype=torch.long)
        local_indices = start + offsets * interval
        nested_episode_index = self._episode_id_to_nested_id[episode]
        episode_start = int(self.episode_data_index["from"][nested_episode_index])
        global_indices = episode_start + local_indices
        rows = self.hf_dataset.select(global_indices.tolist())
        if view.view_key in self.meta.video_keys:
            timestamps = [float(value) for value in rows["timestamp"]]
            video = self._query_videos({view.view_key: timestamps}, episode)[view.view_key]
        else:
            frames = [torch.as_tensor(value) for value in rows[view.view_key]]
            video = torch.stack(frames)
            if video.ndim != 4:
                raise ValueError(f"Expected [T,H,W,C] or [T,C,H,W], got {tuple(video.shape)}")
            if video.shape[-1] == 3:
                video = video.permute(0, 3, 1, 2)
            elif video.shape[1] != 3:
                raise ValueError(f"Expected RGB frames, got {tuple(video.shape)}")
            video = video.float()
            if float(video.max()) > 1.0:
                video = video / 255.0
        video = _resize_short_side(video, self.image_size)
        video = _random_crop(video, self.image_size, generator)
        if self.random_flip and bool(torch.rand((), generator=generator) < 0.5):
            video = video.flip(-1)
        return {
            "video": video.contiguous(),
            "label": self._resolve_label(view, episode),
            "sample_id": f"episode-{episode:06d}",
            "frame_indices": global_indices,
            "episode_index": episode,
            "view": view.view_key,
            "frame_interval": interval,
        }

    def _resolve_label(self, view: LiberoManifestView, episode: int) -> int:
        if view.task_index is not None:
            task_index = int(view.task_index)
        else:
            episode_tasks = self._episode_rows[episode].get("tasks")
            if not episode_tasks:
                raise ValueError(f"Manifest row for episode {episode} has no task_index")
            task_ref = episode_tasks[0]
            try:
                task_row = self.meta.tasks.loc[task_ref]
            except (KeyError, TypeError):
                task_row = self.meta.tasks.iloc[int(task_ref)]
            task_index = int(task_row["task_index"])
        try:
            return self.task_index_to_class_id[task_index]
        except KeyError as error:
            raise ValueError(f"Unknown task_index {task_index} for episode {episode}") from error


__all__ = ["LiberoManifestView", "ReconstructionDataset", "load_libero_manifest"]
