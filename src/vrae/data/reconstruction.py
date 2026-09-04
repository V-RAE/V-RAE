"""Manifest-backed LeRobot dataset for reconstruction training."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing import Value
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import datasets
import pyarrow.dataset as pa_dataset
from torchvision.io import decode_image
from vrae.data.lerobot3.lerobot_dataset import LeRobotDatasetMetadata
from vrae.data.lerobot3.utils import get_hf_features_from_features
from vrae.data.lerobot3.video_utils import decode_video_frames, get_safe_default_codec


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
    if target == (height, width):
        return video
    if video.dtype == torch.uint8:
        video = video.float().div_(255.0)
    return F.interpolate(video, size=target, mode="bilinear", align_corners=False, antialias=True)


def _random_crop(video: torch.Tensor, image_size: int, generator: torch.Generator) -> torch.Tensor:
    _, _, height, width = video.shape
    top = int(torch.randint(height - image_size + 1, (), generator=generator).item())
    left = int(torch.randint(width - image_size + 1, (), generator=generator).item())
    return video[:, :, top : top + image_size, left : left + image_size]


class ReconstructionDataset(torch.utils.data.Dataset):
    """Sample strict, random, single-view clips from a LeRobot manifest.

    This reader loads only the metadata and raw columns needed for temporal
    sampling and reconstruction augmentations.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        repo_id: str = "libero",
        num_frames: int = 16,
        view_key: str | None = None,
        camera_keys: Sequence[dict[str, Any]] | None = None,
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
        self.repo_id = repo_id
        self.root = Path(root)
        self.meta = LeRobotDatasetMetadata(repo_id, root=self.root)
        self.tolerance_s = 1e-4
        self.video_backend = get_safe_default_codec()
        self.during_training = True
        self._manifest_episodes = load_libero_manifest(manifest)
        requested_cameras = camera_keys or []
        self.camera_keys = tuple(
            {
                "key": str(camera["key"]),
                "name": str(camera.get("name", camera["key"])),
                "stream_id": int(camera.get("stream_id", index)),
            }
            for index, camera in enumerate(requested_cameras)
        )
        if view_key is not None:
            requested_view = str(view_key)
            self._manifest_episodes = {
                episode: tuple(view for view in views if view.view_key == requested_view)
                for episode, views in self._manifest_episodes.items()
            }
            self._manifest_episodes = {
                episode: views for episode, views in self._manifest_episodes.items() if views
            }
        if self.camera_keys:
            requested_views = tuple(camera["key"] for camera in self.camera_keys)
            self._manifest_episodes = {
                episode: tuple(
                    next(view for view in views if view.view_key == view_key)
                    for view_key in requested_views
                )
                for episode, views in self._manifest_episodes.items()
                if all(any(view.view_key == view_key for view in views) for view_key in requested_views)
            }
        self.episodes = list(range(self.meta.total_episodes))
        manifest_camera_keys = [
            view.view_key
            for views in self._manifest_episodes.values()
            for view in views
        ]
        data_columns = list(dict.fromkeys(
            [camera["key"] for camera in self.camera_keys] + manifest_camera_keys
            + ["timestamp"]
        ))
        data_paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}.")
        # Reuse the normal LeRobot parquet cache, then keep only the columns
        # needed by reconstruction.  Changing the input column list would
        # force datasets to rebuild a second 33 GB-derived Arrow cache.
        parquet_dataset = datasets.Dataset.from_parquet(
            [str(path) for path in data_paths],
            features=get_hf_features_from_features(self.meta.features),
            columns=list(self.meta.features),
            filters=pa_dataset.field("episode_index").isin(self.episodes),
        )
        self.hf_dataset = parquet_dataset.select_columns(data_columns)
        episode_lengths = torch.tensor(
            [int(self.meta.episodes[index]["length"]) for index in self.episodes],
            dtype=torch.long,
        )
        episode_ends = torch.cumsum(episode_lengths, dim=0)
        self.episode_data_index = {
            "from": torch.cat((torch.zeros(1, dtype=torch.long), episode_ends[:-1])),
            "to": episode_ends,
        }
        self._episode_id_to_nested_id = {
            episode_id: nested_id for nested_id, episode_id in enumerate(self.episodes)
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
                if all(view.frame_count >= min_required for view in views)
            }
        else:
            required = 1 + (self.reconstruction_num_frames - 1) * self.frame_interval
            valid = {
                episode: views
                for episode, views in manifest_views.items()
                if all(view.frame_count >= required for view in views)
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
        self._profile_lock = threading.Lock()
        self._profile_totals = defaultdict(float)
        self._profile_sample_count = 0

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

    def _record_profile(self, name: str, seconds: float) -> None:
        with self._profile_lock:
            self._profile_totals[name] += float(seconds)

    def profile_snapshot(self, *, reset: bool = False) -> dict[str, float]:
        with self._profile_lock:
            sample_count = int(self._profile_sample_count)
            totals = dict(self._profile_totals)
            if reset:
                self._profile_sample_count = 0
                self._profile_totals.clear()
        result = {"profile/data_detail.sample_count": float(sample_count)}
        for name, seconds in totals.items():
            result[f"profile/data_detail.{name}_ms"] = (
                float(seconds) / max(sample_count, 1) * 1000.0
            )
        return result

    def _decode_image_frame(self, key: str, index: int) -> torch.Tensor:
        # Access Arrow storage directly so reading one timestamp does not run
        # the Hugging Face transform over every image column in the selected rows.
        value = self.hf_dataset.data[key][int(index)].as_py()
        if isinstance(value, dict) and value.get("bytes") is not None:
            encoded = torch.frombuffer(
                bytearray(value["bytes"]), dtype=torch.uint8
            )
            return decode_image(encoded, mode="RGB")
        raise ValueError(
            f"Expected inline image bytes for {key!r}; got {type(value).__name__}"
        )

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], episode_index: int
    ) -> dict[str, torch.Tensor]:
        episode = self.meta.episodes[int(episode_index)]
        result = {}
        for video_key, timestamps in query_timestamps.items():
            start_timestamp = episode[f"videos/{video_key}/from_timestamp"]
            shifted_timestamps = [start_timestamp + timestamp for timestamp in timestamps]
            video_path = self.root / self.meta.get_video_file_path(episode_index, video_key)
            result[video_key] = decode_video_frames(
                video_path,
                shifted_timestamps,
                self.tolerance_s,
                self.video_backend,
            ).squeeze(0)
        return result

    def _choose_view_and_interval(
        self, views: Sequence[LiberoManifestView], generator: torch.Generator
    ) -> tuple[tuple[LiberoManifestView, ...], int] | None:
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
            if valid and len(valid) == len(valid_views):
                return tuple(valid), interval
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_started_at = time.perf_counter()
        episode = self._episodes[int(index)]
        generator = self._generator(index)
        selected = self._choose_view_and_interval(self._manifest_episodes[episode], generator)
        if selected is None:
            raise IndexError(f"episode {episode} has no valid clip")
        selected_views, interval = selected
        view = selected_views[0]
        required = 1 + (self.reconstruction_num_frames - 1) * interval
        max_start = view.frame_count - required
        start = int(torch.randint(max_start + 1, (), generator=generator).item())
        offsets = torch.arange(self.reconstruction_num_frames, dtype=torch.long)
        local_indices = start + offsets * interval
        nested_episode_index = self._episode_id_to_nested_id[episode]
        episode_start = int(self.episode_data_index["from"][nested_episode_index])
        global_indices = episode_start + local_indices
        videos: list[torch.Tensor] = []
        crop_top = crop_left = None
        rows = None
        for selected_view in selected_views:
            decode_started_at = time.perf_counter()
            if selected_view.view_key in self.meta.video_keys:
                if rows is None:
                    rows = self.hf_dataset.select(global_indices.tolist())
                timestamps = [float(value) for value in rows["timestamp"]]
                video = self._query_videos(
                    {selected_view.view_key: timestamps}, episode
                )[selected_view.view_key]
            else:
                frames = [
                    self._decode_image_frame(selected_view.view_key, frame_index)
                    for frame_index in global_indices
                ]
                video = torch.stack(frames)
                if video.ndim != 4:
                    raise ValueError(f"Expected [T,H,W,C] or [T,C,H,W], got {tuple(video.shape)}")
                if video.shape[-1] == 3:
                    video = video.permute(0, 3, 1, 2)
                elif video.shape[1] != 3:
                    raise ValueError(f"Expected RGB frames, got {tuple(video.shape)}")
            self._record_profile("sample.decode", time.perf_counter() - decode_started_at)
            preprocess_started_at = time.perf_counter()
            video = _resize_short_side(video, self.image_size)
            if crop_top is None:
                _, _, height, width = video.shape
                crop_top = int(torch.randint(height - self.image_size + 1, (), generator=generator).item())
                crop_left = int(torch.randint(width - self.image_size + 1, (), generator=generator).item())
            video = video[:, :, crop_top : crop_top + self.image_size, crop_left : crop_left + self.image_size]
            videos.append(video)
            self._record_profile(
                "sample.resize_crop", time.perf_counter() - preprocess_started_at
            )
        assemble_started_at = time.perf_counter()
        video = torch.stack(videos, dim=1) if len(self.camera_keys) > 1 else videos[0]
        if self.random_flip and bool(torch.rand((), generator=generator) < 0.5):
            video = video.flip(-1)
        self._record_profile(
            "sample.assemble", time.perf_counter() - assemble_started_at
        )
        result = {
            "video": video.contiguous(),
            "label": self._resolve_label(view, episode),
            "sample_id": f"episode-{episode:06d}",
            "frame_indices": global_indices,
            "episode_index": episode,
            "view": view.view_key,
            "frame_interval": interval,
            **({"stream_ids": torch.tensor([camera["stream_id"] for camera in self.camera_keys], dtype=torch.long)} if self.camera_keys else {}),
        }
        with self._profile_lock:
            self._profile_sample_count += 1
            self._profile_totals["sample.total"] += time.perf_counter() - sample_started_at
        return result

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
