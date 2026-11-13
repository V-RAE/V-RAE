"""NanoVVM raw-video clips as one observed RGB frame plus sixteen future frames.

A manifest record with ``start_frame = s`` covers source frames ``[s, s+49)``.
The clip decodes exactly ``s + 3*k`` for ``k = 0..16`` (decord, as NanoVVM
counted and decodes them), applies one spatial transform to all 17 frames
(short side to 256 with bilinear, antialias off; 256x256 center crop; no flip),
and splits it:

    condition = clip[0]      # [3,256,256]    source frame s
    future    = clip[1:17]   # [16,3,256,256] source frames s+3 ... s+48

Frames stay uint8; the frozen V-RAE encoder applies its own pixel
normalization. Multi-source data is a clip-level concatenation, so an epoch
mixes sources in proportion to their clip counts.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vrae.data.transforms import center_crop_video, random_crop_video, resize_short_side
from vrae.paths import ProjectPaths
from vrae.training.common.contracts import resolve_batch_contract
from vrae.training.common.sampler import StatefulDistributedBatchSampler
from vrae.training.nanovvm_video_pred.manifest import (
    SOURCE_KEYS,
    ClipManifest,
    clip_span,
    frame_indices,
    load_clip_manifest,
    validate_sources,
)

DATASET_KEY = "nanovvm"
RGB_SEQ_LEN = 17
FRAME_INTERVAL = 3
CONDITION_FRAMES = 1
FUTURE_FRAMES = 16
SOURCE_FRAME_SPAN = clip_span(RGB_SEQ_LEN, FRAME_INTERVAL)
IMAGE_SIZE = 256
PREPROCESSING: dict[str, Any] = {
    "horizontal_flip_prob": 0.0,
    "resize": {"mode": "short_side", "interpolation": "bilinear", "antialias": False},
    "crop": {"mode": "center"},
}
VIDEO_DECODERS = ("decord",)
SPLITS = ("train", "val")
SINGLE_SOURCE_KEYS = ("root", "manifest", "manifest_meta")
DEFAULT_SOURCE_NAME = "default"


def frame_protocol_metadata() -> dict[str, Any]:
    """The temporal contract recorded with statistics and checkpoints."""

    offsets = [FRAME_INTERVAL * k for k in range(RGB_SEQ_LEN)]
    return {
        "rgb_seq_len": RGB_SEQ_LEN,
        "frame_interval": FRAME_INTERVAL,
        "source_frame_span": SOURCE_FRAME_SPAN,
        "condition_frames": CONDITION_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "condition_source_offsets": offsets[:CONDITION_FRAMES],
        "future_source_offsets": offsets[CONDITION_FRAMES:],
    }


def validate_frame_protocol(data: Mapping[str, Any]) -> None:
    expected = {
        "rgb_seq_len": RGB_SEQ_LEN,
        "frame_interval": FRAME_INTERVAL,
        "condition_frames": CONDITION_FRAMES,
        "future_frames": FUTURE_FRAMES,
    }
    for key, value in expected.items():
        actual = data.get(key, value)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != value:
            raise ValueError(
                f"data.{key} must be {value} for NanoVVM cond1gen16_interval3 prediction, "
                f"got {actual!r}"
            )


def validate_preprocessing(config: object) -> None:
    """NanoVVM's preprocessing schema (flip, resize, crop); image_size is the output size."""

    if not isinstance(config, Mapping):
        raise ValueError("data.preprocessing must be a mapping")
    unknown = set(config) - {"horizontal_flip_prob", "resize", "crop"}
    if unknown:
        raise ValueError(f"Unknown preprocessing keys: {sorted(unknown)}")
    prob = config.get("horizontal_flip_prob", 0.5)
    if (
        isinstance(prob, bool)
        or not isinstance(prob, (int, float))
        or not math.isfinite(prob)
        or not 0 <= prob <= 1
    ):
        raise ValueError("preprocessing.horizontal_flip_prob must be between zero and one")
    resize = config.get("resize", {})
    if not isinstance(resize, Mapping) or set(resize) - {"mode", "interpolation", "antialias"}:
        raise ValueError("preprocessing.resize accepts mode, interpolation and antialias")
    if resize.get("mode", "short_side") not in ("short_side", "none"):
        raise ValueError("preprocessing.resize.mode must be short_side or none")
    if resize.get("interpolation", "bilinear") not in ("bilinear", "bicubic"):
        raise ValueError("preprocessing.resize.interpolation must be bilinear or bicubic")
    if not isinstance(resize.get("antialias", False), bool):
        raise ValueError("preprocessing.resize.antialias must be boolean")
    crop = config.get("crop", {})
    if (
        not isinstance(crop, Mapping)
        or set(crop) - {"mode"}
        or crop.get("mode", "random") not in ("random", "center")
    ):
        raise ValueError("preprocessing.crop.mode must be random or center")


def spatial_preprocess(
    video: torch.Tensor,
    image_size: int,
    *,
    preprocessing: Mapping[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one flip/resize/crop to every frame of ``[T,3,H,W]`` (NanoVVM order)."""

    config = {} if preprocessing is None else dict(preprocessing)
    validate_preprocessing(config)
    prob = config.get("horizontal_flip_prob", 0.5)
    if prob > 0 and bool(torch.rand((), generator=generator) < prob):
        video = video.flip(-1)
    resize = config.get("resize", {})
    if resize.get("mode", "short_side") == "short_side":
        video = resize_short_side(
            video,
            int(image_size),
            mode=resize.get("interpolation", "bilinear"),
            antialias=resize.get("antialias", False),
        )
    if config.get("crop", {}).get("mode", "random") == "random":
        return random_crop_video(video, int(image_size), generator=generator)
    return center_crop_video(video, int(image_size))


def split_condition_future(clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[...,17,3,H,W]`` -> (frame 0 ``[...,3,H,W]``, frames 1..16 ``[...,16,3,H,W]``)."""

    if clip.ndim < 4 or clip.shape[-4] != RGB_SEQ_LEN or clip.shape[-3] != 3:
        raise ValueError(f"clip must be [...,{RGB_SEQ_LEN},3,H,W], got {tuple(clip.shape)}")
    condition = clip.select(-4, 0).contiguous()
    future = clip.narrow(-4, CONDITION_FRAMES, FUTURE_FRAMES).contiguous()
    return condition, future


def decode_clip_frames(
    path: Path,
    start_frame: int,
    *,
    where: str,
    rgb_seq_len: int = RGB_SEQ_LEN,
    frame_interval: int = FRAME_INTERVAL,
) -> tuple[torch.Tensor, np.ndarray]:
    """Decode exact source frames ``start + interval*k`` as uint8 ``[T,3,H,W]`` with decord."""

    from decord import VideoReader, cpu

    if not path.is_file():
        raise FileNotFoundError(
            f"Source video missing for {where}. Check the data root and that the dataset kept "
            "its directory structure."
        )
    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    except Exception as error:  # decord raises its own error types
        raise RuntimeError(f"Could not open the video for {where}: {error}") from error
    try:
        indices = frame_indices(len(reader), rgb_seq_len, frame_interval, int(start_frame))
    except ValueError:
        raise ValueError(
            f"The video has {len(reader)} frames, too few for {where}; the manifest does not "
            "match this file."
        ) from None
    try:
        frames = reader.get_batch(indices).asnumpy()
    except Exception as error:
        raise RuntimeError(
            f"Could not decode frames {indices.tolist()} for {where}: {error}"
        ) from error
    if frames.ndim != 4 or frames.shape[0] != rgb_seq_len or frames.shape[-1] != 3:
        raise RuntimeError(f"decord returned {frames.shape} for {where}; expected [T,H,W,3]")
    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).contiguous()
    return video, indices


class ManifestClipSource:
    """One NanoVVM manifest with its own video root."""

    def __init__(
        self,
        *,
        name: str,
        dataset: str,
        root: Path,
        manifest: Path,
        manifest_meta: Path,
        clips: ClipManifest,
        configured: Mapping[str, str] | None = None,
    ) -> None:
        self.name = str(name)
        self.dataset = str(dataset)
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.manifest_meta = Path(manifest_meta)
        self.clips = clips
        self.configured = dict(configured or {})

    def __len__(self) -> int:
        return len(self.clips)

    def identity(self) -> dict[str, Any]:
        meta = self.clips.meta
        return {
            "name": self.name,
            "dataset": self.dataset,
            "manifest": self.configured.get("manifest", str(self.manifest)),
            "manifest_sha256": str(meta.get("manifest", {}).get("sha256")),
            "manifest_split": meta.get("split"),
            "total_clips": len(self.clips),
        }


def _reject_interpolation(value: str, where: str) -> None:
    if "${" in value:
        raise ValueError(
            f"{where}={value!r} contains an unresolved ${{...}} interpolation. V-RAE reads YAML "
            "with PyYAML only; write paths relative to the NanoVVM data root "
            "(paths.datasets.nanovvm or data.data_root) instead."
        )


def resolve_data_root(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    """``data.data_root`` when set, otherwise the ``nanovvm`` dataset path of ``--paths``."""

    configured = config["data"].get("data_root")
    if configured:
        _reject_interpolation(str(configured), "data.data_root")
        root = Path(str(configured)).expanduser()
        if not root.is_absolute():
            root = paths.project_root / root
    else:
        root = paths.dataset(DATASET_KEY, require_exists=False)
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"NanoVVM data root is not a directory: {root}. Set datasets.{DATASET_KEY} in the "
            "--paths file or data.data_root in the config."
        )
    return root


def split_source_entries(config: Mapping[str, Any], split: str) -> list[dict[str, str]]:
    """The configured sources of one split, as ``{name, dataset, root, manifest, manifest_meta}``.

    ``data.<split>`` is either ``{sources: [...]}`` (NanoVVM mix configs) or a
    single ``{root, manifest, manifest_meta}`` with optional ``name``/``dataset``.
    """

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    value = config["data"].get(split)
    where = f"data.{split}"
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping with sources or root/manifest/manifest_meta")
    if "sources" in value:
        if set(value) != {"sources"}:
            raise ValueError(
                f"{where}.sources cannot be combined with {sorted(set(value) - {'sources'})}"
            )
        sources = [dict(item) if isinstance(item, Mapping) else item for item in value["sources"]]
        validate_sources(sources, f"{where}.sources")
        entries = [{key: str(source[key]) for key in SOURCE_KEYS} for source in sources]
    else:
        unknown = set(value) - set(SINGLE_SOURCE_KEYS) - {"name", "dataset"}
        missing = [key for key in SINGLE_SOURCE_KEYS if not value.get(key)]
        if unknown or missing:
            raise ValueError(
                f"{where} needs root, manifest and manifest_meta (optional name, dataset); "
                f"missing={missing} unknown={sorted(unknown)}"
            )
        entry = {
            "name": str(value.get("name", DEFAULT_SOURCE_NAME)),
            "dataset": str(value.get("dataset", DEFAULT_SOURCE_NAME)),
            **{key: str(value[key]) for key in SINGLE_SOURCE_KEYS},
        }
        validate_sources([entry], where)
        entries = [entry]
    for index, entry in enumerate(entries):
        for key in ("root", "manifest", "manifest_meta"):
            _reject_interpolation(entry[key], f"{where}[{index}].{key}")
    return entries


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_split_sources(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    split: str,
) -> list[dict[str, Any]]:
    """Resolve every source path of a split against the data root and check that it exists."""

    base = resolve_data_root(config, paths)
    resolved: list[dict[str, Any]] = []
    problems: list[str] = []
    for entry in split_source_entries(config, split):
        item = {
            "name": entry["name"],
            "dataset": entry["dataset"],
            "configured": entry,
            "root": _resolve(entry["root"], base),
            "manifest": _resolve(entry["manifest"], base),
            "manifest_meta": _resolve(entry["manifest_meta"], base),
        }
        if not item["root"].is_dir():
            problems.append(f"{entry['name']}: video root is not a directory: {item['root']}")
        for key in ("manifest", "manifest_meta"):
            if not item[key].is_file():
                problems.append(f"{entry['name']}: {key} is not a file: {item[key]}")
        resolved.append(item)
    if problems:
        raise FileNotFoundError(
            f"data.{split} has {len(problems)} missing path(s) under data root {base}:\n  "
            + "\n  ".join(problems)
        )
    return resolved


def load_split_sources(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    split: str,
) -> list[ManifestClipSource]:
    sources = []
    for item in resolve_split_sources(config, paths, split):
        clips = load_clip_manifest(
            item["manifest"],
            item["manifest_meta"],
            rgb_seq_len=RGB_SEQ_LEN,
            frame_interval=FRAME_INTERVAL,
        )
        sources.append(
            ManifestClipSource(
                name=item["name"],
                dataset=item["dataset"],
                root=item["root"],
                manifest=item["manifest"],
                manifest_meta=item["manifest_meta"],
                clips=clips,
                configured=item["configured"],
            )
        )
    return sources


class NanoVVMVideoPredictionDataset(Dataset[dict[str, Any]]):
    """Clip-level concatenation of manifest sources; one item per manifest record."""

    def __init__(
        self,
        sources: Sequence[ManifestClipSource],
        *,
        split: str,
        image_size: int = IMAGE_SIZE,
        preprocessing: Mapping[str, Any] | None = None,
        video_decoder: str = "decord",
        check_first_video: bool = True,
    ) -> None:
        if not sources:
            raise ValueError("at least one manifest source is required")
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"repeated source names: {names}")
        geometry = {(s.clips.rgb_seq_len, s.clips.frame_interval, s.clips.span) for s in sources}
        if geometry != {(RGB_SEQ_LEN, FRAME_INTERVAL, SOURCE_FRAME_SPAN)}:
            raise ValueError(f"sources disagree with the 17-frame interval-3 protocol: {geometry}")
        if str(video_decoder) not in VIDEO_DECODERS:
            raise ValueError(f"data.video_decoder must be one of {VIDEO_DECODERS}")
        self.sources = list(sources)
        self.split = str(split)
        self.image_size = int(image_size)
        self.preprocessing = dict(PREPROCESSING if preprocessing is None else preprocessing)
        validate_preprocessing(self.preprocessing)
        self.video_decoder = str(video_decoder)
        self.cumulative_sizes: list[int] = []
        total = 0
        for source in self.sources:
            if len(source) == 0:
                raise ValueError(f"source {source.name} has no clips")
            total += len(source)
            self.cumulative_sizes.append(total)
            if check_first_video:
                first = source.root / source.clips.video_paths[0]
                if not first.is_file():
                    raise FileNotFoundError(
                        f"Manifest video {source.clips.video_paths[0]!r} of source {source.name} "
                        f"is not under {source.root}; check the data root."
                    )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def set_epoch(self, epoch: int) -> None:
        del epoch  # clip starts and spatial transforms are deterministic

    def locate(self, index: int) -> tuple[ManifestClipSource, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self.cumulative_sizes, index)
        offset = self.cumulative_sizes[source_index - 1] if source_index else 0
        return self.sources[source_index], index - offset

    def clip_record(self, index: int) -> dict[str, Any]:
        source, local = self.locate(index)
        record = source.clips.record(local)
        # The source name keeps identical file names and frame numbers of
        # different cameras/sources distinct.
        sample_id = f"{source.name}:{record['video_path']}:{record['start_frame']}"
        return {
            **record,
            "sample_id": sample_id,
            "source": source.name,
            "dataset": source.dataset,
            "split": self.split,
            "local_index": local,
        }

    def load_clip(self, index: int) -> tuple[torch.Tensor, dict[str, Any], np.ndarray]:
        source, _ = self.locate(index)
        record = self.clip_record(index)
        path = source.root / record["video_path"]
        where = (
            f"{source.name} clip {record['local_index']} "
            f"[{record['start_frame']}, {record['end_frame_exclusive']}) of {path}"
        )
        frames, indices = decode_clip_frames(path, record["start_frame"], where=where)
        clip = spatial_preprocess(frames, self.image_size, preprocessing=self.preprocessing)
        expected = (RGB_SEQ_LEN, 3, self.image_size, self.image_size)
        if tuple(clip.shape) != expected or clip.dtype != torch.uint8:
            raise RuntimeError(
                f"{where}: preprocessed clip {tuple(clip.shape)}, expected {expected}"
            )
        return clip, record, indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip, record, indices = self.load_clip(index)
        condition, future = split_condition_future(clip)
        return {
            "condition": condition,
            "future": future,
            "frame_indices": torch.from_numpy(indices.copy()),
            "sample_id": record["sample_id"],
            "index": int(index),
            "metadata": record,
        }

    def identity(self) -> list[dict[str, Any]]:
        return [source.identity() for source in self.sources]

    def summary(self) -> list[str]:
        total = len(self)
        return [
            f"{s.name} ({s.dataset}): {len(s)} clips ({100 * len(s) / total:.2f}%), "
            f"root {s.root}, manifest {s.manifest}"
            for s in self.sources
        ]


def build_nanovvm_dataset(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    split: str,
) -> NanoVVMVideoPredictionDataset:
    data = config["data"]
    validate_frame_protocol(data)
    return NanoVVMVideoPredictionDataset(
        load_split_sources(config, paths, split),
        split=split,
        image_size=int(data.get("image_size", IMAGE_SIZE)),
        preprocessing=data.get("preprocessing", PREPROCESSING),
        video_decoder=str(data.get("video_decoder", "decord")),
    )


def collate_nanovvm(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty NanoVVM batch")
    return {
        "condition": torch.stack([item["condition"] for item in items]),
        "future": torch.stack([item["future"] for item in items]),
        "frame_indices": torch.stack([item["frame_indices"] for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
        "metadata": [item["metadata"] for item in items],
    }


def build_nanovvm_loader(
    dataset: Dataset[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader[dict[str, Any]], StatefulDistributedBatchSampler]:
    batch = resolve_batch_contract(config["training"], world_size=int(world_size))
    sampler = StatefulDistributedBatchSampler(
        len(dataset),
        batch.local_micro_batch_size,
        rank=int(rank),
        world_size=int(world_size),
        seed=int(config["data"].get("seed", 3407)),
        shuffle=shuffle,
        drop_last=drop_last,
        gradient_accumulation_steps=batch.gradient_accumulation_steps,
    )
    workers = int(config["training"].get("num_workers", 4))
    extra: dict[str, Any] = {}
    if workers > 0:
        extra["prefetch_factor"] = int(config["training"].get("prefetch_factor", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_nanovvm,
        persistent_workers=False,
        generator=sampler.loader_generator,
        **extra,
    )
    return loader, sampler


__all__ = [
    "CONDITION_FRAMES",
    "DATASET_KEY",
    "FRAME_INTERVAL",
    "FUTURE_FRAMES",
    "IMAGE_SIZE",
    "ManifestClipSource",
    "NanoVVMVideoPredictionDataset",
    "PREPROCESSING",
    "RGB_SEQ_LEN",
    "SOURCE_FRAME_SPAN",
    "build_nanovvm_dataset",
    "build_nanovvm_loader",
    "collate_nanovvm",
    "decode_clip_frames",
    "frame_protocol_metadata",
    "load_split_sources",
    "resolve_data_root",
    "resolve_split_sources",
    "spatial_preprocess",
    "split_condition_future",
    "split_source_entries",
    "validate_frame_protocol",
    "validate_preprocessing",
]
