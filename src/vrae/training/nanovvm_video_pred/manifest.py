"""NanoVVM offline clip manifests, read without installing NanoVVM.

Ported from nanovvm@37c2377 ``data/manifest.py`` and ``data/frames.py`` so that
the schema, sampling-parameter, frame-range, record-count, and SHA-256 checks
match the trainer that wrote the manifests (see nanovvm ``docs/clip_manifests.md``).

``manifest.json`` is a JSON array of records
``{"video_path", "start_frame", "end_frame_exclusive", "label"}`` whose
``video_path`` is relative to the source's video root; ``meta.yaml`` records the
schema version, sampling parameters, clip count, and the manifest's SHA-256.
Clips start on the ``span`` grid of their video and are chronological and
non-overlapping, but need not be contiguous (holdout subsets skip clips).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
SEGMENTATION = "sequential_non_overlapping"
INCOMPLETE_CLIP_POLICY = "discard"
RECORD_KEYS = ("video_path", "start_frame", "end_frame_exclusive", "label")
SOURCE_KEYS = ("name", "dataset", "root", "manifest", "manifest_meta")
BUILD_HINT = (
    "Build it with NanoVVM's data_processing/build_clip_manifest.py (and "
    "split_clip_manifest.py for holdouts); see nanovvm docs/clip_manifests.md."
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def clip_span(num_frames: int, frame_interval: int = 1) -> int:
    for name, value in (("num_frames", num_frames), ("frame_interval", frame_interval)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return 1 + (int(num_frames) - 1) * int(frame_interval)


def frame_indices(
    length: int,
    num_frames: int,
    frame_interval: int = 1,
    start: int = 0,
) -> np.ndarray:
    """Source-frame indices ``start + frame_interval * [0..num_frames-1]``."""

    span = clip_span(num_frames, frame_interval)
    if start < 0 or start + span > length:
        raise ValueError(
            f"{length} frames cannot supply {num_frames} frames at "
            f"interval={frame_interval}, start={start}"
        )
    return int(start) + np.arange(num_frames, dtype=np.int64) * int(frame_interval)


@dataclass(frozen=True, eq=False)
class ClipManifest:
    """Clip ``i`` decodes ``video_paths[video_index[i]]`` from ``start_frames[i]``.

    Clips are held as arrays rather than one dict per clip so that DataLoader
    workers do not touch (and copy) millions of Python objects.
    """

    video_paths: tuple[str, ...]
    video_index: np.ndarray
    start_frames: np.ndarray
    labels: np.ndarray
    rgb_seq_len: int
    frame_interval: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not len(self.video_index) == len(self.start_frames) == len(self.labels):
            raise ValueError("ClipManifest arrays must have one entry per clip")

    @property
    def span(self) -> int:
        return clip_span(self.rgb_seq_len, self.frame_interval)

    def __len__(self) -> int:
        return len(self.start_frames)

    def record(self, index: int) -> dict[str, Any]:
        start = int(self.start_frames[index])
        return {
            "video_path": self.video_paths[int(self.video_index[index])],
            "start_frame": start,
            "end_frame_exclusive": start + self.span,
            "label": int(self.labels[index]),
        }


def check_relative_path(path: object, where: str) -> None:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(
            f"{where}: invalid video_path {path!r}; manifest paths must be relative to the "
            "video root, without a leading '/', empty, '.' or '..' components"
        )


def _get(meta: dict[str, Any], dotted: str, source: object) -> Any:
    value: Any = meta
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Clip manifest metadata {source} has no {dotted}; {BUILD_HINT}")
        value = value[key]
    return value


def read_meta(meta_path: str | Path) -> dict[str, Any]:
    meta_path = Path(meta_path).expanduser()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Clip manifest metadata not found: {meta_path}. {BUILD_HINT}")
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise ValueError(f"Malformed clip manifest metadata {meta_path}: {error}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"Malformed clip manifest metadata {meta_path}: expected a YAML mapping")
    return meta


def validate_meta(
    meta: dict[str, Any],
    *,
    rgb_seq_len: int,
    frame_interval: int,
    source: object = "meta.yaml",
) -> None:
    """Check the metadata's own consistency, then its compatibility with the clip length."""

    version = meta.get("schema_version")
    if not _is_int(version) or version != SCHEMA_VERSION:
        raise ValueError(
            f"Clip manifest metadata {source} has unsupported schema_version {version!r}; "
            f"this code reads version {SCHEMA_VERSION}. {BUILD_HINT}"
        )
    sampling = {
        key: _get(meta, f"sampling.{key}", source)
        for key in ("rgb_seq_len", "frame_interval", "source_frame_span", "clip_stride")
    }
    for key, value in sampling.items():
        if not _is_int(value) or value < 1:
            raise ValueError(
                f"Clip manifest metadata {source}: sampling.{key} must be a positive integer, "
                f"got {value!r}"
            )
    span = clip_span(sampling["rgb_seq_len"], sampling["frame_interval"])
    expected = {
        "sampling.source_frame_span": span,
        "sampling.clip_stride": span,
        "sampling.segmentation": SEGMENTATION,
        "sampling.incomplete_clip_policy": INCOMPLETE_CLIP_POLICY,
        "manifest.path_type": "relative",
        "manifest.start_frame_inclusive": True,
        "manifest.end_frame_exclusive": True,
    }
    for key, value in expected.items():
        actual = _get(meta, key, source)
        if actual != value or type(actual) is not type(value):
            raise ValueError(
                f"Clip manifest metadata {source}: {key} is {actual!r}; this code supports "
                f"only {value!r}. {BUILD_HINT}"
            )
    for key in ("manifest.filename", "manifest.sha256"):
        if not isinstance(_get(meta, key, source), str):
            raise ValueError(f"Clip manifest metadata {source}: {key} must be a string")
    total = _get(meta, "statistics.total_clips", source)
    if not _is_int(total) or total < 0:
        raise ValueError(
            f"Clip manifest metadata {source}: statistics.total_clips must be a "
            "non-negative integer"
        )
    mismatched = [
        f"{key}={sampling[key]} (this task uses {value})"
        for key, value in (("rgb_seq_len", rgb_seq_len), ("frame_interval", frame_interval))
        if sampling[key] != value
    ]
    if mismatched:
        raise ValueError(
            f"Clip manifest {source} was built for "
            + ", ".join(mismatched)
            + f"; this task needs clips of {rgb_seq_len} RGB frames at "
            f"frame_interval={frame_interval} ({clip_span(rgb_seq_len, frame_interval)} "
            f"source frames). {BUILD_HINT}"
        )


def parse_records(
    records: object,
    *,
    rgb_seq_len: int,
    frame_interval: int,
    source: object = "manifest.json",
    meta: dict[str, Any] | None = None,
) -> ClipManifest:
    """Validate decoded manifest JSON (structure, paths, frame ranges) and pack it."""

    span = clip_span(rgb_seq_len, frame_interval)
    if not isinstance(records, list):
        raise ValueError(
            f"Malformed clip manifest {source}: the top level must be a JSON array of clip records"
        )
    if not records:
        raise ValueError(f"Clip manifest {source} is empty. {BUILD_HINT}")
    keys = set(RECORD_KEYS)
    video_index = np.empty(len(records), dtype=np.int32)
    start_frames = np.empty(len(records), dtype=np.int64)
    labels = np.empty(len(records), dtype=np.int64)
    paths: list[str] = []
    lookup: dict[str, int] = {}
    last_start: list[int] = []
    for i, record in enumerate(records):
        where = f"{source}, record {i}"
        if not isinstance(record, dict) or record.keys() != keys:
            raise ValueError(
                f"{where}: expected an object with exactly the keys {list(RECORD_KEYS)}, "
                f"got {record!r}"
            )
        path, start, end, label = (record[key] for key in RECORD_KEYS)
        if not (_is_int(start) and _is_int(end) and _is_int(label)) or label < 0:
            raise ValueError(
                f"{where}: start_frame, end_frame_exclusive and label must be integers "
                f"(label non-negative), got {record!r}"
            )
        if start < 0 or end - start != span or start % span:
            raise ValueError(
                f"{where} ({path!r}): frame range [{start}, {end}) is invalid; clips of "
                f"rgb_seq_len={rgb_seq_len} at frame_interval={frame_interval} span {span} "
                f"source frames and start at a non-negative multiple of {span}"
            )
        index = lookup.get(path)
        if index is None:
            check_relative_path(path, where)
            index = lookup[path] = len(paths)
            paths.append(path)
            last_start.append(-1)
        elif start <= last_start[index]:
            raise ValueError(
                f"{where} ({path!r}): clip starting at {start} repeats or precedes an earlier "
                "clip of the same video; clips must be chronological and non-overlapping"
            )
        last_start[index] = start
        video_index[i], start_frames[i], labels[i] = index, start, label
    return ClipManifest(
        tuple(paths),
        video_index,
        start_frames,
        labels,
        int(rgb_seq_len),
        int(frame_interval),
        dict(meta or {}),
    )


def load_clip_manifest(
    manifest_path: str | Path,
    meta_path: str | Path,
    *,
    rgb_seq_len: int,
    frame_interval: int,
) -> ClipManifest:
    """Read, validate, and pack one manifest pair; no video is opened."""

    manifest_path = Path(manifest_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clip manifest not found: {manifest_path}. {BUILD_HINT}")
    meta_path = Path(meta_path).expanduser()
    meta = read_meta(meta_path)
    validate_meta(meta, rgb_seq_len=rgb_seq_len, frame_interval=frame_interval, source=meta_path)
    if meta["manifest"]["filename"] != manifest_path.name:
        raise ValueError(
            f"Clip manifest metadata {meta_path} describes {meta['manifest']['filename']!r}, "
            f"not {manifest_path.name!r}"
        )
    data = manifest_path.read_bytes()
    try:
        records = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Malformed clip manifest {manifest_path}: {error}") from None
    clips = parse_records(
        records,
        rgb_seq_len=rgb_seq_len,
        frame_interval=frame_interval,
        source=manifest_path,
        meta=meta,
    )
    if (
        len(clips) != meta["statistics"]["total_clips"]
        or hashlib.sha256(data).hexdigest() != meta["manifest"]["sha256"]
    ):
        raise ValueError(
            f"Clip manifest {manifest_path} ({len(clips)} clips) does not match {meta_path} "
            f"(total_clips={meta['statistics']['total_clips']}, sha256 differs); the two files "
            f"were not generated together. {BUILD_HINT}"
        )
    return clips


def validate_sources(sources: object, where: str) -> None:
    """A non-empty list of ``{name, dataset, root, manifest, manifest_meta}`` with unique names."""

    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"{where} must be a non-empty list of {{{', '.join(SOURCE_KEYS)}}} mappings"
        )
    names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != set(SOURCE_KEYS):
            raise ValueError(
                f"{where}[{index}] must have exactly the keys {list(SOURCE_KEYS)}, got {source!r}"
            )
        bad = [key for key in SOURCE_KEYS if not isinstance(source[key], str) or not source[key]]
        if bad:
            raise ValueError(f"{where}[{index}]: {', '.join(bad)} must be non-empty strings")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", source["dataset"]):
            raise ValueError(
                f"{where}[{index}]: dataset {source['dataset']!r} may use only letters, digits, "
                "'_', '.', '-'"
            )
        if source["name"] in names:
            raise ValueError(f"{where}: repeated source name {source['name']!r}")
        names.add(source["name"])


__all__ = [
    "BUILD_HINT",
    "ClipManifest",
    "RECORD_KEYS",
    "SCHEMA_VERSION",
    "SOURCE_KEYS",
    "check_relative_path",
    "clip_span",
    "frame_indices",
    "load_clip_manifest",
    "parse_records",
    "read_meta",
    "validate_meta",
    "validate_sources",
]
