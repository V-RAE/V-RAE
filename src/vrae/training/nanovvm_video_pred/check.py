"""Check a NanoVVM prediction config against the files it names.

Without flags: resolve every train/val source against the data root and check
that each video root, manifest, and meta.yaml exists, reading only meta.yaml.
--load-manifests also runs the full manifest validation (schema, sampling,
frame ranges, record count, SHA-256) and checks each source's first video.
--decode N decodes the first clip of each source for N sources and reports the
source frames of the condition and the future.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from vrae.config import load_config
from vrae.paths import find_project_root, load_project_paths
from vrae.training.nanovvm_video_pred.data import (
    NanoVVMVideoPredictionDataset,
    load_split_sources,
    resolve_data_root,
    resolve_split_sources,
)
from vrae.training.nanovvm_video_pred.manifest import read_meta
from vrae.training.nanovvm_video_pred.train import validate_build


def check_config(
    config: dict[str, Any],
    paths: Any,
    *,
    load_manifests: bool = False,
    decode: int = 0,
) -> dict[str, Any]:
    validate_build(config)
    report: dict[str, Any] = {"data_root": str(resolve_data_root(config, paths)), "splits": {}}
    splits = ["train"] + (["val"] if config["data"].get("val") is not None else [])
    for split in splits:
        entries = []
        for item in resolve_split_sources(config, paths, split):
            meta = read_meta(item["manifest_meta"])
            entries.append(
                {
                    "name": item["name"],
                    "dataset": item["dataset"],
                    "root": str(item["root"]),
                    "manifest": str(item["manifest"]),
                    "manifest_meta": str(item["manifest_meta"]),
                    "meta_split": meta.get("split"),
                    "meta_total_clips": meta.get("statistics", {}).get("total_clips"),
                    "meta_sampling": {
                        key: meta.get("sampling", {}).get(key)
                        for key in ("rgb_seq_len", "frame_interval", "source_frame_span")
                    },
                }
            )
        summary: dict[str, Any] = {
            "sources": entries,
            "meta_total_clips": sum(int(entry["meta_total_clips"] or 0) for entry in entries),
        }
        if load_manifests or decode:
            sources = load_split_sources(config, paths, split)
            dataset = NanoVVMVideoPredictionDataset(
                sources,
                split=split,
                image_size=int(config["data"].get("image_size", 256)),
                preprocessing=config["data"].get("preprocessing"),
            )
            summary["validated_clips"] = len(dataset)
            summary["shares"] = dataset.summary()
            decoded = []
            offset = 0
            for source in sources[: max(0, int(decode))]:
                item = dataset[offset]
                decoded.append(
                    {
                        "sample_id": item["sample_id"],
                        "condition_source_frame": int(item["frame_indices"][0]),
                        "future_source_frames": [int(v) for v in item["frame_indices"][1:]],
                        "condition": [list(item["condition"].shape), str(item["condition"].dtype)],
                        "future": [list(item["future"].shape), str(item["future"].dtype)],
                    }
                )
                offset += len(source)
            if decoded:
                summary["decoded"] = decoded
        report["splits"][split] = summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check NanoVVM prediction data paths and manifests"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--load-manifests", action="store_true")
    parser.add_argument("--decode", type=int, default=0, help="decode the first clip of N sources")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    report = check_config(
        config, paths, load_manifests=arguments.load_manifests, decode=arguments.decode
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
