#!/usr/bin/env python3
"""Create a small episode/view manifest from a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vrae.data.lerobot3.lerobot_dataset import LeRobotDatasetMetadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="libero")
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    views = metadata.image_keys
    if not views:
        raise ValueError("LeRobot dataset has no image features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("episode_index", "view", "frame_count", "task_index", "path"),
        )
        writer.writeheader()
        for episode in metadata.episodes:
            episode_index = int(episode["episode_index"])
            task_ref = episode["tasks"][0]
            try:
                task_index = int(metadata.tasks.loc[task_ref, "task_index"])
            except (KeyError, TypeError):
                task_index = int(metadata.tasks.iloc[int(task_ref)]["task_index"])
            for view in views:
                writer.writerow(
                    {
                        "episode_index": episode_index,
                        "view": view,
                        "frame_count": int(episode["length"]),
                        "task_index": task_index,
                        "path": str(metadata.get_data_file_path(episode_index)),
                    }
                )


if __name__ == "__main__":
    main()
