"""Read and inspect one batch from the local V-RAE LIBERO dataset."""

from torch.utils.data import DataLoader

from vrae.data import LeRobotVideoDataset
from vrae.data.lerobot import build_temporal_indices


def test_build_temporal_indices_uses_frame_interval() -> None:
    assert build_temporal_indices(16, 4) == list(range(0, 64, 4))


def test_build_temporal_indices_rejects_non_positive_interval() -> None:
    try:
        build_temporal_indices(16, 0)
    except ValueError as error:
        assert "frame_interval" in str(error)
    else:
        raise AssertionError("expected frame_interval validation to fail")


def _describe(name: str, value: object) -> None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        print(f"{name}: {type(value).__name__} = {value!r}")
    else:
        print(f"{name}: shape={tuple(shape)}, dtype={dtype}")


def main() -> None:
    import ipdb;ipdb.set_trace()
    dataset = LeRobotVideoDataset(
        root="/zsh/cache/data/Lerobot/libero",
        repo_id="libero",
        frame_num=16,
        camera_keys=[
            {"key": "observation.images.image", "name": "head", "stream_id": 0},
            {"key": "observation.images.image2", "name": "wrist", "stream_id": 1},
        ],
    )

    print(f"dataset_length: {len(dataset)}")
    print(f"frame_num: {dataset.frame_num}")
    print(f"delta_indices: {dataset.delta_indices}")
    print(f"image_transforms: {dataset.image_transforms}")

    item = dataset[0]
    print("\n--- item[0] ---")
    for key, value in item.items():
        _describe(key, value)
    print(
        "video_range:",
        float(item["video"].min()),
        float(item["video"].max()),
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("\n--- batch[0:2] ---")
    for key, value in batch.items():
        _describe(key, value)


if __name__ == "__main__":
    main()
