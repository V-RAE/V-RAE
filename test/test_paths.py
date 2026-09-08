from __future__ import annotations

from pathlib import Path

from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_CONFIG = PROJECT_ROOT / "configs/training/reconstruction/vjepa2_1_libero_reconstruction_random_interval_bs8_pretrained.yaml"
GENERATION_CONFIG = PROJECT_ROOT / "configs/training/generation/libero_videodit.yaml"


def build_libero_paths(reconstruction: dict) -> ProjectPaths:
    """Build the path registry from the checked-in LIBERO configuration."""

    return load_project_paths(
        reconstruction,
        project_root=PROJECT_ROOT,
        override={
            "project_root": str(PROJECT_ROOT),
            "datasets": {
                "lerobot": reconstruction["data"]["root"],
            },
            "third_party": {
                "vjepa2_1": str(PROJECT_ROOT / "third_party/vjepa2"),
            },
        },
    )


def main() -> None:
    reconstruction = load_config(RECONSTRUCTION_CONFIG)
    generation = load_config(GENERATION_CONFIG)

    reconstruction_root = find_project_root(RECONSTRUCTION_CONFIG)
    generation_root = find_project_root(GENERATION_CONFIG)
    if reconstruction_root != PROJECT_ROOT or generation_root != PROJECT_ROOT:
        raise RuntimeError("LIBERO training configs did not resolve to the repository root")

    shared_data_fields = (
        "root",
        "repo_id",
        "split",
        "camera_keys",
        "num_frames",
        "image_size",
        "fps",
        "seed",
    )
    for field in shared_data_fields:
        if reconstruction["data"].get(field) != generation["data"].get(field):
            raise RuntimeError(f"Reconstruction and generation disagree on data.{field}")

    paths = build_libero_paths(reconstruction)
    libero_dataset = paths.dataset("lerobot")
    vjepa_source = paths.source("vjepa2_1")
    encoder_checkpoint = paths.checkpoint(
        reconstruction["model"]["encoder"]["checkpoint"]
    )

    reconstruction_run = paths.training_run(
        reconstruction["task"],
        reconstruction["run_name"],
    )
    generation_run = paths.training_run(
        generation["task"],
        generation["run_name"],
    )
    stage1_checkpoint = paths.checkpoint(
        generation["stage1"]["checkpoint"],
        require_exists=False,
    )
    latent_statistics = paths.checkpoint(
        generation["latent_normalizer"]["path"],
        require_exists=False,
    )

    expected_stage1_checkpoint = reconstruction_run / "checkpoints/latest.pt"
    if stage1_checkpoint != expected_stage1_checkpoint:
        raise RuntimeError(
            "Generation stage1 checkpoint does not point to the reconstruction run: "
            f"{stage1_checkpoint} != {expected_stage1_checkpoint}"
        )

    print(f"project_root:          {paths.project_root}")
    print(f"libero_dataset:        {libero_dataset}")
    print(f"vjepa_source:           {vjepa_source}")
    print(f"encoder_checkpoint:    {encoder_checkpoint}")
    print(f"reconstruction_run:    {reconstruction_run}")
    print(f"generation_run:        {generation_run}")
    print(f"stage1_checkpoint:     {stage1_checkpoint}")
    print(f"latent_statistics:     {latent_statistics}")


if __name__ == "__main__":
    main()
