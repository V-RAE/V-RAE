from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from vrae.training.common.visualization import save_video_mp4


def fixed_sample_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    count = min(int(num_samples), int(dataset_size))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dataset_size, generator=generator)[:count].tolist()


def _pose_series(
    action: torch.Tensor,
    initial_state: torch.Tensor,
    layout: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position = action[:, layout["position_indices"]]
    orientation = action[:, layout["orientation_indices"]]
    gripper = action[:, int(layout["gripper_index"])]
    mode = str(layout["mode"])
    if mode == "delta":
        initial_position = initial_state[layout["state_position_indices"]]
        position = initial_position.unsqueeze(0) + position.cumsum(dim=0)
        orientation = orientation.cumsum(dim=0)
    elif mode != "absolute":
        raise ValueError("sampling.action_layout.mode must be delta or absolute")
    return position, orientation, gripper


def save_action_plot(
    prediction: torch.Tensor,
    target: torch.Tensor,
    initial_state: torch.Tensor,
    path: Path,
    layout: dict[str, Any],
) -> Path:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "Predicted and ground-truth actions must have equal [T,D] shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    pred_position, pred_orientation, pred_gripper = _pose_series(
        prediction,
        initial_state,
        layout,
    )
    true_position, true_orientation, true_gripper = _pose_series(
        target,
        initial_state,
        layout,
    )

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 5))
    position_axis = figure.add_subplot(1, 2, 1, projection="3d")
    signals_axis = figure.add_subplot(1, 2, 2)
    position_axis.plot(*pred_position.T.numpy(), label="prediction")
    position_axis.plot(*true_position.T.numpy(), label="ground truth")
    position_axis.set_title("3D end-effector position")
    position_axis.set_xlabel("x")
    position_axis.set_ylabel("y")
    position_axis.set_zlabel("z")
    position_axis.legend()
    colors = ("tab:red", "tab:green", "tab:blue")
    for index, (name, color) in enumerate(
        zip(("rot_x", "rot_y", "rot_z"), colors, strict=True)
    ):
        signals_axis.plot(
            pred_orientation[:, index].numpy(),
            color=color,
            label=f"pred {name}",
        )
        signals_axis.plot(
            true_orientation[:, index].numpy(),
            color=color,
            linestyle="--",
            label=f"gt {name}",
        )
    signals_axis.plot(pred_gripper.numpy(), color="black", label="pred gripper")
    signals_axis.plot(
        true_gripper.numpy(),
        color="black",
        linestyle="--",
        label="gt gripper",
    )
    signals_axis.set_title("Orientation and gripper")
    signals_axis.set_xlabel("action step")
    signals_axis.legend(ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_sampling(
    model: nn.Module,
    dataset,
    indices: list[int],
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    wandb_logger,
) -> tuple[float, float]:
    started_at = time.perf_counter()
    video_paths: list[Path] = []
    plot_paths: list[Path] = []
    seed = int(config.get("seed", 3407))
    fps = int(config.get("fps", 10))
    num_frames = int(config["num_video_frames"])
    action_horizon = int(config["action_horizon"])
    layout = dict(config["action_layout"])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sample_index, dataset_index in enumerate(indices):
                sample = dataset[dataset_index]
                video = sample["video"]
                if video.ndim != 4 or video.shape[0] != 3:
                    raise ValueError(
                        f"Sampling video must be [3,T,H,W], got {tuple(video.shape)}"
                    )
                infer_kwargs = {
                    "prompt": None if "context" in sample else sample["prompt"],
                    "input_image": video[:, 0],
                    "num_video_frames": num_frames,
                    "action_horizon": action_horizon,
                    "action": sample["action"],
                    "proprio": sample["proprio"][0],
                    "context": sample.get("context"),
                    "context_mask": sample.get("context_mask"),
                    "num_inference_steps": int(config.get("num_inference_steps", 20)),
                    "seed": seed + sample_index,
                }
                result = model.infer_joint(**infer_kwargs)
                frames = result["video"]
                predicted_video = torch.stack(
                    [pil_to_tensor(frame.convert("RGB")) for frame in frames]
                )
                if predicted_video.shape[0] != num_frames:
                    raise ValueError(
                        "Sample output frame count mismatch: "
                        f"got {predicted_video.shape[0]}, expected {num_frames}"
                    )
                video_path = output_dir / f"step-{step:08d}-sample-{sample_index:02d}.mp4"
                if bool(config.get("save_video", True)):
                    save_video_mp4(predicted_video, video_path, fps=fps)
                    video_paths.append(video_path)

                predicted_action = dataset.denormalize_action(result["action"])
                target_action = dataset.denormalize_action(sample["action"])
                initial_state = dataset.denormalize_state(sample["proprio"][0])
                plot_path = (
                    output_dir
                    / f"step-{step:08d}-sample-{sample_index:02d}-trajectory.png"
                )
                if bool(config.get("save_trajectory", True)):
                    save_action_plot(
                        predicted_action,
                        target_action,
                        initial_state,
                        plot_path,
                        layout,
                    )
                    plot_paths.append(plot_path)
    finally:
        model.train(was_training)
    sample_ms = (time.perf_counter() - started_at) * 1000.0

    upload_started_at = time.perf_counter()
    if bool(config.get("upload_wandb", True)):
        wandb_logger.log_video_files("sample", video_paths, step=step)
        wandb_logger.log_image_files("sample/trajectory", plot_paths, step=step)
        wandb_logger.log({"sample/step": step}, step=step)
    upload_ms = (time.perf_counter() - upload_started_at) * 1000.0
    return sample_ms, upload_ms
