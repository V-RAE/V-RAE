from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from vrae.training.common.visualization import save_video_mp4


def fixed_sample_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    count = min(int(num_samples), int(dataset_size))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dataset_size, generator=generator)[:count].tolist()


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(
            f"Axis-angle rotations must have shape [T,3], got {tuple(axis_angle.shape)}"
        )
    theta_squared = axis_angle.square().sum(dim=1, keepdim=True)
    theta = theta_squared.sqrt()
    denominator = theta.clamp_min(1.0e-8)
    sine_scale = torch.where(
        theta_squared > 1.0e-12,
        torch.sin(theta) / denominator,
        1.0 - theta_squared / 6.0,
    )
    cosine_scale = torch.where(
        theta_squared > 1.0e-12,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1.0e-12),
        0.5 - theta_squared / 24.0,
    )
    x, y, z = axis_angle.unbind(dim=1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return (
        identity.unsqueeze(0)
        + sine_scale.unsqueeze(2) * skew
        + cosine_scale.unsqueeze(2) * torch.bmm(skew, skew)
    )


def _pose_trajectory(
    action: torch.Tensor,
    initial_state: torch.Tensor,
    layout: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    position = action[:, layout["position_indices"]]
    orientation = action[:, layout["orientation_indices"]]
    initial_position = initial_state[layout["state_position_indices"]]
    state_orientation_indices = layout.get(
        "state_orientation_indices",
        layout["orientation_indices"],
    )
    initial_orientation = initial_state[state_orientation_indices]
    initial_rotation = _axis_angle_to_matrix(initial_orientation.unsqueeze(0))[0]
    mode = str(layout["mode"])
    if mode == "delta":
        position = torch.cat(
            (
                initial_position.unsqueeze(0),
                initial_position.unsqueeze(0) + position.cumsum(dim=0),
            ),
            dim=0,
        )
        rotations = [initial_rotation]
        for delta_rotation in _axis_angle_to_matrix(orientation):
            # LIBERO's OSC controller applies world-frame rotation deltas on the left.
            rotations.append(delta_rotation @ rotations[-1])
        rotation = torch.stack(rotations)
    elif mode != "absolute":
        raise ValueError("sampling.action_layout.mode must be delta or absolute")
    else:
        position = torch.cat((initial_position.unsqueeze(0), position), dim=0)
        rotation = torch.cat(
            (
                initial_rotation.unsqueeze(0),
                _axis_angle_to_matrix(orientation),
            ),
            dim=0,
        )
    return position, rotation


def _trajectory_bounds(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    points = torch.cat((prediction, target), dim=0)
    minimum = points.amin(dim=0)
    maximum = points.amax(dim=0)
    center = (minimum + maximum) * 0.5
    span = max(float((maximum - minimum).max()), 1.0e-3)
    return center, span


def _project_pose_axes(
    axis,
    position: np.ndarray,
    rotation: np.ndarray,
    length: float,
    frame_height: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    from mpl_toolkits.mplot3d import proj3d

    endpoints = position[None, :] + rotation.T * length  # [3 axes, xyz]
    points = np.concatenate((position[None, :], endpoints), axis=0)
    projected_x, projected_y, _ = proj3d.proj_transform(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        axis.get_proj(),
    )
    display_points = axis.transData.transform(
        np.stack((projected_x, projected_y), axis=1)
    )
    display_points[:, 1] = frame_height - display_points[:, 1]
    origin = tuple(display_points[0])
    return [
        (origin, tuple(display_points[axis_index + 1]))
        for axis_index in range(3)
    ]


def _draw_projected_pose(
    draw: ImageDraw.ImageDraw,
    projected_axes: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    alpha: int,
    width: int,
    show_labels: bool,
) -> None:
    colors = ((214, 39, 40), (44, 160, 44), (31, 119, 180))
    for axis_index, (origin, endpoint) in enumerate(projected_axes):
        color = (*colors[axis_index], alpha)
        draw.line((origin, endpoint), fill=color, width=width)
        direction = np.asarray(endpoint) - np.asarray(origin)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            unit = direction / norm
            perpendicular = np.asarray((-unit[1], unit[0]))
            arrow_base = np.asarray(endpoint) - unit * 7.0
            arrow = (
                endpoint,
                tuple(arrow_base + perpendicular * 3.0),
                tuple(arrow_base - perpendicular * 3.0),
            )
            draw.polygon(arrow, fill=color)
        if show_labels:
            draw.text(
                (endpoint[0] + 3.0, endpoint[1] - 8.0),
                "xyz"[axis_index],
                fill=color,
            )


def save_action_video(
    prediction: torch.Tensor,
    target: torch.Tensor,
    initial_state: torch.Tensor,
    path: Path,
    layout: dict[str, Any],
    *,
    fps: int,
) -> Path:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "Predicted and ground-truth actions must have equal [T,D] shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    pred_position, pred_rotation = _pose_trajectory(
        prediction,
        initial_state,
        layout,
    )
    true_position, true_rotation = _pose_trajectory(
        target,
        initial_state,
        layout,
    )

    import matplotlib.pyplot as plt

    pred_position = pred_position.detach().cpu()
    pred_rotation = pred_rotation.detach().cpu()
    true_position = true_position.detach().cpu()
    true_rotation = true_rotation.detach().cpu()
    center, span = _trajectory_bounds(pred_position, true_position)
    half_extent = span * 0.65
    pose_axis_length = span * 0.09
    total_steps = pred_position.shape[0]
    intro_frames = max(int(fps), 1)
    figure = plt.figure(figsize=(7, 7), dpi=120)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    axis.plot(
        *pred_position.T.numpy(),
        color="#e66101",
        linewidth=2.4,
        label="prediction",
    )
    axis.plot(
        *true_position.T.numpy(),
        color="#5e3c99",
        linewidth=2.0,
        linestyle="--",
        label="ground truth",
    )
    axis.set_xlim(float(center[0] - half_extent), float(center[0] + half_extent))
    axis.set_ylim(float(center[1] - half_extent), float(center[1] + half_extent))
    axis.set_zlim(float(center[2] - half_extent), float(center[2] + half_extent))
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend(loc="upper right")
    axis.set_title("EEF pose trajectory")
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.94)

    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    base_frame = rgba[..., :3].copy()
    frame_height = base_frame.shape[0]
    intro_frame = torch.from_numpy(base_frame.copy()).permute(2, 0, 1)
    frames = [intro_frame.clone() for _ in range(intro_frames)]
    for pose_step in range(total_steps):
        frame = Image.fromarray(base_frame.copy())
        draw = ImageDraw.Draw(frame, mode="RGBA")
        true_axes = _project_pose_axes(
            axis,
            true_position[pose_step].numpy(),
            true_rotation[pose_step].numpy(),
            pose_axis_length,
            frame_height,
        )
        _draw_projected_pose(
            draw,
            true_axes,
            alpha=85,
            width=2,
            show_labels=False,
        )
        predicted_axes = _project_pose_axes(
            axis,
            pred_position[pose_step].numpy(),
            pred_rotation[pose_step].numpy(),
            pose_axis_length,
            frame_height,
        )
        _draw_projected_pose(
            draw,
            predicted_axes,
            alpha=255,
            width=3,
            show_labels=True,
        )
        frame_array = np.asarray(frame)
        frames.append(torch.from_numpy(frame_array.copy()).permute(2, 0, 1))
    try:
        return save_video_mp4(torch.stack(frames), path, fps=fps)
    finally:
        plt.close(figure)


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
    trajectory_paths: list[Path] = []
    seed = int(config.get("seed", 3407))
    fps = int(config.get("fps", 10))
    num_frames = int(config["num_video_frames"])
    action_horizon = int(config["action_horizon"])
    layout = dict(config["action_layout"])
    training_modes = {
        module: module.training
        for module in model.modules()
    }
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
                trajectory_path = (
                    output_dir
                    / f"step-{step:08d}-sample-{sample_index:02d}-trajectory.mp4"
                )
                if bool(config.get("save_trajectory", True)):
                    save_action_video(
                        predicted_action,
                        target_action,
                        initial_state,
                        trajectory_path,
                        layout,
                        fps=fps,
                    )
                    trajectory_paths.append(trajectory_path)
    finally:
        for module, training in training_modes.items():
            module.training = training
    sample_ms = (time.perf_counter() - started_at) * 1000.0

    upload_started_at = time.perf_counter()
    if bool(config.get("upload_wandb", True)):
        wandb_logger.log_video_files("sample", video_paths, step=step)
        wandb_logger.log_video_files(
            "sample/trajectory",
            trajectory_paths,
            step=step,
        )
        wandb_logger.log({"sample/step": step}, step=step)
    upload_ms = (time.perf_counter() - upload_started_at) * 1000.0
    return sample_ms, upload_ms
