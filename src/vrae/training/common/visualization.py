from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 6:
        batch, time, views, channels, height, width = video.shape
        video = video.permute(0, 2, 1, 3, 4, 5).reshape(batch * views, time, channels, height, width)
    if video.ndim != 5:
        raise ValueError("Expected video [B,T,C,H,W] or [B,T,V,C,H,W]")
    if video.dtype == torch.uint8:
        return video
    return video.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)


def comparison_video(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    if real.shape != reconstructed.shape:
        raise ValueError("Real and reconstructed videos must have equal shape")
    return torch.cat((video_to_uint8(real), video_to_uint8(reconstructed)), dim=-1)


def save_video_mp4(video: torch.Tensor, path: str | Path, *, fps: int) -> Path:
    """Encode one ``[T,C,H,W]`` uint8/RGB clip as an MP4 file."""

    if video.ndim != 4:
        raise ValueError(f"Expected one video [T,C,H,W], got {tuple(video.shape)}")
    video = video_to_uint8(video.unsqueeze(0))[0]
    frames = video.permute(0, 2, 3, 1).contiguous().numpy()
    if frames.shape[-1] == 1:
        frames = frames.repeat(3, axis=-1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise ValueError("MP4 encoding requires one, three, or four channels")

    import av

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with av.open(str(temporary), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                packet = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
                for encoded in packet:
                    container.mux(encoded)
            for encoded in stream.encode():
                container.mux(encoded)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def save_sample_tensors(tensors: Mapping[str, torch.Tensor], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save({name: value.detach().cpu() for name, value in tensors.items()}, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
