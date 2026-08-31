from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REQUIRED_WANDB_FIELDS = (
    "project",
    "group",
    "tags",
    "mode",
    "resume",
    "log_interval",
    "sample_interval",
)


def bind_wandb_run_name(config: Mapping[str, Any], run_name: str) -> dict[str, Any]:
    """Return a W&B config whose run name follows the training run name."""
    name = str(run_name)
    if not name:
        raise ValueError("run_name must be non-empty before binding the W&B run name")
    bound = dict(config)
    bound["name"] = name
    return bound


def validate_wandb_config(config: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_WANDB_FIELDS if field not in config]
    if missing:
        raise ValueError(f"W&B config is missing fields: {missing}")
    if int(config["log_interval"]) <= 0 or int(config["sample_interval"]) <= 0:
        raise ValueError("W&B log_interval and sample_interval must be positive")
    if str(config["resume"]) not in {"allow", "must", "never", "auto"}:
        raise ValueError("W&B resume must be allow, must, never, or auto")


def _api_key_source(wandb_module: Any) -> str | None:
    """Return where an API key was found without exposing the key itself."""

    if os.environ.get("WANDB_API_KEY", "").strip():
        return "WANDB_API_KEY"
    api = getattr(wandb_module, "api", None)
    if api is not None and getattr(api, "api_key", None):
        return "saved login"
    return None


class WandbLogger:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        enabled: bool,
        run_dir: str,
        run_id: str | None = None,
        exact_resume: bool = False,
        resume_from_step: int | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.run = None
        self._wandb = None
        configured_mode = str(config.get("mode", "offline")).strip().lower()
        runtime_mode = os.environ.get("WANDB_MODE", configured_mode).strip().lower()
        if not enabled:
            return
        if configured_mode == "disabled" or runtime_mode == "disabled":
            print("[wandb] Logging is disabled.", flush=True)
            return
        validate_wandb_config(config)
        import wandb

        self._wandb = wandb
        if runtime_mode == "online":
            key_source = _api_key_source(wandb)
            if key_source is None:
                runtime_mode = "offline"
                os.environ["WANDB_MODE"] = runtime_mode
                print(
                    "[wandb] No API key detected; automatically switched online -> offline.",
                    flush=True,
                )
            else:
                print(
                    f"[wandb] API key detected ({key_source}); online logging enabled.",
                    flush=True,
                )
        elif runtime_mode == "offline":
            print("[wandb] Offline logging enabled.", flush=True)

        init_kwargs: dict[str, Any] = {
            "project": str(config.get("project", "V-RAE")),
            "name": config.get("name"),
            "group": config.get("group"),
            "tags": config.get("tags"),
            "mode": runtime_mode,
            "dir": run_dir,
            "id": run_id,
        }
        if run_config is not None:
            init_kwargs["config"] = dict(run_config)
        start_new_on_resume = os.environ.get("WANDB_START_NEW_RUN_ON_RESUME", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        enable_rewind = os.environ.get("WANDB_ENABLE_REWIND", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if exact_resume and start_new_on_resume:
            init_kwargs["resume"] = "allow"
        elif (
            exact_resume
            and enable_rewind
            and resume_from_step is not None
            and runtime_mode == "online"
        ):
            if run_id is None:
                raise ValueError("Online checkpoint rewind requires a W&B run id")
            init_kwargs["resume_from"] = f"{run_id}?_step={int(resume_from_step)}"
        else:
            init_kwargs["resume"] = "must" if exact_resume else str(config["resume"])
        self.run = wandb.init(
            **init_kwargs,
        )

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        if self.run is not None:
            self.run.log(dict(values), step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def log_video(
        self,
        name: str,
        video: torch.Tensor,
        *,
        step: int,
        fps: int,
        caption: str | None = None,
    ) -> None:
        if self.run is None or self._wandb is None:
            return
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[2] not in {1, 3, 4}:
            raise ValueError("W&B video must be [B,T,C,H,W]")
        values = video.detach().cpu()
        if values.dtype != torch.uint8:
            values = values.float().clamp(0, 1).mul(255).round().to(torch.uint8)
        prefix = str(name).rstrip("/")
        media = {
            f"{prefix}/video_{index:03d}": self._wandb.Video(
                item,
                fps=int(fps),
                format="mp4",
                caption=caption,
            )
            for index, item in enumerate(values)
        }
        self.run.log(media, step=int(step))

    def log_video_files(
        self,
        name: str,
        paths: Sequence[str | Path],
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        """Upload already encoded MP4 files without requiring MoviePy."""

        if self.run is None or self._wandb is None:
            return
        prefix = str(name).rstrip("/")
        media = {
            f"{prefix}/video_{index:03d}": self._wandb.Video(
                str(path),
                format="mp4",
                caption=caption,
            )
            for index, path in enumerate(paths)
        }
        if media:
            self.run.log(media, step=int(step))
