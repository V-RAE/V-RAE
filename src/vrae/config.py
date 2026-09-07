from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a checked-in or resolved configuration violates the contract."""


def _checkpoint_relative_to_project(value: object, field: str) -> None:
    if value is None or value == "":
        return
    path = Path(str(value))
    if path.is_absolute() or not path.parts or path.parts[0] != "ckpts":
        raise ConfigError(f"{field} must be a project-relative path under ckpts/")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Cyclic config include: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, MutableMapping):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    includes = payload.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, Sequence):
        raise ConfigError(f"include must be a path or path list: {path}")
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = deep_merge(merged, _load_yaml_file(include_path, (*stack, path)))
    return deep_merge(merged, payload)


def load_config(path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = _load_yaml_file(Path(path), ())
    if overrides:
        config = deep_merge(config, overrides)
    validate_config(config)
    return config


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
    temporary.replace(output)
    return output


def validate_config(config: Mapping[str, Any]) -> None:
    task = str(config.get("task", ""))
    trainer = config.get("trainer")
    if task in {"ci2v", "c2v", "ti2v", "t2v"} and trainer not in {None, "libero_videogen"}:
        raise ConfigError("Video generation tasks require trainer=libero_videogen")
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    pooling = model.get("pooling", {})
    if isinstance(pooling, Mapping):
        if pooling.get("output_norm_affine", False) is not False:
            raise ConfigError("model.pooling.output_norm_affine must be false")
        if pooling.get("name", "temporal_attention") != "temporal_attention":
            raise ConfigError("Only the temporal_attention pooler is registered")
    encoder = model.get("encoder")
    if isinstance(encoder, Mapping) and "name" in encoder:
        allowed = {"vjepa2_1"}
        encoder_name = str(encoder["name"])
        if encoder_name not in allowed:
            raise ConfigError(
                f"Unknown encoder {encoder['name']!r}; expected one of {sorted(allowed)}"
            )
        expected_tubelet = 2 if encoder_name == "vjepa2_1" else 1
        if int(encoder.get("encoder_tubelet_size", expected_tubelet)) != expected_tubelet:
            raise ConfigError(f"{encoder_name} has a fixed encoder_tubelet_size={expected_tubelet}")
        expected_group = 2 if encoder_name == "vjepa2_1" else 4
        if int(pooling.get("group_size", expected_group)) != expected_group:
            raise ConfigError(f"{encoder_name} requires pooling.group_size={expected_group}")
        _checkpoint_relative_to_project(encoder.get("checkpoint"), "model.encoder.checkpoint")
    decoder = model.get("decoder", {})
    if isinstance(decoder, Mapping):
        attention_mode = decoder.get("attention_mode", "chunk_causal")
        if attention_mode not in {"chunk_causal", "full"}:
            raise ConfigError("decoder.attention_mode must be chunk_causal or full")
        init = decoder.get("init", "scratch")
        if init not in {"scratch", "raev2_image"}:
            raise ConfigError("decoder.init must be scratch or raev2_image")
        if "tubelet_size" in decoder and int(decoder["tubelet_size"]) != 4:
            raise ConfigError("decoder.tubelet_size must be 4")
        backend = str(decoder.get("attention_backend", "auto"))
        if backend not in {"sdpa", "fa3", "fa3_fwd", "fa4_cute", "auto"}:
            raise ConfigError(f"Unknown decoder attention backend: {backend}")
        _checkpoint_relative_to_project(decoder.get("checkpoint"), "model.decoder.checkpoint")
    multiview = model.get("multiview", {})
    if multiview:
        if not isinstance(multiview, Mapping):
            raise ConfigError("model.multiview must be a mapping")
        num_views = int(multiview.get("num_views", 1))
        num_streams = int(multiview.get("num_streams", num_views))
        if num_views <= 0 or num_streams <= 0 or num_views > num_streams:
            raise ConfigError("model.multiview requires 0 < num_views <= num_streams")
    data = config.get("data", {})
    if isinstance(data, Mapping):
        if task == "ci2v":
            if str(data.get("dataset", "")) != "reconstruction":
                raise ConfigError("CI2V requires data.dataset=reconstruction")
            num_frames = int(data.get("num_frames", 0))
            if num_frames != 21:
                raise ConfigError("CI2V data.num_frames must be 21 (1 condition + 20 target frames)")
            task_config = config.get("task_config", {})
            ci2v_config = task_config.get("ci2v", {}) if isinstance(task_config, Mapping) else {}
            if isinstance(ci2v_config, Mapping):
                target_frames = ci2v_config.get("video_target", {}).get("frames", num_frames - 1)
                if int(target_frames) != num_frames - 1:
                    raise ConfigError("CI2V video_target.frames must equal data.num_frames - 1")
                decode_frames = ci2v_config.get("sampling", {}).get("decode_frames", num_frames - 1)
                if int(decode_frames) != num_frames - 1:
                    raise ConfigError("CI2V sampling.decode_frames must equal data.num_frames - 1")
        if int(data.get("frame_interval", 1)) <= 0:
            raise ConfigError("data.frame_interval must be positive")
        if str(data.get("dataset", "")) == "lerobot":
            cameras = data.get("camera_keys")
            if (
                not isinstance(cameras, Sequence)
                or isinstance(cameras, (str, bytes))
                or not cameras
            ):
                raise ConfigError("data.camera_keys must be a non-empty list for LeRobot")
            multiview = model.get("multiview", {}) if isinstance(model, Mapping) else {}
            enabled = (
                bool(multiview.get("enabled", len(cameras) > 1))
                if isinstance(multiview, Mapping)
                else len(cameras) > 1
            )
            if enabled and int(multiview.get("num_views", len(cameras))) != len(cameras):
                raise ConfigError("model.multiview.num_views must equal len(data.camera_keys)")
            if any(not isinstance(camera, Mapping) or "key" not in camera for camera in cameras):
                raise ConfigError("each data.camera_keys entry must be a mapping with key")
        if str(data.get("dataset", "")) == "lerobot_reconstruction":
            if not data.get("manifest"):
                raise ConfigError("data.manifest is required for lerobot_reconstruction")
            interval_min = int(data.get("frame_interval_min", 1))
            interval_max = int(data.get("frame_interval_max", 4))
            if interval_min <= 0 or interval_max < interval_min:
                raise ConfigError("data.frame_interval_min/max are invalid")
    training = config.get("training", {})
    if isinstance(training, Mapping) and training.get("resume") and training.get("init_from"):
        raise ConfigError("training.resume and training.init_from are mutually exclusive")
    if isinstance(training, Mapping):
        if int(training.get("num_workers", 4)) < 0:
            raise ConfigError("training.num_workers must be non-negative")
        if int(training.get("prefetch_factor", 4)) <= 0:
            raise ConfigError("training.prefetch_factor must be positive")
        if float(training.get("dataloader_timeout_seconds", 300.0)) <= 0:
            raise ConfigError("training.dataloader_timeout_seconds must be positive")
        _checkpoint_relative_to_project(training.get("resume"), "training.resume")
        _checkpoint_relative_to_project(training.get("init_from"), "training.init_from")
        checkpoint_interval = training.get("checkpoint_interval")
        checkpoint_interval_epochs = training.get("checkpoint_interval_epochs")
        if checkpoint_interval is not None and checkpoint_interval_epochs is not None:
            raise ConfigError(
                "training.checkpoint_interval and training.checkpoint_interval_epochs "
                "are mutually exclusive"
            )
        if checkpoint_interval is not None and int(checkpoint_interval) <= 0:
            raise ConfigError("training.checkpoint_interval must be positive")
        if checkpoint_interval_epochs is not None and int(checkpoint_interval_epochs) <= 0:
            raise ConfigError("training.checkpoint_interval_epochs must be positive")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ConfigError("runtime must be a mapping")
    data_pipeline = runtime.get("data_pipeline", {})
    if not isinstance(data_pipeline, Mapping):
        raise ConfigError("runtime.data_pipeline must be a mapping")
    if data_pipeline:
        if data_pipeline.get("kind") != "lerobot_bounded":
            raise ConfigError("runtime.data_pipeline.kind must be lerobot_bounded")
        for key in (
            "worker_threads",
            "max_inflight",
            "max_buffered_batches",
            "async_prefetch_batches",
            "max_decode_attempts_per_batch",
            "glibc_arena_max",
            "glibc_trim_threshold_bytes",
        ):
            if int(data_pipeline.get(key, 1)) <= 0:
                raise ConfigError(f"runtime.data_pipeline.{key} must be positive")
    host_memory = runtime.get("host_memory", {})
    if not isinstance(host_memory, Mapping):
        raise ConfigError("runtime.host_memory must be a mapping")
    if float(host_memory.get("min_available_gb", 0.0)) < 0.0:
        raise ConfigError("runtime.host_memory.min_available_gb must be non-negative")
    stage1 = config.get("stage1", {})
    if isinstance(stage1, Mapping):
        _checkpoint_relative_to_project(stage1.get("checkpoint"), "stage1.checkpoint")
    latent_normalizer = config.get("latent_normalizer", {})
    if isinstance(latent_normalizer, Mapping):
        _checkpoint_relative_to_project(
            latent_normalizer.get("path"),
            "latent_normalizer.path",
        )
    loss = config.get("loss", {})
    if isinstance(loss, Mapping):
        if "perceptual_frames" in loss and int(loss["perceptual_frames"]) <= 0:
            raise ConfigError("loss.perceptual_frames must be positive")
        if "perceptual_frames" in loss and "perceptual_frames_per_chunk" in loss:
            raise ConfigError(
                "loss.perceptual_frames and loss.perceptual_frames_per_chunk are mutually exclusive"
            )
        if "perceptual_frames_per_chunk" in loss:
            frames_per_chunk = int(loss["perceptual_frames_per_chunk"])
            chunk_size = int(loss.get("perceptual_chunk_size", 4))
            if not 0 < frames_per_chunk <= chunk_size:
                raise ConfigError(
                    "loss.perceptual_frames_per_chunk must be between 1 and perceptual_chunk_size"
                )
        _checkpoint_relative_to_project(
            loss.get("backbone_checkpoint"),
            "loss.backbone_checkpoint",
        )
        _checkpoint_relative_to_project(
            loss.get("calibration_checkpoint"),
            "loss.calibration_checkpoint",
        )


def get_required(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required config field: {dotted_key}")
        current = current[part]
    return current
