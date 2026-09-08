from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vrae.checkpoint import CheckpointError
from vrae.config import save_resolved_config

STAGE1_STRUCTURE_FIELDS = (
    "name",
    "variant",
    "layers",
    "fusion",
    "hidden_size",
    "pixel_normalization",
    "image_size",
    "patch_size",
    "encoder_tubelet_size",
    "pool_group",
    "final_norm_affine",
    "decoder_input_dim",
    "decoder_hidden_size",
    "decoder_depth",
    "decoder_num_heads",
    "decoder_mlp_ratio",
    "decoder_patch_size",
    "decoder_tubelet",
    "decoder_image_size",
    "decoder_num_channels",
    "decoder_layer_norm_eps",
    "decoder_attention_dropout",
    "decoder_attention_mode",
    "decoder_rope_theta",
    "decoder_spatial_position_kind",
    "decoder_spatial_position_trainable_during_stage1",
    "decoder_spatial_position_resize",
    "temporal_compression_ratio",
    "checkpoint_weight_source",
    "runtime_image_size",
    "runtime_grid_size",
    "multiview_enabled",
    "num_views",
    "num_streams",
    "camera_keys",
)

# The declared runtime geometry affects latent statistics and exact resume, but
# it does not affect any V-RAE parameter shape now that encoder and decoder
# grids are derived from each input.  These fields are therefore intentionally
# omitted when transferring reconstruction weights to another resolution.
STAGE1_WEIGHT_COMPATIBILITY_FIELDS = tuple(
    field
    for field in STAGE1_STRUCTURE_FIELDS
    if field not in {"runtime_image_size", "runtime_grid_size"}
)


@dataclass(frozen=True)
class BatchContract:
    global_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    local_micro_batch_size: int

    @property
    def global_micro_batch_size(self) -> int:
        return self.local_micro_batch_size * self.world_size


def resolve_batch_contract(training: Mapping[str, Any], *, world_size: int) -> BatchContract:
    global_batch_size = int(training["global_batch_size"])
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    denominator = int(world_size) * accumulation_steps
    if denominator <= 0 or global_batch_size % denominator:
        raise ValueError(
            "global_batch_size must be divisible by world_size * gradient_accumulation_steps"
        )
    return BatchContract(
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        world_size=int(world_size),
        local_micro_batch_size=global_batch_size // denominator,
    )


def resolve_checkpoint_interval_steps(training: Mapping[str, Any], *, steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    step_value = training.get("checkpoint_interval")
    epoch_value = training.get("checkpoint_interval_epochs")
    if step_value is not None and epoch_value is not None:
        raise ValueError(
            "training.checkpoint_interval and training.checkpoint_interval_epochs "
            "are mutually exclusive"
        )
    if epoch_value is not None:
        epoch_interval = int(epoch_value)
        if epoch_interval <= 0:
            raise ValueError("training.checkpoint_interval_epochs must be positive")
        return epoch_interval * int(steps_per_epoch)
    step_interval = int(step_value if step_value is not None else 1000)
    if step_interval <= 0:
        raise ValueError("training.checkpoint_interval must be positive")
    return step_interval


def structural_stage1_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only latent-definition fields, excluding runtime decoder provenance."""

    return {
        field: copy.deepcopy(metadata[field])
        for field in STAGE1_STRUCTURE_FIELDS
        if field in metadata
    }


def canonical_resume_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize invocation-only fields before exact-resume config comparison."""

    result = copy.deepcopy(dict(config))
    # The path mapping is machine-local and can also be overridden with
    # ``--paths``. Moving a run between storage roots must not change its
    # model/optimizer identity.
    result.pop("paths", None)
    # Runtime resource controls deliberately do not change model, optimizer,
    # sampler, or numerical checkpoint identity. This lets an exact run resume
    # after a safer loader/checkpoint implementation is deployed.
    result.pop("runtime", None)
    training = result.get("training")
    if isinstance(training, dict):
        training["resume"] = None
        training["init_from"] = None
        for runtime_only_key in (
            "num_workers",
            "pin_memory",
            "prefetch_factor",
            "persistent_workers",
            "dataloader_timeout_seconds",
        ):
            training.pop(runtime_only_key, None)
    return result


def _config_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{name}: unexpected in current config")
            elif key not in actual:
                differences.append(f"{name}: missing from current config")
            else:
                differences.extend(_config_differences(expected[key], actual[key], name))
        return differences
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return [f"{prefix}: expected a sequence"]
        if len(expected) != len(actual):
            return [f"{prefix}: length {len(expected)} != {len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_config_differences(left, right, f"{prefix}[{index}]"))
        return differences
    if expected != actual:
        return [f"{prefix}: checkpoint={expected!r}, current={actual!r}"]
    return []


def compare_resolved_configs(
    checkpoint_config: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    differences = _config_differences(
        canonical_resume_config(checkpoint_config),
        canonical_resume_config(current_config),
    )
    if differences:
        detail = "\n".join(differences[:32])
        raise CheckpointError(f"Exact-resume resolved_config mismatch:\n{detail}")


def run_identity(config: Mapping[str, Any]) -> dict[str, str]:
    return {"task": str(config["task"]), "run_name": str(config["run_name"])}


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    run_directory: str | Path | None = None,
) -> None:
    expected_identity = run_identity(config)
    if str(payload.get("task")) != expected_identity["task"]:
        raise CheckpointError(
            f"Checkpoint task {payload.get('task')!r} does not match {expected_identity['task']!r}"
        )
    recorded_identity = payload.get("run_identity")
    if recorded_identity != expected_identity:
        raise CheckpointError(
            f"Checkpoint run identity {recorded_identity!r} does not match {expected_identity!r}"
        )
    resolved_config = payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise CheckpointError("Checkpoint resolved_config must be a mapping")
    compare_resolved_configs(resolved_config, config)
    if checkpoint_path is not None and run_directory is not None:
        checkpoint = Path(checkpoint_path).resolve()
        expected_parent = (Path(run_directory) / "checkpoints").resolve()
        if checkpoint.parent != expected_parent:
            raise CheckpointError(
                "Exact resume must load a checkpoint from the same task/run directory"
            )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return dict(value)


def create_run_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    decoder = model.get("decoder", {}) if isinstance(model, Mapping) else {}
    hidden_size = int(decoder.get("hidden_size", 0)) if isinstance(decoder, Mapping) else 0
    num_heads = int(decoder.get("num_heads", 1)) if isinstance(decoder, Mapping) else 1
    head_dim = hidden_size // num_heads if hidden_size and num_heads else None
    return {
        "run_identity": run_identity(config),
        "wandb_run_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "gradient_accumulation_steps": int(
            config.get("training", {}).get("gradient_accumulation_steps", 1)
        ),
        "global_batch_size": int(config.get("training", {}).get("global_batch_size", 0)),
        "training_dtype": str(config.get("training", {}).get("precision", "fp32")),
        "decoder_execution": {
            "attention_backend_requested": (
                str(decoder.get("attention_backend", "auto"))
                if isinstance(decoder, Mapping)
                else "unavailable"
            ),
            "attention_backend_resolved": "unresolved",
            "attention_dtype": "unresolved",
            "attention_head_dim": head_dim,
            "attention_padding_route": "unresolved",
        },
    }


def update_run_metadata(run: str | Path, values: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(run) / "run_metadata.yaml"
    current = _load_yaml_mapping(path) if path.is_file() else {}
    for key, value in values.items():
        current[key] = copy.deepcopy(value)
    save_resolved_config(current, path)
    return current


def update_decoder_execution_metadata(
    run: str | Path,
    decoder_execution: Mapping[str, Any],
    *,
    training_dtype: str,
) -> dict[str, Any]:
    execution = copy.deepcopy(dict(decoder_execution))
    execution["training_dtype"] = str(training_dtype)
    return update_run_metadata(run, {"decoder_execution": execution})


def load_run_metadata(run: str | Path) -> dict[str, Any]:
    path = Path(run) / "run_metadata.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_yaml_mapping(path)
