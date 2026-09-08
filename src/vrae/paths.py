from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vrae.config import ConfigError

TRAINING_TASKS = {
    "libero_videogen",
    "recon_training",
    "fastwam",
}

DEFAULT_DATASET_PATHS = {
    "lerobot": Path("data/lerobot"),
}
DEFAULT_THIRD_PARTY_PATHS = {
    "vjepa2_1": Path("third_party/vjepa2"),
}


def find_project_root(start: str | Path) -> Path:
    """Find the repository root that owns a config, script, or source path."""

    candidate = Path(start).expanduser().resolve()
    if candidate.is_file() or candidate.suffix:
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src" / "vrae").is_dir():
            return directory
    return Path.cwd().resolve()


def load_project_paths(
    config: Mapping[str, Any] | None = None,
    *,
    override: str | Path | Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> ProjectPaths:
    """Load an explicit path override or use the repository-local layout."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    configured: object = override
    if configured is None and config is not None:
        configured = config.get("paths")
    if configured is None or configured == "":
        return ProjectPaths(project_root=root)
    if isinstance(configured, Mapping):
        values = dict(configured)
        values.setdefault("project_root", str(root))
        return ProjectPaths.from_mapping(values)

    path_file = Path(str(configured)).expanduser()
    if not path_file.is_absolute():
        path_file = root / path_file
    if not path_file.is_file():
        raise FileNotFoundError(path_file)
    from vrae.config import load_config

    values = load_config(path_file)
    values.setdefault("project_root", str(root))
    return ProjectPaths.from_mapping(values, config_file=path_file)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``.``/``..`` without dereferencing symlinks."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _artifact_path(
    candidate: Path,
    root: Path,
    *,
    require_exists: bool,
    description: str,
) -> Path:
    lexical_root = _lexical_absolute(root)
    lexical_path = _lexical_absolute(candidate)
    if not _inside(lexical_path, lexical_root):
        raise ConfigError(f"{description} must be lexically inside {lexical_root}: {candidate}")
    if require_exists:
        if not lexical_path.exists():
            raise FileNotFoundError(lexical_path)
        return lexical_path

    resolved_root = lexical_root.resolve()
    resolved_path = lexical_path.resolve(strict=False)
    if not _inside(resolved_path, resolved_root):
        raise ConfigError(
            f"Writable {description.lower()} cannot follow a symlink outside {lexical_root}: "
            f"{candidate}"
        )
    return lexical_path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    datasets: Mapping[str, Path] = field(default_factory=dict)
    third_party: Mapping[str, Path] = field(default_factory=dict)
    checkpoint_write_root: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        config_file: str | Path | None = None,
    ) -> ProjectPaths:
        default_root = Path(config_file).resolve().parents[1] if config_file else Path.cwd()
        root = Path(value.get("project_root", default_root)).expanduser().resolve()

        def configured_path(path: object) -> Path:
            configured = Path(str(path)).expanduser()
            if not configured.is_absolute():
                configured = root / configured
            return configured.resolve()

        datasets = {
            str(key): configured_path(path) for key, path in value.get("datasets", {}).items()
        }
        third_party = {
            str(key): configured_path(path) for key, path in value.get("third_party", {}).items()
        }
        checkpoint_write_root_value = value.get("checkpoint_write_root")
        checkpoint_write_root = (
            configured_path(checkpoint_write_root_value) if checkpoint_write_root_value else None
        )
        return cls(
            project_root=root,
            datasets=datasets,
            third_party=third_party,
            checkpoint_write_root=checkpoint_write_root,
        )

    @property
    def checkpoint_root(self) -> Path:
        return self.project_root / "ckpts"

    @property
    def writable_checkpoint_root(self) -> Path:
        return self.checkpoint_write_root or self.checkpoint_root

    def dataset(self, name: str, *, require_exists: bool = True) -> Path:
        path = self.datasets.get(name)
        if path is None:
            relative = DEFAULT_DATASET_PATHS.get(name)
            if relative is None:
                raise ConfigError(f"Unknown dataset path: {name!r}")
            path = self.project_root / relative
        if require_exists and not path.exists():
            raise FileNotFoundError(path)
        return path

    def source(self, name: str, *, require_exists: bool = True) -> Path:
        path = self.third_party.get(name)
        if path is None:
            relative = DEFAULT_THIRD_PARTY_PATHS.get(name)
            if relative is None:
                raise ConfigError(f"Unknown third-party source: {name!r}")
            path = self.project_root / relative
        if require_exists and not path.exists():
            raise FileNotFoundError(path)
        return path

    def checkpoint(self, relative: str | Path, *, require_exists: bool = True) -> Path:
        raw = Path(relative).expanduser()
        primary_root = _lexical_absolute(self.checkpoint_root)
        write_root = _lexical_absolute(self.writable_checkpoint_root)

        if raw.is_absolute():
            candidate = _lexical_absolute(raw)
            if _inside(candidate, write_root):
                root = write_root
            else:
                root = primary_root
            return _artifact_path(
                candidate,
                root,
                require_exists=require_exists,
                description="Model checkpoint",
            )

        primary_candidate = _lexical_absolute(self.project_root / raw)
        if not _inside(primary_candidate, primary_root):
            raise ConfigError(
                f"Model checkpoint must be lexically inside {primary_root}: {relative}"
            )
        checkpoint_relative = primary_candidate.relative_to(primary_root)
        write_candidate = write_root / checkpoint_relative

        if not require_exists:
            return _artifact_path(
                write_candidate,
                write_root,
                require_exists=False,
                description="Model checkpoint",
            )
        if write_candidate.exists():
            return _artifact_path(
                write_candidate,
                write_root,
                require_exists=True,
                description="Model checkpoint",
            )
        return _artifact_path(
            primary_candidate,
            primary_root,
            require_exists=True,
            description="Model checkpoint",
        )

    def training_run(self, task: str, run_name: str, *, create: bool = False) -> Path:
        if task not in TRAINING_TASKS:
            raise ConfigError(f"Unknown training task: {task}")
        self._validate_run_name(run_name)
        checkpoint_root = self.writable_checkpoint_root
        path = _artifact_path(
            checkpoint_root / task / run_name,
            checkpoint_root,
            require_exists=False,
            description="Training run",
        )
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _validate_run_name(run_name: str) -> None:
        candidate = Path(run_name)
        if not run_name or candidate.name != run_name or run_name in {".", ".."}:
            raise ConfigError(f"run_name must be one path component, got {run_name!r}")
