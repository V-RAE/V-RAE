from __future__ import annotations

import os
from pathlib import Path


def _checkpoint_root(checkpoint_root: str | Path) -> Path:
    return Path(os.path.abspath(Path(checkpoint_root).expanduser()))


def local_checkpoint_entry(
    checkpoint: str | Path,
    *,
    checkpoint_root: str | Path,
) -> Path:
    """Validate the lexical entry while allowing its read-only symlink target to be external."""

    entry = Path(os.path.abspath(Path(checkpoint).expanduser()))
    expected_root = _checkpoint_root(checkpoint_root)
    if expected_root not in entry.parents:
        raise ValueError(f"evaluation checkpoint must be inside {expected_root}")
    if not entry.is_file():
        raise FileNotFoundError(entry)
    return entry


def local_checkpoint_directory(
    checkpoint: str | Path,
    *,
    checkpoint_root: str | Path,
) -> Path:
    """Require a local evaluation-weight directory under ``ckpts/eval_models``."""

    entry = Path(os.path.abspath(Path(checkpoint).expanduser()))
    expected_root = _checkpoint_root(checkpoint_root)
    if expected_root not in entry.parents:
        raise ValueError(f"evaluation checkpoint must be inside {expected_root}")
    if not entry.is_dir():
        raise FileNotFoundError(entry)
    return entry


__all__ = ["local_checkpoint_directory", "local_checkpoint_entry"]
