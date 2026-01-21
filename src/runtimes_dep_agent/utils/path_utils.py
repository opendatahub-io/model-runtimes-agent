"""Shared path utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def detect_repo_root(start_paths: Iterable[str | Path] | None = None) -> Path:
    """
    Return the repository root by searching for the nearest folder that
    contains `pyproject.toml`. Falls back to the current working directory.
    """

    candidates: list[Path] = []

    if start_paths:
        for raw in start_paths:
            if not raw:
                continue
            path = Path(raw).resolve()
            if path not in candidates:
                candidates.append(path)
            for parent in path.parents:
                if parent not in candidates:
                    candidates.append(parent)

    cwd = Path.cwd().resolve()
    if cwd not in candidates:
        candidates.append(cwd)
    for parent in cwd.parents:
        if parent not in candidates:
            candidates.append(parent)

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate

    return cwd


__all__ = ["detect_repo_root"]
