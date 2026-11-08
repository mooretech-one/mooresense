"""Locate a MooreSense checkout.

The supported install is an editable install from a clone. Seeds and the
public dev split live at the repo root, not inside the wheel.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Return the checkout that contains ``pyproject.toml`` and ``seeds/``."""
    env = os.environ.get("MOORESENSE_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / "seeds").is_dir():
            raise FileNotFoundError(f"MOORESENSE_ROOT={root} has no seeds/ directory")
        return root

    starts: list[Path] = []
    if start is not None:
        starts.append(start)
    starts.append(Path.cwd())
    starts.append(Path(__file__).resolve().parents[2])
    seen: set[Path] = set()
    for begin in starts:
        for candidate in [begin, *begin.parents]:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (candidate / "seeds").is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not find the MooreSense checkout (expected pyproject.toml and seeds/). "
        "Run from a clone or set MOORESENSE_ROOT."
    )


def default_dev_path(root: Path | None = None) -> Path:
    return (root or find_root()) / "data" / "dev.jsonl"


def default_split_path(split: str, root: Path | None = None) -> Path:
    root = root or find_root()
    if split == "dev":
        return root / "data" / "dev.jsonl"
    if split == "fresh":
        return root / "data" / "fresh" / "fresh.jsonl"
    if split == "test":
        return root / "data" / "private" / "test.jsonl"
    raise ValueError(f"unknown split {split!r}")
