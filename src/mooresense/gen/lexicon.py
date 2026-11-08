"""Load the shared slot lexicons."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_lexicon(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} should be a mapping of group name to words")
    lexicon: dict[str, list[str]] = {}
    for name, values in data.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"lexicon group {name} is empty")
        words = [str(value) for value in values]
        if len(set(words)) != len(words):
            raise ValueError(f"lexicon group {name} has duplicates")
        lexicon[str(name)] = words
    return lexicon
