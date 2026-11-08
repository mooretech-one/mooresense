"""Validate a JSONL split: schema, canary, and resimulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from mooresense import __version__
from mooresense.canary import CANARY, CANARY_GUID
from mooresense.gen.generate import generate_items
from mooresense.paths import find_root
from mooresense.schema import Item, dumps_item, json_schema, load_jsonl


def _mismatch(left: Any, right: Any, path: str = "") -> str | None:
    if type(left) is not type(right):
        return f"{path or '<root>'}: {type(left).__name__} vs {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            here = f"{path}.{key}" if path else str(key)
            if key not in left:
                return f"{here} missing on the file"
            if key not in right:
                return f"{here} missing on the resimulation"
            found = _mismatch(left[key], right[key], here)
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path or '<root>'} length {len(left)} vs {len(right)}"
        for index, (item, other) in enumerate(zip(left, right, strict=True)):
            found = _mismatch(item, other, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        shown_left = repr(left)
        shown_right = repr(right)
        if len(shown_left) > 180:
            shown_left = shown_left[:180] + "..."
        if len(shown_right) > 180:
            shown_right = shown_right[:180] + "..."
        return f"{path or '<root>'}: {shown_left} != {shown_right}"
    return None


def validate_path(path: Path, *, resimulate: bool = True, root: Path | None = None) -> list[str]:
    """Return a list of human-readable errors. Empty means the file is valid."""
    root = root or find_root()
    errors: list[str] = []
    try:
        items = load_jsonl(path)
    except (OSError, ValueError, ValidationError) as exc:
        return [str(exc)]
    if not items:
        errors.append(f"{path} has no items")

    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        errors.append("duplicate item ids")

    schema_path = root / "schema" / "item.schema.json"
    if not schema_path.is_file():
        errors.append(f"missing {schema_path}")
    else:
        document = json.loads(schema_path.read_text(encoding="utf-8"))
        if document != json_schema():
            errors.append("schema/item.schema.json does not match Item.model_json_schema()")
        validator = jsonschema.Draft202012Validator(document)
        for item in items:
            payload = json.loads(dumps_item(item))
            problems = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
            if problems:
                first = problems[0]
                loc = ".".join(str(part) for part in first.path) or "<root>"
                errors.append(f"{item.id}: json schema: {loc}: {first.message}")
                break

    canary_path = root / "CANARY.txt"
    if not canary_path.is_file():
        errors.append("missing CANARY.txt")
    else:
        text = canary_path.read_text(encoding="utf-8")
        if CANARY not in text or CANARY_GUID not in text:
            errors.append("CANARY.txt does not contain the library canary")

    if any(item.canary != CANARY for item in items):
        errors.append("an item canary does not match the library canary")

    meta_path = path.with_suffix(".meta.json")
    meta: dict[str, Any] | None = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_items") != len(items):
            errors.append(
                f"meta n_items is {meta.get('n_items')} but the file has {len(items)} items"
            )
        if meta.get("generator_version") not in (None, __version__):
            errors.append(
                f"meta generator_version {meta.get('generator_version')} != {__version__}"
            )
    elif resimulate:
        errors.append(f"missing {meta_path}; cannot resimulate")

    if resimulate and meta is not None and not errors:
        fresh = generate_items(
            root=root,
            split=str(meta["split"]),
            rng_seed=int(meta["rng_seed"]),
            variants=int(meta["variants"]),
        )
        if [item.id for item in fresh] != [item.id for item in items]:
            errors.append(
                f"resimulation produced {len(fresh)} items, file has {len(items)}; id mismatch"
            )
        else:
            for original, new in zip(items, fresh, strict=True):
                left = original.model_dump(mode="json")
                right = new.model_dump(mode="json")
                found = _mismatch(left, right)
                if found:
                    errors.append(f"{original.id} does not match resimulation: {found}")
                    break
    return errors


def assert_item_roundtrip(item: Item) -> None:
    """Used by tests: pydantic dump validates again."""
    Item.model_validate(item.model_dump(mode="json"))
