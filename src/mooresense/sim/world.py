"""Shared world-state helpers.

Entity ids in the public item are slugs of the surface name (``red box``
becomes ``red_box``). While a simulator is running it may use the seed's
local keys; ``export_state`` rewrites relations to those public ids.
"""

from __future__ import annotations

import copy
import re
from typing import Any

RELATION_KEYS = ("in", "on", "at", "held_by")

_SLUG = re.compile(r"[^a-z0-9]+")
_CUP_LABEL = re.compile(r"cup [A-Z]")


def slug(name: str) -> str:
    text = _SLUG.sub("_", name.strip().lower()).strip("_")
    if not text or not text[0].isalpha():
        raise ValueError(f"cannot slug {name!r}")
    return text


def needs_the(name: str) -> bool:
    """Labeled cups ('cup A') are used without an article."""
    return _CUP_LABEL.fullmatch(name) is None


def noun_with_article(name: str, article: str) -> str:
    if article == "the":
        if needs_the(name):
            return f"the {name}"
        return name
    if article == "a":
        word = "an" if name[:1].lower() in "aeiou" else "a"
        return f"{word} {name}"
    raise ValueError(article)


def in_phrase(name: str) -> str:
    if needs_the(name):
        return f"in the {name}"
    return f"in {name}"


def on_phrase(name: str) -> str:
    if needs_the(name):
        return f"on the {name}"
    return f"on {name}"


def export_state(state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rewrite local entity keys to public slugs."""
    idmap: dict[str, str] = {}
    owner: dict[str, str] = {}
    for key, attrs in state.items():
        sid = slug(str(attrs["name"]))
        if sid in owner:
            raise ValueError(f"slug collision between {owner[sid]!r} and {key!r}: {sid}")
        owner[sid] = key
        idmap[key] = sid
    exported: dict[str, dict[str, Any]] = {}
    for key, attrs in state.items():
        out: dict[str, Any] = {}
        for attr, value in attrs.items():
            if attr in {"slot"}:
                continue
            if attr in RELATION_KEYS:
                if value not in idmap:
                    raise ValueError(f"{key}.{attr} references unknown entity {value!r}")
                out[attr] = idmap[value]
            else:
                out[attr] = value
        if "kind" not in out:
            raise ValueError(f"{key} missing kind")
        exported[idmap[key]] = out
    return copy.deepcopy(exported)


def snapshot(state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return export_state(state)


def trace_step(
    index: int,
    event: str,
    op: str,
    args: dict[str, Any],
    state: dict[str, dict[str, Any]],
    *,
    already_exported: bool = False,
) -> dict[str, Any]:
    state_after = state if already_exported else snapshot(state)
    return {
        "index": index,
        "event": event,
        "op": op,
        "args": copy.deepcopy(args),
        "state_after": copy.deepcopy(state_after),
    }
