"""Fill `{slot}`, `{a_slot}`, and `{the_slot}` placeholders."""

from __future__ import annotations

import re
from typing import Any

from mooresense.sim.world import noun_with_article

_A = re.compile(r"\{a_([A-Za-z0-9_]+)\}")
_THE = re.compile(r"\{the_([A-Za-z0-9_]+)\}")
_SLOT = re.compile(r"\{([A-Za-z0-9_]+)\}")
_WHOLE = re.compile(r"^\{([A-Za-z0-9_]+)\}$")


def capitalise(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def render_text(template: str, bindings: dict[str, Any]) -> str:
    def article(match: re.Match[str]) -> str:
        return noun_with_article(str(bindings[match.group(1)]), "a")

    def the(match: re.Match[str]) -> str:
        return noun_with_article(str(bindings[match.group(1)]), "the")

    def slot(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in bindings:
            raise KeyError(key)
        return str(bindings[key])

    try:
        text = _A.sub(article, template)
        text = _THE.sub(the, text)
        text = _SLOT.sub(slot, text)
    except KeyError as exc:
        raise ValueError(f"missing slot {exc} in {template!r}") from exc
    if "{" in text or "}" in text:
        raise ValueError(f"unresolved placeholder in {text!r}")
    return text


def fill_value(value: Any, bindings: dict[str, Any]) -> Any:
    """Replace placeholders. A string that is only `{slot}` keeps the binding's type."""
    if isinstance(value, str):
        whole = _WHOLE.fullmatch(value)
        if whole:
            key = whole.group(1)
            if key not in bindings:
                raise ValueError(f"missing slot {key!r}")
            return bindings[key]
        if "{" in value:
            return render_text(value, bindings)
        return value
    if isinstance(value, list):
        return [fill_value(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: fill_value(item, bindings) for key, item in value.items()}
    return value
