"""Simulator output, kept separate to avoid import cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Simulation:
    context: str
    question: str
    answer_text: str
    accept: list[str] = field(default_factory=list)
    choice_texts: list[str] | None = None
    world_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
