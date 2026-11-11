"""Learner adapter for the external teach-then-query arm.

``reset`` starts a session, ``teach`` stores assertions, and ``answer``
returns a string or a ``ChatResult`` the current scorer can parse. PR6 ships
``LookupLearner``, a test double that stores assertions and reads one fact
back. It is not an atomic-thought agent: it does not walk inheritance, join
a chain, or fade unused facts.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Protocol

from mooresense.schema import Assertion, Item
from mooresense.sim.learn import LOCATION_TYPES, relation_phrase

_EARLIER = ("earlier", "before", "start", "previous")
_COLOR = ("colour", "color")


class Learner(Protocol):
    def reset(self) -> None:
        """Drop every fact stored for the session that just finished."""

    def teach(self, assertions: list[Assertion]) -> None:
        """Store this episode's assertions. They are available immediately."""

    def answer(self, item: Item) -> str | Any:
        """Answer from stored facts. A string or a ChatResult; never the gold label."""


class LookupLearner:
    """Store assertions and answer from one exact fact.

    A where-question uses the latest location of the named source. A question
    that says the fact was earlier uses the location before that. A colour
    question uses a colour stored on that source and does not inherit one.
    """

    def __init__(self) -> None:
        self.assertions: list[Assertion] = []

    def reset(self) -> None:
        self.assertions = []

    def teach(self, assertions: list[Assertion]) -> None:
        self.assertions.extend(assertions)

    def answer(self, item: Item) -> str:
        fact = self._lookup(item.question)
        if fact is None:
            return "I don't know"
        phrase = _phrase(fact)
        if item.choices:
            for choice in item.choices:
                if choice.text.casefold() == phrase.casefold():
                    return choice.label
        return phrase

    def _lookup(self, question: str) -> Assertion | None:
        named = [fact for fact in self.assertions if _mentioned(_surface(fact.source), question)]
        if not named:
            return None
        earlier = any(cue in question.casefold() for cue in _EARLIER)
        if any(cue in question.casefold() for cue in _COLOR):
            colours = [fact for fact in named if fact.type == "color"]
            return colours[-1] if colours else None
        locations = [fact for fact in named if fact.type in LOCATION_TYPES]
        if earlier:
            return locations[-2] if len(locations) >= 2 else None
        return locations[-1] if locations else None


def load_learner(spec: str) -> Learner:
    """Import ``module:Class`` and construct it with no arguments."""
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("--learner must look like module:Class")
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{spec} was not found") from exc
    learner = cls()
    for method in ("reset", "teach", "answer"):
        if not callable(getattr(learner, method, None)):
            raise ValueError(f"{spec} is missing {method}()")
    return learner


def _surface(slug: str) -> str:
    return slug.replace("_", " ")


def _mentioned(name: str, question: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE) is not None


def _phrase(fact: Assertion) -> str:
    if fact.type == "color":
        return fact.target
    return relation_phrase(fact.type, _surface(fact.target))
