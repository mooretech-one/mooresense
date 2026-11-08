"""Container and location simulator.

Operations update an explicit world. The gold answer is read back from that
world, so a templated variant stays correct when names change.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from mooresense.sim.result import Simulation
from mooresense.sim.world import in_phrase, on_phrase, slug, snapshot, trace_step

_LOCATION_ATTRS = ("in", "on", "at", "held_by")


def _clear_location(attrs: dict[str, Any]) -> None:
    for attr in _LOCATION_ATTRS:
        attrs.pop(attr, None)


def _require(state: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    if key not in state:
        raise ValueError(f"unknown entity {key!r}")
    return state[key]


def apply_op(state: dict[str, dict[str, Any]], op: dict[str, Any]) -> None:
    kind = op["op"]
    if kind == "noop":
        return
    if kind == "put_in":
        obj = _require(state, op["object"])
        container = _require(state, op["container"])
        if container.get("closed"):
            raise ValueError(f"cannot put {op['object']} into closed {op['container']}")
        _clear_location(obj)
        obj["in"] = op["container"]
        return
    if kind == "put_on":
        obj = _require(state, op["object"])
        _require(state, op["surface"])
        _clear_location(obj)
        obj["on"] = op["surface"]
        return
    if kind == "move":
        entity = _require(state, op["entity"])
        _require(state, op["place"])
        _clear_location(entity)
        entity["at"] = op["place"]
        return
    if kind == "hold":
        obj = _require(state, op["object"])
        _require(state, op["agent"])
        _clear_location(obj)
        obj["held_by"] = op["agent"]
        return
    if kind == "swap_locations":
        left = _require(state, op["a"])
        right = _require(state, op["b"])
        for attr in ("at", "position"):
            if attr in left or attr in right:
                left[attr], right[attr] = right.get(attr), left.get(attr)
                if left.get(attr) is None:
                    left.pop(attr, None)
                if right.get(attr) is None:
                    right.pop(attr, None)
        return
    if kind == "tip":
        container = _require(state, op["container"])
        _require(state, op["onto"])
        container["upright"] = False
        if container.get("closed"):
            return
        moving = [key for key, attrs in state.items() if attrs.get("in") == op["container"]]
        for key in moving:
            state[key].pop("in", None)
            state[key]["on"] = op["onto"]
        return
    if kind == "close":
        _require(state, op["container"])["closed"] = True
        return
    if kind == "open":
        _require(state, op["container"])["closed"] = False
        return
    raise ValueError(f"unknown state op {kind!r}")


def _any_tipped(state: dict[str, dict[str, Any]]) -> bool:
    return any(attrs.get("upright") is False for attrs in state.values())


def describe_where(state: dict[str, dict[str, Any]], start: str) -> str:
    if start not in state:
        raise ValueError(f"unknown entity {start}")
    if "held_by" in state[start]:
        holder = state[state[start]["held_by"]]["name"]
        return f"held by {holder}"

    parts: list[str] = []
    current = start
    seen: set[str] = set()
    first_container = True
    while current not in seen:
        seen.add(current)
        attrs = state[current]
        if "in" in attrs:
            nxt = attrs["in"]
            name = state[nxt]["name"]
            phrase = in_phrase(name)
            if first_container:
                upright = state[nxt].get("upright")
                if upright is False:
                    phrase += ", which is tipped over"
                elif upright is True and _any_tipped(state):
                    phrase += ", which is upright"
                first_container = False
            parts.append(phrase)
            current = nxt
            continue
        if "on" in attrs:
            nxt = attrs["on"]
            parts.append(on_phrase(state[nxt]["name"]))
            current = nxt
            continue
        if "at" in attrs:
            place = state[attrs["at"]]
            if place.get("place_type") == "room":
                parts.append(f"in the {place['name']}")
            elif place.get("place_type") == "surface" and current == start:
                parts.append(on_phrase(place["name"]))
            break
        if "held_by" in attrs:
            holder = state[attrs["held_by"]]["name"]
            parts.append(f"held by {holder}")
            break
        break
    if not parts:
        raise ValueError(f"no location for {start}")
    return ", ".join(parts)


def _english_list(names: list[str]) -> str:
    labeled = [f"the {name}" for name in names]
    if len(labeled) == 1:
        return labeled[0]
    if len(labeled) == 2:
        return f"{labeled[0]} and {labeled[1]}"
    return ", ".join(labeled[:-1]) + f", and {labeled[-1]}"


def _content_names(state: dict[str, dict[str, Any]], container: str) -> list[str]:
    """Direct contents only. A box inside the bag counts; the key inside that box does not."""
    names = [
        attrs["name"]
        for attrs in state.values()
        if attrs.get("in") == container and attrs.get("kind") in {"object", "container"}
    ]
    return sorted(names, key=str.lower)


def _list_aliases(names: list[str]) -> list[str]:
    if not names:
        return ["nothing", "empty", "none"]
    aliases: set[str] = set()
    for perm in permutations(names):
        pretty = _english_list(list(perm))
        aliases.add(pretty)
        aliases.add(pretty.replace("the ", ""))
        aliases.add(", ".join(perm))
        aliases.add(" and ".join(perm))
    return sorted(aliases)


def _where_choices(state: dict[str, dict[str, Any]], gold: str) -> list[str]:
    options = [gold]
    for attrs in state.values():
        kind = attrs.get("kind")
        if kind == "container":
            phrase = in_phrase(attrs["name"])
            if attrs.get("upright") is False:
                phrase += ", which is tipped over"
            elif attrs.get("upright") is True and _any_tipped(state):
                phrase += ", which is upright"
            options.append(phrase)
        elif kind == "place" and attrs.get("place_type") == "room":
            options.append(f"in the {attrs['name']}")
        elif kind == "place" and attrs.get("place_type") == "surface":
            options.append(on_phrase(attrs["name"]))
        elif kind == "agent":
            options.append(f"held by {attrs['name']}")
    deduped: list[str] = []
    seen: set[str] = set()
    for option in options:
        if option not in seen:
            seen.add(option)
            deduped.append(option)
    if len(deduped) > 4:
        deduped = [gold] + [option for option in deduped if option != gold][:3]
    for filler in ("somewhere else", "nowhere in the scene"):
        if len(deduped) >= 3:
            break
        if filler not in seen:
            deduped.append(filler)
    return deduped


def _export_ops(
    ops: list[dict[str, Any]], state: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    idmap = {key: slug(str(attrs["name"])) for key, attrs in state.items()}
    exported = []
    for op in ops:
        item = {}
        for key, value in op.items():
            if isinstance(value, str) and value in idmap and key != "op":
                item[key] = idmap[value]
            else:
                item[key] = value
        exported.append(item)
    return exported


def simulate_state(instance: dict[str, Any]) -> Simulation:
    state = {key: dict(attrs) for key, attrs in instance["entities"].items()}
    for attrs in state.values():
        attrs.pop("slot", None)
    trace: list[dict[str, Any]] = []
    sentences: list[str] = []
    distractors: list[str] = []
    index = 0

    setup = instance.get("setup") or []
    if not setup:
        trace.append(trace_step(index, "Initial state.", "setup", {}, state))
        index += 1
    else:
        initial = snapshot(state)
        for piece in setup:
            sentences.append(piece["text"])
            trace.append(
                trace_step(
                    index,
                    piece["text"],
                    "setup",
                    {},
                    initial,
                    already_exported=True,
                )
            )
            index += 1

    for step in instance["steps"]:
        ops = step.get("ops") or []
        if step.get("distractor") and any(op["op"] != "noop" for op in ops):
            raise ValueError("distractor steps must be no-ops")
        if not step.get("distractor") and not ops:
            raise ValueError("state steps need ops")
        for op in ops:
            apply_op(state, op)
        text = step["text"]
        sentences.append(text)
        if step.get("distractor"):
            distractors.append(text)
        op_name = "+".join(op["op"] for op in ops) if ops else "noop"
        trace.append(trace_step(index, text, op_name, {"ops": _export_ops(ops, state)}, state))
        index += 1

    query = instance["query"]
    qtype = query["type"]
    if qtype == "where":
        gold = describe_where(state, query["entity"])
        choices: list[str] | None = _where_choices(state, gold)
        accept = [gold]
    elif qtype == "contents":
        names = _content_names(state, query["container"])
        gold = "nothing" if not names else _english_list(names)
        choices = None
        accept = _list_aliases(names)
    else:
        raise ValueError(f"unknown state query {qtype!r}")

    return Simulation(
        context=" ".join(sentences),
        question=instance["question"],
        answer_text=gold,
        accept=accept,
        choice_texts=choices,
        world_state=snapshot(state),
        trace=trace,
        distractors=distractors,
    )
