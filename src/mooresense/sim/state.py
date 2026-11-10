"""Container and location simulator.

Operations update an explicit world. The gold answer is read back from that
world, or from an earlier snapshot on the trace, so a templated variant stays
correct when names change.

Liquids and ice are amounts on a container (``contents_amount``), not extra
entities, so a pour can split one substance. One cup of ice melts into one
cup of water. A solid does not pour, and tipping spills only liquids. A
closed vessel keeps its contents when tipped. Breaking or burning spills
whatever was inside, including a solid and including a closed vessel, because
the vessel no longer holds anything. Those amounts move; they are not deleted.
"""

from __future__ import annotations

import copy
from itertools import permutations
from typing import Any

from mooresense.sim.result import Simulation
from mooresense.sim.world import in_phrase, on_phrase, slug, snapshot, trace_step

_LOCATION_ATTRS = ("in", "on", "at", "held_by")
_BURNABLE = frozenset({"paper", "wood", "cloth", "card", "wax"})
_BREAKABLE = frozenset({"glass", "ceramic", "wood"})


def _clear_location(attrs: dict[str, Any]) -> None:
    for attr in _LOCATION_ATTRS:
        attrs.pop(attr, None)


def _require(state: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    if key not in state:
        raise ValueError(f"unknown entity {key!r}")
    return state[key]


def _amount_map(attrs: dict[str, Any]) -> dict[str, int]:
    raw = attrs.get("contents_amount")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("contents_amount must be a mapping")
    return {str(key): int(value) for key, value in raw.items()}


def _phase_map(attrs: dict[str, Any]) -> dict[str, str]:
    raw = attrs.get("contents_phase")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("contents_phase must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def _store_contents(attrs: dict[str, Any], amounts: dict[str, int], phases: dict[str, str]) -> None:
    kept = {key: int(value) for key, value in amounts.items() if int(value) > 0}
    kept_phases = {key: phases[key] for key in kept if key in phases}
    if kept:
        attrs["contents_amount"] = kept
    else:
        attrs.pop("contents_amount", None)
    if kept_phases:
        attrs["contents_phase"] = kept_phases
    else:
        attrs.pop("contents_phase", None)


def _change_amount(
    attrs: dict[str, Any], substance: str, delta: int, *, phase: str | None = None
) -> None:
    amounts = _amount_map(attrs)
    phases = _phase_map(attrs)
    amounts[substance] = amounts.get(substance, 0) + int(delta)
    if amounts[substance] <= 0:
        amounts.pop(substance, None)
        phases.pop(substance, None)
    elif phase is not None:
        phases[substance] = phase
    _store_contents(attrs, amounts, phases)


def _transfer_amounts(
    state: dict[str, dict[str, Any]], source: str, dest: str, *, liquids_only: bool
) -> None:
    src = state[source]
    amounts = _amount_map(src)
    phases = _phase_map(src)
    moving = [
        (name, amount, phases.get(name, "liquid"))
        for name, amount in amounts.items()
        if not liquids_only or phases.get(name, "liquid") != "solid"
    ]
    for name, amount, phase in moving:
        _change_amount(src, name, -amount)
        _change_amount(state[dest], name, amount, phase=phase)


def _reject_destroyed(attrs: dict[str, Any], what: str) -> None:
    if attrs.get("broken") or attrs.get("burned"):
        raise ValueError(f"cannot {what} a broken or burned vessel")


def _spill_objects(state: dict[str, dict[str, Any]], container: str, onto: str) -> None:
    moving = [key for key, attrs in state.items() if attrs.get("in") == container]
    for key in moving:
        state[key].pop("in", None)
        state[key]["on"] = onto


def _destroy(state: dict[str, dict[str, Any]], op: dict[str, Any], *, mode: str) -> None:
    key = op["object"]
    obj = _require(state, key)
    if obj.get("broken") or obj.get("burned"):
        raise ValueError(f"{key} is already destroyed")
    material = str(obj.get("material", ""))
    allowed = _BREAKABLE if mode == "break" else _BURNABLE
    if material not in allowed:
        raise ValueError(f"no {mode} rule for {material or key}")
    onto = op.get("onto")
    if not onto:
        raise ValueError(f"{mode} needs onto")
    _require(state, onto)
    _spill_objects(state, key, onto)
    _transfer_amounts(state, key, onto, liquids_only=False)
    obj["upright"] = False
    _clear_location(obj)
    obj["on"] = onto
    if mode == "break":
        obj["broken"] = True
    else:
        obj["burned"] = True
        obj["form"] = "ash"


def apply_op(state: dict[str, dict[str, Any]], op: dict[str, Any]) -> None:
    kind = op["op"]
    if kind == "noop":
        return
    if kind == "put_in":
        obj = _require(state, op["object"])
        container = _require(state, op["container"])
        if container.get("closed"):
            raise ValueError(f"cannot put {op['object']} into closed {op['container']}")
        _reject_destroyed(container, "put anything into")
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
        _spill_objects(state, op["container"], op["onto"])
        _transfer_amounts(state, op["container"], op["onto"], liquids_only=True)
        return
    if kind == "close":
        _require(state, op["container"])["closed"] = True
        return
    if kind == "open":
        _require(state, op["container"])["closed"] = False
        return
    if kind == "fill":
        container = _require(state, op["container"])
        _reject_destroyed(container, "fill")
        amount = int(op["amount"])
        if amount < 1:
            raise ValueError("fill amount must be positive")
        phase = str(op.get("phase", "liquid"))
        if phase not in {"liquid", "solid"}:
            raise ValueError(f"unknown phase {phase}")
        _change_amount(container, str(op["substance"]), amount, phase=phase)
        return
    if kind == "pour":
        src = _require(state, op["source"])
        dst = _require(state, op["dest"])
        if op["source"] == op["dest"]:
            raise ValueError("cannot pour into the same vessel")
        substance = str(op["substance"])
        amount = int(op["amount"])
        if amount < 1:
            raise ValueError("pour amount must be positive")
        if src.get("closed") or dst.get("closed"):
            raise ValueError("cannot pour through a closed vessel")
        _reject_destroyed(src, "pour from")
        _reject_destroyed(dst, "pour into")
        if _phase_map(src).get(substance, "liquid") == "solid":
            raise ValueError(f"solid {substance} does not pour")
        have = _amount_map(src).get(substance, 0)
        if amount > have:
            raise ValueError(f"cannot pour {amount} {substance}; only {have} is there")
        _change_amount(src, substance, -amount)
        _change_amount(dst, substance, amount, phase="liquid")
        return
    if kind == "mix":
        container = _require(state, op["container"])
        _reject_destroyed(container, "mix in")
        parts = [str(part) for part in op["parts"]]
        if len(parts) < 2 or len(set(parts)) != len(parts):
            raise ValueError("mix needs two different substances")
        product = str(op["product"])
        if product in parts:
            raise ValueError("mix product must be a new substance")
        total = 0
        for part in parts:
            if _phase_map(container).get(part, "liquid") == "solid":
                raise ValueError(f"cannot mix solid {part}")
            have = _amount_map(container).get(part, 0)
            if have <= 0:
                raise ValueError(f"no {part} to mix")
            total += have
        for part in parts:
            _change_amount(container, part, -_amount_map(container)[part])
        _change_amount(container, product, total, phase="liquid")
        return
    if kind == "freeze":
        container = _require(state, op["container"])
        substance = str(op.get("substance", "water"))
        if substance != "water":
            raise ValueError(f"no freeze rule for {substance}")
        if _amount_map(container).get(substance, 0) <= 0:
            raise ValueError("nothing to freeze")
        _change_amount(container, substance, 0, phase="solid")
        return
    if kind == "melt":
        _melt(state, op)
        return
    if kind in {"break", "burn"}:
        _destroy(state, op, mode=kind)
        return
    raise ValueError(f"unknown state op {kind!r}")


def _melt(state: dict[str, dict[str, Any]], op: dict[str, Any]) -> None:
    container = _require(state, op["container"])
    substance = str(op["substance"])
    have = _amount_map(container).get(substance, 0)
    if have <= 0:
        raise ValueError(f"nothing to melt in {op['container']}")
    phase = _phase_map(container).get(substance, "liquid")
    if substance == "water":
        if phase != "solid":
            raise ValueError("water melts only from a solid")
        _change_amount(container, "water", 0, phase="liquid")
        return
    if substance != "ice" and phase != "solid":
        raise ValueError(f"no melt rule for {substance}")
    becomes = str(op.get("becomes", "water" if substance == "ice" else substance))
    _change_amount(container, substance, -have)
    _change_amount(container, becomes, have, phase="liquid")


def _any_tipped(state: dict[str, dict[str, Any]]) -> bool:
    return any(attrs.get("upright") is False for attrs in state.values())


def describe_where(state: dict[str, dict[str, Any]], start: str) -> str:
    if start not in state:
        raise ValueError(f"unknown entity {start}")
    if "held_by" in state[start]:
        holder = state[state[start]["held_by"]]["name"]
        phrase = f"held by {holder}"
        if state[start].get("burned"):
            return f"burned, {phrase}"
        return phrase

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
    phrase = ", ".join(parts)
    if state[start].get("burned"):
        return f"burned, {phrase}"
    return phrase


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


def _cap_options(gold: str, options: list[str], *, limit: int = 4) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for option in [gold, *options]:
        if option and option not in seen:
            seen.add(option)
            deduped.append(option)
    if len(deduped) > limit:
        deduped = deduped[:limit]
    for filler in ("somewhere else", "nowhere in the scene"):
        if len(deduped) >= 3:
            break
        if filler not in seen:
            deduped.append(filler)
            seen.add(filler)
    return deduped


def _head_phrase(state: dict[str, dict[str, Any]], holder: str) -> str:
    attrs = state[holder]
    name = str(attrs["name"])
    if attrs.get("kind") == "place" and attrs.get("place_type") == "surface":
        return on_phrase(name)
    if attrs.get("kind") == "place" and attrs.get("place_type") == "room":
        return f"in the {name}"
    return in_phrase(name)


def _outer_phrase(state: dict[str, dict[str, Any]], holder: str) -> str:
    if not any(key in state[holder] for key in _LOCATION_ATTRS):
        return ""
    try:
        return describe_where(state, holder)
    except ValueError:
        return ""


def _place_phrase(state: dict[str, dict[str, Any]], holder: str) -> str:
    head = _head_phrase(state, holder)
    outer = _outer_phrase(state, holder)
    if outer:
        return f"{head}, {outer}"
    return head


def _holders(state: dict[str, dict[str, Any]], substance: str) -> list[tuple[str, int]]:
    found = [
        (key, amount)
        for key, attrs in state.items()
        if (amount := _amount_map(attrs).get(substance, 0))
    ]
    found.sort(key=lambda item: str(state[item[0]]["name"]).lower())
    return found


def describe_liquid(state: dict[str, dict[str, Any]], substance: str) -> str:
    holders = _holders(state, substance)
    if not holders:
        return "nowhere"
    if len(holders) == 1:
        return _place_phrase(state, holders[0][0])
    bits = []
    for key, amount in holders:
        unit = "cup" if amount == 1 else "cups"
        bits.append(f"{amount} {unit} {_head_phrase(state, key)}")
    if len(bits) == 2:
        return f"{bits[0]} and {bits[1]}"
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"


def _seen_amounts(history: list[dict[str, dict[str, Any]]]) -> set[int]:
    seen = {0}
    for snapshot_state in history:
        substances: set[str] = set()
        for attrs in snapshot_state.values():
            amounts = _amount_map(attrs)
            seen.update(amounts.values())
            substances.update(amounts)
        for substance in substances:
            seen.add(sum(_amount_map(attrs).get(substance, 0) for attrs in snapshot_state.values()))
    return seen


def _number_choices(gold_n: int, seen: set[int]) -> list[str]:
    pool = set(seen)
    pool.add(0)
    pool.add(gold_n + 1)
    rest = [number for number in pool if number != gold_n]
    rest.sort(key=lambda number: (0 if number > gold_n else 1, abs(number - gold_n), number))
    return _cap_options(str(gold_n), [str(number) for number in rest], limit=5)


def _contents_choices(
    state: dict[str, dict[str, Any]], container: str, names: list[str], gold: str
) -> list[str]:
    options = [gold]
    if names:
        options.append(_english_list(names[:1]))
        if len(names) >= 2:
            options.append(_english_list(names[:-1]))
    direct_boxes = [
        key
        for key, attrs in state.items()
        if attrs.get("in") == container and attrs.get("kind") == "container"
    ]
    nested = sorted(
        (
            attrs["name"]
            for attrs in state.values()
            if attrs.get("in") in direct_boxes and attrs.get("kind") in {"object", "container"}
        ),
        key=str.lower,
    )
    if nested:
        mixed = sorted(set(names).union(nested), key=str.lower)
        options.append(_english_list(mixed))
    for key, attrs in state.items():
        if key == container or attrs.get("kind") != "container":
            continue
        other = _content_names(state, key)
        if other and other != names:
            options.append(_english_list(other))
            break
    options.append("nothing")
    return _cap_options(gold, options)


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
    history: list[dict[str, dict[str, Any]]] = []
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
        history.append(copy.deepcopy(state))
        index += 1

    query = instance["query"]
    qtype = query["type"]
    answer_state = state
    if "after_step" in query:
        step_n = int(query["after_step"])
        if step_n < 1 or step_n > len(history):
            raise ValueError(f"after_step {step_n} is outside 1..{len(history)}")
        answer_state = history[step_n - 1]
    seen = _seen_amounts(history)
    if qtype == "where":
        gold = describe_where(answer_state, query["entity"])
        choices: list[str] | None = _where_choices(answer_state, gold)
        if answer_state is not state:
            final = describe_where(state, query["entity"])
            choices = _cap_options(gold, [*choices, final])
        accept = [gold]
    elif qtype == "contents":
        names = _content_names(answer_state, query["container"])
        gold = "nothing" if not names else _english_list(names)
        choices = _contents_choices(answer_state, str(query["container"]), names, gold)
        if answer_state is not state:
            final_names = _content_names(state, query["container"])
            final_gold = "nothing" if not final_names else _english_list(final_names)
            choices = _cap_options(gold, [final_gold, *choices])
        accept = _list_aliases(names)
    elif qtype == "where_liquid":
        substance = str(query["substance"])
        gold = describe_liquid(answer_state, substance)
        options = [gold, "nowhere"]
        for key, _amount in _holders(answer_state, substance):
            options.append(_place_phrase(answer_state, key))
        for key, attrs in answer_state.items():
            if attrs.get("kind") in {"container", "place"}:
                options.append(_place_phrase(answer_state, key))
        if answer_state is not state:
            options.append(describe_liquid(state, substance))
        choices = _cap_options(gold, options)
        accept = [gold]
    elif qtype in {"amount", "amount_total"}:
        substance = str(query["substance"])
        if qtype == "amount":
            gold_n = _amount_map(answer_state[query["container"]]).get(substance, 0)
        else:
            gold_n = sum(amount for _key, amount in _holders(answer_state, substance))
        gold = str(gold_n)
        choices = _number_choices(gold_n, seen)
        accept = [gold, f"{gold} cups"]
        if gold == "1":
            accept.append("1 cup")
    elif qtype == "phase":
        substance = str(query["substance"])
        attrs = answer_state[query["container"]]
        if _amount_map(attrs).get(substance, 0) <= 0:
            gold = "gone"
        else:
            gold = _phase_map(attrs).get(substance, "liquid")
        choices = _cap_options(gold, ["liquid", "solid", "gone"], limit=3)
        accept = [gold]
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
