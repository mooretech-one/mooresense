"""Pours, melts, breaks, and burns. Gold text is read from the world."""

from mooresense.sim.state import describe_where, simulate_state


def _run(steps, query, entities):
    return simulate_state(
        {"entities": entities, "steps": steps, "query": query, "question": "What follows?"}
    )


def _step(op):
    return {"text": "An event.", "ops": [op]}


def test_pour_conserves_and_intermediate_amount_differs_from_the_end():
    entities = {
        "jug": {"kind": "container", "name": "jug", "closed": False, "upright": True},
        "bowl": {"kind": "container", "name": "bowl", "closed": False, "upright": True},
        "mug": {"kind": "container", "name": "mug", "closed": False, "upright": True},
    }
    steps = [
        _step({"op": "fill", "container": "jug", "substance": "water", "amount": 8}),
        _step({"op": "pour", "source": "jug", "dest": "bowl", "substance": "water", "amount": 3}),
        _step({"op": "pour", "source": "jug", "dest": "mug", "substance": "water", "amount": 2}),
    ]
    final = _run(steps, {"type": "amount", "container": "jug", "substance": "water"}, entities)
    assert final.answer_text == "3"
    assert final.world_state["jug"]["contents_amount"]["water"] == 3
    assert final.world_state["bowl"]["contents_amount"]["water"] == 3
    assert final.world_state["mug"]["contents_amount"]["water"] == 2
    mid = _run(
        steps,
        {"type": "amount", "container": "jug", "substance": "water", "after_step": 2},
        entities,
    )
    assert mid.answer_text == "5"
    assert mid.world_state["jug"]["contents_amount"]["water"] == 3
    assert mid.trace[2]["state_after"]["jug"]["contents_amount"]["water"] == 5
    total = _run(steps, {"type": "amount_total", "substance": "water"}, entities)
    assert total.answer_text == "8"


def test_liquid_follows_its_container_and_mixing_adds():
    entities = {
        "jug": {"kind": "container", "name": "jug", "closed": False, "upright": True},
        "bag": {"kind": "container", "name": "bag", "closed": False, "upright": True},
        "kitchen": {"kind": "place", "name": "kitchen", "place_type": "room"},
    }
    carried = _run(
        [
            _step({"op": "fill", "container": "jug", "substance": "water", "amount": 4}),
            _step({"op": "put_in", "object": "jug", "container": "bag"}),
            _step({"op": "move", "entity": "bag", "place": "kitchen"}),
        ],
        {"type": "where_liquid", "substance": "water"},
        entities,
    )
    assert carried.answer_text == "in the jug, in the bag, in the kitchen"
    mixed = _run(
        [
            _step({"op": "fill", "container": "jug", "substance": "milk", "amount": 2}),
            _step({"op": "fill", "container": "jug", "substance": "juice", "amount": 3}),
            _step(
                {"op": "mix", "container": "jug", "parts": ["milk", "juice"], "product": "mixture"}
            ),
        ],
        {"type": "amount", "container": "jug", "substance": "mixture"},
        entities,
    )
    assert mixed.answer_text == "5"
    assert mixed.world_state["jug"]["contents_amount"] == {"mixture": 5}
    assert "2" in (mixed.choice_texts or [])
    assert "3" in (mixed.choice_texts or [])


def test_freeze_stays_when_tipped_and_melted_ice_spills():
    entities = {
        "cup": {"kind": "container", "name": "cup", "closed": False, "upright": True},
        "counter": {"kind": "place", "name": "counter", "place_type": "surface"},
    }
    frozen = _run(
        [
            _step({"op": "fill", "container": "cup", "substance": "water", "amount": 2}),
            _step({"op": "freeze", "container": "cup", "substance": "water"}),
            _step({"op": "tip", "container": "cup", "onto": "counter"}),
        ],
        {"type": "where_liquid", "substance": "water"},
        entities,
    )
    assert frozen.answer_text == "in the cup"
    assert frozen.world_state["cup"]["contents_phase"]["water"] == "solid"
    assert "contents_amount" not in frozen.world_state["counter"]
    melted = _run(
        [
            _step(
                {
                    "op": "fill",
                    "container": "cup",
                    "substance": "ice",
                    "amount": 1,
                    "phase": "solid",
                }
            ),
            _step({"op": "melt", "container": "cup", "substance": "ice", "becomes": "water"}),
            _step({"op": "tip", "container": "cup", "onto": "counter"}),
        ],
        {"type": "where_liquid", "substance": "water"},
        entities,
    )
    assert melted.answer_text == "on the counter"
    assert melted.world_state["counter"]["contents_amount"]["water"] == 1


def test_closed_tip_holds_but_break_and_burn_spill():
    entities = {
        "jar": {
            "kind": "container",
            "name": "glass jar",
            "material": "glass",
            "closed": False,
            "upright": True,
        },
        "marble": {"kind": "object", "name": "marble"},
        "table": {"kind": "place", "name": "table", "place_type": "surface"},
        "envelope": {
            "kind": "container",
            "name": "paper envelope",
            "material": "paper",
            "closed": False,
            "upright": True,
        },
        "key": {"kind": "object", "name": "key"},
        "bowl": {"kind": "container", "name": "bowl", "closed": False, "upright": True},
        "kitchen": {"kind": "place", "name": "kitchen", "place_type": "room"},
    }
    tipped = _run(
        [
            _step({"op": "put_in", "object": "marble", "container": "jar"}),
            _step({"op": "close", "container": "jar"}),
            _step({"op": "tip", "container": "jar", "onto": "table"}),
        ],
        {"type": "where", "entity": "marble"},
        entities,
    )
    assert tipped.answer_text == "in the glass jar, which is tipped over"
    assert tipped.world_state["marble"]["in"] == "glass_jar"
    broken = _run(
        [
            _step({"op": "put_in", "object": "marble", "container": "jar"}),
            _step({"op": "close", "container": "jar"}),
            _step({"op": "break", "object": "jar", "onto": "table"}),
            _step({"op": "put_in", "object": "marble", "container": "bowl"}),
            _step({"op": "move", "entity": "bowl", "place": "kitchen"}),
        ],
        {"type": "where", "entity": "marble"},
        entities,
    )
    assert broken.answer_text == "in the bowl, which is upright, in the kitchen"
    assert broken.world_state["glass_jar"]["broken"] is True
    spilled = next(step for step in broken.trace if step["op"] == "break")
    assert spilled["state_after"]["marble"]["on"] == "table"
    burned = _run(
        [
            _step({"op": "put_in", "object": "key", "container": "envelope"}),
            _step({"op": "burn", "object": "envelope", "onto": "table"}),
        ],
        {"type": "where", "entity": "key"},
        entities,
    )
    assert burned.answer_text == "on the table"
    assert burned.world_state["paper_envelope"]["form"] == "ash"
    assert describe_where(
        {
            "paper_envelope": burned.world_state["paper_envelope"],
            "table": burned.world_state["table"],
        },
        "paper_envelope",
    ).startswith("burned, ")
