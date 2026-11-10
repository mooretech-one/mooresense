"""CLI smoke tests, including a mocked OpenAI-compatible server."""

import json
from pathlib import Path

import httpx
import pytest
from tasks.lm_eval.adapter import to_lm_eval_doc

from mooresense.cli import main
from mooresense.eval.loglikelihood import score_loglikelihood
from mooresense.eval.runner import ChatResult, chat_complete, parse_chat_response
from mooresense.paths import find_root
from mooresense.schema import load_jsonl


def test_generate_and_validate_roundtrip(tmp_path: Path):
    out = tmp_path / "slice.jsonl"
    assert main(["generate", "--variants", "1", "--seed", "0", "--out", str(out)]) == 0
    assert main(["validate", str(out)]) == 0


def test_random_baseline_is_deterministic(tmp_path: Path):
    items = tmp_path / "items.jsonl"
    assert main(["generate", "--variants", "1", "--category", "time", "--out", str(items)]) == 0
    first = tmp_path / "random.jsonl"
    second = tmp_path / "random-again.jsonl"
    args = ["eval", "--baseline", "random", "--seed", "0", "--items", str(items), "--out"]
    assert main([*args, str(first)]) == 0
    assert main([*args, str(second)]) == 0
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    summary = json.loads(first.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert "by_category_tier" in summary
    chance = tmp_path / "chance.jsonl"
    assert main(["eval", "--baseline", "chance", "--items", str(items), "--out", str(chance)]) == 0
    chance_summary = json.loads(chance.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert chance_summary["chance"] == pytest.approx(0.0)


def test_eval_requires_a_model(tmp_path: Path):
    items = find_root() / "data" / "dev.jsonl"
    with pytest.raises(SystemExit):
        main(["eval", "--items", str(items), "--out", str(tmp_path / "out.jsonl")])


def test_test_split_stays_in_data_private(tmp_path: Path):
    with pytest.raises(SystemExit, match="data/private"):
        main(["generate", "--split", "test", "--variants", "1", "--out", str(tmp_path / "t.jsonl")])

    private = find_root() / "data" / "private" / "test.jsonl"
    try:
        assert main(["generate", "--split", "test", "--variants", "1", "--out", str(private)]) == 0
        assert private.is_file()
        assert all(item.split.value == "test" for item in load_jsonl(private))
    finally:
        private.unlink(missing_ok=True)
        private.with_suffix(".meta.json").unlink(missing_ok=True)


def test_chat_completion_posts_temperature_zero():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "demo", "choices": [{"message": {"content": "B"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = chat_complete(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        prompt="Answer with B",
        api_key=None,
        timeout=5,
        seed=0,
        max_tokens=16,
        client=client,
    )
    assert result.content == "B"
    assert result.model == "demo"
    assert result.reasoning == ""
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0
    assert body["model"] == "qwen3:8b"
    assert body["think"] is False
    assert body["reasoning_effort"] == "none"
    assert "last line" in str(body["messages"][0]["content"])
    user = body["messages"][1]["content"]
    assert isinstance(user, str)
    assert user.startswith("/no_think\n")


def test_chat_completion_reads_reasoning_fields_and_can_leave_thinking_on():
    parsed = parse_chat_response(
        {
            "model": "qwen3:1.7b",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "The plant survives.\n\n**Answer:** A",
                    }
                }
            ],
        }
    )
    assert parsed.content == ""
    assert parsed.reasoning.startswith("The plant survives.")
    assert parsed.model == "qwen3:1.7b"

    native = parse_chat_response(
        {"choices": [{"message": {"content": "A", "thinking": "trace", "reasoning": ""}}]}
    )
    assert native.content == "A"
    # An empty reasoning field is skipped in favour of thinking.
    assert native.reasoning == "trace"

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "qwen3:1.7b", "choices": [{"message": {"content": "A"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chat_complete(
        base_url="http://localhost:11434/v1",
        model="qwen3:1.7b",
        prompt="Question",
        api_key=None,
        timeout=5,
        seed=0,
        max_tokens=32,
        think="default",
        client=client,
    )
    body = seen["body"]
    assert isinstance(body, dict)
    assert "think" not in body
    assert "reasoning_effort" not in body
    assert not str(body["messages"][1]["content"]).startswith("/no_think")

    chat_complete(
        base_url="http://localhost:11434/v1",
        model="qwen3:1.7b",
        prompt="Question",
        api_key=None,
        timeout=5,
        seed=0,
        max_tokens=32,
        think="on",
        client=client,
    )
    thinking = seen["body"]
    assert isinstance(thinking, dict)
    assert thinking["think"] is True
    assert "reasoning_effort" not in thinking
    assert str(thinking["messages"][1]["content"]).startswith("/think\n")


def test_eval_records_reasoning_parse_failures_and_default_budget(tmp_path: Path, monkeypatch):
    calls: dict[str, object] = {}

    def fake_chat_complete(**kwargs):
        calls.update(kwargs)
        return ChatResult(content="", reasoning="still thinking about B", model="qwen3:1.7b")

    monkeypatch.setattr("mooresense.cli.chat_complete", fake_chat_complete)
    items = find_root() / "data" / "dev.jsonl"
    out = tmp_path / "empty.jsonl"
    assert (
        main(
            [
                "eval",
                "--model",
                "qwen3:1.7b",
                "--items",
                str(items),
                "--limit",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert calls["think"] == "off"
    assert calls["max_tokens"] == 1024
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["raw"] == ""
    assert row["parse_failed"] is True
    assert row["correct"] is False
    assert row["reasoning"] == "still thinking about B"
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["parse_failure_rate"] == 1.0
    assert summary["meta"]["think"] == "off"
    assert summary["meta"]["max_tokens"] == 1024

    def answering(**kwargs):
        calls["think_on_tokens"] = kwargs["max_tokens"]
        return ChatResult(
            content="<think>B is tempting</think>\n\n**Answer:** A",
            reasoning="",
            model="qwen3:1.7b",
        )

    monkeypatch.setattr("mooresense.cli.chat_complete", answering)
    answered = tmp_path / "answered.jsonl"
    assert (
        main(
            [
                "eval",
                "--model",
                "qwen3:1.7b",
                "--think",
                "on",
                "--items",
                str(items),
                "--limit",
                "1",
                "--out",
                str(answered),
            ]
        )
        == 0
    )
    assert calls["think_on_tokens"] == 8192
    row = json.loads(answered.read_text(encoding="utf-8").splitlines()[0])
    assert row["parsed"] == "A"
    assert row["correct"] is True
    assert row["parse_failed"] is False
    assert "B is tempting" in row["reasoning"]


def _result_row(item_id: str, raw: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "abs_error": None,
        "baseline": None,
        "category": "causal",
        "chance": 0.5,
        "correct": False,
        "format": "binary",
        "gold": "yes",
        "has_distractors": False,
        "id": item_id,
        "model": "qwen3:1.7b",
        "parsed": None,
        "prediction": raw,
        "raw": raw,
        "seed_id": "causal.care.0003",
        "split": "dev",
        "tier": 2,
    }
    row.update(overrides)
    return row


def test_rescore_reparses_without_calling_the_model(tmp_path: Path):
    prose = (
        "B (Owen had worn a hat) does not stop the mud.\n"
        "- **C** (Phone stayed quiet) isn't related to the muddy floor.\n"
        "\n"
        "**Answer:** A"
    )
    # Saved under the old scorer: the prose was recorded as wrong and unparsed.
    # Gold for this id is A. The chance row must stay unscored.
    results = tmp_path / "old.jsonl"
    rows = [
        _result_row(
            "causal.chain.t3.0005-v02",
            prose,
            format="mc",
            gold="The rain had held off",
            chance=1 / 3,
            seed_id="causal.chain.0005",
            tier=3,
            has_distractors=True,
        ),
        _result_row(
            "causal.care.t2.0003-v00",
            "",
            correct=None,
            baseline="chance",
            parsed=None,
        ),
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    out = tmp_path / "rescored.jsonl"
    assert (
        main(
            [
                "rescore",
                "--results",
                str(results),
                "--items",
                str(find_root() / "data" / "dev.jsonl"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rescored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rescored[0]["parsed"] == "A"
    assert rescored[0]["correct"] is True
    assert rescored[0]["parse_failed"] is False
    assert rescored[0]["model"] == "qwen3:1.7b"
    assert rescored[1]["baseline"] == "chance"
    assert rescored[1]["correct"] is None
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["rescored_from"] == str(results)
    # The chance row is not a parse failure; the repaired row parsed.
    assert summary["parse_failure_rate"] == 0.0

    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        json.dumps(_result_row("not.a.real.item", "A")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="not in the items file"):
        main(
            [
                "rescore",
                "--results",
                str(missing),
                "--items",
                str(find_root() / "data" / "dev.jsonl"),
                "--out",
                str(tmp_path / "nope.jsonl"),
            ]
        )


def test_loglikelihood_is_not_on_the_default_path():
    item = load_jsonl(find_root() / "data" / "dev.jsonl")[0]
    with pytest.raises(NotImplementedError):
        score_loglikelihood(item, "model.gguf")


def test_lm_eval_adapter_flattens_choices_only():
    items = load_jsonl(find_root() / "data" / "dev.jsonl")
    mc = next(item for item in items if item.format.value == "mc")
    free = next(item for item in items if item.format.value == "free_text")
    doc = to_lm_eval_doc(mc)
    assert isinstance(doc["choices"], list)
    assert doc["choices"][doc["gold"]] == mc.answer.text
    with pytest.raises(ValueError):
        to_lm_eval_doc(free)
