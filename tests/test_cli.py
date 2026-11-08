"""CLI smoke tests, including a mocked OpenAI-compatible server."""

import json
from pathlib import Path

import httpx
import pytest
from tasks.lm_eval.adapter import to_lm_eval_doc

from mooresense.cli import main
from mooresense.eval.loglikelihood import score_loglikelihood
from mooresense.eval.runner import chat_complete
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
    text, echoed = chat_complete(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        prompt="Answer with B",
        api_key=None,
        timeout=5,
        seed=0,
        max_tokens=16,
        client=client,
    )
    assert text == "B"
    assert echoed == "demo"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0
    assert body["model"] == "qwen3:8b"


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
