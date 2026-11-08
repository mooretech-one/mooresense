"""Log-likelihood scoring for base models.

v0.1 scores chat completions by the parsed answer letter (see
``mooresense.eval.runner``). Ranking choices by log probability, for base
models that do not chat, is intentionally not on the default path: it needs
llama-cpp-python, which is an optional extra so CI does not build it.

Install with ``pip install 'mooresense[ll]'`` and score a GGUF locally when
you need that number. The hook below is the place that call should land.
"""

from __future__ import annotations

from mooresense.schema import Item


def score_loglikelihood(item: Item, model_path: str) -> str:
    """Return the choice label with the highest log-likelihood.

    Not implemented in this foundation. Chat scoring is the supported path.
    """
    if item.format.value not in {"mc", "binary"}:
        raise ValueError("log-likelihood scoring only applies to mc and binary items")
    raise NotImplementedError(
        "Log-likelihood scoring requires the optional 'll' extra (llama-cpp-python) "
        f"and a local GGUF. Requested model file: {model_path}. "
        "Use `mooresense eval` against an OpenAI-compatible chat server for v0.1."
    )
