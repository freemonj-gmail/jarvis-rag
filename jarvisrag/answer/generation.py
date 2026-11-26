"""Output limits for model calls: per-call token budgets and the runaway cap.
"""

from contextlib import contextmanager
from typing import Any

from llama_index.llms.ollama import Ollama


# Safety ceiling for an evidence answer (synthesis, refinement, distillation).
# Not a length target: the longest measured answers are ~1,300 tokens, while a
# runaway generation was observed running into the 600 s request timeout. At
# ~100-160 tokens/s this ends a runaway within about a minute. A capped reply
# is treated as no answer.
_ANSWER_MAX_TOKENS = 8192


def _hit_output_cap(response: Any) -> bool:
    """True when the model stopped because it reached the token cap, not by finishing."""
    raw = getattr(response, "raw", None)
    reason = raw.get("done_reason") if isinstance(raw, dict) else getattr(raw, "done_reason", None)
    return reason == "length"


@contextmanager
def _llm_output_budget(llm_client: Ollama, num_predict: int):
    """Temporarily override the Ollama `num_predict` for a call.

    Pass a positive integer to raise the cap to at least that many tokens.
    Pass -1 to fully uncap generation (model stops on its own EOS token), for
    output that is legitimately long (authored documents, general knowledge).
    Evidence answers use the ``_ANSWER_MAX_TOKENS`` ceiling instead. The prior
    value is always restored on exit so cheaper callsites (review PASS/FAIL,
    speech summary) keep their default cap.
    """
    extra = getattr(llm_client, "additional_kwargs", None)
    if not isinstance(extra, dict):
        yield
        return
    previous = extra.get("num_predict")
    if num_predict == -1:
        extra["num_predict"] = -1
    else:
        extra["num_predict"] = max(int(previous or 0), num_predict)
    try:
        yield
    finally:
        if previous is None:
            extra.pop("num_predict", None)
        else:
            extra["num_predict"] = previous
