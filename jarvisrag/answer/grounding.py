"""Checks that an answer rests on its evidence.

The stated-figure check (every number an answer states must appear in its
evidence) and the model reviewer that judges an answer against the evidence.
"""

import os
import re

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from jarvisrag.common.numbers import canonical_number, NUMBER_PATTERN
from jarvisrag.speech import _SOURCE_REFERENCE_RE
from jarvisrag.text.query import _query_requests_reasoning, _query_requests_refinement
from jarvisrag.answer.verdicts import _response_is_dismissive


# Not inside a word or after a decimal point; the number itself is read by the
# shared rule ("1,234,567" is one number, "Jan 29,2003" a day and a year).
_CONCRETE_VALUE_RE = re.compile(r"(?<![\w.])" + NUMBER_PATTERN)


# A page reference without a file name ("page 05", "pp. 3-4"); file names with
# an optional page are matched by the shared _SOURCE_REFERENCE_RE.
_PAGE_REFERENCE_RE = re.compile(r"\b(?:pages?|pp?\.)\s*\d+(?:\s*[-\u2013]\s*\d+)?", re.IGNORECASE)


def _concrete_values(text: str) -> set[str]:
    """Numbers, amounts, percentages and years in ``text``, normalized so equal
    numbers compare equal however they are written ("7,500.00" == "7500",
    "2.540" == "2.54", "05" == "5")."""
    values = {canonical_number(raw) for raw in _CONCRETE_VALUE_RE.findall(text or "")}
    values.discard("")
    return values


def _unsupported_answer_values(answer: str, evidence_lines: list[str]) -> set[str]:
    """Concrete values the answer states that appear nowhere in its evidence.

    File names and page references in the answer are skipped: they identify
    where a fact came from rather than state one, and a follow-up answer may
    cite a page that this turn's evidence does not include."""
    claims = _PAGE_REFERENCE_RE.sub(" ", _SOURCE_REFERENCE_RE.sub(" ", answer or ""))
    return _concrete_values(claims) - _concrete_values("\n".join(evidence_lines))


def _review_answer_against_evidence(
    user_query: str,
    answer: str,
    evidence_lines: list[str],
    llm_client: Ollama,
) -> tuple[bool, str]:
    # Diagnostic switch only: JARVIS_ENABLE_ANSWER_REVIEW=0 skips the review loop.
    if os.environ.get("JARVIS_ENABLE_ANSWER_REVIEW", "1") != "1":
        return True, "review disabled"
    if _response_is_dismissive(answer):
        return True, "answer is already a not-found response"
    if not evidence_lines:
        return False, "no retrieved evidence available for review"

    reasoning_requested = _query_requests_reasoning(user_query)
    refinement_requested = _query_requests_refinement(user_query) or "\n" in user_query.strip()

    # Deterministic guard: if the latest turn asks for reasoning/implications,
    # an answer that is just a short value+source citation (e.g.
    # "Registration: $125.50 (file.pdf, page 1)") is a recycled value answer,
    # not reasoning. Fail it without an LLM round-trip so the cascade can route
    # to the reasoning extractor or surface the unverified best answer with a
    # caveat.
    if reasoning_requested:
        answer_compact = " ".join((answer or "").split())
        answer_lower = answer_compact.lower()
        reasoning_markers_in_answer = (
            "implication", "imply", "implies", "infer", "suggest", "likely",
            "would", "could", "might", "may ", "hypothes", "because",
            "therefore", "as a result", "cannot determine", "does not state",
            "does not contain", "is silent", "evidence does not",
        )
        has_reasoning_marker = any(m in answer_lower for m in reasoning_markers_in_answer)
        has_money_value = bool(re.search(r"\$\s*\d", answer_compact))
        if has_money_value and not has_reasoning_marker and len(answer_compact.split()) < 25:
            return False, (
                "answer recycles a value/source citation without addressing the requested "
                "reasoning or implications"
            )

    if reasoning_requested:
        support_rule = (
            "The user is asking for evidence-grounded reasoning. Return PASS if the answer separates retrieved facts "
            "from implications or hypotheses, and every implication is a cautious inference from those facts. "
            "Return FAIL if it presents external rules, professional advice, or unsupported predictions as facts.\n"
        )
    elif refinement_requested:
        support_rule = (
            "The user is refining a prior question. Return PASS if the answer engages with the latest clarification "
            "and is grounded in the evidence, even if it must say the evidence does not contain the requested detail. "
            "Return FAIL only if the answer ignores the latest clarification or invents facts not in the evidence.\n"
        )
    else:
        support_rule = "Return PASS only if the answer addresses the requested subject, topic, and requested values/details.\n"

    prompt = (
        "You are a strict answer reviewer for a document-grounded assistant.\n"
        "Decide whether the answer directly answers the user's question and is supported only by the evidence.\n"
        "If the question contains prior turns plus a latest clarification, judge whether the answer addresses the latest clarification.\n"
        f"{support_rule}"
        "HARD RULE: Return FAIL if the answer introduces ANY concrete fact, number, date, name, rule, or definition that is NOT present in the evidence below, "
        "even if it sounds plausible. Outside knowledge counts as hallucination here.\n"
        "Return FAIL if the answer is generic, off-topic, mostly raw unrelated excerpts, or missing requested values/details that ARE in the evidence.\n"
        "Respond with exactly two lines:\n"
        "PASS or FAIL\n"
        "A short reason\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {answer}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
    )
    try:
        response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        review_text = (response.message.content or "").strip()
    except Exception as e:
        return True, f"review unavailable: {e}"

    first_line = review_text.splitlines()[0].strip().upper() if review_text else ""
    if first_line.startswith("PASS"):
        return True, review_text
    if first_line.startswith("FAIL"):
        reason = " ".join(review_text.splitlines()[1:]).strip() or "answer failed evidence review"
        return False, reason
    return True, f"unrecognized review response: {review_text[:120]}"
