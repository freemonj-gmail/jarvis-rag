"""Questions back to the user when one detail would make a turn answerable.
"""

import os
import re
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from jarvisrag.config import debug_print
from jarvisrag.iotools import Spinner
from jarvisrag.answer.generation import _llm_output_budget
from jarvisrag.answer.fallbacks import _not_found_answer


# General quantity/value intent -- a number that can change from year to year.
# Domain-neutral (no tax-specific words) so the period clarification generalizes
# to any corpus: prices, rates, thresholds, limits, fees, etc.
_QUANTITY_INTENT_RE = re.compile(
    r"\b(limit|limits|rate|rates|amount|amounts|maximum|minimum|threshold|thresholds"
    r"|cap|caps|how much|how many|price|prices|cost|costs|fee|fees)\b|[$€£%]",
    re.I,
)


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _period_clarification_from_evidence(user_query: str, evidence_lines: list[str]) -> str:
    """Deterministically ask WHICH year when a quantity question names none but
    the retrieved evidence spans several years.

    A value that varies by year (a limit, rate, threshold, price, ...) cannot be
    pinned down if the question fixes no period and the evidence carries two or
    more distinct years -- and this session proved the local models HEDGE instead
    of guessing. So ask the user for the year rather than infer it. Deterministic
    and DOMAIN-AGNOSTIC: it keys only on 'a quantity question + multiple years in
    the evidence', never on any specific topic, value, or corpus. Returns '' when
    not applicable, so the caller falls through to the general LLM triage.
    """
    q = user_query or ""
    if _YEAR_RE.search(q):
        return ""  # the query already fixes a period
    if not _QUANTITY_INTENT_RE.search(q):
        return ""  # period only matters for a quantity that varies over time
    years = {
        y
        for line in evidence_lines
        for y in _YEAR_RE.findall(str(line))
        if 1990 <= int(y) <= 2100
    }
    if len(years) < 2:
        return ""
    shown = ", ".join(sorted(years)[-4:])
    return (
        "Which year are you asking about? The retrieved evidence covers multiple "
        f"years ({shown}) and the amounts differ by year."
    )


def _clarifying_question_from_evidence(
    user_query: str,
    evidence_lines: list[str],
    llm_client: Any,
) -> str:
    """Ask for the ONE user-supplied detail that would unlock an answer.

    Runs only after the answer+review loop failed while on-topic evidence WAS
    retrieved. Two outcomes are possible and both are useful: the evidence simply
    does not cover the topic (return "" and let the caller show the evidence
    preview), or it covers the topic but is ambiguous without a detail only the
    user holds -- a period, an entity among several, a jurisdiction, a units
    basis, which of several plans/accounts they mean.

    DOMAIN-AGNOSTIC: the model is asked to derive the missing dimension from the
    question and evidence themselves, so this works for any corpus. It must never
    ask for something the corpus should supply -- that would push lookup work
    back onto the user.
    """
    if not (user_query or "").strip():
        return ""
    # Deterministic fast-path: a quantity question with no year, over evidence
    # that spans several years, is period-ambiguous. Ask which year directly
    # rather than relying on the triage model, which tends to decline. Agnostic.
    period_question = _period_clarification_from_evidence(user_query, evidence_lines or [])
    if period_question:
        return period_question
    # Triage only needs enough evidence to see WHAT dimension varies, not the
    # full text; a live MCP result can be tens of thousands of characters.
    budget = int(os.environ.get("JARVIS_CLARIFY_EVIDENCE_CHARS", "1500"))
    material = "\n".join(
        " ".join(str(line).split())[:budget] for line in (evidence_lines or [])[:6]
    )
    if material:
        context_block = f"Evidence excerpts:\n{material}\n\n"
        no_detail_rule = (
            "- If the evidence does not cover the topic at all, or no single detail would "
            "resolve it, reply with exactly: NONE\n"
        )
    else:
        # Nothing was retrieved: the useful question is the one that would make a
        # RETRIEVAL succeed (which document, period, account), not one about text
        # we never found.
        context_block = "No supporting material was retrieved for this question.\n\n"
        no_detail_rule = (
            "- Nothing was retrieved, so ask only if a detail would plausibly let a fresh "
            "search succeed. If the question is simply outside the documents, or is already "
            "fully specific, reply with exactly: NONE\n"
        )
    prompt = (
        "You triage questions that could not be answered from retrieved evidence.\n"
        "Decide whether ONE specific detail that only the USER can supply would make the "
        "evidence sufficient to answer.\n"
        "Such a detail is typically a dimension the evidence splits on: a time period or "
        "year, which entity/account/plan/person is meant among several, a jurisdiction, a "
        "units or currency basis, or which of several documents applies.\n\n"
        "RULES:\n"
        "- The detail must be something the USER knows about their own situation, NEVER "
        "something that should be looked up in the documents.\n"
        "- Ask only if the evidence would genuinely answer the question once the detail is known.\n"
        + no_detail_rule +
        "- Otherwise reply with ONE short question of at most 25 words. No preamble, no "
        "explanation, no bullet points.\n\n"
        f"Question: {user_query}\n\n"
        + context_block +
        "Reply (a single question, or NONE):"
    )
    try:
        with Spinner("Checking what's missing"), _llm_output_budget(llm_client, 120):
            response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        candidate = " ".join((response.message.content or "").split()).strip()
    except Exception as exc:
        debug_print(f"[debug] Clarification triage failed: {exc}")
        return ""
    if not candidate or candidate.upper().startswith("NONE"):
        return ""
    # Extract the first question sentence wherever it appears, THEN validate its
    # length. Validating the whole reply first (the earlier order) threw away a
    # perfectly good question whenever the model wrapped it in a preamble or
    # added a sentence of justification -- the triage ran, cost real time, and
    # silently produced nothing.
    match = re.search(r"([^.?!\n]{8,240}\?)", candidate)
    if not match:
        debug_print(f"[debug] Clarification triage returned no question: {candidate[:120]}")
        return ""
    question = match.group(1).strip(" \t-*•>\"'")
    if len(question.split()) > 40:
        debug_print(f"[debug] Discarded over-long clarification: {question[:120]}")
        return ""
    return question


def _dead_end_response(
    user_query: str,
    llm_client: Any,
    last_turn: dict[str, object],
    reason: str = "",
) -> str:
    """Last resort when nothing could answer the turn: ask for the one missing
    detail rather than simply declaring failure.

    Applies to EVERY dead end, with or without retrieved evidence, so the
    clarify-then-answer behavior is not special-cased to one lane. Falls back to
    the plain not-found response when no single detail would help, and never asks
    twice in a row.
    """
    if not last_turn.get("awaiting_clarification"):
        clarifying = _clarifying_question_from_evidence(user_query, [], llm_client)
        if clarifying:
            debug_print("[debug] Dead end; asking the user one clarifying question.")
            return (
                "I could not answer that from the indexed documents. One detail would "
                f"let me try again properly:\n\n{clarifying}"
            )
    return _not_found_answer(user_query, reason)
