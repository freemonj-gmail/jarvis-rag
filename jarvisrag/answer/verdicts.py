"""Model verdicts on answers and turns, with deterministic fallbacks.

Whether an answer is a non-answer, supplies what the question asked, rests on
evidence about the question at all, and whether a new message continues the
previous exchange. Each is a one-word model verdict with a rule-based fallback.
"""

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from jarvisrag.config import debug_print
from jarvisrag.iotools import Spinner
from jarvisrag.text.query import _looks_like_follow_up, _structural_follow_up


# A genuine "I have nothing" response is short. Longer text that merely contains
# or opens with a hedging phrase is a real answer and must not be discarded.
_DISMISSIVE_MAX_CHARS = 240


def _response_is_dismissive(response_text: str) -> bool:
    text = " ".join((response_text or "").strip().lower().split())
    if not text:
        return True

    exact_markers = {
        "empty response",
        "no information is available",
        "no information available",
        "the provided page is blank",
        "provided page is blank",
        "page is blank",
    }
    if text in exact_markers:
        return True

    # Catch generic refusal-style answers without clobbering long evidence previews.
    prefix_markers = [
        "i could not find a clear answer",
        "i could not find an answer",
        "i cannot find a clear answer",
        "i cannot find an answer",
        "i could not find",
        "i cannot find",
        "i did not find",
        "i don't find",
        "i do not find",
        "the provided evidence does not",
        "the evidence does not",
        "the indexed documents do not",
        "the indexed corpus does not",
        "the documents do not",
        "the provided context does not",
        "the context does not",
        "the supplied context does not",
    ]
    # A genuine refusal is SHORT. A long answer that merely OPENS with a hedge
    # ("The evidence does not give the 2026 figure, but section 219(b)(5)(B)(ii)
    # sets the age-50 catch-up at ...") is a real, useful answer and must not be
    # thrown away -- discarding it costs the user the answer AND replaces it with
    # a raw evidence dump. Same length guard the substring markers below use.
    if len(text) <= _DISMISSIVE_MAX_CHARS and any(text.startswith(marker) for marker in prefix_markers):
        return True

    # As a final safeguard, only treat marker substrings as dismissive when the
    # response is very short (likely a generic fallback, not a real evidence block).
    contains_markers = [
        "empty response",
        "no information is available",
        "provided page is blank",
        "page is blank",
        "no information available",
        "cannot find",
        "could not find",
        "did not find",
        "does not contain",
        "does not mention",
        "do not contain",
        "do not mention",
    ]
    if len(text) <= _DISMISSIVE_MAX_CHARS and any(marker in text for marker in contains_markers):
        return True

    return False


# Phrases an answer uses to report that something specific is absent. Unlike the
# dismissive markers, these appear INSIDE otherwise-substantive answers.
_HEDGE_MARKERS = (
    "does not contain", "do not contain", "does not state", "do not state",
    "does not explicitly", "do not explicitly", "does not specify", "do not specify",
    "does not mention", "do not mention", "does not provide", "do not provide",
    "is not specified", "are not specified", "is not stated", "are not stated",
    "could not determine", "cannot determine", "not enough information",
)


def _answer_hedges_about_missing_detail(response_text: str) -> bool:
    """True when a substantive answer still reports a key detail as absent."""
    text = " ".join((response_text or "").lower().split())
    if not text:
        return False
    return any(marker in text for marker in _HEDGE_MARKERS)


ANSWER_COMPLETE = "answered"


ANSWER_PARTIAL = "partial"


ANSWER_UNANSWERED = "unanswered"


_COMPLETENESS_PROMPT = (
    "You check whether an answer gives the user the information their question asks for.\n"
    "Judge ONLY whether the requested information is present in the answer, not whether it is correct.\n"
    "If the question contains earlier turns followed by a latest message, judge against the latest message.\n"
    "Reply with exactly one word:\n"
    "ANSWERED - the answer supplies every specific value, detail, or fact the question asks for.\n"
    "PARTIAL - the answer supplies some of what was asked, but says or implies that another requested part "
    "is missing, unavailable, not provided, unknown, or could not be found.\n"
    "UNANSWERED - the answer supplies none of what was asked: it reports the information as absent, or "
    "discusses something other than what was asked.\n\n"
    "Question: {question}\n\nAnswer: {answer}\n\nVerdict:"
)


def _answer_completeness(user_query: str, response_text: str, llm_client: Any) -> str:
    """Whether an answer actually supplies what the question asked for.

    A semantic verdict instead of phrase matching: an answer can report a detail
    as absent in endless wordings ("is not provided", "isn't listed", "there's no
    mention", "I wasn't able to locate"), and a marker list misses most of them
    (measured 0/7 on paraphrased non-answers, while the model judged 16/16 for
    the answered-vs-not decision). Falls back to the deterministic markers if the
    model is unavailable or replies off-format."""
    if not (response_text or "").strip() or _response_is_dismissive(response_text):
        return ANSWER_UNANSWERED
    prompt = _COMPLETENESS_PROMPT.format(question=user_query, answer=response_text)
    try:
        with Spinner("Checking the answer"):
            response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        verdict = " ".join((response.message.content or "").split()).upper()
    except Exception as exc:
        debug_print(f"[debug] Answer completeness check failed ({type(exc).__name__}); using markers.")
        verdict = ""
    # UNANSWERED contains ANSWERED, so test it first.
    for word, label in (("UNANSWERED", ANSWER_UNANSWERED), ("PARTIAL", ANSWER_PARTIAL), ("ANSWERED", ANSWER_COMPLETE)):
        if verdict.startswith(word):
            debug_print(f"[debug] Answer completeness: {label}.")
            return label
    if verdict:
        debug_print(f"[debug] Unrecognized completeness verdict: {verdict[:60]}; using markers.")
    return ANSWER_PARTIAL if _answer_hedges_about_missing_detail(response_text) else ANSWER_COMPLETE


_EVIDENCE_TOPIC_PROMPT = (
    "You decide whether excerpts retrieved from a user's documents are about what their question asks.\n"
    "RELATED - the excerpts are about the specific thing the question asks about (the same person, organization, "
    "item, account, rule, program, or event), so they are where its answer would be expected, even if this "
    "particular detail is missing.\n"
    "UNRELATED - the excerpts are about something else: they only share words or a broad field with the question, "
    "and its answer would not be expected in them.\n"
    "Reply with exactly one word: RELATED or UNRELATED.\n\n"
    "Question: {question}\n\nExcerpts:\n{excerpts}\n\nVerdict:"
)


_EVIDENCE_TOPIC_LINES = 12


_EVIDENCE_TOPIC_CHARS = 700


def _evidence_concerns_question(user_query: str, evidence_lines: list[str], llm_client: Any) -> bool:
    """Whether retrieved evidence is about what the question asks, judged by meaning.

    Retrieval admits a chunk on shared words, so "query-supported" evidence can
    be about something else entirely: "the first woman to win a Nobel Prize"
    pulled insurance-policy pages and a resume, then a tax-law section on prize
    income. Showing that as "relevant evidence" is wrong; the question belongs to
    general knowledge. Measured on 72 labelled pairs from three corpora (general
    knowledge, in-corpus, and in-corpus-but-missing-the-detail questions): 72/72
    on each of 3 runs, ~0.45 s per verdict. Keeps the evidence (the previous
    behavior) when the model is unavailable or replies off-format."""
    if not evidence_lines or llm_client is None:
        return bool(evidence_lines)
    excerpts = "\n".join(
        f"{i}. {' '.join(str(line).split())[:_EVIDENCE_TOPIC_CHARS]}"
        for i, line in enumerate(evidence_lines[:_EVIDENCE_TOPIC_LINES], 1)
    )
    prompt = _EVIDENCE_TOPIC_PROMPT.format(question=user_query.strip(), excerpts=excerpts)
    try:
        with Spinner("Checking the evidence"):
            response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        verdict = " ".join((response.message.content or "").split()).upper()
    except Exception as exc:
        debug_print(f"[debug] Evidence topic check failed ({type(exc).__name__}); keeping the evidence.")
        return True
    # UNRELATED contains RELATED, so test it first.
    if verdict.startswith("UNRELATED"):
        return False
    if not verdict.startswith("RELATED"):
        debug_print(f"[debug] Unrecognized evidence topic verdict: {verdict[:60]}; keeping the evidence.")
    return True


_FOLLOW_UP_PROMPT = (
    "You decide whether a user's new message continues the previous exchange or starts a new question.\n"
    "CONTINUES - the new message depends on the previous exchange to be understood: a word or phrase in it "
    "refers back to something named only in the previous question or answer, or it asks to repeat, refine, "
    "correct, extend, or vary that question.\n"
    "NEW - the new message can be understood on its own: anything it refers to is named within the new "
    "message itself, or it asks about a different subject.\n"
    "Reply with exactly one word: CONTINUES or NEW.\n\n"
    "Previous question: {previous_question}\n"
    "Previous answer: {previous_answer}\n\n"
    "New message: {new_message}\n\n"
    "Verdict:"
)


def _turn_continues_previous(user_query: str, last_turn: dict[str, object], llm_client: Any) -> bool:
    """Whether this message continues the previous exchange, judged by meaning.

    Pronoun and marker words cannot tell a back-reference from a reference
    inside the same message ("...the child tax credit, and how much is it?"),
    and they miss follow-ups that use none ("And for heads of household?").
    Measured on 52 labelled pairs across 8 subjects: the model 52/52, the
    wording rules 36/52, at ~0.07 s per verdict; and 14/14 on replies to a
    clarifying question, including a user who moved on to a new question.
    Structural cases are settled
    without the model; the wording rules are the fallback when the model is
    unavailable or replies off-format."""
    structural = _structural_follow_up(user_query, last_turn)
    if structural is not None:
        return structural
    if llm_client is None:
        return _looks_like_follow_up(user_query, last_turn)
    prompt = _FOLLOW_UP_PROMPT.format(
        previous_question=str(last_turn.get("effective_query") or last_turn.get("user_query") or "").strip(),
        previous_answer=str(last_turn.get("response_text") or "").strip(),
        new_message=user_query.strip(),
    )
    try:
        response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        verdict = " ".join((response.message.content or "").split()).upper()
    except Exception as exc:
        debug_print(f"[debug] Follow-up check failed ({type(exc).__name__}); using wording rules.")
        verdict = ""
    if verdict.startswith("CONTINUES"):
        return True
    if verdict.startswith("NEW"):
        return False
    if verdict:
        debug_print(f"[debug] Unrecognized follow-up verdict: {verdict[:60]}; using wording rules.")
    return _looks_like_follow_up(user_query, last_turn)
