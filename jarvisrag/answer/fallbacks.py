"""Answers used when synthesis or review does not produce one.

Extractive summaries built from the evidence, the model distillation of the
evidence, the evidence preview, and the general-knowledge answer for questions
outside the indexed documents.
"""

import os
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from jarvisrag.config import debug_print, JARVIS_DEBUG
from jarvisrag.text.query import (
    _evidence_body_from_evidence_line,
    _is_refinement_turn,
    _latest_user_turn,
    _query_anchor_terms,
    _query_requests_reasoning,
    _query_requests_values,
    _query_terms,
    _query_topic_terms,
    _source_label_from_evidence_line,
)
from jarvisrag.answer.generation import _ANSWER_MAX_TOKENS, _hit_output_cap, _llm_output_budget


def _extractive_reasoning_answer_from_evidence(user_query: str, evidence_lines: list[str]) -> str:
    if not evidence_lines or not _query_requests_reasoning(user_query):
        return ""

    facts = []
    for line in evidence_lines[:5]:
        source_label = _source_label_from_evidence_line(line)
        evidence_body = _evidence_body_from_evidence_line(line)
        snippet = evidence_body[:1600].rstrip()
        if len(evidence_body) > 1600:
            snippet += "..."
        if snippet:
            facts.append(f"- {snippet} ({source_label})")

    if not facts:
        return ""

    terms = _query_terms(user_query)
    topic_terms = _query_topic_terms(terms, _query_anchor_terms(user_query, terms))
    latest_turn = _latest_user_turn(user_query)
    topic_label = ", ".join(topic_terms[:3]) or "the requested topic"

    return (
        f"For your latest question ({latest_turn}), here is the grounded read from the indexed evidence for {topic_label}:\n"
        + "\n".join(facts[:4])
        + "\n\nA cautious implication is that the requested change or scenario would most directly affect the related "
        "values, statuses, dates, people, accounts, or other attributes shown in those facts. The indexed evidence "
        "does not by itself prove the governing rule, causal relationship, or exact outcome unless that relationship "
        "is stated in the evidence, so I would treat any conclusion beyond the listed facts as a hypothesis to verify."
    )


def _fallback_summary_budget() -> tuple[int, int]:
    """Concise budget for the last-resort evidence-preview fallbacks.

    When synthesis fails/empties, these previews must NOT become a wall of raw
    snippets (a full-length dump was the "answer is entirely too long" failure).
    Bound to a few short snippets; env-overridable per deployment. Universal UX
    improvement, so it defaults on rather than behind a feature flag."""
    items = int(os.environ.get("JARVIS_FALLBACK_SUMMARY_ITEMS", "3"))
    chars = int(os.environ.get("JARVIS_FALLBACK_SUMMARY_CHARS", "400"))
    return max(1, items), max(120, chars)


def _extractive_summary_answer_from_evidence(user_query: str, evidence_lines: list[str], max_items: int = 5) -> str:
    if not evidence_lines or _query_requests_values(user_query) or _query_requests_reasoning(user_query):
        return ""

    budget_items, budget_chars = _fallback_summary_budget()
    max_items = min(max_items, budget_items)
    latest_turn = _latest_user_turn(user_query)
    if _is_refinement_turn(user_query):
        lines = [f"Refining that for your latest question ({latest_turn}), here is what the indexed evidence supports:"]
    else:
        lines = ["Here is what I found in the indexed evidence:"]
    for evidence_line in evidence_lines[:max_items]:
        source_label = _source_label_from_evidence_line(evidence_line)
        evidence_body = _evidence_body_from_evidence_line(evidence_line)
        snippet = " ".join(evidence_body.split())[:budget_chars].rstrip()
        if len(evidence_body) > budget_chars:
            snippet += "..."
        if snippet:
            lines.append(f"- {snippet} ({source_label})")

    if len(lines) == 1:
        return ""
    lines.append("I am only summarizing what appears in the retrieved evidence here; I am not adding outside information.")
    return "\n".join(lines)


def _forced_extractive_summary_from_evidence(user_query: str, evidence_lines: list[str], max_items: int = 5) -> str:
    """Last-resort summary that ignores the value/reasoning gates.

    Used only when the LLM answer was rejected as ungrounded AND every other
    extractive helper refused. Returns a curated snippet list with sources so
    the user still sees the relevant evidence rather than a 'could not assemble'
    message.
    """
    if not evidence_lines:
        return ""
    budget_items, budget_chars = _fallback_summary_budget()
    max_items = min(max_items, budget_items)
    lines = ["Here is the closest grounded evidence I found in the indexed documents:"]
    for evidence_line in evidence_lines[:max_items]:
        source_label = _source_label_from_evidence_line(evidence_line)
        evidence_body = _evidence_body_from_evidence_line(evidence_line)
        snippet = " ".join(evidence_body.split())[:budget_chars].rstrip()
        if len(evidence_body) > budget_chars:
            snippet += "..."
        if snippet:
            lines.append(f"- {snippet} ({source_label})")
    if len(lines) == 1:
        return ""
    lines.append("I am only surfacing what appears in the retrieved evidence; verify any specific values directly in the cited sources.")
    return "\n".join(lines)


def _distill_answer_from_evidence(
    user_query: str,
    evidence_lines: list[str],
    llm_client: Ollama,
) -> str:
    """Ask the LLM to extract a SUCCINCT, value-focused answer from the evidence.

    Used after the full review loop fails. Unlike `_synthesize_evidence_answer`,
    this prompt is purely extractive: pick the value(s)/fact(s) from the
    evidence that answer the question, cite the source, and say so plainly if
    nothing in the evidence matches. The output is NOT re-reviewed (the
    evidence itself bounds it), but it is shape-checked for emptiness.
    """
    if not evidence_lines:
        return ""

    prompt = (
        "You are extracting a direct answer from indexed document evidence. Follow ALL rules:\n"
        "1. Use ONLY facts present in the Evidence section. No outside knowledge.\n"
        "2. Pull the specific value(s), number(s), name(s), or short fact(s) that answer the question.\n"
        "3. Be SUCCINCT: 1-3 sentences total. Do not list unrelated evidence lines.\n"
        "4. Cite the source label in parentheses for every concrete fact.\n"
        "5. The evidence may contain OCR artifacts. Treat 'S' immediately before digits as '$' "
        "(e.g., 'S50.00' means '$50.00'). Treat obvious letter/digit confusions (l/I/1, O/0) "
        "in the surrounding context as the digit when they appear inside a number.\n"
        "6. When the question asks 'how much', 'what is the amount', 'what is the total', or similar, "
        "any monetary value (with $, S followed by digits, USD, dollars, or labeled 'amount', "
        "'charged', 'paid', 'donation', 'total', 'balance') near a relevant noun IS a valid answer. "
        "Surface it.\n"
        "7. Only reply with exactly NO_MATCH if the evidence contains no value of the requested kind "
        "(e.g., no number at all when a number is requested). Do not say NO_MATCH just because the "
        "phrasing is indirect.\n\n"
        f"Question: {user_query}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
    )
    def _call(p: str) -> str:
        try:
            with _llm_output_budget(llm_client, _ANSWER_MAX_TOKENS):
                response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=p)])
            if _hit_output_cap(response):
                debug_print(f"[debug] Distillation reached the {_ANSWER_MAX_TOKENS}-token ceiling (runaway); discarding it.")
                return ""
            return (response.message.content or "").strip()
        except Exception as exc:
            debug_print(f"[debug] Distillation LLM call failed: {exc}")
            return ""
    answer = _call(prompt)
    if not answer:
        # Ollama occasionally returns an empty completion on the first try
        # (long prompt, transient decoding). Retry once with a tightened
        # prompt that keeps only the rules + question + top evidence lines.
        debug_print("[debug] Distillation returned empty response; retrying with tightened prompt.")
        compact_prompt = (
            "Answer ONLY from the evidence. Pull the specific value(s)/fact(s) that answer the question. "
            "Cite the source label in parentheses. Treat 'S' before digits as '$'. Reply NO_MATCH only if "
            "no value of the requested kind exists in the evidence.\n\n"
            f"Question: {user_query}\n\nEvidence:\n"
            + "\n".join(evidence_lines[:6])
        )
        answer = _call(compact_prompt)
    if not answer:
        debug_print("[debug] Distillation retry also returned empty; falling through.")
        return ""
    if answer.strip().upper().startswith("NO_MATCH"):
        debug_print("[debug] Distillation reported NO_MATCH; falling through to extractive summary.")
        return ""
    return answer


def _not_found_answer(user_query: str, reason: str = "") -> str:
    response = (
        "I could not find evidence in the indexed documents that directly answers the question."
    )
    if reason and JARVIS_DEBUG:
        response += f" Review reason: {reason}"
    return response


def _answer_from_general_knowledge(user_query: str, llm_client: Any) -> str:
    """Answer a question using the LLM's own pretraining knowledge, with no
    document grounding. Used as a last-resort fallback after the RAG pipeline
    has confirmed there is no supporting evidence in the indexed corpus."""
    query = (user_query or "").strip()
    if not query:
        return ""
    prompt = (
        "You are a helpful general-knowledge assistant.\n"
        "The user's question is NOT covered by any indexed documents, so answer "
        "from your own training knowledge.\n"
        "Be clear and complete. If you genuinely do not know, say so plainly "
        "instead of guessing. Do not invent citations or claim a source document.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    answer = ""
    try:
        # Uncap num_predict so the fallback can return a full multi-sentence
        # answer instead of being clipped by the default 256-token chat budget.
        with _llm_output_budget(llm_client, -1):
            response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        answer = (response.message.content or "").strip()
    except Exception as e:
        debug_print(f"[debug] General-knowledge LLM call raised: {e}")
        return f"General-knowledge fallback failed: {e}"
    if not answer:
        # Retry once with an even more permissive prompt; some local models
        # occasionally return an empty completion on the first attempt.
        debug_print("[debug] General-knowledge LLM returned empty on first try; retrying once.")
        try:
            with _llm_output_budget(llm_client, -1):
                response = llm_client.chat([
                    ChatMessage(
                        role=MessageRole.USER,
                        content=f"Answer the following question from your general knowledge. Be direct and complete.\n\nQuestion: {query}\n\nAnswer:",
                    )
                ])
            answer = (response.message.content or "").strip()
        except Exception as e:
            debug_print(f"[debug] General-knowledge LLM retry raised: {e}")
            return f"General-knowledge fallback failed: {e}"
    if not answer:
        debug_print("[debug] General-knowledge LLM returned empty after retry; returning empty.")
        return ""
    prefix = (
        "The indexed documents do not contain this information. "
        "Answering from general model knowledge:\n\n"
    )
    return prefix + answer


def _evidence_present_fallback(evidence_lines: list[str], reason: str = "") -> str:
    if not evidence_lines:
        return ""
    lines = [
        "I found relevant evidence in the indexed documents, but could not assemble a complete reviewed answer from it.",
        "",
        "",
        "Relevant evidence found:",
    ]
    # This is a READING preview, not the evidence itself. A live MCP result can be
    # tens of thousands of characters (full statute sections), so printing lines
    # verbatim buries the user in raw text instead of showing why the answer
    # failed. Cap each line to a scannable excerpt.
    preview_chars = int(os.environ.get("JARVIS_EVIDENCE_PREVIEW_CHARS", "800"))
    for line in evidence_lines[:6]:
        compact = " ".join(str(line).split())
        if len(compact) > preview_chars:
            compact = compact[:preview_chars].rstrip() + " …[truncated preview]"
        lines.append(compact)
    if reason and JARVIS_DEBUG:
        lines.append(f"Review reason: {reason}")
    return "\n".join(lines)
