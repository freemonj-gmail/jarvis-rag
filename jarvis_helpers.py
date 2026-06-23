import json
import logging
import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess
import shutil
import warnings
import time
import math
import signal
import hashlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, UTC
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Sequence, cast

import re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract
import easyocr
import asyncio
# Kokoro is imported lazily inside TTSManager so smoke tests and non-TTS
# callers don't pay the model-load cost (and don't require espeak-ng).
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.llms.ollama import Ollama
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.readers.file.unstructured import UnstructuredReader

from jarvis_config import (
    INDEX_MANIFEST_FILE,
    INDEX_MANIFEST_VERSION,
    JARVIS_DEBUG,
    JARVIS_DOMAIN,
    debug_print,
    logger,
    short_text_qa_template,
)


# ---------------------------------------------------------------------------
# LLM Router — two-tier model escalation (DISABLED BY DEFAULT)
# ---------------------------------------------------------------------------
# WHAT THIS DOES
#   Provides an optional second model "tier" intended for future multi-GPU
#   hardware. The PRIMARY model (small/fast, e.g. gemma) keeps answering
#   everything that can be grounded in the local RAG corpus. When a query falls
#   OUTSIDE the indexed corpus and Jarvis must fall back to parametric
#   ("general") knowledge, the router escalates that single call to a larger
#   ESCALATION model (e.g. llama3:70b) with broader world knowledge.
#
# WHY ESCALATE ONLY ON THE CORPUS-MISS PATH
#   Retrieval is shared: both models read the same vector store, so running the
#   big model adds nothing when good evidence already exists. The large model
#   only helps when the answer must come from the model's own training
#   knowledge — i.e. exactly the `_answer_from_general_knowledge` fallback.
#   Routing only there keeps the expensive 70B model idle for the common
#   (grounded) case and active only for genuine corpus misses, minimizing VRAM
#   pressure and latency.
#
# ACTIVATION
#   Fully gated behind JARVIS_ROUTER=1. While unset/"0" (the default), Jarvis
#   behaves exactly as before: the primary client handles every call, the
#   escalation model is never constructed, and no extra VRAM or latency is
#   incurred. Flip the switch on after the hardware upgrade to enable it.
#
# CONFIGURATION (environment variables)
#   JARVIS_ROUTER                 "0"           enable ("1") / disable the tier
#   JARVIS_ESCALATION_MODEL       "llama3:70b"  Ollama tag for the escalation model
#   JARVIS_ESCALATION_TIMEOUT     "600"         request timeout (seconds)
#   JARVIS_ESCALATION_TEMPERATURE "0.2"         sampling temperature
#
# FAILURE HANDLING
#   If the escalation client cannot be built (Ollama down, model not pulled),
#   the router logs the reason and transparently returns the primary client, so
#   an answer is never blocked by a missing big model.
# ---------------------------------------------------------------------------
_escalation_llm_cache: dict[str, Any] = {}


def _router_enabled() -> bool:
    return os.environ.get("JARVIS_ROUTER", "0") == "1"


def _build_escalation_llm() -> Any:
    """Lazily construct (and cache) the escalation Ollama client. Returns None
    if the model tag is empty or the client cannot be created."""
    model = os.environ.get("JARVIS_ESCALATION_MODEL", "llama3:70b").strip()
    if not model:
        return None
    if model in _escalation_llm_cache:
        return _escalation_llm_cache[model]
    client = None
    try:
        timeout = float(os.environ.get("JARVIS_ESCALATION_TIMEOUT", "600"))
        temperature = float(os.environ.get("JARVIS_ESCALATION_TEMPERATURE", "0.2"))
        client = Ollama(
            model=model,
            request_timeout=timeout,
            temperature=temperature,
        )
    except Exception as e:
        debug_print(f"[debug] Router: failed to build escalation LLM '{model}': {e}")
        client = None
    _escalation_llm_cache[model] = client
    return client


def route_general_knowledge_llm(primary_llm: Any) -> Any:
    """Pick the LLM client for the out-of-corpus (general-knowledge) fallback.

    Returns the larger escalation model when the router is enabled
    (JARVIS_ROUTER=1) and that model can be constructed; otherwise returns the
    unchanged primary client. This is the single decision point that makes the
    smaller primary model "defer to" the larger one for questions that are not
    covered by the indexed corpus."""
    if not _router_enabled():
        return primary_llm
    escalation = _build_escalation_llm()
    if escalation is None:
        debug_print("[debug] Router enabled but escalation LLM unavailable; using primary client.")
        return primary_llm
    model = os.environ.get("JARVIS_ESCALATION_MODEL", "llama3:70b").strip()
    debug_print(f"[debug] Router: escalating out-of-corpus query to '{model}'.")
    return escalation


GENERIC_DETAIL_KEYWORDS = [
    "account", "account number", "address", "amount", "balance", "certificate",
    "city", "contract", "credit", "customer", "date", "deduction", "description",
    "document", "due", "email", "employer", "fee", "id", "identifier", "invoice",
    "mailing", "name", "number", "organization", "owner", "payment", "phone",
    "property", "recipient", "registration", "statement", "total", "withheld",
]

DOMAIN_DETAIL_KEYWORDS = {
    "tax": [
        "w-2", "1099", "1098", "1040", "dividend", "distribution", "federal income",
        "form", "interest", "medicare wages", "mortgage", "payer", "principal residence",
        "section 199a", "social security", "socialsecuritynumber", "state wages", "tax",
        "wages",
    ],
}

GENERIC_BOILERPLATE_MARKERS = [
    ("all rights reserved", 40),
    ("continued", 15),
    ("disclosure", 30),
    ("for your records", 35),
    ("instructions", 40),
    ("privacy", 25),
    ("see instructions", 35),
    ("see the", 20),
    ("terms and conditions", 35),
]

DOMAIN_BOILERPLATE_MARKERS = {
    "tax": [
        ("department of the treasury", 60),
        ("internal revenue service", 45),
        ("keep for your records", 60),
        ("this is important tax information", 70),
        ("understanding your statement", 45),
        ("www.irs.gov", 60),
    ],
}


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
    if any(text.startswith(marker) for marker in prefix_markers):
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
    if len(text) <= 240 and any(marker in text for marker in contains_markers):
        return True

    return False


def _salvage_answer_from_sources(user_query: str, response: object, llm_client: Ollama) -> str:
    source_nodes = getattr(response, "source_nodes", None) or []
    if not source_nodes:
        return ""

    excerpts = []
    for idx, node in enumerate(source_nodes[:6], start=1):
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text:
            continue
        if is_low_information_text(node_text):
            continue
        compact_text = " ".join(node_text.split())
        excerpts.append(f"EXCERPT {idx}: {compact_text[:900]}")

    if not excerpts:
        return ""

    salvage_prompt = (
        "Extract any useful facts from these excerpts. Do not claim the page is blank unless excerpts are truly empty. "
        "Prefer concrete details such as names, dates, amounts, account/form numbers, addresses, and organizations. "
        "If facts are limited, return whatever is available and note limits briefly.\n\n"
        f"Question: {user_query}\n\n"
        f"Excerpts:\n{os.linesep.join(excerpts)}"
    )

    try:
        salvage_response = llm_client.chat([
            ChatMessage(role=MessageRole.USER, content=salvage_prompt)
        ])
        return (salvage_response.message.content or "").strip()
    except Exception:
        return ""


def _build_evidence_preview(response: object, max_items: int = 4) -> str:
    source_nodes = getattr(response, "source_nodes", None) or []
    if not source_nodes:
        return ""

    lines = [
        "I retrieved matching content but could not synthesize a clean answer. Here are direct evidence snippets:",
    ]

    added = 0
    for node in source_nodes:
        if added >= max_items:
            break
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text:
            continue
        if is_low_information_text(node_text):
            continue

        metadata = getattr(node, "metadata", {}) or {}
        file_name = metadata.get("file_name") or metadata.get("file_path") or "unknown file"
        page_number = metadata.get("page_number")
        score = getattr(node, "score", None)

        compact = " ".join(normalize_text(node_text).split())
        snippet = compact[:1600].rstrip()
        if len(compact) > 1600:
            snippet += "..."

        label = f"- {file_name}"
        if page_number is not None:
            label += f" (page {page_number})"
        if JARVIS_DEBUG and score is not None:
            try:
                label += f" [score={float(score):.3f}]"
            except Exception:
                pass

        lines.append(label)
        lines.append(f"  {snippet}")
        added += 1

    return "\n".join(lines) if added else ""


def _source_nodes_to_evidence_lines(source_nodes: list[object], max_items: int = 8) -> list[str]:
    evidence_lines = []
    for node in source_nodes[:max_items]:
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text or is_low_information_text(node_text):
            continue

        metadata = getattr(node, "metadata", {}) or {}
        file_name = os.path.basename(
            str(metadata.get("file_name") or metadata.get("file_path") or "unknown file")
        )
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"

        compact = " ".join(normalize_text(node_text).split())
        evidence_lines.append(f"- {label}: {compact[:4000]}")
    return evidence_lines


def _source_nodes_to_query_evidence_lines(
    user_query: str,
    source_nodes: list[object],
    max_items: int = 8,
    source_hints: list[str] | None = None,
) -> list[str]:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    require_value = _query_requests_values(user_query)
    hint_basenames = {os.path.basename(h).lower() for h in (source_hints or []) if h}

    ranked: list[tuple[int, set[str], str]] = []
    forced: list[str] = []  # source-hint matches bypass the score filter
    for node in source_nodes:
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text or is_low_information_text(node_text):
            continue

        metadata = getattr(node, "metadata", {}) or {}
        node_fname = str(metadata.get("file_name") or "").lower()
        node_fpath = str(metadata.get("file_path") or "").lower()
        is_source_hinted = bool(hint_basenames) and any(
            h in node_fname or h in node_fpath for h in hint_basenames
        )

        search_text = node_text
        for key in ("file_name", "file_path", "parent_file", "source"):
            value = metadata.get(key)
            if value:
                search_text += f"\n{value}"

        file_name = os.path.basename(
            str(metadata.get("file_name") or metadata.get("file_path") or "unknown file")
        )
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"

        if is_source_hinted:
            # Bypass score filter for user-named files; send the full chunk text
            # (capped at JARVIS_SOURCE_HINT_CHUNK_CHARS, default 4000) so employer
            # names, dates, and lists aren't severed mid-word.
            max_chars = int(os.environ.get("JARVIS_SOURCE_HINT_CHUNK_CHARS", "16000"))
            snippet = " ".join(node_text.split())[:max_chars].rstrip()
            if len(node_text) > max_chars:
                snippet += "..."
            forced.append(f"- {label}: {snippet}")
            continue

        score = _query_candidate_score(search_text, terms, anchor_terms, require_value)
        if score <= 0:
            continue

        snippet = _query_focused_snippet(node_text, terms, anchor_terms, require_value)
        if not _query_evidence_is_supported(snippet, terms, anchor_terms, require_value):
            continue

        coverage = _query_topic_coverage(snippet, terms, anchor_terms)
        ranked.append((score, coverage, f"- {label}: {snippet}"))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = list(forced)  # source-hinted evidence first
    covered_topics: set[str] = set()

    for _, coverage, line in ranked:
        if line in selected:
            continue
        if coverage - covered_topics:
            selected.append(line)
            covered_topics.update(coverage)
            if len(selected) >= max_items:
                return selected

    for _, _, line in ranked:
        if line in selected:
            continue
        selected.append(line)
        if len(selected) >= max_items:
            break

    return selected


def _normalize_source_hint(source_hint: str) -> str:
    normalized = " ".join(source_hint.strip(" '\".,;:()[]{}").split())
    lowered = normalized.lower()
    for marker in (" document ", " file ", " entitled ", " named ", " called "):
        marker_index = lowered.rfind(marker)
        if marker_index >= 0:
            normalized = normalized[marker_index + len(marker):].strip()
            lowered = normalized.lower()
            break
    # The greedy regex used in _extract_source_hints may capture leading words
    # like "please review file.docx"; the actual filename is always the final
    # whitespace-separated token ending in the extension.
    tokens = normalized.split()
    if len(tokens) > 1 and "." in tokens[-1]:
        normalized = tokens[-1]
    base_name = os.path.basename(normalized)
    return base_name.lower() if base_name else lowered


def _extract_source_hints(user_query: str) -> list[str]:
    hints = []
    for match in re.findall(r"[\w .#&()\-]+\.(?:pdf|png|jpe?g|txt|docx?|xlsx?|csv)", user_query, re.I):
        normalized = _normalize_source_hint(match)
        if normalized and normalized not in hints:
            hints.append(normalized)
    for match in re.findall(r"['\"]([^'\"]+\.(?:pdf|png|jpe?g|txt|docx?|xlsx?|csv))['\"]", user_query, re.I):
        normalized = _normalize_source_hint(match)
        if normalized and normalized not in hints:
            hints.append(normalized)
    return hints[:4]


def _line_matches_source_hints(evidence_line: str, source_hints: list[str]) -> bool:
    if not source_hints:
        return False
    lowered = evidence_line.lower()
    return any(hint in lowered or os.path.basename(hint) in lowered for hint in source_hints)


def _prioritize_source_hinted_evidence(evidence_lines: list[str], source_hints: list[str]) -> list[str]:
    if not evidence_lines or not source_hints:
        return evidence_lines
    hinted = [line for line in evidence_lines if _line_matches_source_hints(line, source_hints)]
    other = [line for line in evidence_lines if line not in hinted]
    return hinted + other


def _query_requests_refinement(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    refinement_markers = (
        "be more specific", "clarify", "correction", "dig deeper", "drill down",
        "explain that", "follow up", "go deeper", "instead", "narrow", "refine",
        "rephrase", "same evidence", "tell me more", "use that", "what about",
        "what does that mean", "why", "you missed",
    )
    return any(marker in lowered for marker in refinement_markers)


# Pronouns / anaphora that imply the user is still talking about the prior
# turn's subject without naming it (e.g. "how long did HE work THERE?").
# Matched with word boundaries so we never false-trigger on substrings like
# "the", "where", "then" inside other words. Only consulted when a prior turn
# exists, so a brand-new first question can't be mistaken for a follow-up.
_FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|there|this|those|these)\b"
)


# Lowercased words that are never, on their own, a "new subject" proper noun:
# question/lead words, conjunctions, articles, and the anaphoric pronouns. Used
# by `_names_new_subject` to avoid mistaking these for a topic switch.
_NON_SUBJECT_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "can", "could", "would", "should", "do", "does", "did", "is", "are",
    "was", "were", "will", "tell", "show", "find", "and", "but", "also",
    "additionally", "conversely", "however", "so", "then", "the", "a", "an",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "theirs", "there", "this", "those", "these", "it", "its", "that",
})


def _names_new_subject(user_query: str, last_turn: dict[str, object]) -> bool:
    """True if `user_query` introduces a Title-case proper-noun subject that does
    not appear in the prior turn — i.e. the user switched topics rather than
    continuing the previous one.

    Used to stop follow-up context from bleeding into an unrelated question. The
    first word is skipped (sentence-initial capitalization is not a reliable
    proper-noun signal), and anaphoric pronouns / lead/question words are never
    treated as subjects.
    """
    prior_text = " ".join(
        str(last_turn.get(key) or "") for key in ("user_query", "effective_query")
    ).lower()
    for idx, raw in enumerate(str(user_query or "").split()):
        if idx == 0:
            continue  # sentence-initial word is unreliable (verb/common noun)
        token = re.sub(r"'s$", "", raw.strip(".,;:!?'\"()[]"))
        if len(token) < 2 or not re.match(r"^[A-Z][a-z]", token):
            continue
        if token.lower() in _NON_SUBJECT_WORDS:
            continue
        if token.lower() not in prior_text:
            return True
    return False


def _looks_like_follow_up(user_query: str, last_turn: dict[str, object]) -> bool:
    if not last_turn.get("user_query"):
        return False
    lowered = " ".join(user_query.lower().split())
    if not lowered:
        return False
    follow_up_markers = (
        "that", "it", "its", "same", "previous", "earlier", "answer", "evidence",
        "relevant evidence", "file", "filename", "entitled", "document", "pdf",
        "you said", "you missed", "look at", "use the", "in the evidence",
        "clarify", "dig deeper", "drill down", "instead", "narrow", "refine",
        "tell me more", "what about", "what does that mean",
    )
    has_source_hint = bool(_extract_source_hints(user_query))
    wants_refinement = _query_requests_reasoning(user_query) or _query_requests_refinement(user_query)
    has_pronoun = bool(_FOLLOW_UP_PRONOUN_RE.search(lowered))
    has_marker = any(marker in lowered for marker in follow_up_markers)

    # An explicit source reference ("the answer is in X.pdf") is a deliberate
    # continuation and always stays a follow-up.
    if has_source_hint:
        return True

    # Topic-switch guard: if the user names a NEW proper-noun subject and does
    # NOT refer back with a pronoun, treat it as a fresh question so the prior
    # turn's context does not bleed in — even when a generic marker like
    # "what about"/"instead"/"it" is present (these double as refinement and as
    # topic-switch phrasing). A back-referring pronoun ("his GPA at Johns
    # Hopkins") keeps the thread because the subject is still continuous.
    if not has_pronoun and _names_new_subject(user_query, last_turn):
        return False

    return has_marker or has_pronoun or wants_refinement


def _is_negative_evidence_feedback(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    negative_markers = (
        "bad evidence", "does not answer", "doesn't answer", "has no reference",
        "has no references", "irrelevant", "not relevant", "no reference",
        "no references", "not in the evidence", "wrong evidence", "you provided has no",
    )
    return any(marker in lowered for marker in negative_markers)


def _query_requests_reasoning(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    reasoning_markers = (
        "based on", "could mean", "do you think", "hypothesis", "hypothesize",
        "implication", "implications", "imply", "implies", "infer", "inference", "likely", "may mean",
        "might mean", "should i", "tax wise", "tax-wise", "what do you think",
        "what would", "would be", "would happen",
    )
    conditional_reasoning = bool(re.search(r"\bif\b.+\b(then|would|could|might|may)\b", lowered))
    return conditional_reasoning or any(marker in lowered for marker in reasoning_markers)


def _effective_query_for_turn(user_query: str, last_turn: dict[str, object]) -> tuple[str, bool, list[str]]:
    source_hints = _extract_source_hints(user_query)
    if not _looks_like_follow_up(user_query, last_turn):
        return user_query, False, source_hints

    prior_query = str(last_turn.get("effective_query") or last_turn.get("user_query") or "").strip()
    if not prior_query:
        return user_query, False, source_hints

    lowered = " ".join(user_query.lower().split())
    asks_new_question = bool(re.search(r"\b(what|which|who|when|where|why|how|can|could|show|tell|find)\b", lowered))
    source_correction = bool(source_hints) or any(
        marker in lowered
        for marker in (
            "answer is in", "in the evidence", "entitled", "same evidence",
            "same file", "use the file", "look at the file",
        )
    )
    if source_correction and not asks_new_question:
        return prior_query, True, source_hints

    effective_query = f"{prior_query}\n{user_query}"
    return effective_query, True, source_hints


def _latest_user_turn(user_query: str) -> str:
    lines = [line.strip() for line in str(user_query or "").splitlines() if line.strip()]
    return lines[-1] if lines else str(user_query or "").strip()


def _is_refinement_turn(user_query: str) -> bool:
    return "\n" in str(user_query or "") or _query_requests_refinement(user_query)


def _evidence_lines_for_turn(
    evidence_lines: list[str],
    last_turn: dict[str, object],
    is_follow_up: bool,
    negative_evidence_feedback: bool,
    source_hints: list[str],
) -> list[str]:
    merged_evidence_lines = list(evidence_lines)
    if is_follow_up and not negative_evidence_feedback:
        prior_evidence_lines = [
            line
            for line in cast(list[object], last_turn.get("evidence_lines", []))
            if isinstance(line, str)
        ]
        for line in prior_evidence_lines:
            if line not in merged_evidence_lines:
                merged_evidence_lines.append(line)
    return _prioritize_source_hinted_evidence(merged_evidence_lines, source_hints)


def _review_answer_against_evidence(
    user_query: str,
    answer: str,
    evidence_lines: list[str],
    llm_client: Ollama,
) -> tuple[bool, str]:
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


def _query_requests_corpus_details(user_query: str) -> bool:
    text = " ".join(user_query.lower().split())
    detail_terms = ["details", "facts", "information", "what is present", "what's present"]
    corpus_terms = ["retrieved documents", "indexed documents", "documents", "data", "files"]
    return any(term in text for term in detail_terms) and any(term in text for term in corpus_terms)


def _query_requests_source_details(user_query: str) -> bool:
    if not _extract_source_hints(user_query):
        return False
    text = " ".join(user_query.lower().split())
    source_detail_markers = (
        "in the document", "in the file", "inside the document", "inside the file",
        "what is in", "what's in", "what does", "tell me what", "summarize",
        "summary", "contents", "about this document", "about the document",
    )
    return any(marker in text for marker in source_detail_markers)


def _query_terms(user_query: str) -> list[str]:
    stop_words = {
        "about", "all", "amount", "amounts", "and", "answer", "apply", "applies", "are", "can",
        "could", "detail", "details", "did", "do", "does", "during", "entitled", "evidence", "file",
        "filename", "find", "follow", "for", "from", "have", "how", "information", "is", "much", "present",
        "pdf", "png", "jpg", "jpeg", "txt", "doc", "docx", "xlsx", "csv",
        "provided", "references", "reference", "show", "tell", "that", "the", "their", "there",
        "these", "this", "user", "was", "were", "what", "which", "with", "you", "year",
        "based", "hypothesis", "hypothesize", "implication", "implications", "imply", "implies", "increase", "increased",
        "increases", "increasing", "infer", "inference", "into", "likely", "maybe", "might",
        "should", "taxwise", "think", "wise", "would",
        "pertain", "pertains", "pertaining",
    }
    terms = []
    normalized_query = re.sub(r"\btax\s+year\b", "year", user_query.lower())
    for term in re.findall(r"\b(?:\d{2,}|[a-zA-Z][a-zA-Z0-9-]{2,})\b", normalized_query):
        if term not in stop_words and term not in terms:
            terms.append(term)
    # Also preserve short all-uppercase acronyms from the original (case-sensitive) query
    # so 2-3 letter abbreviations like "EV", "IRS", "HSA", "TOD" are searchable.
    for acronym in re.findall(r"\b[A-Z]{2,5}\b", user_query):
        lowered = acronym.lower()
        if lowered in stop_words or lowered in terms:
            continue
        terms.append(lowered)
    return terms[:8]


def _query_anchor_terms(user_query: str, terms: list[str]) -> list[str]:
    question_words = {
        "Answer", "Can", "Could", "Evidence", "File", "Find", "How", "It",
        "Show", "Tell", "The", "This", "What", "Which", "You",
    }
    anchors = []

    for term in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", user_query):
        if term in question_words:
            continue
        lowered = term.lower()
        if lowered not in anchors:
            anchors.append(lowered)

    return anchors[:4]


def _query_phrase_anchors(user_query: str) -> list[str]:
    """Multi-word phrases from the query that should be preferred verbatim in evidence.

    Captures:
      * Runs of 2+ capitalized words (e.g. "Adjusted Gross Income", "Doctors Without Borders").
      * Quoted substrings (single or double quoted).
      * Short ALL-CAPS acronyms (>= 2 letters) — these often expand to a phrase in source documents.

    Returned phrases are lowercased and de-duplicated. Common question stems
    like "The" or "This" at the start are stripped.
    """
    phrases: list[str] = []
    leading_stop = {
        "the", "this", "that", "these", "those", "a", "an", "my", "our", "their", "your",
    }

    def _add(phrase: str) -> None:
        normalized = " ".join(phrase.lower().split())
        if not normalized:
            return
        parts = normalized.split()
        while parts and parts[0] in leading_stop:
            parts.pop(0)
        if len(parts) < 2:
            return
        cleaned = " ".join(parts)
        if cleaned not in phrases:
            phrases.append(cleaned)

    for match in re.finditer(r"(?:\b[A-Z][a-zA-Z]+\b\s+){1,5}\b[A-Z][a-zA-Z]+\b", user_query):
        _add(match.group(0))
    for match in re.finditer(r"['\"]([^'\"]{3,80})['\"]", user_query):
        _add(match.group(1))

    acronyms: list[str] = []
    for acronym in re.findall(r"\b[A-Z]{2,5}\b", user_query):
        lowered = acronym.lower()
        if lowered not in acronyms:
            acronyms.append(lowered)

    return phrases[:4] + acronyms[:4]


def _query_page_hints(user_query: str) -> list[int]:
    """Page numbers explicitly named in the query (e.g. 'page 8', 'on page 12')."""
    pages: list[int] = []
    for match in re.finditer(r"\bpage[s]?\s+(\d{1,5})\b", user_query, re.I):
        try:
            page = int(match.group(1))
        except ValueError:
            continue
        if page and page not in pages:
            pages.append(page)
    return pages[:4]


def _evidence_line_matches_phrase(evidence_body: str, phrase: str) -> bool:
    if not phrase:
        return False
    return phrase in evidence_body.lower()


def _evidence_line_matches_page(evidence_line: str, pages: list[int]) -> bool:
    if not pages:
        return False
    lowered = evidence_line.lower()
    return any(re.search(rf",\s*page\s+{page}\b", lowered) for page in pages)


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term in lowered for term in terms)


def _document_search_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    parts = [doc.text or ""]
    for key in ("file_name", "file_path", "parent_file", "source"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _query_requests_values(user_query: str) -> bool:
    return bool(re.search(
        r"\b(amounts?|balances?|costs?|deductions?|due|expenses?|how much|income|owe|owed|payments?|totals?|values?|withhold(?:ing)?)\b",
        user_query,
        re.I,
    ))


def _query_topic_terms(terms: list[str], anchor_terms: list[str]) -> list[str]:
    anchor_set = set(anchor_terms)
    return [term for term in terms if term not in anchor_set]


def _specific_query_terms(terms: list[str], anchor_terms: list[str]) -> list[str]:
    generic_terms = {
        "amount", "amounts", "balance", "balances", "due", "federal", "income",
        "owe", "owed", "payment", "payments", "plan", "plans", "state", "tax",
        "taxes", "total", "totals", "value", "values",
    }
    return [
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if term not in generic_terms and (any(char.isdigit() for char in term) or len(term) > 4)
    ]


def _term_family(term: str) -> list[str]:
    families = {
        "charitable": ["charit", "contribut", "donat", "gift"],
        "charities": ["charit", "contribut", "donat", "gift"],
        "charity": ["charit", "contribut", "donat", "gift"],
        "contribution": ["contribut", "charit", "donat", "gift"],
        "contributions": ["contribut", "charit", "donat", "gift"],
        "donated": ["donat", "contribut", "charit", "gift"],
        "donation": ["donat", "contribut", "charit", "gift"],
        "donations": ["donat", "contribut", "charit", "gift"],
        "due": ["due", "owe", "owed", "payment", "remittance", "debit", "withdraw", "balance due"],
        "federal": ["irs", "treasury", "1040", "u.s. individual income tax", "united states treasury"],
        "529": ["529", "maryland529", "qualified tuition", "qtp", "esa", "college 529", "state qtp"],
        "owed": ["owe", "owed", "amount you owe", "payment", "remittance", "debit", "withdraw", "balance due"],
        "owe": ["owe", "owed", "amount you owe", "payment", "remittance", "debit", "withdraw", "balance due"],
        "payment": ["payment", "payments", "remittance", "debit", "withdraw", "amount you owe"],
        "payments": ["payment", "payments", "remittance", "debit", "withdraw", "amount you owe"],
        "investment": ["investment", "invest", "portfolio", "account", "contribution", "contributions", "principal", "earnings"],
        "investments": ["investment", "invest", "portfolio", "account", "contribution", "contributions", "principal", "earnings"],
        "plan": ["plan", "program", "account", "qtp", "esa"],
        "state": ["maryland", "md", "comptroller", "form 502", "502/505", "local tax", "state tax", "state income"],
        "tax": ["tax", "taxes", "1040", "irs", "treasury"],
        "taxes": ["tax", "taxes", "1040", "irs", "treasury"],
        "total": ["total", "amount", "payment", "remittance", "balance", "line 37"],
        "totals": ["total", "amount", "payment", "remittance", "balance", "line 37"],
    }
    if term in families:
        return families[term]
    if len(term) > 5:
        return [term, term.rstrip("s")]
    return [term]


def _term_family_hit(text: str, term: str) -> bool:
    lowered = text.lower().replace("\\n", " ")
    if term == "federal":
        if any(alias in lowered for alias in _term_family(term)):
            return True
        without_bank_name = re.sub(r"\bnavy\s+federal(?:\s+credit\s+union)?\b", "", lowered)
        return bool(re.search(r"\bfederal\b", without_bank_name))
    for alias in _term_family(term):
        alias = alias.lower()
        if len(alias) <= 3 and re.fullmatch(r"[a-z0-9]+", alias):
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return True
        elif alias in lowered:
            return True
    return False


def _query_topic_coverage(text: str, terms: list[str], anchor_terms: list[str]) -> set[str]:
    lowered = text.lower()
    return {
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if _term_family_hit(lowered, term)
    }


PAYMENT_CONTEXT_MARKERS = (
    "amount due", "amount of tax payment", "amount you owe", "balance due",
    "balance owed", "debit", "direct debit", "due/-refund", "electronic funds withdrawal",
    "owed", "payment", "remittance", "tax balances", "withdraw", "withdrawal",
)


def _query_value_facets(terms: list[str], anchor_terms: list[str]) -> list[str]:
    value_intent_terms = {
        "amount", "amounts", "balance", "balances", "due", "owe", "owed", "payment",
        "payments", "tax", "taxes", "total", "totals", "value", "values",
    }
    facets = [
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if term not in value_intent_terms
    ]
    return facets or _query_topic_terms(terms, anchor_terms)


def _money_values(text: str) -> list[str]:
    values = []
    for value in re.findall(r"\$\s*\d[\d,]*(?:\.\d+)?", text):
        normalized = re.sub(r"\$\s*", "$", value.strip())
        if normalized not in values:
            values.append(normalized)
    return values


def _value_amounts(text: str, require_money: bool = False) -> list[str]:
    money_values = _money_values(text)
    if require_money:
        return money_values

    values = list(money_values)
    for value in _value_numbers(text):
        normalized = value.strip()
        if not normalized:
            continue
        if normalized.startswith("$"):
            normalized = re.sub(r"\$\s*", "$", normalized)
        if normalized not in values:
            values.append(normalized)
    return values


def _format_amount(value: str) -> str:
    value = value.strip()
    if value.startswith("$"):
        return value
    return f"${value}"


def _amount_segment_score(segment: str, facet: str, full_text_has_facet: bool) -> int:
    score = _payment_context_score(segment) * 20
    if _term_family_hit(segment, facet):
        score += 60
    elif full_text_has_facet:
        score += 4
    if re.search(r"\b(amount|balance|due|owe|owed|payment|refund|remittance|scheduled|withdraw|withdrawn)\b", segment, re.I):
        score += 12
    if _money_values(segment):
        score += 10
    if any("," in value for value in _value_amounts(segment)):
        score += 3
    return score


def _facet_row_amount_value(facet: str, text: str) -> str:
    amount_pattern = r"(?:\$\s*)?\d[\d,]*(?:\.\d+)?%?"
    if facet == "federal":
        label_pattern = r"federal"
        next_label_pattern = r"federal|md|maryland|state|total|due/-refund|breakout"
    elif facet == "state":
        label_pattern = r"md|maryland|state"
        next_label_pattern = r"federal|md|maryland|state|total|due/-refund|breakout"
    else:
        label_pattern = re.escape(facet)
        next_label_pattern = r"total|due/-refund|breakout"

    flattened = " ".join(text.replace("\\n", "\n").split())
    row_pattern = (
        rf"\b(?:{label_pattern})\b"
        rf"(?:\s+{amount_pattern}){{1,6}}"
        rf"(?=\s+(?:{next_label_pattern})\b|$)"
    )
    for match in re.finditer(row_pattern, flattened, re.I):
        row_money = _money_values(match.group(0))
        if row_money:
            return row_money[-1]
        row_values = _value_amounts(match.group(0))
        if row_values:
            return row_values[-1]
    return ""


def _facet_amount_value(facet: str, text: str, values: list[str]) -> str:
    row_value = _facet_row_amount_value(facet, text)
    if row_value:
        return row_value

    full_text_has_facet = _term_family_hit(text, facet)
    best_segment: tuple[int, list[str]] | None = None
    separators = r"\n|\\n|\||;|(?<=[.!?])\s+"
    for segment in re.split(separators, text):
        segment = segment.strip()
        if not segment:
            continue
        if facet == "federal":
            facet_hit = re.search(r"\bfederal\b", segment, re.I) or _term_family_hit(segment, facet)
        elif facet == "state":
            facet_hit = re.search(r"\b(?:md|maryland|state)\b", segment, re.I) or _term_family_hit(segment, facet)
        else:
            facet_hit = _term_family_hit(segment, facet)
        context_score = _payment_context_score(segment)
        if not facet_hit and not (full_text_has_facet and context_score > 0):
            continue
        segment_values = _value_amounts(segment)
        if segment_values:
            # Prefer money-formatted values ($1,234.56) over bare integers in the
            # same segment so that nearby identifiers like ZIP codes, invoice
            # numbers, or quantities don't masquerade as currency.
            money_only = [v for v in segment_values if v.startswith("$")]
            if money_only:
                segment_values = money_only
            score = _amount_segment_score(segment, facet, full_text_has_facet)
            if best_segment is None or score > best_segment[0]:
                best_segment = (score, segment_values)

    if best_segment:
        return best_segment[1][-1]

    money_fallback = [v for v in values if v.startswith("$")]
    if money_fallback:
        return money_fallback[-1]
    return values[-1] if values else ""


def _payment_context_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for marker in PAYMENT_CONTEXT_MARKERS if marker in lowered)


def _facet_label(facet: str, evidence_line: str) -> str:
    lowered = evidence_line.lower()
    if facet == "state" and "maryland" in lowered:
        return "State/Maryland"
    return facet.replace("-", " ").title()


def _source_label_from_evidence_line(evidence_line: str) -> str:
    compact = evidence_line[2:] if evidence_line.startswith("- ") else evidence_line
    return compact.split(": ", 1)[0].strip()


def _evidence_body_from_evidence_line(evidence_line: str) -> str:
    compact = evidence_line[2:] if evidence_line.startswith("- ") else evidence_line
    parts = compact.split(": ", 1)
    return parts[1] if len(parts) == 2 else compact


def _extractive_value_answer_from_evidence(user_query: str, evidence_lines: list[str]) -> str:
    if not evidence_lines or not _query_requests_values(user_query):
        return ""

    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    facets = _query_value_facets(terms, anchor_terms)
    if not facets:
        return ""

    phrase_anchors = _query_phrase_anchors(user_query)
    page_hints = _query_page_hints(user_query)

    best_by_facet: dict[str, tuple[int, str, list[str]]] = {}
    for line in evidence_lines:
        evidence_body = _evidence_body_from_evidence_line(line)
        values = _value_amounts(evidence_body)
        payment_score = _payment_context_score(evidence_body)
        if not values or payment_score == 0:
            continue
        coverage = _query_topic_coverage(evidence_body, terms, anchor_terms)
        # Phrase + page hint bonuses make exact-match evidence dominate over
        # generic value-dense lines that merely share a topic word.
        phrase_hit = any(_evidence_line_matches_phrase(evidence_body, phrase) for phrase in phrase_anchors)
        page_hit = _evidence_line_matches_page(line, page_hints)
        bonus = (200 if phrase_hit else 0) + (300 if page_hit else 0)
        for facet in facets:
            if facet not in coverage and not (phrase_hit or page_hit):
                continue
            score = payment_score * 20 + len(values) * 4 + bonus
            current = best_by_facet.get(facet)
            if current is None or score > current[0]:
                best_by_facet[facet] = (score, line, values)

    # Strict all-facets coverage only when the query enumerates multiple value buckets
    # (e.g., "federal and state amounts owed"). For other multi-term queries
    # (e.g., proper-noun org names like "Doctors without borders donation"),
    # returning whichever facet matched is more useful than refusing outright.
    enumerates_multiple = bool(re.search(r"\b(and|or)\b|,", user_query))
    if enumerates_multiple and len(facets) > 1 and any(facet not in best_by_facet for facet in facets):
        return ""
    if not best_by_facet:
        return ""

    parts = []
    for facet in facets:
        match = best_by_facet.get(facet)
        if not match:
            continue
        _, line, values = match
        source_label = _source_label_from_evidence_line(line)
        evidence_body = _evidence_body_from_evidence_line(line)
        answer_value = _facet_amount_value(facet, evidence_body, values)
        if answer_value:
            parts.append(f"{_facet_label(facet, line)}: {_format_amount(answer_value)} ({source_label})")

    if not parts:
        return ""
    return "From the retrieved evidence, " + "; ".join(parts) + "."


def _extractive_value_answer_from_documents(user_query: str, documents: list[Document], max_items: int = 12) -> str:
    if not documents or not _query_requests_values(user_query):
        return ""

    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    ranked: list[tuple[int, str]] = []

    for doc in documents:
        text = doc.text or ""
        if not text or is_low_information_text(text):
            continue

        metadata = doc.metadata or {}
        search_text = _document_search_text(doc)
        score = _query_candidate_score(search_text, terms, anchor_terms, require_value=True)
        if score <= 0:
            continue

        snippet = _query_focused_snippet(text, terms, anchor_terms, require_value=True, max_chars=4000)
        if not _value_amounts(snippet) or _payment_context_score(snippet) == 0:
            compact = " ".join(normalize_text(text).split())[:4000].rstrip()
            if _value_amounts(compact) and _payment_context_score(compact) > 0:
                snippet = compact
            else:
                continue

        coverage = _query_topic_coverage(snippet, terms, anchor_terms)
        if not coverage:
            continue

        file_name = os.path.basename(str(metadata.get("file_name") or metadata.get("file_path") or "unknown file"))
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"
        ranked.append((score + _payment_context_score(snippet) * 20, f"- {label}: {snippet}"))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return _extractive_value_answer_from_evidence(
        user_query,
        [line for _, line in ranked[:max_items]],
    )


def _prepare_retrieval_query(user_query: str) -> str:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    require_value = _query_requests_values(user_query)

    focus_terms = []
    for term in anchor_terms:
        focus_terms.append(term)
    for term in topic_terms:
        focus_terms.extend(_term_family(term))
    if require_value:
        focus_terms.extend([
            "amount", "amounts", "total", "totals", "value", "values", "dollars",
            "deduction", "payment", "remittance", "debit", "withdraw", "balance due",
        ])

    deduped_focus = []
    for term in focus_terms:
        normalized = term.strip().lower()
        if normalized and normalized not in deduped_focus:
            deduped_focus.append(normalized)

    if not deduped_focus:
        return " ".join(user_query.split())

    return (
        "Find indexed evidence for: "
        + " ".join(deduped_focus)
        + ". Original question: "
        + " ".join(user_query.split())
    )


def _retrieval_query_variants(user_query: str) -> list[str]:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    require_value = _query_requests_values(user_query)

    topic_family_terms = []
    for term in topic_terms:
        topic_family_terms.extend(_term_family(term))

    value_terms = [
        "amount", "amounts", "total", "totals", "value", "values", "dollars",
        "deduction", "payment", "remittance", "debit", "withdraw", "balance due",
    ] if require_value else []
    query_parts = [
        _prepare_retrieval_query(user_query),
        " ".join(user_query.split()),
        " ".join(anchor_terms + topic_family_terms + value_terms),
        " ".join(anchor_terms + topic_terms + value_terms),
    ]
    for topic_term in topic_terms:
        query_parts.append(" ".join(anchor_terms + _term_family(topic_term) + value_terms))
    query_parts.extend([
        " ".join(topic_family_terms + value_terms),
        " ".join(anchor_terms),
    ])

    variants = []
    for query in query_parts:
        normalized = " ".join(query.split())
        if normalized and normalized not in variants:
            variants.append(normalized)
    return variants


def _retrieve_query_supported_evidence(
    user_query: str,
    retriever: object,
) -> tuple[list[str], list[object]]:
    max_queries = max(1, int(os.environ.get("JARVIS_RETRIEVAL_ATTEMPTS", "20")))
    max_evidence = max(1, int(os.environ.get("JARVIS_MAX_EVIDENCE_LINES", "30")))
    all_nodes = []
    seen_node_ids = set()

    for attempt, retrieval_query in enumerate(_retrieval_query_variants(user_query)[:max_queries], start=1):
        debug_print(f"[debug] Retrieval attempt {attempt}: {retrieval_query}")
        try:
            nodes = retriever.retrieve(retrieval_query)
        except Exception as e:
            debug_print(f"[debug] Retrieval attempt {attempt} failed: {e}")
            continue

        for node in nodes:
            node_id = getattr(node, "node_id", None) or getattr(getattr(node, "node", None), "node_id", None) or id(node)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            all_nodes.append(node)

    # When the user explicitly names a file (e.g. "review Resume_2026.docx"),
    # pure vector ranking can miss it if the natural-language phrasing matches
    # other documents better. Force-include all chunks from the named file(s).
    source_hints = _extract_source_hints(user_query)
    if source_hints:
        forced_nodes = _force_include_source_hinted_nodes(
            retriever, source_hints, seen_node_ids
        )
        all_nodes.extend(forced_nodes)
        # Allow extra evidence slots so the forced chunks don't crowd everything else out.
        max_evidence = max(max_evidence, len(forced_nodes) + 4)

    evidence_lines = _source_nodes_to_query_evidence_lines(
        user_query,
        all_nodes,
        max_items=max_evidence,
        source_hints=source_hints,
    )
    return evidence_lines, all_nodes


def _force_include_source_hinted_nodes(
    retriever: object,
    source_hints: list[str],
    seen_node_ids: set,
) -> list[object]:
    """Pull every chunk whose file_name/file_path matches a source hint,
    bypassing vector ranking so user-named files always reach the LLM.

    llama-index `VectorIndexRetriever` exposes the docstore as both
    `retriever._docstore` and `retriever._index.docstore`; we try both.
    A hybrid `QueryFusionRetriever` exposes neither, so `build_hybrid_retriever`
    attaches the docstore as `_jarvis_docstore`; we check that first.
    """
    extra: list[object] = []
    docstore = (
        getattr(retriever, "_jarvis_docstore", None)
        or getattr(retriever, "_docstore", None)
        or getattr(getattr(retriever, "_index", None), "docstore", None)
        or getattr(getattr(retriever, "index", None), "docstore", None)
    )
    if docstore is None:
        print(
            f"[jarvis] source-hint force-include: no docstore on retriever "
            f"(type={type(retriever).__name__}); skipping"
        )
        return extra
    try:
        all_docs = docstore.docs or {}
    except Exception as e:
        print(f"[jarvis] source-hint force-include: docstore.docs access failed: {e}")
        return extra
    hint_basenames = {os.path.basename(h).lower() for h in source_hints if h}
    if not hint_basenames:
        return extra
    for node_id, node in all_docs.items():
        if node_id in seen_node_ids:
            continue
        metadata = getattr(node, "metadata", None) or {}
        fname = str(metadata.get("file_name") or "").lower()
        fpath = str(metadata.get("file_path") or "").lower()
        if not fname and not fpath:
            continue
        if any(h in fname or h in fpath for h in hint_basenames):
            seen_node_ids.add(node_id)
            extra.append(node)
    print(
        f"[jarvis] source-hint force-include: hints={sorted(hint_basenames)} "
        f"docstore_size={len(all_docs)} matched={len(extra)}"
    )
    return extra


def _hybrid_enabled() -> bool:
    """Hybrid (BM25 + dense) retrieval is the native runtime. Set
    JARVIS_HYBRID=0 to opt out and fall back to dense-only vector retrieval."""
    return os.environ.get("JARVIS_HYBRID", "1") != "0"


def build_hybrid_retriever(
    index: object,
    dense_top_k: int,
    bm25_top_k: Optional[int] = None,
    fusion_top_k: Optional[int] = None,
    llm: Optional[Any] = None,
) -> object:
    """Build Jarvis's default retriever: a hybrid fusion of BM25 (sparse
    lexical) and dense vector retrieval, merged with Reciprocal Rank Fusion.

    Hybrid is the native runtime because this corpus (resumes, contracts, PDFs)
    is full of exact tokens -- names, dates, dollar amounts, IDs, defined terms
    -- where embedding search underperforms. BM25 closes that recall gap at the
    candidate-generation stage while the dense path keeps semantic matching.

    Opt out with JARVIS_HYBRID=0 to use dense-only.

    Degrades gracefully: if the BM25 package is missing, the docstore is empty,
    or fusion construction fails for any reason, this logs the cause and returns
    the plain dense retriever so retrieval never hard-fails.
    """
    dense_retriever = index.as_retriever(similarity_top_k=dense_top_k)

    if not _hybrid_enabled():
        print("[jarvis] Hybrid retrieval disabled (JARVIS_HYBRID=0); using dense-only vector retrieval.")
        return dense_retriever

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
    except Exception as e:
        print(f"[jarvis] Hybrid retrieval unavailable (BM25/fusion import failed: {e}); using dense-only.")
        return dense_retriever

    # BM25 builds from the docstore (node text), which is loaded on every
    # startup independent of the Milvus embedding/reindex-skip path.
    docstore = (
        getattr(getattr(index, "storage_context", None), "docstore", None)
        or getattr(index, "docstore", None)
    )
    doc_count = 0
    if docstore is not None:
        try:
            doc_count = len(docstore.docs or {})
        except Exception:
            doc_count = 0
    if docstore is None or doc_count == 0:
        print(f"[jarvis] Hybrid retrieval: docstore empty or missing (docs={doc_count}); using dense-only.")
        return dense_retriever

    bm25_k = bm25_top_k or dense_top_k
    fusion_k = fusion_top_k or dense_top_k
    try:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=bm25_k,
        )
        fusion = QueryFusionRetriever(
            [dense_retriever, bm25_retriever],
            similarity_top_k=fusion_k,
            # Upstream _retrieve_query_supported_evidence already expands the
            # query into many variants; num_queries=1 avoids an extra LLM call
            # inside the fusion retriever (keeps the hot path LLM-free).
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=False,
            # Pass the local LLM explicitly so the constructor never resolves a
            # default OpenAI client (it isn't used at query time when num_queries=1).
            llm=llm,
        )
        # Expose the docstore so _force_include_source_hinted_nodes still works
        # through the fusion retriever (which has no _index/_docstore attr).
        try:
            fusion._jarvis_docstore = docstore
        except Exception:
            pass
        print(
            f"[jarvis] Hybrid retrieval active: BM25(top_k={bm25_k}) + "
            f"dense(top_k={dense_top_k}) -> RRF(top_k={fusion_k}) over {doc_count} chunks."
        )
        return fusion
    except Exception as e:
        print(f"[jarvis] Hybrid retrieval construction failed ({e}); using dense-only.")
        return dense_retriever


def _query_candidate_score(
    text: str,
    terms: list[str],
    anchor_terms: list[str],
    require_value: bool,
    require_anchors: bool = True,
) -> int:
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0

    lowered = compact.lower()
    anchor_hits = sum(1 for term in anchor_terms if term in lowered)
    if require_anchors and anchor_terms and anchor_hits == 0:
        return 0

    topic_terms = _query_topic_terms(terms, anchor_terms)
    topic_hits = sum(1 for term in topic_terms if _term_family_hit(lowered, term))
    specific_terms = _specific_query_terms(terms, anchor_terms)
    specific_hits = sum(1 for term in specific_terms if _term_family_hit(lowered, term))
    value_hits = len(set(_value_numbers(compact)))
    payment_context_hits = _payment_context_score(lowered) if require_value else 0

    if require_value and value_hits == 0:
        return 0
    if topic_terms and topic_hits == 0 and anchor_hits == 0:
        return 0
    if specific_terms and specific_hits == 0 and anchor_hits == 0:
        return 0
    if not topic_terms and not anchor_hits:
        return 0

    return (
        anchor_hits * 30
        + topic_hits * 20
        + min(payment_context_hits, 6) * 18
        + min(value_hits, 8) * 6
        + min(len(compact) // 180, 6)
    )


def _query_focused_snippet(text: str, terms: list[str], anchor_terms: list[str], require_value: bool, max_chars: int = 2000) -> str:
    candidates = _detail_candidates(text)
    scored = [
        (_query_candidate_score(candidate, terms, anchor_terms, require_value, require_anchors=False), candidate)
        for candidate in candidates
    ]
    best_score, best_text = max(scored, key=lambda item: item[0]) if scored else (0, "")
    if best_score == 0:
        best_text = " ".join(normalize_text(text).split())

    snippet = " ".join(best_text.split())[:max_chars].rstrip()
    if len(best_text) > max_chars:
        snippet += "..."
    return snippet


def _query_evidence_is_supported(snippet: str, terms: list[str], anchor_terms: list[str], require_value: bool) -> bool:
    lowered = snippet.lower()
    anchor_hits = sum(1 for term in anchor_terms if term in lowered)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    topic_hit = any(_term_family_hit(lowered, term) for term in topic_terms) if topic_terms else False
    specific_terms = _specific_query_terms(terms, anchor_terms)
    specific_hit = any(_term_family_hit(lowered, term) for term in specific_terms) if specific_terms else False

    if anchor_terms and anchor_hits == 0 and not topic_hit:
        return False
    if topic_terms and not topic_hit and anchor_hits == 0:
        return False
    if specific_terms and not specific_hit and anchor_hits == 0 and not topic_hit:
        return False
    if require_value and not _value_numbers(snippet):
        return False
    return True


def _reasoning_guidance(user_query: str) -> str:
    refinement_instruction = ""
    if _is_refinement_turn(user_query):
        refinement_instruction = (
            "This is a continuing conversation. Treat the final user line as the latest clarification. "
            "Use earlier lines only as context, and refine the answer instead of repeating the prior lookup. "
        )

    if not _query_requests_reasoning(user_query):
        return (
            refinement_instruction +
            "Answer the user's question using only the evidence below. "
            "Write a cohesive summary, not a raw list of unrelated excerpts. "
            "Use concrete values, names, dates, and sources when present. "
            "If the evidence does not contain enough information to answer, say exactly what is missing. "
            "Do not add facts from outside the evidence."
        )

    return (
        refinement_instruction +
        "Answer the user's question by reasoning from only the evidence below. "
        "First state the retrieved facts that matter, then state cautious implications or hypotheses. "
        "Use phrases like 'the evidence suggests', 'a reasonable implication is', or 'I cannot determine' where appropriate. "
        "Do not present outside rules, professional advice, projections, or guarantees unless those rules or values are explicitly in the evidence. "
        "If the evidence supports directionally useful reasoning but not a precise result, say so plainly."
    )


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


def _extractive_summary_answer_from_evidence(user_query: str, evidence_lines: list[str], max_items: int = 5) -> str:
    if not evidence_lines or _query_requests_values(user_query) or _query_requests_reasoning(user_query):
        return ""

    latest_turn = _latest_user_turn(user_query)
    if _is_refinement_turn(user_query):
        lines = [f"Refining that for your latest question ({latest_turn}), here is what the indexed evidence supports:"]
    else:
        lines = ["Here is what I found in the indexed evidence:"]
    for evidence_line in evidence_lines[:max_items]:
        source_label = _source_label_from_evidence_line(evidence_line)
        evidence_body = _evidence_body_from_evidence_line(evidence_line)
        snippet = " ".join(evidence_body.split())[:1600].rstrip()
        if len(evidence_body) > 1600:
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
    lines = ["Here is the closest grounded evidence I found in the indexed documents:"]
    for evidence_line in evidence_lines[:max_items]:
        source_label = _source_label_from_evidence_line(evidence_line)
        evidence_body = _evidence_body_from_evidence_line(evidence_line)
        snippet = " ".join(evidence_body.split())[:1600].rstrip()
        if len(evidence_body) > 1600:
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
            with _llm_output_budget(llm_client, -1):
                response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=p)])
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


@contextmanager
def _llm_output_budget(llm_client: Ollama, num_predict: int):
    """Temporarily override the Ollama `num_predict` for a call.

    Pass a positive integer to raise the cap to at least that many tokens.
    Pass -1 to fully uncap generation (model stops on its own EOS token);
    this is the right choice for synthesis and distillation on a local LLM
    where we care more about answer completeness than token cost. The prior
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


def _synthesize_evidence_answer(
    user_query: str,
    evidence_lines: list[str],
    llm_client: Ollama,
    review_feedback: str = "",
    prior_answer: str = "",
) -> str:
    if not evidence_lines:
        return ""

    refinement_block = ""
    latest_turn_text = _latest_user_turn(user_query)
    latest_turn_is_reasoning = _query_requests_reasoning(latest_turn_text)
    if prior_answer and latest_turn_is_reasoning:
        # The latest turn asks for reasoning/implications. Do NOT anchor the
        # model to the prior value-shaped answer; that causes it to restate the
        # dollar amount instead of reasoning. Instead, explicitly forbid
        # recycling the prior answer.
        refinement_block = (
            "This is a follow-up turn. The user's PREVIOUS turn asked for a value/fact and was answered. "
            "The user's LATEST turn asks a DIFFERENT KIND of question (reasoning / implications / inference).\n"
            "- Do NOT restate the prior value answer as if it answers the new question.\n"
            "- Address the latest turn directly: state what the evidence DOES say that is relevant to the new question, "
            "then state cautious implications grounded in that evidence, then plainly note what the evidence does NOT establish.\n"
            "- If the evidence is silent on the new question (e.g., contains a registration amount but no tax-treatment language), "
            "say so explicitly and explain the limit of what can be inferred from what IS present.\n\n"
            f"Prior value answer (for context only, do NOT repeat verbatim):\n{prior_answer}\n\n"
        )
    elif prior_answer:
        refinement_block = (
            "This is a follow-up turn. Treat the user's latest turn as human feedback that should REFINE "
            "your previous answer, not restart from scratch.\n"
            "- Keep what still holds from the previous answer ONLY if it is still supported by the evidence below.\n"
            "- Directly address the new clarification, question, or correction.\n"
            "- If the requested detail is not in the evidence, say so plainly and offer the closest grounded reading.\n"
            "- Do not simply repeat the previous answer or dump raw snippets.\n\n"
            f"Previous answer to refine:\n{prior_answer}\n\n"
        )

    grounding_contract = (
        "STRICT GROUNDING CONTRACT (do not violate):\n"
        "- Use ONLY facts that appear in the Evidence section below (retrieved from the indexed Milvus vector store and local document manifest).\n"
        "- Do NOT use outside knowledge, prior training data, general tax/legal/financial rules, or assumptions.\n"
        "- Every concrete claim, number, date, name, or rule MUST be traceable to a specific evidence line. Cite the source label in parentheses when stating it.\n"
        "- Use whatever IS present in the evidence to answer as completely as possible. Partial evidence is still useful — extract names, amounts, line items, dates, and forms that match the question even if the exact phrasing differs.\n"
        "- Only after you have surfaced the relevant evidence should you note any specific sub-detail that the evidence does not contain. Never refuse outright when related evidence is present.\n"
        "- Cautious inference is allowed ONLY when the user explicitly asked for reasoning/implications, and must be labeled as inference from the cited evidence.\n\n"
    )

    prompt = (
        _reasoning_guidance(user_query)
        + "\n\n"
        + grounding_contract
        + refinement_block
        + f"Question: {user_query}\n\n"
        f"Latest user turn to answer: {_latest_user_turn(user_query)}\n\n"
        + (f"Prior review feedback to fix: {review_feedback}\n\n" if review_feedback else "")
        +
        "Evidence (the ONLY allowed source of facts):\n"
        + "\n".join(evidence_lines)
    )
    try:
        with _llm_output_budget(llm_client, -1):
            response = llm_client.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        answer = (response.message.content or "").strip()
        if not answer:
            debug_print("[debug] Synthesis LLM returned empty completion despite uncapped num_predict.")
        return answer
    except Exception as exc:
        debug_print(f"[debug] Synthesis LLM call failed: {exc}")
        return ""


def _answer_with_review_loop(
    user_query: str,
    evidence_lines: list[str],
    llm_client: Ollama,
    prior_answer: str = "",
) -> tuple[str, bool, str]:
    max_attempts = max(1, int(os.environ.get("JARVIS_ANSWER_REVIEW_ATTEMPTS", "3")))
    review_feedback = ""
    last_review_reason = ""
    best_answer = ""
    # When the user is asking for reasoning/implications, the value-shaped
    # fallbacks (distillation, value extractor) will recycle a dollar figure
    # instead of producing an inference. Gate those off so reasoning queries
    # route to the reasoning extractor or surface the unverified LLM answer
    # with a caveat instead.
    reasoning_requested = _query_requests_reasoning(user_query)

    for attempt in range(1, max_attempts + 1):
        answer = _synthesize_evidence_answer(
            user_query,
            evidence_lines,
            llm_client,
            review_feedback=review_feedback,
            prior_answer=prior_answer,
        )
        if not answer:
            fallback_answer = _extractive_reasoning_answer_from_evidence(user_query, evidence_lines)
            if fallback_answer:
                debug_print(f"[debug] Used extractive reasoning fallback on attempt {attempt}.")
                return fallback_answer, True, "answered with extractive reasoning fallback"
            fallback_answer = _extractive_summary_answer_from_evidence(user_query, evidence_lines)
            if fallback_answer:
                debug_print(f"[debug] Used extractive summary fallback on attempt {attempt}.")
                return fallback_answer, True, "answered with extractive evidence summary fallback"
            distilled = _distill_answer_from_evidence(user_query, evidence_lines, llm_client)
            if distilled:
                debug_print(f"[debug] Used LLM distillation fallback on attempt {attempt}.")
                return distilled, True, "answered with LLM-distilled evidence fallback"
            fallback_answer = _forced_extractive_summary_from_evidence(user_query, evidence_lines)
            if fallback_answer:
                debug_print(f"[debug] Used forced extractive summary fallback on attempt {attempt}.")
                return fallback_answer, True, "answered with forced extractive evidence summary fallback"
            last_review_reason = "answer synthesis returned no text"
            break

        if not best_answer:
            best_answer = answer

        review_passed, review_reason = _review_answer_against_evidence(
            user_query,
            answer,
            evidence_lines,
            llm_client,
        )
        if review_passed:
            debug_print(f"[debug] Answer review passed on attempt {attempt}.")
            return answer, True, review_reason

        debug_print(f"[debug] Answer review failed on attempt {attempt}: {review_reason}")
        last_review_reason = review_reason
        review_feedback = review_reason
        best_answer = answer

    if best_answer:
        hallucination_markers = ("hallucinat", "outside", "not present", "not in the evidence", "invent", "external", "made up", "fabricat")
        reason_lower = last_review_reason.lower()
        looks_hallucinated = any(marker in reason_lower for marker in hallucination_markers)
        if looks_hallucinated:
            debug_print("[debug] Best LLM answer was flagged as hallucinated; falling back to extractive evidence summary.")
            if reasoning_requested:
                extractive = (
                    _extractive_reasoning_answer_from_evidence(user_query, evidence_lines)
                    or _extractive_summary_answer_from_evidence(user_query, evidence_lines)
                    or _forced_extractive_summary_from_evidence(user_query, evidence_lines)
                )
            else:
                extractive = (
                    _extractive_value_answer_from_evidence(user_query, evidence_lines)
                    or _extractive_reasoning_answer_from_evidence(user_query, evidence_lines)
                    or _extractive_summary_answer_from_evidence(user_query, evidence_lines)
                    or _distill_answer_from_evidence(user_query, evidence_lines, llm_client)
                    or _forced_extractive_summary_from_evidence(user_query, evidence_lines)
                )
            if extractive:
                return extractive, True, f"replaced ungrounded LLM answer with extractive evidence: {last_review_reason}"
            return "", False, last_review_reason
        # Review failed for non-hallucination reasons (generic, off-topic, missing value).
        # Prefer a strict evidence-only distillation over surfacing the unreviewed best_answer.
        # For reasoning queries, distillation is value-shaped and would recycle a number;
        # try the reasoning extractor first instead.
        if reasoning_requested:
            reasoning_answer = _extractive_reasoning_answer_from_evidence(user_query, evidence_lines)
            if reasoning_answer:
                debug_print("[debug] Used extractive reasoning answer instead of value-shaped distillation.")
                return reasoning_answer, True, f"reasoning extractor (review failed): {last_review_reason}"
        distilled = _distill_answer_from_evidence(user_query, evidence_lines, llm_client)
        if distilled:
            debug_print("[debug] Used LLM-distilled evidence answer instead of unverified best answer.")
            return distilled, True, f"distilled answer (review failed): {last_review_reason}"
        debug_print("[debug] All review attempts failed and distillation empty; returning best LLM reasoning with caveat.")
        caveat = (
            "Note: the reviewer was not fully satisfied this answer is strictly grounded in the indexed evidence; "
            "treat the reasoning below as my best read of the retrieved snippets and verify against the cited sources."
        )
        composed = f"{best_answer}\n\n{caveat}"
        return composed, True, f"unverified reasoning surfaced: {last_review_reason}"

    # Final unconditional safety net: prefer an LLM-distilled succinct answer
    # over a raw snippet dump. The distillation prompt is strict evidence-only
    # and only re-asks the model to pick the value(s) that answer the question.
    # Reasoning queries skip distillation + value extractor (both value-shaped)
    # and use the reasoning extractor as their last-chance grounded answer.
    if reasoning_requested:
        reasoning_answer = _extractive_reasoning_answer_from_evidence(user_query, evidence_lines)
        if reasoning_answer:
            debug_print("[debug] Returning extractive reasoning answer as final safety net.")
            return reasoning_answer, True, f"reasoning extractor (review never passed): {last_review_reason or 'no answer produced'}"
    else:
        distilled = _distill_answer_from_evidence(user_query, evidence_lines, llm_client)
        if distilled:
            debug_print("[debug] Returning LLM-distilled evidence answer as final safety net.")
            return distilled, True, f"distilled answer (review never passed): {last_review_reason or 'no answer produced'}"
        # Heuristic value extractor (with phrase/page-hint bonuses) as a last
        # resort BEFORE dumping raw snippets, so value-intent queries still get a
        # concrete "X: $N (source)" answer even when the LLM produced no text.
        value_answer = _extractive_value_answer_from_evidence(user_query, evidence_lines)
        if value_answer:
            debug_print("[debug] Returning heuristic value extractor answer as final safety net.")
            return value_answer, True, f"heuristic value extractor (LLM empty): {last_review_reason or 'no answer produced'}"
    forced = _forced_extractive_summary_from_evidence(user_query, evidence_lines)
    if forced:
        print(
            "[jarvis] The LLM returned no synthesized answer for this query; "
            "showing the matching evidence snippets so you can read them directly."
        )
        debug_print("[debug] Returning forced extractive summary as final safety net.")
        return forced, True, f"forced extractive summary (review never passed): {last_review_reason or 'no answer produced'}"

    return "", False, last_review_reason or "answer synthesis failed"


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
    # Out-of-corpus fallback: optionally escalate this single call to the larger
    # router model (JARVIS_ROUTER=1). No-op passthrough when the router is off.
    llm_client = route_general_knowledge_llm(llm_client)
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
    lines.extend(evidence_lines[:6])
    if reason and JARVIS_DEBUG:
        lines.append(f"Review reason: {reason}")
    return "\n".join(lines)


# Inline parenthetical source citations, e.g. "(Resume_2026.docx)",
# "(report.pdf, page 3)", or "(a.docx, b.docx, c.docx)". Matched groups contain
# at least one filename-with-extension or a page reference. Used to clean spoken
# text only; printed/displayed output keeps citations untouched.
# The extension set mirrors every file type Jarvis can ingest (documents +
# images); add new extensions here if the ingest list in Jarvis.py grows.
_CITATION_FILE_EXT = (
    r"pdf|docx?|xlsx?|pptx?|csv|txt|rtf"          # documents
    r"|png|jpe?g|tiff?|bmp|gif|webp|heic"          # images
)
_INLINE_CITATION_RE = re.compile(
    r"\s*\([^()\n]*?"
    r"(?:\.(?:" + _CITATION_FILE_EXT + r")\b"
    r"|\bpages?\s*\d+|\bpp?\.\s*\d+)"
    r"[^()\n]*?\)",
    re.IGNORECASE,
)


def _strip_inline_citations(text: str) -> str:
    """Remove inline parenthetical source citations from spoken text.

    Drops groups like "(Resume_2026.docx)", "(report.pdf, page 3)", or
    "(a.docx, b.docx, c.docx)" so the TTS engine reads the prose answer without
    spelling out filenames and extensions ("Resume 2026 dot docx").

    Only the spoken string is cleaned; printed/displayed output is unaffected.
    The factual sentence content outside the parentheses is preserved, so the
    answer is never truncated. Parentheticals that are NOT citations (no file
    extension or page reference) are left in place.
    """
    if not text:
        return text
    stripped = _INLINE_CITATION_RE.sub("", text)
    # Tidy artifacts left by removal: space before punctuation and space runs.
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return stripped


# Runs of "the document" joined by commas/"and" that should collapse to the
# plural "the documents" after bare filenames are neutralized for speech.
_SPEECH_DOC_RUN_RE = re.compile(
    r"the document(?:\s*(?:,|and|, and)\s*the document)+",
    re.IGNORECASE,
)


def _strip_source_filenames_for_speech(text: str) -> str:
    """Neutralize bare in-prose source filenames so TTS never reads them aloud.

    `_strip_inline_citations` already removes fully parenthetical citations like
    "(report.pdf, page 3)". This handles filenames the model weaves directly
    into sentence grammar — as an object ("listed in Application.pdf") or even
    the subject ("JOB APPLICATION.pdf lists 3.65") — where simply deleting the
    token would leave a broken spoken sentence. Each bare filename (with any
    trailing page reference) is replaced by the neutral phrase "the document",
    and consecutive replacements joined by commas/"and" collapse to "the
    documents". Spoken path only; printed output keeps the real filenames and
    the dedicated 'References' block. Casing is irrelevant because the result is
    fed to the speech engine, never displayed."""
    if not text:
        return text
    spoken = _SOURCE_REFERENCE_RE.sub("the document", text)
    spoken = _SPEECH_DOC_RUN_RE.sub("the documents", spoken)
    # Tidy artifacts: space before punctuation and collapsed space runs.
    spoken = re.sub(r"\s+([.,;:!?])", r"\1", spoken)
    spoken = re.sub(r"[ \t]{2,}", " ", spoken)
    return spoken


# --- Spoken-text abbreviation / numeral normalization -----------------------
# Kokoro (and the espeak/misaki G2P front-end) has no semantic understanding of
# context-dependent abbreviations: it cannot tell "Dr. Smith" (Doctor) from
# "Elm Dr." (Drive), or "Henry III" (the Third) from "Section III" (three).
# That disambiguation must happen upstream, so we expand the common cases here
# with lightweight, context-aware rules BEFORE the text reaches the TTS engine.
# This runs on the spoken path only; printed/displayed output is unaffected.

# Roman-numeral name suffixes -> ordinal words (e.g. "Henry III" -> "the Third").
_ROMAN_NAME_ORDINALS = {
    "II": "the Second",
    "III": "the Third",
    "IV": "the Fourth",
    "V": "the Fifth",
    "VI": "the Sixth",
    "VII": "the Seventh",
    "VIII": "the Eighth",
}

# Words commonly followed by a Roman numeral that are NOT personal names, so we
# must NOT rewrite "Section III" -> "Section the Third" or "World War II" ->
# "World War the Second".
_NON_NAME_BEFORE_ROMAN = frozenset({
    "section", "chapter", "part", "volume", "article", "phase", "type",
    "class", "appendix", "act", "book", "title", "figure", "table", "war",
    "level", "grade", "stage", "group", "round", "episode", "step", "tier",
})

# Spoken word for each digit, used to read ZIP codes one digit at a time.
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# US state / territory postal abbreviations. A 5-digit number immediately
# following one of these is treated as a ZIP code (read digit by digit).
_US_STATE_ABBR = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
})


def _spell_digits(digits: str) -> str:
    """Render a run of digits as space-separated number words, e.g.
    "98052" -> "nine eight zero five two" (read one digit at a time)."""
    return " ".join(_DIGIT_WORDS[d] for d in digits if d in _DIGIT_WORDS)


# Rate units that turn a "/" into the spoken word "per" (e.g. "$7,500/month" ->
# "7500 dollars per month"). Scoped to known denominators so ordinary slashes
# (dates like "12/25", "and/or", "TCP/IP") are left untouched.
_PER_UNITS = (
    "month", "months", "mo", "year", "years", "yr", "yrs", "annum",
    "week", "weeks", "wk", "wks", "day", "days", "hour", "hours", "hr", "hrs",
    "minute", "minutes", "min", "mins", "second", "seconds", "sec",
    "quarter", "quarters", "capita", "person", "unit", "units", "share",
    "shares", "mile", "miles", "gallon", "gallons", "acre", "acres",
)
_PER_SLASH_RE = re.compile(
    r"\s*/\s*(?=(?:" + "|".join(_PER_UNITS) + r")\b)",
    re.IGNORECASE,
)


def _normalize_for_speech(text: str) -> str:
    """Expand context-dependent abbreviations and numerals for natural speech.

    Handles the cases TTS gets wrong on its own:
      * Titles before a name:  "Dr./Mr./Mrs./Ms./Prof./St. <Name>" -> spoken word
      * Address abbreviations: remaining "Dr." -> "Drive", "St." -> "Street"
      * Generational suffixes: "Jr."/"J-R" -> "Junior", "Sr."/"S-R" -> "Senior"
      * Name numerals:         "<Name> III"/"I-I-I" -> "<Name> the Third"

    Rules are case-sensitive where it matters (e.g. "Ms" -> "Miss" but "MS",
    which may mean Microsoft or Missouri, is left untouched). Only the spoken
    string is affected; printed output keeps the original text.
    """
    if not text:
        return text
    s = text

    # 1) Honorific / title abbreviations that are FOLLOWED by a capitalized name
    #    read as titles. The name itself is preserved via a zero-width lookahead.
    title_map = (
        (r"Dr\.", "Doctor"),
        (r"Mrs\.", "Missus"),
        (r"Mr\.", "Mister"),
        (r"Ms\.?", "Miss"),
        (r"Prof\.", "Professor"),
        (r"St\.", "Saint"),     # e.g. "St. Louis" -> "Saint Louis"
    )
    for abbr, full in title_map:
        s = re.sub(rf"\b{abbr}\s+(?=[A-Z][a-z])", full + " ", s)

    # 2) Remaining "Dr." / "St." are address abbreviations (not before a name).
    s = re.sub(r"\bDr\.", "Drive", s)
    s = re.sub(r"\bSt\.", "Street", s)

    # 3) Generational suffixes (period-terminated form first so the standalone
    #    boundary form does not re-match the same token).
    s = re.sub(r"\bJr\.", "Junior", s)
    s = re.sub(r"\bJr\b", "Junior", s)
    s = re.sub(r"\bJ-R\b", "Junior", s)
    s = re.sub(r"\bSr\.", "Senior", s)
    s = re.sub(r"\bSr\b", "Senior", s)
    s = re.sub(r"\bS-R\b", "Senior", s)

    # 4) Roman-numeral name suffixes -> ordinal words, skipping structural words
    #    (Section, World War, Type, ...) that are not personal names.
    def _roman_repl(match: "re.Match[str]") -> str:
        prefix_word = match.group(1)
        roman = match.group(2).upper().replace("-", "").replace(".", "")
        if prefix_word.lower() in _NON_NAME_BEFORE_ROMAN:
            return match.group(0)
        ordinal = _ROMAN_NAME_ORDINALS.get(roman)
        if not ordinal:
            return match.group(0)
        return f"{prefix_word} {ordinal}"

    s = re.sub(
        r"\b([A-Za-z]+)\s+"
        r"((?:I-I-I|I-I|I\.I\.I|VIII|VII|VI|IV|III|II|V))\b",
        _roman_repl,
        s,
    )

    # 5) ZIP codes in addresses -> read digit by digit. Only fire on a strong
    #    address signal so ordinary 5-digit numbers (years, counts, IDs) are
    #    left alone: (a) ZIP+4 form "12345-6789", or (b) a 5-digit number
    #    immediately following a US state abbreviation ("WA 98052").
    def _zip4_repl(match: "re.Match[str]") -> str:
        return f"{_spell_digits(match.group(1))} {_spell_digits(match.group(2))}"

    s = re.sub(r"\b(\d{5})-(\d{4})\b", _zip4_repl, s)

    def _state_zip_repl(match: "re.Match[str]") -> str:
        state, zip5, plus4 = match.group(1), match.group(2), match.group(3)
        if state not in _US_STATE_ABBR:
            return match.group(0)
        spoken = _spell_digits(zip5)
        if plus4:
            spoken = f"{spoken} {_spell_digits(plus4)}"
        return f"{state} {spoken}"

    s = re.sub(r"\b([A-Z]{2})\s+(\d{5})(?:-(\d{4}))?\b", _state_zip_repl, s)

    # 6) Rate slash -> "per" before a known unit, e.g. "$7,500/month" ->
    #    "$7,500 per month". Dates ("12/25"), "and/or", "TCP/IP" etc. are
    #    untouched because their right-hand side is not a rate unit.
    s = _PER_SLASH_RE.sub(" per ", s)
    return s


def _speech_text_for_response(response_text: str) -> str:
    if not response_text:
        return ""
    evidence_heading = "Relevant evidence found:"
    if evidence_heading in response_text:
        return response_text.split(evidence_heading, 1)[0].strip()
    return response_text


def _summarize_for_speech(response_text: str, max_sentences: int = 2, max_chars: int = 280) -> str:
    """Produce a concise spoken summary: drop evidence blocks, citation/reference
    lines, and long enumerations; keep at most `max_sentences` sentences and
    `max_chars` characters so TTS stays brief."""
    spoken_body = _speech_text_for_response(response_text)
    if not spoken_body:
        return ""

    citation_prefixes = (
        "source:", "sources:", "reference:", "references:",
        "evidence:", "from:", "see:", "file:", "page:",
    )
    kept_lines: list[str] = []
    for raw_line in spoken_body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(citation_prefixes):
            continue
        # Skip bullet/numbered enumeration lines (kept in printed output, not spoken).
        if line[:2] in ("- ", "* ") or (len(line) > 2 and line[0].isdigit() and line[1] in ".)"):
            continue
        # Skip parenthetical-only file/page citations like "(file.pdf, p.3)".
        if line.startswith("(") and line.endswith(")") and ("." in line or "p." in lowered):
            continue
        kept_lines.append(line)

    cleaned = " ".join(kept_lines).strip()
    if not cleaned:
        # Nothing left after stripping references — fall back to first non-empty line of original.
        for raw_line in spoken_body.splitlines():
            if raw_line.strip():
                cleaned = raw_line.strip()
                break

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s]
    if not sentences:
        return ""
    # Always include the first sentence whole, even if it exceeds max_chars,
    # so the spoken summary never ends mid-word with an ellipsis.
    chosen = [sentences[0]]
    running = len(sentences[0])
    for nxt in sentences[1:max_sentences]:
        projected = running + 1 + len(nxt)
        if projected > max_chars:
            break
        chosen.append(nxt)
        running = projected
    return " ".join(chosen).strip()


def _spoken_response_text(response_text: str) -> str:
    """Return the FULL prose of a response for TTS, dropping only the
    'Relevant evidence found:' block and reference/citation metadata lines.
    Unlike `_summarize_for_speech`, this keeps the entire answer — including
    list/bullet content (with the bullet marker removed) and all sentences —
    so the listener hears the complete response, not just a two-sentence
    preview."""
    if not response_text:
        return ""
    text = _speech_text_for_response(response_text)

    citation_prefixes = (
        "source:", "sources:", "reference:", "references:",
        "evidence:", "from:", "see:", "file:", "page:",
    )

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            cleaned_lines.append("")  # preserve paragraph breaks for prosody
            continue
        lowered = line.lower().lstrip()
        if lowered.startswith(citation_prefixes):
            continue
        stripped = line.strip()
        # Drop parenthetical-only file/page citation lines like "(file.pdf, p.3)".
        if (
            stripped.startswith("(")
            and stripped.endswith(")")
            and ("." in stripped or "p." in lowered)
        ):
            continue
        # Strip leading bullet/list markers but keep the content.
        line = re.sub(r"^\s*[\-\*\u2022]+\s*", "", line)
        line = re.sub(r"^\s*\d+[\.\)]\s+", "", line)
        # Strip markdown emphasis markers — they are noise for TTS.
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"__([^_]+)__", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        # Remove inline file/page citations so TTS doesn't read out filenames
        # like "Resume_2026.docx" as "Resume 2026 dot docx".
        # The printed answer (and its References block) is produced elsewhere
        # and keeps the citations intact.
        line = _strip_inline_citations(line)
        cleaned_lines.append(line)

    # Collapse runs of blank lines so the speech engine doesn't insert
    # awkwardly long silences.
    collapsed: list[str] = []
    blank_run = False
    for ln in cleaned_lines:
        if not ln.strip():
            if blank_run:
                continue
            blank_run = True
            collapsed.append("")
        else:
            blank_run = False
            collapsed.append(ln)
    # Neutralize any bare in-prose filenames the model wove into sentences so
    # the speech engine never reads filenames/extensions aloud.
    return _strip_source_filenames_for_speech("\n".join(collapsed).strip())


# Source references named inline in an answer, e.g. "Application.pdf (page 12)",
# "report.pdf, page 3", a bare "Rutgers_Transcript.JPG", or a spaced filename
# like "JOB APPLICATION.pdf" / "JHU Transcript.pdf". The file group absorbs any
# leading ALL-CAPS/numeric acronym words (filename fragments such as "JOB" or
# "JHU") but deliberately ignores mixed-case prose words ("While", "Also",
# "listed", "in") that merely precede a filename, then matches the final token
# ending in a known ingest extension; an optional trailing page number is
# captured too. This lets the printed 'References' block list each unique source
# once instead of repeating the whole answer. `(?i:...)` keeps the extension and
# page keyword case-insensitive while the leading-acronym class stays uppercase.
_SOURCE_REFERENCE_RE = re.compile(
    r"(?P<file>(?:[A-Z0-9][A-Z0-9._%\-]+\s+)*[^\s(),]+\.(?i:" + _CITATION_FILE_EXT + r"))"
    r"(?:\s*[\(,]\s*(?i:on\s+)?(?i:pages?)\s*(?P<page>\d+)\s*\)?)?",
)


def _extract_source_references(text: str) -> list[str]:
    """Collect unique source citations named in `text` into display lines.

    Scans for filename tokens ending in a known ingest extension (optionally
    followed by a page reference) and returns a de-duplicated, appearance-ordered
    list like ["Application.pdf (page 12)", "Rutgers_Transcript.JPG"]. Used to
    build the printed 'References' block; the answer prose itself is unchanged.
    """
    if not text:
        return []
    seen: set[tuple[str, str]] = set()
    references: list[str] = []
    for match in _SOURCE_REFERENCE_RE.finditer(text):
        file_name = match.group("file").rstrip(".,;:")
        page = match.group("page") or ""
        key = (file_name.lower(), page)
        if key in seen:
            continue
        seen.add(key)
        references.append(f"{file_name} (page {page})" if page else file_name)
    return references


def _format_response_for_display(response_text: str) -> str:
    """Render the full answer followed by a 'References' block that lists each
    unique source (file + page) named in the answer.

    The answer prose — including every value and inline citation — is preserved
    verbatim; only a de-duplicated source list is appended underneath. If the
    answer names no sources, it is returned unchanged. Spoken/TTS output is
    produced elsewhere (`_spoken_response_text` / `_summarize_for_speech`) and is
    unaffected by this function."""
    if not response_text:
        return ""
    body = response_text.strip()
    references = _extract_source_references(body)
    if not references:
        return body
    ref_block = "\n".join(f"  - {ref}" for ref in references)
    return f"{body}\n\nReferences:\n{ref_block}"


def _topic_snippet_score(text: str, terms: list[str]) -> int:
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0
    lowered = compact.lower()
    term_hits = sum(1 for term in terms if term in lowered)
    if term_hits == 0:
        return 0
    value_hits = len(set(_value_numbers(compact)))
    return term_hits * 12 + min(value_hits, 8) * 4 + min(len(compact) // 180, 6)


def _build_topic_details_preview(user_query: str, documents: list[Document], llm_client: Ollama, max_items: int = 4) -> str:
    terms = _query_terms(user_query)
    if not terms:
        return ""

    anchor_terms = _query_anchor_terms(user_query, terms)
    require_value = _query_requests_values(user_query)

    ranked: list[tuple[int, Document, str]] = []
    for doc in documents:
        search_text = _document_search_text(doc)
        if anchor_terms and not _contains_all_terms(search_text, anchor_terms):
            continue
        score = _query_candidate_score(search_text, terms, anchor_terms, require_value)
        if score <= 0:
            continue
        snippet = _query_focused_snippet(doc.text or "", terms, anchor_terms, require_value)
        if not _query_evidence_is_supported(snippet, terms, anchor_terms, require_value):
            continue
        ranked.append((score, doc, snippet))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        topic = ", ".join(anchor_terms + _query_topic_terms(terms, anchor_terms)[:3])
        if require_value:
            return f"I did not find indexed evidence with specific values for {topic}."
        return f"I did not find extracted document text about {topic} in the indexed corpus."

    topic_label = ", ".join(_query_topic_terms(terms, anchor_terms)[:3] or terms[:3])
    if anchor_terms:
        subject_label = " and ".join(term.title() for term in anchor_terms)
        lead = f"I found indexed evidence about {subject_label} related to {topic_label}."
    else:
        lead = f"I found indexed evidence related to {topic_label}."
    if require_value:
        lead += " The matching evidence includes specific values."

    lines = [lead]
    evidence_lines = []
    for _, doc, snippet in ranked[:max_items]:
        metadata = doc.metadata or {}
        file_name = os.path.basename(_canonical_detail_file_key(metadata))
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"
        evidence_lines.append(f"- {label}: {snippet}")
        lines.append(f"- {label}: {snippet}")

    synthesized, review_passed, review_reason = _answer_with_review_loop(
        user_query,
        evidence_lines,
        llm_client,
    )
    if synthesized and review_passed:
        return synthesized
    debug_print(f"[debug] Corpus fallback answer failed review: {review_reason}")

    return _not_found_answer(user_query, review_reason)


def _build_source_details_preview(user_query: str, documents: list[Document], llm_client: Ollama, max_items: int = 8) -> str:
    source_hints = _extract_source_hints(user_query)
    if not source_hints or not documents:
        return ""

    matching_docs = []
    for doc in documents:
        metadata = doc.metadata or {}
        file_key = _canonical_detail_file_key(metadata).lower()
        base_name = os.path.basename(file_key)
        if any(hint in file_key or hint in base_name for hint in source_hints):
            matching_docs.append(doc)

    if not matching_docs:
        # Distinguish "not extracted" from "extracted but filtered as low-information / OCR noise".
        # The summary document (metadata.is_file_list=True) lists every file the loaders successfully read.
        extracted_filenames: set[str] = set()
        for doc in documents:
            metadata = doc.metadata or {}
            if metadata.get("is_file_list"):
                for line in (doc.text or "").splitlines():
                    stripped = line.strip().lstrip("-").strip()
                    if stripped:
                        extracted_filenames.add(os.path.basename(stripped).lower())
        was_extracted = any(
            hint in extracted_filenames or any(hint in name for name in extracted_filenames)
            for hint in source_hints
        )
        if was_extracted:
            return (
                f"{', '.join(source_hints)} was loaded from disk, but every chunk extracted from it "
                "was dropped by the informative-content / OCR-garbage filter before indexing. "
                "Re-run with JARVIS_OCR_GARBAGE_FILTER=0 (or lower JARVIS_OCR_VOCAB_THRESHOLD) and "
                "JARVIS_REBUILD_INDEX=1 to retain noisier text from this file."
            )
        return (
            f"I did not find extracted indexed text for {', '.join(source_hints)}. "
            "Verify the filename and that the file exists in the indexed directory."
        )

    # When the user explicitly names a file, send full chunk text (capped at
    # JARVIS_SOURCE_HINT_CHUNK_CHARS, default 4000) ordered by chunk_index so
    # sections like "Certifications" that sit mid-chunk aren't cropped out by
    # the generic detail-density scorer. Matches the source-hint force-include
    # behavior in _source_nodes_to_query_evidence_lines.
    max_chars = int(os.environ.get("JARVIS_SOURCE_HINT_CHUNK_CHARS", "16000"))
    ordered: list[tuple[tuple[str, int, int], str]] = []
    for doc in matching_docs:
        text = doc.text or ""
        if not text or is_low_information_text(text):
            continue
        metadata = doc.metadata or {}
        file_name = os.path.basename(_canonical_detail_file_key(metadata))
        page_number = metadata.get("page_number")
        chunk_index = metadata.get("chunk_index")
        try:
            chunk_sort = int(chunk_index) if chunk_index is not None else 0
        except (TypeError, ValueError):
            chunk_sort = 0
        try:
            page_sort = int(page_number) if page_number is not None else 0
        except (TypeError, ValueError):
            page_sort = 0
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"
        compact = " ".join(text.split())[:max_chars].rstrip()
        if len(text) > max_chars:
            compact += "..."
        ordered.append(((file_name.lower(), page_sort, chunk_sort), f"- {label}: {compact}"))

    if not ordered:
        return f"I found {', '.join(source_hints)}, but the extracted text was too sparse to summarize."

    ordered.sort(key=lambda item: item[0])
    # Allow more lines than the default cap so multi-chunk documents (resumes,
    # contracts) reach the LLM in full instead of being truncated to ~8 chunks.
    effective_cap = max(max_items, len(ordered))
    evidence_lines = [line for _, line in ordered[:effective_cap]]
    synthesized, review_passed, review_reason = _answer_with_review_loop(
        user_query,
        evidence_lines,
        llm_client,
    )
    if synthesized and review_passed:
        return synthesized

    debug_print(f"[debug] Source details answer used extractive fallback: {review_reason}")
    return _extractive_summary_answer_from_evidence(user_query, evidence_lines) or _evidence_present_fallback(
        evidence_lines,
        review_reason,
    )


def _value_numbers(text: str) -> list[str]:
    normalized_text = text.replace("\\n", " ")
    numbers = re.findall(r"(?<![\w*/.-])(?:\$\s*)?\d[\d,]*(?:\.\d+)?%?(?![\w/.-])", normalized_text)
    values = []
    for number in numbers:
        stripped = number.strip()
        digits_only = re.sub(r"\D", "", number)
        if not digits_only:
            continue
        numeric_value = int(digits_only)
        if 1900 <= numeric_value <= 2099:
            continue
        has_amount_marker = stripped.startswith("$") or "," in stripped or "." in stripped or stripped.endswith("%")
        if not has_amount_marker and numeric_value < 100:
            continue
        if not has_amount_marker and len(digits_only) > 6:
            continue
        values.append(stripped)
    return values


def _detail_score(text: str) -> int:
    artifact_penalty = text_artifact_score(text)
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0
    if artifact_penalty >= 100:
        return 0

    lowered = compact.lower()
    boilerplate_markers = list(GENERIC_BOILERPLATE_MARKERS)
    boilerplate_markers.extend(DOMAIN_BOILERPLATE_MARKERS.get(JARVIS_DOMAIN, []))
    boilerplate_penalty = sum(
        penalty
        for marker, penalty in boilerplate_markers
        if marker in lowered
    )
    tokens = compact.split()
    numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
    readable_words = re.findall(r"[A-Za-z]{3,}", compact)
    unique_readable_words = set(word.lower() for word in readable_words)
    detail_keywords = list(GENERIC_DETAIL_KEYWORDS)
    detail_keywords.extend(DOMAIN_DETAIL_KEYWORDS.get(JARVIS_DOMAIN, []))
    keyword_hits = sum(
        1
        for keyword in detail_keywords
        if keyword in lowered
    )

    if len(unique_readable_words) < 3 and keyword_hits == 0:
        return 0
    if numeric_tokens and len(numeric_tokens) > max(len(readable_words) * 2, 8) and keyword_hits < 2:
        return 0

    repeated_token_penalty = max(len(tokens) - len(set(tokens)), 0)
    value_score = min(len(set(_value_numbers(compact))), 8) * 5
    label_score = 6 if ":" in compact else 0
    word_score = min(len(unique_readable_words), 24)
    score = value_score + keyword_hits * 8 + label_score + word_score + min(len(compact) // 160, 6)
    score -= min(repeated_token_penalty, 20)
    score -= boilerplate_penalty
    return max(score - artifact_penalty, 0)


def _detail_candidates(text: str) -> list[str]:
    text = normalize_text(text)
    lines = [line.strip() for line in re.split(r"[\n;]+", text) if line.strip()]
    candidates = list(lines)
    for window_size in (2, 3):
        for start in range(0, max(len(lines) - window_size + 1, 0)):
            candidates.append(" | ".join(lines[start:start + window_size]))
    compact = " ".join(text.split())
    if compact:
        candidates.append(compact)
    return candidates or [" ".join(text.split())]


def _best_detail_candidate(text: str) -> tuple[int, str]:
    candidates = _detail_candidates(text)
    best_candidate = max(candidates, key=_detail_score)
    return _detail_score(best_candidate), best_candidate


def _best_detail_snippet(text: str, max_chars: int = 1600) -> str:
    score, best_text = _best_detail_candidate(text)
    if score == 0:
        best_text = " ".join(normalize_text(text).split())

    snippet = " ".join(best_text.split())[:max_chars].rstrip()
    if len(best_text) > max_chars:
        snippet += "..."
    return snippet


def _canonical_detail_file_key(metadata: dict) -> str:
    file_path = metadata.get("file_path") or metadata.get("file_name") or "unknown file"
    directory = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    base_name = re.sub(r" \(\d+\)(?=\.[^.]+$)", "", base_name)
    return os.path.join(directory, base_name) if directory else base_name


def _build_corpus_details_preview(documents: list[Document], max_items: int = 8) -> str:
    best_by_file: dict[str, tuple[int, Document, str]] = {}
    for doc in documents:
        text = doc.text or ""
        score, snippet_text = _best_detail_candidate(text)
        if score <= 0:
            continue

        metadata = doc.metadata or {}
        file_key = _canonical_detail_file_key(metadata)
        current = best_by_file.get(file_key)
        if current is None or score > current[0]:
            best_by_file[file_key] = (score, doc, snippet_text)

    ranked_docs = sorted(best_by_file.values(), key=lambda item: item[0], reverse=True)[:max_items]
    if not ranked_docs:
        return "I could not find concrete extracted details in the indexed documents."

    lines = [
        "Here are concrete details present in the indexed documents from a direct corpus scan:",
    ]
    for score, doc, snippet_text in ranked_docs:
        metadata = doc.metadata or {}
        file_name = os.path.basename(_canonical_detail_file_key(metadata))
        page_number = metadata.get("page_number")
        label = f"- {file_name}"
        if page_number is not None:
            label += f" (page {page_number})"
        if JARVIS_DEBUG:
            label += f" [detail_score={score}]"
        lines.append(label)
        lines.append(f"  {_best_detail_snippet(snippet_text)}")

    return "\n".join(lines)


def _purge_chat_memory_vectors(vector_store: object) -> None:
    try:
        vector_store._connect()
        client = vector_store._client
        collection = vector_store.collection_name
        if client is None or not client.has_collection(collection):
            return

        rows = client.query(
            collection,
            filter='source == "chat_memory"',
            output_fields=["node_id"],
            limit=1000,
        )
        node_ids = [row.get("node_id") for row in rows if row.get("node_id")]
        if not node_ids:
            debug_print("[debug] No chat-memory vectors found in Milvus document collection.")
            return

        client.delete(collection_name=collection, ids=node_ids)
        print(f"Purged {len(node_ids)} chat-memory vectors from Milvus document collection.")
    except Exception as e:
        print(f"Warning: Unable to purge chat-memory vectors: {e}")

# Surya / TrOCR optional imports
try:
    from surya.ocr import run_ocr  # type: ignore[import-not-found]
    from surya.model.detection import load_model as load_detection_model  # type: ignore[import-not-found]
    from surya.model.recognition import load_model as load_recognition_model  # type: ignore[import-not-found]
    surya_available = True
except Exception:
    surya_available = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_available = True
except Exception:
    trocr_available = False


from jarvis_io import (
    JavaWarningFilter,
    suppress_external_output,
    suppress_external_stderr,
    suppress_java_warnings,
    time_limit,
)


class TTSManager:
    # Curated Kokoro voice list (en-US + en-GB). Daniel is the default
    # British male voice requested by the user.
    _KOKORO_VOICES = (
        "bm_daniel", "bm_george", "bm_lewis", "bm_fable",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_puck",
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    )
    _DEFAULT_VOICE = "bm_daniel"

    async def list_voices(self):
        """Return the curated Kokoro voice list. Kept async to preserve the
        prior public signature even though Kokoro voices are static."""
        return [{"ShortName": v} for v in self._KOKORO_VOICES]

    async def get_valid_voice(self, desired_voice):
        """Validate a requested voice against the curated Kokoro voice list.
        Falls back to bm_daniel if the requested voice is not available."""
        if desired_voice in self._KOKORO_VOICES:
            return desired_voice
        return self._DEFAULT_VOICE

    def __init__(self):
        self.enabled = os.environ.get("JARVIS_ENABLE_TTS", "1") == "1"
        if not self.enabled:
            self.initialized = False
            self.tts_available = False
            self.executor = None
            self.audio_player = None
            self.pipeline = None
            return

        try:
            desired_voice = os.environ.get("JARVIS_TTS_VOICE", self._DEFAULT_VOICE)
            try:
                self.voice = asyncio.run(self.get_valid_voice(desired_voice))
            except Exception:
                self.voice = self._DEFAULT_VOICE

            # British English voices use lang_code 'b'; American voices use 'a'.
            self.lang_code = "b" if self.voice.startswith(("bm_", "bf_")) else "a"

            # Cadence: Kokoro takes a `speed` multiplier (1.0 == default).
            # We retain `self.rate`/`self.volume` strings for backwards
            # compatibility with any external callers/inspectors, but Kokoro
            # uses `self.speed` for actual cadence control.
            self.rate = "+0%"
            self.volume = "+0%"
            try:
                self.speed = float(os.environ.get("JARVIS_TTS_SPEED", "1.0"))
            except (TypeError, ValueError):
                self.speed = 1.0

            self.audio_player = None
            if sys.platform == "darwin":
                self.audio_player = shutil.which("afplay")
            elif sys.platform.startswith("linux"):
                self.audio_player = shutil.which("ffplay") or shutil.which("mpg123") or shutil.which("aplay")
            elif sys.platform == "win32":
                self.audio_player = "powershell"

            # Lazy-import + eagerly construct the Kokoro pipeline so the model
            # weights are downloaded/loaded once at startup instead of on the
            # first user query.
            #
            # Silence Kokoro/torch/phonemizer chatter unless JARVIS_TTS_DEBUG=1:
            #   * torch UserWarning about LSTM dropout / weight_norm at load
            #   * Kokoro 'Defaulting repo_id' warning (resolved by passing
            #     repo_id explicitly below)
            #   * phonemizer 'words count mismatch' warnings on every speak()
            tts_debug = os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"
            if not tts_debug:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
                for noisy in (
                    "phonemizer", "kokoro", "espeakng_loader",
                    "huggingface_hub", "httpx", "urllib3",
                ):
                    logging.getLogger(noisy).setLevel(logging.ERROR)
                # phonemizer's own logger setup runs lazily on first speak()
                # and resets handlers + setLevel(WARNING), overriding the line
                # above. A Filter survives that reset. Logger-level filters
                # only catch records emitted on that exact logger, so attach
                # the filter to (a) the 'phonemizer' logger (where the record
                # originates) AND (b) every handler on the root logger
                # (catches the record as it propagates up to the root format
                # that Jarvis configured: "asctime - levelname - message").
                class _DropWordsMismatch(logging.Filter):
                    def filter(self, record: logging.LogRecord) -> bool:
                        return "words count mismatch" not in record.getMessage()
                _drop = _DropWordsMismatch()
                logging.getLogger("phonemizer").addFilter(_drop)
                for _h in list(logging.getLogger().handlers):
                    _h.addFilter(_drop)
            from kokoro import KPipeline  # noqa: WPS433 — local import keeps
            # the kokoro/torch import cost off any non-TTS code path.
            kpipeline_kwargs = {"lang_code": self.lang_code, "repo_id": "hexgrad/Kokoro-82M"}
            if not tts_debug:
                # Kokoro prints a few INFO-level lines to stderr during load
                # that aren't covered by the logger silencing above; route
                # them to /dev/null so the REPL prompt stays clean.
                with suppress_external_output():
                    self.pipeline = KPipeline(**kpipeline_kwargs)
            else:
                self.pipeline = KPipeline(**kpipeline_kwargs)
            self.sample_rate = 24000  # Kokoro emits 24 kHz mono float32.

            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
            self.initialized = True
            self.tts_available = True
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_available = False
            self.executor = None
            self.pipeline = None

    # Words that, when present in the same sentence as a comma-grouped number,
    # strongly imply the number is a dollar amount and should be spoken as such.
    _TTS_MONEY_KEYWORDS = (
        "income", "wages", "salary", "salaries", "tax", "taxes", "withheld",
        "withholding", "deduction", "deductions", "credit", "credits", "refund",
        "refunds", "paid", "payment", "payments", "balance", "due", "owed",
        "owing", "amount", "amounts", "total", "totals", "subtotal", "fee",
        "fees", "cost", "costs", "price", "expense", "expenses", "revenue",
        "revenues", "earnings", "compensation", "interest", "dividend",
        "dividends", "contribution", "contributions", "agi", "gross",
        "adjusted", "exemption", "exemptions", "loan", "principal", "mortgage",
        "premium", "rebate", "value", "values", "dollars", "dollar",
    )

    @classmethod
    def _add_dollar_sign_in_financial_context(cls, text: str) -> str:
        """Prepend `$` to comma-grouped numbers (e.g. `453,172`, `8,577.50`)
        when their sentence mentions a money keyword. This lets the existing
        `$N` -> 'N dollars' rule fire so the spoken output says 'dollars'
        instead of just reading a bare number.

        We only touch comma-grouped numbers (>= 1,000 with thousands separator)
        to avoid mangling years, line numbers, phone digits, dates, etc. Numbers
        already written with the word "dollar(s)" are skipped so we don't produce
        "dollars dollars"."""
        money_pattern = re.compile(
            r"(?<![\w$,])(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)(?!\w)(?![,\d])(?!\s*[Dd]ollars?\b)"
        )
        keywords = cls._TTS_MONEY_KEYWORDS
        # Split on sentence boundaries while keeping the boundary characters so
        # we can rejoin without altering pause cues.
        parts = re.split(r"(?<=[.!?])\s+", text)
        rewritten: list[str] = []
        for sentence in parts:
            lowered = sentence.lower()
            has_money_word = any(kw in lowered for kw in keywords)
            if has_money_word:
                sentence = money_pattern.sub(r"$\1", sentence)
            rewritten.append(sentence)
        return " ".join(rewritten)

    def clean_text(self, text):
        cleaned = str(text or "")
        # Expand context-dependent abbreviations/numerals (Dr., Jr., III, ...)
        # before any punctuation/hyphen stripping, so the rules can see periods
        # and hyphens (e.g. "Dr.", "J-R", "I-I-I").
        cleaned = _normalize_for_speech(cleaned)
        cleaned = re.sub(r"https?://\S+", " link omitted ", cleaned)
        # Recognize money-shaped numbers in financial sentences (no `$` needed).
        cleaned = self._add_dollar_sign_in_financial_context(cleaned)
        cleaned = re.sub(r"\$\s*(\d[\d,]*)(?:\.(\d{1,2}))?", self._spoken_dollar_amount, cleaned)
        cleaned = re.sub(r"\b(\d[\d,]*)\.(\d+)\s*%", lambda m: f"{m.group(1).replace(',', '')} point {m.group(2)} percent", cleaned)
        cleaned = re.sub(r"\b(\d[\d,]*)\s*%", lambda m: f"{m.group(1).replace(',', '')} percent", cleaned)
        cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
        cleaned = re.sub(r"(?<=\d)\.(?=\d)", " point ", cleaned)
        cleaned = re.sub(r"[\u2010-\u2015\-]+", " ", cleaned)
        cleaned = cleaned.replace("'", "")
        cleaned = cleaned.replace("\u2019", "")
        # Keep sentence/clause punctuation (, . ; : ? !) so the Neural voice has
        # natural pause and intonation cues. Strip everything else.
        cleaned = re.sub(r"[^A-Za-z0-9\s,\.;:?!]", " ", cleaned)
        # Collapse repeats of the same punctuation (e.g. "..", "!!").
        cleaned = re.sub(r"([,\.;:?!])\1+", r"\1", cleaned)
        # Drop stray spaces before punctuation introduced by replacements above.
        cleaned = re.sub(r"\s+([,\.;:?!])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _spoken_dollar_amount(match):
        dollars = match.group(1).replace(",", "")
        cents_raw = (match.group(2) or "").ljust(2, "0")[:2]
        cents = int(cents_raw) if cents_raw else 0
        if cents:
            return f"{dollars} dollars and {cents} cents"
        return f"{dollars} dollars"

    def say_goodbye(self):
        if not getattr(self, 'initialized', False):
            return
        if os.environ.get("JARVIS_TTS_GOODBYE", "0") != "1":
            return
        try:
            self.speak("Goodbye!")
        except Exception as e:
            logger.error(f"Error during goodbye: {e}")

    def cleanup(self):
        if not getattr(self, 'initialized', False):
            return
        try:
            executor = getattr(self, 'executor', None)
            if executor is not None:
                executor.shutdown(wait=True)
                self.executor = None
            self.initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def speak(self, text):
        if not getattr(self, 'enabled', True):
            return
        if not getattr(self, 'tts_available', False):
            if os.environ.get("JARVIS_TTS_DEBUG", "0") == "1":
                logger.debug("TTS is unavailable, cannot speak text.")
            return
        executor = getattr(self, 'executor', None)
        if executor is None:
            return
        try:
            executor.submit(self._speak_async, text)
        except RuntimeError:
            # Executor already shut down; ignore.
            pass

    def _speak_async(self, text):
        try:
            tts_debug = os.environ.get("JARVIS_TTS_DEBUG", "0") == "1"
            clean_text = self.clean_text(text)
            if not clean_text:
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                generation_succeeded = False
                try:
                    import soundfile as sf  # noqa: WPS433 — local import
                    audio_chunks: list[np.ndarray] = []
                    # Suppress phonemizer 'words count mismatch' warnings and
                    # any other Kokoro stderr chatter on each speak() call.
                    # stderr-only: this runs on a background worker thread, so it
                    # must NOT touch the process-global sys.stdout that the main
                    # REPL thread uses for the interactive "You: " prompt.
                    pipeline_ctx = (
                        suppress_external_stderr() if not tts_debug
                        else nullcontext()
                    )
                    with pipeline_ctx:
                        for _gs, _ps, audio in self.pipeline(
                            clean_text,
                            voice=self.voice,
                            speed=self.speed,
                        ):
                            if audio is None:
                                continue
                            # Kokoro yields torch tensors; convert defensively.
                            try:
                                audio_np = audio.detach().cpu().numpy()
                            except AttributeError:
                                audio_np = np.asarray(audio)
                            audio_chunks.append(audio_np)
                    if audio_chunks:
                        arr = np.concatenate(audio_chunks)
                        sf.write(tmp_file.name, arr, self.sample_rate)
                        generation_succeeded = True
                    elif tts_debug:
                        logger.debug("Kokoro produced no audio chunks for input.")
                except Exception as e:
                    if tts_debug:
                        logger.debug(f"TTS generation failed: {e}")
                playback_succeeded = False
                if generation_succeeded and self.audio_player:
                    try:
                        if sys.platform == "win32":
                            subprocess.run(
                                [self.audio_player, "-c", f"(New-Object Media.SoundPlayer '{tmp_file.name}').PlaySync();"],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        else:
                            subprocess.run(
                                [self.audio_player, tmp_file.name],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        playback_succeeded = True
                    except Exception as e:
                        if tts_debug:
                            logger.debug(f"Audio playback failed with player {self.audio_player}: {e}")
                elif generation_succeeded and tts_debug:
                    logger.debug(f"Speech audio generated and saved to {tmp_file.name}")

                if tts_debug:
                    try:
                        stat = os.stat(tmp_file.name)
                        logger.debug(f"TTS temp file: {tmp_file.name} size={stat.st_size}")
                    except Exception:
                        pass

                if playback_succeeded or not tts_debug:
                    try:
                        os.unlink(tmp_file.name)
                    except Exception as e:
                        if tts_debug:
                            logger.debug(f"Failed to clean up temporary TTS file: {e}")
                elif tts_debug:
                    logger.debug(f"Left TTS temp file for inspection: {tmp_file.name}")
        except Exception as e:
            if os.environ.get("JARVIS_TTS_DEBUG", "0") == "1":
                logger.debug(f"Failed to generate or play speech: {e}")


from jarvis_textproc import (
    is_low_information_text,
    normalize_text,
    text_artifact_score,
    _is_ocr_garbage_text,
    _load_system_english_vocab,
)


def filter_informative_documents(documents: List[Document]) -> List[Document]:
    kept: List[Document] = []
    for doc in documents:
        text = doc.text or ""
        if is_low_information_text(text):
            continue
        is_garbage, reason = _is_ocr_garbage_text(text)
        if is_garbage:
            metadata = doc.metadata or {}
            source = metadata.get("file_name") or metadata.get("file_path") or doc.id_
            extracted_by = metadata.get("extracted_by", "unknown")
            debug_print(
                f"[debug] OCR-garbage filter dropped chunk from {source} "
                f"[{extracted_by}]: {reason}"
            )
            continue
        kept.append(doc)
    return kept


def _document_source_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("file_path") or metadata.get("file_name") or doc.id_)


def _document_sort_key(doc: Document) -> tuple[str, int, int, int]:
    metadata = doc.metadata or {}
    page_number = metadata.get("page_number")
    chunk_index = metadata.get("chunk_index")
    table_index = metadata.get("table_index")
    return (
        _document_source_key(doc),
        page_number if isinstance(page_number, int) else 0,
        chunk_index if isinstance(chunk_index, int) else 0,
        table_index if isinstance(table_index, int) else 0,
    )


def _format_context_piece(doc: Document) -> str:
    metadata = doc.metadata or {}
    labels = []
    page_number = metadata.get("page_number")
    extracted_by = metadata.get("extracted_by")
    table_index = metadata.get("table_index")
    if page_number is not None:
        labels.append(f"page {page_number}")
    if table_index is not None:
        labels.append(f"table {table_index}")
    if extracted_by:
        labels.append(str(extracted_by))

    text = normalize_text(doc.text or "")
    if not labels:
        return text
    return f"[{' | '.join(labels)}]\n{text}"


def group_documents_by_source_windows(
    documents: List[Document],
    max_words: int = 2500,
) -> List[Document]:
    grouped: Dict[str, List[Document]] = {}
    standalone_docs: List[Document] = []
    for doc in documents:
        metadata = doc.metadata or {}
        if metadata.get("is_file_list"):
            standalone_docs.append(doc)
            continue
        grouped.setdefault(_document_source_key(doc), []).append(doc)

    context_docs: List[Document] = list(standalone_docs)
    for source_key in sorted(grouped):
        source_docs = sorted(grouped[source_key], key=_document_sort_key)
        current_pieces: List[str] = []
        current_word_count = 0
        window_index = 0
        window_start_page: Optional[int] = None
        window_end_page: Optional[int] = None

        def emit_window() -> None:
            nonlocal current_pieces, current_word_count, window_index, window_start_page, window_end_page
            if not current_pieces:
                return
            first_metadata = dict(source_docs[0].metadata or {})
            first_metadata.update({
                "context_mode": "source_window",
                "source_document_count": len(current_pieces),
                "source_window_index": window_index,
                "source_window_start_page": window_start_page,
                "source_window_end_page": window_end_page,
            })
            context_docs.append(Document(text="\n\n".join(current_pieces), metadata=first_metadata))
            window_index += 1
            current_pieces = []
            current_word_count = 0
            window_start_page = None
            window_end_page = None

        for doc in source_docs:
            piece = _format_context_piece(doc)
            if not piece:
                continue
            piece_word_count = len(piece.split())
            if current_pieces and current_word_count + piece_word_count > max_words:
                emit_window()

            metadata = doc.metadata or {}
            page_number = metadata.get("page_number")
            if isinstance(page_number, int):
                window_start_page = page_number if window_start_page is None else min(window_start_page, page_number)
                window_end_page = page_number if window_end_page is None else max(window_end_page, page_number)
            current_pieces.append(piece)
            current_word_count += piece_word_count

        emit_window()

    return context_docs


def normalize_vector(embedding: Sequence[float]) -> List[float]:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        return (vector / norm).tolist()
    return vector.tolist()


def _metadata_filters_to_milvus_expr(filters: Optional[MetadataFilters]) -> Optional[str]:
    if filters is None:
        return None

    try:
        filter_list = filters.legacy_filters()
    except Exception as exc:
        raise ValueError(
            "Milvus vector store only supports exact-match metadata filters."
        ) from exc

    expr_pieces: List[str] = []
    for filt in filter_list:
        key = filt.key
        value = filt.value
        if key not in {"ref_doc_id", "source", "file_path", "parent_file"}:
            raise ValueError(
                "Milvus vector store only supports filters on ref_doc_id, source, file_path, or parent_file."
            )

        if isinstance(value, str):
            safe_value = value.replace('"', '\\"')
            expr_pieces.append(f'{key} == "{safe_value}"')
        else:
            expr_pieces.append(f"{key} == {value}")

    condition = filters.condition.value if getattr(filters.condition, 'value', None) else 'and'
    return f" {condition} ".join(expr_pieces)


class MilvusVectorStore(BasePydanticVectorStore):
    stores_text: bool = False
    is_embedding_query: bool = True
    auto_flush: bool = True

    collection_name: str = "jarvis_context"
    host: str = "127.0.0.1"
    port: str = "19530"
    dim: Optional[int] = None
    max_vectors: int = 5000  # Community Edition limit
    _collection: Optional[Any] = PrivateAttr(default=None)
    _client: Optional[MilvusClient] = PrivateAttr(default=None)

    def __init__(
        self,
        collection_name: str = "jarvis_context",
        host: str = "127.0.0.1",
        port: str = "19530",
        dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(collection_name=collection_name, host=host, port=port, dim=dim, **kwargs)
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dim = dim
        self._collection = None
        self._client = None

    @property
    def client(self) -> Optional[Any]:
        return self._collection

    def has_collection(self) -> bool:
        self._connect()
        return self._client is not None and self._client.has_collection(self.collection_name)

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            # Prefer the new MilvusClient HTTP interface
            self._client = MilvusClient(f"http://{self.host}:{self.port}")

        except Exception as exc:
            # Try fallback port if user mistakenly provided 2379
            if self.port == "2379":
                try:
                    self._client = MilvusClient(f"http://{self.host}:19530")
                    self.port = "19530"
                    return
                except Exception:
                    pass
            raise RuntimeError(f"Failed to connect to Milvus at {self.host}:{self.port}: {exc}") from exc

    def _ensure_collection(self, dim: int) -> None:
        self._connect()

        if self._client is None:
            raise RuntimeError("Milvus client is not initialized.")

        # -------------------------------------------------
        # EXISTING COLLECTION
        # -------------------------------------------------
        if self._client.has_collection(self.collection_name):

            try:
                desc = self._client.describe_collection(self.collection_name)

                fields = desc.get("fields", []) if isinstance(desc, dict) else []

                vector_field = next(
                    (f for f in fields if f.get("name") == "embedding"),
                    None,
                )

                if vector_field is None:
                    raise RuntimeError(
                        "Milvus collection schema does not contain an embedding field."
                    )

                self.dim = int(vector_field.get("params", {}).get("dim"))
                if dim and self.dim != dim:
                    raise RuntimeError(
                        f"Milvus collection embedding dimension is {self.dim}, "
                        f"but the active embedding model produced {dim}. "
                        "Use a matching embedding model or rebuild/drop the collection."
                    )

            except Exception as e:
                if self.dim is None:
                    raise RuntimeError(
                        f"Could not determine embedding dimension: {e}"
                    ) from e
                raise

            # ---------------------------------------------
            # ONLY CREATE INDEX IF ONE DOES NOT EXIST
            # ---------------------------------------------
            try:
                existing_indexes = self._client.list_indexes(self.collection_name)

                if not existing_indexes:

                    from pymilvus.milvus_client.index import IndexParams

                    idx = IndexParams()

                    idx.add_index(
                        field_name="embedding",
                        index_type="HNSW",
                        index_name="embedding_hnsw",
                        metric_type="IP",
                        params={
                            "M": 16,
                            "efConstruction": 100,
                        },
                    )

                    self._client.create_index(
                        collection_name=self.collection_name,
                        index_params=idx,
                    )

                    logger.info(
                        f"Created HNSW index on collection {self.collection_name}"
                    )

            except Exception as e:
                logger.warning(f"Index verification failed: {e}")

            try:
                self._client.load_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Failed to load collection: {e}")

            self._collection = self._client
            return

        # -------------------------------------------------
        # CREATE NEW COLLECTION
        # -------------------------------------------------
        self.dim = dim

        if self.dim is None:
            raise ValueError(
                "Milvus collection does not exist and no embedding dimension was provided."
            )

        fields = [
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=1024,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            FieldSchema(
                name="ref_doc_id",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
            FieldSchema(
                name="file_path",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="parent_file",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="metadata_json",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
        ]

        schema = CollectionSchema(
            fields,
            description="Jarvis Milvus vector store",
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # -------------------------------------------------
        # CREATE INDEX
        # -------------------------------------------------
        from pymilvus.milvus_client.index import IndexParams

        idx = IndexParams()

        idx.add_index(
            field_name="embedding",
            index_type="HNSW",
            index_name="embedding_hnsw",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 100,
            },
        )

        self._client.create_index(
            collection_name=self.collection_name,
            index_params=idx,
        )

        self._client.load_collection(self.collection_name)

        logger.info(
            f"Created Milvus collection and index: {self.collection_name}"
        )

        self._collection = self._client

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError("MilvusVectorStore does not store text nodes directly.")

    def _get_collection_size(self) -> int:
        """Get the current number of vectors in the collection."""
        try:
            self._connect()
        except Exception:
            return 0

        # Prefer client stats when available
        try:
            if self._client is not None and self._client.has_collection(self.collection_name):
                stats = self._client.get_collection_stats(self.collection_name)
                return int(stats.get("row_count", 0))
        except Exception:
            pass
        # Fallback to 0 if unknown
        return 0

    def _make_room_for_new_vectors(self, num_new_vectors: int) -> None:
        """Delete oldest vectors if adding new_vectors would exceed max_vectors."""
        current_size = self._get_collection_size()
        if current_size + num_new_vectors <= self.max_vectors:
            return

        num_to_delete = (current_size + num_new_vectors) - self.max_vectors + 10  # Delete 10 extra for headroom
        logger.info(f"Collection at {current_size}/{self.max_vectors} vectors. Deleting {num_to_delete} oldest entries to make room.")

        try:
            # Use client.query to retrieve node ids and metadata
            all_rows = []
            if self._client is not None and self._client.has_collection(self.collection_name):
                try:
                    all_rows = self._client.query(
                        self.collection_name,
                        filter='node_id != ""',
                        output_fields=["node_id", "metadata_json"],
                        limit=max(current_size, num_to_delete * 4, 1000),
                    )
                except Exception as e:
                    logger.warning(f"Failed to enumerate vectors for eviction: {e}")
                    all_rows = []

            if all_rows:
                entries_with_timestamps = []
                for row in all_rows:
                    node_id = row.get("node_id")
                    metadata_json_str = row.get("metadata_json", "{}")
                    try:
                        metadata = json.loads(metadata_json_str) if isinstance(metadata_json_str, str) else {}
                    except Exception:
                        metadata = {}

                    timestamp_str = metadata.get("insertion_timestamp", "1970-01-01T00:00:00Z")
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except Exception:
                        timestamp = datetime(1970, 1, 1)

                    entries_with_timestamps.append((node_id, timestamp))

                entries_with_timestamps.sort(key=lambda x: x[1])
                oldest_node_ids = [node_id for node_id, _ in entries_with_timestamps[:num_to_delete]]

                if oldest_node_ids:
                    quoted_ids = ", ".join([f'"{nid.replace("\"", "\\\"")}"' for nid in oldest_node_ids])
                    expr = f"node_id in [{quoted_ids}]"
                    deleted_ok = False
                    try:
                        self._client.delete(self.collection_name, filter=expr)
                        deleted_ok = True
                    except Exception as e:
                        logger.warning(f"Failed to delete oldest vectors via client: {e}")
                    if deleted_ok:
                        logger.info(f"Deleted {len(oldest_node_ids)} oldest vectors to maintain capacity.")
                    else:
                        logger.warning(
                            f"Eviction attempted for {len(oldest_node_ids)} vectors but the delete call failed; "
                            "collection may exceed max_vectors."
                        )
            else:
                logger.warning(
                    "Eviction skipped: could not enumerate existing vectors. "
                    "Collection may exceed max_vectors after insert."
                )
        except Exception as e:
            logger.warning(f"Failed to delete oldest vectors: {e}. Proceeding with insert anyway.")

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:

        if not nodes:
            return []

        if self._collection is None:
            self._ensure_collection(dim=len(nodes[0].get_embedding()))

        self._make_room_for_new_vectors(len(nodes))

        rows = []

        for node in nodes:

            metadata = dict(node.metadata or {})
            embedding = normalize_vector(node.get_embedding())
            if self.dim is not None and len(embedding) != self.dim:
                raise RuntimeError(
                    f"Node {node.node_id} has embedding dimension {len(embedding)}, "
                    f"but Milvus collection expects {self.dim}."
                )

            if "insertion_timestamp" not in metadata:
                metadata["insertion_timestamp"] = (
                    datetime.utcnow().isoformat() + "Z"
                )

            rows.append({
                "node_id": node.node_id,
                "embedding": embedding,
                "ref_doc_id": str(
                    node.ref_doc_id
                    or metadata.get("ref_doc_id")
                    or metadata.get("file_path")
                    or node.node_id
                ),
                "source": str(metadata.get("source", "file")),
                "file_path": str(metadata.get("file_path", "")),
                "parent_file": str(metadata.get("parent_file", "")),
                "metadata_json": json.dumps(metadata, default=str),
            })

        try:

            if self._client is not None and self._client.has_collection(
                self.collection_name
            ):

                self._client.insert(
                    self.collection_name,
                    rows,
                )
                if self.auto_flush:
                    self.flush()

            else:
                raise RuntimeError(
                    "Milvus collection is unavailable."
                )

        except Exception as e:
            raise RuntimeError(f"Milvus insert failed: {e}") from e

        return [row["node_id"] for row in rows]

    def flush(self) -> None:
        self._connect()
        if self._client is None or not self._client.has_collection(self.collection_name):
            return
        try:
            self._client.flush(collection_name=self.collection_name)
        except TypeError:
            self._client.flush(self.collection_name)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._connect()
        expr = f'ref_doc_id == "{ref_doc_id.replace("\"", "\\\"")}"'
        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                self._client.delete(self.collection_name, filter=expr)
            except Exception as e:
                logger.warning(f"Milvus client delete failed: {e}")
        else:
            raise RuntimeError("Milvus client is not initialized or collection does not exist; cannot perform delete.")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._connect()

        expr: Optional[str] = None
        if node_ids is not None:
            quoted_ids = ", ".join([f'"{nid.replace("\"", "\\\"")}"' for nid in node_ids])
            expr = f"node_id in [{quoted_ids}]"
        elif filters is not None:
            expr = _metadata_filters_to_milvus_expr(filters)

        if expr is None and node_ids is None:
            raise ValueError("MilvusVectorStore.delete_nodes requires node_ids or metadata filters.")

        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                if node_ids is not None:
                    self._client.delete(self.collection_name, ids=node_ids)
                else:
                    self._client.delete(self.collection_name, filter=expr)
            except Exception as e:
                logger.warning(f"Milvus client delete_nodes failed: {e}")
        else:
            raise RuntimeError("Milvus client is not initialized or collection does not exist; cannot perform delete_nodes.")

    def clear(self) -> None:
        self._connect()
        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                self._client.drop_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Milvus client drop_collection failed: {e}")
        else:
            if self._collection is not None:
                try:
                    self._collection.drop()
                except Exception:
                    pass
        self._collection = None

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("MilvusVectorStore requires a query embedding.")

        self._connect()
        if self._client is None or not self._client.has_collection(self.collection_name):
            raise RuntimeError("Milvus collection does not exist or client is not initialized.")

        query_embedding = normalize_vector(query.query_embedding)
        if self.dim is not None and len(query_embedding) != self.dim:
            raise RuntimeError(
                f"Query embedding dimension is {len(query_embedding)}, "
                f"but Milvus collection expects {self.dim}."
            )
        expr = _metadata_filters_to_milvus_expr(query.filters)

        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64},
        }
        # Prefer client.search and retry once for transient QueryNode/channel issues.
        top_ids: List[str] = []
        top_similarities: List[float] = []
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                if self._client is None or not self._client.has_collection(self.collection_name):
                    raise RuntimeError("Milvus collection is unavailable for search.")

                results = self._client.search(
                    self.collection_name,
                    data=[query_embedding],
                    anns_field="embedding",
                    search_params=search_params,
                    limit=query.similarity_top_k,
                    filter=expr or "",
                    output_fields=["node_id"],
                )

                if results and len(results) > 0 and results[0]:
                    for hit in results[0]:
                        nid = hit.get("node_id") or (hit.get("entity") or {}).get("node_id")
                        dist = hit.get("distance")
                        if nid is not None and dist is not None:
                            top_ids.append(str(nid))
                            top_similarities.append(float(dist))

                return VectorStoreQueryResult(ids=top_ids, similarities=top_similarities)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                transient = (
                    "channel distribution is not serviceable" in msg
                    or "channel not available" in msg
                    or "code=503" in msg
                )
                if attempt == 0 and transient:
                    logger.warning(
                        f"Transient Milvus search failure. Retrying after reload: {e}"
                    )
                    try:
                        if self._client is not None and self._client.has_collection(self.collection_name):
                            self._client.load_collection(self.collection_name)
                    except Exception as load_exc:
                        logger.warning(f"Milvus reload before retry failed: {load_exc}")
                    time.sleep(0.5)
                    continue
                break

        raise RuntimeError(f"Milvus search failed: {last_error}")

    def persist(
        self,
        persist_path: str,
        fs: Optional[Any] = None,
    ) -> None:
        return None


def ensure_document_metadata(doc: Document) -> None:
    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, dict):
            return {str(k): _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_json_safe(v) for v in value]
        return value

    if doc.metadata is None:
        doc.metadata = {}
    else:
        doc.metadata = _to_json_safe(doc.metadata)

    if "ref_doc_id" not in doc.metadata:
        file_path = doc.metadata.get("file_path")
        doc.metadata["ref_doc_id"] = _to_json_safe(file_path) or doc.id_

    if "source" not in doc.metadata:
        if doc.metadata.get("file_path"):
            doc.metadata["source"] = "file"
        elif doc.metadata.get("extracted_by"):
            doc.metadata["source"] = "image"
        else:
            doc.metadata["source"] = "unknown"


def _stable_document_id(text: str, metadata: Dict[str, Any]) -> str:
    identity = {
        "file_path": metadata.get("file_path"),
        "file_name": metadata.get("file_name"),
        "page_number": metadata.get("page_number"),
        "sheet": metadata.get("sheet"),
        "slide_index": metadata.get("slide_index"),
        "table_index": metadata.get("table_index"),
        "chunk_index": metadata.get("chunk_index"),
        "extracted_by": metadata.get("extracted_by"),
        "source": metadata.get("source"),
        "text_hash": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(),
    }
    payload = json.dumps(identity, sort_keys=True, default=str)
    return "jarvis-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _document_with_stable_id(text: str, metadata: Dict[str, Any]) -> Document:
    metadata = dict(metadata)
    metadata["word_count"] = len((text or "").split())
    doc_id = _stable_document_id(text, metadata)
    return Document(text=text, metadata=metadata, id_=doc_id)


def _prepare_index_document(doc: Document, chunk_index: int = 0) -> Document:
    text = normalize_text(doc.text or "")
    metadata = dict(doc.metadata or {})
    metadata.setdefault("chunk_index", chunk_index)
    metadata.setdefault("parent_file", metadata.get("file_path"))
    metadata_doc = Document(text=text, metadata=metadata)
    ensure_document_metadata(metadata_doc)
    metadata = dict(metadata_doc.metadata or {})
    prepared = _document_with_stable_id(text, metadata)
    ensure_document_metadata(prepared)
    return prepared


def sync_documents_with_index(index: VectorStoreIndex, documents: List[Document]) -> int:
    try:
        existing_hashes = set(index.docstore.get_all_document_hashes().values())
    except Exception:
        existing_hashes = set()

    new_docs = [doc for doc in documents if doc.hash not in existing_hashes]
    for doc in new_docs:
        index.insert(doc)
    return len(new_docs)


def _count_simple_vector_embeddings(vector_store: BasePydanticVectorStore) -> Optional[int]:
    if not isinstance(vector_store, SimpleVectorStore):
        return None

    data = getattr(vector_store, "data", None)
    embedding_dict = getattr(data, "embedding_dict", None)
    if embedding_dict is None:
        return 0
    return len(embedding_dict)


def _count_storage_context_embeddings(storage_context: StorageContext) -> Optional[int]:
    counts = [
        count
        for count in (
            _count_simple_vector_embeddings(vector_store)
            for vector_store in storage_context.vector_stores.values()
        )
        if count is not None
    ]
    if not counts:
        return None
    return sum(counts)


def _count_storage_context_documents(storage_context: StorageContext) -> Optional[int]:
    docstore = getattr(storage_context, "docstore", None)
    if docstore is None:
        return None
    docs = getattr(docstore, "docs", None)
    if docs is not None:
        try:
            return len(docs)
        except Exception:
            pass
    try:
        hashes = docstore.get_all_document_hashes()
        return len(hashes)
    except Exception:
        return None


def _vector_store_embedding_count(vector_store: BasePydanticVectorStore) -> Optional[int]:
    simple_count = _count_simple_vector_embeddings(vector_store)
    if simple_count is not None:
        return simple_count

    get_collection_size = getattr(vector_store, "_get_collection_size", None)
    if callable(get_collection_size):
        try:
            return int(get_collection_size())
        except Exception:
            return None

    return None


def _index_manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, INDEX_MANIFEST_FILE)


def _embed_model_signature(embed_model: Optional[Any]) -> str:
    if embed_model is None:
        return "default"
    model_name = getattr(embed_model, "model_name", None) or getattr(embed_model, "_model_name", None)
    payload = {
        "class": f"{embed_model.__class__.__module__}.{embed_model.__class__.__name__}",
        "model_name": model_name,
        "query_instruction": getattr(embed_model, "query_instruction", None),
        "text_instruction": getattr(embed_model, "text_instruction", None),
        "normalize": getattr(embed_model, "normalize", None),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _vector_store_signature(vector_store: BasePydanticVectorStore) -> Dict[str, Any]:
    return {
        "class": f"{vector_store.__class__.__module__}.{vector_store.__class__.__name__}",
        "collection_name": getattr(vector_store, "collection_name", None),
        "host": getattr(vector_store, "host", None),
        "port": getattr(vector_store, "port", None),
        "dim": getattr(vector_store, "dim", None),
    }


def _build_index_manifest(
    documents: List[Document],
    vector_store: BasePydanticVectorStore,
    embed_model: Optional[Any],
) -> Dict[str, Any]:
    doc_entries = [
        {
            "id": doc.id_,
            "hash": doc.hash,
            "ref_doc_id": (doc.metadata or {}).get("ref_doc_id"),
            "file_path": (doc.metadata or {}).get("file_path"),
            "chunk_index": (doc.metadata or {}).get("chunk_index"),
        }
        for doc in documents
    ]
    doc_entries.sort(key=lambda item: str(item.get("id")))
    payload = {
        "version": INDEX_MANIFEST_VERSION,
        "embed_model": _embed_model_signature(embed_model),
        "vector_store": _vector_store_signature(vector_store),
        "documents": doc_entries,
    }
    payload["signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def _load_index_manifest(persist_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_index_manifest_path(persist_dir), "r", encoding="utf-8") as file:
            manifest = json.load(file)
        return manifest if isinstance(manifest, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(f"Unable to read index manifest: {exc}")
        return None


def _write_index_manifest(persist_dir: str, manifest: Dict[str, Any]) -> None:
    try:
        with open(_index_manifest_path(persist_dir), "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True, default=str)
    except Exception as exc:
        logger.warning(f"Unable to write index manifest: {exc}")


def _manifest_matches(current: Dict[str, Any], persisted: Optional[Dict[str, Any]]) -> bool:
    if not persisted:
        return False
    return persisted.get("version") == INDEX_MANIFEST_VERSION and persisted.get("signature") == current.get("signature")


def _scope_file_paths(manifest: Dict[str, Any]) -> set[str]:
    """file_paths present in the current run's manifest. Persisted entries for
    files OUTSIDE this set come from previous runs against other directories
    and must never be touched (no delete, no manifest drop)."""
    return {
        entry.get("file_path")
        for entry in manifest.get("documents", [])
        if entry.get("file_path")
    }


def _scoped_manifests_match(
    current: Dict[str, Any],
    persisted: Optional[Dict[str, Any]],
    scope_file_paths: set[str],
) -> bool:
    """True when the (id, hash) sets agree for every file in scope_file_paths.
    Used as the fast-path test: if in-scope chunks are unchanged, no Milvus
    work is needed even when out-of-scope persisted entries differ."""
    if not persisted or not scope_file_paths:
        return False
    cur = {
        (entry["id"], entry.get("hash"))
        for entry in current.get("documents", [])
        if entry.get("id") and entry.get("file_path") in scope_file_paths
    }
    per = {
        (entry["id"], entry.get("hash"))
        for entry in persisted.get("documents", [])
        if entry.get("id") and entry.get("file_path") in scope_file_paths
    }
    return cur == per


def _manifest_for_persist(
    current_manifest: Dict[str, Any],
    persisted_manifest: Optional[Dict[str, Any]],
    scope_file_paths: set[str],
) -> Dict[str, Any]:
    """Merge current-run entries with persisted entries from out-of-scope files
    so the written manifest never forgets vectors from previous runs against
    other directories."""
    if not persisted_manifest:
        return current_manifest
    persisted_oos = [
        entry
        for entry in persisted_manifest.get("documents", [])
        if entry.get("file_path") and entry.get("file_path") not in scope_file_paths
    ]
    if not persisted_oos:
        return current_manifest
    merged_entries = list(current_manifest.get("documents", [])) + persisted_oos
    merged_entries.sort(key=lambda item: str(item.get("id")))
    merged = {
        "version": current_manifest.get("version"),
        "embed_model": current_manifest.get("embed_model"),
        "vector_store": current_manifest.get("vector_store"),
        "documents": merged_entries,
    }
    merged["signature"] = hashlib.sha256(
        json.dumps(merged, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return merged


def _structural_signatures_match(
    current: Dict[str, Any], persisted: Optional[Dict[str, Any]]
) -> bool:
    """True when embed model + vector store signatures (and manifest version) match.

    Used to decide whether a delta-sync is safe vs. a full rebuild. The chunk
    set may differ; only the embedding/vector-store schema must be compatible.
    """
    if not persisted:
        return False
    if persisted.get("version") != INDEX_MANIFEST_VERSION:
        return False
    return (
        persisted.get("embed_model") == current.get("embed_model")
        and persisted.get("vector_store") == current.get("vector_store")
    )


def _diff_manifest_documents(
    current: Dict[str, Any],
    persisted: Optional[Dict[str, Any]],
    scope_file_paths: Optional[set[str]] = None,
) -> tuple[list[str], list[str]]:
    """Return (ids_to_insert, ids_to_delete) by comparing chunk id+hash sets.

    When scope_file_paths is provided, persisted chunks whose file_path is NOT
    in that set are ignored — they came from a previous run against a different
    directory and must be preserved.
    """
    current_entries = {
        entry["id"]: entry for entry in current.get("documents", []) if entry.get("id")
    }
    persisted_entries = {
        entry["id"]: entry
        for entry in (persisted or {}).get("documents", [])
        if entry.get("id")
    }
    if scope_file_paths is not None:
        persisted_in_scope = {
            doc_id: entry
            for doc_id, entry in persisted_entries.items()
            if entry.get("file_path") in scope_file_paths
        }
    else:
        persisted_in_scope = persisted_entries
    to_delete = [
        doc_id
        for doc_id, entry in persisted_in_scope.items()
        if doc_id not in current_entries
        or current_entries[doc_id].get("hash") != entry.get("hash")
    ]
    to_insert = [
        doc_id
        for doc_id, entry in current_entries.items()
        if doc_id not in persisted_entries
        or persisted_entries[doc_id].get("hash") != entry.get("hash")
    ]
    return to_insert, to_delete


def _apply_manifest_delta(
    index: VectorStoreIndex,
    current_documents: List[Document],
    current_manifest: Dict[str, Any],
    persisted_manifest: Optional[Dict[str, Any]],
    scope_file_paths: Optional[set[str]] = None,
) -> tuple[int, int]:
    """Insert new/changed chunks and delete removed chunks against an existing
    index. When scope_file_paths is supplied, deletions are restricted to that
    set so vectors indexed by previous runs against other directories survive.

    Emits progress every 25 inserts / 50 deletes so long delta-syncs are never
    silent. Per-doc errors (e.g. metadata-too-long) are logged and skipped
    rather than aborting the whole run.
    """
    to_insert_ids, to_delete_ids = _diff_manifest_documents(
        current_manifest, persisted_manifest, scope_file_paths=scope_file_paths
    )

    deleted_count = 0
    if to_delete_ids:
        persisted_by_id = {
            entry["id"]: entry
            for entry in (persisted_manifest or {}).get("documents", [])
            if entry.get("id")
        }
        ref_doc_ids = sorted({
            persisted_by_id[doc_id].get("ref_doc_id") or doc_id
            for doc_id in to_delete_ids
            if doc_id in persisted_by_id
        })
        total = len(ref_doc_ids)
        print(f"[index] delta-sync: deleting {total} stale ref_doc(s) from Milvus...")
        for i, ref_id in enumerate(ref_doc_ids, start=1):
            try:
                index.delete_ref_doc(ref_id, delete_from_docstore=True)
                deleted_count += 1
            except Exception as exc:
                logger.warning(f"Failed to delete stale ref_doc {ref_id}: {exc}")
            if i % 50 == 0 or i == total:
                print(f"[index] delta-sync: deleted {i}/{total} ref_doc(s)")

    inserted_count = 0
    skipped_count = 0
    if to_insert_ids:
        insert_set = set(to_insert_ids)
        new_docs = [doc for doc in current_documents if doc.id_ in insert_set]
        total = len(new_docs)
        print(f"[index] delta-sync: embedding {total} new/changed chunk(s)...")
        for i, doc in enumerate(new_docs, start=1):
            try:
                index.insert(doc)
                inserted_count += 1
            except Exception as exc:
                skipped_count += 1
                meta = doc.metadata or {}
                logger.warning(
                    f"Skipped chunk {doc.id_} from "
                    f"{meta.get('file_name', meta.get('file_path', '?'))}: {exc}"
                )
            if i % 25 == 0 or i == total:
                print(
                    f"[index] delta-sync: embedded {i}/{total} chunk(s) "
                    f"(skipped={skipped_count})"
                )

    return inserted_count, deleted_count


def _reset_persistent_storage(persist_dir: str, vector_store: BasePydanticVectorStore) -> StorageContext:
    # Preserve the per-file extraction cache (and any other sibling caches) across
    # an index rebuild. Only delete files known to belong to the LlamaIndex storage
    # context + our manifest, NOT the whole directory tree.
    if os.path.isdir(persist_dir):
        preserved_subdirs = {"extraction_cache"}
        for name in os.listdir(persist_dir):
            full = os.path.join(persist_dir, name)
            if name in preserved_subdirs:
                continue
            try:
                if os.path.isdir(full) and not os.path.islink(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
            except OSError as exc:
                logger.warning(f"Could not remove {full} during persistent storage reset: {exc}")
    os.makedirs(persist_dir, exist_ok=True)
    return StorageContext.from_defaults(vector_store=vector_store)


def _clear_vector_store(vector_store: BasePydanticVectorStore) -> None:
    if hasattr(vector_store, "clear"):
        try:
            vector_store.clear()
        except Exception as e:
            logger.warning(f"Unable to clear vector store before rebuild: {e}")


def _flush_vector_store(vector_store: BasePydanticVectorStore) -> None:
    flush = getattr(vector_store, "flush", None)
    if callable(flush):
        flush()


def build_or_load_persistent_index(
    documents: List[Document],
    vector_store: BasePydanticVectorStore,
    persist_dir: str,
    embed_model: Optional[Any] = None,
    sync_existing: bool = False,
    rebuild: bool = False,
) -> tuple[VectorStoreIndex, StorageContext]:
    print(f"[index] Preparing {len(documents)} source documents for indexing...")
    for doc in documents:
        ensure_document_metadata(doc)

    print("[index] Splitting large documents into chunks...")
    split_documents = split_large_docs(documents)
    print(f"[index] Filtering low-information chunks (from {len(split_documents)} chunks)...")
    index_documents = filter_informative_documents(split_documents)
    print(f"[index] Building manifest signature for {len(index_documents)} chunks...")
    current_manifest = _build_index_manifest(index_documents, vector_store, embed_model)

    if not os.path.isdir(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)

    if rebuild:
        print("[index] Rebuild requested: clearing vector store and persistent storage...")
        _clear_vector_store(vector_store)
        storage_context = _reset_persistent_storage(persist_dir, vector_store)
    else:
        storage_context = None

    if storage_context is None:
        print("[index] Loading persistent storage context from disk...")
        is_simple_vector_store = isinstance(vector_store, SimpleVectorStore)
        try:
            if is_simple_vector_store:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            else:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=persist_dir,
                )
        except (FileNotFoundError, AssertionError, ValueError, OSError) as exc:
            print(f"[index] No usable persistent storage on disk ({type(exc).__name__}); starting fresh storage context.")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )

    print("[index] Comparing current document signature against persisted manifest...")
    persisted_manifest = _load_index_manifest(persist_dir)
    scope_file_paths = _scope_file_paths(current_manifest)
    structural_ok = _structural_signatures_match(current_manifest, persisted_manifest)
    # The scoped fast path must NEVER skip a rebuild when the embed model or
    # vector store schema changed — old vectors would be queried with a new
    # embedding model and return garbage. Gate it on structural_ok.
    manifest_ok = _manifest_matches(current_manifest, persisted_manifest) or (
        structural_ok
        and _scoped_manifests_match(current_manifest, persisted_manifest, scope_file_paths)
    )
    if not rebuild and not manifest_ok and not structural_ok:
        if persisted_manifest:
            print(
                "Persistent index manifest is structurally incompatible "
                "(embed model or vector store signature changed). "
                "Rebuilding active vector store and local metadata..."
            )
        else:
            print(
                "Persistent index manifest is missing. "
                "Building active vector store and local metadata..."
            )
        print("[index] Clearing existing vector store contents (this can take a while for large Milvus collections)...")
        _clear_vector_store(vector_store)
        print("[index] Resetting on-disk persistent storage (extraction_cache/ is preserved)...")
        storage_context = _reset_persistent_storage(persist_dir, vector_store)

    index_struct = None
    try:
        index_struct = storage_context.index_store.get_index_struct()
    except AssertionError:
        index_struct = None

    is_collection_store = (
        hasattr(vector_store, "collection_name")
        and hasattr(vector_store, "has_collection")
    )
    collection_exists = True
    if is_collection_store:
        try:
            collection_exists = bool(vector_store.has_collection())
        except Exception:
            collection_exists = False

    if index_struct is not None and collection_exists:
        persisted_embedding_count = _vector_store_embedding_count(vector_store)
        if persisted_embedding_count is None:
            persisted_embedding_count = _count_storage_context_embeddings(storage_context)
        expected_embedding_count = _count_storage_context_documents(storage_context)
        if expected_embedding_count is None:
            expected_embedding_count = len(index_documents)
        if persisted_embedding_count is not None and persisted_embedding_count != expected_embedding_count:
            store_label = "Milvus collection" if is_collection_store else "local vector store"
            print(
                f"Persistent index metadata found, but {store_label} has "
                f"{persisted_embedding_count}/{expected_embedding_count} vectors. "
                "Rebuilding vector store from source documents..."
            )
            _clear_vector_store(vector_store)
            storage_context = _reset_persistent_storage(persist_dir, vector_store)
            index_struct = None
        else:
            index = VectorStoreIndex(
                index_struct=index_struct,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            if manifest_ok:
                print("Loaded existing persistent index.")
            else:
                inserted, deleted = _apply_manifest_delta(
                    index,
                    index_documents,
                    current_manifest,
                    persisted_manifest,
                    scope_file_paths=scope_file_paths,
                )
                if inserted or deleted:
                    print(
                        f"Delta-synced persistent index: {inserted} chunk(s) added, "
                        f"{deleted} removed (scoped to current run; other-directory "
                        "vectors preserved)."
                    )
                    _flush_vector_store(vector_store)
                    storage_context.persist(persist_dir)
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
                    )
                else:
                    print("Loaded existing persistent index.")
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
                    )
            if sync_existing:
                print("Syncing existing index with current documents...")
                inserted = sync_documents_with_index(index, index_documents)
                if inserted:
                    print(f"Inserted {inserted} new documents into existing index.")
                    storage_context.persist(persist_dir)
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
                    )
            return index, storage_context

    if index_struct is not None and not collection_exists:
        print("Persistent index metadata found, but Milvus collection is missing. Rebuilding Milvus collection from source documents...")

    index = None
    docs_per_batch = int(os.environ.get("JARVIS_INDEX_BATCH_SIZE", "64"))
    docs_per_batch = max(docs_per_batch, 1)
    batches = chunk_documents(index_documents, docs_per_chunk=docs_per_batch)
    total_batches = len(batches)
    print(
        f"Embedding {len(index_documents)} chunks into the vector store in "
        f"{total_batches} batch(es) of up to {docs_per_batch}. This is the slow step "
        "on first build; subsequent runs reuse the index when the manifest matches."
    )
    previous_auto_flush = getattr(vector_store, "auto_flush", None)
    if previous_auto_flush is not None:
        setattr(vector_store, "auto_flush", False)
    skipped_in_rebuild = 0
    try:
        for batch_number, batch in enumerate(batches, start=1):
            if not batch:
                continue
            print(f"Indexing batch {batch_number}/{total_batches} ({len(batch)} documents)...")
            if index is None:
                # First batch establishes the index; embed docs one-by-one so a single
                # bad doc (e.g. metadata-too-long) can't abort the whole rebuild.
                # nodes=[] satisfies VectorStoreIndex's "one of nodes/objects/index_struct"
                # requirement without forcing a from_documents call that has no per-doc guard.
                index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
            for doc in batch:
                try:
                    index.insert(doc)
                except Exception as exc:
                    skipped_in_rebuild += 1
                    meta = doc.metadata or {}
                    logger.warning(
                        f"Skipped chunk {doc.id_} from "
                        f"{meta.get('file_name', meta.get('file_path', '?'))}: {exc}"
                    )

        if index is None:
            index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=embed_model,
            )
        if skipped_in_rebuild:
            print(
                f"[index] Rebuild completed with {skipped_in_rebuild} chunk(s) skipped "
                "(see warnings above)."
            )
        _flush_vector_store(vector_store)
        storage_context.persist(persist_dir)
        _write_index_manifest(
            persist_dir,
            _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
        )
    finally:
        if previous_auto_flush is not None:
            setattr(vector_store, "auto_flush", previous_auto_flush)

    return index, storage_context


def create_milvus_vector_store(
    collection_name: str = "jarvis_context",
    host: str = "127.0.0.1",
    port: str = "19530",
    dim: Optional[int] = None,
) -> MilvusVectorStore:
    return MilvusVectorStore(collection_name=collection_name, host=host, port=port, dim=dim)


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200):
    text = normalize_text(text)
    if not text:
        return []
    chunk_size = max(int(chunk_size), 1)
    chunk_overlap = max(min(int(chunk_overlap), chunk_size - 1), 0)
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    segments = [segment.strip() for segment in re.split(r"\n+", text) if segment.strip()]
    if len(segments) <= 1:
        segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    if len(segments) > 1:
        chunks: List[str] = []
        current_segments: List[str] = []
        current_word_count = 0

        def emit_current() -> None:
            nonlocal current_segments, current_word_count
            if not current_segments:
                return
            chunks.append("\n".join(current_segments).strip())
            if chunk_overlap:
                overlap_words = " ".join(current_segments).split()[-chunk_overlap:]
                current_segments = [" ".join(overlap_words)] if overlap_words else []
                current_word_count = len(overlap_words)
            else:
                current_segments = []
                current_word_count = 0

        for segment in segments:
            segment_words = segment.split()
            if len(segment_words) > chunk_size:
                emit_current()
                start = 0
                while start < len(segment_words):
                    end = min(start + chunk_size, len(segment_words))
                    chunks.append(" ".join(segment_words[start:end]))
                    if end == len(segment_words):
                        break
                    start = max(end - chunk_overlap, 0)
                current_segments = []
                current_word_count = 0
                continue

            if current_segments and current_word_count + len(segment_words) > chunk_size:
                emit_current()
            current_segments.append(segment)
            current_word_count += len(segment_words)

        emit_current()
        return chunks

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - chunk_overlap, 0)
    return chunks


def split_large_docs(documents: List[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Document]:
    split_docs: List[Document] = []
    for doc in documents:
        text = normalize_text(doc.text or "")
        if not text:
            continue
        metadata = dict(doc.metadata or {})
        metadata.setdefault("parent_file", metadata.get("file_path"))
        if len(text.split()) <= chunk_size:
            split_docs.append(_prepare_index_document(Document(text=text, metadata=metadata), chunk_index=metadata.get("chunk_index", 0)))
            continue
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "chunk_index": idx,
                "parent_file": chunk_metadata.get("file_path"),
            })
            split_docs.append(_prepare_index_document(Document(text=chunk, metadata=chunk_metadata), chunk_index=idx))
    return split_docs


def chunk_documents(documents: List[Document], docs_per_chunk: int = 40) -> List[List[Document]]:
    return [documents[i:i + docs_per_chunk] for i in range(0, len(documents), docs_per_chunk)]


def build_chunk_query_engines(
    documents: List[Document],
    docs_per_chunk: int = 40,
    llm: Optional[Any] = None,
    short_text_qa_template: Optional[Any] = None,
) -> List[Any]:
    if not documents:
        return []
    documents = split_large_docs(documents)
    doc_chunks = chunk_documents(documents, docs_per_chunk=docs_per_chunk)
    chunk_engines = []
    for chunk_idx, chunk_docs in enumerate(doc_chunks, 1):
        if not chunk_docs:
            continue
        index = VectorStoreIndex.from_documents(chunk_docs)
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            response_mode="compact",
            text_qa_template=short_text_qa_template,
        )
        chunk_engines.append((f"chunk_{chunk_idx}", query_engine, len(chunk_docs)))
    return chunk_engines


def summarize_partial_responses(query: str, partial_responses: List[str], llm: Any) -> str:
    combined_text = "\n\n".join(
        [f"PARTIAL ANSWER {idx + 1}:\n{response}" for idx, response in enumerate(partial_responses)]
    )
    summary_prompt = (
        f"You are an assistant that condenses multiple partial answers into a concise summary of the evidence. "
        f"The original question is: {query}\n\n"
        f"Here are the partial answers:\n\n{combined_text}\n\n"
        "Summarize the key facts, findings, and any disagreements or uncertainties. "
        "Do not answer the question yet; only return a clear, compact summary of the content."
    )
    summary_response = llm.chat([
        ChatMessage(role=MessageRole.USER, content=summary_prompt)
    ])
    return summary_response.message.content or ""


def stitch_chunk_responses(query: str, partial_responses: List[str], llm: Any) -> str:
    if not partial_responses:
        return ""
    if len(partial_responses) == 1:
        return partial_responses[0]

    evidence_summary = summarize_partial_responses(query, partial_responses, llm)
    final_prompt = (
        f"You are an assistant that produces a final answer using summarized evidence from document chunks. "
        f"The original question is: {query}\n\n"
        f"Here is the summarized evidence from the partial answers:\n\n{evidence_summary}\n\n"
        "Provide one concise and coherent final answer based on this evidence. "
        "If the evidence is incomplete, note what is missing. If there are contradictions, explain them briefly."
    )
    final_response = llm.chat([
        ChatMessage(role=MessageRole.USER, content=final_prompt)
    ])
    return final_response.message.content or ""


def _is_port_open(host: str, port: str, timeout: float = 2.0) -> bool:
    import socket as _socket
    try:
        with _socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


class ZipExtractor:
    def __init__(self, file_extractors):
        self.file_extractors = file_extractors

    def load_data(self, file_path, **kwargs):
        docs = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                for root, _, files in os.walk(tmpdirname):
                    for name in files:
                        ext = os.path.splitext(name)[1].lower()
                        extractor = self.file_extractors.get(ext)
                        if extractor:
                            extracted_docs = extractor.load_data(os.path.join(root, name), **kwargs)
                            docs.extend(extracted_docs)
        return docs


class ExcelReader:
    def load_data(self, file_path, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                text = normalize_text(df.to_string(index=False))
                if not text:
                    continue
                for chunk_index, chunk in enumerate(chunk_text(text)):
                    docs.append(Document(text=chunk, metadata={
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "sheet": sheet_name,
                        "chunk_index": chunk_index,
                        "extracted_by": "excel"
                    }))
        except Exception as e:
            logger.error(f"Failed to read Excel file {file_path}: {e}")
        return docs


from pptx import Presentation


class PowerPointReader:
    def load_data(self, file_path, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)
        try:
            prs = Presentation(file_path)
            for slide_index, slide in enumerate(prs.slides, start=1):
                slide_text = ""
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                slide_text = normalize_text(slide_text)
                if not slide_text:
                    continue
                for chunk_index, chunk in enumerate(chunk_text(slide_text)):
                    docs.append(Document(text=chunk, metadata={
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "slide_index": slide_index,
                        "chunk_index": chunk_index,
                        "extracted_by": "powerpoint"
                    }))
        except Exception as e:
            logger.error(f"Failed to read PowerPoint file {file_path}: {e}")
        return docs


class JpegExtractor:

    # -------------------------------------------------
    # CLASS-LEVEL SHARED MODELS
    # -------------------------------------------------

    _easyocr_reader = None

    _surya_det_model = None
    _surya_rec_model = None

    _trocr_processor = None
    _trocr_model = None

    _models_initialized = False

    # -------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------

    def __init__(self):

        if not JpegExtractor._models_initialized:

            logger.info("Initializing OCR models (one-time startup)...")

            # -----------------------------------------
            # EasyOCR
            # -----------------------------------------
            try:
                JpegExtractor._easyocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False
                )

                logger.info("EasyOCR initialized.")

            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")

            # -----------------------------------------
            # Surya
            # -----------------------------------------
            if surya_available:

                try:
                    JpegExtractor._surya_det_model = load_detection_model()

                    JpegExtractor._surya_rec_model = (
                        load_recognition_model()
                    )

                    logger.info("Surya OCR initialized.")

                except Exception as e:
                    logger.warning(
                        f"Surya initialization failed: {e}"
                    )

            # -----------------------------------------
            # TrOCR
            # -----------------------------------------
            if trocr_available:

                try:

                    logger.info(
                        "Loading TrOCR processor..."
                    )

                    JpegExtractor._trocr_processor = (
                        TrOCRProcessor.from_pretrained(
                            'microsoft/trocr-base-printed'
                        )
                    )

                    logger.info(
                        "Loading TrOCR model..."
                    )

                    JpegExtractor._trocr_model = (
                        VisionEncoderDecoderModel.from_pretrained(
                            'microsoft/trocr-base-printed'
                        )
                    )

                    logger.info("TrOCR initialized.")

                except Exception as e:
                    logger.warning(
                        f"TrOCR initialization failed: {e}"
                    )

            JpegExtractor._models_initialized = True

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------

    def _preprocess_image(self, image):

        cv_image = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY
        )

        h, w = gray.shape[:2]

        max_dim = max(h, w)

        if max_dim < 1600:

            scale = 1600 / max_dim

            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(
            enhanced,
            h=3
        )

        blurred = cv2.medianBlur(
            denoised,
            3
        )

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return Image.fromarray(thresh)

    # -------------------------------------------------
    # IMAGE LOAD
    # -------------------------------------------------

    def _load_image(self, file_path_str):

        try:

            image = Image.open(file_path_str)

            return image.convert("RGB")

        except Exception as e:

            logger.error(
                f"Unable to open image {file_path_str}: {e}"
            )

            return None

    # -------------------------------------------------
    # SURYA OCR
    # -------------------------------------------------

    def _ocr_with_surya(self, image):

        if (
            not surya_available
            or JpegExtractor._surya_det_model is None
            or JpegExtractor._surya_rec_model is None
        ):
            return None

        try:

            predictions = run_ocr(
                [image],
                [JpegExtractor._surya_det_model],
                [JpegExtractor._surya_rec_model],
                langs=['en']
            )

            if predictions and predictions[0]:

                text = "\n".join(
                    [line.text for line in predictions[0]]
                )

                return text

        except Exception as e:

            logger.error(f"Surya OCR failed: {e}")

        return None

    # -------------------------------------------------
    # TrOCR
    # -------------------------------------------------

    def _ocr_with_trocr(self, image):

        if (
            not trocr_available
            or JpegExtractor._trocr_processor is None
            or JpegExtractor._trocr_model is None
        ):
            return None

        try:

            pixel_values = (
                JpegExtractor._trocr_processor(
                    image,
                    return_tensors="pt"
                ).pixel_values
            )

            generated_ids = (
                JpegExtractor._trocr_model.generate(
                    pixel_values,
                    max_new_tokens=256
                )
            )

            generated_text = (
                JpegExtractor._trocr_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            )

            return generated_text

        except Exception as e:

            logger.error(f"TrOCR failed: {e}")

        return None

    # -------------------------------------------------
    # OCR QUALITY CHECK
    # -------------------------------------------------

    def _is_valid_ocr_text(self, text: str) -> bool:

        if not text:
            return False

        normalized = normalize_text(text)

        if len(normalized) < 20:

            keywords = [
                "dob",
                "date",
                "birth",
                "license",
                "name",
                "expiry",
                "issue",
                "address",
            ]

            lower = normalized.lower()

            if (
                any(keyword in lower for keyword in keywords)
                and len(re.findall(r"\w+", normalized)) >= 2
            ):
                return True

            return False

        if re.fullmatch(r"[\W_]+", normalized):
            return False

        tokens = re.findall(r"\w+", normalized)

        return len(tokens) >= 3

    # -------------------------------------------------
    # MAIN LOAD
    # -------------------------------------------------

    def load_data(self, file_path, **kwargs):

        file_path_str = str(file_path)

        file_name = os.path.basename(file_path)

        documents = []

        image = self._load_image(file_path_str)

        if image is None:
            return documents

        processed_image = self._preprocess_image(image)

        # -----------------------------------------
        # SURYA
        # -----------------------------------------

        text_surya = self._ocr_with_surya(
            processed_image
        )

        if text_surya and text_surya.strip():

            text = normalize_text(text_surya)

            for chunk_index, chunk in enumerate(
                chunk_text(text)
            ):

                documents.append(Document(
                    text=chunk,
                    metadata={
                        "file_path": file_path_str,
                        "file_name": file_name,
                        "chunk_index": chunk_index,
                        "extracted_by": "surya"
                    }
                ))

            return documents

        # -----------------------------------------
        # TrOCR
        # -----------------------------------------

        text_trocr = self._ocr_with_trocr(
            processed_image
        )

        if text_trocr and text_trocr.strip():

            if self._is_valid_ocr_text(text_trocr):

                text = normalize_text(text_trocr)

                for chunk_index, chunk in enumerate(
                    chunk_text(text)
                ):

                    documents.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "trocr"
                        }
                    ))

                return documents

            logger.info(
                f"TrOCR result for {file_name} "
                f"appears too short/low-quality; "
                f"falling back to additional OCR engines. "
                f"Output: {repr(text_trocr)}"
            )

        # -----------------------------------------
        # PYTESSERACT
        # -----------------------------------------

        pyt_docs = []

        try:

            text_pt = normalize_text(
                pytesseract.image_to_string(
                    processed_image
                )
            )

            if text_pt:

                for chunk_index, chunk in enumerate(
                    chunk_text(text_pt)
                ):

                    pyt_docs.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "pytesseract"
                        }
                    ))

        except Exception as e:

            logger.error(
                f"pytesseract failed on {file_path}: {e}"
            )

        # -----------------------------------------
        # EASYOCR
        # -----------------------------------------

        eo_docs = []

        try:

            image_array = np.array(processed_image)

            result = (
                JpegExtractor._easyocr_reader.readtext(
                    image_array,
                    detail=0
                )
            )

            text_eo = normalize_text(
                "\n".join(result)
            )

            if text_eo:

                for chunk_index, chunk in enumerate(
                    chunk_text(text_eo)
                ):

                    eo_docs.append(Document(
                        text=chunk,
                        metadata={
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "extracted_by": "easyocr"
                        }
                    ))

        except Exception as e:

            logger.error(
                f"easyocr failed on {file_path}: {e}"
            )

        documents.extend(pyt_docs)
        documents.extend(eo_docs)

        return documents


class TablePDFReader:
    def __init__(self):
        self._tabula_available = None
        self._tabula = None

    def _java_available(self) -> bool:
        if shutil.which("java") is None:
            return False
        try:
            result = subprocess.run(
                ["java", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_tabula_available(self):
        if self._tabula_available is not None:
            return self._tabula_available

        if not self._java_available():
            logger.info("Java/JDK is not installed; disabling tabula-py table extraction.")
            self._tabula_available = False
            return False

        try:
            import tabula
            self._tabula = tabula
            self._tabula_available = True
            return True
        except ImportError:
            logger.info("tabula-py is not installed; disabling tabula-py table extraction.")
            self._tabula_available = False
            return False
        except Exception as e:
            logger.info(f"tabula-py initialization failed: {e}. Disabling tabula-py table extraction.")
            self._tabula_available = False
            return False

    def _disable_tabula(self, reason: str):
        if self._tabula_available is not False:
            logger.info(f"Disabling tabula-py table extraction: {reason}")
        self._tabula_available = False
        self._tabula = None

    def load_data(self, file_path: str, **kwargs):
        docs = []
        file_name = os.path.basename(file_path)

        if not self._check_tabula_available():
            tabula = None
        else:
            tabula = self._tabula

        if tabula is not None:
            try:
                with suppress_java_warnings():
                    tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                for i, table in enumerate(tables):
                    if not table.empty:
                        table_text = table.to_string(index=False)
                        if table_text.strip():
                            docs.append(Document(text=table_text, metadata={
                                "file_path": file_path,
                                "file_name": file_name,
                                "page_number": None,
                                "table_index": i,
                                "extracted_by": "tabula"
                            }))
                if docs:
                    return docs
            except Exception as e:
                err = str(e).lower()
                if any(keyword in err for keyword in ["jvm dll not found", "jvm", "java", "jpype"]):
                    self._disable_tabula(
                        "Java/JVM is unavailable or incorrectly configured for tabula-py. "
                        "Install a valid JDK only if you need tabula table extraction."
                    )
                else:
                    logger.info(f"Note: tabula table extraction failed for {file_path}: {e}")

        try:
            with suppress_java_warnings():
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        page_tables = page.extract_tables()
                        for i, table in enumerate(page_tables):
                            if table:
                                table_text = pd.DataFrame(table).to_string(index=False)
                                if table_text.strip():
                                    docs.append(Document(text=table_text, metadata={
                                        "file_path": file_path,
                                        "file_name": file_name,
                                        "page_number": page_num,
                                        "table_index": i,
                                        "extracted_by": "pdfplumber_table"
                                    }))
        except Exception as e:
            logger.info(f"Note: pdfplumber table extraction failed for {file_path}: {e}")
        return docs


class CombinedPDFReader:
    def __init__(self):
        self.table_reader = TablePDFReader()
        self.easyocr_reader = None
        self.pdf_file_timeout_seconds = float(os.environ.get("JARVIS_PDF_FILE_TIMEOUT_SECONDS", "90"))

    def _get_easyocr_reader(self):
        if self.easyocr_reader is None:
            logger.info("Initializing PDF EasyOCR fallback model...")
            self.easyocr_reader = easyocr.Reader(['en'])
            logger.info("PDF EasyOCR fallback model initialized.")
        return self.easyocr_reader

    def _build_page_docs(self, text: str, metadata: dict, source: str):
        docs = []
        chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": chunk_index,
                "source": source,
            })
            docs.append(Document(text=chunk, metadata=chunk_meta))
        return docs

    def load_data(self, file_path: str, **kwargs):
        import pdfplumber
        all_docs = []
        file_name = os.path.basename(file_path)
        started = time.time()
        logger.info(f"Reading PDF: {file_name}")

        try:
            with time_limit(self.pdf_file_timeout_seconds, f"PDF extraction for {file_name}"):
                with suppress_java_warnings():
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, start=1):
                            page_text = normalize_text(page.extract_text())
                            page_metadata = {
                                "file_path": file_path,
                                "file_name": file_name,
                                "page_number": page_num,
                            }
                            if page_text:
                                all_docs.extend(self._build_page_docs(page_text, {**page_metadata, "extracted_by": "pdfplumber"}, "pdfplumber"))
                    table_docs = self.table_reader.load_data(file_path, **kwargs)
                    if table_docs:
                        all_docs.extend(table_docs)
        except TimeoutError as e:
            logger.warning(f"{e}. Returning partial content for {file_name}.")
        except Exception as e:
            logger.info(f"pdfplumber extraction error: {e}")

        if not all_docs:
            logger.info(f"No text extracted from {file_name} via text/table extraction. Attempting OCR fallback...")
            try:
                with time_limit(self.pdf_file_timeout_seconds, f"OCR extraction for {file_name}"):
                    with suppress_java_warnings():
                        with pdfplumber.open(file_path) as pdf:
                            for page_num, page in enumerate(pdf.pages, start=1):
                                page_metadata = {
                                    "file_path": file_path,
                                    "file_name": file_name,
                                    "page_number": page_num,
                                }
                                try:
                                    page_image = page.to_image(resolution=300).original.convert("RGB")
                                    image_array = np.array(page_image)
                                    result = self._get_easyocr_reader().readtext(image_array, detail=0)
                                    page_text = "\n".join(result)
                                    if page_text.strip():
                                        all_docs.extend(self._build_page_docs(page_text, {**page_metadata, "extracted_by": "easyocr"}, "easyocr"))
                                except Exception as e:
                                    logger.info(f"EasyOCR failed on page {page_num} of {file_path}: {e}")
            except TimeoutError as e:
                logger.warning(f"{e}. Returning partial OCR content for {file_name}.")
            except Exception as e:
                logger.info(f"OCR extraction error: {e}")

        if not all_docs:
            logger.warning(f"No content extracted from {file_path}")
        logger.info(
            f"Finished PDF: {file_name} -> {len(all_docs)} chunks in {time.time() - started:.1f}s"
        )
        return all_docs


class DeferredExtractor:
    def __init__(self, factory):
        self._factory = factory
        self._extractor = None

    def load_data(self, *args, **kwargs):
        if self._extractor is None:
            self._extractor = self._factory()
        return self._extractor.load_data(*args, **kwargs)


class DoclingReader:
    """Parse modern Office / structured documents via Docling and emit markdown
    chunks suitable for vector embeddings. Supports the Docling-native formats
    (.docx, .pptx, .xlsx, .html, .md, .csv, etc.); legacy binary formats
    (.doc/.xls/.ppt) and .rtf must be routed to a different reader. Falls back
    to UnstructuredReader when docling can't recognize the file (e.g. a legacy
    .doc renamed with a .docx extension)."""

    _converter = None
    _fallback = None

    @classmethod
    def _get_converter(cls):
        if cls._converter is None:
            from docling.document_converter import DocumentConverter
            cls._converter = DocumentConverter()
        return cls._converter

    @classmethod
    def _get_fallback(cls):
        if cls._fallback is None:
            cls._fallback = UnstructuredReader()
        return cls._fallback

    def _fallback_load(self, file_path_str, file_name, reason):
        logger.info(
            f"Docling could not parse {file_name} ({reason}); falling back to UnstructuredReader."
        )
        try:
            fb_docs = self._get_fallback().load_data(file_path_str) or []
        except Exception as e:
            logger.error(f"UnstructuredReader fallback failed for {file_path_str}: {e}")
            return []
        out: List[Document] = []
        for fb in fb_docs:
            text = normalize_text(getattr(fb, "text", "") or "")
            if not text:
                continue
            for chunk_index, chunk in enumerate(
                chunk_text(text, chunk_size=800, chunk_overlap=160)
            ):
                out.append(Document(text=chunk, metadata={
                    "file_path": file_path_str,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "extracted_by": "unstructured_fallback",
                }))
        return out

    def load_data(self, file_path, **kwargs):
        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        try:
            result = self._get_converter().convert(file_path_str)
            text = normalize_text(result.document.export_to_markdown())
        except Exception as e:
            return self._fallback_load(file_path_str, file_name, str(e))
        if not text:
            return self._fallback_load(file_path_str, file_name, "empty docling output")
        docs: List[Document] = []
        # ~800-word chunks with 160-word overlap fits comfortably in bge-m3's
        # 8192-token context window (~6x headroom) while keeping per-section
        # semantic focus. Going larger would dilute retrieval ranking; going
        # smaller would split section headers from their items. Send bigger
        # context to the LLM downstream via JARVIS_SOURCE_HINT_CHUNK_CHARS,
        # JARVIS_MAX_EVIDENCE_LINES, and JARVIS_RETRIEVAL_TOP_K instead.
        for chunk_index, chunk in enumerate(
            chunk_text(text, chunk_size=800, chunk_overlap=160)
        ):
            docs.append(Document(text=chunk, metadata={
                "file_path": file_path_str,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "extracted_by": "docling",
            }))
        return docs


jpeg_extractor = DeferredExtractor(JpegExtractor)
docling_reader = DoclingReader()
file_extractor = {
    ".pdf": CombinedPDFReader(),
    ".docx": docling_reader,
    ".doc": UnstructuredReader(),
    ".rtf": UnstructuredReader(),
    ".csv": UnstructuredReader(),
    ".txt": None,
    ".jpeg": jpeg_extractor,
    ".jpg": jpeg_extractor,
    ".png": jpeg_extractor,
    ".gif": jpeg_extractor,
    ".tif": jpeg_extractor,
    ".tiff": jpeg_extractor,
    ".bmp": jpeg_extractor,
    ".xlsx": docling_reader,
    ".xls": ExcelReader(),
    ".pptx": docling_reader,
    ".ppt": UnstructuredReader(),
}
file_extractor[".zip"] = ZipExtractor(file_extractor)


def has_files_with_extensions(directory, extensions):
    if not os.path.exists(directory):
        return False
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                return True
    return False


EXTRACTION_CACHE_VERSION = 1


def _extraction_cache_path(cache_dir: str, file_path: str) -> str:
    key = hashlib.sha256(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def _serialize_documents_for_cache(docs: List[Document]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        safe_meta: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, os.PathLike):
                safe_meta[str(k)] = os.fspath(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                safe_meta[str(k)] = v
            else:
                safe_meta[str(k)] = str(v)
        out.append({
            "id_": doc.id_,
            "text": doc.text or "",
            "metadata": safe_meta,
        })
    return out


def _deserialize_documents_from_cache(entries: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for entry in entries or []:
        doc = Document(
            text=entry.get("text", "") or "",
            metadata=dict(entry.get("metadata") or {}),
        )
        cached_id = entry.get("id_")
        if cached_id:
            doc.id_ = cached_id
        docs.append(doc)
    return docs


def _walk_files(input_dir: str, extensions: List[str], recursive: bool = True) -> List[str]:
    matched: List[str] = []
    if not os.path.exists(input_dir):
        return matched
    lowered = tuple(ext.lower() for ext in extensions)
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(lowered):
                    matched.append(os.path.join(root, file))
    else:
        for name in os.listdir(input_dir):
            full = os.path.join(input_dir, name)
            if os.path.isfile(full) and name.lower().endswith(lowered):
                matched.append(full)
    matched.sort()
    return matched


def load_documents_with_cache(
    input_dir: str,
    required_exts: List[str],
    file_extractor: Dict[str, Any],
    cache_dir: str,
    recursive: bool = True,
) -> List[Document]:
    """Extract documents from a directory with per-file (path, mtime, size) caching.

    On cache hit, the original Documents are restored from disk without
    re-running OCR / PDF parsing. On miss or stale (mtime or size changed),
    the file is re-extracted via SimpleDirectoryReader using the provided
    file_extractor mapping, and the cache entry is rewritten.

    Set JARVIS_DISABLE_EXTRACTION_CACHE=1 to bypass the cache.
    """
    from llama_index.core import SimpleDirectoryReader  # local import to avoid cycles

    if os.environ.get("JARVIS_DISABLE_EXTRACTION_CACHE", "0") == "1":
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            recursive=recursive,
            required_exts=required_exts,
            file_extractor=file_extractor,
        )
        return reader.load_data()

    os.makedirs(cache_dir, exist_ok=True)
    files = _walk_files(input_dir, required_exts, recursive=recursive)
    if not files:
        return []

    all_docs: List[Document] = []
    cache_hits = 0
    cache_misses = 0
    extract_failures = 0

    for file_path in files:
        try:
            stat = os.stat(file_path)
        except OSError as exc:
            print(f"[jarvis] Skipping unreadable file {file_path}: {exc}")
            continue
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        size = stat.st_size
        cache_path = _extraction_cache_path(cache_dir, file_path)
        cached: Optional[Dict[str, Any]] = None
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
            except Exception:
                cached = None

        if (
            cached
            and cached.get("version") == EXTRACTION_CACHE_VERSION
            and cached.get("mtime_ns") == mtime_ns
            and cached.get("size") == size
            and cached.get("file_path") == os.path.abspath(file_path)
        ):
            docs = _deserialize_documents_from_cache(cached.get("documents") or [])
            all_docs.extend(docs)
            cache_hits += 1
            continue

        cache_misses += 1
        if JARVIS_DEBUG:
            if cached is None:
                miss_reason = "no cache file"
            else:
                reasons = []
                if cached.get("version") != EXTRACTION_CACHE_VERSION:
                    reasons.append(f"version({cached.get('version')}!={EXTRACTION_CACHE_VERSION})")
                if cached.get("mtime_ns") != mtime_ns:
                    reasons.append(f"mtime({cached.get('mtime_ns')}!={mtime_ns})")
                if cached.get("size") != size:
                    reasons.append(f"size({cached.get('size')}!={size})")
                if cached.get("file_path") != os.path.abspath(file_path):
                    reasons.append(f"path({cached.get('file_path')}!={os.path.abspath(file_path)})")
                miss_reason = ",".join(reasons) or "unknown"
            print(f"[debug] extraction-cache MISS {os.path.basename(file_path)}: {miss_reason}")
        try:
            single_file_extractor = {
                ext: reader for ext, reader in (file_extractor or {}).items() if reader is not None
            }
            reader = SimpleDirectoryReader(
                input_files=[file_path],
                file_extractor=single_file_extractor,
            )
            docs = reader.load_data()
        except Exception as exc:
            extract_failures += 1
            print(f"[jarvis] Extraction failed for {file_path}: {exc}")
            continue

        all_docs.extend(docs)
        payload = {
            "version": EXTRACTION_CACHE_VERSION,
            "file_path": os.path.abspath(file_path),
            "mtime_ns": mtime_ns,
            "size": size,
            "documents": _serialize_documents_for_cache(docs),
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception as exc:
            print(f"[jarvis] Could not write extraction cache for {file_path}: {exc}")

    print(
        f"[jarvis] Extraction cache: {cache_hits} hit, {cache_misses} miss"
        + (f", {extract_failures} failed" if extract_failures else "")
        + f" (out of {len(files)} files)."
    )
    return all_docs


# ---------------------------------------------------------------------------
# Query handling pipeline
# ---------------------------------------------------------------------------
@dataclass
class QueryRuntime:
    """Bundle of the initialized runtime objects `handle_query` needs.

    The main application (`Jarvis.py`) builds the index/retriever/LLM/TTS once at
    startup and passes them in here, keeping the entry script thin while the full
    RAG + fallback pipeline lives in this helpers module."""
    all_docs: list
    llm: object
    doc_retriever: object | None
    doc_chat_engine: object | None = None
    doc_index: object | None = None
    storage_context: object | None = None
    response_mode: str = "query"
    tts: object | None = None
    persist_dir: str = "jarvis_storage"


class _SilentTTS:
    """Fallback no-op TTS used when a runtime has no speech manager attached."""

    def speak(self, _text: str) -> None:  # pragma: no cover - trivial
        pass


def handle_query(
    user_query: str,
    runtime: QueryRuntime,
    prior_turn: dict[str, object] | None = None,
) -> dict[str, object]:
    """Process a single user query through the full RAG + fallback pipeline.

    Stateless: `prior_turn` supplies the previous conversational turn (or None /
    {} for a fresh start) and the new turn dict is returned, so callers (the REPL
    loop, smoke tests) can chain turns themselves. Uses the cached
    index/retriever/LLM carried on `runtime` — no re-initialization."""
    last_turn: dict[str, object] = dict(prior_turn) if prior_turn else {}
    tts = runtime.tts or _SilentTTS()

    print("\n[📄 Context Results]")
    try:
        effective_user_query, is_follow_up, source_hints = _effective_query_for_turn(user_query, last_turn)
        if is_follow_up:
            debug_print("[debug] Treated user input as a follow-up clarification.")
            debug_print(f"[debug] Effective query: {effective_user_query}")
        if source_hints:
            debug_print(f"[debug] Source hints: {source_hints}")
        negative_evidence_feedback = _is_negative_evidence_feedback(user_query)
        if negative_evidence_feedback:
            debug_print("[debug] User feedback rejected previous evidence; not merging prior evidence lines.")

        if _query_requests_corpus_details(user_query):
            response_text = _build_corpus_details_preview(runtime.all_docs)
            debug_print("[debug] Used direct corpus details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts.speak(_spoken_response_text(response_text))
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if _query_requests_source_details(user_query):
            response_text = _build_source_details_preview(effective_user_query, runtime.all_docs, runtime.llm)
            debug_print("[debug] Used direct source details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts.speak(_spoken_response_text(response_text))
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if runtime.response_mode == "chat":
            if runtime.doc_chat_engine is None:
                raise RuntimeError("No chat engine available")

            response = runtime.doc_chat_engine.chat(user_query)
            response_text = str(response).strip()
            debug_print("[debug] Used legacy chat response mode.")
            if _response_is_dismissive(response_text):
                if os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] Chat-mode response was dismissive; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, runtime.llm)
                    if gk_answer:
                        response_text = gk_answer
                    else:
                        response_text = (
                            "I could not find a clear answer in the indexed documents. "
                            "Please try rephrasing the question or asking for the relevant source file first."
                        )
                else:
                    response_text = (
                        "I could not find a clear answer in the indexed documents. "
                        "Please try rephrasing the question or asking for the relevant source file first."
                    )
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts.speak(_spoken_response_text(response_text))
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if runtime.doc_retriever is None:
            raise RuntimeError("No retriever available")

        evidence_lines, source_nodes = _retrieve_query_supported_evidence(effective_user_query, runtime.doc_retriever)
        evidence_lines = _evidence_lines_for_turn(
            evidence_lines,
            last_turn,
            is_follow_up,
            negative_evidence_feedback,
            source_hints,
        )
        debug_print(f"[debug] Retrieved source chunks: {len(source_nodes)}")
        debug_print(f"[debug] Query-supported evidence lines: {len(evidence_lines)}")
        for evidence_line in evidence_lines:
            debug_print(f"[debug] Evidence: {evidence_line}")
        if evidence_lines:
            phrase_anchors = _query_phrase_anchors(effective_user_query)
            page_hints = _query_page_hints(effective_user_query)
            latest_turn_is_reasoning = _query_requests_reasoning(
                _latest_user_turn(effective_user_query)
            )
            allow_keyword_extractor = (
                not phrase_anchors
                and not page_hints
                and not latest_turn_is_reasoning
            )
            response_text = (
                _extractive_value_answer_from_evidence(effective_user_query, evidence_lines)
                if allow_keyword_extractor
                else ""
            )
            if response_text:
                review_passed = True
                review_reason = "answered directly from retrieved value evidence"
                debug_print("[debug] Used extractive value answer from retrieved evidence.")
            else:
                if not allow_keyword_extractor:
                    debug_print(
                        "[debug] Skipped keyword extractor due to phrase/page hints or reasoning intent; routing to LLM review loop."
                    )
                doc_scan_answer = (
                    _extractive_value_answer_from_documents(effective_user_query, runtime.all_docs)
                    if allow_keyword_extractor
                    else ""
                )
                if doc_scan_answer:
                    response_text = doc_scan_answer
                    review_passed = True
                    review_reason = "answered directly from local value evidence scan"
                    debug_print("[debug] Used extractive value answer from local document scan.")
                else:
                    prior_answer_for_refinement = ""
                    if is_follow_up:
                        prior_answer_for_refinement = str(last_turn.get("response_text") or "")
                    response_text, review_passed, review_reason = _answer_with_review_loop(
                        effective_user_query,
                        evidence_lines,
                        runtime.llm,
                        prior_answer=prior_answer_for_refinement,
                    )
        else:
            response_text = _extractive_value_answer_from_documents(effective_user_query, runtime.all_docs)
            if response_text:
                review_passed = True
                review_reason = "answered directly from local value evidence scan"
                debug_print("[debug] Used extractive value answer from local document scan.")
            else:
                response_text = ""
                review_passed = False
                review_reason = "no query-supported evidence was retrieved"

        answer_was_dismissive = bool(response_text) and _response_is_dismissive(response_text)

        if review_passed and answer_was_dismissive:
            debug_print("[debug] Reviewed answer is dismissive; demoting to trigger fallback chain.")
            review_passed = False
            review_reason = review_reason or "reviewed answer indicated no information in evidence"
            response_text = ""

        if not review_passed:
            debug_print(f"[debug] Initial answer did not pass review: {review_reason}")
            if evidence_lines and not answer_was_dismissive:
                debug_print("[debug] Returning evidence-present fallback instead of no-evidence response.")
                response_text = _evidence_present_fallback(evidence_lines, review_reason)
            else:
                if answer_was_dismissive:
                    debug_print("[debug] Evidence chunks were rejected by the answerer as irrelevant; bypassing evidence fallback.")
                topic_preview = _build_topic_details_preview(effective_user_query, runtime.all_docs, runtime.llm)
                if topic_preview and not _response_is_dismissive(topic_preview):
                    debug_print("[debug] Applied direct topic corpus fallback.")
                    response_text = topic_preview
                elif os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] No corpus evidence found; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, runtime.llm)
                    response_text = gk_answer or _not_found_answer(effective_user_query, review_reason)
                else:
                    response_text = _not_found_answer(effective_user_query, review_reason)
        else:
            debug_print("[debug] Returning reviewed answer.")

        print(f"Jarvis: {_format_response_for_display(response_text)}\n")
        tts.speak(_spoken_response_text(response_text))
        last_turn = {
            "user_query": user_query,
            "effective_query": effective_user_query,
            "response_text": response_text,
            "evidence_lines": evidence_lines,
        }

        if os.environ.get("JARVIS_ENABLE_CHAT_MEMORY", "0") == "1" and runtime.doc_index is not None and runtime.storage_context is not None:
            if not review_passed or _response_is_dismissive(response_text):
                debug_print("[debug] Skipping chat memory persistence (response was a fallback or not reviewed).")
            else:
                chat_doc = Document(
                    text=f"User: {user_query}\nAssistant: {response_text}",
                    metadata={
                        "source": "chat_memory",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
                try:
                    runtime.doc_index.insert(chat_doc)
                    runtime.storage_context.persist(runtime.persist_dir)
                except Exception as e:
                    print(f"Warning: chat memory persistence failed: {e}")
    except Exception as e:
        print("Context engine failed:", e)
        tts.speak("Sorry, I encountered an error processing your request.")
    return last_turn
