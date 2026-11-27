"""Answers about the indexed corpus itself: topic, source, and corpus previews.
"""

import os
import re

from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama

from jarvisrag.config import debug_print, JARVIS_DEBUG
from jarvisrag.retrieval import (
    _query_candidate_score,
    _query_evidence_is_supported,
    _query_focused_snippet,
    _source_hint_chunk_chars,
)
from jarvisrag.text.query import (
    _best_detail_candidate,
    _best_detail_snippet,
    _contains_all_terms,
    _document_search_text,
    _extract_source_hints,
    _query_anchor_terms,
    _query_requests_values,
    _query_terms,
    _query_topic_terms,
)
from jarvisrag.text.textproc import is_low_information_text
from jarvisrag.answer.fallbacks import (
    _evidence_present_fallback,
    _extractive_summary_answer_from_evidence,
    _not_found_answer,
)
from jarvisrag.answer.synthesis import _answer_with_review_loop


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
                "Lower JARVIS_OCR_VOCAB_THRESHOLD and re-run with JARVIS_REBUILD_INDEX=1 to "
                "retain noisier text from this file."
            )
        return (
            f"I did not find extracted indexed text for {', '.join(source_hints)}. "
            "Verify the filename and that the file exists in the indexed directory."
        )

    # When the user explicitly names a file, send full chunk text (capped at
    # JARVIS_SOURCE_HINT_CHUNK_CHARS, default 16000) ordered by chunk_index so
    # sections like "Certifications" that sit mid-chunk aren't cropped out by
    # the generic detail-density scorer. Matches the source-hint force-include
    # behavior in _source_nodes_to_query_evidence_lines.
    max_chars = _source_hint_chunk_chars()
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
