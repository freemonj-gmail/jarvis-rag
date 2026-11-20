"""Document preparation and chunking for Jarvis.

Owns everything that shapes the chunks that get embedded:
- low-information / OCR-garbage filtering and source-window grouping,
- stable document-id assignment and metadata normalisation,
- word-budget chunking helpers (``chunk_text`` / ``split_large_docs`` /
  ``chunk_documents``).

The persistent index lifecycle (manifests, delta-sync, rebuilds) lives in
:mod:`jarvisrag.extraction.persistence`. Keep it out of this module: the mail
ledger hashes this file's source as its preparation signature, so any edit
here re-prepares and re-embeds every mail chunk.
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document

from jarvisrag.config import debug_print
from jarvisrag.text.textproc import (
    _is_ocr_garbage_text,
    is_low_information_text,
    normalize_text,
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


# Metadata is concatenated into the text the embedder and the LLM see, so it
# competes with the document's own content for the node parser's chunk budget.
# A value large enough to fill that budget makes the parser reject the chunk
# outright, so it never reaches the index. Readers vary in how much they attach
# (raw HTML, per-element dumps), so cap by measured size rather than by key
# name: the rule then holds for any reader and any corpus.
#
# The budget is in characters and the parser's is in tokens, so an equal number
# leaves metadata using roughly a quarter of the chunk. Measured on a mixed
# corpus, this trimmed only the pathological values; lowering it started
# stripping paths and ids that cost recall without fixing anything.
_METADATA_EMBED_BUDGET_CHARS = int(
    os.environ.get("JARVIS_METADATA_EMBED_BUDGET_CHARS", "1024")
)


def _limit_embedded_metadata(doc: Document, budget: Optional[int] = None) -> None:
    """Exclude oversized metadata values from the embedded/LLM text.

    Excluded keys stay in ``doc.metadata`` so they remain stored and filterable;
    only the string built for embedding and prompting shrinks. Largest values go
    first, which leaves compact identifying keys (file name, page) intact.
    """
    budget = _METADATA_EMBED_BUDGET_CHARS if budget is None else budget
    metadata = doc.metadata or {}
    if not metadata or budget < 0:
        return

    # Mirrors the "key: value" line the node parser measures per metadata entry.
    sizes = {key: len(f"{key}: {value}") for key, value in metadata.items()}
    excluded = set(doc.excluded_embed_metadata_keys or [])
    excluded |= set(doc.excluded_llm_metadata_keys or [])
    remaining = sum(size for key, size in sizes.items() if key not in excluded)
    if remaining <= budget:
        return

    for key, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        if remaining <= budget:
            break
        if key in excluded:
            continue
        excluded.add(key)
        remaining -= size

    # The parser measures the longer of the embed and LLM strings, so trim both.
    doc.excluded_embed_metadata_keys = sorted(excluded)
    doc.excluded_llm_metadata_keys = sorted(excluded)


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
    _limit_embedded_metadata(prepared)
    return prepared



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

        def emit_current(next_segment_size: int = 0) -> None:
            nonlocal current_segments, current_word_count
            if not current_segments:
                return
            chunks.append("\n".join(current_segments).strip())
            # Carry only as much overlap as still leaves room for the next segment.
            allowance = min(chunk_overlap, max(chunk_size - next_segment_size, 0))
            overlap_words = " ".join(current_segments).split()[-allowance:] if allowance else []
            current_segments = [" ".join(overlap_words)] if overlap_words else []
            current_word_count = len(overlap_words)

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
                emit_current(len(segment_words))
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
