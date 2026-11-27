"""Document extraction and ingestion for Jarvis.

The ``file_extractor`` extension -> reader map and the directory loader with
per-file (path, mtime, size) extraction caching. The readers live in
``office``, ``pdf``, ``images``, and ``mail``; the cache format in ``cache``.
"""

import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document
from llama_index.readers.file.unstructured import UnstructuredReader

from jarvisrag.config import JARVIS_DEBUG
from jarvisrag.extraction.indexing import chunk_text
from jarvisrag.extraction.office import (
    DeferredExtractor,
    DoclingReader,
    ExcelReader,
    MHTMLReader,
    ZipExtractor,
)
from jarvisrag.extraction.pdf import CombinedPDFReader
from jarvisrag.extraction.images import JpegExtractor
from jarvisrag.extraction.mail import EmlxReader
from jarvisrag.extraction.cache import (
    _deserialize_documents_from_cache,
    _extraction_cache_path,
    _extraction_mode_signature,
    _extractor_signature,
    _serialize_documents_for_cache,
    EXTRACTION_CACHE_VERSION,
)


jpeg_extractor = DeferredExtractor(JpegExtractor)


docling_reader = DoclingReader()


mhtml_reader = MHTMLReader(docling_reader)


file_extractor = {
    ".pdf": CombinedPDFReader(),
    ".docx": docling_reader,
    ".doc": UnstructuredReader(),
    ".rtf": UnstructuredReader(),
    ".csv": UnstructuredReader(),
    ".txt": None,
    ".html": docling_reader,
    ".htm": docling_reader,
    ".mhtml": mhtml_reader,
    ".mht": mhtml_reader,
    ".emlx": EmlxReader(),
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


# A sentence terminator: . ! or ? plus any closing quote/bracket, then whitespace.
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s")


_TEXT_ENDS_SENTENCE_RE = re.compile(r"[.!?][\"')\]]*$")


# Carry at most what chunk_text already overlaps between neighbouring chunks.
# Read from its signature: indexing.py's bytes are part of the mail ledger's
# preparation signature, so editing that file would re-embed every mail vector.
_PAGE_TAIL_MAX_WORDS = inspect.signature(chunk_text).parameters["chunk_overlap"].default


def _unfinished_tail(text: str, max_words: int) -> str:
    """The trailing text after the last finished sentence ('' if the text ends one)."""
    stripped = (text or "").rstrip()
    if not stripped or _TEXT_ENDS_SENTENCE_RE.search(stripped):
        return ""
    last = None
    for last in _SENTENCE_END_RE.finditer(stripped):
        pass
    tail = stripped[last.end():] if last else stripped
    return " ".join(tail.split()[-max_words:])


def carry_unfinished_page_tails(
    docs: List[Document], max_words: int = _PAGE_TAIL_MAX_WORDS
) -> List[Document]:
    """Prefix each page's first chunk with the unfinished tail of the page before.

    Pages are chunked one at a time, so a heading, table caption or sentence
    that ends a page is cut off from the text it introduces on the next page
    (e.g. a table title at the bottom of one page, its rows on the next). Only
    page-numbered text-flow chunks take part; tables (``table_index`` or no page)
    are left alone. Returns new Document objects for the changed chunks; the
    inputs are never mutated, so cached extractions stay raw.
    """
    flows: Dict[Any, Dict[int, List[int]]] = {}
    for position, doc in enumerate(docs):
        metadata = doc.metadata or {}
        page = metadata.get("page_number")
        if not isinstance(page, int) or "table_index" in metadata:
            continue
        pages = flows.setdefault(metadata.get("file_path"), {})
        pages.setdefault(page, []).append(position)
    if not flows:
        return docs

    out = list(docs)
    for pages in flows.values():
        for positions in pages.values():
            positions.sort(key=lambda i: (docs[i].metadata or {}).get("chunk_index", 0))
        for page in sorted(pages):
            previous = pages.get(page - 1)
            if not previous:
                continue
            tail = _unfinished_tail(docs[previous[-1]].text, max_words)
            if not tail:
                continue
            first = docs[pages[page][0]]
            out[pages[page][0]] = Document(
                text=f"{tail}\n{first.text}",
                metadata=dict(first.metadata or {}),
                id_=first.id_,
            )
    return out


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
        return carry_unfinished_page_tails(reader.load_data())

    os.makedirs(cache_dir, exist_ok=True)
    files = _walk_files(input_dir, required_exts, recursive=recursive)
    if not files:
        return []

    all_docs: List[Document] = []
    cache_hits = 0
    cache_misses = 0
    extract_failures = 0
    mode_sig = _extraction_mode_signature()

    for file_path in files:
        try:
            stat = os.stat(file_path)
        except OSError as exc:
            print(f"[jarvis] Skipping unreadable file {file_path}: {exc}")
            continue
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        size = stat.st_size
        extractor_sig = _extractor_signature(file_path)
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
            and cached.get("extraction_mode") == mode_sig
            and (not extractor_sig or cached.get("extractor_sig") == extractor_sig)
            and cached.get("mtime_ns") == mtime_ns
            and cached.get("size") == size
            and cached.get("file_path") == os.path.abspath(file_path)
        ):
            docs = _deserialize_documents_from_cache(cached.get("documents") or [])
            all_docs.extend(carry_unfinished_page_tails(docs))
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
                if cached.get("extraction_mode") != mode_sig:
                    reasons.append(f"mode({cached.get('extraction_mode')}!={mode_sig})")
                if extractor_sig and cached.get("extractor_sig") != extractor_sig:
                    reasons.append(f"extractor({cached.get('extractor_sig')}!={extractor_sig})")
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

        all_docs.extend(carry_unfinished_page_tails(docs))
        # The cache keeps the raw extraction; the carry is reapplied on every load.
        payload = {
            "version": EXTRACTION_CACHE_VERSION,
            "extraction_mode": mode_sig,
            "extractor_sig": extractor_sig,
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
