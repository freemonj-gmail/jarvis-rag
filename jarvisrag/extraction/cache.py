"""The per-file extraction cache: versions, signatures, and serialization.

A cached extraction is reused only while the file's (path, mtime, size), the
cache version, and the extraction-mode signature all still match.
"""

import hashlib
import os
from typing import Any, Dict, List

from llama_index.core.schema import Document

from jarvisrag.config import (
    JARVIS_APPLE_OCR_LANGS,
    JARVIS_APPLE_OCR_LEVEL,
    JARVIS_ENABLE_APPLE_OCR,
    JARVIS_ENABLE_PAGE_PIPELINE,
    JARVIS_ENABLE_SURYA,
    JARVIS_ENABLE_VLM,
    JARVIS_OCR_ESCALATE_ON_FAILURE,
    JARVIS_OCR_MIN_CONF,
    JARVIS_PDF_MIN_CHARS_PER_PAGE,
    JARVIS_PDF_OCR_DPI,
    JARVIS_VLM_CAPTION,
    JARVIS_VLM_MODEL,
)


# Bump whenever extraction OUTPUT changes in a way not captured by
# _extraction_mode_signature() (e.g. VLM dtype/preprocessing/generation changes
# in jarvisrag/extraction/vlm.py). v3: SmolVLM2 float32 + EXIF transpose +
# downscale + repetition controls. v4: Apple Vision OCR backend on the text path.
# v5: Apple Vision OCR row-band assembly (preserves table rows/columns).
# v6: document-level OCR validator + escalation on whole-page failure.
# v7: HTML (.html/.htm via Docling) + MHTML (.mhtml/.mht) extraction support.
EXTRACTION_CACHE_VERSION = 7


# Bump when EMAIL (.emlx) extraction OUTPUT changes without a global
# EXTRACTION_CACHE_VERSION bump, so ONLY email caches invalidate (a global bump
# would needlessly re-OCR every PDF/image). v1: HTML body cleaning -- strip
# <style>/<script> and their contents, decode entities, drop zero-width chars,
# collapse whitespace.
EMAIL_EXTRACTION_VERSION = 1


_EMAIL_EXTENSIONS = (".emlx",)


def _extractor_signature(file_path: str) -> str:
    """A per-file extractor signature that varies only for file types whose
    reader output can change independently of the global cache version. Empty
    for every other file so unrelated caches are never invalidated."""
    if file_path.lower().endswith(_EMAIL_EXTENSIONS):
        return f"emlx{EMAIL_EXTRACTION_VERSION}"
    return ""


def _extraction_cache_path(cache_dir: str, file_path: str) -> str:
    key = hashlib.sha256(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def _extraction_mode_signature() -> str:
    """Signature of the settings that change extraction OUTPUT.

    Folded into the cache validation so that toggling the page pipeline, Surya,
    or the OCR/native-text thresholds automatically invalidates stale cached
    extractions (otherwise a prior run's docs would be replayed under a new
    mode, masking the change).
    """
    return (
        f"pipeline={int(JARVIS_ENABLE_PAGE_PIPELINE)};"
        f"surya={int(JARVIS_ENABLE_SURYA)};"
        f"minconf={JARVIS_OCR_MIN_CONF};"
        f"minchars={JARVIS_PDF_MIN_CHARS_PER_PAGE};"
        f"dpi={JARVIS_PDF_OCR_DPI};"
        f"vlm={int(JARVIS_ENABLE_VLM)};"
        f"vlmmodel={JARVIS_VLM_MODEL if JARVIS_ENABLE_VLM else ''};"
        f"vlmcaption={int(JARVIS_VLM_CAPTION)};"
        f"appleocr={int(JARVIS_ENABLE_APPLE_OCR)};"
        f"applelangs={JARVIS_APPLE_OCR_LANGS if JARVIS_ENABLE_APPLE_OCR else ''};"
        f"applelevel={JARVIS_APPLE_OCR_LEVEL if JARVIS_ENABLE_APPLE_OCR else ''};"
        f"ocrescalate={int(JARVIS_OCR_ESCALATE_ON_FAILURE)}"
    )


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
