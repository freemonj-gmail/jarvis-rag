"""How an answer is printed: the answer body and its References block.
"""

import re

from jarvisrag.speech import _INLINE_CITATION_RE, _SOURCE_REFERENCE_RE, _SPEECH_DOC_RUN_RE


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


def _strip_citations_for_display(text: str) -> str:
    """Remove inline source citations from the answer BODY for display.

    Fully parenthetical citations like "(Resume.docx, page 3)" or "(page 12)"
    are deleted outright; bare filenames woven into sentence grammar ("listed in
    Application.pdf") are replaced with the neutral phrase "the document(s)" so
    the prose stays grammatical. Factual content (values, dates, names) is kept
    verbatim — only the source labels are removed, since they are listed once in
    the dedicated References block underneath. This mirrors the TTS citation
    suppression so the printed and spoken answers read consistently.
    """
    if not text:
        return text
    # 1. Drop fully parenthetical citations wholesale (incl. the parentheses).
    cleaned = _INLINE_CITATION_RE.sub("", text)
    # 2. Neutralize bare in-prose filenames (+ any trailing page) to a placeholder.
    cleaned = _SOURCE_REFERENCE_RE.sub("the document", cleaned)
    cleaned = _SPEECH_DOC_RUN_RE.sub("the documents", cleaned)
    # 3. Recapitalize the placeholder when it begins a sentence or line.
    cleaned = re.sub(
        r"(^|[.!?]\s+|\n)\s*the (documents?)\b",
        lambda m: f"{m.group(1)}The {m.group(2)}",
        cleaned,
    )
    # 4. Tidy artifacts left by removal: space before punctuation, space runs.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _excluded_reference_lines(excluded_files: list[str] | None, max_items: int = 25) -> list[str]:
    """Render launch-folder files that physically exist but produced no indexed
    chunk (unreadable / no extractable data) as annotated References entries, so
    they are surfaced next to the real sources rather than in the answer body.
    Returns an empty list when there is nothing to report."""
    if not excluded_files:
        return []
    shown = excluded_files[:max_items]
    lines = [
        f"{name} — present but not indexed (no extractable data)"
        for name in shown
    ]
    extra = len(excluded_files) - len(shown)
    if extra > 0:
        lines.append(f"...and {extra} more present but not indexed (no extractable data)")
    return lines


def _home_files_in_evidence(
    evidence_lines: list[str] | None, home_indexed_files: list[str] | None
) -> list[str]:
    """Return the launch-folder file basenames that actually appear in the
    retrieved evidence fed to the model this turn. These are genuine sources for
    the answer and must be representable in References regardless of whether the
    LLM happened to type their (often space-containing) filename in prose."""
    if not evidence_lines or not home_indexed_files:
        return []
    return [name for name in home_indexed_files if any(name in line for line in evidence_lines)]


def _ref_basename(ref: str) -> str:
    """Strip a trailing ' (page N)' suffix from a reference line."""
    return re.sub(r"\s*\(page\s*\d+\)\s*$", "", ref).strip()


def _ensure_reference_lines(
    references: list[str], ensure_references: list[str] | None
) -> list[str]:
    """Guarantee each filename in `ensure_references` appears in `references`.

    `references` are extracted from the LLM prose (and may miss files the model
    cited descriptively, or capture only a fragment of a space-containing
    filename). For each guaranteed file: if its basename is already represented,
    leave the existing (possibly page-annotated) entry; otherwise drop any prose
    fragment that is a trailing piece of the full name (e.g. '11.07.25 AM.png'
    captured from 'Screenshot ... 11.07.25 AM.png') and append the full name."""
    if not ensure_references:
        return references
    merged = list(references)
    existing_lower = {_ref_basename(r).lower() for r in merged}
    for fname in ensure_references:
        fl = fname.lower()
        if fl in existing_lower:
            continue
        if " " in fname:
            merged = [
                r for r in merged
                if not (_ref_basename(r).lower() != fl and fl.endswith(_ref_basename(r).lower()))
            ]
        merged.append(fname)
        existing_lower.add(fl)
    return merged


def _format_response_for_display(
    response_text: str,
    excluded_files: list[str] | None = None,
    ensure_references: list[str] | None = None,
    drop_references: set[str] | None = None,
) -> str:
    """Render the answer followed by a 'References' block that lists each unique
    source (file + page) named in the answer.

    Inline source citations are stripped from the answer prose (see
    `_strip_citations_for_display`) and collected once into the de-duplicated
    References list appended underneath — every value, date, and name in the
    answer is preserved. If the answer names no sources, it is returned
    unchanged. Spoken/TTS output is produced elsewhere (`_spoken_response_text`)
    from the raw answer and is unaffected by this function. `ensure_references`
    forces launch-folder evidence files into the list even when the LLM did not
    cite them in prose; `excluded_files` lists launch-folder files that could not
    be indexed (bad/no extractable data), annotated. Both live inside the
    References block, never in the answer body."""
    excluded_refs = _excluded_reference_lines(excluded_files)
    body = response_text.strip() if response_text else ""
    references = _extract_source_references(body) if body else []
    if drop_references:
        references = [
            ref for ref in references if _ref_basename(ref).lower() not in drop_references
        ]
    references = _ensure_reference_lines(references, ensure_references)
    display_body = _strip_citations_for_display(body) if body else ""
    all_refs = references + excluded_refs
    if not all_refs:
        return display_body
    ref_block = "\n".join(f"  - {ref}" for ref in all_refs)
    if display_body:
        return f"{display_body}\n\nReferences:\n{ref_block}"
    return f"References:\n{ref_block}"
