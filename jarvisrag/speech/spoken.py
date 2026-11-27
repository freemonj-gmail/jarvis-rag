"""The text to speak for an answer: citations removed, long answers summarized.
"""

import re
from typing import Iterable

from jarvisrag.speech.citations import (
    _source_label_words,
    _strip_inline_citations,
    _strip_label_citations,
    _strip_source_filenames_for_speech,
)


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


def _spoken_response_text(response_text: str, source_labels: Iterable[str] = ()) -> str:
    """Return the FULL prose of a response for TTS, dropping only the
    'Relevant evidence found:' block and reference/citation metadata lines.
    Unlike `_summarize_for_speech`, this keeps the entire answer — including
    list/bullet content (with the bullet marker removed) and all sentences —
    so the listener hears the complete response, not just a two-sentence
    preview. ``source_labels`` are the labels of the answer's evidence; a
    citation written with their words is not read aloud."""
    if not response_text:
        return ""
    text = _speech_text_for_response(response_text)
    label_words = _source_label_words(source_labels)

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
        line = _strip_label_citations(line, label_words)
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
