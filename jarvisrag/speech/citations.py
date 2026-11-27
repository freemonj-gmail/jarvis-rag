"""Citation patterns and their removal from text that will be spoken.

The citation regexes are shared with the printed-display formatter.
"""

import re
from typing import Iterable


# Inline parenthetical source citations the model emits in prose, e.g.
# "(Resume_2026.docx)", "(report.pdf, page 3)", or "(a.docx, b.docx, c.docx)".
# Matched groups contain at least one filename-with-extension or a page
# reference. Used to clean spoken text only; printed/displayed output keeps
# citations untouched. The extension set mirrors every file type Jarvis can
# ingest (documents + images); add new extensions here if the ingest list in
# Jarvis.py grows.
_CITATION_FILE_EXT = (
    r"pdf|docx?|xlsx?|pptx?|csv|txt|rtf"          # documents
    r"|mht(?:ml)?|html?|emlx?"                    # web archives + mail messages
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


def _strip_label_citations(text: str, label_words: frozenset[str]) -> str:
    """Remove parenthetical citations made of the answer's own source labels.

    The model cites sources in its own shorthand, e.g. "(irc)" or "(irc,
    Result 3)" for a source labelled "taxmcp.io (IRC/Treasury Regs/IRS)", which
    has no file extension or page for ``_strip_inline_citations`` to key on. A
    parenthetical is dropped when one of its comma- or semicolon-separated parts
    consists only of words from those labels and contains a word, not just a
    number; parentheses that say something else are kept. A label token that
    mixes letters and digits counts as a word, so a message or file id such as
    "7c41d0e9b2fa" cited without its extension is dropped too, while "(2026)"
    or "(17)" stays.
    """
    if not text or not label_words:
        return text

    def is_citation(match: "re.Match[str]") -> bool:
        for part in re.split(r"[,;]", match.group(1)):
            words = re.findall(r"[a-z0-9]+", part.lower())
            if words and any(not w.isdigit() for w in words) and all(w in label_words for w in words):
                return True
        return False

    # A citation follows a space ("... 17 (irc)"), unlike "152(c)", and may hold
    # one level of nested parentheses ("(irc, Section 24(c)(1))").
    stripped = re.sub(
        r"\s+\(((?:[^()\n]|\([^()\n]*\))*)\)", lambda m: "" if is_citation(m) else m.group(0), text
    )
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)
    return re.sub(r"[ \t]{2,}", " ", stripped)


def _source_label_words(source_labels: Iterable[str]) -> frozenset[str]:
    """Lowercase words of the source labels, the vocabulary of a citation."""
    return frozenset(
        word for label in source_labels or () for word in re.findall(r"[a-z0-9]+", str(label).lower())
    )


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
