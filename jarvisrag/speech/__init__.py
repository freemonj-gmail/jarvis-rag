"""Speech synthesis (Kokoro TTS) and spoken-text normalization for Jarvis.

Owns everything on the spoken-output path:
  * ``TTSManager`` — the Kokoro pipeline wrapper that runs synthesis on a
    single background worker thread and plays audio via the platform player.
  * The spoken-text transforms that prepare an answer for the speech engine
    (citation stripping, abbreviation/numeral expansion, summary/full prose).
  * The shared citation regexes (``_CITATION_FILE_EXT``, ``_INLINE_CITATION_RE``,
    ``_SOURCE_REFERENCE_RE``) — also consumed by the printed-display formatter.
  * ``_SilentTTS`` — the no-op fallback used when a runtime has no TTS attached.

Depends on the standard library plus ``jarvis_config`` (logger) and
``jarvis_io`` (native-output suppression context managers).
"""
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext

import numpy as np

from jarvisrag.config import logger
from jarvisrag.iotools import suppress_external_output, suppress_external_stderr


# Inline parenthetical source citations the model emits in prose, e.g.
# "(Resume_2026.docx)", "(report.pdf, page 3)", or "(a.docx, b.docx, c.docx)".
# Matched groups contain at least one filename-with-extension or a page
# reference. Used to clean spoken text only; printed/displayed output keeps
# citations untouched. The extension set mirrors every file type Jarvis can
# ingest (documents + images); add new extensions here if the ingest list in
# Jarvis.py grows.
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

# Magnitude scales that may trail a dollar amount, as either a spelled-out word
# ("$4 million") or a single-letter suffix ("$96k", "$4m"). Maps every accepted
# token to the spoken word so "$4 million" reads "4 million dollars" and "$96k"
# reads "96 thousand dollars" instead of dropping the scale.
_SCALE_WORDS = {
    "k": "thousand", "m": "million", "b": "billion", "t": "trillion",
    "thousand": "thousand", "million": "million",
    "billion": "billion", "trillion": "trillion",
}
# Optional scale after a "$N" amount: a letter suffix touching the digits
# (guarded so "million" is not read as a bare "m") or a spelled-out word after
# whitespace. Kept out of the number capture so the existing cents path is
# unaffected when no scale is present.
_DOLLAR_SCALE_SUFFIX = (
    r"(?:\s*(?P<abbr>[kKmMbBtT])(?![A-Za-z])"
    r"|\s+(?P<word>[Tt]housand|[Mm]illion|[Bb]illion|[Tt]rillion))?"
)

# Sub-second time-unit abbreviations -> spoken words. Only expanded when the
# unit is attached to a number (e.g. "200 ms" / "5us" / "10 ns"), so the plain
# English word "us" ("tell us") and unrelated tokens are left untouched. The
# micro sign (µ) and the letter "u" both stand in for "micro". Case-sensitive
# match ("ms" not "MS") so "MS" (Microsoft/Missouri/degree) is unaffected.
_TIME_UNIT_WORDS = {
    "ms": "millisecond",
    "us": "microsecond",
    "\u00b5s": "microsecond",  # µs (micro sign)
    "ns": "nanosecond",
    "ps": "picosecond",
}
_TIME_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*(ms|us|\u00b5s|ns|ps)\b"
)


def _expand_time_units(text: str) -> str:
    """Read attached sub-second durations in full: "200 ms" -> "200 milliseconds",
    "1 ms" -> "1 millisecond". Pluralizes based on the numeric value."""
    def _repl(match: "re.Match[str]") -> str:
        number, unit = match.group(1), match.group(2)
        word = _TIME_UNIT_WORDS[unit]
        try:
            plural = float(number) != 1.0
        except ValueError:
            plural = True
        return f"{number} {word}{'s' if plural else ''}"

    return _TIME_UNIT_RE.sub(_repl, text)


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

    # 7) Sub-second time units attached to a number: "200 ms" -> "200
    #    milliseconds", "5us" -> "5 microseconds", "10 ns" -> "10 nanoseconds".
    s = _expand_time_units(s)
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
        cleaned = re.sub(r"\$\s*(\d[\d,]*)(?:\.(\d{1,2}))?" + _DOLLAR_SCALE_SUFFIX, self._spoken_dollar_amount, cleaned)
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
        whole = match.group(1).replace(",", "")
        frac = match.group(2) or ""
        scale_word = _SCALE_WORDS.get(
            (match.group("abbr") or match.group("word") or "").lower(), ""
        )
        if scale_word:
            # A scaled amount: the decimal is a fractional multiplier, not cents
            # ("$4.5 million" -> "4 point 5 million dollars").
            number = f"{whole} point {frac}" if frac else whole
            return f"{number} {scale_word} dollars"
        cents_raw = frac.ljust(2, "0")[:2]
        cents = int(cents_raw) if cents_raw else 0
        if cents:
            return f"{whole} dollars and {cents} cents"
        return f"{whole} dollars"

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


class _SilentTTS:
    """Fallback no-op TTS used when a runtime has no speech manager attached or
    when speech is disabled via JARVIS_DISABLE_TTS=1 (e.g. smoke tests)."""

    def speak(self, _text: str) -> None:  # pragma: no cover - trivial
        pass

    def say_goodbye(self) -> None:  # pragma: no cover - trivial
        pass

    def cleanup(self) -> None:  # pragma: no cover - trivial
        pass
