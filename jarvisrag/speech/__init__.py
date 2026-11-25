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
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from typing import Iterable

import numpy as np

from jarvisrag.config import debug_print, logger
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


def _cardinal_words(digits: str) -> str:
    """Spell an integer string as cardinal words so a speech engine never applies
    a year-style reading (e.g. 4534 -> 'four thousand ...' not 'forty-five
    thirty-four'). Falls back to the digits if num2words is unavailable."""
    try:
        from num2words import num2words

        return num2words(int(digits)).replace(",", "")
    except Exception:
        return digits

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


# US-style slash dates with a four-digit year: "1/15/2026" (month/day/year) or
# "1/2026" (month/year). Year-less forms like "12/25" stay as written because
# they cannot be told apart from fractions, and a slash inside a larger token
# (paths, URLs, "1/2026/3") is left alone.
_SLASH_DATE_RE = re.compile(r"(?<![\w/.\-])(\d{1,2})/(?:(\d{1,2})/)?(\d{4})(?![\w/\-])")
# A month/year pair has no day to validate it, so "1/1000" (a ratio) must not be
# read as a month and year; only calendar years of modern documents qualify.
_MONTH_YEAR_RANGE = range(1900, 2100)


def _expand_slash_dates(text: str) -> str:
    """Speak slash dates as a person says them: "1/15/2026" -> "January
    fifteenth, twenty twenty-six" and "1/2026" -> "January twenty twenty-six".
    Impossible dates (month 13, February 30) are left as written."""
    try:
        from num2words import num2words
    except Exception:
        return text
    import calendar
    import datetime

    def _repl(match: "re.Match[str]") -> str:
        month, day, year = int(match.group(1)), match.group(2), int(match.group(3))
        if not 1 <= month <= 12:
            return match.group(0)
        if day is None:
            if year not in _MONTH_YEAR_RANGE:
                return match.group(0)
            return f"{calendar.month_name[month]} {num2words(year, to='year')}"
        try:
            datetime.date(year, month, int(day))
        except ValueError:
            return match.group(0)
        return (
            f"{calendar.month_name[month]} {num2words(int(day), to='ordinal')}, "
            f"{num2words(year, to='year')}"
        )

    return _SLASH_DATE_RE.sub(_repl, text)


# Street abbreviations that collide with a title ("Dr." Drive/Doctor, "St."
# Street/Saint). A house number in front is what distinguishes the address.
_STREET_WORDS = {"Dr": "Drive", "St": "Street"}
_STREET_ADDRESS_RE = re.compile(r"(\d+\s+(?:[A-Z][\w'\u2019-]*\s+)*)(Dr|St)\.")

# Spoken names for the two-letter state codes, expanded only inside an address
# so ordinary words that share a code ("OR", "IN", "ME") are left alone.
_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico",
}
# "City, ST" followed by a ZIP, punctuation or the end of the text. Requiring
# that trailing context keeps prose like "sales, OR marketing" untouched.
_STATE_IN_ADDRESS_RE = re.compile(
    r",\s*([A-Z]{2})\b(?=\s*(?:\d{5}(?:-\d{4})?|[.,;:!?)]|$))"
)

# A ZIP is read digit-by-digit when it follows a state, whether written as the
# 2-letter code ("WA 98052") or the full name that _STATE_IN_ADDRESS_RE expands
# the code into before step 5 runs ("Maryland 20878"). One pattern, reused by
# step 5's _state_zip_repl alongside _US_STATE_ABBR / _spell_digits.
_STATE_NAME_SET = frozenset(_STATE_NAMES.values())
_STATE_ZIP_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(n) for n in _STATE_NAME_SET), key=len, reverse=True))
    + r"|[A-Z]{2})\s+(\d{5})(?:-(\d{4}))?\b"
)


def _normalize_for_speech(text: str, roman_numerals: bool = True) -> str:
    """Expand context-dependent abbreviations and numerals for natural speech.

    Handles the cases TTS gets wrong on its own:
      * Titles before a name:  "Dr./Mr./Mrs./Ms./Prof./St. <Name>" -> spoken word
      * Address abbreviations: remaining "Dr." -> "Drive", "St." -> "Street"
      * Generational suffixes: "Jr."/"J-R" -> "Junior", "Sr."/"S-R" -> "Senior"
      * Name numerals:         "<Name> III"/"I-I-I" -> "<Name> the Third"
      * Slash dates:           "1/15/2026" -> "January fifteenth, twenty twenty-six"

    Rules are case-sensitive where it matters (e.g. "Ms" -> "Miss" but "MS",
    which may mean Microsoft or Missouri, is left untouched). Only the spoken
    string is affected; printed output keeps the original text.
    """
    if not text:
        return text
    s = text

    # 0) A street number ahead of the abbreviation settles it as an address, so
    #    resolve those before the title rules below can claim them: in
    #    "11955 Freedom Dr. Reston" the following word is a city, not a name.
    s = _STREET_ADDRESS_RE.sub(
        lambda m: m.group(1) + _STREET_WORDS[m.group(2)], s
    )

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

    # 2b) State codes in a "City, ST" address read as the full state name.
    s = _STATE_IN_ADDRESS_RE.sub(
        lambda m: ", " + _STATE_NAMES[m.group(1)] if m.group(1) in _STATE_NAMES else m.group(0),
        s,
    )

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

    #    When the context step runs, it reads Roman numerals by meaning instead
    #    ("World War Two", "Rocky Four", "Henry the Eighth"); this rule is the
    #    fallback for when the model is unavailable.
    if roman_numerals:
        s = re.sub(
            r"\b([A-Za-z]+)\s+"
            r"((?:I-I-I|I-I|I\.I\.I|VIII|VII|VI|IV|III|II|V))\b",
            _roman_repl,
            s,
        )

    # 5) ZIP codes in addresses -> read digit by digit. Only fire on a strong
    #    address signal so ordinary 5-digit numbers (years, counts, IDs) are
    #    left alone: (a) ZIP+4 form "12345-6789", or (b) a 5-digit number
    #    immediately following a US state, as the 2-letter code ("WA 98052") or
    #    the full name step 2b expanded it into ("Maryland 20878").
    def _zip4_repl(match: "re.Match[str]") -> str:
        return f"{_spell_digits(match.group(1))} {_spell_digits(match.group(2))}"

    s = re.sub(r"\b(\d{5})-(\d{4})\b", _zip4_repl, s)

    def _state_zip_repl(match: "re.Match[str]") -> str:
        state, zip5, plus4 = match.group(1), match.group(2), match.group(3)
        if state not in _US_STATE_ABBR and state not in _STATE_NAME_SET:
            return match.group(0)
        spoken = _spell_digits(zip5)
        if plus4:
            spoken = f"{spoken} {_spell_digits(plus4)}"
        return f"{state} {spoken}"

    s = _STATE_ZIP_RE.sub(_state_zip_repl, s)

    # 6) Rate slash -> "per" before a known unit, e.g. "$7,500/month" ->
    #    "$7,500 per month". Dates ("12/25"), "and/or", "TCP/IP" etc. are
    #    untouched because their right-hand side is not a rate unit.
    s = _PER_SLASH_RE.sub(" per ", s)

    # 7) Sub-second time units attached to a number: "200 ms" -> "200
    #    milliseconds", "5us" -> "5 microseconds", "10 ns" -> "10 nanoseconds".
    s = _expand_time_units(s)

    # 8) Slash dates -> spoken dates: "1/1/2026" -> "January first, twenty
    #    twenty-six", "1/2026" -> "January twenty twenty-six". Runs after the
    #    rate-slash rule, whose units never follow a date's slash.
    s = _expand_slash_dates(s)
    return s


# -- Context step: how tokens whose reading depends on meaning are spoken -------
#
# Fixed rules cannot tell "Ct." in an address (Court) from "ct." on a diamond
# (carat), a year from a count, or a postal code from a quantity. The model that
# wrote the answer can, so it is asked how each such token is said in its
# sentence. Tokens are found by shape alone (digits, symbols, short abbreviations
# with a period, runs of capitals, a short unit after a number), and the model may
# change only those tokens. A reading is used only if it is plain words that keep
# every number's exact value; anything else stays for the fixed rules.

_SPEECH_TOKEN_RE = re.compile(r"\S+")
_LEADING_WRAPPERS = "\"'\u201c\u2018[{("
_TRAILING_WRAPPERS = "\"'\u201d\u2019]},;:!?"
_DIGIT_NAMES = ("zero one two three four five six seven eight nine".split(),
                "oh one two three four five six seven eight nine".split())
_SPOKEN_FORM_PROMPT = (
    "You prepare text to be read aloud by a speech engine that can only pronounce plain words.\n"
    "Tokens to resolve are marked [[n|token]]. For each one, write exactly how a careful native English "
    "speaker would say it aloud IN THIS CONTEXT, as plain words only (no digits, symbols or abbreviations), "
    "keeping the exact value of every number. Decide from the meaning of the surrounding text:\n"
    "- an abbreviation, symbol or unit: the word it stands for here;\n"
    "- capital letters: if people pronounce them as a word, write that word in lowercase; if people say the "
    "letters, write each letter as a capital letter, separated by spaces (\"F B I\"), never spelled out as "
    "letter names;\n"
    "- an abbreviation written with periods between its letters: its capital letters, separated by spaces, "
    "unless it stands for words people say in full;\n"
    "- a year: the way years are spoken;\n"
    "- a number that counts or measures something: an ordinary quantity, even when it looks like a year;\n"
    "- an identifier such as a postal code, phone, account or serial number: digit by digit;\n"
    "- a form, section, flight, room or model number: the way people say such codes, usually in pairs of digits;\n"
    "- the day in a date, a rank or a position: an ordinal;\n"
    "- a Roman numeral: after the name of an individual person (a monarch, pope or other named person), \"the\" "
    "and an ordinal (\"Henry V\" -> \"the fifth\"); after anything else -- an event, a vehicle, a product, a "
    "title, a series or a category -- the plain cardinal number (\"Episode IV\" -> \"four\");\n"
    "- a score, range, ratio or time: the way people say it.\n"
    "Reply with JSON only. Use each marker's contents exactly as written as the key and its spoken words as "
    "the value, e.g. {{\"1|token\": \"...\", \"2|token\": \"...\"}}.\n\n"
    "Text:\n{text}\n\nJSON:"
)


def _speech_tokens(text: str) -> list[tuple[int, int, str]]:
    """Tokens whose spoken form depends on context, as (start, end, token)."""
    found: list[tuple[int, int, str]] = []
    after_number = False
    previous = ""
    for match in _SPEECH_TOKEN_RE.finditer(text):
        token, start, end = match.group(0), match.start(), match.end()
        # A unit follows its number within a sentence ("8849 m", "6 hrs."),
        # never across a sentence end or a line break.
        if after_number and (previous.endswith((".", "!", "?")) or "\n" in text[match.start() - 2:match.start()]):
            after_number = False
        raw = token
        if token.startswith("(") and token.endswith(")"):
            token, start, end = token[1:-1], start + 1, end - 1
        while token and token[0] in _LEADING_WRAPPERS and not (token[0] == "(" and ")" in token):
            token, start = token[1:], start + 1
        while token and (
            token[-1] in _TRAILING_WRAPPERS or (token[-1] == ")" and "(" not in token)
        ):
            token, end = token[:-1], end - 1
        # A full stop after a number or a run of capitals ends the sentence; it is
        # not part of the token ("in 1911." / "Henry VIII."), unlike "Ph.D." or "Ct.".
        if token.endswith(".") and "." not in token[:-1] and (
            any(ch.isdigit() for ch in token) or token[:-1].isupper()
        ):
            token, end = token[:-1], end - 1
        if not token:
            after_number = False
            continue
        letters = re.sub(r"[^A-Za-z]", "", token)
        # Money and percentages already have one reading; the fixed rules say them.
        settled = token.startswith("$") or token.rstrip(".").endswith("%")
        has_digit = any(ch.isdigit() for ch in token)
        if (
            (has_digit and not settled)
            or any(ch in token for ch in "\u00a7\u00b0#&")
            or (token.endswith(".") and 0 < len(letters) <= 5 and re.fullmatch(r"[A-Za-z.]+", token))
            or sum(ch.isupper() for ch in token) >= 2
            or (after_number and re.fullmatch(r"[A-Za-z]{1,4}\.?", token))
            # A one-letter Roman numeral after a capitalized word ("World War I", "Henry V").
            or (token in ("I", "V", "X") and previous[:1].isupper() and previous.isalpha())
        ):
            found.append((start, end, token))
        after_number = has_digit and not settled
        previous = raw
    return found


def _number_readings(digits: str) -> set[str]:
    """Every standard spoken form of a run of digits, as plain lowercase words."""
    from num2words import num2words

    value = int(digits)
    forms = {num2words(value), num2words(value, to="ordinal")}
    if len(digits) == 4 or value >= 1000:
        forms.add(num2words(value, to="year"))
    for names in _DIGIT_NAMES:
        forms.add(" ".join(names[int(d)] for d in digits))
    if len(digits) == 4 and value % 100 and value // 100 % 10:  # "nineteen hundred three", "fifteen hundred"
        forms.add(f"{num2words(value // 100)} hundred {num2words(value % 100)}")
    if len(digits) == 4 and not value % 100:
        forms.add(f"{num2words(value // 100)} hundred")
    if len(digits) == 3:  # codes said "four oh one", "three seventy"
        head, tail = num2words(int(digits[0])), digits[1:]
        if tail == "00":
            forms.add(f"{head} hundred")
        elif tail[0] == "0":
            forms.add(f"{head} oh {num2words(int(tail[1]))}")
        else:
            forms.add(f"{head} {num2words(int(tail))}")
    if value == 2:
        forms |= {"half", "halves"}
    if value == 4:
        forms |= {"quarter", "quarters"}
    for form in list(forms):
        if form.startswith(("one hundred", "one thousand")):
            forms |= {form[4:], "a " + form[4:]}
    readings = set()
    for form in forms:
        words = " ".join(w for w in form.lower().replace("-", " ").replace(",", "").split() if w != "and")
        readings |= {words, words + "s", words[:-1] + "ies" if words.endswith("y") else words}
    return readings


def _reading_keeps_numbers(token: str, words: str) -> bool:
    """True when every number in ``token`` is said in ``words`` with its own value."""
    said = " " + " ".join(
        w for w in re.sub(r"[^a-z ]", " ", words.lower().replace("-", " ")).split() if w != "and"
    ) + " "
    # Only a thousands separator joins digits: "29,2003" is a day and a year.
    plain = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", token)
    return all(
        any(f" {reading} " in said for reading in _number_readings(run))
        for run in re.findall(r"\d+", plain)
    )


def _speech_llm(llm_client, num_predict: int):
    """A copy of the answer model with its own output budget. The context step
    runs on the speech thread, so it must not change the shared client's
    settings while an answer is being generated; the copy keeps the same model
    and context size, so Ollama does not load anything new."""
    try:
        options = dict(getattr(llm_client, "additional_kwargs", None) or {})
        options["num_predict"] = num_predict
        return llm_client.model_copy(update={"additional_kwargs": options})
    except Exception:
        return None


def _resolve_spoken_tokens(text: str, llm_client) -> str:
    """Word ``text`` for speech, saying each context-dependent token as its
    sentence means it. Falls back to the fixed rules alone when the model is
    unavailable or replies off-format."""
    if not text:
        return text
    if llm_client is None:
        return _normalize_for_speech(text)
    worded = _normalize_for_speech(text, roman_numerals=False)
    tokens = _speech_tokens(worded)
    if not tokens:
        return worded
    marked, last = [], 0
    for number, (start, end, token) in enumerate(tokens, 1):
        marked += [worded[last:start], f"[[{number}|{token}]]"]
        last = end
    marked.append(worded[last:])
    llm = _speech_llm(llm_client, 64 + 24 * len(tokens))
    if llm is None:
        return _normalize_for_speech(text)
    try:
        from llama_index.core.llms import ChatMessage, MessageRole

        reply = llm.chat(
            [ChatMessage(role=MessageRole.USER, content=_SPOKEN_FORM_PROMPT.format(text="".join(marked)))]
        )
        content = reply.message.content or ""
        found = re.search(r"\{.*\}", content, re.S)
        try:
            readings = json.loads(found.group(0)) if found else {}
        except ValueError:
            # A reply with a stray syntax error still names its readings; keep
            # every well-formed pair instead of losing the whole paragraph.
            readings = dict(re.findall(r'"(\d+\|[^"\n]*)"\s*:\s*"([^"\n]*)"', content))
    except Exception as error:
        debug_print(f"[debug] Speech context step skipped ({type(error).__name__}: {error}); using the fixed rules.")
        return _normalize_for_speech(text)
    if not isinstance(readings, dict) or not readings:
        debug_print("[debug] Speech context step gave no readings; using the fixed rules.")
        return _normalize_for_speech(text)
    readings = {str(key).strip().rstrip("."): value for key, value in readings.items()}
    pieces, last = [], 0
    for number, (start, end, token) in enumerate(tokens, 1):
        pieces.append(worded[last:start])
        # Keyed by number AND token, so a miscounted reply cannot land on another token.
        said = readings.get(f"{number}|{token}".rstrip("."))
        usable = (
            isinstance(said, str)
            and said.strip()
            and re.fullmatch(r"[A-Za-z][A-Za-z' .,-]*", said.strip())
            and _reading_keeps_numbers(token, said)
        )
        if usable:
            said = " ".join(said.split())
            # Keep the full stop when the abbreviation ended the line; mid-line, a
            # capital after it is usually a name ("v. Jones"), not a new sentence.
            rest = worded[end:]
            if token.endswith(".") and not said.endswith(".") and (not rest.strip() or re.match(r"[ \t]*\n", rest)):
                said += "."
            pieces.append(said)
        else:
            if isinstance(said, str) and said.strip():
                debug_print(f"[debug] Speech reading rejected for {token!r}: {said!r}; using the fixed rules.")
            pieces.append(token)
        last = end
    pieces.append(worded[last:])
    return "".join(pieces)


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


def _strip_label_citations(text: str, label_words: frozenset[str]) -> str:
    """Remove parenthetical citations made of the answer's own source labels.

    The model cites sources in its own shorthand, e.g. "(irc)" or "(irc,
    Result 3)" for a source labelled "taxmcp.io (IRC/Treasury Regs/IRS)", which
    has no file extension or page for ``_strip_inline_citations`` to key on. A
    parenthetical is dropped when one of its comma- or semicolon-separated parts
    consists only of words from those labels and contains a word, not just a
    number; parentheses that say something else are kept.
    """
    if not text or not label_words:
        return text

    def is_citation(match: "re.Match[str]") -> bool:
        for part in re.split(r"[,;]", match.group(1)):
            words = re.findall(r"[a-z0-9]+", part.lower())
            if words and any(w.isalpha() for w in words) and all(w in label_words for w in words):
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


def _resolve_audio_player():
    """Return the platform's command-line audio player, or None if absent."""
    if sys.platform == "darwin":
        return shutil.which("afplay")
    if sys.platform.startswith("linux"):
        return shutil.which("ffplay") or shutil.which("mpg123") or shutil.which("aplay")
    if sys.platform == "win32":
        return "powershell"
    return None


def _audio_player_command(player, path):
    """Build the argv that plays `path` to completion with `player`."""
    if sys.platform == "win32":
        return [player, "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync();"]
    return [player, path]


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

            self.audio_player = _resolve_audio_player()

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
        cleaned = _normalize_spoken_semantics(text)
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
        # Spell the amount out so the engine never reads a 4-digit figure as a year.
        dollars = _cardinal_words(whole)
        if cents:
            return f"{dollars} dollars and {_cardinal_words(str(cents))} cents"
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
            # Deferred wording runs here, on the speech thread, not the REPL's.
            if callable(text):
                text = text()
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
                        subprocess.run(
                            _audio_player_command(self.audio_player, tmp_file.name),
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


def _normalize_spoken_semantics(text: str) -> str:
    """Rewrite text into the words a speech engine should actually say.

    Covers what a listener would otherwise hear wrong regardless of engine:
    context-dependent abbreviations (Dr., Jr., III), URLs, money, percentages
    and decimals. Purely orthographic scrubbing -- stripping apostrophes and
    punctuation -- is deliberately excluded, because that is a Kokoro/espeak
    front-end requirement and degrades engines that use punctuation for phrasing.
    """
    cleaned = str(text or "")
    # Expand abbreviations before any punctuation/hyphen stripping downstream, so
    # the rules can still see periods and hyphens ("Dr.", "J-R", "I-I-I").
    cleaned = _normalize_for_speech(cleaned)
    cleaned = re.sub(r"https?://\S+", " link omitted ", cleaned)
    # Recognize money-shaped numbers in financial sentences (no `$` needed).
    cleaned = TTSManager._add_dollar_sign_in_financial_context(cleaned)
    cleaned = re.sub(
        r"\$\s*(\d[\d,]*)(?:\.(\d{1,2}))?" + _DOLLAR_SCALE_SUFFIX,
        TTSManager._spoken_dollar_amount,
        cleaned,
    )
    cleaned = re.sub(
        r"\b(\d[\d,]*)\.(\d+)\s*%",
        lambda m: f"{m.group(1).replace(',', '')} point {m.group(2)} percent",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(\d[\d,]*)\s*%",
        lambda m: f"{m.group(1).replace(',', '')} percent",
        cleaned,
    )
    # Drop only thousands separators: in "Jan 29,2003" the comma separates a
    # day from a year, and removing it would read one large number.
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", ", ", cleaned)
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", " point ", cleaned)
    return cleaned


class _SilentTTS:
    """Fallback no-op TTS used when a runtime has no speech manager attached or
    when speech is disabled via JARVIS_ENABLE_TTS=0 (e.g. smoke tests)."""

    def speak(self, _text: str) -> None:  # pragma: no cover - trivial
        pass

    def say_goodbye(self) -> None:  # pragma: no cover - trivial
        pass

    def cleanup(self) -> None:  # pragma: no cover - trivial
        pass
