"""Shared math vocabulary so symbols and their spelled-out names route alike."""

from __future__ import annotations

import difflib
import re


# Each operator lists its symbol and every spelled-out name, longest phrase
# first so "take away" is consumed before "take" and "difference between"
# before "difference". Wordings that reverse operand order ("subtract A from
# B") are deliberately absent; the word-arithmetic patterns own those. So are
# overloaded words like "over" and "per", which collide with thresholds and
# with "per cent".
MULTIPLICATION_WORDS: tuple[str, ...] = (
    r"multiplied\s+by",
    r"multiply(?:\s+by)?",
    r"by\s+a\s+factor\s+of",
    r"coefficient\s+of",
    r"product\s+of",
    r"scaled\s+by",
    r"times",
    r"[\u00d7\u2715]",
)
DIVISION_WORDS: tuple[str, ...] = (
    r"divided\s+by",
    r"divide(?:\s+by)?",
    r"quotient\s+of",
    r"ratio\s+of",
    r"[\u00f7]",
)
ADDITION_WORDS: tuple[str, ...] = (
    r"increased\s+by",
    r"added\s+to",
    r"add(?:ed)?",
    r"sum\s+of",
    r"total\s+of",
    r"plus",
)
SUBTRACTION_WORDS: tuple[str, ...] = (
    r"difference\s+(?:between|of)",
    r"tak(?:e|es|en|ing)\s+away",
    r"tak(?:e|es|en|ing)\s+off",
    r"decreased\s+by",
    r"reduced\s+by",
    r"subtract(?:ed|s|ing)?",
    r"deduct(?:ed|s|ing)?",
    r"minus",
    r"less",
    r"[\u2212\u2013]",
)

OPERATOR_SYNONYMS: tuple[tuple[str, str], ...] = (
    *((pattern, "*") for pattern in MULTIPLICATION_WORDS),
    *((pattern, "/") for pattern in DIVISION_WORDS),
    *((pattern, "+") for pattern in ADDITION_WORDS),
    *((pattern, "-") for pattern in SUBTRACTION_WORDS),
)

_OPERATOR_RE = re.compile(
    "|".join(
        f"(?P<g{index}>(?<![a-z]){pattern}(?![a-z]))"
        for index, (pattern, _) in enumerate(OPERATOR_SYNONYMS)
    ),
    re.I,
)
_OPERATOR_BY_GROUP = {
    f"g{index}": symbol for index, (_, symbol) in enumerate(OPERATOR_SYNONYMS)
}

# A rate marker, in symbol and word form, so "27%" and "27 percent" behave alike.
PERCENT_MARKER_RE = re.compile(r"%|\bpercent(?:age)?\b|\bper\s+cent\b", re.I)

# Ways a follow-up names the number the previous turn produced.
REFERENCE_RE = re.compile(
    "|".join(
        (
            r"(?:the\s+)?(?:previous|prior|last|preceding|earlier)\s+"
            r"(?:answer|result|value|number|total|amount|figure|calculation|sum)",
            r"(?:the\s+)?(?:answer|result|value|total|amount|figure)\s+"
            r"(?:above|before|from\s+before|you\s+(?:just\s+)?(?:gave|calculated|said))",
            r"(?:that|this)\s+(?:answer|result|value|number|total|amount|figure)",
            r"\bthat\b",
            r"\bthis\b",
            r"\bit\b",
        )
    ),
    re.I,
)

# Connectors that join one calculation step to the next.
CHAIN_SPLIT_RE = re.compile(r",?\s*(?:and\s+then|then|and)\s+", re.I)

# An operator with nothing on its left, meaning the base is the previous answer.
LEADING_OPERATOR_RE = re.compile(r"^[+\-*/]\s")


def to_operator_symbols(text: str) -> str:
    """Rewrite spelled-out operators as their symbols."""

    def replace(match: re.Match[str]) -> str:
        return f" {_OPERATOR_BY_GROUP[match.lastgroup or '']} "

    return " ".join(_OPERATOR_RE.sub(replace, text).split())


# Plain spellings of the operator words, for catching near-miss typos.
OPERATOR_VOCABULARY: tuple[str, ...] = (
    "multiply", "multiplied", "product", "times", "factor", "coefficient", "scaled",
    "divide", "divided", "quotient", "ratio",
    "add", "added", "plus", "sum", "total", "increased",
    "subtract", "subtracted", "minus", "difference", "deduct", "deducted",
    "decreased", "reduced", "less", "percent", "percentage",
)

_WORD_RE = re.compile(r"[a-z]+", re.I)
# "multiple" scores 0.875 against "multiply"; anything looser invites noise.
_SUGGESTION_CUTOFF = 0.86
_MIN_SUGGESTION_LENGTH = 4
_MAX_SUGGESTIONS = 4


def suggest_operator_corrections(text: str) -> tuple[tuple[str, str, str], ...]:
    """Return (corrected text, original word, suggestion) for a lone near-miss.

    Ranked by similarity; the caller decides which respelling is the real one.
    """
    matches: list[tuple[str, list[str]]] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if len(lowered) < _MIN_SUGGESTION_LENGTH or lowered in OPERATOR_VOCABULARY:
            continue
        close = difflib.get_close_matches(
            lowered, OPERATOR_VOCABULARY, n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
        )
        if close:
            matches.append((word, close))
    # More than one near-miss word is too ambiguous to guess from.
    if len(matches) != 1:
        return ()
    word, suggestions = matches[0]
    pattern = rf"(?<![a-z]){re.escape(word)}(?![a-z])"
    return tuple(
        (re.sub(pattern, suggestion, text, count=1, flags=re.I), word, suggestion)
        for suggestion in suggestions
    )
