"""Check that a voice says the words Jarvis sends it.

Each labelled sentence is worded for speech exactly as Jarvis would, rendered by
the real voice, and transcribed by a local Whisper model. Both the words sent
and the words heard pass through Whisper's English normalizer, so "nineteen
oh-three" and "1903" compare equal, and the word error rate between them
measures only what the voice dropped or garbled -- the voice's own behaviour,
which changes with Python, PyTorch, macOS or the voice package, not with Jarvis.
"""

from __future__ import annotations

import difflib
import re
import statistics

# A sentence with more than this share of its words wrong counts as misspoken.
MISSPOKEN_WER = 0.2
# Voices sample randomly, so a slightly higher error is noise rather than drift.
WER_TOLERANCE = 0.03
MISSPOKEN_TOLERANCE = 2


def _letters_and_digits(words: list[str]) -> list[str]:
    """Join runs of single letters ("i r c" -> "irc") and of digit groups
    ("113 1973" -> "1131973"): Whisper writes either form of each, so neither
    is a difference in what was said."""
    folded: list[str] = []
    in_letter_run = False
    for word in words:
        letter = len(word) == 1 and word.isalpha()
        if folded and ((letter and in_letter_run) or (word.isdigit() and folded[-1].isdigit())):
            folded[-1] += word
        else:
            folded.append(word)
            in_letter_run = letter
            continue
        in_letter_run = in_letter_run and letter
    return folded


_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _roman_value(word: str) -> int | None:
    """The value of a Roman numeral of two or more letters, else None."""
    if len(word) < 2 or any(ch not in _ROMAN_VALUES for ch in word):
        return None
    total = 0
    for ch, nxt in zip(word, word[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt in _ROMAN_VALUES and _ROMAN_VALUES[nxt] > value else value
    return total if total > 0 else None


def _same_notation(words: list[str]) -> list[str]:
    """Write numbers one way: Whisper transcribes a spoken "the eighth" as "VIII"
    and "eighth" as "8th", which are the same words said."""
    out: list[str] = []
    for word in words:
        roman = _roman_value(word)
        if roman is not None:
            word = str(roman)
        word = re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", word)
        if word.isdigit() and out and out[-1] == "the":
            out.pop()
        out.append(word)
    return out


def word_error_rate(sent: str, heard: str) -> float:
    """Word-level edit distance between two normalized texts, per sent word."""
    want = _letters_and_digits(_same_notation(sent.split()))
    got = _letters_and_digits(_same_notation(heard.split()))
    if not want:
        return 0.0 if not got else 1.0
    matcher = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    errors = 0
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op != "equal":
            errors += max(a1 - a0, b1 - b0)
    return errors / len(want)


def voice_report(engine: str, scores: list[tuple[str, float]]) -> dict:
    """Aggregate per-sentence (subject, word error rate) scores."""
    rates = [rate for _, rate in scores]
    return {
        "engine": engine,
        "metrics": {
            "sentences": len(rates),
            "mean_wer": statistics.fmean(rates) if rates else 0.0,
            "misspoken": sum(rate > MISSPOKEN_WER for rate in rates),
        },
        "per_sentence": [{"subject": subject, "wer": round(rate, 3)} for subject, rate in scores],
    }


def compare_voice_reports(baseline: dict, current: dict) -> list[str]:
    """Ways ``current`` is worse than ``baseline``, as messages."""
    if baseline.get("engine") != current.get("engine"):
        return [f"engine differs: {baseline.get('engine')} -> {current.get('engine')}"]
    base, now = baseline["metrics"], current["metrics"]
    if base["sentences"] != now["sentences"]:
        return [f"sentence sets differ ({base['sentences']} vs {now['sentences']}); record a new reference"]
    problems = []
    if now["mean_wer"] > base["mean_wer"] + WER_TOLERANCE:
        problems.append(f"mean_wer rose {base['mean_wer']:.3f} -> {now['mean_wer']:.3f}")
    if now["misspoken"] > base["misspoken"] + MISSPOKEN_TOLERANCE:
        problems.append(f"misspoken rose {base['misspoken']} -> {now['misspoken']}")
    return problems
