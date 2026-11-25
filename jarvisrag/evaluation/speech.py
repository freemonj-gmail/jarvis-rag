"""Score how answers are worded for speech against labelled readings.

Each case is a sentence and the tokens in it whose spoken form depends on
context, with the readings a listener would accept: ``accept`` holds the
natural readings, ``acceptable`` further readings that are unidiomatic but not
wrong or misleading. Grading is deterministic: a token counts as read correctly
when one of its readings appears in the words handed to the speech engine.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable

# Readings measured run to run vary by about three tokens in a hundred, so a
# smaller drop is noise rather than a regression.
RATE_TOLERANCE = 0.03
# The speech step may slow by half, or by a second, before it counts.
LATENCY_RATIO_LIMIT = 1.5
LATENCY_SLACK_MS = 1000.0


@dataclass(frozen=True)
class SpokenToken:
    token: str
    accept: tuple[str, ...]
    acceptable: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeechCase:
    subject: str
    text: str
    tokens: tuple[SpokenToken, ...]


def load_speech_cases(path: str) -> list[SpeechCase]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        SpeechCase(
            subject=str(item["subject"]),
            text=str(item["text"]),
            tokens=tuple(
                SpokenToken(
                    token=str(t["token"]),
                    accept=tuple(t["accept"]),
                    acceptable=tuple(t.get("acceptable") or ()),
                )
                for t in item["tokens"]
            ),
        )
        for item in raw
    ]


def _words(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower().replace("-", " "))
    return " " + " ".join(w for w in text.split() if w != "and") + " "


def _readings(token: SpokenToken, lenient: bool) -> list[str]:
    readings = list(token.accept) + (list(token.acceptable) if lenient else [])
    letters = [ch for ch in token.token if ch.isalpha()]
    # An initialism the voice spells on its own may be left as written.
    if any(_words(r).split() == [ch.lower() for ch in letters] for r in readings):
        readings.append("".join(letters))
    return readings


def token_read_correctly(token: SpokenToken, spoken: str, lenient: bool = False) -> bool:
    said = _words(spoken)
    return any(_words(reading) in said for reading in _readings(token, lenient))


def evaluate_speech(
    cases: Iterable[SpeechCase],
    speak: Callable[[str], tuple[str, float]],
    group: int = 5,
    context_step: bool = True,
) -> dict:
    """Word every case for speech and grade it.

    ``speak(text)`` returns the words for the speech engine and the seconds the
    context step took; ``context_step`` records whether the code has that step,
    since its time can only be compared with code that also has it. Cases are worded ``group`` sentences at a time, one per
    line, the way a paragraph of an answer is.
    """
    cases = list(cases)
    natural = acceptable = total = 0
    latencies: list[float] = []
    subjects: dict[str, list[int]] = {}
    for start in range(0, len(cases), max(1, group)):
        block = cases[start:start + group]
        spoken, seconds = speak("\n".join(case.text for case in block))
        latencies.append(seconds * 1000.0)
        lines = spoken.split("\n")
        if len(lines) != len(block):
            raise ValueError("the worded text must keep one line per sentence")
        for case, line in zip(block, lines):
            tally = subjects.setdefault(case.subject, [0, 0])
            for token in case.tokens:
                total += 1
                tally[1] += 1
                if token_read_correctly(token, line):
                    natural += 1
                    tally[0] += 1
                if token_read_correctly(token, line, lenient=True):
                    acceptable += 1
    return {
        "context_step": context_step,
        "metrics": {
            "tokens": total,
            "natural_rate": natural / total if total else 0.0,
            "acceptable_rate": acceptable / total if total else 0.0,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
            "latency_ms_max": max(latencies) if latencies else 0.0,
        },
        "subjects": {name: {"natural": hit, "tokens": n} for name, (hit, n) in sorted(subjects.items())},
    }


def compare_speech_reports(baseline: dict, current: dict) -> list[str]:
    """Metrics in ``current`` that are worse than ``baseline``, as messages."""
    base, now = baseline["metrics"], current["metrics"]
    problems = []
    for name in ("acceptable_rate", "natural_rate"):
        if now[name] < base[name] - RATE_TOLERANCE:
            problems.append(f"{name} fell {base[name]:.3f} -> {now[name]:.3f}")
    limit = max(base["latency_ms_p50"] * LATENCY_RATIO_LIMIT, base["latency_ms_p50"] + LATENCY_SLACK_MS)
    timed_alike = baseline.get("context_step", True) == current.get("context_step", True)
    if timed_alike and now["latency_ms_p50"] > limit:
        problems.append(f"latency_ms_p50 rose {base['latency_ms_p50']:.0f} -> {now['latency_ms_p50']:.0f}")
    return problems
