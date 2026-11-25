"""Measure how Jarvis words answers for speech against labelled readings.

Runs the text side of the Chatterbox path -- the context step that decides how
abbreviations, codes and numbers are read, then the fixed wording rules -- on
labelled sentences, and grades the words handed to the voice. No audio is made.

Regression gate: pass ``--baseline <earlier report>``. A metric that looks worse
is re-measured once and counts only if it reproduces; the run then exits 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("JARVIS_ENABLE_TTS", "0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jarvisrag import speech
from jarvisrag.evaluation.speech import compare_speech_reports, evaluate_speech, load_speech_cases
from jarvisrag.speech.chatterbox import _expand_numbers_to_words


def _speaker():
    resolve = getattr(speech, "_resolve_spoken_tokens", None)
    llm = None
    if resolve is not None:
        from jarvisrag.answer.bootstrap import _build_llm

        llm = _build_llm()

    def speak(text: str) -> tuple[str, float]:
        started = time.monotonic()
        worded = resolve(text, llm) if resolve is not None else text
        seconds = time.monotonic() - started
        # Word each line on its own so a rule that joins lines cannot let one
        # sentence be graded against its neighbour's words.
        lines = [_expand_numbers_to_words(speech._normalize_spoken_semantics(line)) for line in worded.split("\n")]
        return "\n".join(lines), seconds

    speak.context_step = resolve is not None
    return speak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", help="labelled speech cases (JSON)")
    parser.add_argument("--output", required=True, help="where to write the report")
    parser.add_argument("--baseline", help="earlier report to compare with")
    parser.add_argument("--group", type=int, default=5, help="sentences worded together, like one paragraph")
    args = parser.parse_args()

    cases = load_speech_cases(args.cases)
    speak = _speaker()
    report = evaluate_speech(cases, speak, group=args.group, context_step=speak.context_step)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    m = report["metrics"]
    print(
        f"Speech report written: natural {m['natural_rate']:.3f}, acceptable {m['acceptable_rate']:.3f}, "
        f"p50 {m['latency_ms_p50']:.0f} ms over {m['tokens']} tokens."
    )
    if not args.baseline:
        return 0
    with open(args.baseline, encoding="utf-8") as handle:
        baseline = json.load(handle)
    problems = compare_speech_reports(baseline, report)
    if problems:
        print(f"{len(problems)} metric(s) worse than the baseline; confirming with a second run...")
        confirm = evaluate_speech(cases, speak, group=args.group, context_step=speak.context_step)
        with open(args.output.replace(".json", ".confirm.json"), "w", encoding="utf-8") as handle:
            json.dump(confirm, handle, indent=1)
        repeated = {p.split()[0] for p in compare_speech_reports(baseline, confirm)}
        problems = [p for p in problems if p.split()[0] in repeated]
    for problem in problems:
        print(f"REGRESSION: {problem}")
    if problems:
        print(f"Regression gate FAILED: {len(problems)} metric(s) worse than the baseline.")
        return 1
    print("Regression gate passed: no metric worse than the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
