"""Deterministic answer-quality metrics for Jarvis experiments.

Retrieval metrics stop at "did the right passage reach the prompt". This module
measures the answer itself: required facts present, forbidden claims absent, the
response not a refusal, and expected sources cited. When the answer callable
reports more about a turn (its evidence, spoken text, whether it continued the
previous turn) the report also covers hallucinated figures, answer length, the
clarification rate, and follow-up handling, and ``compare_answer_reports`` turns
two reports into a regression verdict.

Grading is deterministic normalized substring matching - no model judges an
answer, so results are reproducible and cheap to interpret. The evaluator
receives answers from an injected callable, so its own tests need no models,
index, or corpus. Reports contain aggregate metrics by default so private
ground truth can stay local and untracked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Callable, Iterable

from jarvisrag.evaluation.retrieval import _normalized, _percentile


@dataclass(frozen=True)
class AnswerCase:
    """One question plus the assertions a correct answer must satisfy."""

    case_id: str
    query: str
    # Each group is a set of acceptable surface forms; the group passes when any
    # form appears, so harmless paraphrase does not fail an otherwise correct answer.
    must_include: tuple[tuple[str, ...], ...] = ()
    must_not_include: tuple[str, ...] = ()
    must_cite: tuple[str, ...] = ()
    must_not_refuse: bool = True
    # Conversation turns only: whether this turn should be read as continuing the
    # previous one. None leaves follow-up handling ungraded.
    expect_follow_up: bool | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AnswerCase":
        case_id = value.get("id")
        query = value.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("answer case requires a non-empty id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"answer case {case_id} requires a non-empty query")

        raw_groups = value.get("must_include") or []
        if not isinstance(raw_groups, list):
            raise ValueError(f"answer case {case_id} must_include must be a list")
        groups: list[tuple[str, ...]] = []
        for group in raw_groups:
            # A bare string is shorthand for a single-alternative group.
            forms = [group] if isinstance(group, str) else group
            if not isinstance(forms, list) or not forms:
                raise ValueError(f"answer case {case_id} must_include entries must be non-empty")
            if not all(isinstance(form, str) and form.strip() for form in forms):
                raise ValueError(f"answer case {case_id} must_include forms must be non-empty strings")
            groups.append(tuple(forms))

        forbidden = value.get("must_not_include") or []
        citations = value.get("must_cite") or []
        for name, items in (("must_not_include", forbidden), ("must_cite", citations)):
            if not isinstance(items, list) or not all(
                isinstance(item, str) and item.strip() for item in items
            ):
                raise ValueError(f"answer case {case_id} {name} must be a list of non-empty strings")

        refuse = value.get("must_not_refuse", True)
        if not isinstance(refuse, bool):
            raise ValueError(f"answer case {case_id} must_not_refuse must be a boolean")
        follow_up = value.get("expect_follow_up")
        if follow_up is not None and not isinstance(follow_up, bool):
            raise ValueError(f"answer case {case_id} expect_follow_up must be a boolean")

        return cls(
            case_id=case_id,
            query=query,
            must_include=tuple(groups),
            must_not_include=tuple(forbidden),
            must_cite=tuple(citations),
            must_not_refuse=refuse,
            expect_follow_up=follow_up,
        )


@dataclass(frozen=True)
class AnswerConversation:
    """Turns asked in order within one conversation, each graded like a case."""

    conversation_id: str
    turns: tuple[AnswerCase, ...]


def load_answer_cases(path: str) -> list[AnswerCase]:
    """Load versioned answer ground truth from a local JSON file."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("answer case file must use schema_version 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("answer case file must contain a non-empty cases list")
    cases = [AnswerCase.from_mapping(item) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("answer case ids must be unique")
    return cases


def load_answer_conversations(path: str) -> list[AnswerConversation]:
    """Load the optional ``conversations`` list from an answer case file.

    Each conversation is ``{"id": ..., "turns": [<case>, ...]}``; turns use the
    case schema plus an optional ``expect_follow_up``. A file without
    conversations yields an empty list."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("answer case file must be a JSON object")
    raw = payload.get("conversations") or []
    if not isinstance(raw, list):
        raise ValueError("conversations must be a list")
    conversations: list[AnswerConversation] = []
    for item in raw:
        conversation_id = item.get("id") if isinstance(item, dict) else None
        turns = item.get("turns") if isinstance(item, dict) else None
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation requires a non-empty id")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"conversation {conversation_id} requires a non-empty turns list")
        conversations.append(AnswerConversation(
            conversation_id=conversation_id,
            turns=tuple(AnswerCase.from_mapping(turn) for turn in turns),
        ))
    ids = [c.conversation_id for c in conversations]
    if len(ids) != len(set(ids)):
        raise ValueError("conversation ids must be unique")
    return conversations


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _default_is_refusal(answer: str) -> bool:
    return not _normalized(answer)


@dataclass(frozen=True)
class _Grade:
    satisfied_groups: int
    forbidden_hits: int
    cited: int
    refused: bool


def _grade(case: AnswerCase, answer: str, is_refusal: Callable[[str], bool]) -> _Grade:
    normalized = _normalized(answer)
    satisfied = sum(
        1
        for group in case.must_include
        if any(_normalized(form) in normalized for form in group)
    )
    forbidden = sum(1 for claim in case.must_not_include if _normalized(claim) in normalized)
    # Lenient citation: the source may appear inline or in the references block.
    cited = sum(1 for source in case.must_cite if _normalized(_basename(source)) in normalized)
    return _Grade(
        satisfied_groups=satisfied,
        forbidden_hits=forbidden,
        cited=cited,
        refused=bool(is_refusal(answer)),
    )


def _case_passed(case: AnswerCase, grade: _Grade) -> bool:
    if grade.satisfied_groups != len(case.must_include):
        return False
    if grade.forbidden_hits:
        return False
    if grade.cited != len(case.must_cite):
        return False
    if case.must_not_refuse and grade.refused:
        return False
    return True


# An answer callable returns the response text, or a dict describing the turn:
#   {"answer": str, "evidence": [str, ...], "spoken": str,
#    "continued": bool, "asked_clarification": bool}
# Every key but "answer" is optional; a metric is reported only when its input
# was supplied, so a text-only callable keeps the original report.
AnswerResult = Any


def _turn_fields(result: AnswerResult) -> dict[str, Any]:
    if isinstance(result, dict):
        fields = dict(result)
        fields["answer"] = str(fields.get("answer") or "")
        return fields
    return {"answer": str(result or "")}


def _rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_answers(
    cases: Iterable[AnswerCase],
    answer_query: Callable[[str], AnswerResult],
    *,
    repeat: int = 1,
    is_refusal: Callable[[str], bool] | None = None,
    metadata: dict[str, Any] | None = None,
    include_case_details: bool = False,
    conversations: Iterable[AnswerConversation] = (),
    start_conversation: Callable[[], Callable[[str], AnswerResult]] | None = None,
    find_unsupported: Callable[[str, list[str]], Iterable[str]] | None = None,
    sample_memory: Callable[[], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Score answers against required facts, forbidden claims, and citations.

    `answer_query` answers one independent question. Conversations are asked
    turn by turn through a fresh callable from `start_conversation`, so each
    turn sees the previous ones. `find_unsupported(answer, evidence)` returns
    the figures an answer states that its evidence lacks (a hallucination
    signal). `sample_memory()` returns ``{"rss_mb", "available_pct", "swap_mb"}``
    and is called before the first turn and after every turn, so steady growth
    across identical questions (a leak) and swap use (the step before memory
    exhaustion) show up. `repeat` runs everything several times so
    temperature-driven instability shows up as a rate rather than a phantom
    regression.
    """

    case_list = list(cases)
    conversation_list = list(conversations)
    if not case_list and not conversation_list:
        raise ValueError("at least one answer case is required")
    if conversation_list and start_conversation is None:
        raise ValueError("conversations require start_conversation")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    refusal_check = is_refusal or _default_is_refusal

    runs: list[dict[str, Any]] = []
    memory: list[dict[str, float]] = []

    def take_memory_sample() -> None:
        if sample_memory is None:
            return
        try:
            memory.append(dict(sample_memory()))
        except Exception:  # a failed sample must never fail the evaluation
            pass

    take_memory_sample()

    def run_once(case: AnswerCase, ask: Callable[[str], AnswerResult]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            fields = _turn_fields(ask(case.query))
            error = None
        except Exception as exc:  # a failed turn is a failed case, never a crash
            fields = {"answer": ""}
            error = type(exc).__name__
        answer = fields["answer"]
        grade = _grade(case, answer, refusal_check)
        run: dict[str, Any] = {
            "case": case,
            "passed": _case_passed(case, grade) and error is None,
            "grade": grade,
            "error": error,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "chars": len(answer),
            "empty": not answer.strip(),
        }
        if "spoken" in fields:
            spoken = str(fields.get("spoken") or "")
            run["spoken_chars"] = len(spoken)
            # Brackets are never meant to be read aloud: they mark leaked
            # citation tags or labels the speech cleanup missed.
            run["spoken_leak"] = "[" in spoken or "]" in spoken
        if find_unsupported is not None and fields.get("evidence") is not None:
            flagged = sorted({str(v) for v in find_unsupported(answer, list(fields["evidence"]))})
            run["unsupported"] = bool(flagged)
            run["unsupported_values"] = flagged
        if "asked_clarification" in fields:
            run["clarification"] = bool(fields["asked_clarification"])
        if case.expect_follow_up is not None and "continued" in fields:
            run["follow_up_ok"] = bool(fields["continued"]) == case.expect_follow_up
        take_memory_sample()
        return run

    for case in case_list:
        for _ in range(repeat):
            runs.append(run_once(case, answer_query))
    for conversation in conversation_list:
        for _ in range(repeat):
            ask = start_conversation()
            for turn in conversation.turns:
                run = run_once(turn, ask)
                run["detail_id"] = f"{conversation.conversation_id}/{turn.case_id}"
                runs.append(run)

    graded_items: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        graded_items.setdefault(run.get("detail_id") or run["case"].case_id, []).append(run)

    total_groups = sum(len(items[0]["case"].must_include) for items in graded_items.values())
    total_citations = sum(len(items[0]["case"].must_cite) for items in graded_items.values())
    satisfied_groups = sum(
        sum(r["grade"].satisfied_groups for r in items) / len(items) for items in graded_items.values()
    )
    satisfied_citations = sum(
        sum(r["grade"].cited for r in items) / len(items) for items in graded_items.values()
    )
    flaky_cases = sum(1 for items in graded_items.values() if len({r["passed"] for r in items}) > 1)
    total_runs = len(runs)
    latencies = [r["latency_ms"] for r in runs]

    def optional(key: str) -> list[Any]:
        return [r[key] for r in runs if key in r]

    spoken_chars = optional("spoken_chars")
    unsupported = optional("unsupported")
    clarifications = optional("clarification")
    follow_ups = optional("follow_up_ok")
    leaks = optional("spoken_leak")

    report: dict[str, Any] = {
        "schema_version": 1,
        "case_count": len(case_list),
        "conversation_count": len(conversation_list),
        "conversation_turn_count": sum(len(c.turns) for c in conversation_list),
        "repeat": repeat,
        "run_count": total_runs,
        "answer_errors": sum(1 for r in runs if r["error"]),
        "metrics": {
            "case_pass_rate": _rate(sum(r["passed"] for r in runs), total_runs),
            "fact_group_coverage": satisfied_groups / total_groups if total_groups else None,
            "citation_hit_rate": satisfied_citations / total_citations if total_citations else None,
            "forbidden_claim_rate": _rate(sum(bool(r["grade"].forbidden_hits) for r in runs), total_runs),
            "unsupported_value_rate": _rate(sum(unsupported), len(unsupported)),
            "refusal_rate": _rate(sum(r["grade"].refused for r in runs), total_runs),
            "empty_answer_rate": _rate(sum(r["empty"] for r in runs), total_runs),
            "clarification_rate": _rate(sum(clarifications), len(clarifications)),
            "follow_up_accuracy": _rate(sum(follow_ups), len(follow_ups)),
            "spoken_leak_rate": _rate(sum(leaks), len(leaks)),
            "flaky_case_count": flaky_cases,
            "answer_chars_mean": mean(r["chars"] for r in runs) if runs else None,
            "spoken_chars_mean": mean(spoken_chars) if spoken_chars else None,
            "latency_ms_mean": mean(latencies) if latencies else None,
            "latency_ms_p50": _percentile(latencies, 0.50) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95) if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
        },
    }
    report["metrics"].update(_memory_metrics(memory))
    if metadata:
        report["metadata"] = metadata
    if include_case_details:
        details = []
        for detail_id, items in graded_items.items():
            case = items[0]["case"]
            count = len(items)
            detail = {
                "id": detail_id,
                "pass_rate": round(sum(r["passed"] for r in items) / count, 4),
                "fact_groups_expected": len(case.must_include),
                "fact_groups_satisfied_mean": round(sum(r["grade"].satisfied_groups for r in items) / count, 4),
                "citations_expected": len(case.must_cite),
                "citations_satisfied_mean": round(sum(r["grade"].cited for r in items) / count, 4),
                "forbidden_claim_runs": sum(bool(r["grade"].forbidden_hits) for r in items),
                "refusal_runs": sum(r["grade"].refused for r in items),
                "flaky": len({r["passed"] for r in items}) > 1,
                "latency_ms_mean": round(mean(r["latency_ms"] for r in items), 1),
                "answer_chars_mean": round(mean(r["chars"] for r in items), 1),
            }
            flagged = sorted({v for r in items for v in r.get("unsupported_values", ())})
            if flagged:
                # The figures themselves, so a hallucination flag can be traced.
                detail["unsupported_values"] = flagged
            spoken = [r["spoken_chars"] for r in items if "spoken_chars" in r]
            if spoken:
                detail["spoken_chars_mean"] = round(mean(spoken), 1)
            for key, name in (
                ("unsupported", "unsupported_value_runs"),
                ("follow_up_ok", "follow_up_ok_runs"),
                ("clarification", "clarification_runs"),
            ):
                values = [r[key] for r in items if key in r]
                if values:
                    detail[name] = sum(values)
            details.append(detail)
        report["cases"] = details
    return report


def _memory_metrics(samples: list[dict[str, float]]) -> dict[str, float | None]:
    """Peak, growth, headroom, and swap from the per-turn memory samples.

    Growth is measured from the first post-turn sample, so one-time model and
    index loading during the first answer is not mistaken for a leak."""

    def series(key: str) -> list[float]:
        return [float(s[key]) for s in samples if s.get(key) is not None]

    rss, available, swap = series("rss_mb"), series("available_pct"), series("swap_mb")
    after_turns = rss[1:]
    return {
        "memory_rss_mb_peak": max(rss) if rss else None,
        "memory_rss_mb_growth": after_turns[-1] - after_turns[0] if len(after_turns) > 1 else None,
        "memory_available_pct_min": min(available) if available else None,
        "swap_mb_max": max(swap) if swap else None,
    }


def _paired_case_ratios(base_cases: list[dict[str, Any]], cur_cases: list[dict[str, Any]]) -> dict[str, float]:
    """Median over shared questions of current/baseline per-question means."""

    before = {c["id"]: c for c in base_cases}
    ratios: dict[str, float] = {}
    for key in ("latency_ms_mean", "answer_chars_mean", "spoken_chars_mean"):
        values = [
            c[key] / before[c["id"]][key]
            for c in cur_cases
            if c["id"] in before and c.get(key) is not None and (before[c["id"]].get(key) or 0) > 0
        ]
        if values:
            ratios[key] = median(values)
    return ratios


# Metrics where higher is better, and where lower is better. Hallucination
# metrics get no allowance at all: one more invented figure is a regression.
_HIGHER_IS_BETTER = ("case_pass_rate", "fact_group_coverage", "citation_hit_rate", "follow_up_accuracy")
_LOWER_IS_BETTER = ("refusal_rate", "empty_answer_rate", "clarification_rate", "spoken_leak_rate")
_ZERO_TOLERANCE = ("forbidden_claim_rate", "unsupported_value_rate")


def compare_answer_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    latency_tolerance: float = 0.25,
    tail_latency_tolerance: float = 0.50,
    length_tolerance: float = 0.25,
    memory_tolerance: float = 0.25,
    memory_growth_floor_mb: float = 256.0,
    swap_growth_mb: float = 512.0,
    available_drop_pct: float = 10.0,
) -> tuple[list[str], list[str]]:
    """Return ``(regressions, warnings)`` for ``current`` against ``baseline``.

    Rates may move by one run's worth (1 / run_count) before counting, because
    a single temperature-driven flip is noise; hallucination rates may not move
    at all. Latency, length, and memory use tolerances that are tuned defaults
    rather than universal limits; memory also depends on what else the machine
    is running, so compare runs made under similar load. Reports built from
    different case sets, repeat counts, or models are not comparable and say so
    instead.
    """

    regressions: list[str] = []
    warnings: list[str] = []
    shape = ("case_count", "conversation_turn_count", "repeat")
    for key in shape:
        if baseline.get(key, 0) != current.get(key, 0):
            return [f"not comparable: {key} {baseline.get(key)} -> {current.get(key)}"], []
    base_model = (baseline.get("metadata") or {}).get("llm_model")
    cur_model = (current.get("metadata") or {}).get("llm_model")
    if base_model and cur_model and base_model != cur_model:
        return [f"not comparable: llm_model {base_model} -> {cur_model}"], []

    base, cur = baseline.get("metrics") or {}, current.get("metrics") or {}
    slack = 1.0 / max(1, int(current.get("run_count") or 1)) + 1e-9

    def both(name: str) -> bool:
        return base.get(name) is not None and cur.get(name) is not None

    for name in _HIGHER_IS_BETTER:
        if both(name) and cur[name] < base[name] - slack:
            regressions.append(f"{name} fell {base[name]:.3f} -> {cur[name]:.3f}")
    for name in _LOWER_IS_BETTER:
        if both(name) and cur[name] > base[name] + slack:
            regressions.append(f"{name} rose {base[name]:.3f} -> {cur[name]:.3f}")
    for name in _ZERO_TOLERANCE:
        if both(name) and cur[name] > base[name] + 1e-9:
            regressions.append(f"{name} rose {base[name]:.3f} -> {cur[name]:.3f} (hallucination signal)")
    if (current.get("answer_errors") or 0) > (baseline.get("answer_errors") or 0):
        regressions.append(f"answer_errors rose {baseline.get('answer_errors')} -> {current.get('answer_errors')}")
    if both("flaky_case_count") and cur["flaky_case_count"] > base["flaky_case_count"] + 1:
        regressions.append(f"flaky_case_count rose {base['flaky_case_count']} -> {cur['flaky_case_count']}")
    paired = _paired_case_ratios(baseline.get("cases") or [], current.get("cases") or [])
    checks = [
        ("latency_ms_p95", tail_latency_tolerance),
        ("latency_ms_max", tail_latency_tolerance),
        ("memory_rss_mb_peak", memory_tolerance),
    ]
    if paired:
        # Per-question comparison: the median of each question's own ratio.
        # The median over all runs mixes fast and slow questions, so on a small
        # run it jumps between the two groups with no real change.
        for key, label, tolerance in (
            ("latency_ms_mean", "time to answer", latency_tolerance),
            ("answer_chars_mean", "answer length", length_tolerance),
            ("spoken_chars_mean", "spoken length", length_tolerance),
        ):
            ratio = paired.get(key)
            if ratio is not None and ratio > 1 + tolerance:
                regressions.append(
                    f"{label} per question rose +{ratio - 1:.0%} (median of per-question ratios, limit +{tolerance:.0%})"
                )
    else:
        checks += [
            ("latency_ms_p50", latency_tolerance),
            ("answer_chars_mean", length_tolerance),
            ("spoken_chars_mean", length_tolerance),
        ]
    for name, tolerance in checks:
        if both(name) and base[name] > 0 and cur[name] > base[name] * (1 + tolerance):
            regressions.append(
                f"{name} rose {base[name]:.0f} -> {cur[name]:.0f} (+{cur[name] / base[name] - 1:.0%}, limit +{tolerance:.0%})"
            )
    if both("memory_rss_mb_growth"):
        limit = max(base["memory_rss_mb_growth"], 0.0) * 1.5 + memory_growth_floor_mb
        if cur["memory_rss_mb_growth"] > limit:
            regressions.append(
                f"memory_rss_mb_growth rose {base['memory_rss_mb_growth']:.0f} -> "
                f"{cur['memory_rss_mb_growth']:.0f} MB across the run (possible leak)"
            )
    if both("swap_mb_max") and cur["swap_mb_max"] > base["swap_mb_max"] + swap_growth_mb:
        regressions.append(f"swap_mb_max rose {base['swap_mb_max']:.0f} -> {cur['swap_mb_max']:.0f} MB")
    if both("memory_available_pct_min") and (
        cur["memory_available_pct_min"] < base["memory_available_pct_min"] - available_drop_pct
    ):
        regressions.append(
            f"memory_available_pct_min fell {base['memory_available_pct_min']:.0f}% -> "
            f"{cur['memory_available_pct_min']:.0f}% (less headroom before memory exhaustion)"
        )

    base_cases = {c["id"]: c for c in baseline.get("cases") or []}
    for case in current.get("cases") or []:
        before = base_cases.get(case["id"])
        if not before or case["pass_rate"] >= before["pass_rate"]:
            continue
        message = f"case {case['id']} pass rate {before['pass_rate']:.2f} -> {case['pass_rate']:.2f}"
        if before["pass_rate"] == 1.0 and case["pass_rate"] == 0.0:
            regressions.append(message)
        else:
            warnings.append(message)
    return regressions, warnings


def _regression_key(message: str) -> str:
    """The metric or case a regression message is about, e.g. 'case_pass_rate'."""
    for marker in (" rose", " fell", " pass rate", ":"):
        if marker in message:
            return message.split(marker, 1)[0]
    return message


# Rate metrics are averages over questions, so the same rate can fall in two runs
# for unrelated reasons (a different flaky question each time). These reproduce
# only when at least one question is worse than the baseline in both runs.
_PER_QUESTION_CONFIRMED = (
    "case_pass_rate", "fact_group_coverage", "citation_hit_rate", "follow_up_accuracy",
    "refusal_rate", "empty_answer_rate", "clarification_rate",
)


def _worse_questions(baseline: dict[str, Any], report: dict[str, Any]) -> set[str]:
    """Questions whose pass rate, facts, follow-ups, or clarifications got worse."""

    before = {c["id"]: c for c in baseline.get("cases") or []}
    worse = set()
    for case in report.get("cases") or []:
        old = before.get(case["id"])
        if not old:
            continue
        if (
            case.get("pass_rate", 0) < old.get("pass_rate", 0)
            or case.get("fact_groups_satisfied_mean", 0) < old.get("fact_groups_satisfied_mean", 0)
            or case.get("citations_satisfied_mean", 0) < old.get("citations_satisfied_mean", 0)
            or case.get("follow_up_ok_runs", 0) < old.get("follow_up_ok_runs", 0)
            or case.get("refusal_runs", 0) > old.get("refusal_runs", 0)
            or case.get("clarification_runs", 0) > old.get("clarification_runs", 0)
        ):
            worse.add(case["id"])
    return worse


def confirm_regressions(
    baseline: dict[str, Any], first: dict[str, Any], second: dict[str, Any], **tolerances: Any
) -> tuple[list[str], list[str]]:
    """Regressions that ``first`` and a confirmation run ``second`` both show.

    With a few runs per question one flaky answer can cross a threshold, so a
    regression must reproduce to count: the same metric must regress again and,
    for rate metrics, the same question must be worse in both runs. One that
    does not reproduce becomes a warning; a hallucination flag is never dropped
    silently -- its warning lists the flagged figures per question for review.
    """

    first_regressions, first_warnings = compare_answer_reports(baseline, first, **tolerances)
    second_regressions, _ = compare_answer_reports(baseline, second, **tolerances)
    reproduced = {_regression_key(m) for m in second_regressions}
    both_worse = _worse_questions(baseline, first) & _worse_questions(baseline, second)
    confirmed: list[str] = []
    warnings = list(first_warnings)
    for message in first_regressions:
        key = _regression_key(message)
        if key in reproduced and (key not in _PER_QUESTION_CONFIRMED or both_worse):
            confirmed.append(message)
            continue
        note = f"did not reproduce in the confirmation run: {message}"
        if key in reproduced:
            note = f"different questions regressed in each run (not reproduced): {message}"
        if key in _ZERO_TOLERANCE:
            flagged = [
                f"{case['id']}: {', '.join(case['unsupported_values'])}"
                for report in (first, second)
                for case in report.get("cases") or []
                if case.get("unsupported_values")
            ]
            note += f" -- review the flagged figures: {'; '.join(flagged) or 'none recorded'}"
        warnings.append(note)
    return confirmed, warnings
