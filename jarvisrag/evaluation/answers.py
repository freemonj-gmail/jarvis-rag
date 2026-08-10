"""Deterministic answer-quality metrics for Jarvis experiments.

Retrieval metrics stop at "did the right passage reach the prompt". This module
measures the answer itself: required facts present, forbidden claims absent, the
response not a refusal, and expected sources cited.

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
from statistics import mean
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

        return cls(
            case_id=case_id,
            query=query,
            must_include=tuple(groups),
            must_not_include=tuple(forbidden),
            must_cite=tuple(citations),
            must_not_refuse=refuse,
        )


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


def evaluate_answers(
    cases: Iterable[AnswerCase],
    answer_query: Callable[[str], str],
    *,
    repeat: int = 1,
    is_refusal: Callable[[str], bool] | None = None,
    metadata: dict[str, Any] | None = None,
    include_case_details: bool = False,
) -> dict[str, Any]:
    """Score answers against required facts, forbidden claims, and citations.

    `answer_query` returns the response text for a query. `repeat` runs each case
    several times so temperature-driven instability shows up as a pass rate
    rather than a phantom regression.
    """

    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one answer case is required")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    refusal_check = is_refusal or _default_is_refusal

    total_groups = sum(len(case.must_include) for case in case_list)
    total_citations = sum(len(case.must_cite) for case in case_list)
    satisfied_groups = 0
    satisfied_citations = 0
    forbidden_runs = 0
    refusal_runs = 0
    passed_runs = 0
    total_runs = 0
    answer_errors = 0
    flaky_cases = 0
    latencies: list[float] = []
    case_details: list[dict[str, Any]] = []

    for case in case_list:
        outcomes: list[bool] = []
        case_groups = 0
        case_citations = 0
        case_forbidden = 0
        case_refusals = 0
        for _ in range(repeat):
            started = time.perf_counter()
            try:
                answer = answer_query(case.query)
                error = None
            except Exception as exc:  # a failed turn is a failed case, never a crash
                answer = ""
                error = type(exc).__name__
                answer_errors += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)

            grade = _grade(case, answer, refusal_check)
            passed = _case_passed(case, grade) and error is None
            outcomes.append(passed)
            total_runs += 1
            passed_runs += int(passed)
            case_groups += grade.satisfied_groups
            case_citations += grade.cited
            case_forbidden += int(bool(grade.forbidden_hits))
            case_refusals += int(grade.refused)

        satisfied_groups += case_groups / repeat
        satisfied_citations += case_citations / repeat
        forbidden_runs += case_forbidden
        refusal_runs += case_refusals
        if len(set(outcomes)) > 1:
            flaky_cases += 1

        if include_case_details:
            case_details.append({
                "id": case.case_id,
                "pass_rate": round(sum(outcomes) / repeat, 4),
                "fact_groups_expected": len(case.must_include),
                "fact_groups_satisfied_mean": round(case_groups / repeat, 4),
                "citations_expected": len(case.must_cite),
                "citations_satisfied_mean": round(case_citations / repeat, 4),
                "forbidden_claim_runs": case_forbidden,
                "refusal_runs": case_refusals,
                "flaky": len(set(outcomes)) > 1,
            })

    report: dict[str, Any] = {
        "schema_version": 1,
        "case_count": len(case_list),
        "repeat": repeat,
        "run_count": total_runs,
        "answer_errors": answer_errors,
        "metrics": {
            "case_pass_rate": passed_runs / total_runs if total_runs else None,
            "fact_group_coverage": satisfied_groups / total_groups if total_groups else None,
            "citation_hit_rate": satisfied_citations / total_citations if total_citations else None,
            "forbidden_claim_rate": forbidden_runs / total_runs if total_runs else None,
            "refusal_rate": refusal_runs / total_runs if total_runs else None,
            "flaky_case_count": flaky_cases,
            "latency_ms_mean": mean(latencies) if latencies else None,
            "latency_ms_p50": _percentile(latencies, 0.50) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95) if latencies else None,
        },
    }
    if metadata:
        report["metadata"] = metadata
    if include_case_details:
        report["cases"] = case_details
    return report
