"""Deterministic retrieval metrics for sequential Jarvis experiments.

The evaluator receives evidence lines from an injected retrieval callable. It
does not initialize models, contact MCP services, mutate an index, or print
queries and document contents. Reports contain aggregate metrics by default so
private ground-truth cases can remain local and untracked.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from jarvisrag.text.query import (
    _evidence_body_from_evidence_line,
    _source_label_from_evidence_line,
)


_PAGE_SUFFIX_RE = re.compile(r",\s*page\s+(\d+)\s*$", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceTarget:
    """One source passage that should appear in retrieved evidence."""

    source: str | None = None
    page: int | None = None
    contains_all: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvidenceTarget":
        source = value.get("source")
        page = value.get("page")
        contains = value.get("contains_all") or []
        if source is not None and not isinstance(source, str):
            raise ValueError("target source must be a string")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
            raise ValueError("target page must be an integer")
        if not isinstance(contains, list) or not all(isinstance(item, str) for item in contains):
            raise ValueError("target contains_all must be a list of strings")
        if not source and page is None and not contains:
            raise ValueError("target must define source, page, or contains_all")
        return cls(source=source, page=page, contains_all=tuple(contains))


@dataclass(frozen=True)
class RetrievalCase:
    """A private-safe retrieval query and its expected evidence targets."""

    case_id: str
    query: str
    relevant: tuple[EvidenceTarget, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RetrievalCase":
        case_id = value.get("id")
        query = value.get("query")
        relevant = value.get("relevant")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case id must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"case {case_id!r} query must be a non-empty string")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"case {case_id!r} relevant must be a non-empty list")
        targets = tuple(EvidenceTarget.from_mapping(item) for item in relevant)
        return cls(case_id=case_id, query=query, relevant=targets)


def load_retrieval_cases(path: str) -> list[RetrievalCase]:
    """Load versioned retrieval ground truth from a local JSON file."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("retrieval case file must use schema_version 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("retrieval case file must contain a non-empty cases list")
    cases = [RetrievalCase.from_mapping(item) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval case ids must be unique")
    return cases


def _normalized(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _evidence_parts(evidence_line: str) -> tuple[str, int | None, str]:
    label = _source_label_from_evidence_line(evidence_line)
    page_match = _PAGE_SUFFIX_RE.search(label)
    page = int(page_match.group(1)) if page_match else None
    source_label = _PAGE_SUFFIX_RE.sub("", label).strip()
    body = _evidence_body_from_evidence_line(evidence_line)
    return source_label, page, body


def _target_matches(target: EvidenceTarget, evidence_line: str) -> bool:
    source_label, page, body = _evidence_parts(evidence_line)
    if target.source:
        expected_source = os.path.basename(target.source).casefold()
        actual_source = os.path.basename(source_label).casefold()
        if expected_source != actual_source:
            return False
    if target.page is not None and target.page != page:
        return False
    normalized_body = _normalized(body)
    return all(_normalized(fragment) in normalized_body for fragment in target.contains_all)


def _first_matching_rank(targets: Sequence[EvidenceTarget], evidence: Sequence[str]) -> int | None:
    for rank, line in enumerate(evidence, start=1):
        if any(_target_matches(target, line) for target in targets):
            return rank
    return None


def _max_distinct_matches(targets: Sequence[EvidenceTarget], evidence: Sequence[str]) -> int:
    """Largest number of targets satisfiable by *distinct* evidence lines.

    Without this, several expected passages could all be credited to one
    retrieved line and overstate recall. Maximum bipartite matching keeps one
    line per target.
    """

    if not targets or not evidence:
        return 0
    candidates = [
        [index for index, line in enumerate(evidence) if _target_matches(target, line)]
        for target in targets
    ]
    line_owner: dict[int, int] = {}

    def _assign(target_index: int, visited: set[int]) -> bool:
        for line_index in candidates[target_index]:
            if line_index in visited:
                continue
            visited.add(line_index)
            owner = line_owner.get(line_index)
            if owner is None or _assign(owner, visited):
                line_owner[line_index] = target_index
                return True
        return False

    return sum(_assign(index, set()) for index in range(len(targets)))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _word_set(value: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD_RE.finditer(value)}


def _near_duplicate_count(evidence: Sequence[str], threshold: float) -> int:
    accepted: list[set[str]] = []
    duplicate_count = 0
    for line in evidence:
        words = _word_set(_evidence_body_from_evidence_line(line))
        is_duplicate = False
        for prior in accepted:
            union = words | prior
            similarity = len(words & prior) / len(union) if union else 1.0
            if similarity >= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            duplicate_count += 1
        else:
            accepted.append(words)
    return duplicate_count


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def evaluate_retrieval(
    cases: Iterable[RetrievalCase],
    retrieve_evidence: Callable[[str], Sequence[str]],
    *,
    cutoffs: Sequence[int] = (5, 10, 30),
    near_duplicate_threshold: float = 0.9,
    include_case_details: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an unchanged retrieval path and return JSON-safe metrics.

    Recall is micro-averaged over all expected evidence targets, each of which
    must be satisfied by a distinct evidence line. MRR and case hit rate use the
    first matching target for each query. Retrieval exceptions are counted as
    misses so one bad case does not erase the rest of a run.
    """

    case_list = list(cases)
    normalized_cutoffs = tuple(sorted({int(value) for value in cutoffs if int(value) > 0}))
    if not case_list:
        raise ValueError("at least one retrieval case is required")
    if not normalized_cutoffs:
        raise ValueError("at least one positive recall cutoff is required")
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    target_hits = {cutoff: 0 for cutoff in normalized_cutoffs}
    total_targets = sum(len(case.relevant) for case in case_list)
    page_target_count = 0
    page_target_hits = 0
    reciprocal_ranks: list[float] = []
    latency_ms: list[float] = []
    evidence_counts: list[float] = []
    evidence_chars: list[float] = []
    duplicate_lines = 0
    total_evidence_lines = 0
    retrieval_errors = 0
    case_details: list[dict[str, Any]] = []
    largest_cutoff = normalized_cutoffs[-1]

    for case in case_list:
        started = time.perf_counter()
        error = False
        try:
            evidence = list(retrieve_evidence(case.query))
        except Exception:
            evidence = []
            error = True
            retrieval_errors += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)
        evidence_counts.append(float(len(evidence)))
        char_count = sum(len(line) for line in evidence)
        evidence_chars.append(float(char_count))
        total_evidence_lines += len(evidence)
        duplicate_lines += _near_duplicate_count(evidence, near_duplicate_threshold)

        first_rank = _first_matching_rank(case.relevant, evidence)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for cutoff in normalized_cutoffs:
            target_hits[cutoff] += _max_distinct_matches(case.relevant, evidence[:cutoff])
        page_targets = [target for target in case.relevant if target.page is not None]
        if page_targets:
            page_target_count += len(page_targets)
            page_target_hits += _max_distinct_matches(page_targets, evidence[:largest_cutoff])

        if include_case_details:
            case_details.append({
                "id": case.case_id,
                "first_relevant_rank": first_rank,
                "evidence_lines": len(evidence),
                "evidence_chars": char_count,
                "latency_ms": round(elapsed_ms, 3),
                "error": error,
            })

    duplicate_ratio = duplicate_lines / total_evidence_lines if total_evidence_lines else None
    report: dict[str, Any] = {
        "schema_version": 1,
        "case_count": len(case_list),
        "target_count": total_targets,
        "retrieval_errors": retrieval_errors,
        "metrics": {
            "recall_at": {
                str(cutoff): target_hits[cutoff] / total_targets
                for cutoff in normalized_cutoffs
            },
            "mrr": mean(reciprocal_ranks),
            "case_hit_rate_at_max_k": sum(rank > 0 for rank in reciprocal_ranks) / len(case_list),
            "source_page_hit_rate_at_max_k": (
                page_target_hits / page_target_count if page_target_count else None
            ),
            "near_duplicate_evidence_ratio": duplicate_ratio,
            "unique_evidence_ratio": None if duplicate_ratio is None else 1.0 - duplicate_ratio,
            "evidence_lines_mean": mean(evidence_counts),
            "evidence_lines_p95": _percentile(evidence_counts, 0.95),
            "evidence_chars_mean": mean(evidence_chars),
            "evidence_chars_p95": _percentile(evidence_chars, 0.95),
            "latency_ms_mean": mean(latency_ms),
            "latency_ms_p50": _percentile(latency_ms, 0.50),
            "latency_ms_p95": _percentile(latency_ms, 0.95),
            "process_peak_rss_bytes_with_model_load": _peak_rss_bytes(),
        },
    }
    if metadata:
        report["metadata"] = dict(metadata)
    if include_case_details:
        report["cases"] = case_details
    return report
