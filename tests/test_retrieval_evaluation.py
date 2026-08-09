"""Focused tests for privacy-safe retrieval evaluation metrics."""

import json
import io
import logging
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvisrag.evaluation import (
    EvidenceTarget,
    RetrievalCase,
    evaluate_retrieval,
    load_retrieval_cases,
)
from scripts.evaluate_retrieval import _private_runtime_output


def run_tests() -> None:
    cases = [
        RetrievalCase(
            case_id="policy-condition",
            query="What condition applies to the policy?",
            relevant=(
                EvidenceTarget(
                    source="handbook.pdf",
                    page=2,
                    contains_all=("three years", "twenty days"),
                ),
            ),
        ),
        RetrievalCase(
            case_id="complete-list",
            query="List all recorded items.",
            relevant=(
                EvidenceTarget(source="records.pdf", page=7, contains_all=("first item",)),
                EvidenceTarget(source="appendix.pdf", page=9, contains_all=("final item",)),
            ),
        ),
    ]
    evidence_by_query = {
        cases[0].query: [
            "- overview.pdf, page 1: General policy introduction.",
            "- handbook.pdf, page 2: After three years, employees receive twenty days.",
            "- copy.pdf, page 4: After three years, employees receive twenty days.",
        ],
        cases[1].query: [
            "- records.pdf, page 7: First item in the register.",
            "- notes.pdf, page 1: Background note alpha.",
            "- notes.pdf, page 2: Background note beta.",
            "- notes.pdf, page 3: Background note gamma.",
            "- notes.pdf, page 4: Background note delta.",
            "- appendix.pdf, page 9: Final item in the appendix.",
        ],
    }

    report = evaluate_retrieval(cases, evidence_by_query.__getitem__)
    metrics = report["metrics"]
    assert report["case_count"] == 2
    assert report["target_count"] == 3
    assert report["retrieval_errors"] == 0
    assert metrics["recall_at"] == {"5": 2 / 3, "10": 1.0, "30": 1.0}
    assert metrics["mrr"] == 0.75
    assert metrics["case_hit_rate_at_max_k"] == 1.0
    assert metrics["source_page_hit_rate_at_max_k"] == 1.0
    assert metrics["near_duplicate_evidence_ratio"] == 1 / 9
    assert metrics["unique_evidence_ratio"] == 8 / 9
    assert metrics["evidence_lines_mean"] == 4.5
    assert metrics["evidence_lines_p95"] == 6.0
    assert metrics["latency_ms_p50"] >= 0.0
    assert metrics["latency_ms_p95"] >= metrics["latency_ms_p50"]
    assert metrics["process_peak_rss_bytes_with_model_load"] > 0
    assert "cases" not in report
    serialized = json.dumps(report)
    assert cases[0].query not in serialized
    assert "handbook.pdf" not in serialized
    assert "twenty days" not in serialized

    detailed = evaluate_retrieval(
        cases,
        evidence_by_query.__getitem__,
        include_case_details=True,
        metadata={"variant": "synthetic-baseline"},
    )
    assert detailed["metadata"] == {"variant": "synthetic-baseline"}
    assert [item["id"] for item in detailed["cases"]] == [
        "policy-condition",
        "complete-list",
    ]
    assert detailed["cases"][0]["first_relevant_rank"] == 2
    assert detailed["cases"][1]["first_relevant_rank"] == 1
    assert "query" not in detailed["cases"][0]

    payload = {
        "schema_version": 1,
        "cases": [
            {
                "id": "schema-check",
                "query": "Which section contains the rule?",
                "relevant": [
                    {
                        "source": "manual.pdf",
                        "page": 3,
                        "contains_all": ["required condition"],
                    }
                ],
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        loaded = load_retrieval_cases(handle.name)
    assert loaded == [
        RetrievalCase(
            case_id="schema-check",
            query="Which section contains the rule?",
            relevant=(
                EvidenceTarget(
                    source="manual.pdf",
                    page=3,
                    contains_all=("required condition",),
                ),
            ),
        )
    ]

    error_case = RetrievalCase(
        case_id="retrieval-error",
        query="Trigger a retrieval failure.",
        relevant=(EvidenceTarget(source="missing.pdf"),),
    )

    def failing_retriever(_query: str) -> list[str]:
        raise RuntimeError("synthetic failure")

    error_report = evaluate_retrieval([error_case], failing_retriever)
    assert error_report["retrieval_errors"] == 1
    assert error_report["metrics"]["recall_at"] == {"5": 0.0, "10": 0.0, "30": 0.0}
    assert error_report["metrics"]["mrr"] == 0.0

    shared_line_case = RetrievalCase(
        case_id="shared-line",
        query="Which two conditions apply?",
        relevant=(
            EvidenceTarget(source="combined.pdf", contains_all=("first condition",)),
            EvidenceTarget(source="combined.pdf", contains_all=("second condition",)),
        ),
    )
    shared = evaluate_retrieval(
        [shared_line_case],
        lambda _query: ["- combined.pdf: first condition and second condition together"],
    )
    assert shared["metrics"]["recall_at"]["5"] == 0.5, "each target needs its own evidence line"

    split = evaluate_retrieval(
        [shared_line_case],
        lambda _query: [
            "- combined.pdf: first condition stated here",
            "- combined.pdf: second condition stated separately",
        ],
    )
    assert split["metrics"]["recall_at"]["5"] == 1.0

    empty_report = evaluate_retrieval([shared_line_case], lambda _query: [])
    assert empty_report["metrics"]["near_duplicate_evidence_ratio"] is None
    assert empty_report["metrics"]["unique_evidence_ratio"] is None

    hidden_stdout = io.StringIO()
    hidden_stderr = io.StringIO()
    with redirect_stdout(hidden_stdout), redirect_stderr(hidden_stderr):
        with _private_runtime_output(False):
            print("private extracted text")
            print("private warning", file=sys.stderr)
            logging.getLogger("jarvis-test").warning("private log")
    assert hidden_stdout.getvalue() == ""
    assert hidden_stderr.getvalue() == ""

    print("retrieval evaluation tests passed (3 targets, 9 evidence lines)")


if __name__ == "__main__":
    run_tests()
