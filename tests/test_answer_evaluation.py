"""Focused tests for deterministic answer-quality metrics.

Uses a stub answer callable, so no model, index, or corpus is required.
"""

import json
import os
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvisrag.evaluation import AnswerCase, evaluate_answers, load_answer_cases


def run_tests() -> None:
    cases = [
        AnswerCase(
            case_id="employment",
            query="Which organizations were worked for?",
            must_include=(("Acme Corporation", "Acme Corp"), ("Bedrock Institute",)),
            must_not_include=("Initech",),
            must_cite=("example-cv.docx",),
        ),
        AnswerCase(
            case_id="degree",
            query="What graduate degree was earned?",
            must_include=(("Master of Science",),),
        ),
    ]

    answers = {
        # Uses the second accepted surface form, omits nothing, cites the source.
        "Which organizations were worked for?": (
            "Worked at Acme Corp and the Bedrock Institute (example-cv.docx)."
        ),
        "What graduate degree was earned?": "The record shows a Master of Science.",
    }
    report = evaluate_answers(cases, lambda query: answers[query])
    metrics = report["metrics"]
    assert metrics["case_pass_rate"] == 1.0, metrics
    assert metrics["fact_group_coverage"] == 1.0, metrics
    assert metrics["citation_hit_rate"] == 1.0, metrics
    assert metrics["forbidden_claim_rate"] == 0.0, metrics
    assert metrics["flaky_case_count"] == 0
    print("- alternative surface forms and lenient citation OK")

    # A fluent, confident, wrong answer: the failure class preference scoring misses.
    confabulated = dict(answers)
    confabulated["Which organizations were worked for?"] = (
        "Worked at Acme Corporation, Bedrock Institute, and Initech (example-cv.docx)."
    )
    report = evaluate_answers(cases, lambda query: confabulated[query])
    assert report["metrics"]["forbidden_claim_rate"] == 0.5, report["metrics"]
    assert report["metrics"]["case_pass_rate"] == 0.5, report["metrics"]
    print("- forbidden claim detected despite all required facts present")

    # A missing citation fails the case even when every fact is present.
    uncited = dict(answers)
    uncited["Which organizations were worked for?"] = "Worked at Acme Corp and Bedrock Institute."
    report = evaluate_answers(cases, lambda query: uncited[query], include_case_details=True)
    assert report["metrics"]["fact_group_coverage"] == 1.0
    assert report["metrics"]["citation_hit_rate"] == 0.0
    detail = next(case for case in report["cases"] if case["id"] == "employment")
    assert detail["pass_rate"] == 0.0
    print("- missing citation fails the case")

    # Refusals are graded by the pipeline's own notion of a dismissive answer.
    refusal_report = evaluate_answers(
        [cases[1]],
        lambda query: "I could not find that in the documents.",
        is_refusal=lambda answer: "could not find" in answer,
    )
    assert refusal_report["metrics"]["refusal_rate"] == 1.0
    assert refusal_report["metrics"]["case_pass_rate"] == 0.0
    print("- refusal detection OK")

    # An exception during a turn is a failed case, never a crashed run.
    def explode(_query: str) -> str:
        raise RuntimeError("model unavailable")

    error_report = evaluate_answers([cases[1]], explode)
    assert error_report["answer_errors"] == 1
    assert error_report["metrics"]["case_pass_rate"] == 0.0
    print("- retrieval/answer errors counted as failures")

    # Instability shows up as a pass rate, not a phantom regression.
    toggle = {"n": 0}

    def flaky(_query: str) -> str:
        toggle["n"] += 1
        return "Master of Science" if toggle["n"] % 2 else "no comment"

    flaky_report = evaluate_answers([cases[1]], flaky, repeat=2, include_case_details=True)
    assert flaky_report["metrics"]["flaky_case_count"] == 1
    assert flaky_report["metrics"]["case_pass_rate"] == 0.5
    assert flaky_report["cases"][0]["flaky"] is True
    print("- repeat exposes flaky cases")

    # Reports must not leak queries or answers.
    serialized = json.dumps(
        evaluate_answers(cases, lambda query: answers[query], include_case_details=True)
    )
    for secret in ("Acme", "Bedrock", "organizations were worked for", "example-cv"):
        assert secret not in serialized, f"report leaked {secret!r}"
    print("- report contains no query, answer, or source text")

    # Schema validation, including the bare-string shorthand for a single form.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cases.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "cases": [{"id": "a", "query": "q", "must_include": ["one form"]}],
            }, handle)
        loaded = load_answer_cases(path)
        assert loaded[0].must_include == (("one form",),)

        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 2, "cases": []}, handle)
        try:
            load_answer_cases(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected schema version rejection")
    print("- case schema loading and validation OK")

    print(f"answer evaluation tests passed ({len(cases)} cases graded deterministically)")


if __name__ == "__main__":
    run_tests()
