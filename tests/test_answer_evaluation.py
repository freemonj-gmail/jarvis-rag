"""Focused tests for deterministic answer-quality metrics.

Uses a stub answer callable, so no model, index, or corpus is required.
"""

import json
import os
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvisrag.evaluation import (
    AnswerCase,
    compare_answer_reports,
    confirm_regressions,
    evaluate_answers,
    load_answer_cases,
    load_answer_conversations,
)


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

    run_regression_tests(cases, answers)

    print(f"answer evaluation tests passed ({len(cases)} cases graded deterministically)")


def run_regression_tests(cases, answers) -> None:
    # Turn details beyond the text: evidence, spoken form, and clarification.
    degree = cases[1]

    def rich(query: str) -> dict:
        return {
            "answer": "A Master of Science, awarded in 2019 with 120 credits.",
            "evidence": ["The record shows a Master of Science, 2019."],
            "spoken": "A Master of Science [source tag].",
            "asked_clarification": False,
        }

    report = evaluate_answers(
        [degree], rich, include_case_details=True,
        find_unsupported=lambda answer, evidence: {"120"} if "120" not in " ".join(evidence) else set(),
    )
    metrics = report["metrics"]
    assert report["cases"][0]["unsupported_values"] == ["120"], report["cases"]
    assert metrics["unsupported_value_rate"] == 1.0, metrics  # 120 is not in the evidence
    assert metrics["spoken_leak_rate"] == 1.0, metrics  # a bracketed tag would be read aloud
    assert metrics["clarification_rate"] == 0.0, metrics
    assert metrics["answer_chars_mean"] == len(rich("")["answer"]), metrics
    assert metrics["latency_ms_max"] is not None
    # A text-only callable reports none of the optional metrics.
    plain = evaluate_answers([degree], lambda query: answers[query])["metrics"]
    assert plain["unsupported_value_rate"] is None and plain["spoken_leak_rate"] is None
    print("- hallucinated figures, spoken leaks, and length are measured")

    # Memory: sampled before the first turn and after each one. Growth starts at
    # the first post-turn sample, so one-time loading is not counted as a leak.
    rss = iter([500.0, 2000.0, 2010.0, 2400.0])
    swap = iter([1300.0, 1300.0, 1400.0, 1600.0])  # 1.3 GB left behind by other apps
    mem_report = evaluate_answers(
        [degree], lambda query: answers[query], repeat=3,
        sample_memory=lambda: {"rss_mb": next(rss), "available_pct": 60.0, "swap_mb": next(swap)},
    )["metrics"]
    assert mem_report["memory_rss_mb_peak"] == 2400.0, mem_report
    assert mem_report["memory_rss_mb_growth"] == 400.0, mem_report
    assert mem_report["swap_mb_max"] == 1600.0 and mem_report["memory_available_pct_min"] == 60.0
    # Only the swap the run itself adds is counted, not what was there before it.
    assert mem_report["swap_mb_growth"] == 300.0, mem_report
    def broken_sampler():
        raise OSError("no permission")
    assert evaluate_answers([degree], lambda q: answers[q], sample_memory=broken_sampler)["metrics"][
        "memory_rss_mb_peak"] is None, "a failing sampler must not fail the run"
    print("- memory peak, growth across the run, headroom, and swap are measured")

    # Conversations: each turn sees the previous ones; follow-up handling is graded.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cases.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "cases": [{"id": "a", "query": "q", "must_include": ["x"]}],
                "conversations": [{"id": "c1", "turns": [
                    {"id": "t1", "query": "What is the limit?", "must_include": ["$1,000"], "expect_follow_up": False},
                    {"id": "t2", "query": "And next year?", "must_include": ["$1,100"], "expect_follow_up": True},
                ]}],
            }, handle)
        conversations = load_answer_conversations(path)
    assert [t.case_id for t in conversations[0].turns] == ["t1", "t2"]

    def start():
        history: list[str] = []

        def ask(query: str) -> dict:
            history.append(query)
            answer = "The limit is $1,000." if len(history) == 1 else "Next year it is $1,100."
            return {"answer": answer, "continued": len(history) > 1}

        return ask

    conv_report = evaluate_answers(
        [], None, conversations=conversations, start_conversation=start, repeat=2, include_case_details=True
    )
    assert conv_report["conversation_turn_count"] == 2 and conv_report["run_count"] == 4
    assert conv_report["metrics"]["follow_up_accuracy"] == 1.0
    assert conv_report["metrics"]["case_pass_rate"] == 1.0
    assert {c["id"] for c in conv_report["cases"]} == {"c1/t1", "c1/t2"}
    serialized = json.dumps(conv_report)
    assert "limit" not in serialized and "1,100" not in serialized, "conversation report leaked text"
    print("- conversations graded turn by turn, follow-ups checked, no text in the report")

    # The comparator: worse on any metric is a regression; one run of noise is not.
    def report_with(**metrics):
        base = {
            "case_pass_rate": 1.0, "fact_group_coverage": 1.0, "citation_hit_rate": 1.0,
            "forbidden_claim_rate": 0.0, "unsupported_value_rate": 0.0, "refusal_rate": 0.0,
            "empty_answer_rate": 0.0, "clarification_rate": 0.1, "follow_up_accuracy": 1.0,
            "spoken_leak_rate": 0.0, "flaky_case_count": 0, "answer_chars_mean": 400.0,
            "spoken_chars_mean": 380.0, "latency_ms_p50": 20000.0, "latency_ms_p95": 40000.0,
            "latency_ms_max": 50000.0, "memory_rss_mb_peak": 8000.0, "memory_rss_mb_growth": 100.0,
            "memory_available_pct_min": 60.0, "swap_mb_max": 200.0, "swap_mb_growth": 50.0,
        }
        base.update(metrics)
        return {"case_count": 10, "conversation_turn_count": 0, "repeat": 2, "run_count": 20,
                "answer_errors": 0, "metrics": base, "metadata": {"llm_model": "m"}}

    baseline = report_with()
    assert compare_answer_reports(baseline, report_with()) == ([], [])
    within = report_with(case_pass_rate=0.95, latency_ms_p50=24000.0, answer_chars_mean=480.0,
                         memory_rss_mb_peak=9000.0, memory_rss_mb_growth=300.0, swap_mb_growth=500.0,
                         memory_available_pct_min=55.0)
    assert compare_answer_reports(baseline, within)[0] == [], "one flipped run or small drift is noise"
    for worse, metric in (
        (dict(case_pass_rate=0.9), "case_pass_rate"),
        (dict(forbidden_claim_rate=0.05), "forbidden_claim_rate"),
        (dict(unsupported_value_rate=0.05), "unsupported_value_rate"),
        (dict(latency_ms_p50=26000.0), "latency_ms_p50"),
        (dict(latency_ms_max=80000.0), "latency_ms_max"),
        (dict(answer_chars_mean=520.0), "answer_chars_mean"),
        (dict(spoken_chars_mean=500.0), "spoken_chars_mean"),
        (dict(follow_up_accuracy=0.8), "follow_up_accuracy"),
        (dict(refusal_rate=0.15), "refusal_rate"),
        (dict(memory_rss_mb_peak=10500.0), "memory_rss_mb_peak"),
        (dict(memory_rss_mb_growth=900.0), "memory_rss_mb_growth"),
        (dict(swap_mb_growth=1000.0), "swap_mb_growth"),
        (dict(memory_available_pct_min=45.0), "memory_available_pct_min"),
    ):
        regressions, _ = compare_answer_reports(baseline, report_with(**worse))
        assert any(metric in r for r in regressions), (metric, regressions)
    # Startup must be clearly slower to count: a second of cache warmth is noise.
    timed = lambda seconds: dict(baseline, metadata={"llm_model": "m", "startup_seconds": seconds})
    assert compare_answer_reports(timed(4.0), timed(5.5))[0] == []
    assert any("startup_seconds" in r for r in compare_answer_reports(timed(4.0), timed(9.0))[0])
    # Swap left over from before the run is not a regression; swap the run adds is.
    assert compare_answer_reports(baseline, report_with(swap_mb_max=1800.0))[0] == []
    baseline_cases = dict(baseline, cases=[{"id": "a", "pass_rate": 1.0}, {"id": "b", "pass_rate": 1.0}])
    current_cases = dict(report_with(), cases=[{"id": "a", "pass_rate": 0.0}, {"id": "b", "pass_rate": 0.5}])
    regressions, warnings = compare_answer_reports(baseline_cases, current_cases)
    assert any("case a" in r for r in regressions) and any("case b" in w for w in warnings)
    other_set = dict(report_with(), case_count=11)
    assert compare_answer_reports(baseline, other_set)[0][0].startswith("not comparable")

    # With per-question details, time and length are compared question by question.
    def detailed(times, chars, p50):
        r = report_with(latency_ms_p50=p50)
        r["cases"] = [{"id": f"q{i}", "pass_rate": 1.0, "latency_ms_mean": t, "answer_chars_mean": c}
                      for i, (t, c) in enumerate(zip(times, chars))]
        return r
    fast_slow = [6000.0, 8000.0, 10000.0, 30000.0, 40000.0, 50000.0]
    lengths = [300.0] * 6
    # The overall median moved from the fast group to the slow one; no question got slower.
    same_speed = detailed(fast_slow, lengths, p50=10000.0), detailed(fast_slow, lengths, p50=30000.0)
    assert compare_answer_reports(*same_speed)[0] == [], "median jump with unchanged questions is not a regression"
    slower = detailed([t * 1.4 for t in fast_slow], lengths, p50=14000.0)
    regressions, _ = compare_answer_reports(same_speed[0], slower)
    assert any("time to answer per question" in r for r in regressions), regressions
    longer = detailed(fast_slow, [450.0] * 6, p50=10000.0)
    assert any("answer length per question" in r for r in compare_answer_reports(same_speed[0], longer)[0])

    # Confirmation: a regression counts only if a second run reproduces it; a
    # hallucination flag that does not reproduce is kept as a warning with its figures.
    cased = dict(baseline, cases=[{"id": "q1", "pass_rate": 1.0}, {"id": "q2", "pass_rate": 1.0}])

    def run_with(q1, q2, **metrics):
        return dict(report_with(**metrics), cases=[
            {"id": "q1", "pass_rate": q1, "unsupported_values": ["4,321"] if metrics.get("unsupported_value_rate") else []},
            {"id": "q2", "pass_rate": q2},
        ])
    flicker = run_with(1.0, 1.0, unsupported_value_rate=0.05)
    confirmed, warnings = confirm_regressions(cased, flicker, run_with(1.0, 1.0))
    assert confirmed == [], confirmed
    assert any("did not reproduce" in w and "q1: 4,321" in w for w in warnings), warnings
    # The same question worse in both runs is a confirmed regression...
    confirmed, _ = confirm_regressions(cased, run_with(0.0, 1.0, case_pass_rate=0.85), run_with(0.5, 1.0, case_pass_rate=0.85))
    assert [c.split(" ")[0] for c in confirmed] == ["case_pass_rate"], confirmed
    # ...but a different flaky question in each run is not.
    confirmed, warnings = confirm_regressions(cased, run_with(0.0, 1.0, case_pass_rate=0.85), run_with(1.0, 0.5, case_pass_rate=0.85))
    assert confirmed == [] and any("different questions" in w for w in warnings), (confirmed, warnings)
    print("- regression comparator flags worse metrics and tolerates one run of noise")

    # Speech wording: a token counts as read correctly when one of its readings
    # reaches the voice; an initialism left as written counts as its spoken letters.
    from jarvisrag.evaluation.speech import SpeechCase, SpokenToken, compare_speech_reports, evaluate_speech

    cases = [SpeechCase("x", "A 2 ct. ring from 1903.", (
        SpokenToken("ct.", ("carat",)), SpokenToken("1903", ("nineteen oh three",), ("one nine zero three",)),
    )), SpeechCase("y", "Ask the IRS.", (SpokenToken("IRS", ("i r s",)),))]
    worded = {"A 2 ct. ring from 1903.": "A two carat ring from one nine zero three.", "Ask the IRS.": "Ask the IRS."}
    report = evaluate_speech(cases, lambda text: ("\n".join(worded[t] for t in text.split("\n")), 0.5))
    assert report["metrics"]["natural_rate"] == 2 / 3, report
    assert report["metrics"]["acceptable_rate"] == 1.0, report
    slower = dict(report, metrics=dict(report["metrics"], latency_ms_p50=report["metrics"]["latency_ms_p50"] + 5000))
    worse = dict(report, metrics=dict(report["metrics"], acceptable_rate=0.9))
    assert compare_speech_reports(report, report) == []
    assert [p.split()[0] for p in compare_speech_reports(report, worse)] == ["acceptable_rate"]
    assert [p.split()[0] for p in compare_speech_reports(report, slower)] == ["latency_ms_p50"]
    # Time is compared only between versions that both have the context step.
    assert compare_speech_reports(dict(report, context_step=False), slower) == []
    print("- speech wording scores readings and flags worse metrics")

    # Voice check: word error rate between the words sent and the words heard.
    from jarvisrag.evaluation.voice import compare_voice_reports, voice_report, word_error_rate

    assert word_error_rate("won in 1903 again", "won in 1903 again") == 0.0
    assert word_error_rate("won in 1903 again", "won in again") == 0.25  # a dropped word
    # The same words written in another notation are not an error.
    assert word_error_rate("henry the 8th married", "henry viii married") == 0.0
    assert word_error_rate("world war 2 ended", "world war ii ended") == 0.0
    assert compare_voice_reports(voice_report("chatterbox", [("x", 0.0)]), voice_report("chatterbox", [("x", 0.0), ("y", 0.0)]))[0].startswith("sentence sets differ")
    base = voice_report("chatterbox", [("x", 0.0), ("y", 0.1)])
    assert compare_voice_reports(base, voice_report("chatterbox", [("x", 0.05), ("y", 0.1)])) == []
    drifted = voice_report("chatterbox", [("x", 0.5), ("y", 0.6)])
    assert [p.split()[0] for p in compare_voice_reports(base, drifted)] == ["mean_wer"]
    assert compare_voice_reports(base, voice_report("kokoro", [("x", 0.0)]))[0].startswith("engine differs")
    print("- voice check scores dropped words and flags drift")


if __name__ == "__main__":
    run_tests()
