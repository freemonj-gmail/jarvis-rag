"""Measure Jarvis answer quality against local ground truth.

Runs the real `handle_query` pipeline so grounding, the review loop,
distillation, and the fallback lanes are all exercised. Grading is
deterministic; no model judges the answer.

MCP is force-disabled: it makes outbound calls carrying the query text, which
would make runs non-reproducible and send private questions to a third party.

Reports contain aggregate metrics only. Answers quote the corpus, so neither
answers nor queries are ever written to the report.

Regression gate: pass ``--baseline <earlier report>`` to compare this run with a
saved one. A metric that looks worse is re-measured in a second run (written
next to the report as ``*.confirm.json``) and counts only if it reproduces; the
run then exits 1 when any metric is confirmed worse: accuracy, hallucination
(forbidden claims, figures absent from the evidence), time to answer, answer and
spoken length, follow-up handling, refusals, empty answers, errors, or memory
(process peak and growth across the run, lowest free system memory, swap added during the run).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time

# Must happen before any jarvisrag import: config freezes these at import time.
# Hermetic run - no outbound calls, and no query text leaving the machine.
os.environ["JARVIS_ENABLE_MCP"] = "0"
os.environ["JARVIS_MCP_CONFIG"] = "jarvis-mcp-disabled-for-evaluation.json"
os.environ.setdefault("JARVIS_ENABLE_TTS", "0")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jarvisrag.evaluation import (
    compare_answer_reports,
    confirm_regressions,
    evaluate_answers,
    load_answer_cases,
    load_answer_conversations,
)


@contextlib.contextmanager
def _private_runtime_output(verbose: bool):
    """Hide pipeline logs and printed answers, which contain private text."""

    if verbose:
        yield
        return
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            with contextlib.redirect_stdout(null_stream), contextlib.redirect_stderr(null_stream):
                yield
    finally:
        logging.disable(previous_disable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Jarvis answer quality through the real query pipeline. "
            "The report omits queries, answers, and document text."
        )
    )
    parser.add_argument("corpus", help="local corpus directory used to build the runtime")
    parser.add_argument("cases", help="local schema_version=1 answer case JSON file")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument(
        "--persist-dir",
        help="isolated local metadata/cache directory (sets JARVIS_PERSIST_DIR)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per case; use 3 when baselining to expose temperature-driven flakiness",
    )
    parser.add_argument(
        "--baseline",
        help="earlier report from the same case file; exit 1 if this run is worse on any metric",
    )
    parser.add_argument(
        "--case-details",
        action="store_true",
        help="include case IDs, pass rates, and counts; never includes query or answer text",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show pipeline logs and answers; exposes private document content",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = load_answer_cases(args.cases)
    conversations = load_answer_conversations(args.cases)

    if args.persist_dir:
        os.environ["JARVIS_PERSIST_DIR"] = os.path.abspath(os.path.expanduser(args.persist_dir))

    from jarvisrag.answer.answer import (
        _evidence_source_labels,
        _response_is_dismissive,
        _unsupported_answer_values,
        handle_query,
    )
    from jarvisrag.speech import _spoken_response_text
    import psutil

    process = psutil.Process()

    def sample_memory() -> dict:
        system = psutil.virtual_memory()
        return {
            "rss_mb": process.memory_info().rss / 2**20,
            "available_pct": system.available / system.total * 100.0,
            "swap_mb": psutil.swap_memory().used / 2**20,
        }
    from jarvisrag.answer.bootstrap import DEFAULT_RETRIEVAL_TOP_K, build_runtime

    print("Building answer runtime...", flush=True)
    startup_started = time.perf_counter()
    with _private_runtime_output(args.verbose):
        runtime = build_runtime(args.corpus)
    startup_seconds = time.perf_counter() - startup_started

    if getattr(runtime, "mcp_registry", None) is not None:
        active = getattr(runtime.mcp_registry, "active", None)
        if callable(active) and active():
            print("Refusing to run: MCP layer is active and would leak query text.", file=sys.stderr)
            return 2

    def ask(query: str, prior: dict | None) -> tuple[dict, dict]:
        with _private_runtime_output(args.verbose):
            turn = handle_query(query, runtime, prior)
        answer = str(turn.get("response_text") or "")
        evidence = list(turn.get("evidence_lines") or [])
        result = {
            "answer": answer,
            # Answers from the math or general-knowledge paths carry no evidence,
            # so there is nothing to check their figures against.
            "evidence": evidence or None,
            # Spoken exactly as Jarvis speaks it, so the gate measures the real speech.
            "spoken": _spoken_response_text(answer, _evidence_source_labels(evidence)),
            "asked_clarification": bool(turn.get("awaiting_clarification")),
            "continued": str(turn.get("effective_query") or query).strip() != query.strip(),
        }
        return result, turn

    def answer_query(query: str) -> dict:
        # Each case is an independent turn; no prior turn is chained in, so
        # follow-up carry-over cannot leak one case's state into the next.
        return ask(query, None)[0]

    def start_conversation():
        state: dict = {"turn": None}

        def ask_turn(query: str) -> dict:
            result, state["turn"] = ask(query, state["turn"])
            return result

        return ask_turn

    turn_count = sum(len(c.turns) for c in conversations)
    print(
        f"Answering {len(cases)} cases + {turn_count} conversation turns x{args.repeat} "
        "(this runs the real model)...",
        flush=True,
    )
    def run_evaluation() -> dict:
        return evaluate_answers(
            cases,
            answer_query,
            repeat=args.repeat,
            is_refusal=_response_is_dismissive,
            include_case_details=args.case_details,
            conversations=conversations,
            start_conversation=start_conversation,
            find_unsupported=_unsupported_answer_values,
            sample_memory=sample_memory,
            metadata={
                "startup_seconds": round(startup_seconds, 3),
                # Report the model that actually answered, not a duplicated default.
                "llm_model": getattr(runtime.llm, "model", "unknown"),
                "retrieval_top_k": int(os.environ.get("JARVIS_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K)),
                "max_evidence_lines": int(os.environ.get("JARVIS_MAX_EVIDENCE_LINES", "30")),
                "suppress_duplicate_evidence": True,
                "milvus_collection": os.environ.get("MILVUS_COLLECTION", "jarvis_context"),
            },
        )

    report = run_evaluation()

    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        print(f"Answer evaluation report written ({len(cases)} aggregate cases).")
    else:
        print(serialized)

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        regressions, warnings = compare_answer_reports(baseline, report)
        if regressions:
            # A few runs per question let one flaky answer cross a threshold, so a
            # regression must reproduce in a second run to count.
            print(f"{len(regressions)} metric(s) worse than the baseline; confirming with a second run...", flush=True)
            confirmation = run_evaluation()
            if args.output:
                root, ext = os.path.splitext(args.output)
                with open(f"{root}.confirm{ext or '.json'}", "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(confirmation, indent=2, sort_keys=True) + "\n")
            regressions, warnings = confirm_regressions(baseline, report, confirmation)
        for message in warnings:
            print(f"WARNING: {message}")
        for message in regressions:
            print(f"REGRESSION: {message}")
        if regressions:
            print(f"Regression gate FAILED: {len(regressions)} metric(s) worse than the baseline.")
            return 1
        print("Regression gate passed: no metric worse than the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
