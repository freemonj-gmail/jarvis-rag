"""Answer-pipeline checks: synthesis, review and refine, figure grounding,
fallbacks, the domain cascade, evidence budgets and topic, routing, and
clarifying questions.

Uses mocked models and synthetic evidence, never real documents, a network
service, or an LLM. Run with ``make answer-test`` or directly through uv.
"""

import contextlib
import io
import os
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from tests.helpers import _patch_answer


class AnswerBudgetTests(unittest.TestCase):
    """The per-turn wall-clock budget must bound the LLM fallback chain: once the
    budget is spent, no further distillation (a slow model call) is STARTED, yet a
    non-empty answer still comes back from the fast pure-Python extractive
    fallbacks. This is the guard against the stacked-generation hang where a slow
    synthesis returned empty and the fallback chain then launched more multi-minute
    model calls with no time bound.
    """

    def _run_with_empty_synthesis(self, budget):
        from jarvisrag.answer import pipeline as answer_module

        evidence = ["- doc.pdf: The 2026 elective deferral limit is $24,500."]
        distill_calls = {"n": 0}

        def fake_distill(*_a, **_k):
            distill_calls["n"] += 1
            return "DISTILLED ANSWER"

        with patch.dict(os.environ, {"JARVIS_ANSWER_TIME_BUDGET_SECONDS": str(budget)}), \
                _patch_answer("_synthesize_evidence_answer", return_value=""), \
                _patch_answer("_extractive_reasoning_answer_from_evidence", return_value=""), \
                _patch_answer("_extractive_summary_answer_from_evidence", return_value=""), \
                _patch_answer("_distill_answer_from_evidence", side_effect=fake_distill), \
                _patch_answer("_forced_extractive_summary_from_evidence", return_value="FORCED SUMMARY"):
            answer, ok, _reason = answer_module._answer_with_review_loop("what is the limit", evidence, object())
        return answer, ok, distill_calls["n"]

    def test_spent_budget_skips_the_slow_distillation_call(self):
        answer, ok, distills = self._run_with_empty_synthesis(budget=0)
        self.assertEqual(distills, 0)  # no slow LLM call started once the budget is gone
        self.assertTrue(ok)
        self.assertEqual(answer, "FORCED SUMMARY")  # fast fallback still yields an answer

    def test_available_budget_still_uses_distillation(self):
        answer, ok, distills = self._run_with_empty_synthesis(budget=1000)
        self.assertEqual(distills, 1)  # with budget remaining, the distill IS attempted
        self.assertEqual(answer, "DISTILLED ANSWER")


class RunawayGenerationTests(unittest.TestCase):
    """An evidence answer that reaches the token ceiling has run away: it is
    discarded, and a runaway refinement falls back to the earlier draft."""

    class _CapLLM:
        def __init__(self, done_reason):
            self.additional_kwargs = {"num_predict": 256}
            self.done_reason, self.cap_seen = done_reason, None

        def chat(self, messages):
            self.cap_seen = self.additional_kwargs.get("num_predict")
            resp = Mock()
            resp.message = Mock()
            resp.message.content = "The limit is $1,000 " * 50
            resp.raw = {"done_reason": self.done_reason}
            return resp

    def test_capped_synthesis_is_discarded(self):
        from jarvisrag.answer.generation import _ANSWER_MAX_TOKENS
        from jarvisrag.answer.synthesis import _synthesize_evidence_answer
        evidence = ["- doc.pdf: The limit is $1,000."]
        runaway = self._CapLLM("length")
        self.assertEqual(_synthesize_evidence_answer("What is the limit?", evidence, runaway), "")
        self.assertEqual(runaway.cap_seen, _ANSWER_MAX_TOKENS)
        self.assertEqual(runaway.additional_kwargs["num_predict"], 256)  # restored afterwards
        finished = self._CapLLM("stop")
        self.assertTrue(_synthesize_evidence_answer("What is the limit?", evidence, finished))

    def test_runaway_refinement_keeps_the_earlier_draft(self):
        from jarvisrag.answer import pipeline as answer_module
        draft = "The limit is $1,000."
        reviewer = lambda *_a, **_k: (False, "the answer is too generic")
        with _patch_answer("_synthesize_evidence_answer", side_effect=[draft, ""]), \
                _patch_answer("_review_answer_against_evidence", side_effect=reviewer), \
                _patch_answer("_extractive_reasoning_answer_from_evidence", return_value=""), \
                _patch_answer("_distill_answer_from_evidence", return_value=""), \
                _patch_answer("_forced_extractive_summary_from_evidence", return_value="FORCED SUMMARY"):
            answer, ok, _reason = answer_module._answer_with_review_loop(
                "What is the limit?", ["- doc.pdf: The limit is $1,000."], object()
            )
        self.assertTrue(ok)
        self.assertTrue(answer.startswith(draft))
        self.assertNotIn("FORCED SUMMARY", answer)


class ReviewerCorroborationTests(unittest.TestCase):
    """A reviewer's hallucination claim discards an answer only when the evidence agrees."""

    EVIDENCE = [
        "- limits.html: For 2026 the 401(k) elective deferral limit is $24,500 and the catch-up is $8,000.",
        "- ira.html: The 2026 IRA limit is $7,500 ($8,600 if age 50 or older).",
    ]

    def _run(self, answer_text):
        from jarvisrag.answer import pipeline as answer_module

        reviewer = lambda *_a, **_k: (False, "these values are not present in the evidence")
        with _patch_answer("_synthesize_evidence_answer", return_value=answer_text), \
                _patch_answer("_review_answer_against_evidence", side_effect=reviewer), \
                _patch_answer("_extractive_reasoning_answer_from_evidence", return_value=""), \
                _patch_answer("_extractive_summary_answer_from_evidence", return_value="EXTRACTIVE"), \
                _patch_answer("_distill_answer_from_evidence", return_value=""), \
                _patch_answer("_forced_extractive_summary_from_evidence", return_value=""):
            return answer_module._answer_with_review_loop("are the limits the same", self.EVIDENCE, object())

    def test_answer_is_kept_when_every_value_is_in_the_evidence(self):
        text = "No. The 2026 IRA limit is $7,500 ($8,600 at 50+); the 401(k) limit is $24,500 plus $8,000."
        answer, ok, reason = self._run(text)
        self.assertEqual(answer, text)
        self.assertTrue(ok)
        self.assertIn("not corroborated", reason)

    def test_answer_is_discarded_when_a_value_is_missing_from_the_evidence(self):
        answer, ok, _reason = self._run("The 2026 IRA limit is $9,999.")
        self.assertEqual(answer, "EXTRACTIVE")
        self.assertTrue(ok)

    def test_value_extraction_normalizes_formatting(self):
        from jarvisrag.answer.grounding import _unsupported_answer_values

        self.assertEqual(_unsupported_answer_values("Limits: $7,500, $24,500.", self.EVIDENCE), set())
        self.assertEqual(_unsupported_answer_values("A rise of 12% to $1,234.50.", self.EVIDENCE), {"12", "1234.5"})
        # Equal numbers match however they are written.
        self.assertEqual(_unsupported_answer_values("GPA 2.540, limit $7500.00", ["GPA 2.54; $7,500"]), set())
        # A comma groups thousands only before three digits: a date written without a
        # space ("Jan 29,2003") is a day and a year, not 292003.
        self.assertEqual(_unsupported_answer_values("Dated Jan 29, 2003.", ["Report, Dated Jan  29,2003,as 1.21Ct."]), set())
        self.assertEqual(_unsupported_answer_values("It cost $1,000, then $24,500.", ["$1,000 and $24,500"]), set())

    def test_passed_answer_with_an_invented_value_is_sent_back(self):
        from jarvisrag.answer import pipeline as answer_module

        drafts = iter(["The cumulative GPA is 2.731.", "The cumulative GPA is 2.54."])
        feedback = []

        def synthesize(*_a, review_feedback="", **_k):
            feedback.append(review_feedback)
            return next(drafts)

        with _patch_answer("_synthesize_evidence_answer", side_effect=synthesize), \
                _patch_answer("_review_answer_against_evidence", return_value=(True, "ok")):
            answer, ok, _reason = answer_module._answer_with_review_loop(
                "What cumulative GPA appears in the 2019 transcript?", ["- transcript.pdf: Cumulative 2.540"], object()
            )
        self.assertEqual(answer, "The cumulative GPA is 2.54.")
        self.assertTrue(ok)
        self.assertIn("2.731", feedback[1])

    def _failed_review_run(self, draft, distilled):
        from jarvisrag.answer import pipeline as answer_module

        # Review fails for a reason that is not about invented content, so the
        # loop falls back to distillation and then to the draft itself.
        with _patch_answer("_synthesize_evidence_answer", return_value=draft), \
                _patch_answer("_review_answer_against_evidence", return_value=(False, "does not answer the question")), \
                _patch_answer("_distill_answer_from_evidence", return_value=distilled), \
                _patch_answer("_forced_extractive_summary_from_evidence", return_value="EXTRACTIVE"):
            return answer_module._answer_with_review_loop(
                "What cumulative averages appear?", ["- transcript.pdf: Cumulative 2.731"], object()
            )

    def test_distilled_answer_with_an_invented_value_is_not_returned(self):
        answer, _, _ = self._failed_review_run("Cumulative 2.731.", "The cumulative average is 2.54.")
        # The invented distillation is dropped; the grounded draft is surfaced instead.
        self.assertTrue(answer.startswith("Cumulative 2.731."))
        self.assertNotIn("2.54", answer)

    def test_unverified_draft_with_an_invented_value_is_not_surfaced(self):
        answer, ok, _ = self._failed_review_run("The cumulative average is 2.54.", "")
        self.assertEqual(answer, "EXTRACTIVE")
        self.assertTrue(ok)

    def test_grounded_distillation_is_still_used(self):
        answer, _, reason = self._failed_review_run("Cumulative 2.731.", "The cumulative average is 2.731.")
        self.assertEqual(answer, "The cumulative average is 2.731.")
        self.assertIn("distilled", reason)

    def test_values_from_the_question_are_not_invented(self):
        from jarvisrag.answer import pipeline as answer_module

        with _patch_answer("_synthesize_evidence_answer", return_value="The evidence has no 2031 figure."), \
                _patch_answer("_review_answer_against_evidence", return_value=(True, "ok")):
            answer, ok, _ = answer_module._answer_with_review_loop(
                "What is the 2031 standard deduction?", ["- rp.pdf: 2026 standard deduction $16,100"], object()
            )
        self.assertEqual(answer, "The evidence has no 2031 figure.")
        self.assertTrue(ok)

    def test_source_references_are_not_claims(self):
        from jarvisrag.answer.grounding import _unsupported_answer_values

        # Page numbers and file names cite a source; they are not stated values.
        cited = "The IRA limit is $7,500 (Notice_2026-66.pdf, page 05); see also page 12 and pp. 3-4."
        self.assertEqual(_unsupported_answer_values(cited, self.EVIDENCE), set())
        # A figure inside the same citation is still checked.
        self.assertEqual(
            _unsupported_answer_values("The IRA limit is $7,500 (Notice_2026-66.pdf, page 05, $9,999).", self.EVIDENCE),
            {"9999"},
        )


class ValueQuestionSynthesisTests(unittest.TestCase):
    """A value question with evidence is answered by synthesis + review, never by a
    pre-model keyword shortcut that pairs the question's own words with nearby numbers."""

    EVIDENCE = [
        "- rates.pdf, page 3: Schedule A. Not over $10,000: 5% of the amount. Over $10,000 but not over $40,000: "
        "$500 plus 9% of the excess over $10,000. Penalty for late payment: $350.",
        "- rates.pdf, page 4: Schedule B. Not over $5,000: 5% of the amount. Payment due within 30 days.",
        "- notes.pdf: Filing fees are $10 per form; amounts owed are withdrawn by direct debit.",
    ]

    def test_value_question_goes_to_the_review_loop(self):
        import jarvisrag.answer.pipeline as ans
        runtime = ans.QueryRuntime(all_docs=[], llm=Mock(), doc_retriever=object(), domain_retrievers=[])
        loop = Mock(return_value=("SYNTHESIZED ANSWER", True, "ok"))
        with _patch_answer("_retrieve_query_supported_evidence", return_value=(list(self.EVIDENCE), [])
        ), _patch_answer("fetch_mcp_evidence", return_value=[]), _patch_answer("_answer_with_review_loop", loop
        ), _patch_answer("_answer_completeness", return_value=ans.ANSWER_COMPLETE), \
                contextlib.redirect_stdout(io.StringIO()):
            turn = ans.handle_query("What are the schedule A rates for filing amounts owed?", runtime)
        loop.assert_called_once()
        self.assertEqual(turn["response_text"], "SYNTHESIZED ANSWER")


class EvidenceBudgetTests(unittest.TestCase):
    """The global evidence ceiling must bound the diluting per-source pile WITHOUT
    ever truncating the authoritative (MCP + domain) tiers -- so a value the domain
    retriever ranked deep in its own fully-retained list can never be cut here.
    Pure, domain-agnostic list assembly.
    """

    def _lines(self, tag, n):
        return [f"- {tag}{i}.pdf: {tag} evidence {i}" for i in range(n)]

    def test_domain_is_never_truncated_even_when_ranked_deep(self):
        from jarvisrag.answer.evidence import _budget_combined_evidence

        mcp = self._lines("mcp", 1)
        domain = self._lines("dom", 30)  # answer sits at domain position 21 (index 20)
        main = self._lines("main", 30)
        out = _budget_combined_evidence(mcp, domain, main, budget=40, main_reserve=6)
        # every domain line survives, in order, right after MCP
        self.assertEqual(out[: len(mcp) + len(domain)], mcp + domain)
        self.assertIn("- dom20.pdf: dom evidence 20", out)  # the deep linking chunk
        self.assertLessEqual(len(out), 40)

    def test_main_is_reserved_when_a_domain_routed(self):
        from jarvisrag.answer.evidence import _budget_combined_evidence

        out = _budget_combined_evidence(self._lines("mcp", 1), self._lines("dom", 30),
                                        self._lines("main", 30), budget=40, main_reserve=6)
        main_kept = [l for l in out if l.startswith("- main")]
        self.assertEqual(len(main_kept), 6)  # main capped to the reserve, not flooding

    def test_authoritative_over_budget_is_kept_and_main_dropped(self):
        from jarvisrag.answer.evidence import _budget_combined_evidence

        # MCP(12) + domain(30) = 42 authoritative lines > budget 40
        out = _budget_combined_evidence(self._lines("mcp", 12), self._lines("dom", 30),
                                        self._lines("main", 30), budget=40, main_reserve=6)
        self.assertEqual(len(out), 42)  # authoritative never sacrificed
        self.assertFalse([l for l in out if l.startswith("- main")])  # main dropped to fit

    def test_no_domain_leaves_main_untouched(self):
        from jarvisrag.answer.evidence import _budget_combined_evidence

        out = _budget_combined_evidence([], [], self._lines("main", 30), budget=40, main_reserve=6)
        self.assertEqual(len(out), 30)  # single-corpus path unchanged, no reserve applied


class PeriodClarificationTests(unittest.TestCase):
    """Deterministic year clarification: a quantity question with no year, over
    evidence spanning multiple years, must ask WHICH year -- agnostically (keyed
    only on quantity-intent + multiple years, never on a topic/value/corpus).
    """

    def _ev(self, *years):
        return [f"- src{y}.pdf: the limit for {y} is a value" for y in years]

    def test_asks_year_when_quantity_query_lacks_year_and_evidence_spans_years(self):
        from jarvisrag.answer.clarify import _period_clarification_from_evidence

        q = _period_clarification_from_evidence(
            "what is the contribution limit for people over 50", self._ev(2024, 2025, 2026)
        )
        self.assertTrue(q.endswith("?") or "?" in q)
        self.assertIn("2026", q)
        self.assertIn("2024", q)

    def test_silent_when_query_already_names_a_year(self):
        from jarvisrag.answer.clarify import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence(
                "what is the 2026 contribution limit", self._ev(2024, 2025, 2026)
            ),
            "",
        )

    def test_silent_when_evidence_spans_one_year(self):
        from jarvisrag.answer.clarify import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence("what is the limit", self._ev(2026)), ""
        )

    def test_silent_for_non_quantity_question(self):
        from jarvisrag.answer.clarify import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence(
                "who is eligible to participate", self._ev(2024, 2025, 2026)
            ),
            "",
        )

    def test_currency_symbol_counts_as_quantity_intent(self):
        from jarvisrag.answer.clarify import _period_clarification_from_evidence

        q = _period_clarification_from_evidence(
            "is the $8,000 the same for everyone", self._ev(2025, 2026)
        )
        self.assertIn("year", q.lower())


class FallbackConcisionTests(unittest.TestCase):
    """The last-resort evidence-preview fallbacks must stay CONCISE: an empty
    synthesis must never emit a wall of raw snippets (the 'answer entirely too
    long' failure). Bounded to a few short snippets, env-overridable."""

    def _long_lines(self, n):
        return [f"- Doc{i}.pdf: " + ("word " * 800) for i in range(n)]

    def test_forced_summary_is_capped_in_items_and_length(self):
        from jarvisrag.answer.fallbacks import _forced_extractive_summary_from_evidence

        out = _forced_extractive_summary_from_evidence("what is the limit", self._long_lines(10))
        self.assertLessEqual(out.count("\n- "), 3)          # at most 3 snippet bullets
        self.assertLess(len(out), 2000)                     # nowhere near the old ~8KB dump

    def test_summary_respects_env_budget(self):
        from jarvisrag.answer.fallbacks import _forced_extractive_summary_from_evidence
        import os

        os.environ["JARVIS_FALLBACK_SUMMARY_ITEMS"] = "2"
        os.environ["JARVIS_FALLBACK_SUMMARY_CHARS"] = "150"
        try:
            out = _forced_extractive_summary_from_evidence("what is the limit", self._long_lines(10))
            self.assertLessEqual(out.count("\n- "), 2)
        finally:
            del os.environ["JARVIS_FALLBACK_SUMMARY_ITEMS"]
            del os.environ["JARVIS_FALLBACK_SUMMARY_CHARS"]


class SynthesisBoundedOutputTests(unittest.TestCase):
    """The synthesis prompt always carries the bounded-output directive (the fix
    for the runaway 'Synthesizing 10m00s' -> empty -> raw-dump)."""

    class _CapturingLLM:
        def __init__(self):
            self.additional_kwargs = {}
            self.captured = ""

        def chat(self, messages):
            self.captured = messages[0].content
            resp = Mock()
            resp.message = Mock()
            resp.message.content = "A grounded answer."
            return resp

    def _run(self):
        from jarvisrag.answer.synthesis import _synthesize_evidence_answer
        llm = self._CapturingLLM()
        out = _synthesize_evidence_answer(
            "What is the RMA number?",
            ["- Email.emlx: RMA is ECR-1 (Email.emlx)"],
            llm,
        )
        return out, llm.captured

    def test_directive_is_always_present(self):
        out, prompt = self._run()
        self.assertEqual(out, "A grounded answer.")
        self.assertIn("ANSWER SHAPE", prompt)

    def test_status_fidelity_rule_applies_to_any_question(self):
        from jarvisrag.answer.synthesis import _STATUS_FIDELITY_RULE
        _out, prompt = self._run()
        self.assertIn(_STATUS_FIDELITY_RULE.strip(), prompt)


class RoutingTermTests(unittest.TestCase):
    """Routing terms for MCP connectors and domain corpora match at the start of
    a word, never inside an unrelated word."""

    def test_term_matches_at_word_start_only(self):
        from jarvisrag.mcp.registry import routing_term_matches
        for term, text in (("irs", "Who was the first woman to win?"), ("opm", "the development plan"),
                           ("irc", "under these circumstances"), ("tax", "fix the syntax error")):
            self.assertFalse(routing_term_matches(term, text), (term, text))
        for term, text in (("irs", "What did the IRS say?"), ("tax", "Is my pension taxable?"),
                           ("deduction", "Which deductions apply?"), ("401(k)", "my 401(k) limit"),
                           ("w-2", "where is my W-2"), ("tax year", "for the 2026 tax year")):
            self.assertTrue(routing_term_matches(term, text), (term, text))

    def test_connector_and_domain_routing_share_the_rule(self):
        from jarvisrag.answer.domains import _domains_for_turn
        from jarvisrag.mcp.registry import MCPConnector
        connector = MCPConnector(name="tax", enabled=True, transport="http", url="", match_any=["irs", "tax"])
        self.assertFalse(connector.matches("Who was the first person to walk on Mars?"))
        self.assertTrue(connector.matches("When are taxes due?"))
        domains = [{"name": "benefits", "retriever": object(), "match_any": ["opm", "fers"]}]
        self.assertEqual(_domains_for_turn("Summarize the development plan and the offers", domains), ([], []))
        active, matched = _domains_for_turn("What does OPM say about FERS?", domains)
        self.assertEqual([d["name"] for d in matched], ["benefits"])


class DomainCascadeTests(unittest.TestCase):
    """The failure-triggered domain cascade (_cascade_domain_answer) consults
    domains left un-consulted this turn ONLY when the primary answer did not fully
    answer, and adopts a re-synthesis only when it is supported and supplies at
    least part of what was asked. Keyed on answer completeness, not topic
    keywords, so it stays domain-agnostic."""

    @staticmethod
    def _domains():
        return [{"name": "mail", "retriever": object(), "match_any": ["email"]}]

    def test_no_unconsulted_domains_returns_none(self):
        from jarvisrag.answer.domains import _cascade_domain_answer
        self.assertIsNone(_cascade_domain_answer("q", [], [], llm=Mock()))

    def test_no_domain_evidence_returns_none(self):
        import jarvisrag.answer.pipeline as ans
        with _patch_answer("_retrieve_domain_evidence", return_value=([], [], set())):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    def test_supported_answer_is_adopted(self):
        import jarvisrag.answer.pipeline as ans
        with _patch_answer("_retrieve_domain_evidence",
            return_value=(["- Email.emlx: RMA is ECR-1 (Email.emlx)"], ["Email.emlx"], set()),
        ), _patch_answer("_answer_with_review_loop", return_value=("The RMA number is ECR-1.", True, "ok")
        ):
            out = ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
        self.assertIsNotNone(out)
        self.assertEqual(out["response_text"], "The RMA number is ECR-1.")
        self.assertEqual(out["refs"], ["Email.emlx"])
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertTrue(out["evidence_lines"])

    def test_dismissive_resynthesis_returns_none(self):
        import jarvisrag.answer.pipeline as ans
        with _patch_answer("_retrieve_domain_evidence",
            return_value=(["- Email.emlx: unrelated text (Email.emlx)"], ["Email.emlx"], set()),
        ), _patch_answer("_answer_with_review_loop",
            return_value=("The evidence does not contain an RMA number.", True, "ok"),
        ):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    def test_failed_review_returns_none(self):
        import jarvisrag.answer.pipeline as ans
        with _patch_answer("_retrieve_domain_evidence",
            return_value=(["- Email.emlx: x (Email.emlx)"], ["Email.emlx"], set()),
        ), _patch_answer("_answer_with_review_loop", return_value=("", False, "no")):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    @staticmethod
    def _two_domains():
        return [
            {"name": "reference", "retriever": object(), "index": "ref-index"},
            {"name": "mail", "retriever": object(), "index": "mail-index"},
        ]

    @staticmethod
    def _per_domain_evidence(query, domains):
        name = domains[0]["name"]
        return ([f"- {name}.doc: {name} text ({name}.doc)"], [f"{name}.doc"], set())

    def test_each_domain_is_retrieved_in_isolation(self):
        # Pooled retrieval let one corpus fill the shared cap and starve the
        # next; every retrieval call must carry exactly one domain.
        import jarvisrag.answer.pipeline as ans
        calls = []

        def record(query, domains):
            calls.append([d["name"] for d in domains])
            return self._per_domain_evidence(query, domains)

        with _patch_answer("_retrieve_domain_evidence", side_effect=record), _patch_answer("_answer_with_review_loop", return_value=("The evidence does not contain it.", True, "ok")
        ):
            ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(sorted(calls), [["mail"], ["reference"]])

    def test_falls_through_to_domain_that_answers(self):
        import jarvisrag.answer.pipeline as ans

        def answer(query, evidence, llm):
            if any("mail" in line for line in evidence):
                return ("The RMA number is ECR-1.", True, "ok")
            return ("The evidence does not contain an RMA number.", True, "ok")

        with _patch_answer("_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), _patch_answer("_answer_with_review_loop", side_effect=answer), _patch_answer("_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertIsNotNone(out)
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertEqual(out["refs"], ["mail.doc"])
        self.assertFalse(any("reference" in line for line in out["evidence_lines"]))

    def test_most_relevant_domain_is_tried_first(self):
        import jarvisrag.answer.pipeline as ans
        order = []

        def record(query, domains):
            order.append(domains[0]["name"])
            return self._per_domain_evidence(query, domains)

        with _patch_answer("_retrieve_domain_evidence", side_effect=record), _patch_answer("_answer_with_review_loop", return_value=("The RMA number is ECR-1.", True, "ok")
        ), _patch_answer("_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.2, "mail-index": 0.7}[idx]
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(order, ["mail"])  # first domain answered; the other is never paid for
        self.assertEqual(out["domain_names"], ["mail"])

    def test_attachment_passages_join_evidence_before_one_synthesis(self):
        # Attachments are read BEFORE synthesis, not by a retry keyed on the
        # answer's hedge wording (which missed "is not provided").
        import jarvisrag.answer.pipeline as ans
        seen = []

        def synth(query, evidence, llm):
            seen.append(list(evidence))
            return ("The tracking code is ZX-1.", True, "ok")

        attachment = "- mail.doc: [Attachment: label.pdf] TRACKING #: ZX-1"
        with _patch_answer("_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), _patch_answer("_attachment_evidence_lines", return_value=[attachment]), _patch_answer("_answer_with_review_loop", side_effect=synth
        ):
            out = ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], attachment)
        self.assertEqual(out["response_text"], "The tracking code is ZX-1.")

    def test_complete_answer_from_later_domain_beats_earlier_partial(self):
        import jarvisrag.answer.pipeline as ans
        from jarvisrag.answer import verdicts as answer_verdicts
        order = []

        def answer(query, evidence, llm):
            name = "mail" if any("mail" in line for line in evidence) else "reference"
            order.append(name)
            return (f"{name} answer", True, "ok")

        with _patch_answer("_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), _patch_answer("_answer_with_review_loop", side_effect=answer), _patch_answer("_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ), _patch_answer("_answer_completeness",
            side_effect=lambda q, text, llm: answer_verdicts.ANSWER_PARTIAL if text.startswith("reference") else ans.ANSWER_COMPLETE,
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(order, ["reference", "mail"])
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertEqual(out["completeness"], ans.ANSWER_COMPLETE)

    def test_first_partial_is_kept_when_no_domain_answers_fully(self):
        import jarvisrag.answer.pipeline as ans
        from jarvisrag.answer import verdicts as answer_verdicts
        with _patch_answer("_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), _patch_answer("_answer_with_review_loop", side_effect=lambda q, ev, llm: (f"{ev[0][2:8]} answer", True, "ok")
        ), _patch_answer("_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ), _patch_answer("_answer_completeness", return_value=answer_verdicts.ANSWER_PARTIAL):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(out["domain_names"], ["reference"])
        self.assertEqual(out["completeness"], answer_verdicts.ANSWER_PARTIAL)

    def test_long_non_answer_is_not_adopted_whatever_its_wording(self):
        # Longer than the dismissive-length guard and free of hedge markers, so
        # only the meaning-based verdict can reject it.
        import jarvisrag.answer.pipeline as ans
        from jarvisrag.answer import verdicts as answer_verdicts
        non_answer = (
            "These records cover retirement plan contribution limits, survivor annuities and "
            "earnings tests for several years; a lamp return and its tracking number are "
            "nowhere among them, so that part of your question is outside this material entirely."
        )
        self.assertFalse(ans._response_is_dismissive(non_answer))
        self.assertFalse(answer_verdicts._answer_hedges_about_missing_detail(non_answer))
        with _patch_answer("_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), _patch_answer("_answer_with_review_loop", return_value=(non_answer, True, "ok")
        ), _patch_answer("_answer_completeness", return_value=ans.ANSWER_UNANSWERED):
            self.assertIsNone(ans._cascade_domain_answer("q", self._domains(), [], llm=Mock()))


class AnswerCompletenessTests(unittest.TestCase):
    """_answer_completeness judges whether an answer supplies what was asked by
    meaning, so a non-answer is caught however it is worded."""

    class _VerdictLLM:
        def __init__(self, reply=None, error=None):
            self.reply, self.error, self.prompts = reply, error, []

        def chat(self, messages):
            self.prompts.append(messages[0].content)
            if self.error:
                raise self.error
            resp = Mock()
            resp.message = Mock()
            resp.message.content = self.reply
            return resp

    # Hedges in wordings the phrase list never contained.
    _PARAPHRASED = "Your blender order number is BX-1042. The delivery date isn't listed in these records."

    def _verdict(self, reply, answer=_PARAPHRASED, error=None):
        from jarvisrag.answer.verdicts import _answer_completeness
        llm = self._VerdictLLM(reply, error)
        return _answer_completeness("What is the order number and delivery date?", answer, llm), llm

    def test_unanswered_is_not_mistaken_for_answered(self):
        from jarvisrag.answer.verdicts import ANSWER_UNANSWERED
        self.assertEqual(self._verdict("UNANSWERED")[0], ANSWER_UNANSWERED)

    def test_partial_and_answered_verdicts(self):
        from jarvisrag.answer.verdicts import ANSWER_COMPLETE, ANSWER_PARTIAL
        self.assertEqual(self._verdict("PARTIAL")[0], ANSWER_PARTIAL)
        self.assertEqual(self._verdict(" answered.\n")[0], ANSWER_COMPLETE)

    def test_prompt_carries_question_and_answer(self):
        _, llm = self._verdict("PARTIAL")
        self.assertIn("What is the order number and delivery date?", llm.prompts[0])
        self.assertIn(self._PARAPHRASED, llm.prompts[0])

    def test_empty_or_dismissive_answer_needs_no_model_call(self):
        from jarvisrag.answer.verdicts import ANSWER_UNANSWERED
        for answer in ("", "I could not find that."):
            verdict, llm = self._verdict("ANSWERED", answer=answer)
            self.assertEqual(verdict, ANSWER_UNANSWERED)
            self.assertEqual(llm.prompts, [])

    def test_unavailable_model_falls_back_to_phrase_markers(self):
        from jarvisrag.answer.verdicts import ANSWER_COMPLETE, ANSWER_PARTIAL
        hedged = "The order number is BX-1042, but the evidence does not state a delivery date."
        self.assertEqual(self._verdict(None, answer=hedged, error=RuntimeError("down"))[0], ANSWER_PARTIAL)
        self.assertEqual(self._verdict("maybe", answer="The order number is BX-1042.")[0], ANSWER_COMPLETE)


class CascadeTriggerTests(unittest.TestCase):
    """handle_query cascades when the primary answer passed review but did not
    supply what was asked -- judged by meaning, not by hedge phrases."""

    _EVIDENCE = [
        "- receipt.pdf: Lamp purchased June 2 for $45 (receipt.pdf)",
        "- receipt.pdf: Shipped from the Ohio warehouse (receipt.pdf)",
        "- receipt.pdf: Return label follows in a separate message (receipt.pdf)",
    ]
    # Passes review, is too long for the dismissive guard, and uses no hedge marker.
    _PRIMARY = (
        "The lamp was purchased on June 2 for $45 and shipped from the Ohio warehouse. A return "
        "label is mentioned as arriving in a separate message, which isn't among these records, "
        "so the tracking number for the return isn't something I can see here."
    )

    def _run(self, completeness, cascade_result=None):
        import jarvisrag.answer.pipeline as ans
        runtime = ans.QueryRuntime(
            all_docs=[],
            llm=Mock(),
            doc_retriever=object(),
            domain_retrievers=[{"name": "mail", "retriever": object(), "match_any": ["email"]}],
        )
        cascade = Mock(return_value=cascade_result)
        judge = Mock(return_value=completeness)
        with _patch_answer("_retrieve_query_supported_evidence", return_value=(list(self._EVIDENCE), [])
        ), _patch_answer("fetch_mcp_evidence", return_value=[]), _patch_answer("_answer_with_review_loop", return_value=(self._PRIMARY, True, "ok")
        ), _patch_answer("_answer_completeness", judge), _patch_answer("_cascade_domain_answer", cascade
        ), _patch_answer("_clarifying_question_from_evidence", return_value=""
        ), contextlib.redirect_stdout(io.StringIO()):
            turn = ans.handle_query("What is the tracking number for the lamp return?", runtime)
        return turn, cascade, judge

    def test_paraphrased_partial_answer_cascades_and_adopts(self):
        import jarvisrag.answer.pipeline as ans
        from jarvisrag.answer import verdicts as answer_verdicts
        adopted = {
            "response_text": "The return tracking number is ZX-1.",
            "review_reason": "ok",
            "evidence_lines": ["- mail.emlx: tracking ZX-1 (mail.emlx)"],
            "refs": ["mail.emlx"],
            "superseded": set(),
            "domain_names": ["mail"],
            "completeness": ans.ANSWER_COMPLETE,
        }
        turn, cascade, judge = self._run(answer_verdicts.ANSWER_PARTIAL, adopted)
        cascade.assert_called_once()
        self.assertEqual([d["name"] for d in cascade.call_args.args[1]], ["mail"])
        self.assertEqual(turn["response_text"], "The return tracking number is ZX-1.")
        self.assertEqual(judge.call_count, 1)  # cascade's verdict is reused, not recomputed

    def test_complete_answer_never_cascades(self):
        import jarvisrag.answer.pipeline as ans
        turn, cascade, judge = self._run(ans.ANSWER_COMPLETE)
        cascade.assert_not_called()
        self.assertEqual(turn["response_text"], self._PRIMARY)
        self.assertEqual(judge.call_count, 1)

    def test_partial_answer_is_kept_when_cascade_finds_nothing(self):
        import jarvisrag.answer.pipeline as ans
        from jarvisrag.answer import verdicts as answer_verdicts
        turn, cascade, judge = self._run(answer_verdicts.ANSWER_PARTIAL, None)
        cascade.assert_called_once()
        self.assertEqual(turn["response_text"], self._PRIMARY)
        self.assertEqual(judge.call_count, 1)  # the clarification check reuses the verdict


class EvidenceTopicTests(unittest.TestCase):
    """Evidence that shares words with a question but is about something else is
    not presented as relevant: the question is answered from general knowledge."""

    _EVIDENCE = [
        "- policy.pdf, page 9: Policy endorsements: nuclear hazard; strikes, riots and civil commotions",
        "- resume.docx: Winner of the first annual robotics prize; led flight software development",
    ]

    def _run(self, review, on_topic, completeness=None):
        import jarvisrag.answer.pipeline as ans
        runtime = ans.QueryRuntime(all_docs=[], llm=Mock(), doc_retriever=object())
        topic = Mock(return_value=on_topic)
        with _patch_answer("_retrieve_query_supported_evidence", return_value=(list(self._EVIDENCE), [])
        ), _patch_answer("fetch_mcp_evidence", return_value=[]), _patch_answer("_answer_with_review_loop", return_value=review
        ), _patch_answer("_answer_completeness", return_value=completeness), _patch_answer("_evidence_concerns_question", topic
        ), _patch_answer("_clarifying_question_from_evidence", return_value=""), _patch_answer("_answer_from_general_knowledge", return_value="GENERAL: Marie Curie, 1903."
        ), _patch_answer("_build_topic_details_preview") as topic_preview, contextlib.redirect_stdout(
            io.StringIO()
        ) as out:
            turn = ans.handle_query("Who was the first woman to win a Nobel Prize?", runtime)
        return turn, topic, topic_preview, out.getvalue()

    def test_unrelated_evidence_gets_a_general_knowledge_answer(self):
        turn, topic, topic_preview, shown = self._run(("", False, "failed review"), on_topic=False)
        self.assertEqual(turn["response_text"], "GENERAL: Marie Curie, 1903.")
        self.assertEqual(turn["evidence_lines"], [])
        self.assertNotIn("policy.pdf", shown)
        topic.assert_called_once()
        topic_preview.assert_not_called()  # word matching must not re-surface the same text

    def test_related_evidence_is_still_previewed(self):
        turn, _, _, _ = self._run(("", False, "failed review"), on_topic=True)
        self.assertIn("Relevant evidence found", turn["response_text"])
        self.assertIn("policy.pdf", turn["response_text"])

    def test_passed_non_answer_over_unrelated_evidence_is_replaced(self):
        import jarvisrag.answer.pipeline as ans
        non_answer = "The provided evidence describes insurance endorsements and a resume, not Nobel laureates."
        turn, topic, _, _ = self._run((non_answer, True, "ok"), False, ans.ANSWER_UNANSWERED)
        self.assertEqual(turn["response_text"], "GENERAL: Marie Curie, 1903.")
        topic.assert_called_once()

    def test_answered_question_skips_the_check(self):
        import jarvisrag.answer.pipeline as ans
        turn, topic, _, _ = self._run(("Marie Curie (resume.docx).", True, "ok"), False, ans.ANSWER_COMPLETE)
        self.assertEqual(turn["response_text"], "Marie Curie (resume.docx).")
        topic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
