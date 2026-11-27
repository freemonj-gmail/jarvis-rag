"""The top-level query pipeline for Jarvis RAG.

``handle_query`` turns one user turn into a printed and spoken answer. It sits on
top of ``text.query`` (analysis), ``retrieval`` (evidence), and ``speech``
(display/TTS), and composes the answer modules: ``synthesis`` (draft, review,
refine), ``verdicts``, ``grounding``, ``fallbacks``, ``clarify``, ``domains``,
``evidence``, ``previews``, and ``display``. The runtime it uses is
``answer.runtime.QueryRuntime``.
"""

import os

from jarvisrag.answer.math import solve_math_query, suggest_math_correction
from jarvisrag.answer.runtime import QueryRuntime
from jarvisrag.config import debug_print
from jarvisrag.mcp.connector import fetch_mcp_evidence
from jarvisrag.retrieval import _evidence_lines_for_turn, _retrieve_query_supported_evidence
from jarvisrag.speech import _resolve_spoken_tokens, _SilentTTS, _spoken_response_text
from jarvisrag.text.query import (
    _effective_query_for_turn,
    _is_negative_evidence_feedback,
    _query_anchor_terms,
    _query_requests_authoring,
    _query_requests_corpus_details,
    _query_requests_source_details,
    _query_terms,
    _query_topic_terms,
)
from jarvisrag.answer.verdicts import (
    _answer_completeness,
    _evidence_concerns_question,
    _response_is_dismissive,
    _turn_continues_previous,
    ANSWER_COMPLETE,
    ANSWER_UNANSWERED,
)
from jarvisrag.answer.fallbacks import _answer_from_general_knowledge, _evidence_present_fallback
from jarvisrag.answer.synthesis import _answer_with_review_loop, _author_from_evidence
from jarvisrag.answer.clarify import _clarifying_question_from_evidence, _dead_end_response
from jarvisrag.answer.display import _format_response_for_display, _home_files_in_evidence
from jarvisrag.answer.previews import (
    _build_corpus_details_preview,
    _build_source_details_preview,
    _build_topic_details_preview,
)
from jarvisrag.answer.evidence import (
    _budget_combined_evidence,
    _DEFAULT_MAIN_INDEX_RESERVE,
    _DEFAULT_SYNTHESIS_EVIDENCE_BUDGET,
    _evidence_source_labels,
)
from jarvisrag.answer.domains import (
    _cascade_domain_answer,
    _corpus_relevance,
    _domains_for_turn,
    _retrieve_domain_evidence,
    _with_attachment_evidence,
)


# Hedging only signals an intent mismatch when there was plenty to work with; on
# a thin evidence set the answer is simply honest about a sparse corpus.
_HEDGE_MIN_EVIDENCE_LINES = 3


def _speak_answer(tts, llm_client, response_text: str, source_labels=()) -> None:
    """Hand the answer to the voice, worded on the speech thread: the context
    step (how abbreviations, codes and numbers are said here) needs a model
    call, which must not hold up the printed answer or the next prompt. A voice
    that is off never calls it, so no model time is spent."""
    spoken = _spoken_response_text(response_text, source_labels)
    if spoken:
        tts.speak(lambda: _resolve_spoken_tokens(spoken, llm_client))


def handle_query(
    user_query: str,
    runtime: QueryRuntime,
    prior_turn: dict[str, object] | None = None,
) -> dict[str, object]:
    """Process a single user query through the full RAG + fallback pipeline.

    Stateless: `prior_turn` supplies the previous conversational turn (or None /
    {} for a fresh start) and the new turn dict is returned, so callers (the REPL
    loop, smoke tests) can chain turns themselves. Uses the cached
    index/retriever/LLM carried on `runtime` — no re-initialization."""
    last_turn: dict[str, object] = dict(prior_turn) if prior_turn else {}
    tts = runtime.tts or _SilentTTS()
    # Set only when this turn ENDS by asking the user for a missing detail. The
    # next turn reads it to (a) link the reply back to this question no matter how
    # tersely it is phrased, and (b) refuse to ask a second time.
    awaiting_clarification = False

    print("\n[📄 Context Results]")
    try:
        effective_user_query, is_follow_up, source_hints = _effective_query_for_turn(
            user_query, last_turn, _turn_continues_previous(user_query, last_turn, getattr(runtime, "llm", None))
        )
        if is_follow_up:
            debug_print("[debug] Treated user input as a follow-up clarification.")
            debug_print(f"[debug] Effective query: {effective_user_query}")
        if source_hints:
            debug_print(f"[debug] Source hints: {source_hints}")
        negative_evidence_feedback = _is_negative_evidence_feedback(user_query)
        if negative_evidence_feedback:
            debug_print("[debug] User feedback rejected previous evidence; not merging prior evidence lines.")

        calculation = solve_math_query(user_query, last_turn)
        if calculation is not None:
            response_text = calculation.response_text
            debug_print(
                f"[debug] Used deterministic math path: "
                f"{calculation.category}/{calculation.operation}."
            )
            print(f"Jarvis: {response_text}\n")
            _speak_answer(tts, runtime.llm, response_text)
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
                "calculation": calculation.metadata(),
            }

        misspelling = suggest_math_correction(user_query, last_turn)
        if misspelling is not None:
            word, suggestion, corrected = misspelling
            response_text = (
                f'Did you mean "{suggestion}" instead of "{word}"? '
                f"Read that way: {corrected.response_text} "
                "If not, ask again and I will search your documents instead."
            )
            debug_print(f"[debug] Suggested math operator correction: {word} -> {suggestion}.")
            print(f"Jarvis: {response_text}\n")
            _speak_answer(tts, runtime.llm, response_text)
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
                "calculation": corrected.metadata(),
            }

        if _query_requests_corpus_details(user_query):
            response_text = _build_corpus_details_preview(runtime.all_docs)
            debug_print("[debug] Used direct corpus details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text, runtime.home_excluded_files)}\n")
            _speak_answer(tts, runtime.llm, response_text)
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if _query_requests_source_details(user_query):
            response_text = _build_source_details_preview(effective_user_query, runtime.all_docs, runtime.llm)
            debug_print("[debug] Used direct source details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text, runtime.home_excluded_files)}\n")
            _speak_answer(tts, runtime.llm, response_text)
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if runtime.response_mode == "chat":
            if runtime.doc_chat_engine is None:
                raise RuntimeError("No chat engine available")

            response = runtime.doc_chat_engine.chat(user_query)
            response_text = str(response).strip()
            debug_print("[debug] Used legacy chat response mode.")
            if _response_is_dismissive(response_text):
                if os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] Chat-mode response was dismissive; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, runtime.llm)
                    if gk_answer:
                        response_text = gk_answer
                    else:
                        response_text = (
                            "I could not find a clear answer in the indexed documents. "
                            "Please try rephrasing the question or asking for the relevant source file first."
                        )
                else:
                    response_text = (
                        "I could not find a clear answer in the indexed documents. "
                        "Please try rephrasing the question or asking for the relevant source file first."
                    )
            print(f"Jarvis: {_format_response_for_display(response_text, runtime.home_excluded_files)}\n")
            _speak_answer(tts, runtime.llm, response_text)
            return {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }

        if runtime.doc_retriever is None:
            raise RuntimeError("No retriever available")

        # A domain-routed query (e.g. federal benefits) is answered from the
        # domain + MCP sources, so suppress the home-corpus force-include: it
        # otherwise pins every launch-folder file into the evidence, bloating the
        # synthesis context and the References with unrelated local documents.
        # Consult gated domain corpora (e.g. federal-benefits PDFs) kept SEPARATE
        # from the main index. A keyword match routes explicitly. A follow-up that
        # drops the subject keyword STICKS to the previous turn's domain, but only
        # when that domain is MORE relevant to the query than the main index —
        # otherwise it is a real topic change and we fall back to normal routing.
        domains = getattr(runtime, "domains", None)
        prior_domain_names = last_turn.get("domain_names")
        active_domains, keyword_domains = _domains_for_turn(
            effective_user_query, domains, prior_domain_names
        )
        domain_routed = False
        routed_domain_names: list = []
        domain_evidence: list[str] = []
        domain_refs: list[str] = []
        domain_superseded: set[str] = set()
        if keyword_domains:
            domain_routed = True
            routed_domain_names = [d.name for d in keyword_domains]
            domain_evidence, domain_refs, domain_superseded = _retrieve_domain_evidence(
                effective_user_query, keyword_domains
            )
            domain_evidence = _with_attachment_evidence(effective_user_query, domain_evidence, keyword_domains)
        elif active_domains:
            domain_rel = max(
                (_corpus_relevance(effective_user_query, d.index) for d in active_domains),
                default=0.0,
            )
            main_rel = _corpus_relevance(effective_user_query, runtime.doc_index)
            debug_print(f"[debug] sticky arbitration: domain_rel={domain_rel:.4f} main_rel={main_rel:.4f}")
            if domain_rel > main_rel:
                domain_routed = True
                routed_domain_names = [d.name for d in active_domains]
                domain_evidence, domain_refs, domain_superseded = _retrieve_domain_evidence(
                    effective_user_query, active_domains
                )
                domain_evidence = _with_attachment_evidence(effective_user_query, domain_evidence, active_domains)
        main_evidence, source_nodes = _retrieve_query_supported_evidence(
            effective_user_query,
            runtime.doc_retriever,
            home_corpus_dir=(None if domain_routed else runtime.home_corpus_dir),
        )
        main_evidence = _evidence_lines_for_turn(
            main_evidence,
            last_turn,
            is_follow_up,
            negative_evidence_feedback,
            source_hints,
        )
        # Pull fresh authoritative/account intelligence from matching MCP
        # connectors (e.g. taxmcp.io for IRS questions, Truthify for the
        # financial institutions). Fully gated + fail-safe: a non-matching
        # question, a disabled layer, or any fetch failure yields [].
        # Literal full-text back-ends (e.g. eCFR) need salient keywords, not a
        # raw question, so pass a compact set reusing the existing query analysis.
        # On a clarification answer the prior turn's evidence -- including any MCP
        # lines -- is already carried forward by _evidence_lines_for_turn, so skip
        # the fresh MCP round-trip: it only adds latency and can TIME OUT on the
        # follow-up (observed: a 20s taxmcp timeout) for no new information.
        if is_follow_up and last_turn.get("awaiting_clarification"):
            debug_print("[debug] Clarification answer: skipping MCP re-fetch; prior evidence is reused.")
            mcp_evidence: list[str] = []
        else:
            mcp_terms = _query_terms(effective_user_query)
            mcp_anchors = _query_anchor_terms(effective_user_query, mcp_terms)
            mcp_keywords = " ".join((mcp_anchors + _query_topic_terms(mcp_terms, mcp_anchors))[:4])
            mcp_evidence = fetch_mcp_evidence(
                effective_user_query, runtime.mcp_registry, search_terms=mcp_keywords
            )
        # Combine the three tiers under ONE global ceiling. MCP is prepended so it
        # outranks the local corpus and the model's general knowledge; domain is
        # next; the general main index fills the remainder. The authoritative
        # tiers are never truncated -- this only bounds the diluting main-index
        # tail so the per-source caps cannot SUM into a ~70-line pile that buries
        # the answer and stalls synthesis.
        if domain_evidence:
            debug_print(f"[debug] Injected {len(domain_evidence)} domain-corpus evidence line(s).")
        if mcp_evidence:
            debug_print(f"[debug] Injected {len(mcp_evidence)} MCP evidence line(s) at highest precedence.")
        evidence_lines = _budget_combined_evidence(
            mcp_evidence,
            domain_evidence,
            main_evidence,
            budget=int(os.environ.get("JARVIS_SYNTHESIS_EVIDENCE_BUDGET", str(_DEFAULT_SYNTHESIS_EVIDENCE_BUDGET))),
            main_reserve=int(os.environ.get("JARVIS_MAIN_INDEX_RESERVE", str(_DEFAULT_MAIN_INDEX_RESERVE))),
        )
        debug_print(f"[debug] Retrieved source chunks: {len(source_nodes)}")
        debug_print(f"[debug] Query-supported evidence lines: {len(evidence_lines)}")
        for evidence_line in evidence_lines:
            debug_print(f"[debug] Evidence: {evidence_line}")

        # Launch-folder files actually present in this turn's evidence must be
        # representable in References even if the LLM cites them descriptively or
        # their filename contains spaces (which the prose citation regex mangles).
        # Domain-routed answers come from the domain/MCP sources, not the launch
        # folder, so they cite their OWN evidence instead (a staged mail filename
        # is sha1(Message-ID), which the model can never reproduce in prose).
        home_evidence_refs = domain_refs if domain_routed else _home_files_in_evidence(
            evidence_lines, runtime.home_indexed_files
        )
        excluded_for_display = None if domain_routed else runtime.home_excluded_files

        if _query_requests_authoring(user_query):
            response_text = _author_from_evidence(effective_user_query, evidence_lines, runtime.llm)
            if response_text:
                debug_print("[debug] Used authoring/composition path (single pass, no review loop).")
                print(f"Jarvis: {_format_response_for_display(response_text, excluded_for_display, home_evidence_refs, domain_superseded)}\n")
                _speak_answer(tts, runtime.llm, response_text)
                return {
                    "user_query": user_query,
                    "effective_query": effective_user_query,
                    "response_text": response_text,
                    "evidence_lines": evidence_lines,
                }
            debug_print("[debug] Authoring path returned empty; falling through to standard pipeline.")

        if evidence_lines:
            prior_answer_for_refinement = ""
            # A clarifying QUESTION is not a prior answer to refine; feeding
            # it back would have the model rework the question instead of
            # answering it now that the missing detail has arrived.
            if is_follow_up and not last_turn.get("awaiting_clarification"):
                prior_answer_for_refinement = str(last_turn.get("response_text") or "")
            response_text, review_passed, review_reason = _answer_with_review_loop(
                effective_user_query,
                evidence_lines,
                runtime.llm,
                prior_answer=prior_answer_for_refinement,
            )
        else:
            response_text = ""
            review_passed = False
            review_reason = "no query-supported evidence was retrieved"

        answer_was_dismissive = bool(response_text) and _response_is_dismissive(response_text)

        if review_passed and answer_was_dismissive:
            debug_print("[debug] Reviewed answer is dismissive; demoting to trigger fallback chain.")
            review_passed = False
            review_reason = review_reason or "reviewed answer indicated no information in evidence"
            response_text = ""

        # Failure-triggered domain cascade (G1): the primary answer (home/main +
        # any keyword-routed domain) did not fully answer -- either it failed
        # review / was demoted as dismissive, OR it passed review yet does not
        # supply everything the question asked for. Whether it did is judged on
        # MEANING (_answer_completeness), not on hedge phrases: a non-answer can
        # be worded endlessly ("is not provided", "isn't listed", ...) and a
        # phrase list missed most of them. Either way the consulted corpora lack
        # it, so broaden to domains NOT consulted this turn -- e.g. the large
        # personal mail index, kept behind explicit intent so it does not dilute
        # ordinary queries. A complete answer never cascades, and it stays
        # domain-agnostic: it consults whatever is un-consulted, keyed on answer
        # completeness, not any topic keyword. State is mutated ONLY on a
        # successful adoption, so a partial primary answer is preserved when the
        # cascade finds nothing.
        unconsulted = [
            d for d in (domains or []) if d.name not in routed_domain_names
        ]
        cascade_possible = bool(unconsulted) and not (
            is_follow_up and last_turn.get("awaiting_clarification")
        )
        # Computed once per answer and reused by the clarification check below.
        answer_completeness = ""
        if review_passed and cascade_possible:
            answer_completeness = _answer_completeness(effective_user_query, response_text, runtime.llm)
        primary_unanswered = (not review_passed) or answer_completeness != ANSWER_COMPLETE
        if primary_unanswered and cascade_possible:
            cascade = _cascade_domain_answer(
                effective_user_query, unconsulted, mcp_evidence, runtime.llm
            )
            if cascade:
                debug_print(
                    f"[debug] Domain cascade answered from {cascade['domain_names']}; adopting it."
                )
                response_text = cascade["response_text"]
                review_passed = True
                review_reason = cascade["review_reason"]
                evidence_lines = cascade["evidence_lines"]
                home_evidence_refs = cascade["refs"]
                excluded_for_display = None
                domain_superseded = cascade["superseded"]
                answer_completeness = cascade["completeness"]
                routed_domain_names = routed_domain_names + cascade["domain_names"]
                answer_was_dismissive = False

        # Retrieval admits evidence on shared words, so it can be about something
        # else entirely. Whenever that evidence did not yield an answer -- the
        # draft failed review, or passed while supplying none of what was asked --
        # judge by meaning whether it is about the question at all. If not, the
        # question is outside the indexed documents: answer from general
        # knowledge instead of presenting unrelated text as "relevant evidence".
        # An answered question never reaches the check, so it costs nothing there.
        evidence_off_topic = False
        if evidence_lines:
            if review_passed and not answer_completeness:
                answer_completeness = _answer_completeness(effective_user_query, response_text, runtime.llm)
            if (not review_passed or answer_completeness == ANSWER_UNANSWERED) and not _evidence_concerns_question(
                effective_user_query, evidence_lines, runtime.llm
            ):
                debug_print("[debug] Retrieved evidence is about something else; treating the question as uncovered.")
                evidence_off_topic = True
                review_passed = False
                review_reason = "retrieved evidence is about something else"
                evidence_lines = []
                home_evidence_refs = []
                answer_was_dismissive = False

        if not review_passed:
            debug_print(f"[debug] Initial answer did not pass review: {review_reason}")
            # Evidence reaching this branch was judged to be about the question
            # (see the topic check above). Surface it rather than routing to a
            # general-knowledge answer that would wrongly deny the corpus. This
            # holds even when the drafted LLM answer was dismissive because the
            # chunks don't literally spell out the asked-for detail — showing what
            # WAS found is more honest and trustworthy than pretending indexed
            # files don't exist. No explicit filename mention is required to reach
            # this path.
            if evidence_lines:
                if answer_was_dismissive:
                    debug_print(
                        "[debug] Drafted answer was dismissive but on-topic evidence exists; "
                        "showing the evidence preview instead of a general-knowledge denial."
                    )
                else:
                    debug_print("[debug] Returning evidence-present fallback instead of no-evidence response.")
                # Before dumping an evidence preview, check whether ONE detail only
                # the user can supply (a year, which account/plan, a jurisdiction)
                # would make this answerable. Asking beats a wall of text. Capped at
                # one round per chain: if the user already answered a clarification
                # and it still failed, show the evidence instead of asking again.
                clarifying = (
                    ""
                    if last_turn.get("awaiting_clarification")
                    else _clarifying_question_from_evidence(
                        effective_user_query, evidence_lines, runtime.llm
                    )
                )
                if clarifying:
                    debug_print("[debug] Asking the user one clarifying question.")
                    awaiting_clarification = True
                    response_text = (
                        "I have relevant source material, but one detail would let me answer it "
                        f"precisely:\n\n{clarifying}"
                    )
                else:
                    response_text = _evidence_present_fallback(evidence_lines, review_reason)
            else:
                # Evidence already judged unrelated must not be re-surfaced by the
                # word-overlap topic preview, which matches the same way retrieval did.
                topic_preview = (
                    ""
                    if evidence_off_topic
                    else _build_topic_details_preview(effective_user_query, runtime.all_docs, runtime.llm)
                )
                if topic_preview and not _response_is_dismissive(topic_preview):
                    debug_print("[debug] Applied direct topic corpus fallback.")
                    response_text = topic_preview
                elif os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] No corpus evidence found; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, runtime.llm)
                    response_text = gk_answer or _dead_end_response(
                        effective_user_query, runtime.llm, last_turn, review_reason
                    )
                    awaiting_clarification = not gk_answer and response_text.endswith("?")
                else:
                    response_text = _dead_end_response(
                        effective_user_query, runtime.llm, last_turn, review_reason
                    )
                    awaiting_clarification = response_text.endswith("?")
        else:
            debug_print("[debug] Returning reviewed answer.")
            # An answer that PASSED review while still not supplying everything
            # asked for is the signature of an intent mismatch: the question was
            # answered literally, but the retrieved material is about a related
            # frame the user actually meant. The reviewer cannot catch this --
            # the answer is faithful to the evidence, just to the wrong reading.
            # Keep the answer in full and APPEND one disambiguating question, so
            # nothing is lost and the next turn can resolve it. Uses the same
            # meaning-based completeness verdict as the cascade, not hedge phrases.
            if (
                not last_turn.get("awaiting_clarification")
                and len(evidence_lines) >= _HEDGE_MIN_EVIDENCE_LINES
                and (
                    answer_completeness
                    or _answer_completeness(effective_user_query, response_text, runtime.llm)
                ) != ANSWER_COMPLETE
            ):
                clarifying = _clarifying_question_from_evidence(
                    effective_user_query, evidence_lines, runtime.llm
                )
                if clarifying:
                    debug_print("[debug] Answer hedged despite strong evidence; appending a clarifying question.")
                    awaiting_clarification = True
                    response_text = (
                        f"{response_text}\n\n"
                        f"One thing would let me answer this precisely: {clarifying}"
                    )

        print(f"Jarvis: {_format_response_for_display(response_text, excluded_for_display, home_evidence_refs, domain_superseded)}\n")
        _speak_answer(tts, runtime.llm, response_text, _evidence_source_labels(evidence_lines))
        last_turn = {
            "user_query": user_query,
            "effective_query": effective_user_query,
            "response_text": response_text,
            "evidence_lines": evidence_lines,
            "domain_names": routed_domain_names,
            "awaiting_clarification": awaiting_clarification,
        }
    except Exception as e:
        print("Context engine failed:", e)
        tts.speak("Sorry, I encountered an error processing your request.")
    return last_turn
