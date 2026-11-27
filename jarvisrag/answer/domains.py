"""Domain corpora: which to consult, their evidence and references, email
attachments, and the cascade to corpora not consulted on the first pass.
"""

import hashlib
import os
import re

from jarvisrag.config import debug_print
from jarvisrag.iotools import Spinner
from jarvisrag.mcp.registry import routing_term_matches
from jarvisrag.retrieval import (
    _query_evidence_is_supported,
    _query_focused_snippet,
    _retrieve_query_supported_evidence,
)
from jarvisrag.text.query import _query_anchor_terms, _query_requests_values, _query_terms
from jarvisrag.text.textproc import is_low_information_text
from jarvisrag.answer.verdicts import _answer_completeness, ANSWER_COMPLETE, ANSWER_UNANSWERED
from jarvisrag.answer.synthesis import _answer_with_review_loop
from jarvisrag.answer.evidence import (
    _budget_combined_evidence,
    _DEFAULT_MAIN_INDEX_RESERVE,
    _DEFAULT_SYNTHESIS_EVIDENCE_BUDGET,
)


# ---------------------------------------------------------------------------
# Query handling pipeline
# ---------------------------------------------------------------------------


# Domain evidence is capped at the SAME budget the retriever already applies to
# the main index (JARVIS_MAX_EVIDENCE_LINES). A tighter, separate cap here (was
# 15) silently discarded up to HALF the ranked lines the domain retriever had
# already selected and deduped -- and for a cross-document answer the pieces that
# rank lower are exactly a value phrased in different vocabulary than the query
# (e.g. a $-figure in a "402(g) elective deferral" chunk vs a Roth/IRA question),
# so the tighter cap cut precisely the evidence needed to link the answer.
_DEFAULT_DOMAIN_EVIDENCE_LINES = 30


def _domains_for_turn(
    query: str, domain_retrievers: list | None, prior_domain_names: list | None = None
) -> tuple[list, list]:
    """Select the domain corpora to consult this turn.

    Keyword matches take priority (an explicit on-topic query, or a topic switch).
    When nothing matches, fall back to STICKY routing: the domain(s) the previous
    turn used, so a follow-up that drops the subject keyword still reaches the
    right corpus. The sticky domain's own relevance gate then decides whether it
    actually applies. Returns (active_domains, keyword_matched_domains)."""
    if not domain_retrievers:
        return [], []
    keyword_matched = [
        d for d in domain_retrievers
        if any(routing_term_matches(str(term), query) for term in (d.get("match_any") or []) if term)
    ]
    if keyword_matched:
        return keyword_matched, keyword_matched
    if prior_domain_names:
        prior = set(prior_domain_names)
        return [d for d in domain_retrievers if d.get("name") in prior], []
    return [], []


def _corpus_relevance(query: str, index: object) -> float:
    """Top dense cosine similarity of ``query`` against a corpus's vector index.

    Used to arbitrate sticky domain routing. The domain and main indexes share
    the same embedding model, so their dense similarity scores are directly
    comparable, and a keyword-less follow-up is routed to whichever corpus fits
    it better. Fail-safe: any error yields 0.0."""
    if index is None:
        return 0.0
    try:
        nodes = index.as_retriever(similarity_top_k=3).retrieve(query)
    except Exception:  # noqa: BLE001 - arbitration must never break a turn
        return 0.0
    return max((float(getattr(n, "score", 0.0) or 0.0) for n in nodes), default=0.0)


_DOMAIN_EVIDENCE_LABEL_RE = re.compile(r"^-\s*(.+?):\s")


_EVIDENCE_PAGE_SUFFIX_RE = re.compile(r",\s*page\s*\d+\s*$", re.IGNORECASE)


def _email_reference_line(metadata: dict) -> str:
    """Cite a message the way a person can actually find it again.

    A staged message's filename is sha1(Message-ID), which identifies nothing to
    a reader, so ANY message metadata is preferred over it: Subject when present,
    otherwise sender and/or date, otherwise a bare marker. Returns "" only when
    the node carries no message metadata at all — i.e. it is not a message, and
    its own filename is the meaningful citation.
    """
    if not any(str(key).startswith("email_") for key in metadata):
        return ""
    subject = str(metadata.get("email_subject") or "").strip()
    sender = str(metadata.get("email_from") or "").strip()
    date = str(metadata.get("email_date") or "").strip()
    reference = f'Email: "{subject}"' if subject else "Email (no subject)"
    if sender:
        reference += f" from {sender}"
    if date:
        reference += f" ({date})"
    return reference


def _domain_reference_lines(
    evidence_lines: list[str], source_nodes: list | None
) -> tuple[list[str], set[str]]:
    """Guaranteed References for the domain evidence actually sent to the model.

    A domain-routed answer skips the launch-folder reference injection, so without
    this the References fall back to whatever filename the model types in prose —
    and a staged mail filename is sha1(Message-ID), which it will never reproduce.
    Source choice is driven by the node's own metadata, not by corpus name: a
    message cites Subject/From/Date, anything else keeps its file label. Returns
    the references plus the filenames they REPLACE, so the opaque original is not
    also listed when the model happens to name it in prose."""
    if not evidence_lines:
        return [], set()
    by_name: dict[str, dict] = {}
    for node in source_nodes or []:
        metadata = getattr(getattr(node, "node", node), "metadata", None) or {}
        name = os.path.basename(str(metadata.get("file_name") or metadata.get("file_path") or ""))
        if name and name not in by_name:
            by_name[name] = metadata
    references: list[str] = []
    superseded: set[str] = set()
    for line in evidence_lines:
        match = _DOMAIN_EVIDENCE_LABEL_RE.match(line)
        if not match:
            continue
        label = match.group(1).strip()
        name = _EVIDENCE_PAGE_SUFFIX_RE.sub("", label).strip()
        reference = _email_reference_line(by_name.get(name) or {})
        if reference:
            superseded.add(name.lower())
        else:
            reference = label
        if reference not in references:
            references.append(reference)
    return references, superseded


def _retrieve_domain_evidence(
    query: str, domains: list | None
) -> tuple[list[str], list[str], set[str]]:
    """Retrieve evidence from the already-selected domain corpora.

    Additive + fail-safe: no domains or any error yields []. Domain evidence
    carries its own source filenames (e.g. the OPM/SSA PDFs) so it is citeable
    like local evidence, and is merged at high precedence. Each domain index is
    SEPARATE from the main index, and its own relevance gate decides whether it
    contributes — so a sticky (prior-topic) domain that no longer applies simply
    returns nothing and the turn falls back to normal retrieval. Returns the
    evidence lines, the References they must be citeable by, and the filenames
    those references replace."""
    if not domains:
        return [], [], set()
    cap = max(1, int(os.environ.get("JARVIS_MAX_EVIDENCE_LINES", str(_DEFAULT_DOMAIN_EVIDENCE_LINES))))
    per_domain: list[list[str]] = []
    nodes: list = []
    for domain in domains:
        try:
            evidence, source_nodes = _retrieve_query_supported_evidence(query, domain["retriever"])
            per_domain.append(list(evidence))
            nodes.extend(source_nodes or [])
        except Exception as exc:  # noqa: BLE001 - never let a domain lookup break a turn
            debug_print(f"[domain] {domain.get('name')} retrieval failed ({exc}); skipping.")
    # Interleave the domains' ranked lines (A1, B1, A2, B2, ...) under the one cap.
    # Filling the cap in domain order let the first domain's lines starve every
    # later domain, which was then never even searched. One domain = its own order.
    delivered: list[str] = []
    for rank in range(max((len(evidence) for evidence in per_domain), default=0)):
        for evidence in per_domain:
            if rank < len(evidence) and len(delivered) < cap:
                delivered.append(evidence[rank])
        if len(delivered) >= cap:
            break
    references, superseded = _domain_reference_lines(delivered, nodes)
    return delivered, references, superseded


# Per-turn bounds on runtime attachment reading: evidence emails opened,
# attachments read, and the largest file extracted.
_ATTACHMENT_MAX_EMAILS = 3


_ATTACHMENT_MAX_FILES = 5


_ATTACHMENT_MAX_BYTES = 25_000_000


def _evidence_email_paths(evidence_lines: list[str], domains: list) -> list[tuple[str, str]]:
    """(evidence label, staged path) for each distinct email in evidence order.

    Evidence is already relevance-ranked, so the first emails are the likeliest
    carriers of an answer that sits in an attachment. Only labels that resolve to
    a file inside one of ``domains``' corpus directories count, so launch-folder
    and reference-corpus lines are ignored."""
    from jarvisrag.extraction.extractors import _EMAIL_EXTENSIONS

    corpus_dirs = [d.get("corpus_dir") for d in domains or [] if d.get("corpus_dir")]
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in evidence_lines:
        match = _DOMAIN_EVIDENCE_LABEL_RE.match(str(line))
        if not match:
            continue
        label = _EVIDENCE_PAGE_SUFFIX_RE.sub("", match.group(1).strip()).strip()
        if label in seen or not label.lower().endswith(_EMAIL_EXTENSIONS):
            continue
        seen.add(label)
        for corpus_dir in corpus_dirs:
            path = os.path.join(corpus_dir, label)
            if os.path.isfile(path):
                found.append((label, path))
                break
    return found


def _attachment_evidence_lines(query: str, evidence_lines: list[str], domains: list) -> list[str]:
    """Dereference the attachments of the top evidence emails into evidence lines.

    The runtime half of the attachment pointer design: attachment content is
    NEVER embedded in the vector index. For the emails an answer is being built
    from, the attachments of the first few are opened, extracted with the same
    readers and extraction cache as the rest of the corpus, trimmed to the
    query-relevant passage, and labelled with their parent email so they cite as
    that email. A passage is kept only if it passes the same query-support gate
    retrieval applies to every chunk, so an unrelated attachment never dilutes
    the evidence. Bounded per turn by email count, file count, and file size."""
    from jarvisrag.answer.bootstrap import EXTRACTION_CACHE_DIR, file_extractor
    from jarvisrag.extraction.extractors import (
        _EMAIL_EXTENSIONS,
        load_documents_with_cache,
        staged_email_attachments,
    )

    snippet_chars = int(os.environ.get("JARVIS_QUERY_SNIPPET_CHARS", "6000"))
    terms = _query_terms(query)
    anchors = _query_anchor_terms(query, terms)
    require_value = _query_requests_values(query)

    selected: list[tuple[str, str, str, str]] = []
    # The same file is often attached to several emails (a forwarded label, a
    # resent invoice). Read each distinct content once, from its highest-ranked
    # email, so a duplicate neither repeats evidence nor uses up a slot.
    seen_content: set[str] = set()
    emails_used = 0
    for label, staged in _evidence_email_paths(evidence_lines, domains):
        if emails_used >= _ATTACHMENT_MAX_EMAILS or len(selected) >= _ATTACHMENT_MAX_FILES:
            break
        readable = []
        for name, path in staged_email_attachments(staged):
            ext = os.path.splitext(path)[1].lower()
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if ext not in file_extractor or ext in _EMAIL_EXTENSIONS or size > _ATTACHMENT_MAX_BYTES:
                continue
            try:
                with open(path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                continue
            if digest in seen_content:
                continue
            seen_content.add(digest)
            readable.append((label, name, path, ext))
        if readable:
            emails_used += 1
            selected.extend(readable[: _ATTACHMENT_MAX_FILES - len(selected)])
    if not selected:
        return []

    lines: list[str] = []
    with Spinner("Reading attachments"):
        for label, name, path, ext in selected:
            try:
                docs = load_documents_with_cache(
                    input_dir=os.path.dirname(path),
                    required_exts=[ext],
                    file_extractor=file_extractor,
                    cache_dir=EXTRACTION_CACHE_DIR,
                    recursive=False,
                )
            except Exception as exc:  # noqa: BLE001 - one bad attachment never breaks a turn
                debug_print(f"[debug] Attachment extraction failed ({type(exc).__name__}); skipping.")
                continue
            target = os.path.normcase(os.path.abspath(path))
            text = "\n".join(
                str(doc.text or "")
                for doc in docs
                if os.path.normcase(os.path.abspath(str(doc.metadata.get("file_path") or ""))) == target
            ).strip()
            if not text or is_low_information_text(text):
                continue
            snippet = _query_focused_snippet(text, terms, anchors, require_value, max_chars=snippet_chars)
            if _query_evidence_is_supported(snippet, terms, anchors, require_value):
                lines.append(f"- {label}: [Attachment: {name}] {snippet}")
    debug_print(f"[debug] Attachments: read {len(selected)}, kept {len(lines)} query-supported passage(s).")
    return lines


def _with_attachment_evidence(query: str, domain_lines: list[str], domains: list) -> list[str]:
    """Domain evidence with the top emails' relevant attachment passages added.

    Runs BEFORE synthesis, not as a retry triggered by the answer's wording: a
    retry keyed on hedge phrases missed an answer that said a detail "is not
    provided", so the attachment holding it was never opened. Adding the
    query-supported attachment passages up front keeps it to ONE synthesis. The
    passages lead because they were opened specifically for this question. A
    no-op when no evidence email has a readable, relevant attachment."""
    if not domain_lines:
        return domain_lines
    attachment_lines = _attachment_evidence_lines(query, domain_lines, domains)
    return attachment_lines + list(domain_lines) if attachment_lines else domain_lines


def _cascade_domain_answer(
    query: str,
    unconsulted_domains: list,
    mcp_evidence: list[str],
    llm,
) -> dict | None:
    """Second-pass answer over domain corpora NOT consulted on the first pass.

    Runs ONLY when the primary (home/main + keyword-routed) answer did not fully
    answer: it broadens the search to the remaining domains -- e.g. the
    large personal mail index, which is gated behind explicit intent so it never
    dilutes an ordinary query -- and re-synthesizes. DOMAIN-AGNOSTIC and keyed on
    answer WEAKNESS, not any topic keyword: it consults whatever was left
    un-consulted this turn. Fail-safe: no domains, no supported evidence, or a
    failed re-synthesis / one that supplies none of what was asked yields None,
    so the caller simply proceeds to its normal fallback chain.

    The re-synthesis deliberately uses ONLY the un-consulted domain evidence
    (plus any authoritative MCP lines) and NOT the home/main evidence: that
    evidence already failed to answer on the first pass, and re-mixing it drags
    the second pass back into the same wrong framing (measured: a confusable
    same-entity home chunk -- e.g. a product PURCHASE receipt -- caused the mail
    RETURN answer, correct in isolation, to come back dismissive).

    Each un-consulted domain is likewise answered IN ISOLATION rather than pooled:
    pooled retrieval fills one shared evidence cap in domain order, so an
    unrelated corpus that happens to come first can occupy every slot and the
    relevant corpus is never retrieved at all (measured: 30/30 lines from an
    unrelated reference corpus, 0 from the mail index that held the answer at
    rank 0). Domains are tried in descending query relevance so the likeliest
    source is synthesized first; relevance only ORDERS the attempts and never
    excludes a domain. A domain whose answer is complete wins at once; otherwise
    the first answer that supplies at least part of what was asked is kept, and
    an answer that supplies none of it is never adopted, however it is worded.

    Returns a dict (response_text, review_reason, evidence_lines, refs,
    superseded, domain_names, completeness) on a supported answer that supplies
    at least part of what was asked; otherwise None.
    """
    if not unconsulted_domains:
        return None
    ordered = sorted(
        unconsulted_domains,
        key=lambda d: _corpus_relevance(query, d.get("index")),
        reverse=True,
    )
    first_partial = None
    for domain in ordered:
        casc_evidence, casc_refs, casc_superseded = _retrieve_domain_evidence(query, [domain])
        if not casc_evidence:
            continue
        casc_evidence = _with_attachment_evidence(query, casc_evidence, [domain])
        combined = _budget_combined_evidence(
            mcp_evidence,
            casc_evidence,
            [],
            budget=int(os.environ.get("JARVIS_SYNTHESIS_EVIDENCE_BUDGET", str(_DEFAULT_SYNTHESIS_EVIDENCE_BUDGET))),
            main_reserve=int(os.environ.get("JARVIS_MAIN_INDEX_RESERVE", str(_DEFAULT_MAIN_INDEX_RESERVE))),
        )
        text, passed, reason = _answer_with_review_loop(query, combined, llm)
        completeness = _answer_completeness(query, text, llm) if passed else ANSWER_UNANSWERED
        if completeness != ANSWER_UNANSWERED:
            result = {
                "response_text": text,
                "review_reason": reason,
                "evidence_lines": combined,
                "refs": casc_refs,
                "superseded": casc_superseded,
                "domain_names": [domain.get("name")],
                "completeness": completeness,
            }
            if completeness == ANSWER_COMPLETE:
                return result
            if first_partial is None:
                first_partial = result
            debug_print(f"[debug] Domain cascade: '{domain.get('name')}' answered in part; trying next.")
            continue
        debug_print(f"[debug] Domain cascade: '{domain.get('name')}' did not answer; trying next.")
    return first_partial
