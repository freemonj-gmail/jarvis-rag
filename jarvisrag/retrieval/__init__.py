"""Retrieval layer for Jarvis RAG.

Owns hybrid (BM25 + dense) retriever construction and the query-supported
evidence pipeline: turning a user query into ranked, topic-covered evidence
lines via multi-variant retrieval, source-hint force-include, and per-node
scoring/snippeting. Sits between the pure query-analysis layer
(``jarvis_query``) and the answer-synthesis layer (``jarvis_answer``).
"""

import os
import re
from typing import Any, Optional, cast

from jarvisrag.config import debug_print
from jarvisrag.text.textproc import is_low_information_text, normalize_text
from jarvisrag.text.query import (
    _detail_candidates,
    _evidence_body_from_evidence_line,
    _extract_source_hints,
    _payment_context_score,
    _prioritize_source_hinted_evidence,
    _query_anchor_terms,
    _query_requests_enumeration,
    _query_requests_values,
    _query_terms,
    _query_topic_coverage,
    _query_topic_terms,
    _retrieval_query_variants,
    _specific_query_terms,
    _term_family_hit,
    _value_numbers,
)

# Reciprocal-rank-fusion damping constant for aggregating a node's relevance
# across the multiple retrieval-query paraphrases (standard RRF k=60).
_RRF_K = 60


def _node_identity(node: object) -> object:
    """Stable identity for a retrieved node, used to aggregate its relevance
    across query variants and to look that relevance back up when ranking
    evidence. Mirrors the dedup key used during retrieval."""
    return (
        getattr(node, "node_id", None)
        or getattr(getattr(node, "node", None), "node_id", None)
        or id(node)
    )


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _line_tokens(evidence_line: str) -> set[str]:
    body = _evidence_body_from_evidence_line(evidence_line)
    return {match.group(0).casefold() for match in _WORD_RE.finditer(body)}


def _near_duplicate_of(tokens: set[str], signatures: list[set[str]], threshold: float) -> bool:
    """True when these tokens repeat evidence already selected (Jaccard >= threshold)."""
    for prior in signatures:
        union = tokens | prior
        if union and len(tokens & prior) / len(union) >= threshold:
            return True
    return False


def _extraction_quality(metadata: dict) -> float:
    """Extraction cleanliness in [0.0, 1.0], used ONLY as a ranking tiebreaker.

    Native / structured extractions (born-digital PDF text, docx via docling)
    record no OCR uncertainty and are treated as fully clean (1.0). OCR-derived
    text carries an ``ocr_confidence`` reflecting recognition certainty; that
    value is used directly. This is never a primary ranking signal — it only
    separates candidates that already tie on retriever relevance and lexical
    grounding score, gently preferring a cleaner extraction over a noisier one.
    """
    conf = (metadata or {}).get("ocr_confidence")
    if conf is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        return 1.0


def _source_nodes_to_evidence_lines(source_nodes: list[object], max_items: int = 8) -> list[str]:
    evidence_lines = []
    for node in source_nodes[:max_items]:
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text or is_low_information_text(node_text):
            continue

        metadata = getattr(node, "metadata", {}) or {}
        file_name = os.path.basename(
            str(metadata.get("file_name") or metadata.get("file_path") or "unknown file")
        )
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"

        compact = " ".join(normalize_text(node_text).split())
        evidence_lines.append(f"- {label}: {compact[:4000]}")
    return evidence_lines


def _source_nodes_to_query_evidence_lines(
    user_query: str,
    source_nodes: list[object],
    max_items: int = 8,
    source_hints: list[str] | None = None,
    node_relevance: dict[object, float] | None = None,
    home_corpus_dir: str | None = None,
) -> list[str]:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    require_value = _query_requests_values(user_query)
    enumerate_full = _query_requests_enumeration(user_query)
    enum_max_chars = int(os.environ.get("JARVIS_ENUM_CHUNK_CHARS", "8000"))
    hint_basenames = {os.path.basename(h).lower() for h in (source_hints or []) if h}
    node_relevance = node_relevance or {}
    home_root = (os.path.abspath(home_corpus_dir).lower() + os.sep) if home_corpus_dir else None

    # Each candidate carries the hybrid retriever's aggregated cross-variant
    # relevance (RRF) as the PRIMARY ranking signal; the lexical score is kept
    # only as a grounding gate (score > 0) and tiebreaker. The retriever already
    # fuses BM25 + dense semantics, so trusting its relevance prevents the lexical
    # heuristic from burying a concise on-topic section under verbose, tangential
    # chunks that merely repeat the query words. Extraction quality (OCR
    # confidence) is folded in only as a final tiebreaker so a cleaner extraction
    # wins when candidates are otherwise equally relevant and grounded.
    ranked: list[tuple[float, int, float, set[str], object, str]] = []
    forced: list[str] = []  # source-hint matches bypass the score filter
    # Launch-folder ("home") chunks kept as gate-relaxed candidates so a small
    # provided folder is never fully crowded out of the evidence by a larger
    # accumulated cross-directory corpus (see the home-corpus floor below).
    home_candidates: list[tuple[float, int, float, object, str]] = []
    home_node_count = 0
    total_node_count = 0
    selected_ids: set[object] = set()
    for node in source_nodes:
        try:
            node_text = (node.get_content() or "").strip()
        except Exception:
            node_text = ""
        if not node_text or is_low_information_text(node_text):
            continue

        metadata = getattr(node, "metadata", {}) or {}
        node_fname = str(metadata.get("file_name") or "").lower()
        node_fpath = str(metadata.get("file_path") or "").lower()
        is_source_hinted = bool(hint_basenames) and any(
            h in node_fname or h in node_fpath for h in hint_basenames
        )

        search_text = node_text
        for key in ("file_name", "file_path", "parent_file", "source"):
            value = metadata.get(key)
            if value:
                search_text += f"\n{value}"

        file_name = os.path.basename(
            str(metadata.get("file_name") or metadata.get("file_path") or "unknown file")
        )
        page_number = metadata.get("page_number")
        label = file_name
        if page_number is not None:
            label += f", page {page_number}"

        if is_source_hinted:
            # Bypass score filter for user-named files; send the full chunk text
            # (capped at JARVIS_SOURCE_HINT_CHUNK_CHARS, default 4000) so employer
            # names, dates, and lists aren't severed mid-word.
            max_chars = int(os.environ.get("JARVIS_SOURCE_HINT_CHUNK_CHARS", "16000"))
            snippet = " ".join(node_text.split())[:max_chars].rstrip()
            if len(node_text) > max_chars:
                snippet += "..."
            forced.append(f"- {label}: {snippet}")
            continue

        score = _query_candidate_score(search_text, terms, anchor_terms, require_value)
        relevance = node_relevance.get(_node_identity(node), 0.0)
        identity = _node_identity(node)
        quality = _extraction_quality(metadata)
        total_node_count += 1

        is_home = bool(home_root) and bool(node_fpath) and node_fpath.startswith(home_root)
        if is_home:
            home_node_count += 1
            home_max = int(os.environ.get("JARVIS_HOME_CHUNK_CHARS", "8000"))
            home_snippet = " ".join(node_text.split())[:home_max].rstrip()
            if len(node_text) > home_max:
                home_snippet += "..."
            home_candidates.append((relevance, score, quality, identity, f"- {label}: {home_snippet}"))

        if score <= 0:
            continue

        if enumerate_full:
            # Enumeration query: send the full chunk (capped) so older/lower-density
            # entries (e.g. the oldest jobs at the bottom of a resume) aren't severed
            # by the query-centered focused window. score > 0 already gated relevance,
            # so the per-snippet support check is skipped here.
            snippet = " ".join(node_text.split())[:enum_max_chars].rstrip()
            if len(node_text) > enum_max_chars:
                snippet += "..."
        else:
            snippet = _query_focused_snippet(node_text, terms, anchor_terms, require_value)
            if not _query_evidence_is_supported(snippet, terms, anchor_terms, require_value):
                continue

        coverage = _query_topic_coverage(snippet, terms, anchor_terms)
        ranked.append((relevance, score, quality, coverage, identity, f"- {label}: {snippet}"))

    # Primary sort: retriever relevance (RRF). Secondary: lexical grounding score.
    # Tertiary: extraction quality (cleaner OCR breaks ties).
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = list(forced)  # source-hinted evidence first

    # Near-duplicate suppression. This corpus holds many near-identical document
    # versions, so the same passage can occupy several evidence slots and crowd
    # out distinct material. Explicitly requested (source-hinted) lines are kept
    # and only serve as comparison points.
    suppress_duplicates = True  # always on: near-duplicate crowding hurts recall
    duplicate_threshold = float(os.environ.get("JARVIS_DUPLICATE_EVIDENCE_THRESHOLD", "0.9"))
    signatures = [_line_tokens(line) for line in selected] if suppress_duplicates else []

    def accept(line: str) -> bool:
        """False when the line repeats selected evidence; records it when accepted."""
        if not suppress_duplicates:
            return True
        tokens = _line_tokens(line)
        if _near_duplicate_of(tokens, signatures, duplicate_threshold):
            return False
        signatures.append(tokens)
        return True

    # Home-corpus floor: when the launch folder is a MINORITY of the indexed
    # collection (other directories' vectors dominate), reserve a bounded share
    # of slots for the provided folder's own chunks so it is always represented.
    # Skipped when there is no home dir or it is already the majority (the normal
    # single-corpus case), leaving default ranking untouched.
    home_share = (home_node_count / total_node_count) if total_node_count else 0.0
    if home_candidates and 0.0 < home_share < 0.5:
        home_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        reserve = min(len(home_candidates), max(4, max_items // 3))
        for _, _, _, identity, line in home_candidates[:reserve]:
            if identity in selected_ids or line in selected:
                continue
            if not accept(line):
                continue
            selected.append(line)
            selected_ids.add(identity)
            if len(selected) >= max_items:
                break

    if enumerate_full:
        # Aggregation queries must list EVERY matching item, so skip the
        # topic-coverage dedup (which drops same-topic chunks such as additional
        # resume entries) and take the highest-ranked full chunks directly.
        for _, _, _, _, identity, line in ranked:
            if identity in selected_ids or line in selected:
                continue
            if not accept(line):
                continue
            selected.append(line)
            selected_ids.add(identity)
            if len(selected) >= max_items:
                break
        return selected

    covered_topics: set[str] = set()

    for _, _, _, coverage, identity, line in ranked:
        if identity in selected_ids or line in selected:
            continue
        if coverage - covered_topics:
            if not accept(line):
                continue
            selected.append(line)
            selected_ids.add(identity)
            covered_topics.update(coverage)
            if len(selected) >= max_items:
                return selected

    for _, _, _, _, identity, line in ranked:
        if identity in selected_ids or line in selected:
            continue
        if not accept(line):
            continue
        selected.append(line)
        selected_ids.add(identity)
        if len(selected) >= max_items:
            break

    return selected


def _evidence_lines_for_turn(
    evidence_lines: list[str],
    last_turn: dict[str, object],
    is_follow_up: bool,
    negative_evidence_feedback: bool,
    source_hints: list[str],
) -> list[str]:
    merged_evidence_lines = list(evidence_lines)
    if is_follow_up and not negative_evidence_feedback:
        prior_evidence_lines = [
            line
            for line in cast(list[object], last_turn.get("evidence_lines", []))
            if isinstance(line, str)
        ]
        for line in prior_evidence_lines:
            if line not in merged_evidence_lines:
                merged_evidence_lines.append(line)
    return _prioritize_source_hinted_evidence(merged_evidence_lines, source_hints)


def _retrieve_query_supported_evidence(
    user_query: str,
    retriever: object,
    home_corpus_dir: str | None = None,
) -> tuple[list[str], list[object]]:
    max_queries = 20  # fixed ceiling on query paraphrases; the generator yields far fewer
    max_evidence = max(1, int(os.environ.get("JARVIS_MAX_EVIDENCE_LINES", "30")))
    all_nodes = []
    seen_node_ids = set()
    # Aggregate each node's relevance across all query paraphrases with RRF so a
    # chunk consistently retrieved high by the hybrid (BM25 + dense) retriever
    # outranks one that merely repeats the query words. This is the primary
    # evidence-ordering signal (see _source_nodes_to_query_evidence_lines).
    node_relevance: dict[object, float] = {}

    for attempt, retrieval_query in enumerate(_retrieval_query_variants(user_query)[:max_queries], start=1):
        debug_print(f"[debug] Retrieval attempt {attempt}: {retrieval_query}")
        try:
            nodes = retriever.retrieve(retrieval_query)
        except Exception as e:
            debug_print(f"[debug] Retrieval attempt {attempt} failed: {e}")
            continue

        for rank, node in enumerate(nodes):
            node_id = _node_identity(node)
            node_relevance[node_id] = node_relevance.get(node_id, 0.0) + 1.0 / (_RRF_K + rank)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            all_nodes.append(node)

    # When the user explicitly names a file (e.g. "review Resume_2026.docx"),
    # pure vector ranking can miss it if the natural-language phrasing matches
    # other documents better. Force-include all chunks from the named file(s).
    source_hints = _extract_source_hints(user_query)
    if source_hints:
        forced_nodes = _force_include_source_hinted_nodes(
            retriever, source_hints, seen_node_ids
        )
        all_nodes.extend(forced_nodes)
        # Allow extra evidence slots so the forced chunks don't crowd everything else out.
        max_evidence = max(max_evidence, len(forced_nodes) + 4)

    # The shared vector collection accumulates chunks from every directory ever
    # indexed, so for a small launch folder the hybrid retriever may never even
    # surface its (often OCR-noisy) chunks as candidates. Pull them straight from
    # the docstore so the folder the user actually pointed at is always available
    # to the home-corpus floor in _source_nodes_to_query_evidence_lines.
    if home_corpus_dir:
        home_cap = int(os.environ.get("JARVIS_HOME_CORPUS_MAX_CHUNKS", "60"))
        home_nodes = _force_include_home_corpus_nodes(
            retriever, home_corpus_dir, seen_node_ids, home_cap
        )
        all_nodes.extend(home_nodes)

    evidence_lines = _source_nodes_to_query_evidence_lines(
        user_query,
        all_nodes,
        max_items=max_evidence,
        source_hints=source_hints,
        node_relevance=node_relevance,
        home_corpus_dir=home_corpus_dir,
    )
    return evidence_lines, all_nodes


def _force_include_source_hinted_nodes(
    retriever: object,
    source_hints: list[str],
    seen_node_ids: set,
) -> list[object]:
    """Pull every chunk whose file_name/file_path matches a source hint,
    bypassing vector ranking so user-named files always reach the LLM.

    llama-index `VectorIndexRetriever` exposes the docstore as both
    `retriever._docstore` and `retriever._index.docstore`; we try both.
    A hybrid `QueryFusionRetriever` exposes neither, so `build_hybrid_retriever`
    attaches the docstore as `_jarvis_docstore`; we check that first.
    """
    extra: list[object] = []
    docstore = (
        getattr(retriever, "_jarvis_docstore", None)
        or getattr(retriever, "_docstore", None)
        or getattr(getattr(retriever, "_index", None), "docstore", None)
        or getattr(getattr(retriever, "index", None), "docstore", None)
    )
    if docstore is None:
        print(
            f"[jarvis] source-hint force-include: no docstore on retriever "
            f"(type={type(retriever).__name__}); skipping"
        )
        return extra
    try:
        all_docs = docstore.docs or {}
    except Exception as e:
        print(f"[jarvis] source-hint force-include: docstore.docs access failed: {e}")
        return extra
    hint_basenames = {os.path.basename(h).lower() for h in source_hints if h}
    if not hint_basenames:
        return extra
    for node_id, node in all_docs.items():
        if node_id in seen_node_ids:
            continue
        metadata = getattr(node, "metadata", None) or {}
        fname = str(metadata.get("file_name") or "").lower()
        fpath = str(metadata.get("file_path") or "").lower()
        if not fname and not fpath:
            continue
        if any(h in fname or h in fpath for h in hint_basenames):
            seen_node_ids.add(node_id)
            extra.append(node)
    print(
        f"[jarvis] source-hint force-include: hints={sorted(hint_basenames)} "
        f"docstore_size={len(all_docs)} matched={len(extra)}"
    )
    return extra


def _force_include_home_corpus_nodes(
    retriever: object,
    home_corpus_dir: str,
    seen_node_ids: set,
    cap: int,
) -> list[object]:
    """Pull chunks whose file_path is under the launch directory directly from
    the docstore, bypassing vector ranking, so the folder the user pointed at is
    always an evidence candidate -- even for meta-queries that don't lexically
    match the (often OCR-noisy) file contents. Bounded by `cap`. The docstore is
    located the same way as in _force_include_source_hinted_nodes.
    """
    extra: list[object] = []
    docstore = (
        getattr(retriever, "_jarvis_docstore", None)
        or getattr(retriever, "_docstore", None)
        or getattr(getattr(retriever, "_index", None), "docstore", None)
        or getattr(getattr(retriever, "index", None), "docstore", None)
    )
    if docstore is None:
        return extra
    try:
        all_docs = docstore.docs or {}
    except Exception as e:
        print(f"[jarvis] home-corpus force-include: docstore.docs access failed: {e}")
        return extra
    home_root = os.path.abspath(home_corpus_dir).lower() + os.sep
    for node_id, node in all_docs.items():
        if node_id in seen_node_ids:
            continue
        metadata = getattr(node, "metadata", None) or {}
        fpath = str(metadata.get("file_path") or "").lower()
        if fpath and fpath.startswith(home_root):
            seen_node_ids.add(node_id)
            extra.append(node)
            if len(extra) >= cap:
                break
    if extra:
        print(
            f"[jarvis] home-corpus force-include: dir={home_corpus_dir} "
            f"matched={len(extra)} (cap={cap})"
        )
    return extra


def _hybrid_enabled() -> bool:
    """Hybrid (BM25 + dense) retrieval is the native runtime. Diagnostic switch
    only: JARVIS_HYBRID=0 forces dense-only retrieval to help isolate a
    retrieval issue; it is not a user-facing option."""
    return os.environ.get("JARVIS_HYBRID", "1") != "0"


def build_hybrid_retriever(
    index: object,
    dense_top_k: int,
    bm25_top_k: Optional[int] = None,
    fusion_top_k: Optional[int] = None,
    llm: Optional[Any] = None,
) -> object:
    """Build Jarvis's default retriever: a hybrid fusion of BM25 (sparse
    lexical) and dense vector retrieval, merged with Reciprocal Rank Fusion.

    Hybrid is the native runtime because this corpus (resumes, contracts, PDFs)
    is full of exact tokens -- names, dates, dollar amounts, IDs, defined terms
    -- where embedding search underperforms. BM25 closes that recall gap at the
    candidate-generation stage while the dense path keeps semantic matching.

    Opt out with JARVIS_HYBRID=0 to use dense-only.

    Degrades gracefully: if the BM25 package is missing, the docstore is empty,
    or fusion construction fails for any reason, this logs the cause and returns
    the plain dense retriever so retrieval never hard-fails.
    """
    dense_retriever = index.as_retriever(similarity_top_k=dense_top_k)

    if not _hybrid_enabled():
        print("[jarvis] Hybrid retrieval disabled (JARVIS_HYBRID=0); using dense-only vector retrieval.")
        return dense_retriever

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
    except Exception as e:
        print(f"[jarvis] Hybrid retrieval unavailable (BM25/fusion import failed: {e}); using dense-only.")
        return dense_retriever

    # BM25 builds from the docstore (node text), which is loaded on every
    # startup independent of the Milvus embedding/reindex-skip path.
    docstore = (
        getattr(getattr(index, "storage_context", None), "docstore", None)
        or getattr(index, "docstore", None)
    )
    doc_count = 0
    if docstore is not None:
        try:
            doc_count = len(docstore.docs or {})
        except Exception:
            doc_count = 0
    if docstore is None or doc_count == 0:
        print(f"[jarvis] Hybrid retrieval: docstore empty or missing (docs={doc_count}); using dense-only.")
        return dense_retriever

    bm25_k = bm25_top_k or dense_top_k
    fusion_k = fusion_top_k or dense_top_k
    try:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=bm25_k,
        )
        fusion = QueryFusionRetriever(
            [dense_retriever, bm25_retriever],
            similarity_top_k=fusion_k,
            # Upstream _retrieve_query_supported_evidence already expands the
            # query into many variants; num_queries=1 avoids an extra LLM call
            # inside the fusion retriever (keeps the hot path LLM-free).
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=False,
            # Pass the local LLM explicitly so the constructor never resolves a
            # default OpenAI client (it isn't used at query time when num_queries=1).
            llm=llm,
        )
        # Expose the docstore so _force_include_source_hinted_nodes still works
        # through the fusion retriever (which has no _index/_docstore attr).
        try:
            fusion._jarvis_docstore = docstore
        except Exception:
            pass
        print(
            f"[jarvis] Hybrid retrieval active: BM25(top_k={bm25_k}) + "
            f"dense(top_k={dense_top_k}) -> RRF(top_k={fusion_k}) over {doc_count} chunks."
        )
        return fusion
    except Exception as e:
        print(f"[jarvis] Hybrid retrieval construction failed ({e}); using dense-only.")
        return dense_retriever


def _query_candidate_score(
    text: str,
    terms: list[str],
    anchor_terms: list[str],
    require_value: bool,
    require_anchors: bool = True,
) -> int:
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0

    lowered = compact.lower()
    anchor_hits = sum(1 for term in anchor_terms if term in lowered)
    if require_anchors and anchor_terms and anchor_hits == 0:
        return 0

    topic_terms = _query_topic_terms(terms, anchor_terms)
    topic_hits = sum(1 for term in topic_terms if _term_family_hit(lowered, term))
    specific_terms = _specific_query_terms(terms, anchor_terms)
    specific_hits = sum(1 for term in specific_terms if _term_family_hit(lowered, term))
    value_hits = len(set(_value_numbers(compact)))
    payment_context_hits = _payment_context_score(lowered) if require_value else 0

    if require_value and value_hits == 0:
        return 0
    if topic_terms and topic_hits == 0 and anchor_hits == 0:
        return 0
    if specific_terms and specific_hits == 0 and anchor_hits == 0:
        return 0
    if not topic_terms and not anchor_hits:
        return 0

    return (
        anchor_hits * 30
        + topic_hits * 20
        + min(payment_context_hits, 6) * 18
        + min(value_hits, 8) * 6
        + min(len(compact) // 180, 6)
    )


def _query_focused_snippet(text: str, terms: list[str], anchor_terms: list[str], require_value: bool, max_chars: int = 2000) -> str:
    candidates = _detail_candidates(text)
    scored = [
        (_query_candidate_score(candidate, terms, anchor_terms, require_value, require_anchors=False), candidate)
        for candidate in candidates
    ]
    best_score, best_text = max(scored, key=lambda item: item[0]) if scored else (0, "")
    if best_score == 0:
        best_text = " ".join(normalize_text(text).split())

    snippet = " ".join(best_text.split())[:max_chars].rstrip()
    if len(best_text) > max_chars:
        snippet += "..."
    return snippet


def _query_evidence_is_supported(snippet: str, terms: list[str], anchor_terms: list[str], require_value: bool) -> bool:
    lowered = snippet.lower()
    anchor_hits = sum(1 for term in anchor_terms if term in lowered)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    topic_hit = any(_term_family_hit(lowered, term) for term in topic_terms) if topic_terms else False
    specific_terms = _specific_query_terms(terms, anchor_terms)
    specific_hit = any(_term_family_hit(lowered, term) for term in specific_terms) if specific_terms else False

    if anchor_terms and anchor_hits == 0 and not topic_hit:
        return False
    if topic_terms and not topic_hit and anchor_hits == 0:
        return False
    if specific_terms and not specific_hit and anchor_hits == 0 and not topic_hit:
        return False
    if require_value and not _value_numbers(snippet):
        return False
    return True
