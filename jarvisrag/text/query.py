"""Query-time text analysis for Jarvis (dependency leaf).

Pure-text helpers with no LLM, retrieval, or vector-store dependencies: query
parsing/term extraction, follow-up/turn detection, source-hint parsing, value &
facet extraction, retrieval-query preparation, and detail/value scoring. Higher
layers (jarvis_retrieval, jarvis_answer) import from here; this module imports
only shared config/text utilities.
"""

import os
import re

from llama_index.core.schema import Document

from jarvisrag.config import JARVIS_DOMAIN
from jarvisrag.text.textproc import (
    is_low_information_text,
    normalize_text,
    text_artifact_score,
)


GENERIC_DETAIL_KEYWORDS = [
    "account", "account number", "address", "amount", "balance", "certificate",
    "city", "contract", "credit", "customer", "date", "deduction", "description",
    "document", "due", "email", "employer", "fee", "id", "identifier", "invoice",
    "mailing", "name", "number", "organization", "owner", "payment", "phone",
    "property", "recipient", "registration", "statement", "total", "withheld",
]

DOMAIN_DETAIL_KEYWORDS = {
    "tax": [
        "w-2", "1099", "1098", "1040", "dividend", "distribution", "federal income",
        "form", "interest", "medicare wages", "mortgage", "payer", "principal residence",
        "section 199a", "social security", "socialsecuritynumber", "state wages", "tax",
        "wages",
    ],
}

GENERIC_BOILERPLATE_MARKERS = [
    ("all rights reserved", 40),
    ("continued", 15),
    ("disclosure", 30),
    ("for your records", 35),
    ("instructions", 40),
    ("privacy", 25),
    ("see instructions", 35),
    ("see the", 20),
    ("terms and conditions", 35),
]

DOMAIN_BOILERPLATE_MARKERS = {
    "tax": [
        ("department of the treasury", 60),
        ("internal revenue service", 45),
        ("keep for your records", 60),
        ("this is important tax information", 70),
        ("understanding your statement", 45),
        ("www.irs.gov", 60),
    ],
}


def _normalize_source_hint(source_hint: str) -> str:
    normalized = " ".join(source_hint.strip(" '\".,;:()[]{}").split())
    lowered = normalized.lower()
    for marker in (" document ", " file ", " entitled ", " named ", " called "):
        marker_index = lowered.rfind(marker)
        if marker_index >= 0:
            normalized = normalized[marker_index + len(marker):].strip()
            lowered = normalized.lower()
            break
    # The greedy regex used in _extract_source_hints may capture leading words
    # like "please review file.docx"; the actual filename is always the final
    # whitespace-separated token ending in the extension.
    tokens = normalized.split()
    if len(tokens) > 1 and "." in tokens[-1]:
        normalized = tokens[-1]
    base_name = os.path.basename(normalized)
    return base_name.lower() if base_name else lowered


def _is_real_source_hint(normalized: str) -> bool:
    """True only for an actual file reference, not a bare extension.

    A query like "summarize the .pdf files" or "any scanned jpeg" mentions a
    file *type*, not a named file; treating ".pdf" as a hint would force-include
    every PDF in the corpus. A genuine hint must have a non-empty stem before the
    final extension (e.g. "resume.pdf" -> stem "resume")."""
    if not normalized:
        return False
    stem = normalized.rsplit(".", 1)[0].strip()
    return bool(stem)


def _extract_source_hints(user_query: str) -> list[str]:
    hints = []
    for match in re.findall(r"[\w .#&()\-]+\.(?:pdf|png|jpe?g|txt|docx?|xlsx?|csv)", user_query, re.I):
        normalized = _normalize_source_hint(match)
        if _is_real_source_hint(normalized) and normalized not in hints:
            hints.append(normalized)
    for match in re.findall(r"['\"]([^'\"]+\.(?:pdf|png|jpe?g|txt|docx?|xlsx?|csv))['\"]", user_query, re.I):
        normalized = _normalize_source_hint(match)
        if _is_real_source_hint(normalized) and normalized not in hints:
            hints.append(normalized)
    return hints[:4]


def _line_matches_source_hints(evidence_line: str, source_hints: list[str]) -> bool:
    if not source_hints:
        return False
    lowered = evidence_line.lower()
    return any(hint in lowered or os.path.basename(hint) in lowered for hint in source_hints)


def _prioritize_source_hinted_evidence(evidence_lines: list[str], source_hints: list[str]) -> list[str]:
    if not evidence_lines or not source_hints:
        return evidence_lines
    hinted = [line for line in evidence_lines if _line_matches_source_hints(line, source_hints)]
    other = [line for line in evidence_lines if line not in hinted]
    return hinted + other


def _query_requests_refinement(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    refinement_markers = (
        "be more specific", "clarify", "correction", "dig deeper", "drill down",
        "explain that", "follow up", "go deeper", "instead", "narrow", "refine",
        "rephrase", "same evidence", "tell me more", "use that", "what about",
        "what does that mean", "why", "you missed",
    )
    return any(marker in lowered for marker in refinement_markers)


# Pronouns / anaphora that imply the user is still talking about the prior
# turn's subject without naming it (e.g. "how long did HE work THERE?").
# Matched with word boundaries so we never false-trigger on substrings like
# "the", "where", "then" inside other words. Only consulted when a prior turn
# exists, so a brand-new first question can't be mistaken for a follow-up.
_FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|there|this|those|these)\b"
)


# Lowercased words that are never, on their own, a "new subject" proper noun:
# question/lead words, conjunctions, articles, and the anaphoric pronouns. Used
# by `_names_new_subject` to avoid mistaking these for a topic switch.
_NON_SUBJECT_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "can", "could", "would", "should", "do", "does", "did", "is", "are",
    "was", "were", "will", "tell", "show", "find", "and", "but", "also",
    "additionally", "conversely", "however", "so", "then", "the", "a", "an",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "theirs", "there", "this", "those", "these", "it", "its", "that",
})


def _names_new_subject(user_query: str, last_turn: dict[str, object]) -> bool:
    """True if `user_query` introduces a Title-case proper-noun subject that does
    not appear in the prior turn — i.e. the user switched topics rather than
    continuing the previous one.

    Used to stop follow-up context from bleeding into an unrelated question. The
    first word is skipped (sentence-initial capitalization is not a reliable
    proper-noun signal), and anaphoric pronouns / lead/question words are never
    treated as subjects.
    """
    prior_text = " ".join(
        str(last_turn.get(key) or "") for key in ("user_query", "effective_query")
    ).lower()
    for idx, raw in enumerate(str(user_query or "").split()):
        if idx == 0:
            continue  # sentence-initial word is unreliable (verb/common noun)
        token = re.sub(r"'s$", "", raw.strip(".,;:!?'\"()[]"))
        if len(token) < 2 or not re.match(r"^[A-Z][a-z]", token):
            continue
        if token.lower() in _NON_SUBJECT_WORDS:
            continue
        if token.lower() not in prior_text:
            return True
    return False


def _looks_like_follow_up(user_query: str, last_turn: dict[str, object]) -> bool:
    if not last_turn.get("user_query"):
        return False
    lowered = " ".join(user_query.lower().split())
    if not lowered:
        return False
    follow_up_markers = (
        "that", "it", "its", "same", "previous", "earlier", "answer", "evidence",
        "relevant evidence", "file", "filename", "entitled", "document", "pdf",
        "you said", "you missed", "look at", "use the", "in the evidence",
        "clarify", "dig deeper", "drill down", "instead", "narrow", "refine",
        "tell me more", "what about", "what does that mean",
    )
    has_source_hint = bool(_extract_source_hints(user_query))
    wants_refinement = _query_requests_reasoning(user_query) or _query_requests_refinement(user_query)
    has_pronoun = bool(_FOLLOW_UP_PRONOUN_RE.search(lowered))
    has_marker = any(marker in lowered for marker in follow_up_markers)

    # An explicit source reference ("the answer is in X.pdf") is a deliberate
    # continuation and always stays a follow-up.
    if has_source_hint:
        return True

    # Topic-switch guard: if the user names a NEW proper-noun subject and does
    # NOT refer back with a pronoun, treat it as a fresh question so the prior
    # turn's context does not bleed in — even when a generic marker like
    # "what about"/"instead"/"it" is present (these double as refinement and as
    # topic-switch phrasing). A back-referring pronoun ("his GPA at Johns
    # Hopkins") keeps the thread because the subject is still continuous.
    if not has_pronoun and _names_new_subject(user_query, last_turn):
        return False

    return has_marker or has_pronoun or wants_refinement


def _is_negative_evidence_feedback(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    negative_markers = (
        "bad evidence", "does not answer", "doesn't answer", "has no reference",
        "has no references", "irrelevant", "not relevant", "no reference",
        "no references", "not in the evidence", "wrong evidence", "you provided has no",
    )
    return any(marker in lowered for marker in negative_markers)


def _query_requests_reasoning(user_query: str) -> bool:
    lowered = " ".join(user_query.lower().split())
    reasoning_markers = (
        "based on", "could mean", "do you think", "hypothesis", "hypothesize",
        "implication", "implications", "imply", "implies", "infer", "inference", "likely", "may mean",
        "might mean", "should i", "tax wise", "tax-wise", "what do you think",
        "what would", "would be", "would happen",
    )
    conditional_reasoning = bool(re.search(r"\bif\b.+\b(then|would|could|might|may)\b", lowered))
    return conditional_reasoning or any(marker in lowered for marker in reasoning_markers)


# --- Enumeration / "list everything" intent (vocabulary-free) ---------------
# Cues that signal the user wants a COMPLETE list rather than a single fact.
# "Strong" cues imply enumeration on their own; "weak" cues only count when the
# enumerated noun is itself plural (a morphological list signal). NO domain or
# dataset vocabulary is hardcoded — the enumerated noun is read from the query,
# so this works identically for resumes, contracts, recipes, case law, etc.
_STRONG_ENUM_CUES = (
    "list", "enumerate", "every", "each", "all of", "how many",
    "name all", "name every", "name each",
)
_WEAK_ENUM_CUES = (
    "what", "which", "name the", "give me", "show me", "tell me",
)
# Words that are never the SUBJECT being enumerated (command/instruction verbs
# and auxiliaries); excluded when reading the enumerated noun from the query.
# Purely grammatical function words — not tied to any dataset.
_NON_TARGET_WORDS = frozenset({
    "list", "enumerate", "name", "give", "provide", "show", "tell", "find",
    "identify", "itemize", "describe", "summarize", "summarise", "explain",
    "define", "outline", "detail", "compare", "contrast", "create", "write",
    "draft", "compose", "generate", "make", "build", "prepare", "produce",
    "update", "revise", "format", "tailor", "design", "search", "get",
    "display", "print", "return", "gather", "collect", "review",
    "has", "have", "had", "does", "did", "are", "was", "were", "been", "being",
    # Quantifiers / prepositions / fillers that survive the length filter but are
    # never the enumerated subject (kept out so the boost targets stay clean).
    "every", "each", "many", "much", "any", "some", "more", "most", "than",
    "over", "under", "about", "into", "onto", "from", "with", "that", "this",
    "they", "them", "their", "there", "here", "then", "also", "only", "just",
})
# Singular words that end in "s" -> must NOT be mistaken for plurals.
_NON_PLURAL_S_WORDS = frozenset({
    "status", "analysis", "diagnosis", "synopsis", "series", "species",
    "address", "business", "process", "access", "progress", "congress",
    "bonus", "census", "campus", "focus", "virus", "basis", "thesis",
    "crisis", "lens", "news", "always",
})


def _looks_plural(term: str) -> bool:
    """Heuristic morphological test: a content word that signals a list."""
    t = term.lower()
    return len(t) > 3 and t.endswith("s") and t not in _NON_PLURAL_S_WORDS


def _query_enumeration_target(user_query: str) -> list[str]:
    """The content noun(s) the user is asking to enumerate, read straight from
    the query — NOT matched against any fixed vocabulary, so it works for ANY
    dataset (publications, patents, clauses, ingredients, exhibits, ...).

    Drops proper-noun anchors (names like "Jordan"), command/instruction verbs,
    and auxiliaries, keeping the subject the user wants listed. Used to (a) detect
    enumeration intent and (b) boost chunks that actually contain that subject.
    """
    terms = _query_terms(user_query)
    anchors = set(_query_anchor_terms(user_query, terms))
    targets = [
        term
        for term in terms
        if term not in anchors and term not in _NON_TARGET_WORDS and len(term) > 3
    ]
    return targets[:3]


def _query_requests_enumeration(user_query: str) -> bool:
    """True when the query asks to list EVERY matching item across the corpus.

    Vocabulary-free: enumeration is inferred from listing cues plus the grammar
    of the query, never from a curated noun list. A strong listing cue
    ("list", "every", "how many", ...) triggers on its own; a weak cue
    ("what", "which", ...) only triggers when the enumerated noun is plural.

    Enumeration queries must surface the COMPLETE set, so the retrieval layer
    sends fuller chunks, skips topic-coverage dedup, and boosts chunks that
    contain the enumerated subject when this returns True.
    """
    lowered = " ".join((user_query or "").lower().split())
    if not lowered:
        return False
    targets = _query_enumeration_target(user_query)
    if not targets:
        return False
    if any(cue in lowered for cue in _STRONG_ENUM_CUES):
        return True
    if any(cue in lowered for cue in _WEAK_ENUM_CUES) and any(_looks_plural(t) for t in targets):
        return True
    return False


# Verbs that signal the user wants Jarvis to AUTHOR/compose a new document
# (organize + summarize + format) rather than answer a factual lookup. Paired
# with a document-type noun below.
_AUTHORING_VERBS = (
    "create", "write", "draft", "compose", "generate", "build", "make",
    "produce", "prepare", "put together", "assemble", "format", "tailor",
    "update", "revise", "rewrite", "rework", "polish", "design",
)
# Document types Jarvis can author from ANY indexed corpus (domain-agnostic):
# career docs, legal/financial briefs, research summaries, narratives, etc.
_AUTHORING_NOUNS = (
    "document", "summary", "report", "brief", "memo", "memorandum", "abstract",
    "overview", "synopsis", "digest", "outline", "proposal", "narrative",
    "statement", "essay", "letter", "cover letter", "bio", "biography",
    "profile", "resume", "résumé", "cv", "curriculum vitae", "portfolio",
    "linkedin", "elevator pitch", "objective", "headline", "write-up", "writeup",
)


def _query_requests_authoring(user_query: str) -> bool:
    """True when the user asks Jarvis to COMPOSE a new document (summary, report,
    brief, letter, resume, etc.) from the indexed material rather than answer a
    factual question about it.

    DOMAIN-AGNOSTIC: this gates purely on a generic authoring verb + a generic
    document-type noun, with no assumption about subject matter, so it works for
    any corpus (career records, contracts, research papers, financial filings,
    fiction, etc.). Authoring requests need the LLM to organize, summarize, and
    format across many documents — work the strict-grounding answer+review loop
    rejects as "not traceable to a single evidence line". When this returns True
    the pipeline routes to a single uncapped composition pass that uses retrieved
    evidence as source material without the factual-review gate.
    """
    lowered = " ".join((user_query or "").lower().split())
    if not lowered:
        return False
    # Must name a document type to author.
    if not any(re.search(rf"\b{re.escape(noun)}\b", lowered) for noun in _AUTHORING_NOUNS):
        return False
    # The authoring verb must appear as an IMPERATIVE or a request addressed to
    # the assistant — not merely co-occur inside a factual wh-question such as
    # "what statement did he make" or "which report did they generate". This
    # keeps the gate precise across arbitrary corpora and question phrasings.
    verb_alt = "|".join(re.escape(verb) for verb in _AUTHORING_VERBS)
    # (a) Imperative at/near the start, optionally after a polite lead-in.
    if re.match(rf"^(?:please\s+|kindly\s+|pls\s+|now\s+|go\s+ahead\s+and\s+)*(?:{verb_alt})\b", lowered):
        return True
    # (b) Explicit request directed at the assistant, e.g. "can you draft",
    #     "could you please write", "please generate", "i need you to compose".
    request_lead = (
        r"(?:can|could|would|will)\s+you(?:\s+please)?"
        r"|please|kindly"
        r"|i\s+(?:want|need|would\s+like|'d\s+like)\s+(?:you\s+)?to"
        r"|help\s+me(?:\s+to)?"
        r"|let'?s"
    )
    if re.search(rf"\b(?:{request_lead})\s+(?:{verb_alt})\b", lowered):
        return True
    return False


def _effective_query_for_turn(user_query: str, last_turn: dict[str, object]) -> tuple[str, bool, list[str]]:
    source_hints = _extract_source_hints(user_query)
    if not _looks_like_follow_up(user_query, last_turn):
        return user_query, False, source_hints

    prior_query = str(last_turn.get("effective_query") or last_turn.get("user_query") or "").strip()
    if not prior_query:
        return user_query, False, source_hints

    lowered = " ".join(user_query.lower().split())
    asks_new_question = bool(re.search(r"\b(what|which|who|when|where|why|how|can|could|show|tell|find)\b", lowered))
    source_correction = bool(source_hints) or any(
        marker in lowered
        for marker in (
            "answer is in", "in the evidence", "entitled", "same evidence",
            "same file", "use the file", "look at the file",
        )
    )
    if source_correction and not asks_new_question:
        return prior_query, True, source_hints

    effective_query = f"{prior_query}\n{user_query}"
    return effective_query, True, source_hints


def _latest_user_turn(user_query: str) -> str:
    lines = [line.strip() for line in str(user_query or "").splitlines() if line.strip()]
    return lines[-1] if lines else str(user_query or "").strip()


def _is_refinement_turn(user_query: str) -> bool:
    return "\n" in str(user_query or "") or _query_requests_refinement(user_query)


def _query_requests_corpus_details(user_query: str) -> bool:
    text = " ".join(user_query.lower().split())
    detail_terms = ["details", "facts", "information", "what is present", "what's present"]
    corpus_terms = ["retrieved documents", "indexed documents", "documents", "data", "files"]
    return any(term in text for term in detail_terms) and any(term in text for term in corpus_terms)


def _query_requests_source_details(user_query: str) -> bool:
    if not _extract_source_hints(user_query):
        return False
    text = " ".join(user_query.lower().split())
    source_detail_markers = (
        "in the document", "in the file", "inside the document", "inside the file",
        "what is in", "what's in", "what does", "tell me what", "summarize",
        "summary", "contents", "about this document", "about the document",
    )
    return any(marker in text for marker in source_detail_markers)


def _query_terms(user_query: str) -> list[str]:
    stop_words = {
        "about", "all", "amount", "amounts", "and", "answer", "apply", "applies", "are", "can",
        "could", "detail", "details", "did", "do", "does", "during", "entitled", "evidence", "file",
        "filename", "find", "follow", "for", "from", "have", "how", "information", "is", "much", "present",
        "pdf", "png", "jpg", "jpeg", "txt", "doc", "docx", "xlsx", "csv",
        "provided", "references", "reference", "show", "tell", "that", "the", "their", "there",
        "these", "this", "user", "was", "were", "what", "which", "with", "you", "year",
        "based", "hypothesis", "hypothesize", "implication", "implications", "imply", "implies", "increase", "increased",
        "increases", "increasing", "infer", "inference", "into", "likely", "maybe", "might",
        "should", "taxwise", "think", "wise", "would",
        "pertain", "pertains", "pertaining",
    }
    terms = []
    normalized_query = re.sub(r"\btax\s+year\b", "year", user_query.lower())
    for term in re.findall(r"\b(?:\d{2,}|[a-zA-Z][a-zA-Z0-9-]{2,})\b", normalized_query):
        if term not in stop_words and term not in terms:
            terms.append(term)
    # Also preserve short all-uppercase acronyms from the original (case-sensitive) query
    # so 2-3 letter abbreviations like "EV", "IRS", "HSA", "TOD" are searchable.
    for acronym in re.findall(r"\b[A-Z]{2,5}\b", user_query):
        lowered = acronym.lower()
        if lowered in stop_words or lowered in terms:
            continue
        terms.append(lowered)
    return terms[:8]


def _query_anchor_terms(user_query: str, terms: list[str]) -> list[str]:
    # Words that are NOT topical anchors even when capitalized at the start of a
    # query. Anchors gate which chunks are considered relevant, so an imperative
    # lead verb like "List"/"Create" must never become an anchor — otherwise it
    # matches the common word in unrelated documents (e.g. "list" in technical
    # PDFs) and floods retrieval with off-topic chunks, crowding the real answer
    # out of the evidence cap.
    question_words = {
        "Answer", "Can", "Could", "Evidence", "File", "Find", "How", "It",
        "Show", "Tell", "The", "This", "What", "Which", "You",
        # Imperative / instruction lead verbs (enumeration, description, authoring):
        "List", "Enumerate", "Name", "Give", "Provide", "Identify", "Itemize",
        "Describe", "Summarize", "Summarise", "Explain", "Define", "Outline",
        "Detail", "Compare", "Contrast", "Create", "Write", "Draft", "Compose",
        "Generate", "Make", "Build", "Prepare", "Produce", "Update", "Revise",
        "Format", "Tailor", "Design", "Search", "Get", "Display", "Print",
        "Return", "Gather", "Collect", "Review",
    }
    anchors = []

    for term in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", user_query):
        if term in question_words:
            continue
        lowered = term.lower()
        if lowered not in anchors:
            anchors.append(lowered)

    return anchors[:4]


def _query_phrase_anchors(user_query: str) -> list[str]:
    """Multi-word phrases from the query that should be preferred verbatim in evidence.

    Captures:
      * Runs of 2+ capitalized words (e.g. "Adjusted Gross Income", "Doctors Without Borders").
      * Quoted substrings (single or double quoted).
      * Short ALL-CAPS acronyms (>= 2 letters) — these often expand to a phrase in source documents.

    Returned phrases are lowercased and de-duplicated. Common question stems
    like "The" or "This" at the start are stripped.
    """
    phrases: list[str] = []
    leading_stop = {
        "the", "this", "that", "these", "those", "a", "an", "my", "our", "their", "your",
    }

    def _add(phrase: str) -> None:
        normalized = " ".join(phrase.lower().split())
        if not normalized:
            return
        parts = normalized.split()
        while parts and parts[0] in leading_stop:
            parts.pop(0)
        if len(parts) < 2:
            return
        cleaned = " ".join(parts)
        if cleaned not in phrases:
            phrases.append(cleaned)

    for match in re.finditer(r"(?:\b[A-Z][a-zA-Z]+\b\s+){1,5}\b[A-Z][a-zA-Z]+\b", user_query):
        _add(match.group(0))
    for match in re.finditer(r"['\"]([^'\"]{3,80})['\"]", user_query):
        _add(match.group(1))

    acronyms: list[str] = []
    for acronym in re.findall(r"\b[A-Z]{2,5}\b", user_query):
        lowered = acronym.lower()
        if lowered not in acronyms:
            acronyms.append(lowered)

    return phrases[:4] + acronyms[:4]


def _query_page_hints(user_query: str) -> list[int]:
    """Page numbers explicitly named in the query (e.g. 'page 8', 'on page 12')."""
    pages: list[int] = []
    for match in re.finditer(r"\bpage[s]?\s+(\d{1,5})\b", user_query, re.I):
        try:
            page = int(match.group(1))
        except ValueError:
            continue
        if page and page not in pages:
            pages.append(page)
    return pages[:4]


def _evidence_line_matches_phrase(evidence_body: str, phrase: str) -> bool:
    if not phrase:
        return False
    return phrase in evidence_body.lower()


def _evidence_line_matches_page(evidence_line: str, pages: list[int]) -> bool:
    if not pages:
        return False
    lowered = evidence_line.lower()
    return any(re.search(rf",\s*page\s+{page}\b", lowered) for page in pages)


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term in lowered for term in terms)


def _document_search_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    parts = [doc.text or ""]
    for key in ("file_name", "file_path", "parent_file", "source"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _query_requests_values(user_query: str) -> bool:
    return bool(re.search(
        r"\b(amounts?|balances?|costs?|deductions?|due|expenses?|how much|income|owe|owed|payments?|totals?|values?|withhold(?:ing)?)\b",
        user_query,
        re.I,
    ))


def _query_topic_terms(terms: list[str], anchor_terms: list[str]) -> list[str]:
    anchor_set = set(anchor_terms)
    return [term for term in terms if term not in anchor_set]


def _specific_query_terms(terms: list[str], anchor_terms: list[str]) -> list[str]:
    generic_terms = {
        "amount", "amounts", "balance", "balances", "due", "federal", "income",
        "owe", "owed", "payment", "payments", "plan", "plans", "state", "tax",
        "taxes", "total", "totals", "value", "values",
    }
    return [
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if term not in generic_terms and (any(char.isdigit() for char in term) or len(term) > 4)
    ]


def _term_family(term: str) -> list[str]:
    families = {
        "charitable": ["charit", "contribut", "donat", "gift"],
        "charities": ["charit", "contribut", "donat", "gift"],
        "charity": ["charit", "contribut", "donat", "gift"],
        "contribution": ["contribut", "charit", "donat", "gift"],
        "contributions": ["contribut", "charit", "donat", "gift"],
        "donated": ["donat", "contribut", "charit", "gift"],
        "donation": ["donat", "contribut", "charit", "gift"],
        "donations": ["donat", "contribut", "charit", "gift"],
        "due": ["due", "owe", "owed", "payment", "remittance", "debit", "withdraw", "balance due"],
        "federal": ["irs", "treasury", "1040", "u.s. individual income tax", "united states treasury"],
        "529": ["529", "maryland529", "qualified tuition", "qtp", "esa", "college 529", "state qtp"],
        "owed": ["owe", "owed", "amount you owe", "payment", "remittance", "debit", "withdraw", "balance due"],
        "owe": ["owe", "owed", "amount you owe", "payment", "remittance", "debit", "withdraw", "balance due"],
        "payment": ["payment", "payments", "remittance", "debit", "withdraw", "amount you owe"],
        "payments": ["payment", "payments", "remittance", "debit", "withdraw", "amount you owe"],
        "investment": ["investment", "invest", "portfolio", "account", "contribution", "contributions", "principal", "earnings"],
        "investments": ["investment", "invest", "portfolio", "account", "contribution", "contributions", "principal", "earnings"],
        "plan": ["plan", "program", "account", "qtp", "esa"],
        "state": ["maryland", "md", "comptroller", "form 502", "502/505", "local tax", "state tax", "state income"],
        "tax": ["tax", "taxes", "1040", "irs", "treasury"],
        "taxes": ["tax", "taxes", "1040", "irs", "treasury"],
        "total": ["total", "amount", "payment", "remittance", "balance", "line 37"],
        "totals": ["total", "amount", "payment", "remittance", "balance", "line 37"],
    }
    if term in families:
        return families[term]
    if len(term) > 5:
        return [term, term.rstrip("s")]
    return [term]


def _term_family_hit(text: str, term: str) -> bool:
    lowered = text.lower().replace("\\n", " ")
    if term == "federal":
        if any(alias in lowered for alias in _term_family(term)):
            return True
        without_bank_name = re.sub(r"\bnavy\s+federal(?:\s+credit\s+union)?\b", "", lowered)
        return bool(re.search(r"\bfederal\b", without_bank_name))
    for alias in _term_family(term):
        alias = alias.lower()
        if len(alias) <= 3 and re.fullmatch(r"[a-z0-9]+", alias):
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return True
        elif alias in lowered:
            return True
    return False


def _query_topic_coverage(text: str, terms: list[str], anchor_terms: list[str]) -> set[str]:
    lowered = text.lower()
    return {
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if _term_family_hit(lowered, term)
    }


PAYMENT_CONTEXT_MARKERS = (
    "amount due", "amount of tax payment", "amount you owe", "balance due",
    "balance owed", "debit", "direct debit", "due/-refund", "electronic funds withdrawal",
    "owed", "payment", "remittance", "tax balances", "withdraw", "withdrawal",
)


def _query_value_facets(terms: list[str], anchor_terms: list[str]) -> list[str]:
    value_intent_terms = {
        "amount", "amounts", "balance", "balances", "due", "owe", "owed", "payment",
        "payments", "tax", "taxes", "total", "totals", "value", "values",
    }
    facets = [
        term
        for term in _query_topic_terms(terms, anchor_terms)
        if term not in value_intent_terms
    ]
    return facets or _query_topic_terms(terms, anchor_terms)


def _money_values(text: str) -> list[str]:
    values = []
    for value in re.findall(r"\$\s*\d[\d,]*(?:\.\d+)?", text):
        normalized = re.sub(r"\$\s*", "$", value.strip())
        if normalized not in values:
            values.append(normalized)
    return values


def _value_amounts(text: str, require_money: bool = False) -> list[str]:
    money_values = _money_values(text)
    if require_money:
        return money_values

    values = list(money_values)
    for value in _value_numbers(text):
        normalized = value.strip()
        if not normalized:
            continue
        if normalized.startswith("$"):
            normalized = re.sub(r"\$\s*", "$", normalized)
        if normalized not in values:
            values.append(normalized)
    return values


def _format_amount(value: str) -> str:
    value = value.strip()
    if value.startswith("$"):
        return value
    return f"${value}"


def _amount_segment_score(segment: str, facet: str, full_text_has_facet: bool) -> int:
    score = _payment_context_score(segment) * 20
    if _term_family_hit(segment, facet):
        score += 60
    elif full_text_has_facet:
        score += 4
    if re.search(r"\b(amount|balance|due|owe|owed|payment|refund|remittance|scheduled|withdraw|withdrawn)\b", segment, re.I):
        score += 12
    if _money_values(segment):
        score += 10
    if any("," in value for value in _value_amounts(segment)):
        score += 3
    return score


def _facet_row_amount_value(facet: str, text: str) -> str:
    amount_pattern = r"(?:\$\s*)?\d[\d,]*(?:\.\d+)?%?"
    if facet == "federal":
        label_pattern = r"federal"
        next_label_pattern = r"federal|md|maryland|state|total|due/-refund|breakout"
    elif facet == "state":
        label_pattern = r"md|maryland|state"
        next_label_pattern = r"federal|md|maryland|state|total|due/-refund|breakout"
    else:
        label_pattern = re.escape(facet)
        next_label_pattern = r"total|due/-refund|breakout"

    flattened = " ".join(text.replace("\\n", "\n").split())
    row_pattern = (
        rf"\b(?:{label_pattern})\b"
        rf"(?:\s+{amount_pattern}){{1,6}}"
        rf"(?=\s+(?:{next_label_pattern})\b|$)"
    )
    for match in re.finditer(row_pattern, flattened, re.I):
        row_money = _money_values(match.group(0))
        if row_money:
            return row_money[-1]
        row_values = _value_amounts(match.group(0))
        if row_values:
            return row_values[-1]
    return ""


def _facet_amount_value(facet: str, text: str, values: list[str]) -> str:
    row_value = _facet_row_amount_value(facet, text)
    if row_value:
        return row_value

    full_text_has_facet = _term_family_hit(text, facet)
    best_segment: tuple[int, list[str]] | None = None
    separators = r"\n|\\n|\||;|(?<=[.!?])\s+"
    for segment in re.split(separators, text):
        segment = segment.strip()
        if not segment:
            continue
        if facet == "federal":
            facet_hit = re.search(r"\bfederal\b", segment, re.I) or _term_family_hit(segment, facet)
        elif facet == "state":
            facet_hit = re.search(r"\b(?:md|maryland|state)\b", segment, re.I) or _term_family_hit(segment, facet)
        else:
            facet_hit = _term_family_hit(segment, facet)
        context_score = _payment_context_score(segment)
        if not facet_hit and not (full_text_has_facet and context_score > 0):
            continue
        segment_values = _value_amounts(segment)
        if segment_values:
            # Prefer money-formatted values ($1,234.56) over bare integers in the
            # same segment so that nearby identifiers like ZIP codes, invoice
            # numbers, or quantities don't masquerade as currency.
            money_only = [v for v in segment_values if v.startswith("$")]
            if money_only:
                segment_values = money_only
            score = _amount_segment_score(segment, facet, full_text_has_facet)
            if best_segment is None or score > best_segment[0]:
                best_segment = (score, segment_values)

    if best_segment:
        return best_segment[1][-1]

    money_fallback = [v for v in values if v.startswith("$")]
    if money_fallback:
        return money_fallback[-1]
    return values[-1] if values else ""


def _payment_context_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for marker in PAYMENT_CONTEXT_MARKERS if marker in lowered)


def _facet_label(facet: str, evidence_line: str) -> str:
    lowered = evidence_line.lower()
    if facet == "state" and "maryland" in lowered:
        return "State/Maryland"
    return facet.replace("-", " ").title()


def _source_label_from_evidence_line(evidence_line: str) -> str:
    compact = evidence_line[2:] if evidence_line.startswith("- ") else evidence_line
    return compact.split(": ", 1)[0].strip()


def _evidence_body_from_evidence_line(evidence_line: str) -> str:
    compact = evidence_line[2:] if evidence_line.startswith("- ") else evidence_line
    parts = compact.split(": ", 1)
    return parts[1] if len(parts) == 2 else compact


def _prepare_retrieval_query(user_query: str) -> str:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    require_value = _query_requests_values(user_query)

    focus_terms = []
    for term in anchor_terms:
        focus_terms.append(term)
    for term in topic_terms:
        focus_terms.extend(_term_family(term))
    if require_value:
        focus_terms.extend([
            "amount", "amounts", "total", "totals", "value", "values", "dollars",
            "deduction", "payment", "remittance", "debit", "withdraw", "balance due",
        ])

    deduped_focus = []
    for term in focus_terms:
        normalized = term.strip().lower()
        if normalized and normalized not in deduped_focus:
            deduped_focus.append(normalized)

    if not deduped_focus:
        return " ".join(user_query.split())

    return (
        "Find indexed evidence for: "
        + " ".join(deduped_focus)
        + ". Original question: "
        + " ".join(user_query.split())
    )


def _retrieval_query_variants(user_query: str) -> list[str]:
    terms = _query_terms(user_query)
    anchor_terms = _query_anchor_terms(user_query, terms)
    topic_terms = _query_topic_terms(terms, anchor_terms)
    require_value = _query_requests_values(user_query)

    topic_family_terms = []
    for term in topic_terms:
        topic_family_terms.extend(_term_family(term))

    value_terms = [
        "amount", "amounts", "total", "totals", "value", "values", "dollars",
        "deduction", "payment", "remittance", "debit", "withdraw", "balance due",
    ] if require_value else []
    query_parts = [
        _prepare_retrieval_query(user_query),
        " ".join(user_query.split()),
        " ".join(anchor_terms + topic_family_terms + value_terms),
        " ".join(anchor_terms + topic_terms + value_terms),
    ]
    for topic_term in topic_terms:
        query_parts.append(" ".join(anchor_terms + _term_family(topic_term) + value_terms))
    query_parts.extend([
        " ".join(topic_family_terms + value_terms),
        " ".join(anchor_terms),
    ])

    variants = []
    for query in query_parts:
        normalized = " ".join(query.split())
        if normalized and normalized not in variants:
            variants.append(normalized)
    return variants


def _topic_snippet_score(text: str, terms: list[str]) -> int:
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0
    lowered = compact.lower()
    term_hits = sum(1 for term in terms if term in lowered)
    if term_hits == 0:
        return 0
    value_hits = len(set(_value_numbers(compact)))
    return term_hits * 12 + min(value_hits, 8) * 4 + min(len(compact) // 180, 6)


def _value_numbers(text: str) -> list[str]:
    normalized_text = text.replace("\\n", " ")
    numbers = re.findall(r"(?<![\w*/.-])(?:\$\s*)?\d[\d,]*(?:\.\d+)?%?(?![\w/.-])", normalized_text)
    values = []
    for number in numbers:
        stripped = number.strip()
        digits_only = re.sub(r"\D", "", number)
        if not digits_only:
            continue
        numeric_value = int(digits_only)
        if 1900 <= numeric_value <= 2099:
            continue
        has_amount_marker = stripped.startswith("$") or "," in stripped or "." in stripped or stripped.endswith("%")
        if not has_amount_marker and numeric_value < 100:
            continue
        if not has_amount_marker and len(digits_only) > 6:
            continue
        values.append(stripped)
    return values


def _detail_score(text: str) -> int:
    artifact_penalty = text_artifact_score(text)
    compact = " ".join(normalize_text(text).split())
    if not compact or is_low_information_text(compact):
        return 0
    if artifact_penalty >= 100:
        return 0

    lowered = compact.lower()
    boilerplate_markers = list(GENERIC_BOILERPLATE_MARKERS)
    boilerplate_markers.extend(DOMAIN_BOILERPLATE_MARKERS.get(JARVIS_DOMAIN, []))
    boilerplate_penalty = sum(
        penalty
        for marker, penalty in boilerplate_markers
        if marker in lowered
    )
    tokens = compact.split()
    numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
    readable_words = re.findall(r"[A-Za-z]{3,}", compact)
    unique_readable_words = set(word.lower() for word in readable_words)
    detail_keywords = list(GENERIC_DETAIL_KEYWORDS)
    detail_keywords.extend(DOMAIN_DETAIL_KEYWORDS.get(JARVIS_DOMAIN, []))
    keyword_hits = sum(
        1
        for keyword in detail_keywords
        if keyword in lowered
    )

    if len(unique_readable_words) < 3 and keyword_hits == 0:
        return 0
    if numeric_tokens and len(numeric_tokens) > max(len(readable_words) * 2, 8) and keyword_hits < 2:
        return 0

    repeated_token_penalty = max(len(tokens) - len(set(tokens)), 0)
    value_score = min(len(set(_value_numbers(compact))), 8) * 5
    label_score = 6 if ":" in compact else 0
    word_score = min(len(unique_readable_words), 24)
    score = value_score + keyword_hits * 8 + label_score + word_score + min(len(compact) // 160, 6)
    score -= min(repeated_token_penalty, 20)
    score -= boilerplate_penalty
    return max(score - artifact_penalty, 0)


def _detail_candidates(text: str) -> list[str]:
    text = normalize_text(text)
    lines = [line.strip() for line in re.split(r"[\n;]+", text) if line.strip()]
    candidates = list(lines)
    for window_size in (2, 3):
        for start in range(0, max(len(lines) - window_size + 1, 0)):
            candidates.append(" | ".join(lines[start:start + window_size]))
    compact = " ".join(text.split())
    if compact:
        candidates.append(compact)
    return candidates or [" ".join(text.split())]


def _best_detail_candidate(text: str) -> tuple[int, str]:
    candidates = _detail_candidates(text)
    best_candidate = max(candidates, key=_detail_score)
    return _detail_score(best_candidate), best_candidate


def _best_detail_snippet(text: str, max_chars: int = 1600) -> str:
    score, best_text = _best_detail_candidate(text)
    if score == 0:
        best_text = " ".join(normalize_text(text).split())

    snippet = " ".join(best_text.split())[:max_chars].rstrip()
    if len(best_text) > max_chars:
        snippet += "..."
    return snippet
