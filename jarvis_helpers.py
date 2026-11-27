"""Backward-compatible facade for the Jarvis RAG helper modules.

The implementation now lives in focused ``jarvis_*.py`` modules (config, io,
textproc, speech, vectorstore, indexing, extractors, query, retrieval, answer).
This module re-exports their public names so existing imports such as
``from jarvis_helpers import handle_query`` keep working unchanged.
"""

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from jarvisrag.config import (
    INDEX_MANIFEST_FILE,
    INDEX_MANIFEST_VERSION,
    JARVIS_DEBUG,
    debug_print,
    logger,
    short_text_qa_template,
)

from jarvisrag.iotools import (
    JavaWarningFilter,
    _is_port_open,
    read_user_query,
    suppress_external_output,
    suppress_external_stderr,
    suppress_java_warnings,
    time_limit,
)

from jarvisrag.text.textproc import (
    is_low_information_text,
    normalize_text,
    text_artifact_score,
    _is_ocr_garbage_text,
    _load_system_english_vocab,
)

from jarvisrag.speech import (
    KokoroSpeaker,
    _CITATION_FILE_EXT,
    _INLINE_CITATION_RE,
    _SOURCE_REFERENCE_RE,
    SilentSpeaker,
    _normalize_for_speech,
    _speech_text_for_response,
    _spell_digits,
    _spoken_response_text,
    _strip_inline_citations,
    _strip_source_filenames_for_speech,
    _summarize_for_speech,
)

from jarvisrag.vectorstore import (
    MilvusVectorStore,
    _metadata_filters_to_milvus_expr,
    create_milvus_vector_store,
    normalize_vector,
)

from jarvisrag.extraction.indexing import (
    chunk_documents,
    chunk_text,
    ensure_document_metadata,
    filter_informative_documents,
    group_documents_by_source_windows,
    split_large_docs,
)
from jarvisrag.extraction.persistence import (
    build_or_load_persistent_index,
    sync_documents_with_index,
)

from jarvisrag.extraction.extractors import (
    docling_reader,
    file_extractor,
    has_files_with_extensions,
    jpeg_extractor,
    load_documents_with_cache,
)
from jarvisrag.extraction.images import JpegExtractor
from jarvisrag.extraction.office import (
    DeferredExtractor,
    DoclingReader,
    ExcelReader,
    PowerPointReader,
    ZipExtractor,
)
from jarvisrag.extraction.pdf import CombinedPDFReader, TablePDFReader

from jarvisrag.text.query import (
    _FOLLOW_UP_PRONOUN_RE,
    _NON_SUBJECT_WORDS,
    _best_detail_candidate,
    _best_detail_snippet,
    _contains_all_terms,
    _detail_candidates,
    _detail_score,
    _document_search_text,
    _effective_query_for_turn,
    _evidence_body_from_evidence_line,
    _extract_source_hints,
    _is_negative_evidence_feedback,
    _is_refinement_turn,
    _latest_user_turn,
    _line_matches_source_hints,
    _looks_like_follow_up,
    _names_new_subject,
    _normalize_source_hint,
    _prepare_retrieval_query,
    _prioritize_source_hinted_evidence,
    _query_anchor_terms,
    _query_requests_corpus_details,
    _query_requests_reasoning,
    _query_requests_refinement,
    _query_requests_source_details,
    _query_requests_values,
    _query_terms,
    _query_topic_coverage,
    _query_topic_terms,
    _retrieval_query_variants,
    _source_label_from_evidence_line,
    _specific_query_terms,
    _term_family,
    _term_family_hit,
    _topic_snippet_score,
    _value_numbers,
)

from jarvisrag.retrieval import (
    _evidence_lines_for_turn,
    _force_include_source_hinted_nodes,
    _hybrid_enabled,
    _query_candidate_score,
    _query_evidence_is_supported,
    _query_focused_snippet,
    _retrieve_query_supported_evidence,
    _source_nodes_to_evidence_lines,
    _source_nodes_to_query_evidence_lines,
    build_hybrid_retriever,
)

from jarvisrag.answer.runtime import QueryRuntime
from jarvisrag.answer.verdicts import _response_is_dismissive
from jarvisrag.answer.generation import _llm_output_budget
from jarvisrag.answer.grounding import _review_answer_against_evidence
from jarvisrag.answer.fallbacks import (
    _answer_from_general_knowledge,
    _distill_answer_from_evidence,
    _evidence_present_fallback,
    _extractive_reasoning_answer_from_evidence,
    _extractive_summary_answer_from_evidence,
    _forced_extractive_summary_from_evidence,
    _not_found_answer,
)
from jarvisrag.answer.synthesis import (
    _answer_with_review_loop,
    _reasoning_guidance,
    _synthesize_evidence_answer,
)
from jarvisrag.answer.display import (
    _extract_source_references,
    _format_response_for_display,
)
from jarvisrag.answer.previews import (
    _build_corpus_details_preview,
    _build_source_details_preview,
    _build_topic_details_preview,
    _canonical_detail_file_key,
)
from jarvisrag.answer.pipeline import handle_query
