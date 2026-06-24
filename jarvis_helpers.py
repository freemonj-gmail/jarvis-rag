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

from jarvis_config import (
    INDEX_MANIFEST_FILE,
    INDEX_MANIFEST_VERSION,
    JARVIS_DEBUG,
    JARVIS_DOMAIN,
    debug_print,
    logger,
    short_text_qa_template,
)

from jarvis_io import (
    JavaWarningFilter,
    _is_port_open,
    read_user_query,
    suppress_external_output,
    suppress_external_stderr,
    suppress_java_warnings,
    time_limit,
)

from jarvis_textproc import (
    is_low_information_text,
    normalize_text,
    text_artifact_score,
    _is_ocr_garbage_text,
    _load_system_english_vocab,
)

from jarvis_speech import (
    TTSManager,
    _CITATION_FILE_EXT,
    _INLINE_CITATION_RE,
    _SOURCE_REFERENCE_RE,
    _SilentTTS,
    _normalize_for_speech,
    _speech_text_for_response,
    _spell_digits,
    _spoken_response_text,
    _strip_inline_citations,
    _strip_source_filenames_for_speech,
    _summarize_for_speech,
)

from jarvis_vectorstore import (
    MilvusVectorStore,
    _metadata_filters_to_milvus_expr,
    create_milvus_vector_store,
    normalize_vector,
)

from jarvis_indexing import (
    build_or_load_persistent_index,
    chunk_documents,
    chunk_text,
    ensure_document_metadata,
    filter_informative_documents,
    group_documents_by_source_windows,
    split_large_docs,
    sync_documents_with_index,
)

from jarvis_extractors import (
    CombinedPDFReader,
    DeferredExtractor,
    DoclingReader,
    ExcelReader,
    JpegExtractor,
    PowerPointReader,
    TablePDFReader,
    ZipExtractor,
    docling_reader,
    file_extractor,
    has_files_with_extensions,
    jpeg_extractor,
    load_documents_with_cache,
)

from jarvis_query import (
    DOMAIN_BOILERPLATE_MARKERS,
    DOMAIN_DETAIL_KEYWORDS,
    GENERIC_BOILERPLATE_MARKERS,
    GENERIC_DETAIL_KEYWORDS,
    PAYMENT_CONTEXT_MARKERS,
    _FOLLOW_UP_PRONOUN_RE,
    _NON_SUBJECT_WORDS,
    _amount_segment_score,
    _best_detail_candidate,
    _best_detail_snippet,
    _contains_all_terms,
    _detail_candidates,
    _detail_score,
    _document_search_text,
    _effective_query_for_turn,
    _evidence_body_from_evidence_line,
    _evidence_line_matches_page,
    _evidence_line_matches_phrase,
    _extract_source_hints,
    _facet_amount_value,
    _facet_label,
    _facet_row_amount_value,
    _format_amount,
    _is_negative_evidence_feedback,
    _is_refinement_turn,
    _latest_user_turn,
    _line_matches_source_hints,
    _looks_like_follow_up,
    _money_values,
    _names_new_subject,
    _normalize_source_hint,
    _payment_context_score,
    _prepare_retrieval_query,
    _prioritize_source_hinted_evidence,
    _query_anchor_terms,
    _query_page_hints,
    _query_phrase_anchors,
    _query_requests_corpus_details,
    _query_requests_reasoning,
    _query_requests_refinement,
    _query_requests_source_details,
    _query_requests_values,
    _query_terms,
    _query_topic_coverage,
    _query_topic_terms,
    _query_value_facets,
    _retrieval_query_variants,
    _source_label_from_evidence_line,
    _specific_query_terms,
    _term_family,
    _term_family_hit,
    _topic_snippet_score,
    _value_amounts,
    _value_numbers,
)

from jarvis_retrieval import (
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

from jarvis_answer import (
    QueryRuntime,
    _answer_from_general_knowledge,
    _answer_with_review_loop,
    _build_corpus_details_preview,
    _build_escalation_llm,
    _build_evidence_preview,
    _build_source_details_preview,
    _build_topic_details_preview,
    _canonical_detail_file_key,
    _distill_answer_from_evidence,
    _evidence_present_fallback,
    _extract_source_references,
    _extractive_reasoning_answer_from_evidence,
    _extractive_summary_answer_from_evidence,
    _extractive_value_answer_from_documents,
    _extractive_value_answer_from_evidence,
    _forced_extractive_summary_from_evidence,
    _format_response_for_display,
    _llm_output_budget,
    _not_found_answer,
    _purge_chat_memory_vectors,
    _reasoning_guidance,
    _response_is_dismissive,
    _review_answer_against_evidence,
    _router_enabled,
    _salvage_answer_from_sources,
    _synthesize_evidence_answer,
    build_chunk_query_engines,
    handle_query,
    route_general_knowledge_llm,
    stitch_chunk_responses,
    summarize_partial_responses,
)
