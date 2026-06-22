import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
import sys
from datetime import UTC, datetime

if os.environ.get("JARVIS_DEBUG_HTTP", "0") != "1":
    logging.getLogger("httpx").setLevel(logging.WARNING)
from jarvis_helpers import (
    JARVIS_DOMAIN,
    TTSManager,
    debug_print,
    file_extractor,
    has_files_with_extensions,
    load_documents_with_cache,
    short_text_qa_template,
    create_milvus_vector_store,
    build_hybrid_retriever,
    build_or_load_persistent_index,
    filter_informative_documents,
    group_documents_by_source_windows,
    _answer_with_review_loop,
    _answer_from_general_knowledge,
    _build_corpus_details_preview,
    _build_source_details_preview,
    _build_topic_details_preview,
    _effective_query_for_turn,
    _evidence_lines_for_turn,
    _evidence_present_fallback,
    _extractive_value_answer_from_documents,
    _extractive_value_answer_from_evidence,
    _latest_user_turn,
    _query_phrase_anchors,
    _query_page_hints,
    _query_requests_reasoning,
    _is_negative_evidence_feedback,
    _is_port_open,
    _not_found_answer,
    _prioritize_source_hinted_evidence,
    _purge_chat_memory_vectors,
    _query_requests_corpus_details,
    _query_requests_source_details,
    _response_is_dismissive,
    _retrieve_query_supported_evidence,
    _speech_text_for_response,
    _spoken_response_text,
    _summarize_for_speech,
    _format_response_for_display,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

print(f"Python version = {sys.version_info}")
if len(sys.argv) > 1:
    input_dir = sys.argv[1]
else:
    input_dir = r"./documents"


# ---------------------------
# File loaders
# ---------------------------
'''
reader = SimpleDirectoryReader(
    input_dir=r"/path/to/documents/",
    recursive=True,  # Enables traversal into subfolders
    file_extractor=file_extractor
)
'''
# Load documents
#documents = reader.load_data()


# LLM
llm = Ollama(
    model="gemma4:31b",
    request_timeout=600,
    temperature=0.2,
    additional_kwargs={"num_predict": 256},
)  # make sure ollama is running: `ollama run llama3`


# Build index
# Local embedding model — bge-m3: 8192-token context, 1024-dim, multilingual,
# strong retrieval ranking on heterogeneous content (resumes, contracts, PDFs).
# Do NOT add query/text instructions — bge-m3 is not prefix-trained and adding
# them degrades retrieval quality.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
)

if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN is not set. HuggingFaceEmbedding may use anonymous access and be rate-limited.\n" \
          "Set HF_TOKEN in your environment for authenticated downloads and better stability.")

Settings.llm = llm
Settings.embed_model = embed_model

# has_files_with_extensions imported from jarvis_helpers

# Initialize corpus_docs to prevent undefined variable errors
corpus_docs = []

# Helper functions (moved to `jarvis_helpers.py`).


# Load PDFs and documents
document_extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".ppt", ".pptx", ".txt", ".csv", ".rtf"]
EXTRACTION_CACHE_DIR = os.path.join(os.path.dirname(__file__), "jarvis_storage", "extraction_cache")
if has_files_with_extensions(input_dir, document_extensions):
    print("\nProcessing documents...")
    corpus_docs = load_documents_with_cache(
        input_dir=input_dir,
        required_exts=document_extensions,
        file_extractor=file_extractor,
        cache_dir=EXTRACTION_CACHE_DIR,
        recursive=True,
    )
    
    # Track ALL files in input directory that match extensions (even if they produce no docs)
    all_attempted_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in document_extensions):
                all_attempted_files.append(os.path.join(root, file))
    
    # Get unique file paths from extracted documents
    all_extracted_file_paths = sorted({str(doc.metadata.get("file_path")) for doc in corpus_docs if doc.metadata.get("file_path")})
    
    # Create comprehensive summary: files processed + files attempted
    summary_lines = [
        "COMPLETE FILE LIST - ALL PROCESSED DOCUMENTS",
        f"Total files found: {len(all_attempted_files)}",
        f"Total documents extracted: {len(all_extracted_file_paths)}",
        "",
        "FILENAMES:",
    ]
    for path in all_extracted_file_paths:
        filename = os.path.basename(path)
        summary_lines.append(f"- {filename}")
    
    summary_lines.extend(["", "FULL PATHS:"])
    summary_lines.extend(f"- {path}" for path in all_extracted_file_paths)
    
    summary_text = "\n".join(summary_lines)
    summary_doc = Document(
        text=summary_text,
        metadata={"type": "file_list", "file_path": "__summary__", "source": "summary", "is_file_list": True}
    )
    corpus_docs.insert(0, summary_doc)
    print(f"Added summary document for {len(all_extracted_file_paths)} extracted files from {len(all_attempted_files)} total files")
else:
    print("\nNo documents found in directory")

doc_index = None
storage_context = None
doc_chat_engine = None
doc_query_engine = None

# Load images if any exist
image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]
img_chat_engine = None
context_chat_engine = None
all_docs = corpus_docs or []

if has_files_with_extensions(input_dir, image_extensions):
    print("\nFound images, processing image files...")
    try:
        img_docs = load_documents_with_cache(
            input_dir=input_dir,
            required_exts=image_extensions,
            file_extractor=file_extractor,
            cache_dir=EXTRACTION_CACHE_DIR,
            recursive=True,
        )
        all_docs.extend(img_docs)
        if img_docs and os.environ.get("JARVIS_ENABLE_IMAGE_CHAT_ENGINE", "0") == "1":
            img_index = VectorStoreIndex.from_documents(img_docs)
            img_chat_engine = img_index.as_chat_engine(chat_mode="condense_plus_context", llm=llm)
        print(f"Successfully processed {len(img_docs)} images")
        if img_docs:
            print("Image extraction details:")
            for idx, doc in enumerate(img_docs, start=1):
                file_name = doc.metadata.get("file_name", "<unknown>")
                extracted_by = doc.metadata.get("extracted_by", "unknown")
                excerpt = doc.text.replace("\n", " ")
                if len(excerpt) > 180:
                    excerpt = excerpt[:180].rstrip() + "..."
                print(f"  {idx}. {file_name} [{extracted_by}] -> {excerpt}")
    except Exception as e:
        print(f"Warning: Error processing images: {e}")
else:
    print("\nNo image files found in directory")

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "jarvis_storage")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "jarvis_context")

if all_docs:
    raw_doc_count = len(all_docs)
    all_docs = filter_informative_documents(all_docs)
    filtered_doc_count = raw_doc_count - len(all_docs)
    if filtered_doc_count:
        print(f"Filtered {filtered_doc_count} low-information extracted chunks before indexing.")
    if not all_docs:
        raise RuntimeError("All extracted chunks were low-information; nothing remains to index.")

    context_mode = os.environ.get("JARVIS_CONTEXT_MODE", "chunk").strip().lower()
    if context_mode not in {"chunk", "source_window"}:
        raise ValueError("JARVIS_CONTEXT_MODE must be 'chunk' or 'source_window'.")
    if context_mode == "source_window":
        context_window_words = int(os.environ.get("JARVIS_CONTEXT_WINDOW_WORDS", "2500"))
        extracted_doc_count = len(all_docs)
        all_docs = group_documents_by_source_windows(all_docs, max_words=context_window_words)
        print(
            f"Grouped {extracted_doc_count} extracted chunks into {len(all_docs)} "
            f"source-window context documents (max_words={context_window_words})."
        )

    vector_backend = os.environ.get("JARVIS_VECTOR_BACKEND", "milvus").strip().lower()
    if vector_backend not in {"milvus", "local", "memory"}:
        raise ValueError("JARVIS_VECTOR_BACKEND must be 'milvus', 'local', or 'memory'.")

    use_milvus = vector_backend == "milvus" and os.environ.get("USE_MILVUS", "1") == "1"
    require_milvus = use_milvus and os.environ.get("JARVIS_REQUIRE_MILVUS", "1") == "1"
    sync_existing_index = os.environ.get("JARVIS_SYNC_EXISTING_INDEX", "0") == "1"
    rebuild_index = os.environ.get("JARVIS_REBUILD_INDEX", "0") == "1"
    response_mode = os.environ.get(
        "JARVIS_RESPONSE_MODE",
        "query",
    ).strip().lower()
    if response_mode not in {"query", "chat"}:
        raise ValueError("JARVIS_RESPONSE_MODE must be 'query' or 'chat'.")
    if filtered_doc_count and vector_backend != "memory" and not rebuild_index:
        print(
            "Warning: low-information chunks were filtered from the current extraction, "
            "but an existing index may still contain old vectors. Jarvis will rebuild automatically "
            "when the index manifest is missing or stale; set JARVIS_REBUILD_INDEX=1 to force a rebuild."
        )
    vector_store = None
    active_vector_backend = "memory" if vector_backend == "memory" else "local"

    print(
        f"Vector backend -> {vector_backend}. "
        f"Milvus configuration -> enabled={use_milvus}, required={require_milvus}, "
        f"host={MILVUS_HOST}, port={MILVUS_PORT}, collection={MILVUS_COLLECTION}"
    )

    if vector_backend == "memory":
        print("Using in-memory VectorStoreIndex.from_documents backend (pre-Milvus style).")
        doc_index = VectorStoreIndex.from_documents(all_docs, embed_model=embed_model)
        storage_context = None
        active_vector_backend = "memory"
    elif require_milvus and not use_milvus:
        raise RuntimeError(
            "Milvus is required but explicitly disabled (USE_MILVUS=0). "
            "Set USE_MILVUS=1 or JARVIS_REQUIRE_MILVUS=0 to allow local fallback."
        )

    if vector_backend != "memory":
        if use_milvus and not _is_port_open(MILVUS_HOST, MILVUS_PORT, timeout=2.0):
            if require_milvus:
                raise RuntimeError(
                    f"Milvus is required but unreachable at {MILVUS_HOST}:{MILVUS_PORT}. "
                    "Start Milvus or set JARVIS_REQUIRE_MILVUS=0 to allow local fallback."
                )
            print(f"Warning: Cannot reach Milvus at {MILVUS_HOST}:{MILVUS_PORT}. Skipping Milvus and using local fallback.")
            use_milvus = False
        elif use_milvus:
            print(f"Milvus TCP health check passed at {MILVUS_HOST}:{MILVUS_PORT}.")

        try:
            if use_milvus:
                vector_store = create_milvus_vector_store(
                    collection_name=MILVUS_COLLECTION,
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                )
                try:
                    has_collection = vector_store.has_collection()
                    print(f"Milvus collection '{MILVUS_COLLECTION}' exists={has_collection}")
                    if has_collection and os.environ.get("JARVIS_PURGE_CHAT_MEMORY", "1") == "1":
                        _purge_chat_memory_vectors(vector_store)
                except Exception as e:
                    print(f"Warning: Unable to verify Milvus collection state: {e}")
            else:
                from llama_index.core.vector_stores.simple import SimpleVectorStore
                vector_store = SimpleVectorStore()

            print("Building or loading persistent index...")
            doc_index, storage_context = build_or_load_persistent_index(
                all_docs,
                vector_store,
                persist_dir=PERSIST_DIR,
                embed_model=embed_model,
                sync_existing=sync_existing_index,
                rebuild=rebuild_index,
            )
            if use_milvus:
                active_vector_backend = "milvus"
                print(f"Using Milvus vector store at {MILVUS_HOST}:{MILVUS_PORT}/{MILVUS_COLLECTION}")
            else:
                active_vector_backend = "local"
                print("Using local vector store (Milvus disabled by USE_MILVUS).")
        except Exception as e:
            if use_milvus and require_milvus:
                raise RuntimeError(
                    f"Milvus is required and initialization failed: {e}. "
                    "Fix Milvus connectivity/config or set JARVIS_REQUIRE_MILVUS=0 to allow fallback."
                ) from e

            print(f"Warning: Milvus initialization failed, falling back to local index: {e}")
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            local_store = SimpleVectorStore()
            doc_index, storage_context = build_or_load_persistent_index(
                all_docs,
                local_store,
                persist_dir=PERSIST_DIR,
                embed_model=embed_model,
                sync_existing=sync_existing_index,
                rebuild=rebuild_index,
            )
            active_vector_backend = "local"
            print("Using local vector store fallback for persistence.")

    print(f"Active vector backend: {active_vector_backend}")
    print(f"Response mode: {response_mode}")
    print(f"Context mode: {context_mode}")
    print(f"Detail scan domain profile: {JARVIS_DOMAIN}")

    doc_chat_engine = doc_index.as_chat_engine(chat_mode="condense_plus_context", llm=llm)
    doc_retriever = build_hybrid_retriever(
        doc_index,
        dense_top_k=int(os.environ.get("JARVIS_RETRIEVAL_TOP_K", "64")),
        bm25_top_k=int(os.environ.get("JARVIS_BM25_TOP_K", os.environ.get("JARVIS_RETRIEVAL_TOP_K", "64"))),
        fusion_top_k=int(os.environ.get("JARVIS_FUSION_TOP_K", os.environ.get("JARVIS_RETRIEVAL_TOP_K", "64"))),
        llm=llm,
    )
    doc_query_engine = doc_index.as_query_engine(
        llm=llm,
        similarity_top_k=int(os.environ.get("JARVIS_QUERY_TOP_K", "48")),
        response_mode="compact",
        text_qa_template=short_text_qa_template,
    )
    if active_vector_backend == "memory":
        print(f"Successfully processed {len(all_docs)} documents into the in-memory index")
    else:
        print(f"Successfully processed {len(all_docs)} documents into the persistent index")

    print(f"Total raw documents loaded: {len(all_docs)}")
    if doc_query_engine is None:
        raise RuntimeError("No query engine available for document retrieval.")
else:
    print("\nNo documents available for combined context engine")
    doc_chat_engine = None
    doc_query_engine = None
    doc_retriever = None


print("\n\nAsk questions about your data! Type 'exit' to quit.")


class _NoopTTS:
    """TTS stub used when JARVIS_DISABLE_TTS=1 (e.g. smoke tests)."""

    def speak(self, _text): pass
    def say_goodbye(self): pass
    def cleanup(self): pass


# Initialize the TTS Manager (skippable for smoke tests)
if os.environ.get("JARVIS_DISABLE_TTS", "0") == "1":
    tts_manager = _NoopTTS()
else:
    tts_manager = TTSManager()
last_turn: dict[str, object] = {}


def handle_query(user_query: str, prior_turn: dict[str, object] | None = None) -> dict[str, object]:
    """Process a single user query through the full RAG + fallback pipeline.
    Returns the new last_turn dict so callers (e.g. smoke tests) can chain.
    Uses the module-level cached index/retriever/LLM — no re-initialization."""
    global last_turn
    if prior_turn is not None:
        last_turn = prior_turn

    print("\n[📄 Context Results]")
    try:
        effective_user_query, is_follow_up, source_hints = _effective_query_for_turn(user_query, last_turn)
        if is_follow_up:
            debug_print("[debug] Treated user input as a follow-up clarification.")
            debug_print(f"[debug] Effective query: {effective_user_query}")
        if source_hints:
            debug_print(f"[debug] Source hints: {source_hints}")
        negative_evidence_feedback = _is_negative_evidence_feedback(user_query)
        if negative_evidence_feedback:
            debug_print("[debug] User feedback rejected previous evidence; not merging prior evidence lines.")

        if _query_requests_corpus_details(user_query):
            response_text = _build_corpus_details_preview(all_docs)
            debug_print("[debug] Used direct corpus details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts_manager.speak(_spoken_response_text(response_text))
            last_turn = {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }
            return last_turn

        if _query_requests_source_details(user_query):
            response_text = _build_source_details_preview(effective_user_query, all_docs, llm)
            debug_print("[debug] Used direct source details scan.")
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts_manager.speak(_spoken_response_text(response_text))
            last_turn = {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }
            return last_turn

        if response_mode == "chat":
            if doc_chat_engine is None:
                raise RuntimeError("No chat engine available")

            response = doc_chat_engine.chat(user_query)
            response_text = str(response).strip()
            debug_print("[debug] Used legacy chat response mode.")
            if _response_is_dismissive(response_text):
                if os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] Chat-mode response was dismissive; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, llm)
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
            print(f"Jarvis: {_format_response_for_display(response_text)}\n")
            tts_manager.speak(_spoken_response_text(response_text))
            last_turn = {
                "user_query": user_query,
                "effective_query": effective_user_query,
                "response_text": response_text,
                "evidence_lines": [],
            }
            return last_turn

        if doc_retriever is None:
            raise RuntimeError("No retriever available")

        evidence_lines, source_nodes = _retrieve_query_supported_evidence(effective_user_query, doc_retriever)
        evidence_lines = _evidence_lines_for_turn(
            evidence_lines,
            last_turn,
            is_follow_up,
            negative_evidence_feedback,
            source_hints,
        )
        debug_print(f"[debug] Retrieved source chunks: {len(source_nodes)}")
        debug_print(f"[debug] Query-supported evidence lines: {len(evidence_lines)}")
        for evidence_line in evidence_lines:
            debug_print(f"[debug] Evidence: {evidence_line}")
        if evidence_lines:
            phrase_anchors = _query_phrase_anchors(effective_user_query)
            page_hints = _query_page_hints(effective_user_query)
            latest_turn_is_reasoning = _query_requests_reasoning(
                _latest_user_turn(effective_user_query)
            )
            allow_keyword_extractor = (
                not phrase_anchors
                and not page_hints
                and not latest_turn_is_reasoning
            )
            response_text = (
                _extractive_value_answer_from_evidence(effective_user_query, evidence_lines)
                if allow_keyword_extractor
                else ""
            )
            if response_text:
                review_passed = True
                review_reason = "answered directly from retrieved value evidence"
                debug_print("[debug] Used extractive value answer from retrieved evidence.")
            else:
                if not allow_keyword_extractor:
                    debug_print(
                        "[debug] Skipped keyword extractor due to phrase/page hints or reasoning intent; routing to LLM review loop."
                    )
                doc_scan_answer = (
                    _extractive_value_answer_from_documents(effective_user_query, all_docs)
                    if allow_keyword_extractor
                    else ""
                )
                if doc_scan_answer:
                    response_text = doc_scan_answer
                    review_passed = True
                    review_reason = "answered directly from local value evidence scan"
                    debug_print("[debug] Used extractive value answer from local document scan.")
                else:
                    prior_answer_for_refinement = ""
                    if is_follow_up:
                        prior_answer_for_refinement = str(last_turn.get("response_text") or "")
                    response_text, review_passed, review_reason = _answer_with_review_loop(
                        effective_user_query,
                        evidence_lines,
                        llm,
                        prior_answer=prior_answer_for_refinement,
                    )
        else:
            response_text = _extractive_value_answer_from_documents(effective_user_query, all_docs)
            if response_text:
                review_passed = True
                review_reason = "answered directly from local value evidence scan"
                debug_print("[debug] Used extractive value answer from local document scan.")
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

        if not review_passed:
            debug_print(f"[debug] Initial answer did not pass review: {review_reason}")
            if evidence_lines and not answer_was_dismissive:
                debug_print("[debug] Returning evidence-present fallback instead of no-evidence response.")
                response_text = _evidence_present_fallback(evidence_lines, review_reason)
            else:
                if answer_was_dismissive:
                    debug_print("[debug] Evidence chunks were rejected by the answerer as irrelevant; bypassing evidence fallback.")
                topic_preview = _build_topic_details_preview(effective_user_query, all_docs, llm)
                if topic_preview and not _response_is_dismissive(topic_preview):
                    debug_print("[debug] Applied direct topic corpus fallback.")
                    response_text = topic_preview
                elif os.environ.get("JARVIS_ALLOW_GENERAL_KNOWLEDGE", "1") == "1":
                    debug_print("[debug] No corpus evidence found; falling back to general LLM knowledge.")
                    gk_answer = _answer_from_general_knowledge(effective_user_query, llm)
                    response_text = gk_answer or _not_found_answer(effective_user_query, review_reason)
                else:
                    response_text = _not_found_answer(effective_user_query, review_reason)
        else:
            debug_print("[debug] Returning reviewed answer.")

        print(f"Jarvis: {_format_response_for_display(response_text)}\n")
        tts_manager.speak(_spoken_response_text(response_text))
        last_turn = {
            "user_query": user_query,
            "effective_query": effective_user_query,
            "response_text": response_text,
            "evidence_lines": evidence_lines,
        }

        if os.environ.get("JARVIS_ENABLE_CHAT_MEMORY", "0") == "1" and doc_index is not None and storage_context is not None:
            if not review_passed or _response_is_dismissive(response_text):
                debug_print("[debug] Skipping chat memory persistence (response was a fallback or not reviewed).")
            else:
                chat_doc = Document(
                    text=f"User: {user_query}\nAssistant: {response_text}",
                    metadata={
                        "source": "chat_memory",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
                try:
                    doc_index.insert(chat_doc)
                    storage_context.persist(PERSIST_DIR)
                except Exception as e:
                    print(f"Warning: chat memory persistence failed: {e}")
    except Exception as e:
        print("Context engine failed:", e)
        tts_manager.speak("Sorry, I encountered an error processing your request.")
    return last_turn


if __name__ == "__main__":
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            tts_manager.say_goodbye()
            tts_manager.cleanup()
            break
        handle_query(user_query)
