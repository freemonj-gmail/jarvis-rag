import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
import sys

if os.environ.get("JARVIS_DEBUG_HTTP", "0") != "1":
    logging.getLogger("httpx").setLevel(logging.WARNING)
from jarvis_helpers import (
    JARVIS_DOMAIN,
    TTSManager,
    QueryRuntime,
    debug_print,
    handle_query,
    file_extractor,
    has_files_with_extensions,
    load_documents_with_cache,
    short_text_qa_template,
    create_milvus_vector_store,
    build_hybrid_retriever,
    build_or_load_persistent_index,
    filter_informative_documents,
    group_documents_by_source_windows,
    _is_port_open,
    _purge_chat_memory_vectors,
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
    response_mode = "query"


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

# Bundle the initialized runtime objects for the query pipeline (lives in
# jarvis_helpers.py to keep this entry script thin).
runtime = QueryRuntime(
    all_docs=all_docs,
    llm=llm,
    doc_retriever=doc_retriever,
    doc_chat_engine=doc_chat_engine,
    doc_index=doc_index,
    storage_context=storage_context,
    response_mode=response_mode,
    tts=tts_manager,
    persist_dir=PERSIST_DIR,
)
last_turn: dict[str, object] = {}


if __name__ == "__main__":
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            tts_manager.say_goodbye()
            tts_manager.cleanup()
            break
        last_turn = handle_query(user_query, runtime, last_turn)
